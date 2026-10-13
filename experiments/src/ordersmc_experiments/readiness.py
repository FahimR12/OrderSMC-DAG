"""Frozen, paired, MPI-accelerated score-transition diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig, build_candidates, sample
from ordersmc.graphs import optimal_bayes_action_dag
from ordersmc.persistence import save_candidate_table, candidate_table_content_sha256
from .exact_order import exact_order_dp
from .exact_partition_log import exact_partition_log_dp
from .generator import MixedDAGConfig, generate_mixed_dag_dataset
from .paper_metrics import _posterior_metrics
from .readiness_audits import read, write, sha, rmse
from .seeds import protocol_seed_record


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def freeze(config_path: Path, output: Path, smoke: bool = False):
    config = read(config_path)
    if smoke:
        config.update(replicates=1, sample_sizes=[100], method_seeds=2, mpi_ranks=1,
                      status="implementation-smoke-not-scientific-results")
        config["ordersmc"] = dict(particles=128, num_betas=24, mh_steps=3, mh_steps_resample=6)
    repo = Path(__file__).resolve().parents[3]
    source_files = {}
    for folder in ("src/ordersmc", "experiments/src/ordersmc_experiments"):
        source_files.update({str(p.relative_to(repo)): sha(p) for p in sorted((repo / folder).rglob("*.py"))})
    plan = {"schema": "ordersmc-readiness-plan-v1", "config": config,
            "source_files": source_files, "source_sha256": canonical_sha(source_files),
            "environment": {"python": sys.version, "numpy": np.__version__}}
    plan["plan_sha256"] = canonical_sha(plan)
    destination = output / "plan.json"
    if destination.exists():
        if read(destination) != plan:
            raise FileExistsError(f"existing frozen plan differs: {destination}; use a new output root")
        return plan
    output.mkdir(parents=True, exist_ok=True)
    for relative in source_files:
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
    write(destination, plan)
    return plan


def paired_variants(data, types):
    transformed = data.copy()
    for column, kind in types.items():
        if kind == "con":
            transformed[column] = np.sinh(1.25 * np.arcsinh(data[column]) + .5)
    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise ArithmeticError("nonfinite transformed observations")
    return {"unrestricted_native": data.copy(), "paired_nongaussian": transformed}


def score_config(config, seed=0):
    return OrderSMCConfig.protocol_v2(
        lambda_target=config["edge_prior_lambda"], score_method=config["score_method"],
        max_parents=config["maximum_indegree"], screen_k=config["dimension"] - 1,
        parent_pool_max=None, keep_per_child=None, pair_rescue_anchor_k=0,
        pair_rescue_k=0, seed=seed, **config["ordersmc"])


def summarize_posterior(probabilities, truth, nodes, max_indegree):
    # Exact log-space sums and MPI weighted reductions may exceed a boundary
    # by a few ulps. Reject substantive violations; do not clip an invalid fit.
    probabilities = validated_probabilities(probabilities)
    action = optimal_bayes_action_dag(probabilities, nodes, max_indegree=max_indegree)
    selected, actual = set(action.edges()), set(truth.edges())
    f1 = lambda a, b: 2. * len(a & b) / max(1, len(a) + len(b))
    skeleton = lambda edges: {frozenset(e) for e in edges}
    return {**_posterior_metrics(probabilities, truth, tuple(nodes)),
            "directed_f1": f1(selected, actual),
            "skeleton_f1": f1(skeleton(selected), skeleton(actual)),
            "representative_edges": sorted(map(list, selected)),
            "representative_graph_diagnostics": dict(action.graph),
            "edge_probabilities": [{"parent": u, "child": v, "probability": p}
                                   for (u, v), p in probabilities.items()]}


def validated_probabilities(probabilities, tolerance=1e-8):
    result = {}
    for edge, value in probabilities.items():
        if not np.isfinite(value) or value < -tolerance or value > 1. + tolerance:
            raise ArithmeticError(f"invalid posterior probability {edge}: {value}")
        result[edge] = float(np.clip(value, 0., 1.))
    for u, v in sorted(result):
        if u >= v:
            continue
        total = result[u, v] + result.get((v, u), 0.)
        if total > 1. + tolerance:
            raise ArithmeticError(f"incompatible opposite-edge probabilities: {(u, v)} {total}")
        if total > 1.:
            result[u, v] /= total
            result[v, u] = result.get((v, u), 0.) / total
    return result


def root_call(comm, function):
    """Broadcast root-only failures before entering the next collective."""
    value, error = None, None
    if comm is None or comm.rank == 0:
        try:
            value = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if comm is not None:
        error = comm.bcast(error, root=0)
        value = comm.bcast(value, root=0)
    if error:
        raise RuntimeError(error)
    return value


def run_case(output: Path, index: int, mpi: bool = False):
    comm = None
    if mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    plan = root_call(comm, lambda: read(output / "plan.json"))
    config = plan["config"]
    ranks = comm.size if comm is not None else 1
    if ranks != config["mpi_ranks"]:
        raise RuntimeError(f"expected {config['mpi_ranks']} ranks, got {ranks}")
    if not 0 <= index < config["replicates"]:
        raise ValueError("invalid replicate index")

    def prepare():
        # Workers must execute the source snapshot, not a later working-tree edit.
        repo = Path(__file__).resolve().parents[3]
        for path, expected in plan["source_files"].items():
            if sha(repo / path) != expected:
                raise RuntimeError(f"frozen source mismatch: {path}")
        folder = output / "cases" / f"rep{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        generator_config = MixedDAGConfig(
            d=config["dimension"], n_train=max(config["sample_sizes"]), n_test=100,
            max_indegree=config["maximum_indegree"], master_seed=config["master_seed"],
            experiment_id="readiness", cell_id="paired-mixed", replicate_id=index,
            **config["generator"])
        dataset = generate_mixed_dag_dataset(generator_config)
        variants = paired_variants(dataset.train, dataset.variable_types)
        for variant, frame in variants.items():
            frame.to_csv(folder / f"{variant}.csv", index=False)
        write(folder / "generator_manifest.json", dataset.manifest())
        return dataset, variants

    dataset, variants = root_call(comm, prepare)
    folder = output / "cases" / f"rep{index:03d}"
    for variant in config["variants"]:
        for n in config["sample_sizes"]:
            cell = f"{variant}-n{n:05d}"
            destination = folder / f"{cell}.json"
            def existing():
                if not destination.exists():
                    return False
                old = read(destination)
                if old["plan_sha256"] != plan["plan_sha256"] or old["status"] != "complete":
                    raise RuntimeError(f"cannot reuse incompatible result {destination}")
                return True
            if root_call(comm, existing):
                continue
            frame = variants[variant].iloc[:n].copy()
            table = build_candidates(frame, dataset.variable_types, dataset.cardinalities,
                                     config=score_config(config), backend="mpi" if mpi else "local",
                                     comm=comm, verbose=False)
            def references():
                save_candidate_table(table, folder / f"{cell}-candidate.json.gz")
                order = exact_order_dp(table, compute_graph_size=False)
                structure = exact_partition_log_dp(table)
                refs = {"exact-order": summarize_posterior(order.edge_probs, dataset.dag, table.nodes,
                                                           config["maximum_indegree"]),
                        "exact-structure": summarize_posterior(structure.edge_probs, dataset.dag, table.nodes,
                                                               config["maximum_indegree"])}
                return refs, order.edge_probs
            refs, exact = root_call(comm, references)
            records = []
            for seed_index in range(config["method_seeds"]):
                seed_cell = f"paired-n{n:05d}" if config.get("paired_method_seeds", False) else cell
                seed = protocol_seed_record(config["master_seed"], experiment="readiness", cell=seed_cell,
                                            replicate=index, component=f"ordersmc-{seed_index:02d}")
                start = time.perf_counter()
                result = sample(table, config=score_config(config, seed.seed),
                                backend="mpi" if mpi else "local", comm=comm,
                                parent_seed_record=asdict(seed) if mpi else None,
                                verbose=False, provenance_policy="require")
                elapsed = time.perf_counter() - start
                def record_result():
                    local = result.to_local_result() if mpi else result
                    return {"seed_index": seed_index, "seed": asdict(seed),
                            "sampler_wall_seconds": elapsed,
                            "own_target_pip_rmse": rmse(local.edge_probs, exact),
                            **summarize_posterior(local.edge_probs, dataset.dag, table.nodes,
                                                  config["maximum_indegree"])}
                records.append(root_call(comm, record_result))
            def save():
                write(destination, {
                    "schema": "ordersmc-readiness-case-v1", "status": "complete",
                    "plan_sha256": plan["plan_sha256"], "replicate": index,
                    "variant": variant, "n_train": n, "mpi_ranks": ranks,
                    "data_sha256": sha(folder / f"{variant}.csv"),
                    "candidate_sha256": candidate_table_content_sha256(table),
                    "truth_edges": sorted(map(list, dataset.dag.edges())),
                    "variable_types": dataset.variable_types, "references": refs, "smc_runs": records})
                print(f"complete: replicate={index} {cell}", flush=True)
            root_call(comm, save)


def aggregate(output):
    plan = read(output / "plan.json")
    config = plan["config"]
    rows, stability, missing = [], [], []
    for index, variant, n in itertools.product(range(config["replicates"]), config["variants"], config["sample_sizes"]):
        path = output / "cases" / f"rep{index:03d}" / f"{variant}-n{n:05d}.json"
        if not path.exists():
            missing.append(str(path.relative_to(output)))
            continue
        result = read(path)
        if result["plan_sha256"] != plan["plan_sha256"] or result["status"] != "complete":
            raise RuntimeError(f"invalid result: {path}")
        if len(result["smc_runs"]) != config["method_seeds"]:
            raise RuntimeError(f"wrong seed count: {path}")
        common = {"replicate": index, "variant": variant, "n_train": n}
        for method, records in [(m, [r]) for m, r in result["references"].items()] + [("ordersmc", result["smc_runs"])]:
            for record in records:
                rows.append({**common, "method": method,
                             **{key: value for key, value in record.items() if isinstance(value, (int, float))}})
        jaccards, errors = [], []
        for left, right in itertools.combinations(result["smc_runs"], 2):
            a, b = set(map(tuple, left["representative_edges"])), set(map(tuple, right["representative_edges"]))
            jaccards.append(len(a & b) / len(a | b) if a | b else 1.)
            lp = {(r["parent"], r["child"]): r["probability"] for r in left["edge_probabilities"]}
            rp = {(r["parent"], r["child"]): r["probability"] for r in right["edge_probabilities"]}
            errors.append(rmse(lp, rp))
        stability.append({**common, "mean_graph_jaccard": np.mean(jaccards) if jaccards else None,
                          "all_seed_graphs_identical": min(jaccards) == 1. if jaccards else None,
                          "mean_pairwise_pip_rmse": np.mean(errors) if errors else None})
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    write(analysis / "completion.json", {"status": "incomplete" if missing else "complete",
          "expected_datasets": config["replicates"] * len(config["variants"]) * len(config["sample_sizes"]),
          "completed_datasets": len(stability), "completed_smc_runs": sum(r["method"] == "ordersmc" for r in rows),
          "missing": missing, "plan_sha256": plan["plan_sha256"]})
    if missing:
        raise RuntimeError(f"{len(missing)} datasets missing; no final summaries written")
    frame = pd.DataFrame(rows)
    if "own_target_pip_rmse" not in frame:
        frame["own_target_pip_rmse"] = np.nan
    frame.to_csv(analysis / "runs.csv", index=False)
    pd.DataFrame(stability).to_csv(analysis / "stability.csv", index=False)
    metrics = ["directed_auprc", "directed_brier", "directed_f1", "skeleton_f1", "own_target_pip_rmse"]
    dataset = frame.groupby(["variant", "n_train", "replicate", "method"], as_index=False)[metrics].mean()
    dataset.to_csv(analysis / "dataset_means.csv", index=False)
    dataset.groupby(["variant", "n_train", "method"], as_index=False)[metrics].mean().to_csv(analysis / "summary.csv", index=False)
    flags = []
    for (variant, n), group in frame[frame.method == "ordersmc"].groupby(["variant", "n_train"]):
        errors = group.own_target_pip_rmse
        flags.append({"variant": variant, "n_train": int(n), "mean_pip_rmse": errors.mean(),
                      "p95_pip_rmse": errors.quantile(.95),
                      "sampling_flag": bool(errors.mean() > config["diagnostic_flags"]["mean_smc_own_target_pip_rmse"]
                                            or errors.quantile(.95) > config["diagnostic_flags"]["p95_smc_own_target_pip_rmse"])})
    write(analysis / "diagnostic_flags.json", {"status": config["status"], "cells": flags,
          "warning": "Sampling flags are diagnostics, not a recovery or publication acceptance gate. No tuning was performed."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "aggregate"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        if args.config is None:
            parser.error("freeze requires --config")
        print(freeze(args.config, args.output, args.smoke)["plan_sha256"])
    elif args.action == "run":
        try:
            run_case(args.output, args.index, args.mpi)
        except BaseException:
            if args.mpi:
                from mpi4py import MPI
                import traceback
                traceback.print_exc()
                MPI.COMM_WORLD.Abort(1)
            raise
    else:
        aggregate(args.output)


if __name__ == "__main__":
    main()
