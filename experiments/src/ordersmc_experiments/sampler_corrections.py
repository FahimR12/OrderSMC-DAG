"""Frozen, non-overwriting paired diagnostic for the v0.7 sampler corrections."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import networkx as nx

from ordersmc import OrderSMCConfig, sample, validate_candidate_provenance
from ordersmc.confidence import seed_consensus_table
from ordersmc.persistence import load_candidate_table
from .execution import _all_edges, _atomic_json, _json_safe
from .readiness import canonical_sha, summarize_posterior
from .readiness_audits import read, sha, pips

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "experiments/configs/sampler_correction_pilot_v1.json"


def checked_plan(root: Path, *, sources=True):
    plan = read(root / "plan.json")
    if canonical_sha({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise RuntimeError("correction plan hash mismatch")
    if sources:
        for relative, digest in plan["source_files"].items():
            if sha(REPO / relative) != digest:
                raise RuntimeError(f"source changed: {relative}; run from the frozen snapshot")
    return plan


def checked_input(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise RuntimeError(f"frozen input changed: {path}")
    return path


def artifact(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def truth_graph(manifest):
    graph = nx.DiGraph()
    graph.add_nodes_from(manifest["variable_types"])
    graph.add_edges_from(manifest["edges"])
    return graph


def freeze(config_path: Path, output: Path, *, smoke=False):
    if output.exists():
        raise FileExistsError("use a fresh correction output directory; existing results are never overwritten")
    config = read(config_path)
    source = (REPO / config["source"]).resolve()
    completion = read(source / "analysis/completion.json")
    if completion.get("status") != "complete" or completion.get("missing_or_failed_runs", 0):
        raise RuntimeError("source metadata cohort is incomplete")
    source_plan = read(source / "plan.json")
    if source_plan["plan_sha256"] != config["source_plan_sha256"]:
        raise RuntimeError("unexpected metadata source plan")
    if canonical_sha({k: v for k, v in source_plan.items() if k != "plan_sha256"}) != source_plan["plan_sha256"]:
        raise RuntimeError("source plan hash mismatch")
    if smoke:
        config.update(replicates=[0], seed_indices=[0, 1], mpi_ranks=4,
                      status="implementation-smoke-not-scientific-results")
        config["common"].update(particles=32, num_betas=8, mh_steps=1, mh_steps_resample=2)
        config["arms"]["geometric_adaptive"].update(final_target_snapshot_steps=[0, 2])
    inputs, tasks = {}, []
    for replicate in config["replicates"]:
        matching = [d for d in source_plan["datasets"] if d["role"] == "m1_matched" and d["replicate"] == replicate]
        if len(matching) != 1:
            raise RuntimeError("metadata source dataset selection is not unique")
        dataset = matching[0]
        dataset_id = dataset["dataset_id"]
        folder = source / "datasets" / dataset_id
        prepared = read(folder / "prepared.json")
        table = load_candidate_table(folder / "candidate.json.gz", expected_sha256=prepared["candidate_table_sha256"])
        records = {}
        for index in config["seed_indices"]:
            paths = list((source / "runs").glob(f"{dataset_id}-ordersmc-*-s{index:02d}.json"))
            if len(paths) != 1:
                raise RuntimeError("original SMC seed selection is not unique")
            original = read(paths[0])
            if original["status"] != "complete" or original["candidate_table_sha256"] != prepared["candidate_table_sha256"] or original["plan_sha256"] != source_plan["plan_sha256"]:
                raise RuntimeError("original SMC result identity mismatch")
            records[str(index)] = artifact(paths[0])
            for arm, settings in config["arms"].items():
                seed_record = original["run"]["method_seed"]
                cfg = OrderSMCConfig.from_mapping({
                    **table.provenance["candidate_build_config"], **config["common"],
                    **settings, "seed": seed_record["seed"],
                })
                validate_candidate_provenance(table, config=cfg, missing_policy="require")
                tasks.append(dict(task_id=f"rep{replicate:03d}-{arm}-s{index:02d}",
                                  dataset_id=dataset_id, arm=arm, seed_index=index,
                                  seed_record=seed_record, config=cfg.to_dict()))
        inputs[dataset_id] = dict(candidate=artifact(folder / "candidate.json.gz"),
                                 manifest=artifact(folder / "generator_manifest.json"),
                                 prepared=artifact(folder / "prepared.json"),
                                 candidate_table_sha256=prepared["candidate_table_sha256"],
                                 originals=records)
    folders = ("src", "experiments/src", "experiments/configs", "scripts", "tests", "docs")
    files = {str(p.relative_to(REPO)): sha(p) for folder in folders
             for p in sorted((REPO / folder).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = sha(REPO / "pyproject.toml")
    files["README.md"] = sha(REPO / "README.md")
    plan = dict(schema="ordersmc-sampler-correction-plan-v1", config=config,
                source_plan=artifact(source / "plan.json"), inputs=inputs,
                tasks=tasks, source_files=files)
    plan["plan_sha256"] = canonical_sha(plan)
    for relative, digest in files.items():
        dest = output / "code_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)
        if sha(dest) != digest:
            raise RuntimeError("source changed during snapshot")
    _atomic_json(output / "plan.json", plan)
    return plan


def run(root: Path, index: int, comm):
    plan = checked_plan(root)
    if comm.size != plan["config"]["mpi_ranks"]:
        raise RuntimeError("MPI rank count differs from frozen plan")
    task = plan["tasks"][index]
    dest = root / "runs" / task["task_id"]
    existing = read(dest / "result.json") if (dest / "result.json").exists() else None
    if existing is not None:
        if existing.get("status") != "complete" or existing.get("plan_sha256") != plan["plan_sha256"] or existing.get("task") != task:
            raise RuntimeError("existing result identity mismatch")
        return
    inp = plan["inputs"][task["dataset_id"]]
    table = load_candidate_table(checked_input(inp["candidate"]), expected_sha256=inp["candidate_table_sha256"])
    manifest = read(checked_input(inp["manifest"]))
    cfg = OrderSMCConfig.from_mapping(task["config"])
    started = time.perf_counter()
    result = sample(table, config=cfg, backend="mpi", comm=comm,
                    parent_seed_record=task["seed_record"], provenance_policy="require", verbose=False)
    if comm.rank != 0:
        return
    result = result.to_local_result()
    elapsed = time.perf_counter() - started
    summaries = []
    for snapshot in result.final_target_snapshots:
        row = summarize_posterior(snapshot.edge_probs, truth_graph(manifest),
                                  table.nodes, cfg.max_parents)
        row["edge_probabilities"] = _all_edges(table.nodes, snapshot.edge_probs)
        row["final_target_steps"] = snapshot.final_target_mh_sweeps
        summaries.append(row)
    record = dict(status="complete", plan_sha256=plan["plan_sha256"], task=task,
                  candidate_table_sha256=inp["candidate_table_sha256"],
                  runtime_seconds=elapsed, summary=result.summary(), history=result.history,
                  run_info=result.run_info, snapshots=summaries)
    # Completion marker is written only after all posterior artifacts succeed.
    result.save(dest)
    _atomic_json(dest / "result.json", _json_safe(record))


def aggregate(root: Path):
    plan = checked_plan(root)
    rows, missing = [], []
    by_dataset = {}
    for task in plan["tasks"]:
        path = root / "runs" / task["task_id"] / "result.json"
        if not path.exists():
            missing.append(task["task_id"])
            continue
        record = read(path)
        if record.get("status") != "complete" or record["plan_sha256"] != plan["plan_sha256"] or record["task"] != task:
            raise RuntimeError(f"invalid result: {path}")
        for snapshot in record["snapshots"]:
            group = (task["dataset_id"], task["arm"], snapshot["final_target_steps"])
            by_dataset.setdefault(group, []).append(snapshot)
            rows.append({"dataset_id": group[0], "arm": group[1], "final_target_steps": group[2],
                         "seed_index": task["seed_index"], "runtime_seconds_full_run": record["runtime_seconds"],
                         **{k: v for k, v in snapshot.items() if isinstance(v, (int, float))}})
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    _atomic_json(analysis / "completion.json", dict(status="incomplete" if missing else "complete", missing=missing,
                                                    planned=len(plan["tasks"]), completed=len(plan["tasks"])-len(missing)))
    if missing:
        raise RuntimeError(f"{len(missing)} missing runs; no partial-cohort scientific summary generated")
    for dataset_id, inp in plan["inputs"].items():
        manifest = read(checked_input(inp["manifest"]))
        nodes = sorted(manifest["variable_types"])
        for index, old in inp["originals"].items():
            record = read(checked_input(old))
            probabilities = pips(record)
            summary = summarize_posterior(probabilities, truth_graph(manifest), nodes, 2)
            summary["edge_probabilities"] = _all_edges(nodes, probabilities)
            by_dataset.setdefault((dataset_id, "original_v06_global_swaps", 0), []).append(summary)
            rows.append(dict(dataset_id=dataset_id, arm="original_v06_global_swaps", final_target_steps=0,
                             seed_index=int(index), runtime_seconds_full_run=record["runtime_seconds"],
                             **{k: v for k, v in summary.items() if isinstance(v, (int, float))}))
    stability, uncertainty = [], []
    for (dataset_id, arm, steps), records in by_dataset.items():
        maps = [pips(record) for record in records]
        keys = sorted(set().union(*(set(m) for m in maps)))
        vectors = np.array([[m.get(e, 0.) for e in keys] for m in maps])
        edge_sets = [set(map(tuple, r["representative_edges"])) for r in records]
        skeletons = [{tuple(sorted(e)) for e in edges} for edges in edge_sets]
        def jaccard(sets):
            return float(np.mean([len(a & b)/len(a | b) if a | b else 1.
                                  for a,b in itertools.combinations(sets, 2)]))
        stability.append(dict(dataset_id=dataset_id, arm=arm, final_target_steps=steps, seeds=len(records),
                              mean_directed_jaccard=jaccard(edge_sets), mean_skeleton_jaccard=jaccard(skeletons),
                              all_directed_graphs_identical=all(e == edge_sets[0] for e in edge_sets),
                              mean_pairwise_pip_rmse=float(np.mean([np.sqrt(np.mean((a-b)**2)) for a,b in itertools.combinations(vectors,2)])),
                              maximum_edge_seed_sd=float(np.max(np.std(vectors, axis=0, ddof=1)))))
        uncertainty.append(seed_consensus_table(maps).assign(dataset_id=dataset_id, arm=arm, final_target_steps=steps))
    pd.DataFrame(rows).to_csv(analysis / "per_run_metrics.csv", index=False)
    pd.DataFrame(stability).to_csv(analysis / "seed_stability.csv", index=False)
    pd.concat(uncertainty, ignore_index=True).to_csv(analysis / "independent_seed_uncertainty.csv", index=False)
    _atomic_json(analysis / "interpretation.json", dict(
        status="development_diagnostic", convergence_established=False,
        paired_inputs_and_seeds=True, equal_compute_benchmark=False,
        note="Retain all arms and snapshots. Compare recovery AND independent-seed agreement; no automatic winner or publication promotion.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "aggregate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        plan = freeze(args.config, args.output, smoke=args.smoke)
        print(json.dumps(dict(plan_sha256=plan["plan_sha256"], tasks=len(plan["tasks"]))))
    elif args.action == "run":
        from mpi4py import MPI
        if args.task is None:
            parser.error("run requires --task")
        run(args.output, args.task, MPI.COMM_WORLD)
    else:
        aggregate(args.output)


if __name__ == "__main__":
    main()
