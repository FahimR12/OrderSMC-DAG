"""Fixed-artifact, paired-seed SMC budget comparison with exact references."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import itertools
from pathlib import Path
import shutil
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy

from ordersmc import OrderSMCConfig, sample, validate_candidate_provenance
from ordersmc.persistence import load_candidate_table
from .exact_order import exact_order_dp
from .rank_validation import load_suite, stability
from .readiness import canonical_sha, root_call, score_config, summarize_posterior, validated_probabilities
from .readiness_audits import read, write, sha, pips, rmse
from .seeds import protocol_seed_record

REPO = Path(__file__).resolve().parents[3]


def checked_plan(output, *, sources=False):
    plan = read(output / "plan.json")
    if canonical_sha({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise RuntimeError("budget plan hash mismatch")
    if sources:
        for relative, expected in plan["source_files"].items():
            if sha(REPO / relative) != expected:
                raise RuntimeError(f"frozen budget source mismatch: {relative}")
    return plan


def seed_record(plan, dataset, index):
    return protocol_seed_record(plan["config"]["master_seed"], experiment="rank-budget",
                                cell=dataset["noise"].replace("_", "-"),
                                replicate=dataset["replicate"], component=f"ordersmc-{index:02d}")


def sampling_config(plan, dataset, budget, index):
    return OrderSMCConfig.from_mapping(dataset["base_config"]).with_overrides(
        **plan["config"]["budgets"][budget], seed=seed_record(plan, dataset, index).seed)


def task_list(datasets, config):
    return [dict(dataset_id=d["dataset_id"], budget=budget,
                 seed_indices=list(range(start, min(start + config["seeds_per_task"], config["method_seeds"]))))
            for d in datasets for budget in config["budgets"]
            for start in range(0, config["method_seeds"], config["seeds_per_task"])]


def freeze(source, output, config_path, smoke=False):
    source, output = source.resolve(), output.resolve()
    config = read(config_path)
    if smoke:
        config.update(replicates=[0], method_seeds=2, seeds_per_task=2, mpi_ranks=2,
                      status="implementation-smoke-not-scientific-results")
        config["budgets"] = {
            "baseline": dict(particles=128, num_betas=12, mh_steps=1, mh_steps_resample=2),
            "extended": dict(particles=512, num_betas=12, mh_steps=2, mh_steps_resample=4)}
    if (output / "plan.json").exists():
        old = checked_plan(output, sources=True)
        if old["config"] != config or old["source_root"] != str(source):
            raise FileExistsError("budget settings/source differ; use a new output root")
        for relative, expected in old["input_files"].items():
            if sha(output / relative) != expected:
                raise RuntimeError(f"frozen input mismatch: {relative}")
        return old
    if output.exists():
        raise FileExistsError("unfinished/existing output directory; use a new output root")
    suite = load_suite(source)
    complete = read(source / "analysis/completion.json")
    if complete["status"] != "complete" or complete["missing"] or complete["suite_sha256"] != suite["suite_sha256"]:
        raise RuntimeError("budget comparison requires a complete validated source suite")
    source_hashes = read(source / "analysis/artifact_hashes.json")
    pending = []
    datasets = []
    for noise, index in itertools.product(config["noise_families"], config["replicates"]):
        arm = source / f"fresh-{noise}-rank-native"
        arm_plan = read(arm / "plan.json")
        for relative, expected in arm_plan["source_files"].items():
            if relative.startswith("src/ordersmc/") and sha(REPO / relative) != expected:
                raise RuntimeError("model/sampler implementation differs from source; not a pure budget comparison")
        path = arm / "cases" / f"rep{index:03d}" / f"unrestricted_native-n{config['n_train']:05d}.json"
        case = read(path)
        if case["status"] != "complete" or case["plan_sha256"] != arm_plan["plan_sha256"]:
            raise RuntimeError("source case provenance mismatch")
        candidate = path.with_name(path.stem + "-candidate.json.gz")
        paired_path = path.with_name(path.name.replace("unrestricted_native", "paired_nongaussian"))
        paired_candidate = candidate.with_name(candidate.name.replace("unrestricted_native", "paired_nongaussian"))
        csv = path.parent / "unrestricted_native.csv"
        for artifact in (path, candidate, csv, paired_path, paired_candidate):
            if sha(artifact) != source_hashes[str(artifact.relative_to(source))]:
                raise RuntimeError(f"source artifact changed: {artifact}")
        table = load_candidate_table(candidate, expected_sha256=case["candidate_sha256"])
        paired = load_candidate_table(paired_candidate, expected_sha256=read(paired_path)["candidate_sha256"])
        if any(not np.array_equal(a, b) for field in ("parent_sets", "loglik", "logprior")
               for a, b in zip(getattr(table, field), getattr(paired, field))):
            raise RuntimeError("cannot omit transformed copy: candidate targets differ")
        base = score_config(arm_plan["config"])
        if base.score_method != config["score_method"]:
            raise RuntimeError("wrong source score")
        validate_candidate_provenance(table, config=base, missing_policy="require")
        reference_error = rmse(exact_order_dp(table, compute_graph_size=False).edge_probs,
                               pips(case["references"]["exact-order"]))
        if reference_error > 1e-9:
            raise RuntimeError("stored exact reference does not match the frozen target")
        if not smoke and any(getattr(base, k) != v for k, v in config["budgets"]["baseline"].items()):
            raise RuntimeError("baseline budget differs from previous study")
        dataset_id = f"{noise.replace('_', '-')}-rep{index:03d}"
        dataset = dict(dataset_id=dataset_id, noise=noise, replicate=index, n_train=config["n_train"],
                       base_config=base.to_dict(), candidate_sha256=case["candidate_sha256"],
                       source_case=str(path.relative_to(source)), reference_check_rmse=reference_error)
        datasets.append(dataset)
        for src, name in ((path, "reference.json"), (candidate, "candidate.json.gz"), (csv, "data.csv")):
            pending.append((src, f"datasets/{dataset_id}/{name}"))
    source_files = {str(p.relative_to(REPO)): sha(p)
                    for folder in ("src/ordersmc", "experiments/src/ordersmc_experiments")
                    for p in sorted((REPO / folder).rglob("*.py"))}
    plan = dict(schema="ordersmc-rank-budget-plan-v1", config=config,
                source_root=str(source), source_suite_sha256=suite["suite_sha256"],
                source_files=source_files, input_files={dst: sha(src) for src, dst in pending},
                datasets=datasets, tasks=task_list(datasets, config),
                environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                 pandas=pd.__version__, networkx=nx.__version__))
    seeds = [seed_record(plan, d, i).seed for d in datasets for i in range(config["method_seeds"])]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("inference seed collision")
    for d in datasets:
        for budget in config["budgets"]:
            candidate_config = sampling_config(plan, d, budget, 0).candidate_build_dict()
            if candidate_config != OrderSMCConfig.from_mapping(d["base_config"]).candidate_build_dict():
                raise RuntimeError("budget changed the target configuration")
    plan["plan_sha256"] = canonical_sha(plan)
    output.mkdir(parents=True)
    for src, relative in pending:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
    for relative in source_files:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    write(output / "plan.json", plan)
    return plan


def run_path(output, dataset, budget, index):
    return output / "runs" / f"{dataset['dataset_id']}-{budget}-s{index:02d}.json"


def complete_probabilities(probabilities, nodes):
    """The sampler's sparse edge mapping omits zeros; export a full inventory."""
    pairs = {(u, v) for u in nodes for v in nodes if u != v}
    if not set(probabilities) <= pairs:
        raise RuntimeError("sampler returned an unknown node or self edge")
    return validated_probabilities({edge: probabilities.get(edge, 0.) for edge in sorted(pairs)})


def validate_run(record, plan, dataset, budget, index):
    expected = dict(plan_sha256=plan["plan_sha256"], dataset_id=dataset["dataset_id"], budget=budget,
                    seed_index=index, seed=asdict(seed_record(plan, dataset, index)),
                    config=sampling_config(plan, dataset, budget, index).to_dict(),
                    candidate_sha256=dataset["candidate_sha256"], mpi_ranks=plan["config"]["mpi_ranks"],
                    status="complete")
    # JSON serialization converts the tuple-valued snapshot field to a list.
    if canonical_sha({k: record.get(k) for k in expected}) != canonical_sha(expected):
        raise RuntimeError("saved run identity/configuration mismatch")
    probabilities = pips(record)
    if not probabilities or any(not np.isfinite(x) or not 0 <= x <= 1 for x in probabilities.values()):
        raise RuntimeError("invalid saved probabilities")


def run(output, task_index, mpi=False):
    comm = None
    if mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    plan = root_call(comm, lambda: checked_plan(output, sources=True))
    if (comm.size if mpi else 1) != plan["config"]["mpi_ranks"]:
        raise RuntimeError("unexpected MPI layout")
    if not 0 <= task_index < len(plan["tasks"]):
        raise ValueError("invalid task index")
    task = plan["tasks"][task_index]
    dataset = next(d for d in plan["datasets"] if d["dataset_id"] == task["dataset_id"])
    budget = task["budget"]
    def prepare():
        folder = output / "datasets" / dataset["dataset_id"]
        for name in ("reference.json", "candidate.json.gz", "data.csv"):
            path = folder / name
            if sha(path) != plan["input_files"][str(path.relative_to(output))]:
                raise RuntimeError("fixed input changed")
        return load_candidate_table(folder / "candidate.json.gz", expected_sha256=dataset["candidate_sha256"]), read(folder / "reference.json")
    table, reference = root_call(comm, prepare)
    truth = nx.DiGraph()
    truth.add_nodes_from(table.nodes)
    truth.add_edges_from(reference["truth_edges"])
    exact = pips(reference["references"]["exact-order"])
    for index in task["seed_indices"]:
        path = run_path(output, dataset, budget, index)
        def already_complete():
            if not path.exists():
                return False
            validate_run(read(path), plan, dataset, budget, index)
            return True
        if root_call(comm, already_complete):
            continue
        cfg = sampling_config(plan, dataset, budget, index)
        seed = asdict(seed_record(plan, dataset, index))
        start = time.perf_counter()
        result = sample(table, config=cfg, backend="mpi" if mpi else "local", comm=comm,
                        parent_seed_record=seed if mpi else None, verbose=False, provenance_policy="require")
        elapsed = time.perf_counter() - start
        def save():
            local = result.to_local_result() if mpi else result
            probabilities = complete_probabilities(local.edge_probs, table.nodes)
            record = dict(schema="ordersmc-rank-budget-run-v1", status="complete", plan_sha256=plan["plan_sha256"],
                          dataset_id=dataset["dataset_id"], budget=budget, seed_index=index, seed=seed,
                          config=cfg.to_dict(), mpi_ranks=plan["config"]["mpi_ranks"],
                          candidate_sha256=dataset["candidate_sha256"], sampler_wall_seconds=elapsed,
                          rng_lineage=local.run_info.get("rng_lineage"),
                          own_target_pip_rmse=rmse(probabilities, exact),
                          **summarize_posterior(probabilities, truth, table.nodes, cfg.max_parents))
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            write(temporary, record)
            temporary.replace(path)
            print(f"complete: {dataset['dataset_id']} {budget} seed={index}", flush=True)
        root_call(comm, save)


def aggregate(output):
    plan = checked_plan(output, sources=True)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    write(analysis / "completion.json", dict(status="checking", plan_sha256=plan["plan_sha256"]))
    rows, summaries, edge_rows, identities, missing = [], [], [], {}, []
    for dataset in plan["datasets"]:
        folder = output / "datasets" / dataset["dataset_id"]
        reference = read(folder / "reference.json")
        if sha(folder / "reference.json") != plan["input_files"][str((folder / "reference.json").relative_to(output))]:
            raise RuntimeError("fixed reference changed")
        exact = pips(reference["references"]["exact-order"])
        exact_edges = set(map(tuple, reference["references"]["exact-order"]["representative_edges"]))
        common = {k: dataset[k] for k in ("dataset_id", "noise", "replicate", "n_train")}
        for budget in plan["config"]["budgets"]:
            records = []
            for index in range(plan["config"]["method_seeds"]):
                path = run_path(output, dataset, budget, index)
                if not path.exists():
                    missing.append(str(path.relative_to(output)))
                    continue
                record = read(path)
                validate_run(record, plan, dataset, budget, index)
                if set(pips(record)) != set(exact):
                    raise RuntimeError("saved probability inventory differs from exact reference")
                if not np.isclose(record["own_target_pip_rmse"], rmse(pips(record), exact), atol=1e-12, rtol=0):
                    raise RuntimeError("saved reference error mismatch")
                records.append(record)
                identities[str(path.relative_to(output))] = sha(path)
                rows.append(dict(**common, budget=budget, seed_index=index,
                                 **{k: record[k] for k in ("directed_f1", "skeleton_f1", "directed_auprc", "directed_brier", "own_target_pip_rmse")}))
            if len(records) != plan["config"]["method_seeds"]:
                continue
            group = {**reference, "smc_runs": records}
            counts = Counter(tuple(sorted(map(tuple, r["representative_edges"]))) for r in records)
            errors = np.array([r["own_target_pip_rmse"] for r in records])
            flags = plan["config"]["diagnostic_flags"]
            summaries.append(dict(**common, budget=budget, seed_count=len(records),
                unique_directed_graphs=len(counts), modal_graph_fraction=max(counts.values())/len(records),
                mean_own_target_pip_rmse=float(errors.mean()), p95_own_target_pip_rmse=float(np.quantile(errors, .95)),
                sampling_flag=bool(errors.mean() > flags["mean_smc_own_target_pip_rmse"] or np.quantile(errors, .95) > flags["p95_smc_own_target_pip_rmse"]),
                exact_representative_match_fraction=sum(set(edges) == exact_edges for edges in counts.elements())/len(records),
                **stability(group, plan["config"]["orientation_display_threshold"])))
            node_names = sorted(reference["variable_types"])
            for u, v in itertools.combinations(node_names, 2):
                p, q = exact.get((u, v), 0.), exact.get((v, u), 0.)
                probs = [pips(r) for r in records]
                states = Counter("forward" if [u, v] in r["representative_edges"] else
                                 "reverse" if [v, u] in r["representative_edges"] else "absent" for r in records)
                edge_rows.append(dict(**common, budget=budget, u=u, v=v,
                    exact_forward=p, exact_reverse=q, exact_absent=1-p-q,
                    exact_direction_margin=abs(p-q), varied=len(states)>1,
                    forward_count=states["forward"], reverse_count=states["reverse"], absent_count=states["absent"],
                    forward_pip_mean=float(np.mean([r.get((u,v),0.) for r in probs])),
                    forward_pip_sd=float(np.std([r.get((u,v),0.) for r in probs], ddof=1))))
    completion = dict(status="incomplete" if missing else "complete", missing=missing,
                      completed_runs=len(rows), expected_runs=len(plan["datasets"])*len(plan["config"]["budgets"])*plan["config"]["method_seeds"],
                      plan_sha256=plan["plan_sha256"])
    if missing:
        write(analysis / "completion.json", completion)
        raise RuntimeError(f"{len(missing)} runs missing; no complete-cohort summary")
    frame, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    frame.to_csv(analysis / "runs.csv", index=False)
    summary.to_csv(analysis / "stability.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(analysis / "edge_stability.csv", index=False)
    metrics = ["directed_f1", "skeleton_f1", "directed_auprc", "directed_brier", "own_target_pip_rmse"]
    means = frame.groupby(["dataset_id", "noise", "replicate", "budget"], as_index=False)[metrics].mean()
    means.to_csv(analysis / "dataset_means.csv", index=False)
    means.groupby(["noise", "budget"], as_index=False)[metrics].mean().to_csv(analysis / "recovery_summary.csv", index=False)
    measures = ["directed_mean_jaccard", "skeleton_mean_jaccard", "display_mean_jaccard", "mean_pairwise_pip_rmse", "mean_own_target_pip_rmse", "modal_graph_fraction", "unique_directed_graphs"]
    left, right = [summary[summary.budget == b].set_index("dataset_id") for b in ("baseline", "extended")]
    (right[measures] - left[measures]).to_csv(analysis / "paired_extended_minus_baseline.csv")
    write(analysis / "run_hashes.json", identities)
    write(analysis / "completion.json", completion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "aggregate"])
    parser.add_argument("--source", type=Path, default=REPO / "results/rank-validation/20260917-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "experiments/configs/rank_budget_v1.json")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.source, args.output, args.config, args.smoke)["plan_sha256"])
    elif args.action == "aggregate":
        aggregate(args.output)
    else:
        try:
            run(args.output, args.task, args.mpi)
        except BaseException:
            if args.mpi:
                from mpi4py import MPI
                import traceback
                traceback.print_exc()
                MPI.COMM_WORLD.Abort(1)
            raise


if __name__ == "__main__":
    main()
