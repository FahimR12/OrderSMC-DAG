"""Frozen native-versus-rank-native development pilot; never selects a winner."""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy

from ordersmc import uncertainty_aware_graph
from ordersmc.persistence import load_candidate_table
from . import readiness
from .readiness_audits import read, write, sha, pips, rmse
from .readiness_followup import mask_metrics


REPO = Path(__file__).resolve().parents[3]


def arm_configs(spec, base):
    arms = []
    for cohort, noise, seed, count, seeds, ranks in [
        ("replay", "gaussian", spec["replay_master_seed"], spec["replay_replicates"], 0, 1),
        *[("fresh", noise, spec["fresh_master_seed"], spec["fresh_replicates_per_noise"],
           spec["method_seeds"], spec["mpi_ranks"]) for noise in spec["fresh_noise_families"]],
    ]:
        for score in spec["score_methods"]:
            config = copy.deepcopy(base)
            config.update(master_seed=seed, replicates=count, method_seeds=seeds,
                          mpi_ranks=ranks, score_method=score, sample_sizes=spec["sample_sizes"],
                          paired_method_seeds=True, status=spec["status"])
            config["generator"]["continuous_noise"] = noise
            if spec.get("smoke"):
                config["ordersmc"] = dict(particles=128, num_betas=24, mh_steps=3, mh_steps_resample=6)
            arms.append(dict(name=f"{cohort}-{noise}-{score}", cohort=cohort, noise=noise,
                             score=score, config=config))
    return arms


def freeze(output, config_path, smoke=False):
    spec = read(config_path)
    if smoke:
        spec.update(replay_replicates=1, fresh_replicates_per_noise=1,
                    sample_sizes=[80], method_seeds=2, mpi_ranks=2, smoke=True,
                    status="implementation-smoke-not-scientific-results")
    base = read(REPO / "experiments/configs/section2_readiness_v1.json")
    arms = arm_configs(spec, base)
    output.mkdir(parents=True, exist_ok=True)
    previous = read(output / "suite.json") if (output / "suite.json").exists() else None
    if previous and previous["spec"] != spec:
        raise FileExistsError("suite settings differ; use a new output root")
    tasks = []
    for arm in arms:
        folder = output / arm["name"]
        folder.mkdir(exist_ok=True)
        path = folder / "config.json"
        if path.exists() and read(path) != arm["config"]:
            raise FileExistsError(f"arm settings differ: {path}")
        if not path.exists():
            write(path, arm["config"])
        plan = readiness.freeze(path, folder)
        arm["plan_sha256"] = plan["plan_sha256"]
        for index in range(arm["config"]["replicates"]):
            tasks.append(dict(arm=arm["name"], index=index, cohort=arm["cohort"],
                              mpi_ranks=arm["config"]["mpi_ranks"]))
    suite = dict(schema="ordersmc-rank-validation-suite-v1", spec=spec, arms=arms, tasks=tasks,
                 scipy_version=scipy.__version__, pandas_version=pd.__version__,
                 networkx_version=nx.__version__)
    if not smoke:
        historical = REPO / "results/section2-readiness/seed-20260918-v2"
        references = {}
        for path in sorted((historical / "cases").glob("rep*/*-n*.json")):
            old = read(path)
            references[str(path.relative_to(historical / "cases"))] = {
                "data_sha256": old["data_sha256"], "truth_edges": old["truth_edges"],
                "references": old["references"], "source_sha256": sha(path)}
        expected = spec["replay_replicates"] * len(base["variants"]) * len(spec["sample_sizes"])
        if len(references) != expected:
            raise RuntimeError("historical replay requires the complete saved reference cohort")
        suite["historical_references"] = references
    suite["suite_sha256"] = readiness.canonical_sha(suite)
    if previous and previous != suite:
        raise FileExistsError("suite source/environment differs; use a new output root")
    write(output / "suite.json", suite)
    return suite


def load_suite(output):
    suite = read(output / "suite.json")
    payload = {k: v for k, v in suite.items() if k != "suite_sha256"}
    if readiness.canonical_sha(payload) != suite["suite_sha256"]:
        raise RuntimeError("suite hash mismatch")
    for arm in suite["arms"]:
        plan = read(output / arm["name"] / "plan.json")
        if plan["plan_sha256"] != arm["plan_sha256"]:
            raise RuntimeError("arm plan mismatch")
        if readiness.canonical_sha({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
            raise RuntimeError("arm plan hash mismatch")
    return suite


def agreement(edge_sets):
    pairs = [len(a & b) / len(a | b) if a | b else 1.
             for a, b in itertools.combinations(edge_sets, 2)]
    return dict(mean_jaccard=float(np.mean(pairs)), all_identical=min(pairs) == 1.)


def stability(result, threshold):
    directed, skeleton, masked, masks = [], [], [], []
    truth = set(map(tuple, result["truth_edges"]))
    for run in result["smc_runs"]:
        edges = set(map(tuple, run["representative_edges"]))
        graph = nx.DiGraph()
        graph.add_nodes_from(result["variable_types"])
        graph.add_edges_from(edges)
        display = uncertainty_aware_graph(pips(run), graph, orientation_threshold=threshold)
        directed.append(edges)
        skeleton.append({tuple(sorted(e)) for e in edges})
        masked.append({("directed", *e) for e in display.directed_edges}
                      | {("undirected", *e) for e in display.undirected_edges})
        masks.append(mask_metrics(display, truth))
    average = lambda key: (float(np.mean([m[key] for m in masks if m[key] is not None]))
                           if any(m[key] is not None for m in masks) else None)
    return {**{f"{name}_{key}": value for name, sets in
               [("directed", directed), ("skeleton", skeleton), ("display", masked)]
               for key, value in agreement(sets).items()},
            "mean_pairwise_pip_rmse": float(np.mean([
                rmse(pips(a), pips(b)) for a, b in itertools.combinations(result["smc_runs"], 2)])),
            **{key: average(key) for key in masks[0]}}


def aggregate(output):
    suite = load_suite(output)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    missing = []
    for arm in suite["arms"]:
        cfg = arm["config"]
        for index, variant, n in itertools.product(range(cfg["replicates"]), cfg["variants"], cfg["sample_sizes"]):
            path = output / arm["name"] / "cases" / f"rep{index:03d}" / f"{variant}-n{n:05d}.json"
            if not path.exists():
                missing.append(str(path.relative_to(output)))
    write(analysis / "completion.json", dict(status="incomplete" if missing else "checking",
          missing=missing, suite_sha256=suite["suite_sha256"]))
    if missing:
        raise RuntimeError(f"{len(missing)} conditions missing; no complete-cohort summaries written")
    rows, stable, checks, artifacts, paired_data = [], [], [], {}, {}
    for arm in suite["arms"]:
        folder, cfg = output / arm["name"], arm["config"]
        readiness.aggregate(folder)
        data = pd.read_csv(folder / "analysis/runs.csv")
        for key in ("cohort", "noise", "score"):
            data[key] = arm[key]
        rows.append(data)
        for index, n in itertools.product(range(cfg["replicates"]), cfg["sample_sizes"]):
            cases, tables = {}, {}
            for variant in cfg["variants"]:
                path = folder / "cases" / f"rep{index:03d}" / f"{variant}-n{n:05d}.json"
                case = read(path)
                candidate = path.with_name(f"{variant}-n{n:05d}-candidate.json.gz")
                table = load_candidate_table(candidate, expected_sha256=case["candidate_sha256"])
                csv = path.parent / f"{variant}.csv"
                if sha(csv) != case["data_sha256"]:
                    raise RuntimeError(f"dataset hash mismatch: {csv}")
                pair_key = (arm["cohort"], arm["noise"], index, variant, n)
                identity = (case["data_sha256"], case["truth_edges"])
                if pair_key in paired_data and paired_data[pair_key] != identity:
                    raise RuntimeError("the two scores were not evaluated on identical data/truth")
                paired_data[pair_key] = identity
                if arm["cohort"] == "replay" and not suite["spec"].get("smoke"):
                    old = suite["historical_references"][f"rep{index:03d}/{path.name}"]
                    if identity != (old["data_sha256"], old["truth_edges"]):
                        raise RuntimeError("replayed data/truth differ from the historical cohort")
                    if arm["score"] == "native":
                        for method in ("exact-order", "exact-structure"):
                            if rmse(pips(case["references"][method]), pips(old["references"][method])) > 1e-8:
                                raise RuntimeError("unchanged native score does not reproduce historical reference")
                cases[variant], tables[variant] = case, table
                for artifact in (path, candidate, csv):
                    artifacts[str(artifact.relative_to(output))] = sha(artifact)
                if case["smc_runs"]:
                    stable.append(dict(cohort=arm["cohort"], noise=arm["noise"], score=arm["score"],
                                       replicate=index, variant=variant, n_train=n,
                                       **stability(case, suite["spec"]["orientation_display_threshold"])))
            left, right = [tables[v] for v in cfg["variants"]]
            if arm["score"] == "rank-native":
                gap = max(float(np.max(np.abs(a - b))) for a, b in zip(left.loglik, right.loglik))
                pip_gap = rmse(pips(cases[cfg["variants"][0]]["references"]["exact-order"]),
                               pips(cases[cfg["variants"][1]]["references"]["exact-order"]))
                checks.append(dict(arm=arm["name"], replicate=index, n_train=n,
                                   maximum_paired_score_error=gap, paired_exact_pip_rmse=pip_gap))
                if gap > 1e-9 or pip_gap > 1e-9:
                    raise RuntimeError("rank-mode measurement-invariance check failed")
    runs = pd.concat(rows, ignore_index=True)
    runs.to_csv(analysis / "runs.csv", index=False)
    keys = ["cohort", "noise", "score", "variant", "n_train", "replicate", "method"]
    metrics = ["directed_f1", "skeleton_f1", "directed_auprc", "directed_brier", "own_target_pip_rmse"]
    means = runs.groupby(keys, as_index=False)[metrics].mean()
    means.to_csv(analysis / "dataset_means.csv", index=False)
    means.groupby([k for k in keys if k != "replicate"], as_index=False)[metrics].mean().to_csv(analysis / "summary.csv", index=False)
    paired = means.pivot(index=[k for k in keys if k != "score"], columns="score", values=metrics)
    differences = pd.DataFrame({m: paired[m]["rank-native"] - paired[m]["native"] for m in metrics})
    differences.reset_index().to_csv(analysis / "paired_differences_rank_minus_native.csv", index=False)
    pd.DataFrame(stable).to_csv(analysis / "stability.csv", index=False)
    write(analysis / "invariance_checks.json", checks)
    write(analysis / "artifact_hashes.json", artifacts)
    count = len(means[means.method == "exact-order"])
    fits = int((runs.method == "ordersmc").sum())
    write(analysis / "completion.json", dict(status="complete", missing=[], conditions=count,
          completed_smc_runs=fits, suite_sha256=suite["suite_sha256"],
          interpretation="Development pilot; no automatic winner/default change; seed averages within datasets."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "aggregate"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO / "experiments/configs/rank_validation_v1.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task", type=int)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.output, args.config, args.smoke)["suite_sha256"])
    elif args.action == "aggregate":
        aggregate(args.output)
    else:
        try:
            suite = load_suite(args.output)
            if args.task is None or not 0 <= args.task < len(suite["tasks"]):
                raise ValueError("invalid suite task index")
            task = suite["tasks"][args.task]
            readiness.run_case(args.output / task["arm"], task["index"], args.mpi)
        except BaseException:
            if args.mpi:
                from mpi4py import MPI
                import traceback
                traceback.print_exc()
                MPI.COMM_WORLD.Abort(1)
            raise


if __name__ == "__main__":
    main()
