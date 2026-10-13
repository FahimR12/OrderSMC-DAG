"""Paired V1 diagnostic restoring the legacy number of MH proposals, not time."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from collections import defaultdict
from pathlib import Path
import shutil

import networkx as nx
import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig, validate_candidate_provenance
from ordersmc.persistence import load_candidate_table
from . import catchup
from .accuracy_gate import dataset_rmse_gate
from .execution import _atomic_json
from .readiness import canonical_sha, summarize_posterior
from .readiness_audits import read, sha, pips, rmse
from .sampler_corrections import artifact, checked_input
from .rank_budget import complete_probabilities

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "experiments/configs/v1_effort_match_v1.json"


def proposal_count(history):
    counts = [float(row["proposed_moves"]) for row in history]
    if not counts or any(not np.isfinite(x) or x < 0 or not x.is_integer() for x in counts):
        raise ValueError("history must contain finite nonnegative integer proposal counts")
    return sum(map(int, counts))


def effort_steps(legacy_history, adaptive_history, particles):
    if isinstance(particles, bool) or int(particles) != particles or particles < 1:
        raise ValueError("positive integer particles required")
    legacy, adaptive = proposal_count(legacy_history), proposal_count(adaptive_history)
    if legacy < adaptive:
        raise ValueError("adaptive run already exceeds reference effort; cannot remove moves")
    if legacy % particles or adaptive % particles:
        raise ValueError("proposal counts are not whole proposals per particle")
    return dict(legacy_proposals=legacy, adaptive_proposals=adaptive,
                extra_steps=(legacy-adaptive)//particles)


def freeze(root, config_path=DEFAULT_CONFIG):
    if root.exists():
        raise FileExistsError("use a fresh effort-diagnostic root")
    config = read(config_path)
    source = (REPO / config["source"]).resolve()
    original = catchup.checked_source(source / "plan.json")
    if original["plan_sha256"] != config["source_plan_sha256"]:
        raise RuntimeError("unexpected source catch-up plan")
    completion = read(source / "analysis" / config["cohort"] / "completion.json")
    if completion.get("status") != "complete" or completion["plan_sha256"] != original["plan_sha256"] or completion["missing"]:
        raise RuntimeError("effort matching requires the complete source cohort")
    for relative, digest in original["source_files"].items():
        if relative.startswith("src/ordersmc/") and sha(REPO / relative) != digest:
            raise RuntimeError("core sampler changed; this would not be a pure budget diagnostic")
    datasets = copy.deepcopy([d for d in original["datasets"] if d["cohort"] == config["cohort"]])
    selected = {d["key"] for d in datasets}
    for d in datasets:
        table = load_candidate_table(checked_input(d["candidate"]), expected_sha256=d["candidate_sha256"])
        cfg = OrderSMCConfig.from_mapping(table.provenance["candidate_build_config"])
        validate_candidate_provenance(table, config=cfg, missing_policy="require")
    fits = []
    for oldfit in original["fits"]:
        if oldfit["dataset_key"] not in selected:
            continue
        if oldfit["mpi_ranks"] != 1 or "external" in oldfit:
            raise RuntimeError("this paired diagnostic preserves V1's serial layout")
        adaptive, path = catchup.fit_record(source, original, oldfit)
        legacy = catchup.located(oldfit["original"])
        cfg = OrderSMCConfig.from_mapping(oldfit["config"])
        if cfg.final_target_snapshot_steps != (0,):
            raise RuntimeError("source adaptive fit already included extra final-target moves")
        effort = effort_steps(legacy["diagnostics"]["history"], adaptive["history"], cfg.particles)
        fit = copy.deepcopy(oldfit)
        fit.update(fit_id=oldfit["fit_id"] + "--effort-matched", effort=effort,
                   adaptive_source=artifact(path), adaptive_weights=artifact(path.parent / "weights.npy"),
                   config=cfg.with_overrides(final_target_snapshot_steps=tuple(sorted({0, effort["extra_steps"]}))).to_dict())
        fits.append(fit)
    groups = defaultdict(list)
    for f in fits:
        groups[f["dataset_key"]].append(f["fit_id"])
    tasks = [dict(cohort=config["cohort"], mpi_ranks=1, fit_ids=ids[start:start+config["seeds_per_task"]])
             for ids in groups.values() for start in range(0, len(ids), config["seeds_per_task"])]
    folders = ("src", "experiments/src", "experiments/configs", "experiments/adapters",
               "experiments/fixtures", "experiments/validation", "scripts", "tests", "experiments/tests", "docs",
               "summary_statistics_multiopmic_data_pathway_molecule_level")
    files = {str(p.relative_to(REPO)): sha(p) for folder in folders for p in sorted((REPO/folder).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        files[name] = sha(REPO/name)
    plan = dict(schema="ordersmc-v1-effort-match-plan-v1", config={**config,
                    "exact_rmse_flags": original["config"]["exact_rmse_flags"]},
                evidence=[artifact(source/"plan.json"), artifact(source/"analysis"/config["cohort"]/"completion.json")],
                source_root=str(source), datasets=datasets, fits=fits, tasks=tasks,
                comparisons=[r for r in original["comparisons"] if r["dataset_key"] in selected], source_files=files)
    plan["plan_sha256"] = canonical_sha(plan)
    for relative, digest in files.items():
        dest = root / "code_snapshot" / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/relative, dest)
        if sha(dest) != digest:
            raise RuntimeError("source changed while freezing")
    _atomic_json(root/"plan.json", plan)
    shutil.copy2(REPO/"docs/V1_EFFORT_MATCH.md", root/"README.md")
    _atomic_json(root/"inventory.json", dict(plan_sha256=plan["plan_sha256"], datasets=len(datasets),
        fits=len(fits), array_tasks=len(tasks), extra_steps_min=min(f["effort"]["extra_steps"] for f in fits),
        extra_steps_max=max(f["effort"]["extra_steps"] for f in fits)))
    return plan


def preflight(root):
    plan = catchup.checked_plan(root)
    for fit in plan["fits"]:
        adaptive = read(checked_input(fit["adaptive_source"]))
        legacy = catchup.located(fit["original"])
        checked_input(fit["adaptive_weights"])
        if effort_steps(legacy["diagnostics"]["history"], adaptive["history"], fit["config"]["particles"]) != fit["effort"]:
            raise RuntimeError("frozen effort rule no longer matches source histories")
    catchup.preflight(root)


def validate_replay(root, plan, fit):
    record, path = catchup.fit_record(root, plan, fit)
    source = read(checked_input(fit["adaptive_source"]))
    before = next(s for s in record["snapshots"] if s["final_target_steps"] == 0)
    original = next(s for s in source["snapshots"] if s["final_target_steps"] == 0)
    if pips(before) != pips(original):
        raise RuntimeError("zero-extra-move PIPs did not exactly replay the adaptive source")
    if record["run_info"]["tempering_schedule"] != source["run_info"]["tempering_schedule"]:
        raise RuntimeError("adaptive schedule changed during replay")
    tempering = [h for h in record["history"] if h.get("phase") == "tempering"]
    if proposal_count(tempering) != fit["effort"]["adaptive_proposals"]:
        raise RuntimeError("adaptive proposal count changed")
    if proposal_count(record["history"]) != fit["effort"]["legacy_proposals"]:
        raise RuntimeError("final total proposal count is not effort matched")
    if not np.array_equal(np.load(path.parent/"weights.npy"), np.load(checked_input(fit["adaptive_weights"]))):
        raise RuntimeError("final-target MH unexpectedly changed particle weights")
    return dict(status="passed", plan_sha256=plan["plan_sha256"], fit_id=fit["fit_id"],
                result_sha256=sha(path), snapshot_zero_exact_replay=True, weights_unchanged=True,
                **fit["effort"])


def run(root, task_index):
    catchup.run(root, task_index)
    plan = catchup.checked_plan(root)
    for fit_id in plan["tasks"][task_index]["fit_ids"]:
        fit = next(f for f in plan["fits"] if f["fit_id"] == fit_id)
        report = validate_replay(root, plan, fit)
        _atomic_json(root/"validation/replay"/(fit_id+".json"), report)


def aggregate(root):
    plan = catchup.checked_plan(root); analysis = root/"analysis"
    missing, hashes = [], {}
    for f in plan["fits"]:
        try:
            audit = read(root/"validation/replay"/(f["fit_id"]+".json"))
            if audit != validate_replay(root, plan, f):
                raise RuntimeError("replay audit changed")
            hashes[str(root/"runs"/f["fit_id"]/"result.json")] = audit["result_sha256"]
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            missing.append(dict(fit_id=f["fit_id"], reason=str(error)))
    if missing:
        _atomic_json(analysis/"completion.json", dict(status="incomplete", plan_sha256=plan["plan_sha256"], missing=missing))
        raise RuntimeError("missing or unverified paired fits; no incomplete-cohort scientific summary")
    rows, stable, pooled, metrics = [], [], [], list(catchup.METRICS)
    for d in plan["datasets"]:
        truth = nx.DiGraph(); truth.add_nodes_from(d["nodes"]); truth.add_edges_from(d["truth_edges"])
        exact = pips(catchup.located(d["references"]["exact-order"]))
        arms = defaultdict(list)
        for f in [f for f in plan["fits"] if f["dataset_key"] == d["key"]]:
            record, _ = catchup.fit_record(root, plan, f)
            matched = next(s for s in record["snapshots"] if s["final_target_steps"] == f["effort"]["extra_steps"])
            adaptive = next(s for s in record["snapshots"] if s["final_target_steps"] == 0)
            arms["adaptive_matched_effort"].append((f["seed_index"], matched, f["effort"]["extra_steps"]))
            arms["adaptive_original_budget"].append((f["seed_index"], adaptive, 0))
            arms["legacy"].append((f["seed_index"], catchup.located(f["original"]), 0))
        for c in plan["comparisons"]:
            if c["dataset_key"] == d["key"] and c["method"] == "order-mcmc":
                arms["order-mcmc"].append((c["seed_index"], catchup.located(c["original"]), 0))
        common = dict(dataset_key=d["key"], cell_id=d["spec"]["cell_id"], budget="baseline",
                      n_train=d["spec"]["n_train"], dimension=len(d["nodes"]), role="evaluation")
        for arm, records in arms.items():
            maps = [complete_probabilities(pips(r), d["nodes"]) for _,r,_ in records]
            summaries = [summarize_posterior(m, truth, d["nodes"], d["max_parents"]) for m in maps]
            for (seed, _, extra), m, s in zip(records, maps, summaries):
                rows.append(dict(**common, method=arm, seed_index=seed, extra_final_steps=extra,
                    **{k: s.get(k) for k in metrics if k != "own_target_pip_rmse"}, own_target_pip_rmse=rmse(m,exact)))
            edges = [set(map(tuple, s["representative_edges"])) for s in summaries]
            skeletons = [{tuple(sorted(e)) for e in row} for row in edges]
            def jaccard(sets):
                vals = [len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(sets,2)]
                return float(np.mean(vals)) if vals else None
            vectors = np.array([[m[e] for e in maps[0]] for m in maps])
            stable.append(dict(**common, method=arm, seeds=len(records),
                mean_directed_jaccard=jaccard(edges), mean_skeleton_jaccard=jaccard(skeletons),
                all_directed_graphs_identical=all(e == edges[0] for e in edges),
                mean_pairwise_pip_rmse=float(np.mean([np.sqrt(np.mean((a-b)**2)) for a,b in itertools.combinations(vectors,2)]))))
            mean = {e: float(np.mean([m[e] for m in maps])) for e in maps[0]}
            pooled.append(dict(**common, method=arm, **summarize_posterior(mean, truth, d["nodes"], d["max_parents"])))
    analysis.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(analysis/"per_run_metrics.csv", index=False)
    pd.DataFrame(stable).to_csv(analysis/"seed_stability.csv", index=False)
    _atomic_json(analysis/"mean_pip_graphs.json", dict(graphs=pooled))
    dataset_means = frame.groupby(["dataset_key","cell_id","method"],as_index=False)[metrics].mean()
    dataset_means.to_csv(analysis/"dataset_metrics.csv",index=False)
    dataset_means.groupby(["cell_id","method"],as_index=False)[metrics].mean().to_csv(analysis/"cell_summary.csv",index=False)
    gates = {arm: dataset_rmse_gate(frame, plan["config"]["exact_rmse_flags"], method=arm,
                                   group_columns=("cell_id","budget")) for arm in sorted(frame.method.unique())}
    _atomic_json(analysis/"exact_rmse_flags.json", dict(primary_arm=plan["config"]["primary_arm"], arms=gates))
    contrasts=[]
    for cell, group in dataset_means.groupby("cell_id"):
        primary=group[group.method=="adaptive_matched_effort"].set_index("dataset_key")
        for arm in ("adaptive_original_budget","legacy","order-mcmc"):
            other=group[group.method==arm].set_index("dataset_key")
            if set(primary.index)!=set(other.index):
                raise RuntimeError("unpaired comparator datasets")
            for metric in metrics:
                delta=(primary[metric]-other[metric]).to_numpy(dtype=float)
                rng=np.random.Generator(np.random.PCG64DXSM(20260919))
                boot=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
                contrasts.append(dict(cell_id=cell, comparator=arm, metric=metric, datasets=len(delta),
                    matched_minus_comparator=float(delta.mean()),ci_lower=float(np.quantile(boot,.025)),ci_upper=float(np.quantile(boot,.975))))
    pd.DataFrame(contrasts).to_csv(analysis/"paired_dataset_contrasts.csv",index=False)
    _atomic_json(analysis/"artifact_hashes.json", hashes)
    _atomic_json(analysis/"completion.json", dict(status="complete", plan_sha256=plan["plan_sha256"],
        completed_fits=len(plan["fits"]), all_replays_verified=True,
        primary_accuracy_passed=gates[plan["config"]["primary_arm"]]["all_passed"],
        publication_ready=False, equal_compute=False, automatic_default_promotion=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["freeze","preflight","run","aggregate"])
    p.add_argument("--output",type=Path,required=True);p.add_argument("--task",type=int)
    args=p.parse_args()
    if args.action=="freeze":
        plan=freeze(args.output);print(json.dumps(dict(fits=len(plan["fits"]),tasks=len(plan["tasks"]),plan_sha256=plan["plan_sha256"])))
    elif args.action=="preflight":preflight(args.output)
    elif args.action=="run":run(args.output,args.task)
    else:aggregate(args.output)


if __name__=="__main__":main()
