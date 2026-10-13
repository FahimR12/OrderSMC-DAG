"""Strong scaling and truth-based quality/time, without a posterior-reference gate.

Empirical agreement with MCMC is deliberately not called posterior accuracy.
The previous timing edition and its failed convergence checks remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import catchup
from .execution import _atomic_json
from .readiness_audits import read, pips
from .sampler_corrections import checked_input
from .timing_pilot import seed_record


def make_tasks(config, datasets):
    tasks, bundles = [], []
    if config["scaling_particles"] % max(config["ranks"]):
        raise ValueError("scaling particles must divide evenly across ranks")
    if config["arms"] != ["dimension"]:
        raise ValueError("this follow-up freezes the dimension prior only")
    for d in datasets:
        for seed in range(config["seeds"]):
            group = []
            def add(method, ranks, budget, purpose):
                task = dict(dataset_key=d["key"], arm="dimension", seed_index=seed,
                    method=method, ranks=ranks, reference=False, purpose=purpose,
                    **({"particles": budget} if method == "ordersmc" else {"wall_seconds": budget}))
                task["task_id"] = f"{d['key']}--dimension--{method}--{purpose}--r{ranks}--b{budget}--s{seed}"
                group.append(len(tasks)); tasks.append(task)
            for ranks in config["ranks"]:
                add("ordersmc", ranks, config["scaling_particles"], "fixed-work-scaling")
                budgets = list(config["mcmc_seconds"])
                if ranks == 64: budgets.append(config["extra_mcmc_seconds_at_64"])
                for seconds in budgets: add("order-mcmc", ranks, seconds, "accuracy-time")
            for particles in config["adaptive_quality_particles"]:
                add("ordersmc", config["adaptive_quality_ranks"], particles, "accuracy-time")
            record = seed_record(config, d["key"], seed, "schedule")
            order = np.random.Generator(np.random.PCG64DXSM(record["seed"])).permutation(group).tolist()
            bundles.append(dict(dataset_key=d["key"], seed_index=seed, tasks=order, schedule_seed=record))
    return tasks, bundles, []


def digest(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def pair_shd(left, right):
    """Unordered-pair state SHD: reversal, deletion or addition costs one."""
    def states(edges):
        return {tuple(sorted(e)): tuple(e) for e in edges}
    a, b = states(left), states(right)
    return sum(a.get(pair) != b.get(pair) for pair in a.keys() | b.keys())


def variant(task):
    if task["method"] == "order-mcmc": return "MCMC"
    return "SMC fixed" if task["purpose"] == "fixed-work-scaling" else "SMC adaptive"


def complete_group(group, seeds):
    return len(group) == seeds and set(group.seed_index) == set(range(seeds))


def recovery_crossings(frame, config):
    """Descriptive finite-grid comparison against the long-MCMC truth recovery.

    Not posterior accuracy, a formal noninferiority test, or a guarantee that
    the benchmark recovery itself is good. Missing/failed seeds never qualify.
    """
    rows = []
    for dataset, data in frame[frame.arm == "dimension"].groupby("dataset_key"):
        anchor = data[(data.variant == "MCMC") & (data.ranks == 64) &
                      (data.budget == config["extra_mcmc_seconds_at_64"])]
        if not complete_group(anchor, config["seeds"]): continue
        bounds = dict(directed_f1_min=float(anchor.directed_f1.mean()) - config.get("recovery_f1_margin", .02),
            skeleton_f1_min=float(anchor.skeleton_f1.mean()) - config.get("recovery_f1_margin", .02),
            shd_per_true_edge_max=float(anchor.shd_per_true_edge.mean()) + config.get("recovery_shd_per_true_edge_margin", .05))
        def fastest(group):
            choices = []
            for budget, runs in group.groupby("budget"):
                if not complete_group(runs, config["seeds"]): continue
                if ((runs.directed_f1 >= bounds["directed_f1_min"]).all() and
                    (runs.skeleton_f1 >= bounds["skeleton_f1_min"]).all() and
                    (runs.shd_per_true_edge <= bounds["shd_per_true_edge_max"]).all()):
                    choices.append((float(runs.inference_seconds.median()), float(budget)))
            return min(choices) if choices else (None, None)
        for ranks, same_cores in data.groupby("ranks"):
            mt, mb = fastest(same_cores[same_cores.variant == "MCMC"])
            for smc in ("SMC fixed", "SMC adaptive"):
                candidates = same_cores[same_cores.variant == smc]
                if candidates.empty: continue
                st, sb = fastest(candidates)
                rows.append(dict(dataset_key=dataset, ranks=int(ranks), smc_variant=smc,
                    smc_seconds=st, mcmc_seconds=mt, smc_budget=sb, mcmc_budget=mb,
                    speedup=mt/st if st is not None and mt is not None else None,
                    status="both_reached" if st is not None and mt is not None else "not_established_on_tested_grid",
                    anchor_directed_f1=float(anchor.directed_f1.mean()), anchor_skeleton_f1=float(anchor.skeleton_f1.mean()),
                    anchor_shd_per_true_edge=float(anchor.shd_per_true_edge.mean()), **bounds,
                    criterion="truth recovery relative to long MCMC; not posterior accuracy"))
    return pd.DataFrame(rows)


def scaling_table(frame):
    rows = []
    data = frame[(frame.variant == "SMC fixed") & (frame.arm == "dimension")]
    for (dataset, seed), group in data.groupby(["dataset_key", "seed_index"]):
        base = group[group.ranks == 1]
        if len(base) != 1: continue
        base = base.iloc[0]
        for _, row in group.iterrows():
            for field in ("host", "cpu_model", "particles", "proposed_moves", "stages"):
                if row[field] != base[field]: raise RuntimeError(f"scaling comparison changed {field}")
            if row.proposed_moves <= 0: raise RuntimeError("missing fixed-work proposal count")
            speedup = float(base.inference_seconds / row.inference_seconds)
            rows.append(dict(dataset_key=dataset, d=int(row.d), seed_index=int(seed), ranks=int(row.ranks),
                particles=int(row.particles), inference_seconds=row.inference_seconds,
                speedup=speedup, efficiency=speedup/row.ranks,
                directed_f1=row.directed_f1, skeleton_f1=row.skeleton_f1,
                representative_shd=row.representative_shd,
                shd_per_true_edge=row.shd_per_true_edge, proposed_moves=row.proposed_moves,
                directed_f1_change_vs_serial=row.directed_f1-base.directed_f1,
                shd_change_vs_serial=row.representative_shd-base.representative_shd))
    return pd.DataFrame(rows)


def collect(root, allow_partial=False):
    """Read one immutable edition; retain only compact summaries in memory."""
    plan = catchup.checked_source(root / "plan.json")
    for relative, expected in plan["source_files"].items():
        if digest(root / "code_snapshot" / relative) != expected:
            raise RuntimeError("source snapshot changed")
    if read(root / "validation/preflight.json") != dict(all_passed=True, plan_sha256=plan["plan_sha256"]):
        raise RuntimeError("preflight not passed")
    missing_bundles = []
    for index in range(len(plan["bundles"])):
        path = root / "bundles" / f"{index}.json"
        if not path.exists(): missing_bundles.append(index); continue
        value = read(path)
        if value["status"] != "attempted_all" or value["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("invalid bundle")
    if missing_bundles and not allow_partial: raise RuntimeError("incomplete bundle inventory")
    datasets = {d["key"]: d for d in plan["datasets"]}
    preps, exacts, edge_orders, hashes = {}, {}, {}, {}
    for key, d in datasets.items():
        path = root / "prepared" / key / "prepared.json"; prep = read(path)
        if prep["status"] != "complete" or prep["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("invalid preparation")
        hashes[str(path)] = digest(path); preps[key] = prep
        edge_orders[key] = [(u, v) for u in d["nodes"] for v in d["nodes"] if u != v]
        for arm, case in prep["cases"].items():
            checked_input(case["candidate"])
            if "exact" in case:
                probs = pips(read(checked_input(case["exact"])))
                exacts[key, arm] = np.array([probs.get(e, 0.) for e in edge_orders[key]])
    rows, vectors, graphs, failures, missing = [], {}, {}, [], []
    for task in plan["tasks"]:
        tid = task["task_id"]; key = task["dataset_key"]
        path = root / "attempts" / (tid + ".json")
        if not path.exists(): missing.append(tid); continue
        attempt = read(path); hashes[str(path)] = digest(path)
        if attempt["task"] != task or attempt["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("invalid attempt")
        if attempt["status"] != "completed": failures.append(attempt); continue
        path = root / "runs" / tid / "result.json"; r = read(path)
        hashes[str(path)] = digest(path)
        case = preps[key]["cases"][task["arm"]]
        if (r["status"] != "complete" or r["task"] != task or r["plan_sha256"] != plan["plan_sha256"] or
            r["candidate_sha256"] != case["candidate_sha256"]): raise RuntimeError("invalid result/target")
        d = datasets[key]; posterior = r["posterior"]
        probs = pips(posterior); vector = np.array([probs.get(e, 0.) for e in edge_orders[key]])
        if not np.isfinite(vector).all() or np.min(vector) < -1e-8 or np.max(vector) > 1+1e-8:
            raise RuntimeError("invalid probabilities")
        vectors[tid] = vector
        selected = set(map(tuple, posterior["representative_edges"])); graphs[tid] = selected
        truth = set(map(tuple, d["truth_edges"])); shd = pair_shd(selected, truth)
        hist = (r["smc_diagnostics"] or {}).get("history", [])
        reference = exacts.get((key, task["arm"]))
        rows.append(dict(**task, variant=variant(task), d=len(d["nodes"]), truth_edges=len(truth),
            budget=task.get("particles", task.get("wall_seconds")),
            inference_seconds=r["inference_seconds"], setup_seconds=r["setup_seconds"],
            launch_to_saved_result_seconds=attempt["launcher_wall_seconds"],
            composed_end_to_end_seconds=r["shared_preparation_seconds"]+attempt["launcher_wall_seconds"],
            active_core_hours=r["active_core_hours"], reserved_core_hours=attempt["reserved_core_hours"],
            directed_f1=posterior["directed_f1"], skeleton_f1=posterior["skeleton_f1"],
            directed_auprc=posterior.get("directed_auprc"), skeleton_auprc=posterior.get("skeleton_auprc"),
            representative_shd=shd, shd_per_true_edge=shd/max(1, len(truth)),
            exact_pip_rmse=float(np.sqrt(np.mean((vector-reference)**2))) if reference is not None else np.nan,
            exact_pip_max_error=float(np.max(np.abs(vector-reference))) if reference is not None else np.nan,
            host=r["hardware"][0]["host"], cpu_model=r["hardware"][0]["cpu_model"],
            proposed_moves=sum(int(h.get("proposed_moves", 0)) for h in hist),
            stages=len([h for h in hist if h.get("phase") != "final_target"])))
        del r
    if missing and not allow_partial: raise RuntimeError("unattempted planned fits")
    if not rows: raise RuntimeError("no completed fits")
    frame = pd.DataFrame(rows)
    frame["long_mcmc_agreement_rmse"] = np.nan
    frame["long_mcmc_informative_agreement_rmse"] = np.nan
    frame["long_mcmc_max_difference"] = np.nan
    frame["pip_rmse_vs_serial"] = np.nan
    for (key, arm), group in frame.groupby(["dataset_key", "arm"]):
        anchors = group[(group.variant == "MCMC") & (group.ranks == 64) &
                        (group.budget == plan["config"]["extra_mcmc_seconds_at_64"])]
        # Leave the evaluated seed out for BOTH methods, preventing reuse of a
        # shorter chain's prefix inside its own long-chain agreement benchmark.
        if complete_group(anchors, plan["config"]["seeds"]) and len(anchors) > 1:
            for index, row in group.iterrows():
                anchor = np.mean([vectors[a.task_id] for _, a in anchors.iterrows()
                                  if a.seed_index != row.seed_index], axis=0)
                delta = vectors[row.task_id]-anchor; informative = (anchor >= .05) & (anchor <= .95)
                frame.loc[index, "long_mcmc_agreement_rmse"] = np.sqrt(np.mean(delta**2))
                frame.loc[index, "long_mcmc_max_difference"] = np.max(np.abs(delta))
                if informative.any():
                    frame.loc[index, "long_mcmc_informative_agreement_rmse"] = np.sqrt(np.mean(delta[informative]**2))
        for index, row in group[group.variant == "SMC fixed"].iterrows():
            base = group[(group.variant == "SMC fixed") & (group.ranks == 1) & (group.seed_index == row.seed_index)]
            if len(base) == 1:
                frame.loc[index, "pip_rmse_vs_serial"] = np.sqrt(np.mean((vectors[row.task_id]-vectors[base.iloc[0].task_id])**2))
    stability = []
    keys = ["dataset_key", "arm", "variant", "ranks", "budget"]
    for key, group in frame.groupby(keys):
        if not complete_group(group, plan["config"]["seeds"]) or len(group) < 2: continue
        ids = list(group.task_id); pairs = list(itertools.combinations(ids, 2))
        jaccards = [len(graphs[a] & graphs[b])/len(graphs[a] | graphs[b]) if graphs[a] | graphs[b] else 1. for a,b in pairs]
        rmses = [float(np.sqrt(np.mean((vectors[a]-vectors[b])**2))) for a,b in pairs]
        stability.append(dict(zip(keys, key), seeds=len(group), mean_directed_jaccard=float(np.mean(jaccards)),
            min_directed_jaccard=min(jaccards), mean_pairwise_pip_rmse=float(np.mean(rmses))))
    return plan, frame, pd.DataFrame(stability), failures, missing, missing_bundles, hashes


def figures(frame, scaling, stability, output, partial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    data = frame[frame.arm == "dimension"]
    datasets = [(k, int(g.d.iloc[0])) for k,g in data.groupby("dataset_key", sort=False)]
    def axes(rows):
        return plt.subplots(rows, len(datasets), figsize=(5*len(datasets), 3.4*rows), squeeze=False)
    def finish(fig, name):
        fig.suptitle("PARTIAL PILOT — not final" if partial else "Pilot: one dataset per size; three inference seeds")
        fig.tight_layout(rect=(0, 0, 1, .96))
        for suffix in ("pdf", "png"): fig.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
    def ranges(ax, group, x, y, label):
        points = group.groupby(x)[y].agg(["median", "min", "max"]).sort_index()
        ax.errorbar(points.index.to_numpy(), points["median"].to_numpy(),
                    yerr=np.array([points["median"]-points["min"], points["max"]-points["median"]]),
                    marker="o", capsize=3, label=label)
    fig, ax = axes(2)
    for col, (key, d) in enumerate(datasets):
        sub = scaling[scaling.dataset_key == key] if not scaling.empty else scaling
        if not sub.empty:
            ranges(ax[0,col], sub, "ranks", "speedup", "Measured SMC")
            ax[0,col].plot(sorted(sub.ranks.unique()), sorted(sub.ranks.unique()), ":", color="grey", label="Ideal")
            ranges(ax[1,col], sub, "ranks", "efficiency", "Measured SMC")
        count = int(sub.particles.iloc[0]) if not sub.empty else "?"
        ax[0,col].set(title=f"{d} variables; {count} particles", ylabel="Speedup over one core")
        ax[1,col].set(xlabel="Active cores", ylabel="Parallel efficiency")
        for row in range(2): ax[row,col].grid(alpha=.2)
        ax[0,col].legend(fontsize=7)
    finish(fig, "smc_strong_scaling")
    for x, name, xlabel in (("inference_seconds", "recovery_vs_time", "Inference seconds"),
                            ("active_core_hours", "recovery_vs_core_hours", "Active core-hours"),
                            ("composed_end_to_end_seconds", "recovery_vs_end_to_end", "Preparation + launch-to-result seconds")):
        fig, ax = axes(3)
        for col, (key,d) in enumerate(datasets):
            sub = data[data.dataset_key == key]
            for (method, ranks), group in sub.groupby(["variant", "ranks"]):
                for row, metric in enumerate(("directed_f1", "skeleton_f1", "shd_per_true_edge")):
                    points = group.groupby("budget").agg(x=(x,"median"), y=(metric,"median"), low=(metric,"min"), high=(metric,"max")).sort_values("x")
                    ax[row,col].errorbar(points.x.to_numpy(), points.y.to_numpy(),
                        yerr=np.array([points.y-points.low,points.high-points.y]), marker="o", capsize=2,
                        linestyle="--" if method == "MCMC" else "-", label=f"{method}, {ranks} cores")
                    ax[row,col].set(xscale="log", xlabel=xlabel, ylabel=metric.replace("_", " "))
                    ax[row,col].grid(alpha=.2)
            ax[0,col].set_title(f"{d} variables"); ax[0,col].legend(fontsize=6)
        finish(fig, name)
    fig, ax = axes(3)
    for col,(key,d) in enumerate(datasets):
        sub = data[(data.dataset_key == key) & (data.variant == "SMC fixed")]
        for row, metric in enumerate(("directed_f1", "skeleton_f1", "pip_rmse_vs_serial")):
            if not sub.empty: ranges(ax[row,col], sub, "ranks", metric, "Fixed-work SMC")
            ax[row,col].set(xlabel="Active cores", ylabel=metric.replace("_", " "))
            ax[row,col].grid(alpha=.2)
        count = int(sub.particles.iloc[0]) if not sub.empty else "?"
        ax[0,col].set_title(f"{d} variables; {count} particles")
    finish(fig, "smc_quality_vs_cores")
    fig, ax = axes(2)
    for col,(key,d) in enumerate(datasets):
        sub = data[data.dataset_key == key]
        for (method,ranks), group in sub.groupby(["variant","ranks"]):
            for row,metric in enumerate(("long_mcmc_agreement_rmse", "long_mcmc_informative_agreement_rmse")):
                points = group.groupby("budget")[["inference_seconds",metric]].median().dropna().sort_values("inference_seconds")
                if not points.empty: ax[row,col].plot(points.inference_seconds,points[metric],"o-",label=f"{method}, {ranks} cores")
                ax[row,col].set(xscale="log", xlabel="Inference seconds", ylabel=("All edges" if row == 0 else "Informative edges")+" PIP disagreement")
                ax[row,col].grid(alpha=.2)
        ax[0,col].set_title(f"{d} variables: MCMC agreement, NOT accuracy")
        if ax[0,col].lines: ax[0,col].legend(fontsize=6)
    finish(fig, "edge_probability_agreement")
    fig, ax = axes(2)
    for col,(key,d) in enumerate(datasets):
        sub = stability[(stability.dataset_key == key) & (stability.arm == "dimension")] if not stability.empty else stability
        if not sub.empty:
            for (method,budget), group in sub.groupby(["variant","budget"]):
                group=group.sort_values("ranks")
                for row,metric in enumerate(("mean_directed_jaccard","mean_pairwise_pip_rmse")):
                    ax[row,col].plot(group.ranks,group[metric],"o-",label=f"{method}, budget {budget:g}")
        ax[0,col].set(title=f"{d} variables", ylabel="Cross-seed graph Jaccard", ylim=(0,1.02))
        ax[1,col].set(xlabel="Active cores",ylabel="Cross-seed PIP RMSE")
        if ax[0,col].lines: ax[0,col].legend(fontsize=6)
        for row in range(2): ax[row,col].grid(alpha=.2)
    finish(fig, "seed_reproducibility")


def aggregate(root, output=None, allow_partial=False):
    root = Path(root).resolve(); output = Path(output) if output else root / "analysis"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new report directory; earlier reporting editions are preserved")
    plan, frame, stability, failures, missing, missing_bundles, hashes = collect(root, allow_partial)
    partial = bool(missing or missing_bundles)
    scaling = scaling_table(frame); crossings = recovery_crossings(frame, plan["config"])
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("per_run_metrics",frame),("scaling",scaling),("seed_stability",stability),("time_to_recovery",crossings)):
        table.to_csv(output / f"{name}.csv",index=False)
    frame.groupby(["dataset_key","arm","variant","ranks","budget"],as_index=False).agg(
        seeds=("seed_index","size"), median_seconds=("inference_seconds","median"),
        min_seconds=("inference_seconds","min"),max_seconds=("inference_seconds","max"),
        mean_directed_f1=("directed_f1","mean"),mean_skeleton_f1=("skeleton_f1","mean"),
        mean_shd=("representative_shd","mean"),mean_directed_auprc=("directed_auprc","mean"),
        mean_exact_pip_rmse=("exact_pip_rmse","mean"),
        mean_mcmc_pip_disagreement=("long_mcmc_agreement_rmse","mean")).to_csv(output / "summary.csv",index=False)
    _atomic_json(output / "failures.json",dict(failed_attempts=failures,unattempted_task_ids=missing,unfinished_bundles=missing_bundles))
    _atomic_json(output / "artifact_hashes.json",hashes)
    figures(frame,scaling,stability,output/"figures",partial)
    _atomic_json(output / "completion.json",dict(status="partial" if partial else "complete_with_failed_attempts" if failures else "complete",
        source_root=str(root),plan_sha256=plan["plan_sha256"],planned=len(plan["tasks"]),succeeded=len(frame),
        report_code_sha256=digest(__file__),
        failed=len(failures),unattempted=len(missing),confirmatory=False,publication_ready=False,
        posterior_accuracy_claim_for_large_networks=False,graph_recovery_requires_mcmc_convergence=False,
        note="Truth recovery and empirical MCMC agreement are separate. Existing reference failures are not overturned."))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args(); aggregate(args.source,args.output,args.allow_partial)


if __name__ == "__main__": main()
