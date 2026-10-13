"""Complete-inventory timing reports; never manufacture speedups for failed gates."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .execution import _atomic_json
from .readiness_audits import read, pips, sha, rmse
from .sampler_corrections import checked_input
from .timing_pilot import accepted_plan, split_rank_rhat


def maximum_pip_error(probabilities, reference):
    """Sparse sampler output omits zero-probability edges, not observations."""
    return max((abs(probabilities.get(edge, 0.) - value) for edge, value in reference.items()), default=0.)


def reference_check(records, config, nodes):
    groups = [pips(r["posterior"]) for r in records]
    edges = sorted(groups[0]); pooled = {e: float(np.mean([g[e] for g in groups])) for e in edges}
    panel = [e for e in edges if .05 <= pooled[e] <= .95]
    discrepancy = rmse(groups[0], groups[1])
    largest = max(abs(groups[0][e] - groups[1][e]) for e in edges)
    rhats, drifts, enough = [], [], True
    constant_quantities = 0
    for record in records:
        chains = record["chains"]; diagnostics = record["reference_diagnostics"]
        enough &= all(c["stored_samples"] >= config["reference_min_stored_per_chain"] for c in chains)
        # All ranks within a group monitor the same pooled-uncertainty edge panel.
        if any(d["monitored_edges"] != diagnostics[0]["monitored_edges"] for d in diagnostics):
            raise RuntimeError("inconsistent reference diagnostic panels")
        length = min(len(d["trace"]) for d in diagnostics)
        traces = np.array([np.asarray(d["trace"])[np.linspace(0, len(d["trace"]) - 1, length, dtype=int)] for d in diagnostics])
        for column in range(traces.shape[2]):
            values = traces[:, :, column]
            constant_quantities += int(np.ptp(values) == 0)
            rhats.append(split_rank_rhat(values))
        # A constant target trace in any chain cannot establish useful exploration.
        enough &= all(np.ptp(values) > 0 for values in traces[:, :, 0])
        halves = []
        for half in (0, 1):
            maps = [{(x["parent"], x["child"]): x["probability"] for x in d["half_pips"][half]} for d in diagnostics]
            halves.append({e: float(np.mean([m[e] for m in maps])) for e in edges})
        drifts.append(rmse(halves[0], halves[1]))
    maximum_rhat = max(rhats, default=math.inf)
    passed = bool(enough and discrepancy <= config["reference_group_rmse_limit"] and
        largest <= config["reference_group_max_abs_limit"] and maximum_rhat <= config["reference_rhat_limit"] and
        max(drifts) <= config["reference_half_rmse_limit"])
    return pooled, dict(usable=passed, exact=False, convergence_proved=False,
        group_pip_rmse=discrepancy, group_max_abs_difference=largest,
        informative_group_pip_rmse=rmse({e: groups[0][e] for e in panel}, {e: groups[1][e] for e in panel}) if panel else None,
        max_rank_normalized_folded_split_rhat=maximum_rhat if np.isfinite(maximum_rhat) else None,
        rhat_nonfinite=not np.isfinite(maximum_rhat), max_half_pip_drift=max(drifts),
        minimum_chain_length_and_exploration_passed=bool(enough), constant_monitored_quantities=constant_quantities,
        warning="Passing finite diagnostics is not proof of convergence; reference is approximate.")


def quality_crossings(frame, thresholds, seeds):
    """Smallest tested *group-median* time with every repetition below threshold.

    No best-seed selection, interpolation, or extrapolated crossing. These
    finite-grid ratios are not optimal time-to-accuracy estimates.
    """
    rows = []
    for (dataset, arm, ranks), group in frame[frame.purpose == "accuracy-time"].groupby(["dataset_key", "arm", "ranks"]):
        for threshold in thresholds:
            times = {}
            for method in ("ordersmc", "order-mcmc"):
                candidates = []
                for _, budget in group[group.method == method].groupby("budget"):
                    if len(budget) != seeds or budget.seed_index.nunique() != seeds: continue
                    if not budget.reference_usable.all() or not (budget.pip_rmse <= threshold).all(): continue
                    candidates.append(float(budget.inference_seconds.median()))
                times[method] = min(candidates) if candidates else None
            a, b = times["ordersmc"], times["order-mcmc"]
            rows.append(dict(dataset_key=dataset, arm=arm, ranks=ranks, epsilon=threshold,
                smc_seconds=a, mcmc_seconds=b, speedup=b / a if a is not None and b is not None else None,
                status="both_reached" if a is not None and b is not None else "not_established_on_tested_grid"))
    return pd.DataFrame(rows)


def scaling_rows(frame, config):
    rows = []
    subset = frame[(frame.purpose == "fixed-work-scaling") & (frame.arm == "dimension")]
    for (dataset, seed), group in subset.groupby(["dataset_key", "seed_index"]):
        base = group[group.ranks == 1]
        if len(base) != 1: continue
        for _, r in group.iterrows():
            same_cpu = r.cpu_model == base.iloc[0].cpu_model and r.host == base.iloc[0].host
            same_work = r.proposed_moves == base.iloc[0].proposed_moves and r.stages == base.iloc[0].stages
            if not same_cpu or not same_work:
                raise RuntimeError("fixed-work scaling changed hardware or proposal count")
            speedup = float(base.iloc[0].inference_seconds / r.inference_seconds)
            rows.append(dict(dataset_key=dataset, seed_index=seed, ranks=r.ranks,
                particles=r.particles, inference_seconds=r.inference_seconds, speedup=speedup,
                efficiency=speedup / r.ranks, proposed_moves=r.proposed_moves, stages=r.stages,
                pip_rmse=r.pip_rmse, reference_usable=r.reference_usable))
    return pd.DataFrame(rows)


def aggregate(root):
    plan = accepted_plan(root); config = plan["config"]; output = root / "analysis"
    output.mkdir(exist_ok=True); hashes, rows, failures, references, gates = {}, [], [], {}, []
    # Refuse a selective partial report. A failed, explicitly attempted step is
    # an observation; an unattempted step is an incomplete experiment.
    for index in range(len(plan["bundles"])):
        record = read(root / "bundles" / f"{index}.json")
        if record["status"] != "attempted_all" or record["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("incomplete timing bundle")
    for d in plan["datasets"]:
        prep = read(root / "prepared" / d["key"] / "prepared.json")
        if prep["plan_sha256"] != plan["plan_sha256"]: raise RuntimeError("stale preparation report")
        for arm, case in prep["cases"].items():
            if "exact" in case:
                references[d["key"], arm] = pips(read(checked_input(case["exact"])))
                gate = dict(usable=True, exact=True)
            else:
                records = []
                for task in plan["references"]:
                    if (task["dataset_key"], task["arm"]) != (d["key"], arm): continue
                    path = root / "references" / task["task_id"] / "result.json"
                    r = read(path)
                    if r["status"] != "complete" or r["task"] != task or r["plan_sha256"] != plan["plan_sha256"] or r["candidate_sha256"] != case["candidate_sha256"]:
                        raise RuntimeError("invalid reference artifact")
                    records.append(r); hashes[str(path)] = sha(path)
                if len(records) != 2: raise RuntimeError("two independent reference groups required")
                reference, gate = reference_check(records, config, d["nodes"])
                references[d["key"], arm] = reference
            gates.append(dict(dataset_key=d["key"], arm=arm, **gate))
    gate_lookup = {(g["dataset_key"], g["arm"]): g for g in gates}
    for task in plan["tasks"]:
        attempt_path = root / "attempts" / (task["task_id"] + ".json")
        attempt = read(attempt_path); hashes[str(attempt_path)] = sha(attempt_path)
        if attempt["plan_sha256"] != plan["plan_sha256"] or attempt["task"] != task: raise RuntimeError("invalid task attempt")
        if attempt["status"] != "completed":
            failures.append(attempt); continue
        path = root / "runs" / task["task_id"] / "result.json"; r = read(path)
        if r["status"] != "complete" or r["task"] != task or r["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("invalid timing artifact")
        hashes[str(path)] = sha(path)
        prep = read(root / "prepared" / task["dataset_key"] / "prepared.json")
        if r["candidate_sha256"] != prep["cases"][task["arm"]]["candidate_sha256"]:
            raise RuntimeError("comparison target changed")
        ref = references[task["dataset_key"], task["arm"]]; probs = pips(r["posterior"])
        informative = {e: p for e, p in ref.items() if .05 <= p <= .95}
        hist = (r["smc_diagnostics"] or {}).get("history", [])
        # MPI history reports proposal totals per tempering/final-target row.
        proposed = sum(float(h.get("proposed_moves", 0)) for h in hist)
        records = r["chains"]
        rows.append(dict(**task, budget=task.get("particles", task.get("wall_seconds")),
            inference_seconds=r["inference_seconds"], setup_seconds=r["setup_seconds"],
            launch_to_saved_result_seconds=attempt["launcher_wall_seconds"],
            composed_end_to_end_seconds=r["shared_preparation_seconds"] + attempt["launcher_wall_seconds"],
            active_core_hours=r["active_core_hours"], reserved_core_hours=attempt["reserved_core_hours"],
            reference_usable=gate_lookup[task["dataset_key"], task["arm"]]["usable"],
            reference_exact=gate_lookup[task["dataset_key"], task["arm"]]["exact"],
            pip_rmse=rmse(probs, ref), informative_pip_rmse=rmse(probs, informative) if informative else None,
            max_pip_error=maximum_pip_error(probs, ref),
            expected_edges=sum(probs.values()), expected_edges_error=abs(sum(probs.values()) - sum(ref.values())),
            directed_f1=r["posterior"]["directed_f1"], skeleton_f1=r["posterior"]["skeleton_f1"],
            host=r["hardware"][0]["host"], cpu_model=r["hardware"][0]["cpu_model"],
            proposed_moves=proposed, stages=len([h for h in hist if h.get("phase") != "final_target"]),
            mcmc_stored_draws=sum(c["stored_samples"] for c in records),
            maximum_rank_rss_kib=r["max_rank_peak_rss_kib"]))
    if not rows: raise RuntimeError("no completed benchmark estimates")
    frame = pd.DataFrame(rows); frame.to_csv(output / "per_run_metrics.csv", index=False)
    crossings = quality_crossings(frame, config["accuracy_rmse_thresholds"], config["seeds"])
    crossings.to_csv(output / "time_to_accuracy.csv", index=False)
    scaling = scaling_rows(frame, config); scaling.to_csv(output / "scaling.csv", index=False)
    frame.groupby(["dataset_key", "arm", "method", "purpose", "ranks", "budget"], as_index=False).agg(
        repetitions=("seed_index", "size"), median_seconds=("inference_seconds", "median"),
        min_seconds=("inference_seconds", "min"), max_seconds=("inference_seconds", "max"),
        mean_pip_rmse=("pip_rmse", "mean"), worst_seed_pip_rmse=("pip_rmse", "max"),
        mean_directed_f1=("directed_f1", "mean"), mean_skeleton_f1=("skeleton_f1", "mean"),
        mean_active_core_hours=("active_core_hours", "mean")).to_csv(output / "summary.csv", index=False)
    # Reproducibility across the three independent inference repetitions.
    stability = []
    import itertools
    for key, group in frame.groupby(["dataset_key", "arm", "method", "purpose", "ranks", "budget"]):
        graphs = [set(map(tuple, read(root / "runs" / row.task_id / "result.json")["posterior"]["representative_edges"]))
                  for _, row in group.iterrows()]
        if len(graphs) != config["seeds"] or len(graphs) < 2: continue
        pairs = list(itertools.combinations(graphs, 2))
        stability.append(dict(zip(["dataset_key", "arm", "method", "purpose", "ranks", "budget"], key),
            seeds=len(graphs), mean_directed_jaccard=float(np.mean([len(a & b) / len(a | b) if a | b else 1. for a, b in pairs]))))
    pd.DataFrame(stability).to_csv(output / "seed_stability.csv", index=False)
    _atomic_json(output / "reference_diagnostics.json", dict(gates=gates, references=[
        dict(dataset_key=k[0], arm=k[1], edge_probabilities=[dict(parent=e[0], child=e[1], probability=p) for e, p in v.items()])
        for k, v in references.items()]))
    _atomic_json(output / "failures.json", dict(failed_attempts=failures))
    _atomic_json(output / "artifact_hashes.json", hashes)
    figures(frame, crossings, scaling, output / "figures")
    _atomic_json(output / "completion.json", dict(status="complete" if not failures else "complete_with_failed_attempts",
        plan_sha256=plan["plan_sha256"], attempted=len(plan["tasks"]), succeeded=len(rows), failed=len(failures),
        reference_gates_passed=sum(g["usable"] for g in gates), reference_gates_total=len(gates),
        confirmatory=False, publication_ready=False, universal_speed_claim_established=False))


def figures(frame, crossings, scaling, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(exist_ok=True)
    datasets = list(frame.dataset_key.unique())
    labels = {d: str(d).split("--", 1)[-1] for d in datasets}
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("pdf", "png"): fig.savefig(destination / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
    for axis_column, name, xlabel in (("inference_seconds", "posterior_error_vs_time", "Inference seconds"),
        ("active_core_hours", "posterior_error_vs_core_hours", "Active core-hours"),
        ("composed_end_to_end_seconds", "posterior_error_vs_end_to_end", "Shared preparation + launch-to-result seconds")):
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.3), squeeze=False)
        for ax, d in zip(axes[0], datasets):
            subset = frame[(frame.dataset_key == d) & (frame.arm == "dimension") & (frame.purpose == "accuracy-time")]
            usable = bool(subset.reference_usable.all())
            for (method, ranks), group in subset.groupby(["method", "ranks"]):
                points = group.groupby("budget")[[axis_column, "pip_rmse"]].median().sort_values(axis_column)
                ax.plot(points[axis_column], points.pip_rmse, marker="o" if method == "ordersmc" else "s",
                    linestyle="-" if method == "ordersmc" else "--", label=f"{method}, {ranks} cores")
            ax.set(xscale="log", yscale="log", xlabel=xlabel,
                ylabel="PIP RMSE" if usable else "Disagreement with UNVALIDATED reference",
                title=labels[d] + ("" if usable else "\nREFERENCE CHECK FAILED"))
            ax.legend(fontsize=6); ax.grid(alpha=.2)
        save(fig, name)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        subset = crossings[(crossings.dataset_key == d) & (crossings.arm == "dimension")]
        for epsilon, group in subset.groupby("epsilon"):
            valid = group[group.status == "both_reached"].sort_values("ranks")
            if not valid.empty: ax.plot(valid.ranks, valid.speedup, "o-", label=f"RMSE <= {epsilon}")
        ax.axhline(1., color="grey", linestyle=":")
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        else: ax.text(.5, .5, "No validated matched-accuracy\ncrossing on tested grid", ha="center", transform=ax.transAxes)
        ax.set(title=labels[d], xlabel="Equal active cores", ylabel="MCMC / SMC time")
    save(fig, "quality_matched_speedup")
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False)
    for col, d in enumerate(datasets):
        if not scaling.empty:
            subset = scaling[scaling.dataset_key == d]
            for metric, row in (("speedup", 0), ("efficiency", 1)):
                stats = subset.groupby("ranks")[metric].agg(["median", "min", "max"])
                axes[row, col].errorbar(stats.index.to_numpy(), stats["median"].to_numpy(),
                    yerr=np.array([stats["median"] - stats["min"], stats["max"] - stats["median"]]), marker="o")
                axes[row, col].set(xlabel="MPI ranks", ylabel=metric, title=labels[d])
        axes[0, col].plot([1, 64], [1, 64], ":", color="grey", label="ideal")
        axes[1, col].axhline(1., linestyle=":", color="grey")
    fig.suptitle("Fixed-work scaling probe: 128 particles; 32 temperatures; not the adaptive accuracy arm")
    save(fig, "mpi_scaling")
