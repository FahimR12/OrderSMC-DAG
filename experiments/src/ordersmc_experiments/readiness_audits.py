"""Read-only-source diagnostics before the metadata study; never tune recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ordersmc.persistence import load_candidate_table
from .exact_order import exact_order_dp
from .exact_partition_log import exact_partition_log_dp
from .exact_structure import exact_structure_dp
from .metadata_analysis import TOLERANCES, RELATIVE_METRICS
from .metadata_profiles import (
    MetadataDAGConfig, load_metadata_dag_artifact, load_metadata_profile,
    metadata_fidelity_statistics, _sample,
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pips(record):
    return {(r["parent"], r["child"]): float(r["probability"])
            for r in record["edge_probabilities"]}


def rmse(left, right):
    return float(np.sqrt(np.mean([(left.get(e, 0.) - p) ** 2 for e, p in right.items()])))


def audit_partition(source: Path, output: Path):
    """Assess structure chains against a structure reference, not an order target."""
    completion = read(source / "analysis/completion.json")
    if completion["missing_or_failed_runs"] or completion["planned_runs"] != completion["completed_runs"]:
        raise RuntimeError("the reference audit requires a complete source cohort")
    plan = read(source / "plan.json")
    rows, reference_rows = [], []
    for spec in plan["datasets"]:
        dataset_id = spec["dataset_id"]
        directory = source / "datasets" / dataset_id
        prepared = read(directory / "prepared.json")
        table = load_candidate_table(directory / "candidate.json.gz",
                                     expected_sha256=prepared["candidate_table_sha256"])
        reference = exact_partition_log_dp(table)
        order = exact_order_dp(table, compute_graph_size=False)
        crosscheck = {"sink_reference_status": "passed"}
        try:
            sink = exact_structure_dp(table, compute_graph_size=False)
            discrepancy = rmse(sink.edge_probs, reference.edge_probs)
            logz_error = abs(sink.log_normalizer - reference.log_normalizer)
            crosscheck.update(sink_pip_rmse=discrepancy, sink_logz_error=logz_error)
            if discrepancy > 1e-7 or logz_error > 1e-6:
                crosscheck["sink_reference_status"] = "numerical-disagreement"
        except (FloatingPointError, ArithmeticError) as error:
            crosscheck = {"sink_reference_status": "numerically-unavailable", "reason": str(error)}
        reference_rows.append({
            "dataset_id": dataset_id, "n_train": spec["n_train"],
            "candidate_table_sha256": prepared["candidate_table_sha256"],
            "structure_log_normalizer": reference.log_normalizer,
            "order_structure_pip_rmse": rmse(order.edge_probs, reference.edge_probs),
            "edge_probabilities": [{"source": e[0], "target": e[1], "probability": p}
                                   for e, p in reference.edge_probs.items()],
            **crosscheck,
        })
        for path in sorted((source / "runs").glob(f"{dataset_id}-*.json")):
            run = read(path)
            method = run["run"]["method"]
            if method not in {"partition-mcmc", "rev-mcmc", "order-mcmc", "ordersmc"}:
                continue
            if run["candidate_table_sha256"] != prepared["candidate_table_sha256"]:
                raise RuntimeError(f"candidate identity mismatch: {path}")
            own = reference.edge_probs if method in {"partition-mcmc", "rev-mcmc"} else order.edge_probs
            diagnostics = run["diagnostics"]
            rows.append({
                "dataset_id": dataset_id, "n_train": spec["n_train"], "method": method,
                "run_file": path.name, "run_sha256": sha(path),
                "own_target": "structure" if method in {"partition-mcmc", "rev-mcmc"} else "order",
                "own_target_pip_rmse": rmse(pips(run), own),
                "structure_target_pip_rmse": rmse(pips(run), reference.edge_probs),
                "order_target_pip_rmse": rmse(pips(run), order.edge_probs),
                "acceptance_rate": diagnostics.get("acceptance_rate"),
                "retained_samples": diagnostics.get("retained_samples"),
            })
        print(f"structure reference: {dataset_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "partition_runs.csv", index=False)
    dataset = frame.groupby(["method", "n_train", "dataset_id"], as_index=False)[
        ["own_target_pip_rmse", "structure_target_pip_rmse", "order_target_pip_rmse"]].mean()
    summary = dataset.groupby(["method", "n_train"], as_index=False).agg(
        datasets=("dataset_id", "size"), own_target_pip_rmse=("own_target_pip_rmse", "mean"),
        p95_dataset_rmse=("own_target_pip_rmse", lambda x: x.quantile(.95)),
        structure_target_pip_rmse=("structure_target_pip_rmse", "mean"),
        order_target_pip_rmse=("order_target_pip_rmse", "mean"))
    summary.to_csv(output / "partition_summary.csv", index=False)
    write(output / "partition_reference.json", {
        "schema": "ordersmc-readiness-structure-audit-v1", "status": "diagnostic-complete",
        "source_root": str(source.resolve()), "plan_sha256": plan["plan_sha256"],
        "completion_file_sha256": sha(source / "analysis/completion.json"),
        "reference": "unique source-layer partition recursion; positive log-space arithmetic",
        "unit": "four seeds averaged within each dataset before cohort summaries",
        "warning": "Low acceptance and own-target error diagnose this run configuration; not a general ranking of Partition MCMC.",
        "datasets": reference_rows,
    })


def audit_calibration(source: Path, output: Path, repeats: int = 50, statistics_root: Path | None = None):
    """Conditional repeated-sampling diagnostic, without selecting new mechanisms."""
    if repeats < 20:
        raise ValueError("at least 20 repeat samples are required")
    plan = read(source / "plan.json")
    draws, summaries, provenance = [], [], []
    for spec in plan["datasets"]:
        if spec["role"] != "m1_matched" or spec["metadata_profile"] != "protein_pathway":
            continue
        directory = source / "datasets" / spec["dataset_id"]
        data = load_metadata_dag_artifact(directory)
        manifest = data.manifest
        if manifest["calibration"]["selection_uses_graph_recovery"] is not False:
            raise RuntimeError("calibration must not use recovery")
        config = MetadataDAGConfig(**manifest["config"])
        profile = (load_metadata_profile(config.profile_id, statistics_root)
                   if statistics_root is not None else load_metadata_profile(config.profile_id))
        if dict(profile.source_hashes) != manifest["profile"]["source_hashes"]:
            raise RuntimeError("metadata source changed since preparation")
        bio = [node for node in data.train.columns if node not in {"PD", "female", "age_z"}]
        values = []
        for repeat in range(repeats):
            simulated = _sample(len(data.train), f"readiness-fidelity-{repeat:03d}",
                                manifest["topological_order"], manifest["mechanisms"],
                                data.variable_types, profile, config)
            metrics = metadata_fidelity_statistics(simulated, bio, profile)
            draws.append({"dataset_id": spec["dataset_id"], "repeat": repeat, **metrics})
            values.append(metrics)
        for metric, tolerance in TOLERANCES.items():
            observations = np.array([v[metric] for v in values])
            target = manifest["calibration"]["target"][metric]
            low, high = np.quantile(observations, [.025, .975])
            divisor = max(abs(target), 1e-12) if metric in RELATIVE_METRICS else 1.
            mean = float(observations.mean())
            se = float(observations.std(ddof=1) / np.sqrt(repeats))
            summaries.append({
                "dataset_id": spec["dataset_id"], "metric": metric, "target": target,
                "repeated_sample_mean": mean, "sample_sd": observations.std(ddof=1),
                "monte_carlo_se_of_mean": se, "predictive_q025": low, "predictive_q975": high,
                "target_in_predictive_interval": bool(low <= target <= high),
                "mean_scaled_difference": (mean - target) / divisor,
                "tolerance": tolerance, "mean_within_tolerance": abs(mean - target) / divisor <= tolerance,
                "systematic_tolerance_miss": max(0., abs(mean - target) - 1.96 * se) / divisor > tolerance,
            })
        provenance.append({"dataset_id": spec["dataset_id"],
                           "generator_manifest_sha256": sha(directory / "generator_manifest.json"),
                           "selected_scale": manifest["calibration"]["selected_scale"]})
        print(f"calibration resampling: {spec['dataset_id']}", flush=True)
    if not summaries:
        raise RuntimeError("no prepared protein-pathway M1 datasets")
    pd.DataFrame(draws).to_csv(output / "calibration_draws.csv", index=False)
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "calibration_by_graph.csv", index=False)
    frame.groupby("metric", as_index=False).agg(
        target=("target", "first"), generated_mean=("repeated_sample_mean", "mean"),
        fraction_means_within_tolerance=("mean_within_tolerance", "mean"),
        fraction_systematic_misses=("systematic_tolerance_miss", "mean"),
        fraction_targets_in_predictive_interval=("target_in_predictive_interval", "mean"),
    ).to_csv(output / "calibration_summary.csv", index=False)
    write(output / "calibration_audit.json", {
        "schema": "ordersmc-readiness-calibration-audit-v1", "status": "diagnostic-complete",
        "source_root": str(source.resolve()), "plan_sha256": plan["plan_sha256"],
        "repeat_samples_per_fixed_graph": repeats, "n_per_sample": len(data.train),
        "uses_participant_values": False, "uses_graph_recovery": False,
        "changes_generator_or_selects_new_scale": False,
        "intervals": "Empirical 2.5/97.5 percentiles of independent size-n samples conditional on each of five frozen mechanisms; not uncertainty intervals for the uploaded real-data targets.",
        "systematic_miss_definition": "max(0, abs(repeated-sample mean - target) - 1.96 * Monte Carlo SE) exceeds the original tolerance (relative for rank/components). Descriptive, not a multiplicity-adjusted test.",
        "limitations": "Five graphs only. Source-target sampling uncertainty is unavailable. Repeated samples are not new independent graphs. Female prevalence remains the declared 0.5 assumption.",
        "datasets": provenance,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clg-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statistics-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit_partition(args.clg_root, args.output)
    audit_calibration(args.metadata_root, args.output, statistics_root=args.statistics_root)


if __name__ == "__main__":
    main()
