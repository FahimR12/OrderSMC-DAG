"""Independent finite-sample metadata calibration audit; no inference scores."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .execution import _plan, _execution_config, _source_sha256, _sha, _atomic_json
from .metadata_analysis import TOLERANCES, RELATIVE_METRICS
from .metadata_profiles import (generate_metadata_dag_dataset, metadata_config_for_spec,
                                load_metadata_profile, metadata_fidelity_statistics, _sample)
from .readiness_audits import read, sha


def binding(root, config_path, profile):
    plan, execution = _plan(root / "plan.json"), _execution_config(config_path)
    return plan, execution, dict(plan_sha256=plan["plan_sha256"], execution_config_sha256=_sha(execution),
                                 source_sha256=_source_sha256(), profile=profile)


def primary_specs(plan):
    return [d for d in plan["datasets"] if d.get("role") == "m1_matched"]


def summarize_draws(draws, targets):
    if len(draws) < 2:
        raise ValueError("at least two validation draws are required")
    rows = []
    for metric, tolerance in TOLERANCES.items():
        values = np.asarray([r[metric] for r in draws], dtype=float)
        target = float(targets[metric])
        if not np.all(np.isfinite(values)) or not np.isfinite(target):
            raise ValueError(f"non-finite calibration metric: {metric}")
        divisor = max(abs(target), 1e-12) if metric in RELATIVE_METRICS else 1.
        mean, se = float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values)))
        rows.append(dict(metric=metric, target=target, generated_mean=mean,
                         monte_carlo_se=se, scaled_difference=(mean-target)/divisor,
                         tolerance=tolerance, mean_within_tolerance=abs(mean-target)/divisor <= tolerance,
                         systematic_tolerance_miss=max(0., abs(mean-target)-1.96*se)/divisor > tolerance))
    return rows


def run(root, config_path, profile, index):
    plan, execution, identity = binding(root, config_path, profile)
    specs = primary_specs(plan)
    if not 0 <= index < len(specs):
        raise ValueError("invalid calibration task index")
    spec = specs[index]
    destination = root / "calibration_validation" / f"rep{index:03d}.json"
    if destination.exists():
        old = read(destination)
        if old.get("identity") != identity or old.get("status") != "complete":
            raise FileExistsError("calibration artifact differs; use a new output root")
        return
    settings = execution["generator"]
    pilot_size = execution["smoke"]["pilot_size"] if profile == "smoke" else settings["pilot_size"]
    cfg = metadata_config_for_spec(spec, plan["master_seed"], settings, pilot_size)
    if cfg.calibration_method != "finite-sample-v2":
        raise ValueError("this audit requires finite-sample-v2 calibration")
    metadata = load_metadata_profile(cfg.profile_id)
    data = generate_metadata_dag_dataset(cfg)
    biological = [n for n in data.train if n not in {"age_z", "female", "PD"}]
    gate = execution["metadata_calibration_gate"]
    repeats = int(gate["validation_repeats"])
    if repeats < 20:
        raise ValueError("calibration validation requires >=20 disjoint samples")
    rows, all_draws = [], []
    for prevalence in dict.fromkeys([cfg.female_prevalence, *gate["female_sensitivity"]]):
        primary = prevalence == cfg.female_prevalence
        local_cfg = cfg if primary else replace(cfg, female_prevalence=prevalence,
                            calibration_scales=(data.manifest["calibration"]["selected_scale"],))
        local = data if primary else generate_metadata_dag_dataset(local_cfg)
        draws = []
        for repeat in range(repeats):
            frame = _sample(metadata.participants, f"calibration-validate-{repeat:03d}",
                            local.manifest["topological_order"], local.manifest["mechanisms"],
                            local.variable_types, metadata, local_cfg)
            metrics = metadata_fidelity_statistics(frame, biological, metadata)
            draws.append(metrics)
            all_draws.append(dict(repeat=repeat, female_prevalence=prevalence, primary=primary, **metrics))
        rows.extend(dict(female_prevalence=prevalence, primary=primary, **r)
                    for r in summarize_draws(draws, local.manifest["calibration"]["target"]))
    if _source_sha256() != identity["source_sha256"]:
        raise RuntimeError("source changed during calibration")
    _atomic_json(destination, dict(schema="ordersmc-finite-sample-calibration-validation-v2",
        status="complete", identity=identity, dataset_id=spec["dataset_id"], config=asdict(cfg),
        selected_scale=data.manifest["calibration"]["selected_scale"],
        generator_manifest_sha256=_sha(data.manifest), calibration=data.manifest["calibration"],
        repeats=repeats, uses_graph_recovery=False, rows=rows, draws=all_draws))
    print(f"calibration complete: {spec['dataset_id']}", flush=True)


def aggregate(root, config_path, profile):
    plan, execution, identity = binding(root, config_path, profile)
    records, missing = [], []
    for index, spec in enumerate(primary_specs(plan)):
        path = root / "calibration_validation" / f"rep{index:03d}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        record = read(path)
        if record["status"] != "complete" or record["identity"] != identity or record["dataset_id"] != spec["dataset_id"]:
            raise RuntimeError("calibration validation identity mismatch")
        records.append(record)
    rows = [dict(dataset_id=r["dataset_id"], selected_scale=r["selected_scale"], **row)
            for r in records for row in r["rows"]]
    fails = [r for r in rows if r["primary"] and r["systematic_tolerance_miss"]]
    report = dict(schema="ordersmc-finite-sample-calibration-gate-v2", identity=identity,
                  status="passed" if records and not missing and not fails else "failed",
                  missing=missing, systematic_misses=fails, datasets=len(records),
                  selection_uses_graph_recovery=False, tolerances=dict(TOLERANCES),
                  criterion="no per-graph systematic misses on disjoint validation draws; no graph exclusions",
                  female_prevalence_assumption=execution["generator"]["female_prevalence"],
                  female_sensitivity_is_diagnostic_not_a_fit_selection_rule=True,
                  source_artifacts={f"rep{i:03d}.json":sha(root / "calibration_validation" / f"rep{i:03d}.json")
                                    for i in range(len(primary_specs(plan)))
                                    if (root / "calibration_validation" / f"rep{i:03d}.json").exists()})
    _atomic_json(root / "validation/calibration-gate.json", report)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(root / "calibration_validation/by_graph.csv", index=False)
        frame.groupby(["primary", "female_prevalence", "metric"], as_index=False).agg(
            target=("target", "first"), generated_mean=("generated_mean", "mean"),
            fraction_means_within_tolerance=("mean_within_tolerance", "mean"),
            systematic_misses=("systematic_tolerance_miss", "sum")).to_csv(root / "calibration_validation/summary.csv", index=False)
    if report["status"] != "passed":
        raise RuntimeError(f"calibration gate failed: {len(missing)} missing cases; {len(fails)} systematic misses")
    return report


def validate_gate(root, config_path, profile, prepared=False):
    plan, execution, identity = binding(root, config_path, profile)
    if not execution.get("metadata_calibration_gate", {}).get("required", False):
        return
    gate = read(root / "validation/calibration-gate.json")
    if gate["status"] != "passed" or gate["identity"] != identity:
        raise RuntimeError("metadata calibration gate is missing, failed or stale")
    expected = {f"rep{i:03d}.json" for i in range(len(primary_specs(plan)))}
    if not expected or set(gate["source_artifacts"]) != expected:
        raise RuntimeError("calibration gate has an incomplete artifact inventory")
    for relative, digest in gate["source_artifacts"].items():
        if sha(root / "calibration_validation" / relative) != digest:
            raise RuntimeError("calibration artifact changed after audit")
    if prepared:
        for index, spec in enumerate(primary_specs(plan)):
            reference = read(root / "calibration_validation" / f"rep{index:03d}.json")
            actual = read(root / "datasets" / spec["dataset_id"] / "generator_manifest.json")
            body = {k:v for k,v in actual.items() if k not in {"artifacts", "artifact_fingerprint_sha256"}}
            if _sha(body) != reference["generator_manifest_sha256"]:
                raise RuntimeError("prepared primary generator differs from the audited mechanism")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "aggregate", "validate"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--profile", choices=["smoke", "pilot"], required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--prepared", action="store_true")
    a = p.parse_args()
    if a.action == "run": run(a.root,a.config,a.profile,a.index)
    elif a.action == "aggregate": aggregate(a.root,a.config,a.profile)
    else: validate_gate(a.root,a.config,a.profile,a.prepared)


if __name__ == "__main__":
    main()
