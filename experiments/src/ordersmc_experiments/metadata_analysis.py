"""M0 fidelity audit for Section 2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

TOLERANCES: Mapping[str, float] = {
    "pd_prevalence": .08,
    "age_pd_correlation": .08,
    "female_pd_correlation": .08,
    "absolute_correlation_median": .05,
    "absolute_correlation_q95": .08,
    "effective_rank": .25,
    "components_80pct": .25,
    "skewness_median": .25,
    "excess_kurtosis_median": .75,
    "mean_abs_correlation_with_age_z": .06,
    "mean_abs_correlation_with_female": .06,
    "mean_abs_correlation_with_PD": .06,
}
RELATIVE_METRICS = {"effective_rank", "components_80pct"}


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path) -> dict[str, object]:
    plan = _read(root / "plan.json")
    if plan.get("experiment") != "s2":
        raise ValueError("metadata calibration audit requires an s2 plan")
    rows: list[dict[str, object]] = []
    for spec in plan["datasets"]:
        if spec.get("role") != "m1_matched":
            continue
        manifest = _read(
            root / "datasets" / str(spec["dataset_id"]) / "generator_manifest.json"
        )
        calibration = manifest["calibration"]
        if calibration.get("selection_uses_graph_recovery") is not False:
            raise RuntimeError("generator calibration consulted a recovery result")
        for split in ("train", "independent_test"):
            target, achieved = calibration["target"], calibration[split]
            for metric, tolerance in TOLERANCES.items():
                difference = float(achieved[metric]) - float(target[metric])
                scaled = (
                    difference / max(abs(float(target[metric])), 1e-12)
                    if metric in RELATIVE_METRICS else difference
                )
                rows.append({
                    "dataset_id": spec["dataset_id"],
                    "metadata_profile": spec["metadata_profile"],
                    "replicate": spec["replicate"],
                    "split": split,
                    "metric": metric,
                    "target": float(target[metric]),
                    "achieved": float(achieved[metric]),
                    "difference": difference,
                    "scaled_difference": scaled,
                    "tolerance": tolerance,
                    "within_tolerance": abs(scaled) <= tolerance,
                    "selected_scale": calibration["selected_scale"],
                    "selection_uses_graph_recovery": False,
                })
    if not rows:
        raise RuntimeError("no prepared M1 datasets were found")
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis / "m0_calibration_runs.csv", rows)
    summaries = []
    keys = sorted({
        (str(row["metadata_profile"]), str(row["split"]), str(row["metric"]))
        for row in rows
    })
    for profile, split, metric in keys:
        group = [
            row for row in rows
            if row["metadata_profile"] == profile
            and row["split"] == split and row["metric"] == metric
        ]
        scaled = np.asarray([float(row["scaled_difference"]) for row in group])
        summaries.append({
            "metadata_profile": profile,
            "split": split,
            "metric": metric,
            "target": float(group[0]["target"]),
            "mean_achieved": float(np.mean([float(row["achieved"]) for row in group])),
            "mean_scaled_difference": float(np.mean(scaled)),
            "maximum_absolute_scaled_difference": float(np.max(np.abs(scaled))),
            "fraction_within_tolerance": float(np.mean([
                bool(row["within_tolerance"]) for row in group
            ])),
            "datasets": len(group),
            "tolerance": float(group[0]["tolerance"]),
        })
    _write_csv(analysis / "m0_calibration_summary.csv", summaries)
    report = {
        "schema": "ordersmc-section2-m0-calibration-audit-v1",
        "status": "diagnostic-complete",
        "confirmatory_gate_active": False,
        "reason": "protein pilot precedes the confirmatory calibration freeze",
        "selection_uses_graph_recovery": False,
        "datasets": len({str(row["dataset_id"]) for row in rows}),
        "metrics": len(TOLERANCES),
        "all_individual_values_within_provisional_tolerance": all(
            bool(row["within_tolerance"]) for row in rows
        ),
        "provisional_tolerances": dict(TOLERANCES),
    }
    (analysis / "m0_calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    report = audit(parser.parse_args(argv).root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
