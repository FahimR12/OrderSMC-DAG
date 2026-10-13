"""Prespecified dataset-level summaries and contrasts for paper Section 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


IDENTIFIERS = (
    "dataset_id", "cell_id", "dataset_role", "replicate", "metadata_profile",
    "data_variant", "causal_regime", "negative_control", "sample_multiplier",
    "pairing_id", "method",
)
PRIMARY_METRICS = (
    "directed_auprc",
    "directed_brier",
    "skeleton_f1",
    "directed_f1",
    "posterior_expected_pair_state_shd",
    "directed_cross_type_f1",
    "directed_tpr",
    "directed_fdr",
    "skeleton_tpr",
    "skeleton_fdr",
    "directed_false_positive_count",
    "skeleton_false_positive_count",
    "pd_directed_f1",
    "pd_directed_tpr",
    "pd_directed_fdr",
    "pd_directed_false_positive_count",
)
STABILITY_METRICS = (
    "mean_pairwise_pip_rmse",
    "mean_directed_graph_jaccard",
    "mean_skeleton_graph_jaccard",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if not rows:
            writer.writerow({"status": "not_available_in_this_profile"})
        else:
            writer.writerows(rows)


def _number(row: Mapping[str, object], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = np.mean(values[indices], axis=1)
    return tuple(map(float, np.quantile(means, [.025, .975])))


def _dataset_estimates(analysis: Path) -> list[dict[str, object]]:
    """Use mean-PIP Bayes actions for repeated seeds and raw rows for one-run methods."""

    raw = _read_csv(analysis / "dagbagm_metrics_runs.csv")
    if not raw:
        raise RuntimeError("Section 2 analysis requires dagbagm_metrics_runs.csv")
    metadata: dict[str, dict[str, object]] = {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in raw:
        dataset_id = str(row["dataset_id"])
        current = {field: row.get(field, "") for field in IDENTIFIERS if field != "method"}
        if dataset_id in metadata and metadata[dataset_id] != current:
            raise RuntimeError(f"inconsistent dataset metadata: {dataset_id}")
        metadata[dataset_id] = current
        if row.get("role") != "reproducibility-extension":
            grouped.setdefault((dataset_id, str(row["method"])), []).append(row)

    averaged = {
        (str(row["dataset_id"]), str(row["method"])): row
        for row in _read_csv(analysis / "seed_averaged_dagbagm_metrics.csv")
        if row.get("role") != "reproducibility-extension"
    }
    estimates: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        dataset_id, method = key
        if key in averaged:
            source: Mapping[str, object] = averaged[key]
            aggregation = "mean_pip_then_bayes_action"
            seed_count = int(float(averaged[key]["seed_count"]))
        elif len(rows) == 1:
            source = rows[0]
            aggregation = "single_run"
            seed_count = 1
        else:
            raise RuntimeError(
                f"missing seed-averaged Bayes action for {dataset_id}/{method}"
            )
        record: dict[str, object] = {
            **metadata[dataset_id],
            "method": method,
            "seed_count": seed_count,
            "aggregation": aggregation,
        }
        for metric in PRIMARY_METRICS:
            value = _number(source, metric)
            if value is not None:
                record[metric] = value
        estimates.append(record)
    return estimates


def _summaries(
    rows: Iterable[Mapping[str, object]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        labels = dict(zip(group_fields, key))
        for metric in metrics:
            values = np.asarray(
                [value for row in group if (value := _number(row, metric)) is not None],
                dtype=float,
            )
            if not len(values):
                continue
            lower, upper = _bootstrap_mean(values, rng)
            output.append(
                {
                    **labels,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_independent_datasets": len(values),
                }
            )
    return output


def _paired_contrasts(
    left_rows: Iterable[Mapping[str, object]],
    right_rows: Iterable[Mapping[str, object]],
    *,
    contrast: str,
    formula: str,
    metrics: Sequence[str],
    rng: np.random.Generator,
    match_method: bool = True,
) -> list[dict[str, object]]:
    pairing_fields = ("metadata_profile", "pairing_id", "replicate") + (
        ("method",) if match_method else ()
    )

    def keyed(rows: Iterable[Mapping[str, object]]) -> dict[tuple[str, ...], Mapping[str, object]]:
        result: dict[tuple[str, ...], Mapping[str, object]] = {}
        for row in rows:
            key = tuple(str(row.get(field, "")) for field in pairing_fields)
            if key in result:
                raise RuntimeError(f"duplicate Section 2 pairing key: {key}")
            result[key] = row
        return result

    left, right = keyed(left_rows), keyed(right_rows)
    shared = sorted(set(left) & set(right))
    grouped: dict[tuple[str, str], list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
    for key in shared:
        method = key[3] if match_method else "cross_method"
        grouped.setdefault((key[0], method), []).append((left[key], right[key]))
    output: list[dict[str, object]] = []
    for (profile, method), pairs in sorted(grouped.items()):
        left_method = str(pairs[0][0].get("method", ""))
        right_method = str(pairs[0][1].get("method", ""))
        for metric in metrics:
            differences = []
            for left_row, right_row in pairs:
                left_value = _number(left_row, metric)
                right_value = _number(right_row, metric)
                if left_value is not None and right_value is not None:
                    differences.append(left_value - right_value)
            if not differences:
                continue
            values = np.asarray(differences, dtype=float)
            lower, upper = _bootstrap_mean(values, rng)
            output.append(
                {
                    "metadata_profile": profile,
                    "method": method if match_method else left_method,
                    "left_method": left_method,
                    "right_method": right_method,
                    "contrast": contrast,
                    "metric": metric,
                    "difference_formula": formula,
                    "mean_difference": float(np.mean(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_paired_independent_datasets": len(values),
                }
            )
    return output


def analyse(root: Path) -> None:
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("experiment") != "s2":
        raise ValueError("section2_analysis requires an s2 result tree")
    analysis = root / "analysis"
    output = analysis / "section2"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64DXSM(int(plan["master_seed"])))
    estimates = _dataset_estimates(analysis)
    _write_csv(output / "dataset_estimates.csv", estimates)

    m1 = [row for row in estimates if row["dataset_role"] == "m1_matched"]
    m2 = [row for row in estimates if row["dataset_role"] == "m2_gaussian_pair"]
    m3 = [row for row in estimates if row["dataset_role"] == "m3_sample_size"]
    m4 = [row for row in estimates if row["dataset_role"] == "m4_cg_compatible"]
    m5 = [row for row in estimates if row["dataset_role"] == "m5_negative_controls"]

    _write_csv(
        output / "m1_method_summary.csv",
        _summaries(m1, ("metadata_profile", "method"), PRIMARY_METRICS, rng),
    )
    method_contrasts: list[dict[str, object]] = []
    for comparator in sorted({str(row["method"]) for row in m1} - {"ordersmc"}):
        method_contrasts.extend(
            _paired_contrasts(
                (row for row in m1 if row["method"] == "ordersmc"),
                (row for row in m1 if row["method"] == comparator),
                contrast=f"ordersmc_minus_{comparator}",
                formula="metric(OrderSMC) - metric(comparator), paired by generated DAG/data",
                metrics=PRIMARY_METRICS,
                rng=rng,
                match_method=False,
            )
        )
    _write_csv(output / "m1_method_paired_contrasts.csv", method_contrasts)
    _write_csv(
        output / "m2_metadata_minus_gaussian.csv",
        _paired_contrasts(
            m1,
            m2,
            contrast="metadata_minus_gaussian",
            formula="metric(metadata-calibrated) - metric(Gaussian), same DAG/streams",
            metrics=PRIMARY_METRICS,
            rng=rng,
        ),
    )
    m3_profiles = {str(row["metadata_profile"]) for row in m3}
    sample_rows = [
        row for row in m1 if str(row["metadata_profile"]) in m3_profiles
    ] + m3
    _write_csv(
        output / "m3_information_curve.csv",
        _summaries(
            sample_rows,
            ("metadata_profile", "sample_multiplier", "method"),
            PRIMARY_METRICS,
            rng,
        ),
    )
    sample_contrasts: list[dict[str, object]] = []
    for multiplier in ("0.5", "2.0"):
        sample_contrasts.extend(
            _paired_contrasts(
                (row for row in m3 if str(row["sample_multiplier"]) == multiplier),
                m1,
                contrast=f"n{multiplier}_minus_n1.0",
                formula=f"metric(n={multiplier}*observed) - metric(n=observed), same DAG/nested streams",
                metrics=PRIMARY_METRICS,
                rng=rng,
            )
        )
    _write_csv(output / "m3_paired_contrasts.csv", sample_contrasts)
    m4_profiles = {str(row["metadata_profile"]) for row in m4}
    _write_csv(
        output / "m4_end_to_end_summary.csv",
        _summaries(
            [
                row for row in m1
                if str(row["metadata_profile"]) in m4_profiles
            ] + m4,
            ("metadata_profile", "causal_regime", "method"),
            PRIMARY_METRICS,
            rng,
        ),
    )
    _write_csv(
        output / "m5_negative_control_summary.csv",
        _summaries(
            m5,
            ("metadata_profile", "negative_control", "method"),
            PRIMARY_METRICS,
            rng,
        ),
    )

    metadata = {str(row["dataset_id"]): row for row in estimates}
    stability_rows: list[dict[str, object]] = []
    for row in _read_csv(analysis / "seed_stability_dagbagm_metrics.csv"):
        dataset = metadata.get(str(row.get("dataset_id", "")))
        if dataset is None or dataset["dataset_role"] != "m1_matched":
            continue
        stability_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "metadata_profile": dataset["metadata_profile"],
                "method": row["method"],
                **{
                    metric: float(row[metric])
                    for metric in STABILITY_METRICS
                    if _number(row, metric) is not None
                },
            }
        )
    _write_csv(
        output / "m5_seed_stability_summary.csv",
        _summaries(
            stability_rows, ("metadata_profile", "method"), STABILITY_METRICS, rng
        ),
    )
    manifest = {
        "schema": "ordersmc-section2-analysis-v1",
        "plan_sha256": plan["plan_sha256"],
        "master_seed": int(plan["master_seed"]),
        "statistical_unit": "independent generated DAG/data pair",
        "seed_aggregation": "mean PIPs within dataset, then constrained Bayes-action DAG",
        "paired_difference": "delta_i = metric_i(A) - metric_i(B)",
        "paired_mean": "mean_delta = (1/N) * sum_i(delta_i)",
        "confidence_interval": "2.5% and 97.5% percentiles of 10,000 dataset-level bootstrap means",
        "m4_interpretation": "descriptive end-to-end comparison; not a representation-isolation claim",
        "pilot_interpretation": "exploratory until generator and resources are frozen with a fresh confirmatory seed",
        "dataset_estimates": len(estimates),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    analyse(args.result_root.resolve())


if __name__ == "__main__":
    main()
