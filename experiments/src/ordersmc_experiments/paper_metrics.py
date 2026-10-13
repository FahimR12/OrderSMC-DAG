"""Recompute paper metrics from an immutable experiment result tree."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score

from ordersmc.graphs import optimal_bayes_action_dag
from .metrics import dagbagm_graph_recovery_metrics


# HiGHS solves the large-graph MILPs in floating arithmetic even though
# ``optimal_bayes_action_dag`` audits returned incumbents with exact sums of
# the input binary64 coefficients. This tolerance only decides whether a tiny
# solver discrepancy aborts the paper analysis. When a sub-tolerance
# discrepancy is found, the strictly higher-utility audited incumbent is
# retained; the scientific pair-state Hamming objective is not relaxed.
PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE = 2e-9


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _bayes_action_audit_fields(graph: nx.DiGraph) -> dict[str, object]:
    certification = graph.graph.get("exact_primary_certification", {})
    if not isinstance(certification, Mapping):
        certification = {}
    return {
        "bayes_action_audit_status": str(certification.get("status", "unknown")),
        "bayes_action_milp_objective_audit_tolerance": float(
            graph.graph["milp_objective_audit_tolerance"]
        ),
        "bayes_action_milp_primary_vs_certified_regret": float(
            graph.graph["milp_primary_vs_certified_regret"]
        ),
    }


def _probabilities(result: Mapping[str, object]) -> dict[tuple[str, str], float]:
    return {
        (str(row["parent"]), str(row["child"])): float(
            np.clip(float(row["probability"]), 0.0, 1.0)
        )
        for row in result["edge_probabilities"]
    }


def _jaccard(left: set[object], right: set[object]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _incident_recovery_metrics(
    estimated: set[tuple[str, str]],
    truth: set[tuple[str, str]],
    node: str,
    prefix: str,
) -> dict[str, float]:
    """Recovery metrics for directed edges incident to one prespecified node."""

    estimated_incident = {edge for edge in estimated if node in edge}
    true_incident = {edge for edge in truth if node in edge}
    true_positive = len(estimated_incident & true_incident)
    false_positive = len(estimated_incident - true_incident)
    false_negative = len(true_incident - estimated_incident)
    selected = true_positive + false_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        f"{prefix}_true_edge_count": float(len(true_incident)),
        f"{prefix}_selected_edge_count": float(selected),
        f"{prefix}_false_positive_count": float(false_positive),
        f"{prefix}_tpr": (
            float(true_positive / len(true_incident))
            if true_incident else float("nan")
        ),
        f"{prefix}_fdr": float(false_positive / max(1, selected)),
        f"{prefix}_f1": (
            float(2 * true_positive / denominator)
            if true_incident and denominator else float("nan")
        ),
    }


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    return tuple(map(float, np.quantile(np.mean(values[indices], axis=1), [0.025, 0.975])))


def _dataset_means(
    rows: Iterable[Mapping[str, object]], metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped.setdefault(str(row["dataset_id"]), []).append(float(value))
    return {dataset: float(np.mean(values)) for dataset, values in grouped.items()}


def _skeleton_brier(
    probabilities: Mapping[tuple[str, str], float],
    truth: nx.DiGraph,
    nodes: tuple[str, ...],
) -> float:
    residuals: list[float] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            probability = min(
                1.0,
                probabilities.get((left, right), 0.0)
                + probabilities.get((right, left), 0.0),
            )
            observed = float(truth.has_edge(left, right) or truth.has_edge(right, left))
            residuals.append((probability - observed) ** 2)
    return float(np.mean(residuals))


def _posterior_metrics(
    probabilities: Mapping[tuple[str, str], float],
    truth: nx.DiGraph,
    nodes: tuple[str, ...],
) -> dict[str, float]:
    edges = tuple((left, right) for left in nodes for right in nodes if left != right)
    observed = np.asarray([float(truth.has_edge(*edge)) for edge in edges])
    predicted = np.asarray([float(probabilities.get(edge, 0.0)) for edge in edges])
    skeleton_observed: list[float] = []
    skeleton_predicted: list[float] = []
    expected_shd = 0.0
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            probability = min(
                1.0,
                float(probabilities.get((left, right), 0.0))
                + float(probabilities.get((right, left), 0.0)),
            )
            skeleton_predicted.append(probability)
            skeleton_observed.append(
                float(truth.has_edge(left, right) or truth.has_edge(right, left))
            )
            if truth.has_edge(left, right):
                expected_shd += 1.0 - probabilities.get((left, right), 0.0)
            elif truth.has_edge(right, left):
                expected_shd += 1.0 - probabilities.get((right, left), 0.0)
            else:
                expected_shd += probability
    sk_observed = np.asarray(skeleton_observed)
    sk_predicted = np.asarray(skeleton_predicted)
    return {
        "directed_auprc": float(average_precision_score(observed, predicted)),
        "directed_brier": float(np.mean((predicted - observed) ** 2)),
        "skeleton_auprc": float(
            average_precision_score(sk_observed, sk_predicted)
        ),
        "skeleton_brier": float(np.mean((sk_predicted - sk_observed) ** 2)),
        "posterior_expected_pair_state_shd": float(expected_shd),
    }


def analyse(root: Path) -> None:
    plan = _read_json(root / "plan.json")
    completion = _read_json(root / "analysis" / "completion.json")
    if (
        int(completion.get("planned_runs", -1))
        != int(completion.get("completed_runs", -2))
        or int(completion.get("missing_or_failed_runs", -1)) != 0
    ):
        raise RuntimeError(
            "paper metrics require a complete cohort; inspect analysis/completion.json"
        )
    manifests: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    probability_maps: dict[str, dict[tuple[str, str], float]] = {}
    representative_edges: dict[str, set[tuple[str, str]]] = {}
    posterior_thresholds: dict[str, float] = {}

    for result_path in sorted((root / "runs").glob("*.json")):
        result = _read_json(result_path)
        if result.get("status") != "complete":
            continue
        run = result["run"]
        assert isinstance(run, Mapping)
        dataset_id = str(run["dataset_id"])
        if dataset_id not in manifests:
            manifests[dataset_id] = _read_json(
                root / "datasets" / dataset_id / "generator_manifest.json"
            )
        manifest = manifests[dataset_id]
        variable_types = {
            str(node): str(kind)
            for node, kind in dict(manifest["variable_types"]).items()
        }
        nodes = tuple(sorted(variable_types))
        truth = nx.DiGraph()
        truth.add_nodes_from(nodes)
        truth.add_edges_from(tuple(map(tuple, manifest["edges"])))
        probabilities = _probabilities(result)
        dataset_spec = dict(result["dataset"])
        manifest_config = manifest.get("config", {})
        maximum_indegree = int(
            dict(manifest_config).get("max_indegree", dataset_spec["max_indegree"])
        )
        try:
            representative = optimal_bayes_action_dag(
                probabilities,
                nodes,
                max_indegree=maximum_indegree,
                loss="pair_state_hamming",
                milp_objective_audit_tolerance=(
                    PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE
                ),
            )
        except Exception as error:
            raise RuntimeError(f"Bayes-action reduction failed for {result_path.name}") from error
        recovery = dagbagm_graph_recovery_metrics(
            representative,
            truth,
            nodes,
            node_types=variable_types,
        )
        if "PD" in nodes:
            recovery.update(
                _incident_recovery_metrics(
                    set(representative.edges()), set(truth.edges()), "PD", "pd_directed"
                )
            )
        original_metrics = result.get("metrics", {})
        assert isinstance(original_metrics, Mapping)
        run_id = str(run["run_id"])
        dimension = int(dataset_spec["dimension"])
        prepared = _read_json(root / "datasets" / dataset_id / "prepared.json")
        probability_maps[run_id] = probabilities
        representative_edges[run_id] = set(representative.edges())
        posterior_thresholds[run_id] = float(result.get("posterior_threshold", 0.5))
        record: dict[str, object] = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "cell_id": str(dataset_spec.get("cell_id", f"d{dimension:03d}")),
            "dimension": dimension,
            "n_train": int(dataset_spec["n_train"]),
            "benchmark_n_train": int(dataset_spec.get("benchmark_n_train", dataset_spec["n_train"])),
            "n_over_d": dataset_spec.get("n_over_d", ""),
            "graph_family": str(dataset_spec.get("graph_family", "")),
            "signal": str(dataset_spec.get("signal", "")),
            "continuous_noise": str(dataset_spec.get("continuous_noise", "gaussian")),
            "variable_type_regime": str(
                dataset_spec.get("variable_type_regime", "balanced_mixed")
            ),
            "dataset_role": str(dataset_spec.get("role", "primary")),
            "replicate": int(dataset_spec.get("replicate", 0)),
            "metadata_profile": str(dataset_spec.get("metadata_profile", "")),
            "data_variant": str(dataset_spec.get("data_variant", "")),
            "causal_regime": str(dataset_spec.get("causal_regime", "")),
            "negative_control": str(dataset_spec.get("negative_control", "none")),
            "sample_multiplier": dataset_spec.get("sample_multiplier", ""),
            "pairing_id": str(dataset_spec.get("pairing_id", "")),
            "method": str(run["method"]),
            "target": str(run["target"]),
            "role": str(run["role"]),
            "wall_seconds": "" if run.get("wall_seconds") is None else int(run["wall_seconds"]),
            "seed_index": "" if run.get("seed_index") is None else int(run["seed_index"]),
            "profile": str(result.get("profile", plan.get("settings", {}).get("profile", ""))),
            "runtime_seconds": float(result["runtime_seconds"]),
            "budget_overrun": bool(result.get("budget_overrun", False)),
            "candidate_build_seconds": float(prepared["candidate_build_seconds"]),
            "end_to_end_seconds": float(prepared["candidate_build_seconds"])
            + float(result["runtime_seconds"]),
            "candidate_true_parent_recall": float(
                prepared["true_directed_parent_recall"]
            ),
            "candidate_complete_family_recall": float(prepared["complete_family_recall"]),
            "skeleton_brier": _skeleton_brier(probabilities, truth, nodes),
            **_bayes_action_audit_fields(representative),
            **recovery,
        }
        for name in (
            "directed_auprc",
            "directed_brier",
            "skeleton_auprc",
            "posterior_expected_pair_state_shd",
            "pip_rmse_to_exact_order",
        ):
            if name in original_metrics and original_metrics[name] is not None:
                record[name] = float(original_metrics[name])
        records.append(record)

    exact_probabilities = {
        str(row["dataset_id"]): probability_maps[str(row["run_id"])]
        for row in records
        if row["method"] == "exact-order-dp"
    }
    for row in records:
        reference = exact_probabilities.get(str(row["dataset_id"]))
        if reference is None:
            continue
        current = probability_maps[str(row["run_id"])]
        row["pip_rmse_to_exact_order"] = math.sqrt(
            sum((current[edge] - reference[edge]) ** 2 for edge in reference)
            / len(reference)
        )

    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "paper_metrics_manifest.json",
        {
            "schema": "ordersmc-paper-metrics-policy-v1",
            "plan_sha256": str(plan.get("plan_sha256", "")),
            "master_seed": int(plan["master_seed"]),
            "bayes_action_loss": "pair_state_hamming",
            "milp_objective_audit_tolerance": (
                PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE
            ),
            "milp_audit_policy": (
                "abort above tolerance; below tolerance retain the strictly "
                "higher exact-binary64-rational utility incumbent"
            ),
            "seed_aggregation": "mean_pip_then_bayes_action",
            "bootstrap_replicates": 10_000,
            "bootstrap_bit_generator": "PCG64DXSM",
        },
    )
    _write_csv(output / "dagbagm_metrics_runs.csv", records)

    rng = np.random.Generator(np.random.PCG64DXSM(int(plan["master_seed"])))
    excluded = {
        "run_id", "dataset_id", "cell_id", "dimension", "n_train", "benchmark_n_train", "n_over_d",
        "graph_family", "signal", "continuous_noise", "variable_type_regime",
        "dataset_role", "replicate", "metadata_profile", "data_variant", "causal_regime",
        "negative_control", "sample_multiplier", "pairing_id",
        "method", "target", "role", "wall_seconds", "seed_index",
        "budget_overrun",
        "bayes_action_audit_status",
        "bayes_action_milp_objective_audit_tolerance",
        "bayes_action_milp_primary_vs_certified_regret",
    }
    metric_names = sorted(
        {
            name
            for row in records
            for name, value in row.items()
            if name not in excluded and isinstance(value, (int, float))
        }
    )
    summaries: list[dict[str, object]] = []
    summary_groups = sorted(
        {
            (
                str(row["cell_id"]), int(row["dimension"]), str(row["dataset_role"]),
                str(row["method"]), str(row["role"]), row["wall_seconds"],
            )
            for row in records
        },
        key=lambda key: tuple(map(str, key)),
    )
    for cell_id, dimension, dataset_role, method, role, wall_seconds in summary_groups:
            group = [
                row for row in records
                if int(row["dimension"]) == dimension
                and str(row["cell_id"]) == cell_id
                and str(row["method"]) == method
                and str(row["role"]) == role
                and row["wall_seconds"] == wall_seconds
            ]
            for metric in metric_names:
                by_dataset = _dataset_means(group, metric)
                values = np.asarray(list(by_dataset.values()), dtype=float)
                if not len(values):
                    continue
                lower, upper = _bootstrap_mean(values, rng)
                summaries.append(
                    {
                        "dimension": dimension,
                        "cell_id": cell_id,
                        "dataset_role": dataset_role,
                        "method": method,
                        "role": role,
                        "wall_seconds": wall_seconds,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n_datasets": len(values),
                        "seed_aggregation": "mean_within_dataset",
                    }
                )
    _write_csv(output / "dagbagm_metrics_summary.csv", summaries)

    comparisons: list[dict[str, object]] = []
    comparison_groups = sorted(
        {
            (
                str(row["cell_id"]), int(row["dimension"]), str(row["dataset_role"]),
                str(row["role"]), row["wall_seconds"],
            )
            for row in records
            if row["method"] in {"ordersmc", "order-mcmc"}
        },
        key=lambda key: tuple(map(str, key)),
    )
    for cell_id, dimension, dataset_role, role, wall_seconds in comparison_groups:
        dimension_rows = [
            row for row in records
            if int(row["dimension"]) == dimension
            and str(row["cell_id"]) == cell_id
            and str(row["role"]) == role
            and row["wall_seconds"] == wall_seconds
        ]
        for metric in metric_names:
            order_smc = _dataset_means(
                (row for row in dimension_rows if row["method"] == "ordersmc"), metric
            )
            order_mcmc = _dataset_means(
                (row for row in dimension_rows if row["method"] == "order-mcmc"), metric
            )
            common = sorted(set(order_smc) & set(order_mcmc))
            if not common:
                continue
            differences = np.asarray(
                [order_smc[key] - order_mcmc[key] for key in common], dtype=float
            )
            lower, upper = _bootstrap_mean(differences, rng)
            comparisons.append(
                {
                    "dimension": dimension,
                    "cell_id": cell_id,
                    "dataset_role": dataset_role,
                    "contrast": "ordersmc_minus_order_mcmc",
                    "role": role,
                    "wall_seconds": wall_seconds,
                    "metric": metric,
                    "mean_difference": float(np.mean(differences)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_paired_datasets": len(common),
                }
            )
    _write_csv(output / "dagbagm_metrics_comparisons.csv", comparisons)

    exact_gate: list[dict[str, object]] = []
    gate_rows = [
        row
        for row in records
        if row["method"] in {"ordersmc", "order-mcmc"}
        and isinstance(row.get("pip_rmse_to_exact_order"), (int, float))
    ]
    for cell_id, method in sorted(
        {(str(row["cell_id"]), str(row["method"])) for row in gate_rows}
    ):
        group = [
            row
            for row in gate_rows
            if row["cell_id"] == cell_id and row["method"] == method
        ]
        values = np.asarray(
            list(_dataset_means(group, "pip_rmse_to_exact_order").values()), dtype=float
        )
        if not len(values):
            continue
        mean = float(np.mean(values))
        percentile = float(np.quantile(values, 0.95))
        profile = str(group[0]["profile"])
        evaluable = profile != "smoke" and len(values) >= 3
        passed = mean <= 0.012 and percentile <= 0.020
        exact_gate.append(
            {
                "cell_id": cell_id,
                "dataset_role": str(group[0]["dataset_role"]),
                "dimension": int(group[0]["dimension"]),
                "method": method,
                "n_datasets": len(values),
                "mean_pip_rmse": mean,
                "empirical_p95_pip_rmse": percentile,
                "maximum_pip_rmse": float(np.max(values)),
                "mean_limit": 0.012,
                "p95_limit": 0.020,
                "profile": profile,
                "gate_evaluable": evaluable,
                "gate_status": (
                    "pass"
                    if evaluable and method == "ordersmc" and passed
                    else "fail"
                    if evaluable and method == "ordersmc"
                    else "underpowered_smoke"
                    if method == "ordersmc"
                    else "not_applicable"
                ),
                "passes_ordersmc_gate": (
                    passed
                    if method == "ordersmc" and evaluable
                    else "not_evaluable"
                    if method == "ordersmc"
                    else "not_applicable"
                ),
            }
        )
    _write_csv(output / "exact_pip_gate.csv", exact_gate)

    # Random seeds are repeated measurements, not independent datasets. Average
    # their PIPs first, then construct the representative DAG and paper metrics.
    seed_groups: dict[tuple[str, str, str, object], list[str]] = {}
    record_by_run = {str(row["run_id"]): row for row in records}
    for row in records:
        if row["seed_index"] != "":
            seed_groups.setdefault(
                (
                    str(row["dataset_id"]),
                    str(row["method"]),
                    str(row["role"]),
                    row["wall_seconds"],
                ),
                [],
            ).append(str(row["run_id"]))
    averaged_pips: list[dict[str, object]] = []
    averaged_graphs: list[dict[str, object]] = []
    averaged_metrics: list[dict[str, object]] = []
    ambiguity: list[dict[str, object]] = []
    for (dataset_id, method, role, wall_seconds), run_ids in sorted(
        seed_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        if len(run_ids) < 2:
            continue
        seed_maps = [probability_maps[run_id] for run_id in sorted(run_ids)]
        edges = sorted(seed_maps[0])
        matrix = np.asarray(
            [[values[edge] for edge in edges] for values in seed_maps], dtype=float
        )
        mean_values = np.mean(matrix, axis=0)
        mean_probabilities = dict(zip(edges, map(float, mean_values)))
        thresholds = {posterior_thresholds[run_id] for run_id in run_ids}
        if len(thresholds) != 1:
            raise RuntimeError(f"inconsistent posterior thresholds for {dataset_id}/{method}")
        threshold = thresholds.pop()
        first = record_by_run[run_ids[0]]
        manifest = manifests[dataset_id]
        variable_types = {
            str(node): str(kind)
            for node, kind in dict(manifest["variable_types"]).items()
        }
        nodes = tuple(sorted(variable_types))
        truth = nx.DiGraph()
        truth.add_nodes_from(nodes)
        truth.add_edges_from(tuple(map(tuple, manifest["edges"])))
        maximum_indegree = int(
            dict(manifest.get("config", {})).get(
                "max_indegree", int(first["dimension"])
            )
        )
        representative = optimal_bayes_action_dag(
            mean_probabilities,
            nodes,
            max_indegree=maximum_indegree,
            loss="pair_state_hamming",
            milp_objective_audit_tolerance=(
                PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE
            ),
        )
        selected = set(representative.edges())
        recovery = dagbagm_graph_recovery_metrics(
            representative, truth, nodes, node_types=variable_types
        )
        if "PD" in nodes:
            recovery.update(
                _incident_recovery_metrics(
                    set(representative.edges()), set(truth.edges()), "PD", "pd_directed"
                )
            )
        averaged_metrics.append(
            {
                "dataset_id": dataset_id,
                "cell_id": first["cell_id"],
                "dimension": first["dimension"],
                "dataset_role": first["dataset_role"],
                "replicate": first["replicate"],
                "metadata_profile": first["metadata_profile"],
                "data_variant": first["data_variant"],
                "causal_regime": first["causal_regime"],
                "negative_control": first["negative_control"],
                "sample_multiplier": first["sample_multiplier"],
                "pairing_id": first["pairing_id"],
                "method": method,
                "role": role,
                "wall_seconds": wall_seconds,
                "seed_count": len(run_ids),
                "aggregation": "mean_pip_then_bayes_action",
                **_bayes_action_audit_fields(representative),
                **_posterior_metrics(mean_probabilities, truth, nodes),
                **recovery,
            }
        )
        ambiguous_count = 0
        threshold_disagreement_count = 0
        bayes_disagreement_count = 0
        for edge_index, (parent, child) in enumerate(edges):
            values = matrix[:, edge_index]
            mean_probability = float(mean_values[edge_index])
            threshold_count = int(np.sum(values >= threshold))
            bayes_count = sum(
                int((parent, child) in representative_edges[run_id])
                for run_id in run_ids
            )
            near_threshold = abs(mean_probability - threshold) <= 0.02
            threshold_disagreement = 0 < threshold_count < len(run_ids)
            bayes_disagreement = 0 < bayes_count < len(run_ids)
            ambiguous_count += int(near_threshold)
            threshold_disagreement_count += int(threshold_disagreement)
            bayes_disagreement_count += int(bayes_disagreement)
            averaged_pips.append(
                {
                    "dataset_id": dataset_id,
                    "cell_id": first["cell_id"],
                    "dimension": first["dimension"],
                    "method": method,
                    "role": role,
                    "wall_seconds": wall_seconds,
                    "parent": parent,
                    "child": child,
                    "seed_count": len(run_ids),
                    "mean_pip": mean_probability,
                    "seed_sd": float(np.std(values, ddof=1)),
                    "minimum_pip": float(np.min(values)),
                    "maximum_pip": float(np.max(values)),
                    "posterior_threshold": threshold,
                    "within_0_02_of_threshold": near_threshold,
                    "threshold_selection_frequency": threshold_count / len(run_ids),
                    "bayes_action_selection_frequency": bayes_count / len(run_ids),
                    "selected_in_seed_averaged_bayes_action": (parent, child) in selected,
                }
            )
            if (parent, child) in selected:
                averaged_graphs.append(
                    {
                        "dataset_id": dataset_id,
                        "cell_id": first["cell_id"],
                        "dimension": first["dimension"],
                        "method": method,
                        "role": role,
                        "wall_seconds": wall_seconds,
                        "parent": parent,
                        "child": child,
                        "mean_pip": mean_probability,
                        "seed_count": len(run_ids),
                    }
                )
        ambiguity.append(
            {
                "dataset_id": dataset_id,
                "cell_id": first["cell_id"],
                "dimension": first["dimension"],
                "method": method,
                "role": role,
                "wall_seconds": wall_seconds,
                "seed_count": len(run_ids),
                "edge_count": len(edges),
                "within_0_02_of_threshold_count": ambiguous_count,
                "threshold_selection_disagreement_count": threshold_disagreement_count,
                "bayes_action_selection_disagreement_count": bayes_disagreement_count,
                "mean_edge_seed_sd": float(np.mean(np.std(matrix, axis=0, ddof=1))),
                "maximum_edge_seed_sd": float(np.max(np.std(matrix, axis=0, ddof=1))),
            }
        )
    _write_csv(output / "seed_averaged_pips.csv", averaged_pips)
    _write_csv(output / "seed_averaged_representative_dags.csv", averaged_graphs)
    _write_csv(output / "seed_averaged_dagbagm_metrics.csv", averaged_metrics)
    _write_csv(output / "seed_ambiguity_summary.csv", ambiguity)
    averaged_summaries: list[dict[str, object]] = []
    averaged_groups = sorted(
        {
            (
                str(row["cell_id"]),
                int(row["dimension"]),
                str(row["dataset_role"]),
                str(row["method"]),
                str(row["role"]),
                row["wall_seconds"],
            )
            for row in averaged_metrics
        },
        key=lambda key: tuple(map(str, key)),
    )
    averaged_excluded = {
        "dataset_id",
        "cell_id",
        "dimension",
        "dataset_role",
        "replicate",
        "metadata_profile",
        "data_variant",
        "causal_regime",
        "negative_control",
        "sample_multiplier",
        "pairing_id",
        "method",
        "role",
        "wall_seconds",
        "seed_count",
        "aggregation",
        "bayes_action_audit_status",
        "bayes_action_milp_objective_audit_tolerance",
        "bayes_action_milp_primary_vs_certified_regret",
    }
    for cell_id, dimension, dataset_role, method, role, wall_seconds in averaged_groups:
        group = [
            row
            for row in averaged_metrics
            if row["cell_id"] == cell_id
            and row["method"] == method
            and row["role"] == role
            and row["wall_seconds"] == wall_seconds
        ]
        names = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if key not in averaged_excluded and isinstance(value, (int, float))
            }
        )
        for metric in names:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            lower, upper = _bootstrap_mean(values, rng)
            averaged_summaries.append(
                {
                    "cell_id": cell_id,
                    "dimension": dimension,
                    "dataset_role": dataset_role,
                    "method": method,
                    "role": role,
                    "wall_seconds": wall_seconds,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_datasets": len(values),
                    "seed_aggregation": "mean_pip_then_bayes_action",
                }
            )
    _write_csv(
        output / "seed_averaged_dagbagm_metrics_summary.csv", averaged_summaries
    )

    stability_seed_groups: dict[tuple[str, str, object], list[str]] = {}
    for row in records:
        if row["seed_index"] != "":
            stability_seed_groups.setdefault(
                (str(row["dataset_id"]), str(row["method"]), row["wall_seconds"]),
                [],
            ).append(str(row["run_id"]))
    stability: list[dict[str, object]] = []
    for (dataset_id, method, wall_seconds), run_ids in sorted(
        stability_seed_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        pip_rmse: list[float] = []
        directed_jaccard: list[float] = []
        skeleton_jaccard: list[float] = []
        for left, right in combinations(sorted(run_ids), 2):
            keys = sorted(probability_maps[left])
            pip_rmse.append(
                math.sqrt(
                    sum(
                        (probability_maps[left][key] - probability_maps[right][key]) ** 2
                        for key in keys
                    ) / len(keys)
                )
            )
            left_edges = representative_edges[left]
            right_edges = representative_edges[right]
            directed_jaccard.append(_jaccard(set(left_edges), set(right_edges)))
            skeleton_jaccard.append(
                _jaccard(
                    {frozenset(edge) for edge in left_edges},
                    {frozenset(edge) for edge in right_edges},
                )
            )
        if pip_rmse:
            stability.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "role": "all-seed-stability",
                    "wall_seconds": wall_seconds,
                    "seed_count": len(run_ids),
                    "pair_count": len(pip_rmse),
                    "mean_pairwise_pip_rmse": float(np.mean(pip_rmse)),
                    "mean_directed_graph_jaccard": float(np.mean(directed_jaccard)),
                    "mean_skeleton_graph_jaccard": float(np.mean(skeleton_jaccard)),
                }
            )
    _write_csv(output / "seed_stability_dagbagm_metrics.csv", stability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    analyse(args.result_root.resolve())


if __name__ == "__main__":
    main()
