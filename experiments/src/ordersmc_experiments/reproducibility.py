"""Across-seed posterior and graph stability summaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Mapping, Sequence

import numpy as np

Edge = tuple[str, str]


@dataclass(frozen=True)
class RankLayoutConsistencyResult:
    reference_layout: str
    comparison_layout: str
    dataset_count: int
    minimum_seed_count_per_layout: int
    global_pip_rms_difference: float
    global_pip_rms_ci_lower: float
    global_pip_rms_ci_upper: float
    skeleton_brier_difference: float
    skeleton_brier_ci_lower: float
    skeleton_brier_ci_upper: float
    pip_rms_margin: float
    skeleton_brier_margin: float
    confidence: float
    bootstrap_resamples: int
    equivalent: bool
    paired_seed_set_sha256: str
    aggregation_formula: str = "sqrt(G^-1 sum_g E_g^-1 sum_e delta_ge^2)"


@dataclass(frozen=True)
class MCSECalibrationCell:
    dataset_id: str
    truth_status: str
    pip_stratum: str
    edge_count: int
    ratio_edge_count: int
    seed_count: int
    median_mcse_to_empirical_sd_ratio: float | None
    p90_mcse_to_empirical_sd_ratio: float | None
    nominal_95_coverage: float
    positive_mcse_zero_sd_count: int
    ratio_status: str


@dataclass(frozen=True)
class MCSECalibrationResult:
    cells: tuple[MCSECalibrationCell, ...]
    dataset_count: int
    minimum_seed_count: int
    median_cell_ratio: float | None
    p90_cell_ratio: float | None
    mean_cell_nominal_95_coverage: float
    target_label: str
    support_ids_by_dataset: tuple[tuple[str, str], ...]
    paired_seed_set_sha256: str
    calibration_status: str
    interval_formula: str = "p_hat +/- 1.959963984540054 * internal_MCSE"
    aggregation_formula: str = (
        "summarize edges within dataset/truth/PIP cells; summarize cells within "
        "dataset; average dataset summaries with equal dataset weight"
    )


def _jaccard(left: set[Edge], right: set[Edge]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _complete_pip_vector(
    run: Mapping[Edge, float],
    nodes: tuple[str, ...],
    edges: tuple[Edge, ...],
    *,
    label: str,
) -> list[float]:
    if not isinstance(run, Mapping):
        raise TypeError(f"{label} must be a mapping")
    expected = set(edges)
    observed: set[Edge] = set()
    for raw_edge in run:
        if (
            not isinstance(raw_edge, tuple)
            or len(raw_edge) != 2
            or not all(isinstance(node, str) for node in raw_edge)
        ):
            raise ValueError(f"{label} contains a malformed edge key")
        observed.add((raw_edge[0], raw_edge[1]))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} must contain every directed non-self pair exactly once; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    prepared: dict[Edge, float] = {}
    for edge in edges:
        value = run[edge]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} contains a boolean probability for {edge}")
        probability = float(value)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{label} contains an invalid probability for {edge}")
        prepared[edge] = probability
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            if prepared[(left, right)] + prepared[(right, left)] > 1.0 + 1e-10:
                raise ValueError(
                    f"{label} opposite directed PIPs exceed one for {(left, right)}"
                )
    return [prepared[edge] for edge in edges]


def summarize_seed_stability(
    edge_probability_runs: Sequence[Mapping[Edge, float]],
    nodes: Sequence[str],
    *,
    threshold: float = 0.7,
    top_k: int = 20,
    threshold_margin: float = 0.02,
) -> Dict[str, object]:
    if len(edge_probability_runs) < 2:
        raise ValueError("at least two seed runs are required")
    prepared_nodes = tuple(nodes)
    if (
        len(prepared_nodes) < 2
        or len(set(prepared_nodes)) != len(prepared_nodes)
        or any(not isinstance(node, str) or not node for node in prepared_nodes)
    ):
        raise ValueError("nodes must contain at least two unique non-empty strings")
    if (
        not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
        or not np.isfinite(threshold_margin)
        or threshold_margin < 0
    ):
        raise ValueError("threshold must lie in [0,1] and threshold_margin be non-negative")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    edges = tuple(
        (parent, child)
        for parent in prepared_nodes
        for child in prepared_nodes
        if parent != child
    )
    matrix = np.asarray(
        [
            _complete_pip_vector(
                run,
                prepared_nodes,
                edges,
                label=f"seed run {run_index}",
            )
            for run_index, run in enumerate(edge_probability_runs)
        ],
        dtype=float,
    )
    standard_deviation = np.std(matrix, axis=0, ddof=1)
    pairwise_rms = []
    graph_jaccard = []
    top_jaccard = []
    selected_sets = [
        {edge for edge, probability in zip(edges, row) if probability >= threshold}
        for row in matrix
    ]
    top_sets = [
        {
            edges[index]
            for index in sorted(
                range(len(edges)), key=lambda index: (-float(row[index]), edges[index])
            )[: min(top_k, len(edges))]
        }
        for row in matrix
    ]
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            pairwise_rms.append(float(np.sqrt(np.mean((matrix[left] - matrix[right]) ** 2))))
            graph_jaccard.append(_jaccard(selected_sets[left], selected_sets[right]))
            top_jaccard.append(_jaccard(top_sets[left], top_sets[right]))
    mean_probability = np.mean(matrix, axis=0)
    return {
        "seed_count": len(edge_probability_runs),
        "edge_pip_sd_mean": float(np.mean(standard_deviation)),
        "edge_pip_sd_median": float(np.median(standard_deviation)),
        "edge_pip_sd_p95": float(np.quantile(standard_deviation, 0.95)),
        "edge_pip_sd_max": float(np.max(standard_deviation)),
        "pairwise_pip_rms_mean": float(np.mean(pairwise_rms)),
        "pairwise_pip_rms_max": float(np.max(pairwise_rms)),
        "threshold_graph_jaccard_mean": float(np.mean(graph_jaccard)),
        "threshold_graph_jaccard_min": float(np.min(graph_jaccard)),
        "top_edge_jaccard_mean": float(np.mean(top_jaccard)),
        "top_edge_jaccard_min": float(np.min(top_jaccard)),
        "edges_near_threshold": int(
            np.sum(np.abs(mean_probability - threshold) <= threshold_margin)
        ),
        "edge_selection_frequency": {
            f"{parent}->{child}": float(np.mean(matrix[:, index] >= threshold))
            for index, (parent, child) in enumerate(edges)
        },
    }


def assess_rank_layout_consistency(
    layout_runs: Mapping[
        str, Mapping[str, Mapping[str, Mapping[Edge, float]]]
    ],
    nodes_by_dataset: Mapping[str, Sequence[str]],
    truth_edges_by_dataset: Mapping[str, Sequence[Edge]],
    *,
    reference_layout: str,
    comparison_layout: str,
    minimum_seeds: int = 20,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    pip_rms_margin: float = 0.01,
    skeleton_brier_margin: float = 0.002,
) -> RankLayoutConsistencyResult:
    """Apply the protocol's E10 rank-count equivalence rule.

    Each layout is first averaged over the precommitted independent seeds for
    a dataset. Dataset-level RMS and skeleton-Brier contrasts are then
    bootstrapped by resampling whole paired datasets, never individual edges.
    """

    if reference_layout == comparison_layout:
        raise ValueError("reference and comparison layouts must differ")
    if reference_layout not in layout_runs or comparison_layout not in layout_runs:
        raise ValueError("both requested layouts must be present")
    if isinstance(minimum_seeds, bool) or minimum_seeds < 2:
        raise ValueError("minimum_seeds must be an integer of at least two")
    if isinstance(resamples, bool) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if pip_rms_margin <= 0.0 or skeleton_brier_margin <= 0.0:
        raise ValueError("equivalence margins must be positive")
    reference = layout_runs[reference_layout]
    comparison = layout_runs[comparison_layout]
    dataset_ids = tuple(sorted(set(reference) & set(comparison)))
    if not dataset_ids or set(dataset_ids) != set(nodes_by_dataset) or set(
        dataset_ids
    ) != set(truth_edges_by_dataset):
        raise ValueError("layouts, nodes, and truths must cover the same datasets")
    if set(reference) != set(dataset_ids) or set(comparison) != set(dataset_ids):
        raise ValueError("rank layouts must contain identical paired dataset IDs")

    dataset_mse: list[float] = []
    dataset_brier_difference: list[float] = []
    paired_seed_sets: dict[str, list[str]] = {}
    observed_minimum_seed_count = 10**18
    for dataset_id in dataset_ids:
        nodes = tuple(nodes_by_dataset[dataset_id])
        if len(nodes) < 2 or len(set(nodes)) != len(nodes):
            raise ValueError(f"dataset {dataset_id!r} has invalid nodes")
        directed_edges = tuple(
            (parent, child)
            for parent in nodes
            for child in nodes
            if parent != child
        )

        reference_runs = reference[dataset_id]
        comparison_runs = comparison[dataset_id]
        if not isinstance(reference_runs, Mapping) or not isinstance(
            comparison_runs, Mapping
        ):
            raise ValueError("rank-layout runs must be keyed by method-seed digest")
        reference_seed_ids = set(reference_runs)
        comparison_seed_ids = set(comparison_runs)
        if reference_seed_ids != comparison_seed_ids:
            raise ValueError("rank layouts must use identical paired method-seed digests")
        digest_pattern = re.compile(r"^[0-9a-f]{64}$")
        if any(not digest_pattern.fullmatch(seed_id) for seed_id in reference_seed_ids):
            raise ValueError("method-seed IDs must be lowercase SHA-256 digests")
        ordered_seed_ids = tuple(sorted(reference_seed_ids))
        paired_seed_sets[dataset_id] = list(ordered_seed_ids)

        def mean_run(
            runs: Mapping[str, Mapping[Edge, float]], label: str
        ) -> np.ndarray:
            nonlocal observed_minimum_seed_count
            if len(runs) < minimum_seeds:
                raise ValueError(
                    f"dataset {dataset_id!r} layout {label!r} has fewer than "
                    f"{minimum_seeds} seeds"
                )
            observed_minimum_seed_count = min(observed_minimum_seed_count, len(runs))
            if any(set(runs[seed_id]) != set(directed_edges) for seed_id in ordered_seed_ids):
                raise ValueError(
                    f"dataset {dataset_id!r} layout {label!r} must report every directed PIP"
                )
            matrix = np.asarray(
                [
                    [float(runs[seed_id][edge]) for edge in directed_edges]
                    for seed_id in ordered_seed_ids
                ],
                dtype=float,
            )
            if np.any(~np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
                raise ValueError("edge probabilities must be finite and lie in [0, 1]")
            for left_index in range(len(nodes)):
                for right_index in range(left_index + 1, len(nodes)):
                    forward = left_index * (len(nodes) - 1) + right_index - int(
                        right_index > left_index
                    )
                    reverse = right_index * (len(nodes) - 1) + left_index
                    if np.any(matrix[:, forward] + matrix[:, reverse] > 1.0 + 1e-10):
                        raise ValueError("opposite directed PIPs exceed one")
            return np.mean(matrix, axis=0)

        reference_mean = mean_run(reference[dataset_id], reference_layout)
        comparison_mean = mean_run(comparison[dataset_id], comparison_layout)
        dataset_mse.append(
            float(np.mean((comparison_mean - reference_mean) ** 2))
        )

        edge_index = {edge: index for index, edge in enumerate(directed_edges)}
        truth = set(map(tuple, truth_edges_by_dataset[dataset_id]))
        valid_edges = set(directed_edges)
        if not truth <= valid_edges:
            raise ValueError(f"dataset {dataset_id!r} truth contains an invalid edge")
        reference_losses = []
        comparison_losses = []
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                reference_probability = reference_mean[edge_index[(left, right)]] + reference_mean[
                    edge_index[(right, left)]
                ]
                comparison_probability = comparison_mean[edge_index[(left, right)]] + comparison_mean[
                    edge_index[(right, left)]
                ]
                observed = float((left, right) in truth or (right, left) in truth)
                reference_losses.append((reference_probability - observed) ** 2)
                comparison_losses.append((comparison_probability - observed) ** 2)
        dataset_brier_difference.append(
            float(np.mean(comparison_losses) - np.mean(reference_losses))
        )

    mse_values = np.asarray(dataset_mse, dtype=float)
    brier_values = np.asarray(dataset_brier_difference, dtype=float)
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    indices = rng.integers(0, len(dataset_ids), size=(int(resamples), len(dataset_ids)))
    rms_bootstrap = np.sqrt(np.mean(mse_values[indices], axis=1))
    brier_bootstrap = np.mean(brier_values[indices], axis=1)
    alpha = 1.0 - confidence
    rms_lower, rms_upper = np.quantile(rms_bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    brier_lower, brier_upper = np.quantile(
        brier_bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    equivalent = bool(
        rms_upper < pip_rms_margin
        and brier_lower >= -skeleton_brier_margin
        and brier_upper <= skeleton_brier_margin
    )
    return RankLayoutConsistencyResult(
        reference_layout=reference_layout,
        comparison_layout=comparison_layout,
        dataset_count=len(dataset_ids),
        minimum_seed_count_per_layout=int(observed_minimum_seed_count),
        global_pip_rms_difference=float(np.sqrt(np.mean(mse_values))),
        global_pip_rms_ci_lower=float(rms_lower),
        global_pip_rms_ci_upper=float(rms_upper),
        skeleton_brier_difference=float(np.mean(brier_values)),
        skeleton_brier_ci_lower=float(brier_lower),
        skeleton_brier_ci_upper=float(brier_upper),
        pip_rms_margin=float(pip_rms_margin),
        skeleton_brier_margin=float(skeleton_brier_margin),
        confidence=float(confidence),
        bootstrap_resamples=int(resamples),
        equivalent=equivalent,
        paired_seed_set_sha256=hashlib.sha256(
            json.dumps(
                paired_seed_sets,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )


def assess_internal_mcse_calibration(
    probability_runs_by_dataset: Mapping[
        str, Mapping[str, Mapping[Edge, float]]
    ],
    mcse_runs_by_dataset: Mapping[str, Mapping[str, Mapping[Edge, float]]],
    reference_pips_by_dataset: Mapping[str, Mapping[Edge, float]],
    truth_edges_by_dataset: Mapping[str, Sequence[Edge]],
    nodes_by_dataset: Mapping[str, Sequence[str]],
    *,
    target_label: str,
    support_ids_by_dataset: Mapping[str, str],
    pip_stratum_boundaries: Sequence[float] = (0.0, 0.1, 0.5, 0.9, 1.0),
) -> MCSECalibrationResult:
    """Compare internal MCSE with empirical across-seed variability for E10.

    Ratios and interval coverage are reduced to dataset/truth/PIP-stratum cells
    before any aggregate is formed. Individual edges are therefore never
    treated as independent dataset replicates.
    """

    dataset_ids = tuple(sorted(probability_runs_by_dataset))
    if not dataset_ids or any(
        set(mapping) != set(dataset_ids)
        for mapping in (
            mcse_runs_by_dataset,
            reference_pips_by_dataset,
            truth_edges_by_dataset,
            nodes_by_dataset,
            support_ids_by_dataset,
        )
    ):
        raise ValueError("all MCSE inputs must cover the same non-empty dataset IDs")
    if target_label not in {"order", "structure", "native_external"}:
        raise ValueError("target_label must identify the audited posterior target")
    if any(
        not isinstance(support_ids_by_dataset[dataset_id], str)
        or not support_ids_by_dataset[dataset_id]
        for dataset_id in dataset_ids
    ):
        raise ValueError("every dataset requires a non-empty support identifier")
    boundaries = tuple(float(value) for value in pip_stratum_boundaries)
    if (
        len(boundaries) < 2
        or boundaries[0] != 0.0
        or boundaries[-1] != 1.0
        or any(not np.isfinite(value) for value in boundaries)
        or any(right <= left for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("PIP stratum boundaries must increase strictly from 0 to 1")

    cells: list[MCSECalibrationCell] = []
    minimum_seed_count = 10**18
    paired_seed_sets: dict[str, list[str]] = {}
    normal_975 = 1.959963984540054
    for dataset_id in dataset_ids:
        probability_runs = probability_runs_by_dataset[dataset_id]
        mcse_runs = mcse_runs_by_dataset[dataset_id]
        if not isinstance(probability_runs, Mapping) or not isinstance(mcse_runs, Mapping):
            raise ValueError("probability and MCSE runs must be keyed by seed digest")
        if set(probability_runs) != set(mcse_runs) or len(probability_runs) < 2:
            raise ValueError("each dataset needs at least two paired probability/MCSE seeds")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", seed_id)
            for seed_id in probability_runs
        ):
            raise ValueError("method-seed IDs must be lowercase SHA-256 digests")
        seed_ids = tuple(sorted(probability_runs))
        paired_seed_sets[dataset_id] = list(seed_ids)
        minimum_seed_count = min(minimum_seed_count, len(probability_runs))
        nodes = tuple(nodes_by_dataset[dataset_id])
        if len(nodes) < 2 or len(set(nodes)) != len(nodes):
            raise ValueError(f"dataset {dataset_id!r} has invalid nodes")
        edges = tuple(
            (parent, child)
            for parent in nodes
            for child in nodes
            if parent != child
        )
        if set(reference_pips_by_dataset[dataset_id]) != set(edges):
            raise ValueError("reference PIPs must explicitly report every directed edge")
        if any(
            set(probability_runs[seed_id]) != set(edges)
            or set(mcse_runs[seed_id]) != set(edges)
            for seed_id in seed_ids
        ):
            raise ValueError(
                "every probability/MCSE run must explicitly report every directed edge"
            )
        probability_matrix = np.asarray(
            [
                [float(probability_runs[seed_id][edge]) for edge in edges]
                for seed_id in seed_ids
            ],
            dtype=float,
        )
        mcse_matrix = np.asarray(
            [
                [float(mcse_runs[seed_id][edge]) for edge in edges]
                for seed_id in seed_ids
            ],
            dtype=float,
        )
        reference = np.asarray(
            [float(reference_pips_by_dataset[dataset_id][edge]) for edge in edges],
            dtype=float,
        )
        if (
            np.any(~np.isfinite(probability_matrix))
            or np.any((probability_matrix < 0.0) | (probability_matrix > 1.0))
            or np.any(~np.isfinite(mcse_matrix))
            or np.any(mcse_matrix < 0.0)
            or np.any(~np.isfinite(reference))
            or np.any((reference < 0.0) | (reference > 1.0))
        ):
            raise ValueError("probabilities/reference must be in [0,1] and MCSE non-negative")
        empirical_sd = np.std(probability_matrix, axis=0, ddof=1)
        internal_mcse = np.median(mcse_matrix, axis=0)
        truth = set(map(tuple, truth_edges_by_dataset[dataset_id]))
        if not truth <= set(edges):
            raise ValueError(f"dataset {dataset_id!r} truth contains an invalid edge")

        for truth_status, truth_mask in (
            ("true_edge", np.asarray([edge in truth for edge in edges], dtype=bool)),
            ("nonedge", np.asarray([edge not in truth for edge in edges], dtype=bool)),
        ):
            for lower, upper in zip(boundaries, boundaries[1:]):
                if upper == 1.0:
                    pip_mask = (reference >= lower) & (reference <= upper)
                    label = f"[{lower:g},{upper:g}]"
                else:
                    pip_mask = (reference >= lower) & (reference < upper)
                    label = f"[{lower:g},{upper:g})"
                selected = truth_mask & pip_mask
                edge_count = int(np.count_nonzero(selected))
                if edge_count == 0:
                    continue
                selected_sd = empirical_sd[selected]
                selected_internal = internal_mcse[selected]
                positive_zero_sd = (selected_sd == 0.0) & (selected_internal > 0.0)
                ratios = np.empty(edge_count, dtype=float)
                nonzero_sd = selected_sd > 0.0
                ratios[nonzero_sd] = selected_internal[nonzero_sd] / selected_sd[nonzero_sd]
                ratios[(selected_sd == 0.0) & (selected_internal == 0.0)] = 1.0
                ratios[positive_zero_sd] = np.inf
                lower_interval = (
                    probability_matrix[:, selected]
                    - normal_975 * mcse_matrix[:, selected]
                )
                upper_interval = (
                    probability_matrix[:, selected]
                    + normal_975 * mcse_matrix[:, selected]
                )
                selected_reference = reference[selected][None, :]
                coverage = float(
                    np.mean(
                        (lower_interval <= selected_reference)
                        & (selected_reference <= upper_interval)
                    )
                )
                cells.append(
                    MCSECalibrationCell(
                        dataset_id=dataset_id,
                        truth_status=truth_status,
                        pip_stratum=label,
                        edge_count=edge_count,
                        ratio_edge_count=edge_count,
                        seed_count=len(probability_runs),
                        median_mcse_to_empirical_sd_ratio=(
                            None
                            if np.any(positive_zero_sd)
                            else float(np.median(ratios))
                        ),
                        p90_mcse_to_empirical_sd_ratio=(
                            None
                            if np.any(positive_zero_sd)
                            else float(np.quantile(ratios, 0.90))
                        ),
                        nominal_95_coverage=coverage,
                        positive_mcse_zero_sd_count=int(np.count_nonzero(positive_zero_sd)),
                        ratio_status=(
                            "infinite_positive_mcse_zero_empirical_sd"
                            if np.any(positive_zero_sd)
                            else "finite"
                        ),
                    )
                )

    if not cells:
        raise ValueError("MCSE calibration has no non-degenerate dataset/stratum cells")
    has_infinite_ratio = any(cell.ratio_status != "finite" for cell in cells)
    dataset_medians: list[float] = []
    dataset_p90s: list[float] = []
    dataset_coverages: list[float] = []
    for dataset_id in dataset_ids:
        dataset_cells = [cell for cell in cells if cell.dataset_id == dataset_id]
        dataset_coverages.append(
            float(np.mean([cell.nominal_95_coverage for cell in dataset_cells]))
        )
        if not has_infinite_ratio:
            dataset_medians.append(
                float(
                    np.median(
                        [cell.median_mcse_to_empirical_sd_ratio for cell in dataset_cells]
                    )
                )
            )
            dataset_p90s.append(
                float(
                    np.quantile(
                        [cell.p90_mcse_to_empirical_sd_ratio for cell in dataset_cells],
                        0.90,
                    )
                )
            )
    return MCSECalibrationResult(
        cells=tuple(cells),
        dataset_count=len(dataset_ids),
        minimum_seed_count=int(minimum_seed_count),
        median_cell_ratio=(None if has_infinite_ratio else float(np.mean(dataset_medians))),
        p90_cell_ratio=(None if has_infinite_ratio else float(np.mean(dataset_p90s))),
        mean_cell_nominal_95_coverage=float(np.mean(dataset_coverages)),
        target_label=target_label,
        support_ids_by_dataset=tuple(
            (dataset_id, support_ids_by_dataset[dataset_id])
            for dataset_id in dataset_ids
        ),
        paired_seed_set_sha256=hashlib.sha256(
            json.dumps(
                paired_seed_sets,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        calibration_status=(
            "infinite_ratio_positive_mcse_zero_empirical_sd"
            if has_infinite_ratio
            else "finite"
        ),
    )
