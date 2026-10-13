"""Dataset-weighted reliability diagrams and dataset-level bootstrap bands.

The statistical unit is the complete graph--dataset pair. An explicit node set
binds each dataset to all ``d(d-1)/2`` unordered pairs; incomplete/top-only and
out-of-universe tables are rejected. Every dataset has total weight one,
divided equally among its unordered pairs. Bootstrap draws
resample whole datasets within strata; edges are never resampled independently.

Initial equal-mass bins are weighted-quantile intervals after sorting by
``(PIP, dataset_id, edge_id)``. Low-effective-count bins are merged only with
an adjacent bin, choosing the closest mean PIP and the lower-PIP neighbour on a
tie. Bootstrap draws recompute the initial weighted-quantile boundaries and
summaries. The point estimate's contiguous merge plan is then applied so that
confidence bands have an unambiguous bin-by-bin interpretation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
import numbers
from typing import Hashable, Mapping, Sequence

import numpy as np


UnorderedEdge = tuple[str, str]
EdgeTable = Mapping[UnorderedEdge, float]
TruthTable = Mapping[UnorderedEdge, int | bool]
NodeSets = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class ReliabilityBin:
    """One final, possibly merged, reliability bin."""

    index: int
    initial_bin_indices: tuple[int, ...]
    edge_count: int
    total_weight: float
    mean_probability: float
    observed_frequency: float
    effective_dataset_count: float
    contributing_datasets: int
    probability_min: float
    probability_max: float


@dataclass(frozen=True)
class ReliabilityMerge:
    """Audit record for one deterministic adjacent-bin merge."""

    deficient_initial_bins: tuple[int, ...]
    neighbour_initial_bins: tuple[int, ...]
    result_initial_bins: tuple[int, ...]
    deficient_effective_dataset_count: float


@dataclass(frozen=True)
class ReliabilityResult:
    """Point reliability curve with its complete weighting/merge provenance."""

    bins: tuple[ReliabilityBin, ...]
    merges: tuple[ReliabilityMerge, ...]
    requested_bins: int
    dataset_count: int
    observation_count: int
    minimum_effective_datasets: float
    node_counts: tuple[tuple[str, int], ...]
    expected_pair_counts: tuple[tuple[str, int], ...]
    pair_universe_validation: str = "complete unordered-pair table induced by explicit node set"
    weighting: str = "each dataset has total weight one; equal weight per unordered edge"
    edge_order: str = "ascending (PIP, dataset_id, edge_id)"


@dataclass(frozen=True)
class ReliabilityBand:
    """Pointwise dataset-bootstrap interval for one displayed bin."""

    index: int
    initial_bin_indices: tuple[int, ...]
    mean_probability_lower: float
    mean_probability_upper: float
    observed_frequency_lower: float
    observed_frequency_upper: float
    effective_dataset_count_lower: float
    effective_dataset_count_upper: float


@dataclass(frozen=True)
class ReliabilityBootstrapResult:
    """Reliability point estimate and whole-dataset stratified bootstrap bands."""

    point: ReliabilityResult
    bands: tuple[ReliabilityBand, ...]
    resamples: int
    confidence: float
    seed: int
    stratum_count: int
    bootstrap_unit: str = "complete graph-dataset pair"
    edge_resampling: bool = False
    bins_recomputed_each_resample: bool = True


@dataclass(frozen=True)
class _Observation:
    dataset_id: str
    unit_instance: int
    edge_id: UnorderedEdge
    probability: float
    outcome: int
    weight: float

    @property
    def unit_key(self) -> tuple[str, int]:
        return (self.dataset_id, self.unit_instance)

    @property
    def sort_key(self) -> tuple[float, str, str, str, int]:
        return (
            self.probability,
            self.dataset_id,
            self.edge_id[0],
            self.edge_id[1],
            self.unit_instance,
        )


def _canonical_edge(edge: object) -> UnorderedEdge:
    if not isinstance(edge, tuple) or len(edge) != 2:
        raise ValueError("unordered edge IDs must be two-string tuples")
    left, right = edge
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        raise ValueError("unordered edge endpoints must be non-empty strings")
    if left == right:
        raise ValueError("an unordered edge cannot be a self-edge")
    return (left, right) if left < right else (right, left)


def _canonical_probability_table(table: EdgeTable) -> dict[UnorderedEdge, float]:
    result: dict[UnorderedEdge, float] = {}
    for raw_edge, raw_probability in table.items():
        edge = _canonical_edge(raw_edge)
        if edge in result:
            raise ValueError(f"duplicate unordered edge after canonicalization: {edge}")
        if isinstance(raw_probability, (bool, np.bool_)) or not isinstance(
            raw_probability, numbers.Real
        ):
            raise ValueError(f"probability for {edge} must be a real number")
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability for {edge} must be finite and in [0, 1]")
        result[edge] = probability
    if not result:
        raise ValueError("each dataset must contain at least one unordered edge")
    return result


def _canonical_truth_table(table: TruthTable) -> dict[UnorderedEdge, int]:
    result: dict[UnorderedEdge, int] = {}
    for raw_edge, raw_outcome in table.items():
        edge = _canonical_edge(raw_edge)
        if edge in result:
            raise ValueError(f"duplicate unordered edge after canonicalization: {edge}")
        if not isinstance(raw_outcome, (bool, np.bool_, int, np.integer)) or int(
            raw_outcome
        ) not in (0, 1):
            raise ValueError(f"truth for {edge} must be binary")
        result[edge] = int(raw_outcome)
    if not result:
        raise ValueError("each dataset must contain at least one unordered edge")
    return result


def _canonical_node_set(raw_nodes: Sequence[str], *, dataset_id: str) -> tuple[str, ...]:
    if isinstance(raw_nodes, (str, bytes)):
        raise ValueError(f"node set for {dataset_id} must be a sequence of node IDs")
    try:
        nodes = tuple(raw_nodes)
    except TypeError as error:
        raise ValueError(f"node set for {dataset_id} must be a sequence of node IDs") from error
    if len(nodes) < 2:
        raise ValueError(f"node set for {dataset_id} must contain at least two nodes")
    if any(not isinstance(node, str) or not node for node in nodes):
        raise ValueError(f"node set for {dataset_id} must contain non-empty strings")
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"node set for {dataset_id} contains duplicate node IDs")
    return tuple(sorted(nodes))


def _expected_unordered_pairs(nodes: Sequence[str]) -> set[UnorderedEdge]:
    return set(combinations(nodes, 2))


def _prepare_dataset_observations(
    probabilities: Mapping[str, EdgeTable],
    outcomes: Mapping[str, TruthTable],
    nodes: NodeSets,
) -> tuple[dict[str, tuple[_Observation, ...]], dict[str, tuple[str, ...]]]:
    if not probabilities:
        raise ValueError("at least one dataset is required")
    if set(probabilities) != set(outcomes):
        raise ValueError("probability and truth tables must contain identical dataset IDs")
    if set(probabilities) != set(nodes):
        raise ValueError("node sets must contain exactly the reliability dataset IDs")
    prepared: dict[str, tuple[_Observation, ...]] = {}
    canonical_nodes: dict[str, tuple[str, ...]] = {}
    for dataset_id in sorted(probabilities):
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset IDs must be non-empty strings")
        dataset_nodes = _canonical_node_set(nodes[dataset_id], dataset_id=dataset_id)
        expected_pairs = _expected_unordered_pairs(dataset_nodes)
        probability_table = _canonical_probability_table(probabilities[dataset_id])
        truth_table = _canonical_truth_table(outcomes[dataset_id])
        if set(probability_table) != set(truth_table):
            raise ValueError(f"edge IDs differ between probabilities and truth for {dataset_id}")
        actual_pairs = set(probability_table)
        if actual_pairs != expected_pairs:
            missing = sorted(expected_pairs - actual_pairs)
            extraneous = sorted(actual_pairs - expected_pairs)
            raise ValueError(
                f"{dataset_id} must contain the complete unordered-pair table induced by "
                f"its explicit node set; missing={missing}, extraneous={extraneous}"
            )
        weight = 1.0 / len(probability_table)
        canonical_nodes[dataset_id] = dataset_nodes
        prepared[dataset_id] = tuple(
            _Observation(
                dataset_id=dataset_id,
                unit_instance=0,
                edge_id=edge,
                probability=probability_table[edge],
                outcome=truth_table[edge],
                weight=weight,
            )
            for edge in sorted(probability_table)
        )
    return prepared, canonical_nodes


def _validate_bin_settings(bins: int, minimum_effective_datasets: float) -> None:
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, numbers.Integral):
        raise ValueError("bins must be a positive integer")
    if int(bins) < 1:
        raise ValueError("bins must be a positive integer")
    if not isinstance(minimum_effective_datasets, numbers.Real):
        raise ValueError("minimum_effective_datasets must be finite and non-negative")
    if not math.isfinite(float(minimum_effective_datasets)) or float(
        minimum_effective_datasets
    ) < 0.0:
        raise ValueError("minimum_effective_datasets must be finite and non-negative")


def _weighted_quantile_groups(
    observations: Sequence[_Observation], bins: int
) -> list[list[_Observation]]:
    """Split sorted records at weighted quantiles without splitting an edge."""

    ordered = sorted(observations, key=lambda observation: observation.sort_key)
    if bins > len(ordered):
        raise ValueError("bins cannot exceed the number of unordered-edge observations")
    cumulative = np.cumsum([observation.weight for observation in ordered], dtype=float)
    total_weight = float(cumulative[-1])
    cuts = [0]
    for boundary in range(1, bins):
        target = boundary * total_weight / bins
        minimum_cut = cuts[-1] + 1
        maximum_cut = len(ordered) - (bins - boundary)
        crossing = int(np.searchsorted(cumulative, target, side="left")) + 1
        candidates = {
            min(max(crossing, minimum_cut), maximum_cut),
            min(max(crossing - 1, minimum_cut), maximum_cut),
        }
        cut = min(
            candidates,
            key=lambda candidate: (
                abs(float(cumulative[candidate - 1]) - target),
                candidate,
            ),
        )
        cuts.append(cut)
    cuts.append(len(ordered))
    return [ordered[left:right] for left, right in zip(cuts[:-1], cuts[1:])]


def _summarize_group(
    observations: Sequence[_Observation],
    *,
    index: int,
    initial_bin_indices: tuple[int, ...],
) -> ReliabilityBin:
    weights = np.asarray([observation.weight for observation in observations], dtype=float)
    probabilities = np.asarray(
        [observation.probability for observation in observations], dtype=float
    )
    outcomes = np.asarray([observation.outcome for observation in observations], dtype=float)
    total_weight = float(np.sum(weights))
    dataset_weights: dict[tuple[str, int], float] = defaultdict(float)
    for observation in observations:
        dataset_weights[observation.unit_key] += observation.weight
    mass = np.asarray(list(dataset_weights.values()), dtype=float)
    effective = float(total_weight * total_weight / np.sum(mass * mass))
    return ReliabilityBin(
        index=index,
        initial_bin_indices=initial_bin_indices,
        edge_count=len(observations),
        total_weight=total_weight,
        mean_probability=float(np.dot(weights, probabilities) / total_weight),
        observed_frequency=float(np.dot(weights, outcomes) / total_weight),
        effective_dataset_count=effective,
        contributing_datasets=len(dataset_weights),
        probability_min=float(np.min(probabilities)),
        probability_max=float(np.max(probabilities)),
    )


def _merge_low_effective_bins(
    initial_groups: Sequence[Sequence[_Observation]],
    minimum_effective_datasets: float,
) -> tuple[list[tuple[tuple[int, ...], list[_Observation]]], tuple[ReliabilityMerge, ...]]:
    groups = [((index,), list(observations)) for index, observations in enumerate(initial_groups)]
    merges: list[ReliabilityMerge] = []
    while len(groups) > 1:
        summaries = [
            _summarize_group(observations, index=index, initial_bin_indices=indices)
            for index, (indices, observations) in enumerate(groups)
        ]
        deficient = next(
            (
                index
                for index, summary in enumerate(summaries)
                if summary.effective_dataset_count < minimum_effective_datasets
            ),
            None,
        )
        if deficient is None:
            break
        neighbours = [
            index
            for index in (deficient - 1, deficient + 1)
            if 0 <= index < len(groups)
        ]
        source = summaries[deficient]
        neighbour = min(
            neighbours,
            key=lambda index: (
                abs(summaries[index].mean_probability - source.mean_probability),
                0 if index < deficient else 1,
            ),
        )
        source_indices, source_observations = groups[deficient]
        neighbour_indices, neighbour_observations = groups[neighbour]
        result_indices = tuple(sorted(source_indices + neighbour_indices))
        result_observations = sorted(
            source_observations + neighbour_observations,
            key=lambda observation: observation.sort_key,
        )
        merges.append(
            ReliabilityMerge(
                deficient_initial_bins=source_indices,
                neighbour_initial_bins=neighbour_indices,
                result_initial_bins=result_indices,
                deficient_effective_dataset_count=source.effective_dataset_count,
            )
        )
        lower = min(deficient, neighbour)
        upper = max(deficient, neighbour)
        groups[lower : upper + 1] = [(result_indices, result_observations)]
    if groups:
        final_summaries = [
            _summarize_group(observations, index=index, initial_bin_indices=indices)
            for index, (indices, observations) in enumerate(groups)
        ]
        deficient_final = [
            summary
            for summary in final_summaries
            if summary.effective_dataset_count < minimum_effective_datasets
        ]
        if deficient_final:
            minimum_observed = min(
                summary.effective_dataset_count for summary in deficient_final
            )
            raise ValueError(
                "minimum_effective_datasets cannot be met even after all possible "
                f"adjacent-bin merges: requested={minimum_effective_datasets:.17g}, "
                f"final_effective_dataset_count={minimum_observed:.17g}"
            )
    return groups, tuple(merges)


def _point_from_prepared(
    prepared: Mapping[str, Sequence[_Observation]],
    node_sets: Mapping[str, Sequence[str]],
    *,
    bins: int,
    minimum_effective_datasets: float,
) -> ReliabilityResult:
    observations = [
        observation
        for dataset_id in sorted(prepared)
        for observation in prepared[dataset_id]
    ]
    initial_groups = _weighted_quantile_groups(observations, bins)
    groups, merges = _merge_low_effective_bins(
        initial_groups, minimum_effective_datasets
    )
    summaries = tuple(
        _summarize_group(observations, index=index, initial_bin_indices=indices)
        for index, (indices, observations) in enumerate(groups)
    )
    return ReliabilityResult(
        bins=summaries,
        merges=merges,
        requested_bins=int(bins),
        dataset_count=len(prepared),
        observation_count=len(observations),
        minimum_effective_datasets=float(minimum_effective_datasets),
        node_counts=tuple(
            (dataset_id, len(node_sets[dataset_id])) for dataset_id in sorted(node_sets)
        ),
        expected_pair_counts=tuple(
            (dataset_id, len(node_sets[dataset_id]) * (len(node_sets[dataset_id]) - 1) // 2)
            for dataset_id in sorted(node_sets)
        ),
    )


def fit_reliability(
    probabilities: Mapping[str, EdgeTable],
    outcomes: Mapping[str, TruthTable],
    *,
    nodes: NodeSets,
    bins: int = 10,
    minimum_effective_datasets: float = 10.0,
) -> ReliabilityResult:
    """Fit dataset-weighted equal-mass reliability bins.

    ``nodes`` is required and binds each dataset to its complete unordered-pair
    universe. Probability and truth tables must contain every induced pair
    exactly once and no other pair; top-edge-only inputs are rejected. For
    skeleton PIPs each probability is normally ``P(i->j) + P(j->i)``. Each
    dataset contributes total mass one, so larger node sets cannot dominate.
    """

    _validate_bin_settings(bins, minimum_effective_datasets)
    prepared, canonical_nodes = _prepare_dataset_observations(
        probabilities, outcomes, nodes
    )
    observation_count = sum(len(observations) for observations in prepared.values())
    if int(bins) > observation_count:
        raise ValueError("bins cannot exceed the number of unordered-edge observations")
    return _point_from_prepared(
        prepared,
        canonical_nodes,
        bins=int(bins),
        minimum_effective_datasets=float(minimum_effective_datasets),
    )


def _validate_strata(
    dataset_ids: Sequence[str], strata: Mapping[str, Hashable] | None
) -> dict[str, str]:
    if strata is None:
        return {dataset_id: "all" for dataset_id in dataset_ids}
    if set(strata) != set(dataset_ids):
        raise ValueError("strata must contain exactly the reliability dataset IDs")
    normalized: dict[str, str] = {}
    for dataset_id in dataset_ids:
        label = strata[dataset_id]
        if not isinstance(label, (str, int, float, bool, tuple)):
            raise ValueError("stratum labels must have a deterministic scalar/tuple representation")
        normalized[dataset_id] = repr(label)
    return normalized


def bootstrap_reliability(
    probabilities: Mapping[str, EdgeTable],
    outcomes: Mapping[str, TruthTable],
    *,
    nodes: NodeSets,
    strata: Mapping[str, Hashable] | None = None,
    bins: int = 10,
    minimum_effective_datasets: float = 10.0,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ReliabilityBootstrapResult:
    """Compute stratified whole-dataset bootstrap reliability bands.

    The bootstrap unit is one complete graph--dataset pair. Sampling is with
    replacement within each supplied stratum and preserves that stratum's
    dataset count. All unordered edges belonging to a selected dataset travel
    together; there is deliberately no edge-level resampling. Weighted-quantile
    bin boundaries and bin summaries are recomputed for every bootstrap draw.
    Complete-pair validation against ``nodes`` occurs before any resampling.
    """

    _validate_bin_settings(bins, minimum_effective_datasets)
    if isinstance(resamples, (bool, np.bool_)) or not isinstance(
        resamples, numbers.Integral
    ) or int(resamples) < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not isinstance(confidence, numbers.Real) or not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or int(
        seed
    ) < 0:
        raise ValueError("seed must be a non-negative integer")

    prepared, canonical_nodes = _prepare_dataset_observations(
        probabilities, outcomes, nodes
    )
    dataset_ids = tuple(sorted(prepared))
    observation_count = sum(len(observations) for observations in prepared.values())
    if int(bins) > observation_count:
        raise ValueError("bins cannot exceed the number of unordered-edge observations")
    minimum_bootstrap_observations = len(dataset_ids) * min(
        len(observations) for observations in prepared.values()
    )
    if int(bins) > minimum_bootstrap_observations:
        raise ValueError(
            "too many bins for all possible whole-dataset bootstrap samples"
        )
    point = _point_from_prepared(
        prepared,
        canonical_nodes,
        bins=int(bins),
        minimum_effective_datasets=float(minimum_effective_datasets),
    )
    labels = _validate_strata(dataset_ids, strata)
    stratum_names = sorted(set(labels.values()))
    stratum_members = {
        label: tuple(dataset_id for dataset_id in dataset_ids if labels[dataset_id] == label)
        for label in stratum_names
    }
    merge_plan = tuple(reliability_bin.initial_bin_indices for reliability_bin in point.bins)
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    mean_probability = np.empty((int(resamples), len(merge_plan)), dtype=float)
    observed_frequency = np.empty_like(mean_probability)
    effective_count = np.empty_like(mean_probability)

    for draw in range(int(resamples)):
        sampled_observations: list[_Observation] = []
        unit_instance = 0
        for label in stratum_names:
            members = stratum_members[label]
            sampled_indices = rng.integers(0, len(members), size=len(members))
            for sampled_index in sampled_indices:
                dataset_id = members[int(sampled_index)]
                sampled_observations.extend(
                    _Observation(
                        dataset_id=observation.dataset_id,
                        unit_instance=unit_instance,
                        edge_id=observation.edge_id,
                        probability=observation.probability,
                        outcome=observation.outcome,
                        weight=observation.weight,
                    )
                    for observation in prepared[dataset_id]
                )
                unit_instance += 1
        initial_groups = _weighted_quantile_groups(sampled_observations, int(bins))
        for bin_index, initial_indices in enumerate(merge_plan):
            grouped = [
                observation
                for initial_index in initial_indices
                for observation in initial_groups[initial_index]
            ]
            summary = _summarize_group(
                grouped,
                index=bin_index,
                initial_bin_indices=initial_indices,
            )
            mean_probability[draw, bin_index] = summary.mean_probability
            observed_frequency[draw, bin_index] = summary.observed_frequency
            effective_count[draw, bin_index] = summary.effective_dataset_count

    tail = (1.0 - float(confidence)) / 2.0
    quantiles = (tail, 1.0 - tail)
    bands = tuple(
        ReliabilityBand(
            index=index,
            initial_bin_indices=initial_indices,
            mean_probability_lower=float(np.quantile(mean_probability[:, index], quantiles[0])),
            mean_probability_upper=float(np.quantile(mean_probability[:, index], quantiles[1])),
            observed_frequency_lower=float(
                np.quantile(observed_frequency[:, index], quantiles[0])
            ),
            observed_frequency_upper=float(
                np.quantile(observed_frequency[:, index], quantiles[1])
            ),
            effective_dataset_count_lower=float(
                np.quantile(effective_count[:, index], quantiles[0])
            ),
            effective_dataset_count_upper=float(
                np.quantile(effective_count[:, index], quantiles[1])
            ),
        )
        for index, initial_indices in enumerate(merge_plan)
    )
    return ReliabilityBootstrapResult(
        point=point,
        bands=bands,
        resamples=int(resamples),
        confidence=float(confidence),
        seed=int(seed),
        stratum_count=len(stratum_names),
    )
