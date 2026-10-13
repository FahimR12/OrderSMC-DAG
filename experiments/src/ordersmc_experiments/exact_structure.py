r"""Exact subset DP for the candidate-restricted structure-modular target.

For a node set ``S``, let ``F(S)`` be the total modular weight of DAGs on
``S`` and let

``A_i(U) = sum_{P in C_i, P subset U} w_i(P)``.

Every nonempty DAG has at least one sink. Inclusion--exclusion over the events
"node i is a sink" gives the Robinson sink-set recurrence

``F(S) = sum_{empty != T subset S} (-1)^(|T|+1) F(S\T) prod_{i in T} A_i(S\T)``.

It counts a DAG once regardless of its number of topological orders and hence
targets a structure-modular posterior. Candidate parent sets absent from the
table have exactly zero mass.

Local log weights are shifted by one constant per child before exponentiation;
the product of those constants is restored only in the final log normalizer.
The alternating recurrence uses extended-precision Neumaier summation. Edge
normalizers are derivatives with respect to an edge-specific log multiplier,
computed by a reverse pass through the same recurrence and a superset zeta
transform. This provides every directed-edge PIP without rerunning the
``O(3^d)`` recurrence for every edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numbers
from typing import Dict, Iterable

import numpy as np

from ordersmc.core import (
    GraphSizeDistribution,
    OrderCandidateTable,
    _validate_candidate_table,
    candidate_max_indegree,
)
from ordersmc.persistence import candidate_table_content_sha256


STRUCTURE_TARGET = "candidate_restricted_structure_modular"


@dataclass(frozen=True)
class ExactStructureDPResult:
    edge_probs: Dict[tuple[str, str], float]
    graph_size_probs: GraphSizeDistribution | None
    log_normalizer: float
    log_prior_normalizer: float
    log_evidence_ratio: float
    beta: float
    node_count: int
    subset_count: int
    recurrence_term_count: int
    target: str = STRUCTURE_TARGET
    target_label: str = "structure"
    target_semantics: str = STRUCTURE_TARGET
    arithmetic: str = "per-child log scaling and numpy.longdouble Neumaier summation"


def _family_masks(table: OrderCandidateTable) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for child_idx, parent_sets in enumerate(table.parent_sets):
        child_masks = np.zeros(parent_sets.shape[0], dtype=np.uint64)
        for family_idx, row in enumerate(parent_sets):
            mask = 0
            for raw_parent in row:
                parent = int(raw_parent)
                if parent < 0:
                    continue
                if parent == child_idx:
                    raise ValueError("candidate table contains a self parent")
                mask |= 1 << parent
            child_masks[family_idx] = np.uint64(mask)
        masks.append(child_masks)
    return masks


def _neumaier_sum(values: Iterable[np.longdouble]) -> np.longdouble:
    """Compensated extended-precision sum, including severe sign cancellation."""
    total = np.longdouble(0.0)
    correction = np.longdouble(0.0)
    for raw_value in values:
        value = np.longdouble(raw_value)
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    return total + correction


def _nonempty_subsets(mask: int):
    subset = mask
    while subset:
        yield subset
        subset = (subset - 1) & mask


def _available_family_weights(
    table: OrderCandidateTable,
    beta: float,
    family_masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], np.longdouble]:
    """Return zeta-transformed normalized local sums and exact-family weights."""
    node_count = len(table.nodes)
    subset_count = 1 << node_count
    available = np.zeros((node_count, subset_count), dtype=np.longdouble)
    exact_weights: list[np.ndarray] = []
    log_scale_total = np.longdouble(0.0)

    for child in range(node_count):
        logweights = (
            np.asarray(table.logprior[child], dtype=float)
            + float(beta) * np.asarray(table.loglik[child], dtype=float)
        )
        scale = float(np.max(logweights))
        normalized = np.exp(
            np.asarray(logweights - scale, dtype=np.longdouble), dtype=np.longdouble
        )
        if np.any(normalized == 0.0):
            raise FloatingPointError(
                "local family-weight range exceeds numpy.longdouble exponent range"
            )
        exact_weights.append(normalized)
        log_scale_total += np.longdouble(scale)
        for family_mask, weight in zip(family_masks[child], normalized):
            available[child, int(family_mask)] += weight

        # Subset zeta transform: after all bits, A_i(U) is the sum of exact
        # family weights whose parent mask is a subset of U.
        for bit_index in range(node_count):
            bit = 1 << bit_index
            for parent_pool in range(subset_count):
                if parent_pool & bit:
                    available[child, parent_pool] += available[child, parent_pool ^ bit]

    return available, exact_weights, log_scale_total


def _sink_dp(
    table: OrderCandidateTable,
    beta: float,
    *,
    return_edge_probs: bool,
) -> tuple[float, Dict[tuple[str, str], float]]:
    node_count = len(table.nodes)
    subset_count = 1 << node_count
    full_mask = subset_count - 1
    family_masks = _family_masks(table)
    available, exact_weights, log_scale_total = _available_family_weights(
        table, beta, family_masks
    )

    total_weight = np.zeros(subset_count, dtype=np.longdouble)
    total_weight[0] = np.longdouble(1.0)
    for nodes in range(1, subset_count):
        terms: list[np.longdouble] = []
        for sinks in _nonempty_subsets(nodes):
            remainder = nodes ^ sinks
            local_product = np.longdouble(1.0)
            bits = sinks
            while bits:
                sink_bit = bits & -bits
                sink = sink_bit.bit_length() - 1
                local_product *= available[sink, remainder]
                bits ^= sink_bit
            sign = 1.0 if sinks.bit_count() % 2 == 1 else -1.0
            terms.append(np.longdouble(sign) * total_weight[remainder] * local_product)
        value = _neumaier_sum(terms)
        if not np.isfinite(value) or value <= 0.0:
            raise FloatingPointError(
                "sink-set inclusion-exclusion lost positivity; reduce max_nodes or use higher precision"
            )
        total_weight[nodes] = value

    scaled_normalizer = total_weight[full_mask]
    log_normalizer = float(np.log(scaled_normalizer) + log_scale_total)
    if not return_edge_probs:
        return log_normalizer, {}

    # Reverse-mode derivative of F(V) with respect to every A_i(U).
    total_adjoint = np.zeros(subset_count, dtype=np.longdouble)
    available_adjoint = np.zeros_like(available)
    total_adjoint[full_mask] = np.longdouble(1.0)
    for nodes in range(full_mask, 0, -1):
        output_adjoint = total_adjoint[nodes]
        if output_adjoint == 0.0:
            continue
        for sinks in _nonempty_subsets(nodes):
            remainder = nodes ^ sinks
            sink_indices: list[int] = []
            bits = sinks
            while bits:
                sink_bit = bits & -bits
                sink_indices.append(sink_bit.bit_length() - 1)
                bits ^= sink_bit
            local_values = [available[sink, remainder] for sink in sink_indices]
            prefix = [np.longdouble(1.0)]
            for value in local_values:
                prefix.append(prefix[-1] * value)
            suffix = [np.longdouble(1.0)] * (len(local_values) + 1)
            for index in range(len(local_values) - 1, -1, -1):
                suffix[index] = suffix[index + 1] * local_values[index]
            sign = np.longdouble(1.0 if len(sink_indices) % 2 == 1 else -1.0)
            signed_adjoint = output_adjoint * sign
            total_adjoint[remainder] += signed_adjoint * prefix[-1]
            for index, sink in enumerate(sink_indices):
                product_without_sink = prefix[index] * suffix[index + 1]
                available_adjoint[sink, remainder] += (
                    signed_adjoint * total_weight[remainder] * product_without_sink
                )

    edge_numerators = np.zeros((node_count, node_count), dtype=np.longdouble)
    for child in range(node_count):
        # If A_i(U)=sum_{P subset U} w_i(P), then dZ/dw_i(P) is the
        # superset sum of dZ/dA_i(U) over U containing P.
        family_sensitivity = available_adjoint[child].copy()
        for bit_index in range(node_count):
            bit = 1 << bit_index
            for parent_mask in range(subset_count):
                if not parent_mask & bit:
                    family_sensitivity[parent_mask] += family_sensitivity[parent_mask | bit]
        for family_mask_raw, weight in zip(family_masks[child], exact_weights[child]):
            family_mask = int(family_mask_raw)
            contribution = weight * family_sensitivity[family_mask]
            bits = family_mask
            while bits:
                parent_bit = bits & -bits
                parent = parent_bit.bit_length() - 1
                edge_numerators[parent, child] += contribution
                bits ^= parent_bit

    recurrence_terms = 3**node_count - 2**node_count
    numerical_tolerance = (
        np.longdouble(256.0)
        * np.finfo(np.longdouble).eps
        * np.longdouble(max(1, recurrence_terms))
        * max(np.longdouble(1.0), abs(scaled_normalizer))
    )
    edge_probs: Dict[tuple[str, str], float] = {}
    for parent in range(node_count):
        for child in range(node_count):
            if parent == child:
                continue
            numerator = edge_numerators[parent, child]
            if numerator < 0.0 and abs(numerator) <= numerical_tolerance:
                numerator = np.longdouble(0.0)
            if numerator > scaled_normalizer and numerator - scaled_normalizer <= numerical_tolerance:
                numerator = scaled_normalizer
            if not np.isfinite(numerator) or numerator < 0.0 or numerator > scaled_normalizer:
                raise FloatingPointError("edge marginal left the probability range during reverse DP")
            edge_probs[(table.nodes[parent], table.nodes[child])] = float(
                numerator / scaled_normalizer
            )
    return log_normalizer, edge_probs


def _sink_graph_size_distribution(
    table: OrderCandidateTable,
    beta: float,
) -> tuple[float, Dict[int, float]]:
    """Run the sink-set recurrence over edge-count feature polynomials."""

    node_count = len(table.nodes)
    subset_count = 1 << node_count
    full_mask = subset_count - 1
    family_masks = _family_masks(table)
    _, exact_weights, log_scale_total = _available_family_weights(
        table, beta, family_masks
    )
    family_sizes = [
        np.asarray([int(mask).bit_count() for mask in child_masks], dtype=np.int32)
        for child_masks in family_masks
    ]
    child_capacities = [int(np.max(sizes, initial=0)) for sizes in family_sizes]
    edge_capacity = int(sum(child_capacities))
    max_child_capacity = max(child_capacities, default=0)

    available_by_size = np.zeros(
        (node_count, subset_count, max_child_capacity + 1), dtype=np.longdouble
    )
    for child in range(node_count):
        for family_mask, family_size, weight in zip(
            family_masks[child], family_sizes[child], exact_weights[child]
        ):
            available_by_size[child, int(family_mask), int(family_size)] += weight
        for bit_index in range(node_count):
            bit = 1 << bit_index
            for parent_pool in range(subset_count):
                if parent_pool & bit:
                    available_by_size[child, parent_pool] += available_by_size[
                        child, parent_pool ^ bit
                    ]

    subset_capacities = np.zeros(subset_count, dtype=np.int32)
    for subset in range(1, subset_count):
        bit = subset & -subset
        child = bit.bit_length() - 1
        subset_capacities[subset] = (
            subset_capacities[subset ^ bit] + child_capacities[child]
        )

    total_polynomial = np.zeros(
        (subset_count, edge_capacity + 1), dtype=np.longdouble
    )
    total_polynomial[0, 0] = np.longdouble(1.0)
    for nodes in range(1, subset_count):
        capacity = int(subset_capacities[nodes])
        total = np.zeros(capacity + 1, dtype=np.longdouble)
        correction = np.zeros(capacity + 1, dtype=np.longdouble)
        absolute_term_sum = np.zeros(capacity + 1, dtype=np.longdouble)
        for sinks in _nonempty_subsets(nodes):
            remainder = nodes ^ sinks
            remainder_capacity = int(subset_capacities[remainder])
            local_product = np.asarray([1.0], dtype=np.longdouble)
            bits = sinks
            while bits:
                sink_bit = bits & -bits
                sink = sink_bit.bit_length() - 1
                local = available_by_size[
                    sink, remainder, : child_capacities[sink] + 1
                ]
                local_product = np.convolve(local_product, local)
                bits ^= sink_bit
            term = np.convolve(
                total_polynomial[remainder, : remainder_capacity + 1],
                local_product,
            )
            if sinks.bit_count() % 2 == 0:
                term = -term
            width = len(term)
            previous = total[:width].copy()
            updated = previous + term
            correction[:width] += np.where(
                np.abs(previous) >= np.abs(term),
                (previous - updated) + term,
                (term - updated) + previous,
            )
            total[:width] = updated
            absolute_term_sum[:width] += np.abs(term)

        value = total + correction
        tolerance = (
            np.longdouble(512.0)
            * np.finfo(np.longdouble).eps
            * np.maximum(np.longdouble(1.0), absolute_term_sum)
        )
        materially_negative = value < -tolerance
        if np.any(materially_negative) or not np.all(np.isfinite(value)):
            raise FloatingPointError(
                "sink-set graph-size recurrence lost coefficient positivity"
            )
        value[value < 0.0] = 0.0
        total_polynomial[nodes, : capacity + 1] = value

    coefficients = total_polynomial[full_mask]
    scaled_normalizer = np.sum(coefficients, dtype=np.longdouble)
    if not np.isfinite(scaled_normalizer) or scaled_normalizer <= 0.0:
        raise FloatingPointError("graph-size normalizer is not finite and positive")
    probabilities = coefficients / scaled_normalizer
    graph_size_probs = {
        int(size): float(probability)
        for size, probability in enumerate(probabilities)
        if probability > 0.0
    }
    total_probability = float(sum(graph_size_probs.values()))
    graph_size_probs = {
        size: probability / total_probability
        for size, probability in graph_size_probs.items()
    }
    return float(np.log(scaled_normalizer) + log_scale_total), graph_size_probs


def exact_structure_dp(
    table: OrderCandidateTable,
    *,
    beta: float = 1.0,
    max_nodes: int = 12,
    compute_graph_size: bool = True,
) -> ExactStructureDPResult:
    """Return exact candidate-restricted structure-posterior quantities.

    Runtime is ``O(3^d + d 2^d + number_of_retained_parent_entries)`` and the
    main arrays use ``O(d 2^d)`` memory. The default bound follows protocol v2's
    small-network exact-reference envelope. ``log_evidence_ratio`` means
    ``log Z_beta - log Z_0`` for this fixed candidate space and structural
    prior; data-dependent screening prevents interpreting it as an unrestricted
    marginal likelihood.
    """
    _validate_candidate_table(table)
    node_count = len(table.nodes)
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, numbers.Integral):
        raise ValueError("max_nodes must be a positive integer")
    max_nodes = int(max_nodes)
    if max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    if node_count > max_nodes:
        raise ValueError(
            f"exact structure DP requested for d={node_count}, exceeding max_nodes={max_nodes}"
        )
    if isinstance(beta, bool) or not isinstance(beta, numbers.Real):
        raise ValueError("beta must lie in [0, 1]")
    beta = float(beta)
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if not isinstance(compute_graph_size, bool):
        raise ValueError("compute_graph_size must be boolean")

    log_normalizer, edge_probs = _sink_dp(table, beta, return_edge_probs=True)
    graph_size_probs: GraphSizeDistribution | None = None
    if compute_graph_size:
        size_log_normalizer, graph_size_mass = _sink_graph_size_distribution(table, beta)
        if not np.isclose(size_log_normalizer, log_normalizer, atol=3e-10, rtol=3e-12):
            raise RuntimeError("exact structure scalar and graph-size normalizers disagree")
        source_result_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": "ordersmc-exact-structure-graph-size-result-v1",
                    "candidate_table_sha256": candidate_table_content_sha256(table),
                    "beta_hex": float(beta).hex(),
                    "log_normalizer_hex": float(log_normalizer).hex(),
                    "probabilities": [
                        {"edge_count": count, "probability_hex": float(probability).hex()}
                        for count, probability in sorted(graph_size_mass.items())
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        graph_size_probs = GraphSizeDistribution._from_computed(
            graph_size_mass,
            target_label="structure",
            target_semantics=STRUCTURE_TARGET,
            support_id=f"candidate-sha256:{candidate_table_content_sha256(table)}",
            node_count=node_count,
            max_indegree=candidate_max_indegree(table),
            beta=beta,
            producer="exact_structure_feature_dp",
            source_result_sha256=source_result_sha256,
        )
    if beta == 0.0:
        log_prior_normalizer = log_normalizer
    else:
        log_prior_normalizer, _ = _sink_dp(table, 0.0, return_edge_probs=False)
    return ExactStructureDPResult(
        edge_probs=edge_probs,
        graph_size_probs=graph_size_probs,
        log_normalizer=log_normalizer,
        log_prior_normalizer=log_prior_normalizer,
        log_evidence_ratio=float(log_normalizer - log_prior_normalizer),
        beta=beta,
        node_count=node_count,
        subset_count=1 << node_count,
        recurrence_term_count=3**node_count - 2**node_count,
    )
