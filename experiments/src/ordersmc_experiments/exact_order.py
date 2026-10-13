"""Exact subset dynamic programming for the retained order-modular target."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Dict

import numpy as np
from scipy.special import logsumexp

from ordersmc.core import (
    GraphSizeDistribution,
    OrderCandidateTable,
    _validate_candidate_table,
    candidate_max_indegree,
)
from ordersmc.persistence import candidate_table_content_sha256


@dataclass(frozen=True)
class ExactOrderDPResult:
    edge_probs: Dict[tuple[str, str], float]
    graph_size_probs: GraphSizeDistribution | None
    log_normalizer: float
    log_prior_normalizer: float
    log_evidence_ratio: float
    beta: float
    node_count: int
    subset_count: int
    target_label: str = "order"
    target_semantics: str = "candidate_restricted_order_modular"


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
            child_masks[family_idx] = mask
        masks.append(child_masks)
    return masks


def _log_target_dp(
    table: OrderCandidateTable,
    beta: float,
    *,
    return_edge_probs: bool,
) -> tuple[float, Dict[tuple[str, str], float]]:
    node_count = len(table.nodes)
    subset_count = 1 << node_count
    full_mask = subset_count - 1
    masks = _family_masks(table)
    family_logweights = [
        np.asarray(table.logprior[child], dtype=float)
        + float(beta) * np.asarray(table.loglik[child], dtype=float)
        for child in range(node_count)
    ]
    log_available = np.full((node_count, subset_count), -np.inf, dtype=float)
    for child in range(node_count):
        child_bit = 1 << child
        for subset in range(subset_count):
            if subset & child_bit:
                continue
            compatible = (masks[child] & np.uint64(full_mask ^ subset)) == 0
            if np.any(compatible):
                log_available[child, subset] = float(logsumexp(family_logweights[child][compatible]))

    log_forward = np.full(subset_count, -np.inf, dtype=float)
    log_forward[0] = 0.0
    for subset in range(subset_count):
        if not np.isfinite(log_forward[subset]):
            continue
        for child in range(node_count):
            child_bit = 1 << child
            if subset & child_bit:
                continue
            local = log_available[child, subset]
            if np.isfinite(local):
                destination = subset | child_bit
                log_forward[destination] = np.logaddexp(
                    log_forward[destination], log_forward[subset] + local
                )

    log_backward = np.full(subset_count, -np.inf, dtype=float)
    log_backward[full_mask] = 0.0
    for subset in range(full_mask - 1, -1, -1):
        terms = []
        for child in range(node_count):
            child_bit = 1 << child
            if subset & child_bit:
                continue
            local = log_available[child, subset]
            suffix = log_backward[subset | child_bit]
            if np.isfinite(local) and np.isfinite(suffix):
                terms.append(local + suffix)
        if terms:
            log_backward[subset] = float(logsumexp(terms))

    log_normalizer = float(log_forward[full_mask])
    if not np.isfinite(log_normalizer) or not np.isclose(
        log_normalizer, log_backward[0], atol=1e-9, rtol=1e-11
    ):
        raise RuntimeError("exact order DP forward/backward normalizers disagree")
    if not return_edge_probs:
        return log_normalizer, {}

    log_numerators = np.full((node_count, node_count), -np.inf, dtype=float)
    for child in range(node_count):
        child_bit = 1 << child
        for subset in range(subset_count):
            if subset & child_bit or not np.isfinite(log_forward[subset]):
                continue
            suffix = log_backward[subset | child_bit]
            if not np.isfinite(suffix):
                continue
            compatible_indices = np.flatnonzero(
                (masks[child] & np.uint64(full_mask ^ subset)) == 0
            )
            base = log_forward[subset] + suffix
            for family_idx in compatible_indices:
                family_mask = int(masks[child][family_idx])
                contribution = base + float(family_logweights[child][family_idx])
                while family_mask:
                    parent_bit = family_mask & -family_mask
                    parent = parent_bit.bit_length() - 1
                    log_numerators[parent, child] = np.logaddexp(
                        log_numerators[parent, child], contribution
                    )
                    family_mask ^= parent_bit

    edge_probs: Dict[tuple[str, str], float] = {}
    for parent in range(node_count):
        for child in range(node_count):
            if parent == child:
                continue
            log_numerator = log_numerators[parent, child]
            probability = 0.0 if not np.isfinite(log_numerator) else math.exp(
                float(log_numerator - log_normalizer)
            )
            edge_probs[(table.nodes[parent], table.nodes[child])] = float(
                min(1.0, max(0.0, probability))
            )
    return log_normalizer, edge_probs


def _log_graph_size_dp(
    table: OrderCandidateTable,
    beta: float,
) -> tuple[float, Dict[int, float]]:
    """Propagate an exact edge-count feature polynomial through the order DP."""

    node_count = len(table.nodes)
    subset_count = 1 << node_count
    full_mask = subset_count - 1
    masks = _family_masks(table)
    family_logweights = [
        np.asarray(table.logprior[child], dtype=float)
        + float(beta) * np.asarray(table.loglik[child], dtype=float)
        for child in range(node_count)
    ]
    family_sizes = [
        np.asarray([int(mask).bit_count() for mask in child_masks], dtype=np.int32)
        for child_masks in masks
    ]
    child_capacities = [int(np.max(sizes, initial=0)) for sizes in family_sizes]
    edge_capacity = int(sum(child_capacities))

    log_available: list[np.ndarray] = []
    for child in range(node_count):
        local = np.full(
            (subset_count, child_capacities[child] + 1), -np.inf, dtype=float
        )
        child_bit = 1 << child
        for subset in range(subset_count):
            if subset & child_bit:
                continue
            compatible = np.flatnonzero(
                (masks[child] & np.uint64(full_mask ^ subset)) == 0
            )
            for family_index in compatible:
                size = int(family_sizes[child][family_index])
                local[subset, size] = np.logaddexp(
                    local[subset, size], family_logweights[child][family_index]
                )
        log_available.append(local)

    log_forward = np.full((subset_count, edge_capacity + 1), -np.inf, dtype=float)
    log_forward[0, 0] = 0.0
    subset_capacities = np.zeros(subset_count, dtype=np.int32)
    for subset in range(1, subset_count):
        bit = subset & -subset
        child = bit.bit_length() - 1
        subset_capacities[subset] = (
            subset_capacities[subset ^ bit] + child_capacities[child]
        )

    for subset in range(subset_count):
        source_capacity = int(subset_capacities[subset])
        for child in range(node_count):
            child_bit = 1 << child
            if subset & child_bit:
                continue
            destination = subset | child_bit
            local = log_available[child][subset]
            finite_local = np.flatnonzero(np.isfinite(local))
            for current_size in range(source_capacity + 1):
                current = log_forward[subset, current_size]
                if not np.isfinite(current):
                    continue
                for local_size in finite_local:
                    total_size = current_size + int(local_size)
                    log_forward[destination, total_size] = np.logaddexp(
                        log_forward[destination, total_size],
                        current + local[int(local_size)],
                    )

    coefficients = log_forward[full_mask]
    finite = np.isfinite(coefficients)
    if not np.any(finite):
        raise RuntimeError("exact order graph-size DP has a non-finite normalizer")
    # Normalize in a shifted coordinate system.  Subtracting the rounded
    # ``logsumexp`` result from very large-magnitude log scores can otherwise
    # discard several bits and make probabilities miss one by ~1e-12 even
    # though the feature DP itself is correct.
    maximum = float(np.max(coefficients[finite]))
    shifted_weights = np.zeros_like(coefficients)
    shifted_weights[finite] = np.exp(coefficients[finite] - maximum)
    shifted_total = math.fsum(float(value) for value in shifted_weights)
    if not math.isfinite(shifted_total) or shifted_total <= 0.0:
        raise RuntimeError("exact order graph-size DP has invalid shifted mass")
    probabilities = shifted_weights / shifted_total
    log_normalizer = maximum + math.log(shifted_total)
    graph_size_probs = {
        int(size): float(probability)
        for size, probability in enumerate(probabilities)
        if probability > 0.0
    }
    total = math.fsum(graph_size_probs.values())
    if not np.isclose(total, 1.0, atol=32 * np.finfo(float).eps, rtol=0.0):
        raise RuntimeError("exact order graph-size probabilities do not normalize")
    graph_size_probs = {
        size: probability / total for size, probability in graph_size_probs.items()
    }
    return log_normalizer, graph_size_probs


def exact_order_dp(
    table: OrderCandidateTable,
    *,
    beta: float = 1.0,
    max_nodes: int = 20,
    compute_graph_size: bool = True,
) -> ExactOrderDPResult:
    """Compute exact retained-target PIPs in ``O(d 2^d)`` DP states.

    Family compatibility work remains proportional to the retained family
    table, so the confirmatory protocol limits this reference to ``d <= 12``.
    ``log_evidence_ratio`` is ``log Z_beta - log Z_0``. It is a conventional
    log marginal likelihood only when the candidate space and structural prior
    were fixed before observing these data.
    """

    _validate_candidate_table(table)
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, (int, np.integer)):
        raise ValueError("max_nodes must be a positive integer")
    max_nodes = int(max_nodes)
    if max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    node_count = len(table.nodes)
    if node_count == 0:
        raise ValueError("candidate table must contain at least one node")
    if node_count > max_nodes:
        raise ValueError(
            f"exact order DP requested for d={node_count}, exceeding max_nodes={max_nodes}"
        )
    if isinstance(beta, (bool, np.bool_)):
        raise ValueError("beta must lie in [0, 1]")
    try:
        beta = float(beta)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("beta must lie in [0, 1]") from exc
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if not isinstance(compute_graph_size, bool):
        raise ValueError("compute_graph_size must be boolean")
    log_normalizer, edge_probs = _log_target_dp(table, beta, return_edge_probs=True)
    graph_size_probs: GraphSizeDistribution | None = None
    if compute_graph_size:
        size_log_normalizer, graph_size_mass = _log_graph_size_dp(table, beta)
        if not np.isclose(size_log_normalizer, log_normalizer, atol=2e-10, rtol=2e-12):
            raise RuntimeError("exact order scalar and graph-size normalizers disagree")
        source_result_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": "ordersmc-exact-order-graph-size-result-v1",
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
            target_label="order",
            target_semantics="candidate_restricted_order_modular",
            support_id=f"candidate-sha256:{candidate_table_content_sha256(table)}",
            node_count=node_count,
            max_indegree=candidate_max_indegree(table),
            beta=beta,
            producer="exact_order_feature_dp",
            source_result_sha256=source_result_sha256,
        )
    if beta == 0.0:
        log_prior_normalizer = log_normalizer
    else:
        log_prior_normalizer, _ = _log_target_dp(table, 0.0, return_edge_probs=False)
    return ExactOrderDPResult(
        edge_probs=edge_probs,
        graph_size_probs=graph_size_probs,
        log_normalizer=log_normalizer,
        log_prior_normalizer=log_prior_normalizer,
        log_evidence_ratio=float(log_normalizer - log_prior_normalizer),
        beta=float(beta),
        node_count=node_count,
        subset_count=1 << node_count,
    )
