"""Same-target Order MCMC comparator for the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import time
from typing import Dict, Literal

import numpy as np

from ordersmc.core import (
    OrderCandidateTable,
    best_edges_and_score_for_order,
    edge_probs_for_order,
    order_log_target,
)


@dataclass(frozen=True)
class OrderMCMCConfig:
    iterations: int = 20_000
    burnin: int = 5_000
    thin: int = 1
    global_swap_probability: float = 0.75
    seed: int = 0
    wall_seconds: float | None = None
    wall_burnin_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.burnin < 0 or self.burnin >= self.iterations:
            raise ValueError("require iterations >= 1 and 0 <= burnin < iterations")
        if self.thin < 1:
            raise ValueError("thin must be at least one")
        if not 0.0 <= self.global_swap_probability <= 1.0:
            raise ValueError("global_swap_probability must lie in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.wall_seconds is not None and self.wall_seconds <= 0.0:
            raise ValueError("wall_seconds must be positive")
        if not 0.0 <= self.wall_burnin_fraction < 1.0:
            raise ValueError("wall_burnin_fraction must lie in [0, 1)")


@dataclass
class OrderMCMCResult:
    orders: list[np.ndarray]
    edge_probs: Dict[tuple[str, str], float]
    edge_mcse: Dict[tuple[str, str], float]
    best_order: np.ndarray
    best_edges: list[tuple[str, str]]
    acceptance_rate: float
    acceptance_by_proposal: Dict[str, float]
    retained_samples: int
    stored_samples: int
    effective_sample_size: float
    edge_effective_sample_size: Dict[tuple[str, str], float]
    edge_lag1_autocorrelation: Dict[tuple[str, str], float]
    log_target_effective_sample_size: float
    log_target_lag1_autocorrelation: float
    log_target_trace: np.ndarray
    proposal_counts: Dict[str, Dict[str, int]]
    runtime_seconds: float = float("nan")
    completed_iterations: int = 0
    bit_generator: str = "PCG64DXSM"
    target: Literal["candidate_restricted_order_modular"] = (
        "candidate_restricted_order_modular"
    )
    target_label: Literal["order"] = "order"


@dataclass(frozen=True)
class ExactOrderMCMCKernel:
    """Exact finite-state kernel used to accept the reference comparator.

    This diagnostic is intentionally limited to small node sets.  It exposes
    the proposal and Metropolis transition probabilities separately so the
    protocol tests can verify symmetry, detailed balance, invariance, and
    irreducibility without relying on a long stochastic run.
    """

    orders: tuple[tuple[int, ...], ...]
    proposal_matrix: np.ndarray
    transition_matrix: np.ndarray
    target_probabilities: np.ndarray


def exact_order_mcmc_kernel(
    table: OrderCandidateTable,
    *,
    global_swap_probability: float = 0.75,
    max_nodes: int = 8,
) -> ExactOrderMCMCKernel:
    """Enumerate the exact proposal and MH kernels for a tiny order target.

    Global proposals choose an unordered pair uniformly and adjacent
    proposals choose an adjacent position uniformly.  Both proposal
    components are symmetric.  Rejected Metropolis moves are accumulated on
    the diagonal of the returned transition matrix.
    """

    if not 0.0 <= global_swap_probability <= 1.0:
        raise ValueError("global_swap_probability must lie in [0, 1]")
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least one")
    node_count = len(table.nodes)
    if node_count == 0:
        raise ValueError("candidate table must contain nodes")
    if node_count > max_nodes:
        raise ValueError(
            f"exact kernel is limited to {max_nodes} nodes; got {node_count}"
        )

    orders = tuple(permutations(range(node_count)))
    state_index = {order: index for index, order in enumerate(orders)}
    state_count = len(orders)
    proposal = np.zeros((state_count, state_count), dtype=float)

    if node_count == 1:
        proposal[0, 0] = 1.0
    else:
        global_mass = global_swap_probability / (node_count * (node_count - 1) / 2)
        adjacent_mass = (1.0 - global_swap_probability) / (node_count - 1)
        for source_index, source in enumerate(orders):
            for left in range(node_count):
                for right in range(left + 1, node_count):
                    destination = list(source)
                    destination[left], destination[right] = (
                        destination[right],
                        destination[left],
                    )
                    proposal[source_index, state_index[tuple(destination)]] += global_mass
            for left in range(node_count - 1):
                destination = list(source)
                destination[left], destination[left + 1] = (
                    destination[left + 1],
                    destination[left],
                )
                proposal[source_index, state_index[tuple(destination)]] += adjacent_mass

    log_targets = np.asarray(
        [order_log_target(np.asarray(order, dtype=np.int32), table, 1.0) for order in orders],
        dtype=float,
    )
    log_normalizer = float(np.logaddexp.reduce(log_targets))
    target_probabilities = np.exp(log_targets - log_normalizer)
    transition = np.zeros_like(proposal)
    for source in range(state_count):
        for destination in range(state_count):
            if source == destination or proposal[source, destination] == 0.0:
                continue
            log_acceptance = min(
                0.0, float(log_targets[destination] - log_targets[source])
            )
            transition[source, destination] = (
                proposal[source, destination] * np.exp(log_acceptance)
            )
        transition[source, source] = 1.0 - float(np.sum(transition[source]))

    return ExactOrderMCMCKernel(
        orders=orders,
        proposal_matrix=proposal,
        transition_matrix=transition,
        target_probabilities=target_probabilities,
    )


def _batch_means_mcse(values: np.ndarray) -> tuple[float, float]:
    """Return a conservative batch-means MCSE and implied ESS.

    The estimator uses all values for its mean. Up to 100 contiguous batches
    diagnose serial dependence; a short remainder is excluded only from the
    variance diagnostic. Constant traces have zero MCSE and full ESS.
    """

    series = np.asarray(values, dtype=float)
    count = len(series)
    if count < 4:
        return float("nan"), float(count)
    variance = float(np.var(series, ddof=1))
    if variance <= 0.0:
        return 0.0, float(count)
    batch_count = min(100, max(2, int(np.sqrt(count))))
    batch_size = count // batch_count
    if batch_size < 2:
        return float(np.sqrt(variance / count)), float(count)
    used = batch_count * batch_size
    batch_means = series[:used].reshape(batch_count, batch_size).mean(axis=1)
    variance_of_mean = float(np.var(batch_means, ddof=1) / batch_count)
    variance_of_mean = max(0.0, variance_of_mean)
    mcse = float(np.sqrt(variance_of_mean))
    ess = float(count if variance_of_mean == 0.0 else min(count, variance / variance_of_mean))
    return mcse, max(1.0, ess)


def _lag1_autocorrelation(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=float)
    if len(series) < 2:
        return 0.0
    centered = series - float(np.mean(series))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


def _rao_blackwell_mcmc_summary(
    orders: list[np.ndarray], table: OrderCandidateTable
) -> tuple[
    Dict[tuple[str, str], float],
    Dict[tuple[str, str], float],
    Dict[tuple[str, str], float],
    Dict[tuple[str, str], float],
]:
    edges = sorted(
        {
            (table.nodes[int(parent)], table.nodes[child])
            for child, parent_sets in enumerate(table.parent_sets)
            for parent_set in parent_sets
            for parent in parent_set
            if int(parent) >= 0
        }
    )
    sums = {edge: 0.0 for edge in edges}
    series = {edge: np.empty(len(orders), dtype=float) for edge in edges}
    for index, order in enumerate(orders):
        conditional = edge_probs_for_order(order, table, beta=1.0)
        for edge in edges:
            value = float(conditional.get(edge, 0.0))
            sums[edge] += value
            series[edge][index] = value
    probabilities = {edge: sums[edge] / len(orders) for edge in edges}
    mcse: Dict[tuple[str, str], float] = {}
    ess: Dict[tuple[str, str], float] = {}
    lag1: Dict[tuple[str, str], float] = {}
    for edge in edges:
        mcse[edge], ess[edge] = _batch_means_mcse(series[edge])
        lag1[edge] = _lag1_autocorrelation(series[edge])
    return probabilities, mcse, ess, lag1


def order_mcmc(
    table: OrderCandidateTable,
    config: OrderMCMCConfig = OrderMCMCConfig(),
) -> OrderMCMCResult:
    """Sample the exact ``beta=1`` order target with symmetric swap moves."""

    rng = np.random.Generator(np.random.PCG64DXSM(config.seed))
    node_count = len(table.nodes)
    if node_count == 0:
        raise ValueError("candidate table must contain nodes")
    order = rng.permutation(node_count).astype(np.int32)
    target = order_log_target(order, table, 1.0)
    best_order = order.copy()
    best_edges, best_score = best_edges_and_score_for_order(best_order, table, beta=1.0)
    stored_orders: list[np.ndarray] = []
    target_trace: list[float] = []
    proposed = {"global_swap": 0, "adjacent_swap": 0}
    accepted = {"global_swap": 0, "adjacent_swap": 0}
    started = time.perf_counter()
    burnin_seconds = (
        config.wall_seconds * config.wall_burnin_fraction
        if config.wall_seconds is not None
        else None
    )
    retained_samples = 0
    completed_iterations = 0

    for iteration in range(config.iterations):
        elapsed = time.perf_counter() - started
        if config.wall_seconds is not None and elapsed >= config.wall_seconds and retained_samples:
            break
        proposal = order.copy()
        if node_count < 2:
            proposal_name = "adjacent_swap"
        elif rng.random() < config.global_swap_probability:
            proposal_name = "global_swap"
            left, right = rng.choice(node_count, size=2, replace=False)
            proposal[left], proposal[right] = proposal[right], proposal[left]
        else:
            proposal_name = "adjacent_swap"
            left = int(rng.integers(0, node_count - 1))
            proposal[left], proposal[left + 1] = proposal[left + 1], proposal[left]
        proposed[proposal_name] += 1
        proposal_target = order_log_target(proposal, table, 1.0)
        if np.log(rng.random()) < min(0.0, proposal_target - target):
            order = proposal
            target = proposal_target
            accepted[proposal_name] += 1
            candidate_edges, candidate_score = best_edges_and_score_for_order(
                order, table, beta=1.0
            )
            candidate_key = (tuple(sorted(candidate_edges)), tuple(map(int, order)))
            best_key = (tuple(sorted(best_edges)), tuple(map(int, best_order)))
            if candidate_score > best_score or (
                candidate_score == best_score and candidate_key < best_key
            ):
                best_score = candidate_score
                best_order = order.copy()
                best_edges = candidate_edges
        completed_iterations = iteration + 1
        post_burnin = (
            elapsed >= burnin_seconds
            if burnin_seconds is not None
            else iteration >= config.burnin
        )
        if post_burnin:
            target_trace.append(float(target))
            if retained_samples % config.thin == 0:
                stored_orders.append(order.copy())
            retained_samples += 1

    if not stored_orders:
        raise RuntimeError("Order MCMC retained no post-burn-in samples")
    edge_probs, edge_mcse, edge_ess, edge_lag1 = _rao_blackwell_mcmc_summary(
        stored_orders, table
    )
    target_series = np.asarray(target_trace, dtype=float)
    _, target_ess = _batch_means_mcse(target_series)
    target_lag1 = _lag1_autocorrelation(target_series)
    finite_edge_ess = [value for value in edge_ess.values() if np.isfinite(value)]
    effective_sample_size = min(finite_edge_ess, default=target_ess)
    total_proposed = sum(proposed.values())
    total_accepted = sum(accepted.values())
    acceptance_by_proposal = {
        name: (accepted[name] / proposed[name] if proposed[name] else float("nan"))
        for name in proposed
    }
    return OrderMCMCResult(
        orders=stored_orders,
        edge_probs=edge_probs,
        edge_mcse=edge_mcse,
        best_order=best_order,
        best_edges=best_edges,
        acceptance_rate=total_accepted / total_proposed if total_proposed else float("nan"),
        acceptance_by_proposal=acceptance_by_proposal,
        retained_samples=retained_samples,
        stored_samples=len(stored_orders),
        effective_sample_size=effective_sample_size,
        edge_effective_sample_size=edge_ess,
        edge_lag1_autocorrelation=edge_lag1,
        log_target_effective_sample_size=target_ess,
        log_target_lag1_autocorrelation=target_lag1,
        log_target_trace=target_series,
        proposal_counts={
            name: {"proposed": proposed[name], "accepted": accepted[name]}
            for name in proposed
        },
        runtime_seconds=time.perf_counter() - started,
        completed_iterations=completed_iterations,
    )
