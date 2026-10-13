"""Finite-state acceptance tests for the reference Order MCMC comparator."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from ordersmc.core import OrderCandidateTable
from ordersmc_experiments.exact_order import exact_order_dp
from ordersmc_experiments.order_mcmc import (
    OrderMCMCConfig,
    exact_order_mcmc_kernel,
    order_mcmc,
)


def _tiny_table(node_count: int) -> OrderCandidateTable:
    """Construct a nonuniform, complete tiny candidate table deterministically."""

    rng = np.random.Generator(np.random.PCG64DXSM(18_700 + node_count))
    nodes = [f"X{index}" for index in range(node_count)]
    parent_sets: list[np.ndarray] = []
    loglik: list[np.ndarray] = []
    logprior: list[np.ndarray] = []
    for child in range(node_count):
        possible = [parent for parent in range(node_count) if parent != child]
        families = [parents for size in range(len(possible) + 1) for parents in combinations(possible, size)]
        width = max(1, node_count - 1)
        encoded = np.full((len(families), width), -1, dtype=np.int32)
        for family_index, parents in enumerate(families):
            encoded[family_index, : len(parents)] = parents
        parent_sets.append(encoded)
        loglik.append(rng.normal(scale=0.55, size=len(families)))
        logprior.append(rng.normal(scale=0.17, size=len(families)))
    return OrderCandidateTable(nodes, parent_sets, loglik, logprior)


@pytest.mark.parametrize("node_count", [1, 2, 3, 4])
@pytest.mark.parametrize("global_probability", [0.0, 0.37, 1.0])
def test_enumerated_kernel_is_stochastic_reversible_invariant_and_irreducible(
    node_count: int, global_probability: float
) -> None:
    kernel = exact_order_mcmc_kernel(
        _tiny_table(node_count), global_swap_probability=global_probability
    )
    proposal = kernel.proposal_matrix
    transition = kernel.transition_matrix
    target = kernel.target_probabilities

    assert proposal.shape == transition.shape == (len(kernel.orders), len(kernel.orders))
    assert np.all(proposal >= 0.0)
    assert np.all(transition >= -2e-15)
    assert proposal.sum(axis=1) == pytest.approx(np.ones(len(kernel.orders)), abs=2e-15)
    assert transition.sum(axis=1) == pytest.approx(np.ones(len(kernel.orders)), abs=2e-15)
    assert proposal == pytest.approx(proposal.T, abs=2e-15)

    probability_flux = target[:, None] * transition
    assert probability_flux == pytest.approx(probability_flux.T, abs=3e-15)
    assert target @ transition == pytest.approx(target, abs=3e-15)

    communication_graph = nx.DiGraph(transition > 0.0)
    assert nx.is_strongly_connected(communication_graph)


def test_enumerated_proposal_probabilities_match_the_declared_mixture() -> None:
    global_probability = 0.3
    kernel = exact_order_mcmc_kernel(
        _tiny_table(3), global_swap_probability=global_probability
    )
    source = kernel.orders.index((0, 1, 2))
    destinations = {order: kernel.proposal_matrix[source, index] for index, order in enumerate(kernel.orders)}

    global_mass = global_probability / 3.0
    adjacent_mass = (1.0 - global_probability) / 2.0
    assert destinations[(1, 0, 2)] == pytest.approx(global_mass + adjacent_mass)
    assert destinations[(0, 2, 1)] == pytest.approx(global_mass + adjacent_mass)
    assert destinations[(2, 1, 0)] == pytest.approx(global_mass)
    assert destinations[(0, 1, 2)] == 0.0


@pytest.mark.parametrize("node_count", [2, 3, 4])
def test_enumerated_stationary_pips_match_exact_order_dp(node_count: int) -> None:
    table = _tiny_table(node_count)
    kernel = exact_order_mcmc_kernel(table, global_swap_probability=0.41)
    exact = exact_order_dp(table)

    # The exact DP is algorithmically independent of the enumerated Markov
    # kernel.  Agreement checks that the kernel targets the declared order
    # posterior, not merely that its matrix identities are internally sound.
    from ordersmc.core import edge_probs_for_order

    for edge, expected in exact.edge_probs.items():
        marginal = sum(
            weight
            * edge_probs_for_order(np.asarray(order, dtype=np.int32), table).get(edge, 0.0)
            for order, weight in zip(kernel.orders, kernel.target_probabilities)
        )
        assert marginal == pytest.approx(expected, abs=3e-13)


def test_production_sampler_empirically_matches_the_enumerated_kernel() -> None:
    table = _tiny_table(3)
    global_probability = 0.37
    exact_kernel = exact_order_mcmc_kernel(
        table, global_swap_probability=global_probability
    )
    iterations = 30_000
    sampled = order_mcmc(
        table,
        OrderMCMCConfig(
            iterations=iterations,
            burnin=0,
            thin=1,
            global_swap_probability=global_probability,
            seed=91_227,
        ),
    )
    state_index = {order: index for index, order in enumerate(exact_kernel.orders)}
    counts = np.zeros_like(exact_kernel.transition_matrix)
    for source, destination in zip(sampled.orders, sampled.orders[1:]):
        counts[state_index[tuple(source)], state_index[tuple(destination)]] += 1.0
    row_counts = counts.sum(axis=1)
    assert np.all(row_counts > 1_000)
    empirical = counts / row_counts[:, None]
    assert empirical == pytest.approx(exact_kernel.transition_matrix, abs=0.025)
    global_count = sampled.proposal_counts["global_swap"]["proposed"]
    assert global_count / iterations == pytest.approx(global_probability, abs=0.012)


@pytest.mark.parametrize(
    ("node_count", "global_probability"),
    [(1, 0.83), (4, 0.0), (4, 0.37), (4, 1.0)],
)
def test_stochastic_run_counters_are_complete_and_algebraically_consistent(
    node_count: int, global_probability: float
) -> None:
    iterations = 731
    burnin = 73
    thin = 7
    result = order_mcmc(
        _tiny_table(node_count),
        OrderMCMCConfig(
            iterations=iterations,
            burnin=burnin,
            thin=thin,
            global_swap_probability=global_probability,
            seed=612_907,
        ),
    )
    proposed = sum(item["proposed"] for item in result.proposal_counts.values())
    accepted = sum(item["accepted"] for item in result.proposal_counts.values())
    assert proposed == iterations
    assert 0 <= accepted <= proposed
    assert result.acceptance_rate == pytest.approx(accepted / proposed)
    assert result.retained_samples == iterations - burnin
    assert result.stored_samples == len(range(burnin, iterations, thin))
    assert len(result.orders) == result.stored_samples
    assert len(result.log_target_trace) == result.retained_samples
    assert set(result.edge_effective_sample_size) == set(result.edge_probs)
    assert set(result.edge_lag1_autocorrelation) == set(result.edge_probs)
    for edge in result.edge_probs:
        assert 1.0 <= result.edge_effective_sample_size[edge] <= result.retained_samples
        assert -1.0 <= result.edge_lag1_autocorrelation[edge] <= 1.0
    assert 1.0 <= result.log_target_effective_sample_size <= result.retained_samples
    assert -1.0 <= result.log_target_lag1_autocorrelation <= 1.0
    for name, counts in result.proposal_counts.items():
        assert 0 <= counts["accepted"] <= counts["proposed"]
        expected = (
            counts["accepted"] / counts["proposed"]
            if counts["proposed"]
            else float("nan")
        )
        if np.isnan(expected):
            assert np.isnan(result.acceptance_by_proposal[name])
        else:
            assert result.acceptance_by_proposal[name] == pytest.approx(expected)

    if node_count > 1 and global_probability == 0.0:
        assert result.proposal_counts["global_swap"]["proposed"] == 0
    if node_count > 1 and global_probability == 1.0:
        assert result.proposal_counts["adjacent_swap"]["proposed"] == 0


def test_exact_kernel_rejects_unsafe_or_invalid_enumerations() -> None:
    with pytest.raises(ValueError, match="limited"):
        exact_order_mcmc_kernel(_tiny_table(5), max_nodes=4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        exact_order_mcmc_kernel(_tiny_table(2), global_swap_probability=1.01)
