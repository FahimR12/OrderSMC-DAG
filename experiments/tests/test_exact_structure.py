from __future__ import annotations

from itertools import combinations, product
import math

import networkx as nx
import numpy as np
import pytest
from scipy.special import logsumexp

from ordersmc.core import OrderCandidateTable
from ordersmc_experiments.exact_structure import STRUCTURE_TARGET, exact_structure_dp


def _parent_tuple(row: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(parent) for parent in row if int(parent) >= 0))


def exhaustive_candidate_dags(
    table: OrderCandidateTable,
    beta: float,
) -> tuple[float, dict[tuple[str, str], float], dict[int, float]]:
    """Enumerate every candidate-supported parent-set tuple and retain DAGs."""
    node_count = len(table.nodes)
    dag_logweights: list[float] = []
    dag_edges: list[set[tuple[int, int]]] = []
    family_ranges = [range(len(parent_sets)) for parent_sets in table.parent_sets]
    for selected_families in product(*family_ranges):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(node_count))
        logweight = 0.0
        edges: set[tuple[int, int]] = set()
        for child, family_index in enumerate(selected_families):
            parents = _parent_tuple(table.parent_sets[child][family_index])
            edges.update((parent, child) for parent in parents)
            logweight += float(table.logprior[child][family_index])
            logweight += float(beta) * float(table.loglik[child][family_index])
        graph.add_edges_from(edges)
        if nx.is_directed_acyclic_graph(graph):
            dag_logweights.append(logweight)
            dag_edges.append(edges)

    log_normalizer = float(logsumexp(dag_logweights))
    edge_probs: dict[tuple[str, str], float] = {}
    for parent in range(node_count):
        for child in range(node_count):
            if parent == child:
                continue
            selected = [
                logweight
                for logweight, edges in zip(dag_logweights, dag_edges)
                if (parent, child) in edges
            ]
            probability = 0.0 if not selected else math.exp(
                float(logsumexp(selected) - log_normalizer)
            )
            edge_probs[(table.nodes[parent], table.nodes[child])] = probability
    graph_size_probs: dict[int, float] = {}
    for logweight, edges in zip(dag_logweights, dag_edges):
        size = len(edges)
        graph_size_probs[size] = graph_size_probs.get(size, 0.0) + math.exp(
            logweight - log_normalizer
        )
    return log_normalizer, edge_probs, graph_size_probs


def two_node_table(*, restrict_b_to_a: bool = False, prior_shift: float = 0.0) -> OrderCandidateTable:
    # At beta one: empty DAG weight 1, B->A weight 2, A->B weight 3.
    a_parents = (
        np.array([[-1]], dtype=np.int32)
        if restrict_b_to_a
        else np.array([[-1], [1]], dtype=np.int32)
    )
    a_likelihood = np.array([0.0]) if restrict_b_to_a else np.log([1.0, 2.0])
    return OrderCandidateTable(
        nodes=["A", "B"],
        parent_sets=[a_parents, np.array([[-1], [0]], dtype=np.int32)],
        loglik=[a_likelihood, np.log([1.0, 3.0])],
        logprior=[
            np.full(len(a_parents), prior_shift),
            np.full(2, prior_shift),
        ],
    )


def random_candidate_table(node_count: int, seed: int) -> OrderCandidateTable:
    rng = np.random.default_rng(seed)
    parent_sets = []
    loglik = []
    logprior = []
    for child in range(node_count):
        possible_parents = [node for node in range(node_count) if node != child]
        families = [tuple()]
        for size in range(1, min(2, len(possible_parents)) + 1):
            for family in combinations(possible_parents, size):
                if rng.random() < 0.7:
                    families.append(family)
        width = max(1, max(map(len, families)))
        encoded = np.full((len(families), width), -1, dtype=np.int32)
        for row, family in enumerate(families):
            encoded[row, : len(family)] = family
        parent_sets.append(encoded)
        loglik.append(rng.normal(scale=0.8, size=len(families)))
        logprior.append(rng.normal(scale=0.3, size=len(families)))
    return OrderCandidateTable(
        nodes=[f"X{index}" for index in range(node_count)],
        parent_sets=parent_sets,
        loglik=loglik,
        logprior=logprior,
    )


def sparse_cycle_candidate_table(node_count: int) -> OrderCandidateTable:
    """Two families per child; exhaustive enumeration stays small through d=6."""

    parent_sets = []
    loglik = []
    logprior = []
    for child in range(node_count):
        parent = (child - 1) % node_count
        parent_sets.append(np.array([[-1], [parent]], dtype=np.int32))
        loglik.append(np.array([0.1 * child, 0.2 - 0.03 * child]))
        logprior.append(np.array([0.0, -0.17]))
    return OrderCandidateTable(
        nodes=[f"S{index}" for index in range(node_count)],
        parent_sets=parent_sets,
        loglik=loglik,
        logprior=logprior,
    )


def test_two_node_hand_calculation_and_prior_ratio() -> None:
    result = exact_structure_dp(two_node_table())
    assert result.target == STRUCTURE_TARGET
    assert result.log_normalizer == pytest.approx(math.log(6.0))
    assert result.log_prior_normalizer == pytest.approx(math.log(3.0))
    assert result.log_evidence_ratio == pytest.approx(math.log(2.0))
    assert result.edge_probs[("A", "B")] == pytest.approx(3.0 / 6.0)
    assert result.edge_probs[("B", "A")] == pytest.approx(2.0 / 6.0)
    assert result.graph_size_probs == pytest.approx({0: 1.0 / 6.0, 1: 5.0 / 6.0})
    assert result.recurrence_term_count == 3**2 - 2**2


def test_candidate_restrictions_give_excluded_edges_exactly_zero_mass() -> None:
    result = exact_structure_dp(two_node_table(restrict_b_to_a=True))
    assert result.log_normalizer == pytest.approx(math.log(4.0))
    assert result.log_prior_normalizer == pytest.approx(math.log(2.0))
    assert result.edge_probs[("B", "A")] == 0.0
    assert result.edge_probs[("A", "B")] == pytest.approx(3.0 / 4.0)


@pytest.mark.parametrize("node_count", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [3, 19, 41])
def test_sink_dp_matches_exhaustive_candidate_dag_enumeration(
    node_count: int,
    seed: int,
) -> None:
    table = random_candidate_table(node_count, seed)
    beta = 0.37
    expected_log_z, expected_edges, expected_sizes = exhaustive_candidate_dags(table, beta)
    expected_log_z0, _, _ = exhaustive_candidate_dags(table, 0.0)
    result = exact_structure_dp(table, beta=beta)

    assert result.log_normalizer == pytest.approx(expected_log_z, abs=2e-12)
    assert result.log_prior_normalizer == pytest.approx(expected_log_z0, abs=2e-12)
    assert result.log_evidence_ratio == pytest.approx(expected_log_z - expected_log_z0, abs=3e-12)
    assert result.edge_probs == pytest.approx(expected_edges, abs=3e-12)
    assert result.graph_size_probs == pytest.approx(expected_sizes, abs=3e-12)
    assert sum(size * mass for size, mass in result.graph_size_probs.items()) == pytest.approx(
        sum(result.edge_probs.values()), abs=3e-12
    )


@pytest.mark.parametrize("node_count", [5, 6])
def test_exhaustive_sparse_candidate_parity_extends_through_d6(node_count: int) -> None:
    table = sparse_cycle_candidate_table(node_count)
    expected_log_z, expected_edges, expected_sizes = exhaustive_candidate_dags(table, 0.63)
    result = exact_structure_dp(table, beta=0.63)
    assert result.log_normalizer == pytest.approx(expected_log_z, abs=3e-12)
    assert result.edge_probs == pytest.approx(expected_edges, abs=3e-12)
    assert result.graph_size_probs == pytest.approx(expected_sizes, abs=3e-12)


def test_per_child_scaling_avoids_overflow_without_changing_pips() -> None:
    ordinary = exact_structure_dp(two_node_table())
    shifted = exact_structure_dp(two_node_table(prior_shift=1_000.0))
    assert shifted.log_normalizer == pytest.approx(ordinary.log_normalizer + 2_000.0)
    assert shifted.log_prior_normalizer == pytest.approx(
        ordinary.log_prior_normalizer + 2_000.0
    )
    assert shifted.log_evidence_ratio == pytest.approx(ordinary.log_evidence_ratio)
    assert shifted.edge_probs == pytest.approx(ordinary.edge_probs)


def test_invalid_bounds_and_duplicate_candidate_families_are_rejected() -> None:
    with pytest.raises(ValueError, match="exceeding max_nodes"):
        exact_structure_dp(random_candidate_table(4, 2), max_nodes=3)
    table = two_node_table()
    table.parent_sets[1] = np.array([[-1], [0], [0]], dtype=np.int32)
    table.loglik[1] = np.array([0.0, 1.0, 1.0])
    table.logprior[1] = np.zeros(3)
    with pytest.raises(ValueError, match="duplicate family"):
        exact_structure_dp(table)
