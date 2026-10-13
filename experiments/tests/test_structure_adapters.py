from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from ordersmc.core import OrderCandidateTable
from ordersmc_experiments.exact_structure import exact_structure_dp
from ordersmc_experiments.adapter_validation import validate_algorithm_adapters
from ordersmc_experiments.structure_mcmc import (
    _allowed_rows_after_orphaning,
    _families,
    dag_log_target,
    dag_state_is_valid,
    empty_dag_state,
    enumerate_candidate_dags,
    enumerate_ordered_partitions,
    exact_partition_pips,
    PartitionMCMCConfig,
    REVMCMCConfig,
    partition_mcmc,
    partition_log_target,
    partition_proposal_probability,
    rev_proposal_distribution,
    rev_mcmc,
    standard_structure_proposal_distribution,
    validate_structure_candidate_table,
)


def _complete_three_node_table() -> OrderCandidateTable:
    nodes = ["a", "b", "c"]
    parent_sets = []
    loglik = []
    logprior = []
    for child in range(3):
        possible = [node for node in range(3) if node != child]
        families = [()] + [(node,) for node in possible] + [tuple(possible)]
        rows = np.full((len(families), 2), -1, dtype=np.int32)
        values = []
        priors = []
        for index, family in enumerate(families):
            rows[index, : len(family)] = family
            directional = sum(0.17 * (parent + 1) - 0.09 * (child + 1) for parent in family)
            interaction = 0.21 if len(family) == 2 and child == 2 else 0.0
            values.append(directional + interaction)
            priors.append(-0.31 * len(family))
        parent_sets.append(rows)
        loglik.append(np.asarray(values, dtype=float))
        logprior.append(np.asarray(priors, dtype=float))
    return OrderCandidateTable(nodes, parent_sets, loglik, logprior)


def _assert_detailed_balance(states, log_targets, proposal) -> None:
    log_norm = logsumexp(log_targets)
    target = np.exp(np.asarray(log_targets) - log_norm)
    index = {state: i for i, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)), dtype=float)
    for i, source in enumerate(states):
        distribution = proposal(source)
        assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-12)
        for destination, q_forward in distribution.items():
            j = index[destination]
            if i == j:
                continue
            q_reverse = proposal(destination).get(source, 0.0)
            assert q_reverse > 0.0
            acceptance = min(1.0, target[j] * q_reverse / (target[i] * q_forward))
            transition[i, j] += q_forward * acceptance
        transition[i, i] = 1.0 - transition[i].sum()
    assert np.allclose(transition.sum(axis=1), 1.0, atol=1e-12)
    flow = target[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) < 2e-12
    assert np.allclose(target @ transition, target, atol=2e-12)


def test_partition_representation_matches_exact_structure_pips_and_normalizer() -> None:
    table = _complete_three_node_table()
    validate_structure_candidate_table(table)
    exact = exact_structure_dp(table)
    partitions = enumerate_ordered_partitions(range(3))
    log_targets = np.asarray([partition_log_target(state, table) for state in partitions])
    assert logsumexp(log_targets) == pytest.approx(exact.log_normalizer, abs=2e-12)
    pips = exact_partition_pips(table)
    edges = {(u, v) for u in table.nodes for v in table.nodes if u != v}
    for edge in edges:
        assert pips.get(edge, 0.0) == pytest.approx(exact.edge_probs.get(edge, 0.0), abs=2e-12)


def test_partition_split_merge_kernel_has_exact_detailed_balance() -> None:
    table = _complete_three_node_table()
    states = tuple(
        state
        for state in enumerate_ordered_partitions(range(3))
        if np.isfinite(partition_log_target(state, table))
    )
    log_targets = [partition_log_target(state, table) for state in states]

    def proposal(source):
        result = {
            destination: partition_proposal_probability(source, destination)
            for destination in states
        }
        return {state: mass for state, mass in result.items() if mass > 0.0}

    _assert_detailed_balance(states, log_targets, proposal)


def test_standard_structure_kernel_has_exact_detailed_balance() -> None:
    table = _complete_three_node_table()
    states = enumerate_candidate_dags(table)
    _assert_detailed_balance(
        states,
        [dag_log_target(state, table) for state in states],
        lambda state: standard_structure_proposal_distribution(state, table),
    )


def test_rev_orphan_resample_kernel_has_exact_detailed_balance_on_nonempty_dags() -> None:
    table = _complete_three_node_table()
    all_states = enumerate_candidate_dags(table)
    # REV alone preserves non-emptiness and is not ergodic. Detailed balance is
    # checked on every communicating proposal, while the full sampler mixes it
    # with the ergodic standard structure kernel.
    for source in all_states:
        distribution = rev_proposal_distribution(source, table)
        assert sum(distribution.values()) == pytest.approx(1.0, abs=2e-12)
        for destination, q_forward in distribution.items():
            if source == destination:
                continue
            q_reverse = rev_proposal_distribution(destination, table).get(source, 0.0)
            assert q_reverse > 0.0
            log_flow_forward = dag_log_target(source, table) + np.log(q_forward)
            log_flow_reverse = dag_log_target(destination, table) + np.log(q_reverse)
            acceptance_forward = min(0.0, log_flow_reverse - log_flow_forward)
            acceptance_reverse = min(0.0, log_flow_forward - log_flow_reverse)
            assert log_flow_forward + acceptance_forward == pytest.approx(
                log_flow_reverse + acceptance_reverse, abs=2e-12
            )


def test_descendant_filter_matches_brute_force_orphan_row_validation() -> None:
    table = _complete_three_node_table()
    families = _families(table)
    empty = empty_dag_state(table)
    for state in enumerate_candidate_dags(table):
        for child in range(len(table.nodes)):
            base = list(state)
            base[child] = empty[child]
            base_state = tuple(base)
            for required_parent in (None, *[p for p in range(3) if p != child]):
                expected = []
                for row, parents in enumerate(families[child]):
                    if required_parent is not None and required_parent not in parents:
                        continue
                    proposal = list(base_state)
                    proposal[child] = row
                    if dag_state_is_valid(tuple(proposal), table):
                        expected.append(row)
                assert _allowed_rows_after_orphaning(
                    base_state,
                    child,
                    table,
                    required_parent=required_parent,
                ) == tuple(expected)


def test_downward_closure_is_required_for_fair_common_support() -> None:
    table = _complete_three_node_table()
    # Remove one singleton while retaining the corresponding pair.
    keep = [0, 2, 3]
    table.parent_sets[0] = table.parent_sets[0][keep]
    table.loglik[0] = table.loglik[0][keep]
    table.logprior[0] = table.logprior[0][keep]
    with pytest.raises(ValueError, match="not downward closed"):
        validate_structure_candidate_table(table)


def test_stochastic_structure_adapters_return_finite_normalized_pips() -> None:
    table = _complete_three_node_table()
    results = (
        partition_mcmc(
            table,
            PartitionMCMCConfig(iterations=1_200, burnin=200, thin=2, seed=11),
        ),
        rev_mcmc(
            table,
            REVMCMCConfig(iterations=1_500, burnin=300, thin=2, seed=19),
        ),
    )
    for result in results:
        assert result.stored_samples > 0
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert result.runtime_seconds > 0.0
        assert all(np.isfinite(probability) for probability in result.edge_probs.values())
        assert all(0.0 <= probability <= 1.0 for probability in result.edge_probs.values())


def test_checked_adapter_report_matches_the_executable_validation() -> None:
    checked = Path(__file__).resolve().parents[1] / "validation" / "adapter_validation.json"
    assert json.loads(checked.read_text(encoding="utf-8")) == validate_algorithm_adapters()
