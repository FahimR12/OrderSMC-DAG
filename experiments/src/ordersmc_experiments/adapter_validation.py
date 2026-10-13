"""Deterministic mathematical acceptance report for experiment adapters."""

from __future__ import annotations

import hashlib
import json
from itertools import permutations

import numpy as np
from scipy.special import logsumexp

from ordersmc.core import OrderCandidateTable

from .exact_order import exact_order_dp
from .exact_structure import exact_structure_dp
from .order_mcmc import exact_order_mcmc_kernel
from .structure_mcmc import (
    dag_log_target,
    enumerate_candidate_dags,
    enumerate_ordered_partitions,
    exact_partition_pips,
    partition_log_target,
    partition_proposal_probability,
    rev_proposal_distribution,
    standard_structure_proposal_distribution,
    validate_structure_candidate_table,
)


def adapter_fixture_table() -> OrderCandidateTable:
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
            values.append(
                sum(0.17 * (parent + 1) - 0.09 * (child + 1) for parent in family)
                + (0.21 if len(family) == 2 and child == 2 else 0.0)
            )
            priors.append(-0.31 * len(family))
        parent_sets.append(rows)
        loglik.append(np.asarray(values, dtype=float))
        logprior.append(np.asarray(priors, dtype=float))
    return OrderCandidateTable(nodes, parent_sets, loglik, logprior)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def _mh_residual(states, log_targets, proposal) -> tuple[float, float]:
    target = np.exp(np.asarray(log_targets, dtype=float) - logsumexp(log_targets))
    index = {state: row for row, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)), dtype=float)
    for i, source in enumerate(states):
        distribution = proposal(source)
        if abs(sum(distribution.values()) - 1.0) > 1e-11:
            raise RuntimeError("proposal distribution is not normalized")
        for destination, q_forward in distribution.items():
            j = index[destination]
            if i == j:
                continue
            q_reverse = proposal(destination).get(source, 0.0)
            if q_reverse <= 0.0:
                raise RuntimeError("proposal has no conjugate reverse move")
            acceptance = min(1.0, target[j] * q_reverse / (target[i] * q_forward))
            transition[i, j] += q_forward * acceptance
        transition[i, i] = 1.0 - float(np.sum(transition[i]))
    flow = target[:, None] * transition
    return (
        float(np.max(np.abs(flow - flow.T))),
        float(np.max(np.abs(target @ transition - target))),
    )


def _rev_flow_residual(table: OrderCandidateTable) -> float:
    states = enumerate_candidate_dags(table)
    maximum = 0.0
    for source in states:
        distribution = rev_proposal_distribution(source, table)
        for destination, q_forward in distribution.items():
            if source == destination:
                continue
            q_reverse = rev_proposal_distribution(destination, table).get(source, 0.0)
            if q_reverse <= 0.0:
                raise RuntimeError("REV proposal has no conjugate reverse move")
            forward = dag_log_target(source, table) + np.log(q_forward)
            reverse = dag_log_target(destination, table) + np.log(q_reverse)
            accepted_forward = forward + min(0.0, reverse - forward)
            accepted_reverse = reverse + min(0.0, forward - reverse)
            maximum = max(maximum, abs(accepted_forward - accepted_reverse))
    return float(maximum)


def validate_algorithm_adapters() -> dict[str, object]:
    """Return a fail-closed acceptance report without Monte Carlo uncertainty."""

    table = adapter_fixture_table()
    validate_structure_candidate_table(table)
    tolerance = 2e-11

    order_kernel = exact_order_mcmc_kernel(table, max_nodes=3)
    order_flow = order_kernel.target_probabilities[:, None] * order_kernel.transition_matrix
    order_db = float(np.max(np.abs(order_flow - order_flow.T)))
    order_stationarity = float(
        np.max(
            np.abs(
                order_kernel.target_probabilities @ order_kernel.transition_matrix
                - order_kernel.target_probabilities
            )
        )
    )
    exact_order = exact_order_dp(table)
    direct_order_logz = float(
        logsumexp(
            [
                sum(
                    logsumexp(
                        np.asarray(table.loglik[child])[
                            [
                                row
                                for row, parents in enumerate(table.parent_sets[child])
                                if all(
                                    int(parent) < 0
                                    or order.index(int(parent)) < order.index(child)
                                    for parent in parents
                                )
                            ]
                        ]
                        + np.asarray(table.logprior[child])[
                            [
                                row
                                for row, parents in enumerate(table.parent_sets[child])
                                if all(
                                    int(parent) < 0
                                    or order.index(int(parent)) < order.index(child)
                                    for parent in parents
                                )
                            ]
                        ]
                    )
                    for child in range(3)
                )
                for order in permutations(range(3))
            ]
        )
    )

    exact_structure = exact_structure_dp(table)
    partitions = tuple(
        state
        for state in enumerate_ordered_partitions(range(3))
        if np.isfinite(partition_log_target(state, table))
    )
    partition_log_targets = [partition_log_target(state, table) for state in partitions]

    def partition_proposal(source):
        values = {
            destination: partition_proposal_probability(source, destination)
            for destination in partitions
        }
        return {state: mass for state, mass in values.items() if mass > 0.0}

    partition_db, partition_stationarity = _mh_residual(
        partitions, partition_log_targets, partition_proposal
    )
    partition_logz_error = abs(logsumexp(partition_log_targets) - exact_structure.log_normalizer)
    partition_pips = exact_partition_pips(table)
    edges = {(u, v) for u in table.nodes for v in table.nodes if u != v}
    partition_pip_error = max(
        abs(partition_pips.get(edge, 0.0) - exact_structure.edge_probs.get(edge, 0.0))
        for edge in edges
    )

    dags = enumerate_candidate_dags(table)
    standard_db, standard_stationarity = _mh_residual(
        dags,
        [dag_log_target(state, table) for state in dags],
        lambda state: standard_structure_proposal_distribution(state, table),
    )
    rev_db = _rev_flow_residual(table)

    checks = [
        {
            "adapter": "exact-order-dp",
            "check": "normalizer_vs_direct_order_enumeration",
            "residual": abs(exact_order.log_normalizer - direct_order_logz),
            "tolerance": tolerance,
        },
        {"adapter": "order-mcmc", "check": "detailed_balance", "residual": order_db, "tolerance": tolerance},
        {"adapter": "order-mcmc", "check": "stationarity", "residual": order_stationarity, "tolerance": tolerance},
        {"adapter": "partition-mcmc", "check": "normalizer_vs_exact_structure", "residual": partition_logz_error, "tolerance": tolerance},
        {"adapter": "partition-mcmc", "check": "pips_vs_exact_structure", "residual": partition_pip_error, "tolerance": tolerance},
        {"adapter": "partition-mcmc", "check": "detailed_balance", "residual": partition_db, "tolerance": tolerance},
        {"adapter": "partition-mcmc", "check": "stationarity", "residual": partition_stationarity, "tolerance": tolerance},
        {"adapter": "rev-mcmc-standard-kernel", "check": "detailed_balance", "residual": standard_db, "tolerance": tolerance},
        {"adapter": "rev-mcmc-standard-kernel", "check": "stationarity", "residual": standard_stationarity, "tolerance": tolerance},
        {"adapter": "rev-mcmc-orphan-resample", "check": "conjugate_detailed_balance", "residual": rev_db, "tolerance": tolerance},
    ]
    for check in checks:
        check["status"] = "pass" if check["residual"] <= check["tolerance"] else "fail"
    report: dict[str, object] = {
        "schema": "ordersmc-adapter-validation-v3",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "all_passed": all(row["status"] == "pass" for row in checks),
        "fixture": {
            "nodes": table.nodes,
            "candidate_dag_count": len(dags),
            "positive_partition_count": len(partitions),
        },
        "checks": checks,
        "sources": {
            "partition_mcmc": "Kuipers and Moffa (2017), DOI 10.1080/01621459.2015.1133426",
            "rev_mcmc": "Grzegorczyk and Husmeier (2008), DOI 10.1007/s10994-008-5057-7",
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report
