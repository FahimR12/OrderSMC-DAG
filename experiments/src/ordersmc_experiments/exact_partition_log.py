"""Positive log-space exact structure reference using unique source layers.

Every DAG has one partition obtained by repeatedly removing all sources.
A node in a later layer must have a parent in the immediately preceding
layer. Summing these partitions counts each DAG once, with no alternating
inclusion-exclusion and no exponentiation of absolute family scores.
This deliberately small-network reference costs O(4**d) transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.special import logsumexp

from ordersmc.core import OrderCandidateTable, _validate_candidate_table


@dataclass(frozen=True)
class ExactPartitionLogResult:
    edge_probs: dict[tuple[str, str], float]
    log_normalizer: float
    state_count: int
    target: str = "candidate_restricted_structure_modular"
    arithmetic: str = "positive source-layer recursion in log space"


def exact_partition_log_dp(table: OrderCandidateTable, *, max_nodes: int = 9):
    _validate_candidate_table(table)
    d = len(table.nodes)
    if not 1 <= d <= max_nodes:
        raise ValueError(f"positive partition DP requires 1 <= d <= {max_nodes}")
    full = (1 << d) - 1
    masks, scores, indicators = [], [], []
    for child, families in enumerate(table.parent_sets):
        child_masks, child_indicators = [], []
        for row in families:
            parents = [int(p) for p in row if int(p) >= 0]
            child_masks.append(sum(1 << p for p in parents))
            child_indicators.append([float(p in parents) for p in range(d)])
        masks.append(np.asarray(child_masks, dtype=np.int64))
        indicators.append(np.asarray(child_indicators))
        scores.append(np.asarray(table.loglik[child]) + np.asarray(table.logprior[child]))

    @lru_cache(None)
    def local(child, assigned, previous):
        allowed = (masks[child] & assigned) == masks[child]
        if assigned:
            allowed &= (masks[child] & previous) != 0
        if not np.any(allowed):
            return -math.inf, np.zeros(d)
        values = scores[child][allowed]
        logz = float(logsumexp(values))
        return logz, np.exp(values - logz) @ indicators[child][allowed]

    # Each state stores mass and the conditional mean edge-indicator matrix.
    states = {(0, 0): (0.0, np.zeros((d, d)))}
    for assigned in range(full):
        previous_layers = [0] if assigned == 0 else []
        previous = assigned
        while previous:
            if (assigned, previous) in states:
                previous_layers.append(previous)
            previous = (previous - 1) & assigned
        for previous in previous_layers:
            log_mass, mean = states[(assigned, previous)]
            remaining = full ^ assigned
            child_values = {c: local(c, assigned, previous) for c in range(d) if remaining & (1 << c)}
            block = remaining
            while block:
                children = [c for c in child_values if block & (1 << c)]
                log_transition = sum(child_values[c][0] for c in children)
                if math.isfinite(log_transition):
                    incoming_mean = mean.copy()
                    for child in children:
                        incoming_mean[:, child] += child_values[child][1]
                    incoming_log = log_mass + log_transition
                    key = (assigned | block, block)
                    if key in states:
                        old_log, old_mean = states[key]
                        combined = float(np.logaddexp(old_log, incoming_log))
                        states[key] = (
                            combined,
                            math.exp(old_log - combined) * old_mean
                            + math.exp(incoming_log - combined) * incoming_mean,
                        )
                    else:
                        states[key] = (incoming_log, incoming_mean)
                block = (block - 1) & remaining
    terminal = [value for (assigned, _), value in states.items() if assigned == full]
    logz = float(logsumexp([value[0] for value in terminal]))
    means = sum(math.exp(value[0] - logz) * value[1] for value in terminal)
    if not np.isfinite(means).all() or np.min(means) < -1e-10 or np.max(means) > 1 + 1e-9:
        raise ArithmeticError("positive partition reference produced invalid edge probabilities")
    return ExactPartitionLogResult(
        {(u, v): float(means[i, j]) for i, u in enumerate(table.nodes)
         for j, v in enumerate(table.nodes) if i != j}, logz, len(states)
    )
