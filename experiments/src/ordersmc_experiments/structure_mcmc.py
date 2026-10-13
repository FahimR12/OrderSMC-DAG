"""Validated structure-target adapters for Partition MCMC and REV-MCMC.

The implementations operate on the exact same candidate table as OrderSMC.
Partition MCMC uses the unique source-layer ordered-partition representation of
Kuipers and Moffa (2017). REV-MCMC combines ordinary add/delete/reverse moves
with the orphan-and-resample reversal move of Grzegorczyk and Husmeier (2008).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import time
from typing import Dict, Iterable, Iterator, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from ordersmc.core import OrderCandidateTable

from .order_mcmc import _batch_means_mcse, _lag1_autocorrelation


DAGState = tuple[int, ...]
PartitionState = tuple[tuple[int, ...], ...]
Edge = tuple[str, str]


def _families(table: OrderCandidateTable) -> tuple[tuple[tuple[int, ...], ...], ...]:
    result: list[tuple[tuple[int, ...], ...]] = []
    for rows in table.parent_sets:
        child_rows = []
        for row in np.asarray(rows, dtype=int):
            child_rows.append(tuple(sorted(int(value) for value in row if int(value) >= 0)))
        result.append(tuple(child_rows))
    return tuple(result)


def _local_scores(table: OrderCandidateTable) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(loglik, dtype=float) + np.asarray(logprior, dtype=float)
        for loglik, logprior in zip(table.loglik, table.logprior)
    )


def _family_indices(table: OrderCandidateTable) -> tuple[dict[tuple[int, ...], int], ...]:
    return tuple(
        {parents: index for index, parents in enumerate(rows)}
        for rows in _families(table)
    )


def validate_structure_candidate_table(table: OrderCandidateTable) -> None:
    """Require finite, duplicate-free, downward-closed candidate families."""

    node_count = len(table.nodes)
    if node_count == 0:
        raise ValueError("candidate table must contain nodes")
    families = _families(table)
    scores = _local_scores(table)
    if len(families) != node_count or len(scores) != node_count:
        raise ValueError("candidate table child dimensions are inconsistent")
    for child, (rows, values) in enumerate(zip(families, scores)):
        if len(rows) != len(set(rows)):
            raise ValueError(f"child {child} contains duplicate parent sets")
        if len(rows) != len(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"child {child} has invalid local scores")
        available = set(rows)
        if () not in available:
            raise ValueError(f"child {child} has no empty parent set")
        for parents in rows:
            if child in parents or any(parent < 0 or parent >= node_count for parent in parents):
                raise ValueError(f"child {child} has an invalid parent index")
            for removed in parents:
                subset = tuple(parent for parent in parents if parent != removed)
                if subset not in available:
                    raise ValueError(
                        f"candidate support is not downward closed for child {child}: "
                        f"missing {subset} below {parents}"
                    )


def empty_dag_state(table: OrderCandidateTable) -> DAGState:
    indices = _family_indices(table)
    return tuple(child[()] for child in indices)


def dag_state_edges(state: DAGState, table: OrderCandidateTable) -> tuple[tuple[int, int], ...]:
    families = _families(table)
    return tuple(
        sorted(
            (parent, child)
            for child, row_index in enumerate(state)
            for parent in families[child][row_index]
        )
    )


def dag_state_graph(state: DAGState, table: OrderCandidateTable) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(table.nodes)))
    graph.add_edges_from(dag_state_edges(state, table))
    return graph


def dag_state_is_valid(state: DAGState, table: OrderCandidateTable) -> bool:
    if len(state) != len(table.nodes):
        return False
    if any(index < 0 or index >= len(table.parent_sets[child]) for child, index in enumerate(state)):
        return False
    return nx.is_directed_acyclic_graph(dag_state_graph(state, table))


def dag_log_target(state: DAGState, table: OrderCandidateTable) -> float:
    if not dag_state_is_valid(state, table):
        return float("-inf")
    scores = _local_scores(table)
    return float(sum(scores[child][row] for child, row in enumerate(state)))


def enumerate_candidate_dags(
    table: OrderCandidateTable, *, max_nodes: int = 6
) -> tuple[DAGState, ...]:
    if len(table.nodes) > max_nodes:
        raise ValueError(f"DAG enumeration is limited to {max_nodes} nodes")
    return tuple(
        state
        for state in product(*(range(len(rows)) for rows in table.parent_sets))
        if dag_state_is_valid(tuple(int(value) for value in state), table)
    )


def canonical_partition(state: DAGState, table: OrderCandidateTable) -> PartitionState:
    """Return the unique iterative-source partition of a candidate DAG."""

    graph = dag_state_graph(state, table)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("state is not a DAG")
    remaining = set(graph.nodes())
    blocks: list[tuple[int, ...]] = []
    while remaining:
        block = tuple(
            sorted(node for node in remaining if not (set(graph.predecessors(node)) & remaining))
        )
        if not block:
            raise RuntimeError("source layering failed for an acyclic graph")
        blocks.append(block)
        remaining.difference_update(block)
    return tuple(blocks)


def _partition_allowed_rows(
    partition: PartitionState, table: OrderCandidateTable
) -> tuple[tuple[int, ...], ...]:
    node_count = len(table.nodes)
    flattened = [node for block in partition for node in block]
    if sorted(flattened) != list(range(node_count)) or any(not block for block in partition):
        raise ValueError("partition must contain every node exactly once in nonempty blocks")
    families = _families(table)
    predecessors: set[int] = set()
    allowed: list[tuple[int, ...] | None] = [None] * node_count
    for block_index, block in enumerate(partition):
        previous = set(partition[block_index - 1]) if block_index else set()
        for child in block:
            rows = tuple(
                row_index
                for row_index, parents in enumerate(families[child])
                if set(parents) <= predecessors
                and (block_index == 0 or bool(set(parents) & previous))
            )
            allowed[child] = rows
        predecessors.update(block)
    return tuple(row if row is not None else () for row in allowed)


def partition_log_target(partition: PartitionState, table: OrderCandidateTable) -> float:
    scores = _local_scores(table)
    allowed = _partition_allowed_rows(partition, table)
    if any(not rows for rows in allowed):
        return float("-inf")
    return float(sum(logsumexp(scores[child][list(rows)]) for child, rows in enumerate(allowed)))


def partition_conditional_edge_probs(
    partition: PartitionState, table: OrderCandidateTable
) -> dict[Edge, float]:
    scores = _local_scores(table)
    families = _families(table)
    allowed = _partition_allowed_rows(partition, table)
    result: dict[Edge, float] = {}
    for child, rows in enumerate(allowed):
        if not rows:
            raise ValueError("partition has zero structure-target mass")
        values = scores[child][list(rows)]
        weights = np.exp(values - logsumexp(values))
        for row_index, weight in zip(rows, weights):
            for parent in families[child][row_index]:
                edge = (table.nodes[parent], table.nodes[child])
                result[edge] = result.get(edge, 0.0) + float(weight)
    return result


def _best_dag_in_partition(partition: PartitionState, table: OrderCandidateTable) -> tuple[DAGState, float]:
    scores = _local_scores(table)
    allowed = _partition_allowed_rows(partition, table)
    if any(not rows for rows in allowed):
        raise ValueError("partition has zero structure-target mass")
    state = tuple(
        min(rows, key=lambda index: (-float(scores[child][index]), index))
        for child, rows in enumerate(allowed)
    )
    return state, dag_log_target(state, table)


def _partition_move_probabilities(
    partition: PartitionState, split_probability: float
) -> tuple[float, float, tuple[int, ...]]:
    splitable = tuple(index for index, block in enumerate(partition) if len(block) >= 2)
    can_split = bool(splitable)
    can_merge = len(partition) >= 2
    if can_split and can_merge:
        return split_probability, 1.0 - split_probability, splitable
    if can_split:
        return 1.0, 0.0, splitable
    if can_merge:
        return 0.0, 1.0, splitable
    return 0.0, 0.0, splitable


def _log_nontrivial_binary_splits(size: int) -> float:
    if size < 2:
        raise ValueError("a split requires at least two nodes")
    return size * math.log(2.0) + math.log1p(-2.0 ** (1 - size))


def partition_proposal_probability(
    source: PartitionState,
    destination: PartitionState,
    *,
    split_probability: float = 0.5,
) -> float:
    """Exact split/merge proposal mass for one ordered-partition transition."""

    p_split, p_merge, splitable = _partition_move_probabilities(source, split_probability)
    probability = 0.0
    # Merge one adjacent boundary.
    if p_merge > 0.0:
        for boundary in range(len(source) - 1):
            candidate = (
                source[:boundary]
                + (tuple(sorted(source[boundary] + source[boundary + 1])),)
                + source[boundary + 2 :]
            )
            if candidate == destination:
                probability += p_merge / (len(source) - 1)
    # Split one block into an ordered nonempty pair.
    if p_split > 0.0:
        for block_index in splitable:
            block = source[block_index]
            if len(destination) != len(source) + 1:
                continue
            if source[:block_index] != destination[:block_index] or source[block_index + 1 :] != destination[block_index + 2 :]:
                continue
            left, right = destination[block_index], destination[block_index + 1]
            if left and right and set(left).isdisjoint(right) and set(left) | set(right) == set(block):
                probability += p_split / len(splitable) * math.exp(
                    -_log_nontrivial_binary_splits(len(block))
                )
    return float(probability)


def _sample_partition_proposal(
    partition: PartitionState,
    rng: np.random.Generator,
    split_probability: float,
) -> tuple[PartitionState, float, float, str]:
    p_split, p_merge, splitable = _partition_move_probabilities(partition, split_probability)
    if p_split == p_merge == 0.0:
        return partition, 0.0, 0.0, "self"
    choose_split = p_merge == 0.0 or (p_split > 0.0 and rng.random() < p_split)
    if choose_split:
        block_index = int(splitable[int(rng.integers(0, len(splitable)))])
        block = partition[block_index]
        while True:
            mask = rng.integers(0, 2, size=len(block), dtype=np.int8)
            if np.any(mask == 0) and np.any(mask == 1):
                break
        left = tuple(node for node, bit in zip(block, mask) if bit == 0)
        right = tuple(node for node, bit in zip(block, mask) if bit == 1)
        proposal = partition[:block_index] + (left, right) + partition[block_index + 1 :]
        log_forward = math.log(p_split) - math.log(len(splitable)) - _log_nontrivial_binary_splits(len(block))
        reverse = partition_proposal_probability(proposal, partition, split_probability=split_probability)
        return proposal, log_forward, math.log(reverse), "split"
    boundary = int(rng.integers(0, len(partition) - 1))
    merged = tuple(sorted(partition[boundary] + partition[boundary + 1]))
    proposal = partition[:boundary] + (merged,) + partition[boundary + 2 :]
    log_forward = math.log(p_merge) - math.log(len(partition) - 1)
    reverse = partition_proposal_probability(proposal, partition, split_probability=split_probability)
    return proposal, log_forward, math.log(reverse), "merge"


@dataclass(frozen=True)
class PartitionMCMCConfig:
    iterations: int = 50_000
    burnin: int = 10_000
    thin: int = 10
    split_probability: float = 0.5
    seed: int = 0
    wall_seconds: float | None = None
    wall_burnin_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.burnin < 0 or self.burnin >= self.iterations:
            raise ValueError("require iterations >= 1 and 0 <= burnin < iterations")
        if self.thin < 1 or not 0.0 < self.split_probability < 1.0 or self.seed < 0:
            raise ValueError("invalid Partition-MCMC configuration")
        if self.wall_seconds is not None and self.wall_seconds <= 0.0:
            raise ValueError("wall_seconds must be positive")
        if not 0.0 <= self.wall_burnin_fraction < 1.0:
            raise ValueError("wall_burnin_fraction must lie in [0, 1)")


@dataclass
class StructureMCMCResult:
    method: Literal["partition-mcmc", "rev-mcmc"]
    edge_probs: Dict[Edge, float]
    edge_mcse: Dict[Edge, float]
    best_edges: list[Edge]
    best_log_target: float
    acceptance_rate: float
    acceptance_by_proposal: Dict[str, float]
    retained_samples: int
    stored_samples: int
    effective_sample_size: float
    runtime_seconds: float
    completed_iterations: int
    target: Literal["candidate_restricted_structure_modular"] = (
        "candidate_restricted_structure_modular"
    )
    target_label: Literal["structure"] = "structure"


def _summarize_edge_series(
    series: Mapping[Edge, Sequence[float]], retained: int
) -> tuple[dict[Edge, float], dict[Edge, float], float]:
    probabilities: dict[Edge, float] = {}
    mcse: dict[Edge, float] = {}
    ess_values: list[float] = []
    for edge, values in series.items():
        array = np.asarray(values, dtype=float)
        probabilities[edge] = float(np.mean(array)) if len(array) else 0.0
        edge_mcse, edge_ess = _batch_means_mcse(array)
        mcse[edge] = edge_mcse
        if np.isfinite(edge_ess):
            ess_values.append(edge_ess)
    return probabilities, mcse, min(ess_values, default=float(retained))


def partition_mcmc(
    table: OrderCandidateTable,
    config: PartitionMCMCConfig = PartitionMCMCConfig(),
) -> StructureMCMCResult:
    validate_structure_candidate_table(table)
    rng = np.random.Generator(np.random.PCG64DXSM(config.seed))
    partition: PartitionState = (tuple(range(len(table.nodes))),)
    target = partition_log_target(partition, table)
    if not np.isfinite(target):
        raise RuntimeError("empty-DAG partition unexpectedly has zero mass")
    best_state, best_score = _best_dag_in_partition(partition, table)
    edge_universe = tuple(
        sorted(
            {
            (table.nodes[parent], table.nodes[child])
            for child, rows in enumerate(_families(table))
            for parents in rows
            for parent in parents
            }
        )
    )
    edge_series: dict[Edge, list[float]] = {edge: [] for edge in edge_universe}
    proposed = {"split": 0, "merge": 0}
    accepted = {"split": 0, "merge": 0}
    retained = 0
    stored = 0
    started = time.perf_counter()
    deadline = started + config.wall_seconds if config.wall_seconds is not None else None
    burnin_deadline = (
        started + config.wall_seconds * config.wall_burnin_fraction
        if config.wall_seconds is not None
        else None
    )
    target_cache: dict[PartitionState, float] = {partition: target}
    completed = 0
    for iteration in range(config.iterations):
        now = time.perf_counter()
        if deadline is not None and now >= deadline:
            break
        proposal, log_q_forward, log_q_reverse, move = _sample_partition_proposal(
            partition, rng, config.split_probability
        )
        if move != "self":
            proposed[move] += 1
            proposal_target = target_cache.get(proposal)
            if proposal_target is None:
                proposal_target = partition_log_target(proposal, table)
                target_cache[proposal] = proposal_target
            log_acceptance = proposal_target - target + log_q_reverse - log_q_forward
            if np.log(rng.random()) < min(0.0, log_acceptance):
                partition, target = proposal, proposal_target
                accepted[move] += 1
        completed = iteration + 1
        burnin_complete = (
            now >= burnin_deadline
            if burnin_deadline is not None
            else iteration >= config.burnin
        )
        if burnin_complete:
            retained += 1
            if retained % config.thin == 0:
                conditional = partition_conditional_edge_probs(partition, table)
                for edge in edge_universe:
                    edge_series[edge].append(float(conditional.get(edge, 0.0)))
                stored += 1
                candidate, score = _best_dag_in_partition(partition, table)
                if score > best_score or (score == best_score and candidate < best_state):
                    best_state, best_score = candidate, score
    if stored == 0:
        raise RuntimeError("Partition MCMC stored no post-burn-in samples")
    edge_probs, edge_mcse, ess = _summarize_edge_series(edge_series, retained)
    best_edges = [(table.nodes[u], table.nodes[v]) for u, v in dag_state_edges(best_state, table)]
    return StructureMCMCResult(
        method="partition-mcmc",
        edge_probs=edge_probs,
        edge_mcse=edge_mcse,
        best_edges=best_edges,
        best_log_target=best_score,
        acceptance_rate=sum(accepted.values()) / max(1, sum(proposed.values())),
        acceptance_by_proposal={
            move: accepted[move] / proposed[move] if proposed[move] else float("nan")
            for move in proposed
        },
        retained_samples=retained,
        stored_samples=stored,
        effective_sample_size=ess,
        runtime_seconds=time.perf_counter() - started,
        completed_iterations=completed,
    )


def _replace_parent_set(
    state: DAGState,
    child: int,
    parents: Iterable[int],
    indices: Sequence[Mapping[tuple[int, ...], int]],
) -> DAGState | None:
    key = tuple(sorted(int(parent) for parent in parents))
    row = indices[child].get(key)
    if row is None:
        return None
    values = list(state)
    values[child] = row
    return tuple(values)


def _standard_structure_proposal(
    state: DAGState,
    table: OrderCandidateTable,
    rng: np.random.Generator,
) -> tuple[DAGState, float, float, str]:
    n = len(table.nodes)
    if n < 2:
        return state, 0.0, 0.0, "self"
    families = _families(table)
    indices = _family_indices(table)
    u = int(rng.integers(0, n))
    v_raw = int(rng.integers(0, n - 1))
    v = v_raw + (v_raw >= u)
    parents_v = set(families[v][state[v]])
    parents_u = set(families[u][state[u]])
    if u in parents_v:
        if rng.random() < 0.5:
            proposal = _replace_parent_set(state, v, parents_v - {u}, indices)
            move = "delete"
            log_ratio = math.log(2.0)  # reverse addition has twice the proposal mass
        else:
            interim = _replace_parent_set(state, v, parents_v - {u}, indices)
            proposal = (
                None
                if interim is None
                else _replace_parent_set(interim, u, parents_u | {v}, indices)
            )
            move = "reverse"
            log_ratio = 0.0
    elif v in parents_u:
        return state, 0.0, 0.0, "self"
    else:
        proposal = _replace_parent_set(state, v, parents_v | {u}, indices)
        move = "add"
        log_ratio = -math.log(2.0)  # reverse deletion has half the proposal mass
    if proposal is None or not dag_state_is_valid(proposal, table):
        return state, 0.0, 0.0, "self"
    return proposal, 0.0, log_ratio, move


def standard_structure_proposal_distribution(
    state: DAGState, table: OrderCandidateTable
) -> dict[DAGState, float]:
    """Enumerate the exact ADR proposal kernel for tiny acceptance tests."""

    n = len(table.nodes)
    if n < 2:
        return {state: 1.0}
    families = _families(table)
    indices = _family_indices(table)
    mass = 1.0 / (n * (n - 1))
    result: dict[DAGState, float] = {}

    def add(candidate: DAGState | None, probability: float) -> None:
        destination = (
            candidate
            if candidate is not None and dag_state_is_valid(candidate, table)
            else state
        )
        result[destination] = result.get(destination, 0.0) + probability

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            parents_v = set(families[v][state[v]])
            parents_u = set(families[u][state[u]])
            if u in parents_v:
                add(_replace_parent_set(state, v, parents_v - {u}, indices), 0.5 * mass)
                interim = _replace_parent_set(state, v, parents_v - {u}, indices)
                reversal = (
                    None
                    if interim is None
                    else _replace_parent_set(interim, u, parents_u | {v}, indices)
                )
                add(reversal, 0.5 * mass)
            elif v in parents_u:
                add(None, mass)
            else:
                add(_replace_parent_set(state, v, parents_v | {u}, indices), mass)
    return result


def _allowed_rows_after_orphaning(
    base: DAGState,
    child: int,
    table: OrderCandidateTable,
    *,
    required_parent: int | None,
) -> tuple[int, ...]:
    """Return parent rows that preserve acyclicity after orphaning ``child``.

    ``base`` is already a DAG with the child's incoming edges removed.  Adding
    parents can create a cycle exactly when a proposed parent is reachable from
    the child.  Computing descendants once is equivalent to rebuilding and
    topologically checking one graph per candidate row, and is orders of
    magnitude cheaper for screened tables with thousands of families.
    """

    families = _families(table)
    graph = dag_state_graph(base, table)
    if not nx.is_directed_acyclic_graph(graph):
        return ()
    descendants = nx.descendants(graph, child)
    result = []
    for row, parents in enumerate(families[child]):
        if required_parent is not None and required_parent not in parents:
            continue
        if not descendants.intersection(parents):
            result.append(row)
    return tuple(result)


def _sample_logweights(values: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = np.exp(values - logsumexp(values))
    return int(rng.choice(len(values), p=probabilities))


def rev_log_proposal_probability(
    source: DAGState,
    destination: DAGState,
    table: OrderCandidateTable,
    edge: tuple[int, int],
) -> float:
    """Log proposal mass of one published orphan-and-resample REV move."""

    u, v = edge
    source_edges = dag_state_edges(source, table)
    if edge not in source_edges or (v, u) not in dag_state_edges(destination, table):
        return float("-inf")
    empty = empty_dag_state(table)
    base = list(source)
    base[u] = empty[u]
    base[v] = empty[v]
    target_base = list(destination)
    target_base[u] = empty[u]
    target_base[v] = empty[v]
    if tuple(base) != tuple(target_base):
        return float("-inf")
    base_state = tuple(base)
    scores = _local_scores(table)
    allowed_u = _allowed_rows_after_orphaning(base_state, u, table, required_parent=v)
    if destination[u] not in allowed_u:
        return float("-inf")
    with_u = list(base_state)
    with_u[u] = destination[u]
    allowed_v = _allowed_rows_after_orphaning(tuple(with_u), v, table, required_parent=None)
    if destination[v] not in allowed_v:
        return float("-inf")
    return float(
        -math.log(len(source_edges))
        + scores[u][destination[u]]
        - logsumexp(scores[u][list(allowed_u)])
        + scores[v][destination[v]]
        - logsumexp(scores[v][list(allowed_v)])
    )


def _sample_rev_proposal(
    state: DAGState,
    table: OrderCandidateTable,
    rng: np.random.Generator,
) -> tuple[DAGState, float, float, str]:
    edges = dag_state_edges(state, table)
    if not edges:
        return state, 0.0, 0.0, "self"
    u, v = edges[int(rng.integers(0, len(edges)))]
    empty = empty_dag_state(table)
    base = list(state)
    base[u] = empty[u]
    base[v] = empty[v]
    scores = _local_scores(table)
    allowed_u = _allowed_rows_after_orphaning(tuple(base), u, table, required_parent=v)
    if not allowed_u:
        return state, 0.0, 0.0, "self"
    row_u = allowed_u[_sample_logweights(scores[u][list(allowed_u)], rng)]
    base[u] = row_u
    allowed_v = _allowed_rows_after_orphaning(tuple(base), v, table, required_parent=None)
    if not allowed_v:
        return state, 0.0, 0.0, "self"
    row_v = allowed_v[_sample_logweights(scores[v][list(allowed_v)], rng)]
    base[v] = row_v
    proposal = tuple(base)
    log_forward = rev_log_proposal_probability(state, proposal, table, (u, v))
    log_reverse = rev_log_proposal_probability(proposal, state, table, (v, u))
    if not np.isfinite(log_forward) or not np.isfinite(log_reverse):
        raise RuntimeError("REV proposal lost its conjugate reverse move")
    return proposal, log_forward, log_reverse, "rev"


def rev_proposal_distribution(
    state: DAGState, table: OrderCandidateTable
) -> dict[DAGState, float]:
    """Enumerate the sequential REV proposal kernel for tiny acceptance tests."""

    edges = dag_state_edges(state, table)
    if not edges:
        return {state: 1.0}
    empty = empty_dag_state(table)
    scores = _local_scores(table)
    result: dict[DAGState, float] = {}
    edge_mass = 1.0 / len(edges)
    for u, v in edges:
        base = list(state)
        base[u] = empty[u]
        base[v] = empty[v]
        allowed_u = _allowed_rows_after_orphaning(tuple(base), u, table, required_parent=v)
        if not allowed_u:
            result[state] = result.get(state, 0.0) + edge_mass
            continue
        weights_u = np.exp(
            scores[u][list(allowed_u)] - logsumexp(scores[u][list(allowed_u)])
        )
        for row_u, weight_u in zip(allowed_u, weights_u):
            with_u = list(base)
            with_u[u] = row_u
            allowed_v = _allowed_rows_after_orphaning(
                tuple(with_u), v, table, required_parent=None
            )
            if not allowed_v:
                result[state] = result.get(state, 0.0) + edge_mass * float(weight_u)
                continue
            weights_v = np.exp(
                scores[v][list(allowed_v)] - logsumexp(scores[v][list(allowed_v)])
            )
            for row_v, weight_v in zip(allowed_v, weights_v):
                destination = list(with_u)
                destination[v] = row_v
                key = tuple(destination)
                result[key] = result.get(key, 0.0) + edge_mass * float(weight_u * weight_v)
    return result


@dataclass(frozen=True)
class REVMCMCConfig:
    iterations: int = 100_000
    burnin: int = 20_000
    thin: int = 20
    rev_probability: float = 1.0 / 15.0
    seed: int = 0
    wall_seconds: float | None = None
    wall_burnin_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.burnin < 0 or self.burnin >= self.iterations:
            raise ValueError("require iterations >= 1 and 0 <= burnin < iterations")
        if self.thin < 1 or not 0.0 <= self.rev_probability < 1.0 or self.seed < 0:
            raise ValueError("invalid REV-MCMC configuration")
        if self.wall_seconds is not None and self.wall_seconds <= 0.0:
            raise ValueError("wall_seconds must be positive")
        if not 0.0 <= self.wall_burnin_fraction < 1.0:
            raise ValueError("wall_burnin_fraction must lie in [0, 1)")


def rev_mcmc(
    table: OrderCandidateTable,
    config: REVMCMCConfig = REVMCMCConfig(),
) -> StructureMCMCResult:
    validate_structure_candidate_table(table)
    rng = np.random.Generator(np.random.PCG64DXSM(config.seed))
    state = empty_dag_state(table)
    target = dag_log_target(state, table)
    best_state, best_score = state, target
    edge_universe = tuple(
        sorted(
            {
            (table.nodes[parent], table.nodes[child])
            for child, rows in enumerate(_families(table))
            for parents in rows
            for parent in parents
            }
        )
    )
    edge_series: dict[Edge, list[float]] = {edge: [] for edge in edge_universe}
    proposed = {"add": 0, "delete": 0, "reverse": 0, "rev": 0}
    accepted = {name: 0 for name in proposed}
    retained = stored = completed = 0
    started = time.perf_counter()
    deadline = started + config.wall_seconds if config.wall_seconds is not None else None
    burnin_deadline = (
        started + config.wall_seconds * config.wall_burnin_fraction
        if config.wall_seconds is not None
        else None
    )
    for iteration in range(config.iterations):
        now = time.perf_counter()
        if deadline is not None and now >= deadline:
            break
        if rng.random() < config.rev_probability:
            proposal, log_forward, log_reverse, move = _sample_rev_proposal(state, table, rng)
        else:
            proposal, log_forward, log_reverse, move = _standard_structure_proposal(state, table, rng)
        if move != "self":
            proposed[move] += 1
            proposal_target = dag_log_target(proposal, table)
            log_acceptance = proposal_target - target + log_reverse - log_forward
            if np.log(rng.random()) < min(0.0, log_acceptance):
                state, target = proposal, proposal_target
                accepted[move] += 1
                if target > best_score or (target == best_score and state < best_state):
                    best_state, best_score = state, target
        completed = iteration + 1
        burnin_complete = (
            now >= burnin_deadline
            if burnin_deadline is not None
            else iteration >= config.burnin
        )
        if burnin_complete:
            retained += 1
            if retained % config.thin == 0:
                present = {
                    (table.nodes[u], table.nodes[v]) for u, v in dag_state_edges(state, table)
                }
                for edge in edge_universe:
                    edge_series[edge].append(float(edge in present))
                stored += 1
    if stored == 0:
        raise RuntimeError("REV-MCMC stored no post-burn-in samples")
    edge_probs, edge_mcse, ess = _summarize_edge_series(edge_series, retained)
    return StructureMCMCResult(
        method="rev-mcmc",
        edge_probs=edge_probs,
        edge_mcse=edge_mcse,
        best_edges=[(table.nodes[u], table.nodes[v]) for u, v in dag_state_edges(best_state, table)],
        best_log_target=best_score,
        acceptance_rate=sum(accepted.values()) / max(1, sum(proposed.values())),
        acceptance_by_proposal={
            move: accepted[move] / proposed[move] if proposed[move] else float("nan")
            for move in proposed
        },
        retained_samples=retained,
        stored_samples=stored,
        effective_sample_size=ess,
        runtime_seconds=time.perf_counter() - started,
        completed_iterations=completed,
    )


def enumerate_ordered_partitions(nodes: Sequence[int]) -> tuple[PartitionState, ...]:
    """Enumerate labelled ordered partitions for tiny acceptance tests."""

    values = tuple(nodes)
    if not values:
        return ()
    result: set[PartitionState] = set()
    # A surjection to k ordered labels represents one ordered partition.
    for k in range(1, len(values) + 1):
        for labels in product(range(k), repeat=len(values)):
            if set(labels) != set(range(k)):
                continue
            result.add(
                tuple(
                    tuple(sorted(node for node, label in zip(values, labels) if label == block))
                    for block in range(k)
                )
            )
    return tuple(sorted(result))


def exact_partition_pips(table: OrderCandidateTable, *, max_nodes: int = 6) -> dict[Edge, float]:
    if len(table.nodes) > max_nodes:
        raise ValueError(f"partition enumeration is limited to {max_nodes} nodes")
    partitions = enumerate_ordered_partitions(range(len(table.nodes)))
    log_targets = np.asarray([partition_log_target(row, table) for row in partitions])
    finite = np.isfinite(log_targets)
    weights = np.zeros(len(partitions), dtype=float)
    weights[finite] = np.exp(log_targets[finite] - logsumexp(log_targets[finite]))
    result: dict[Edge, float] = {}
    for partition, weight in zip(partitions, weights):
        if weight == 0.0:
            continue
        for edge, probability in partition_conditional_edge_probs(partition, table).items():
            result[edge] = result.get(edge, 0.0) + float(weight * probability)
    return result
