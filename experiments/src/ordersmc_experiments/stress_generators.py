"""Prespecified stress-generator primitives for protocol-v2 E3 and E6.

This module deliberately separates graph construction and acceptance auditing
from mixed-SEM calibration.  The E3 topology artifact is complete for the two
module-based graph families.  The E6 audit implements the frozen mathematical
acceptance rule for score records produced by a future native-score-integrated
generator.  Neither primitive is advertised as the full experiment generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal, Mapping, Sequence

import networkx as nx
import numpy as np

from .seeds import protocol_seed_record

E3ModuleFamily = Literal["order_blocks", "weak_bridges"]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class E3ModuleTopologyConfig:
    """Frozen topology configuration for the two module-based E3 families."""

    d: int
    family: E3ModuleFamily
    master_seed: int
    cell_id: str
    replicate_id: int
    experiment_id: str = "e3"
    module_count: int = 4
    within_module_mean_indegree: float = 2.0
    bridge_count: int = 3
    max_indegree: int = 3
    max_attempts: int = 100

    def __post_init__(self) -> None:
        if self.d not in {50, 100}:
            raise ValueError("protocol E3 module topologies use d in {50, 100}")
        if self.family not in {"order_blocks", "weak_bridges"}:
            raise ValueError("unsupported E3 module topology family")
        if self.experiment_id != "e3":
            raise ValueError("confirmatory E3 topologies use experiment_id='e3'")
        if self.module_count != 4:
            raise ValueError("protocol E3 fixes four balanced modules")
        if self.within_module_mean_indegree != 2.0:
            raise ValueError("protocol E3 fixes within-module mean indegree two")
        if self.bridge_count != 3:
            raise ValueError("protocol E3 fixes exactly three weak bridges")
        if self.max_indegree != 3:
            raise ValueError("protocol E3 fixes maximum indegree three")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not 0 <= self.master_seed < 10**20:
            raise ValueError("master_seed must lie in [0, 10^20)")
        if not 0 <= self.replicate_id < 10**6:
            raise ValueError("replicate_id must lie in [0, 10^6)")
        if (
            not self.cell_id
            or self.cell_id != self.cell_id.strip().lower()
            or "|" in self.cell_id
        ):
            raise ValueError("cell_id must be a canonical lowercase seed token")


@dataclass(frozen=True)
class E3ModuleTopologyArtifact:
    """Replayable DAG topology with edge-specific signal labels."""

    config: E3ModuleTopologyConfig
    dag: nx.DiGraph
    topological_order: tuple[str, ...]
    module_by_node: Mapping[str, int]
    weak_bridge_edges: tuple[tuple[str, str], ...]
    moderate_within_module_edges: tuple[tuple[str, str], ...]
    accepted_attempt: int
    rejection_log: tuple[Mapping[str, object], ...]

    def manifest(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": "ordersmc-e3-module-topology-v1",
            "config": asdict(self.config),
            "topological_order": list(self.topological_order),
            "module_by_node": dict(sorted(self.module_by_node.items())),
            "edges": [list(edge) for edge in sorted(self.dag.edges())],
            "weak_bridge_edges": [list(edge) for edge in self.weak_bridge_edges],
            "moderate_within_module_edges": [
                list(edge) for edge in self.moderate_within_module_edges
            ],
            "accepted_attempt": self.accepted_attempt,
            "rejection_log": list(self.rejection_log),
            "seed_records": {
                component: asdict(
                    protocol_seed_record(
                        self.config.master_seed,
                        experiment=self.config.experiment_id,
                        cell=self.config.cell_id,
                        replicate=self.config.replicate_id,
                        component=component,
                        attempt=self.accepted_attempt,
                    )
                )
                for component in ("order", "graph")
            },
        }
        body["artifact_fingerprint_sha256"] = _canonical_sha256(body)
        return body


def _rng(config: E3ModuleTopologyConfig, component: str, attempt: int) -> np.random.Generator:
    return protocol_seed_record(
        config.master_seed,
        experiment=config.experiment_id,
        cell=config.cell_id,
        replicate=config.replicate_id,
        component=component,
        attempt=attempt,
    ).generator()


def _balanced_modules(
    nodes: Sequence[str], graph_rng: np.random.Generator, module_count: int
) -> dict[str, int]:
    base_size, remainder = divmod(len(nodes), module_count)
    sizes = [base_size + (index < remainder) for index in range(module_count)]
    permuted_labels = [int(label) for label in graph_rng.permutation(module_count)]
    result: dict[str, int] = {}
    start = 0
    for original_label, size in enumerate(sizes):
        for node in nodes[start : start + size]:
            result[node] = permuted_labels[original_label]
        start += size
    return result


def _tree_edges_from_prufer(
    module_count: int, graph_rng: np.random.Generator
) -> tuple[tuple[int, int], ...]:
    """Draw a labelled uniform tree using its Prufer sequence."""

    sequence = [int(value) for value in graph_rng.integers(0, module_count, module_count - 2)]
    degrees = [1] * module_count
    for value in sequence:
        degrees[value] += 1
    edges: list[tuple[int, int]] = []
    for value in sequence:
        leaf = next(index for index, degree in enumerate(degrees) if degree == 1)
        edges.append(tuple(sorted((leaf, value))))
        degrees[leaf] -= 1
        degrees[value] -= 1
    remaining = [index for index, degree in enumerate(degrees) if degree == 1]
    edges.append(tuple(sorted((remaining[0], remaining[1]))))
    return tuple(sorted(edges))


def _attempt_e3_module_topology(
    config: E3ModuleTopologyConfig, attempt: int
) -> tuple[
    nx.DiGraph,
    tuple[str, ...],
    dict[str, int],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    nodes = tuple(f"X{index:03d}" for index in range(config.d))
    graph_rng = _rng(config, "graph", attempt)
    order_rng = _rng(config, "order", attempt)
    module_by_node = _balanced_modules(nodes, graph_rng, config.module_count)
    # A uniform node permutation randomly interleaves the fixed balanced module
    # memberships, as required by the order-block construction.
    order = tuple(str(node) for node in order_rng.permutation(nodes))
    position = {node: index for index, node in enumerate(order)}
    tree_edges = _tree_edges_from_prufer(config.module_count, graph_rng)

    indegree = {node: 0 for node in nodes}
    bridges: list[tuple[str, str]] = []
    for left_module, right_module in tree_edges:
        candidates: list[tuple[str, str]] = []
        left_nodes = [node for node in nodes if module_by_node[node] == left_module]
        right_nodes = [node for node in nodes if module_by_node[node] == right_module]
        for left_node in left_nodes:
            for right_node in right_nodes:
                edge = (
                    (left_node, right_node)
                    if position[left_node] < position[right_node]
                    else (right_node, left_node)
                )
                if indegree[edge[1]] < config.max_indegree:
                    candidates.append(edge)
        if not candidates:
            raise RuntimeError("no admissible cross-module bridge")
        candidates.sort()
        bridge = candidates[int(graph_rng.integers(0, len(candidates)))]
        bridges.append(bridge)
        indegree[bridge[1]] += 1

    within_edges: list[tuple[str, str]] = []
    for module in range(config.module_count):
        module_nodes = [node for node in nodes if module_by_node[node] == module]
        target_count = int(round(config.within_module_mean_indegree * len(module_nodes)))
        candidates = [
            (parent, child)
            for parent in module_nodes
            for child in module_nodes
            if position[parent] < position[child]
        ]
        priorities = graph_rng.random(len(candidates))
        prioritized = sorted(
            zip(priorities, candidates), key=lambda item: (float(item[0]), item[1])
        )
        selected_count = 0
        for _, edge in prioritized:
            if indegree[edge[1]] >= config.max_indegree:
                continue
            within_edges.append(edge)
            indegree[edge[1]] += 1
            selected_count += 1
            if selected_count == target_count:
                break
        if selected_count != target_count:
            raise RuntimeError(
                f"module {module} retained {selected_count}, expected {target_count} internal edges"
            )

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(bridges)
    graph.add_edges_from(within_edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("ordered E3 construction unexpectedly produced a cycle")
    if max(dict(graph.in_degree()).values(), default=0) > config.max_indegree:
        raise RuntimeError("E3 construction exceeded its maximum indegree")
    return (
        graph,
        order,
        module_by_node,
        tuple(sorted(bridges)),
        tuple(sorted(within_edges)),
    )


def generate_e3_module_topology(
    config: E3ModuleTopologyConfig,
) -> E3ModuleTopologyArtifact:
    """Generate the topology core for E3 order-block or weak-bridge cases."""

    rejection_log: list[dict[str, object]] = []
    for attempt in range(config.max_attempts):
        try:
            graph, order, modules, bridges, internal = _attempt_e3_module_topology(
                config, attempt
            )
        except RuntimeError as error:
            rejection_log.append(
                {
                    "attempt": attempt,
                    "reason": str(error),
                    "seed_records": {
                        component: asdict(
                            protocol_seed_record(
                                config.master_seed,
                                experiment=config.experiment_id,
                                cell=config.cell_id,
                                replicate=config.replicate_id,
                                component=component,
                                attempt=attempt,
                            )
                        )
                        for component in ("order", "graph")
                    },
                }
            )
            continue
        return E3ModuleTopologyArtifact(
            config=config,
            dag=graph,
            topological_order=order,
            module_by_node=modules,
            weak_bridge_edges=bridges,
            moderate_within_module_edges=internal,
            accepted_attempt=attempt,
            rejection_log=tuple(rejection_log),
        )
    raise RuntimeError(
        f"unable to generate E3 {config.family} topology in {config.max_attempts} attempts"
    )


def replay_e3_module_topology(
    manifest: Mapping[str, object],
) -> E3ModuleTopologyArtifact:
    """Regenerate and strictly verify an E3 topology manifest."""

    if manifest.get("schema") != "ordersmc-e3-module-topology-v1":
        raise ValueError("not an OrderSMC E3 module-topology manifest")
    expected = dict(manifest)
    fingerprint = expected.pop("artifact_fingerprint_sha256", None)
    if fingerprint != _canonical_sha256(expected):
        raise ValueError("E3 topology manifest fingerprint mismatch")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("E3 topology manifest has no configuration mapping")
    artifact = generate_e3_module_topology(E3ModuleTopologyConfig(**dict(raw_config)))
    if artifact.manifest() != dict(manifest):
        raise RuntimeError("E3 topology does not strictly reproduce its manifest")
    return artifact


@dataclass(frozen=True)
class JointSignalAttemptRecord:
    """Native-score audit values for one E6 child-parameter attempt."""

    child: str
    designated_parents: tuple[str, str]
    attempt: int
    null_score: float
    one_parent_scores: tuple[float, float]
    two_parent_score: float
    marginal_parent_ranks: tuple[int, int]
    pair_rescue_rank: int
    moderate_signal_target_met: bool
    designated_parents_are_true: bool = True
    calibration_rows: int = 5_000

    def __post_init__(self) -> None:
        if len(set(self.designated_parents)) != 2 or self.child in self.designated_parents:
            raise ValueError("a challenge child requires two distinct designated parents")
        if not 0 <= self.attempt < 100:
            raise ValueError("E6 child attempts must lie in [0, 100)")
        values = (self.null_score, *self.one_parent_scores, self.two_parent_score)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("E6 native scores must be finite")
        if any(rank < 1 for rank in (*self.marginal_parent_ranks, self.pair_rescue_rank)):
            raise ValueError("screening ranks are one-based positive integers")
        if self.calibration_rows != 5_000:
            raise ValueError("protocol E6 fixes a discarded 5,000-row calibration bank")

    @property
    def one_parent_improvements_per_row(self) -> tuple[float, float]:
        return tuple(
            (score - self.null_score) / self.calibration_rows
            for score in self.one_parent_scores
        )

    @property
    def two_parent_improvement_per_row(self) -> float:
        return (self.two_parent_score - self.null_score) / self.calibration_rows

    def rejection_reasons(self) -> tuple[str, ...]:
        pair_improvement = self.two_parent_improvement_per_row
        reasons: list[str] = []
        if pair_improvement <= 0.0:
            reasons.append("two_parent_improvement_not_positive")
        if any(
            improvement > 0.25 * pair_improvement
            for improvement in self.one_parent_improvements_per_row
        ):
            reasons.append("one_parent_improvement_above_quarter_pair_improvement")
        if not self.moderate_signal_target_met:
            reasons.append("moderate_signal_target_not_met")
        if not self.designated_parents_are_true:
            reasons.append("designated_parents_not_true_in_frozen_dag")
        # A marginal screen retains a pair only when both parents are in its top
        # eight; at least one designated parent must therefore fall below it.
        if max(self.marginal_parent_ranks) <= 8:
            reasons.append("designated_pair_not_hidden_from_top8_marginal_screen")
        if self.pair_rescue_rank > 8:
            reasons.append("designated_pair_not_in_top8_pair_rescue")
        return tuple(reasons)

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons()

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "one_parent_improvements_per_row": list(
                self.one_parent_improvements_per_row
            ),
            "two_parent_improvement_per_row": self.two_parent_improvement_per_row,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons()),
        }


@dataclass(frozen=True)
class JointSignalChallengeAudit:
    """First-accepted-attempt audit for the four-child E6 requirement."""

    selected_attempts: tuple[JointSignalAttemptRecord, ...]
    rejected_attempts: tuple[JointSignalAttemptRecord, ...]
    required_children: int

    @property
    def passed(self) -> bool:
        return len(self.selected_attempts) >= self.required_children

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": "ordersmc-e6-joint-signal-audit-v1",
            "required_children": self.required_children,
            "passed": self.passed,
            "selected_attempts": [record.to_dict() for record in self.selected_attempts],
            "rejected_attempts": [record.to_dict() for record in self.rejected_attempts],
        }
        body["audit_fingerprint_sha256"] = _canonical_sha256(body)
        return body


def audit_joint_signal_challenge(
    records: Sequence[JointSignalAttemptRecord],
    *,
    required_children: int = 4,
) -> JointSignalChallengeAudit:
    """Apply E6's frozen acceptance rule without inspecting sampler output.

    Records are grouped by child and processed in increasing attempt order.  A
    child contributes its first accepted attempt; later records for that child
    are forbidden because the generator must stop redrawing it immediately.
    """

    if required_children < 4:
        raise ValueError("protocol E6 requires at least four challenge children")
    by_child: dict[str, list[JointSignalAttemptRecord]] = {}
    for record in records:
        by_child.setdefault(record.child, []).append(record)
    selected: list[JointSignalAttemptRecord] = []
    rejected: list[JointSignalAttemptRecord] = []
    for child in sorted(by_child):
        child_records = sorted(by_child[child], key=lambda record: record.attempt)
        attempts = [record.attempt for record in child_records]
        if len(attempts) != len(set(attempts)):
            raise ValueError(f"duplicate E6 attempt for child {child}")
        if attempts and attempts != list(range(attempts[-1] + 1)):
            raise ValueError(f"E6 attempts for child {child} must be contiguous from zero")
        accepted_index: int | None = None
        for index, record in enumerate(child_records):
            if record.accepted:
                accepted_index = index
                break
            rejected.append(record)
        if accepted_index is not None:
            if accepted_index != len(child_records) - 1:
                raise ValueError(
                    f"child {child} contains attempts after its first accepted attempt"
                )
            selected.append(child_records[accepted_index])
    return JointSignalChallengeAudit(
        selected_attempts=tuple(selected),
        rejected_attempts=tuple(rejected),
        required_children=required_children,
    )
