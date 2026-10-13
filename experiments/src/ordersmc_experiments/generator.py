"""Fully specified mixed-family synthetic DAG generator.

Artifact schema v2 is retained for byte-compatible replay; protocol Version 3
changes the cohort design, not this validated serialization format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import binom

from .seeds import SeedRecord, protocol_seed_record

VariableType = Literal["con", "bin", "cat"]
VariableTypeRegime = Literal[
    "balanced_mixed", "continuous_dominant_mixed", "discrete_dominant_mixed"
]
ContinuousNoise = Literal[
    "gaussian", "student_t_df3", "student_t_df5", "centered_gamma_shape1_scale2"
]
GraphFamily = Literal[
    "random_order", "hub", "modular", "empty", "order_blocks", "weak_bridges"
]
SignalStrength = Literal["weak", "moderate", "strong"]

_E1_EXPECTED_INDEGREE = 1.5
_E1_RANDOM_ORDER_INDEGREE_INTERVAL = (1.0, 1.8)

_R2_RANGES: Mapping[SignalStrength, tuple[float, float]] = {
    "weak": (0.10, 0.20),
    "moderate": (0.20, 0.40),
    "strong": (0.40, 0.60),
}
_ENTROPY_RANGES: Mapping[SignalStrength, tuple[float, float]] = {
    "weak": (0.80, 0.90),
    "moderate": (0.60, 0.80),
    "strong": (0.40, 0.60),
}


class GeneratorFailure(RuntimeError):
    """Explicit generator failure carrying the complete rejection audit."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class MixedDAGConfig:
    """Frozen data-generating configuration.

    ``n_train`` is always the inferential training size. ``n_test`` is an
    independently generated hold-out set and is never removed from
    ``n_train``.
    """

    d: int = 20
    n_train: int = 400
    n_test: int | None = None
    graph_family: GraphFamily = "random_order"
    fixed_topology_id: str | None = None
    expected_indegree: float = 2.0
    max_indegree: int = 3
    signal_strength: SignalStrength = "moderate"
    continuous_noise: ContinuousNoise = "gaussian"
    cross_type_minimum: float = 0.50
    require_all_cross_type_directions: bool = True
    variable_type_regime: VariableTypeRegime = "balanced_mixed"
    categorical_levels: tuple[int, ...] = (3, 4)
    residual_sd: float = 1.0
    pilot_size: int = 5_000
    min_category_marginal: float = 0.05
    min_binary_marginal: float = 0.05
    min_median_entropy_fraction: float = 0.20
    max_extreme_probability_fraction: float = 0.01
    enforce_extreme_probability_fraction: bool = True
    target_binary_range: tuple[float, float] = (0.25, 0.75)
    categorical_dirichlet_concentration: float = 4.0
    hub_power: float = 1.5
    cross_type_weight: float = 1.0
    max_graph_attempts: int = 1_000
    max_parameter_attempts: int = 100
    master_seed: int = 0
    experiment_id: str = "synthetic"
    cell_id: str = "default"
    replicate_id: int = 0

    def __post_init__(self) -> None:
        if self.d < 3:
            raise ValueError("d must be at least 3 for a mixed-family graph")
        if self.n_train < 1 or (self.n_test is not None and self.n_test < 1) or self.pilot_size < 100:
            raise ValueError("n_train >= 1, optional n_test >= 1, and pilot_size >= 100 are required")
        if self.graph_family not in {
            "random_order",
            "hub",
            "modular",
            "empty",
            "order_blocks",
            "weak_bridges",
        }:
            raise ValueError(
                "unsupported graph_family"
            )
        if self.fixed_topology_id is not None:
            if self.fixed_topology_id not in {"asia", "insurance", "alarm", "hailfinder"}:
                raise ValueError("unsupported fixed_topology_id")
            if self.graph_family != "random_order":
                raise ValueError("fixed topologies use graph_family='random_order'")
        if self.signal_strength not in _R2_RANGES:
            raise ValueError("signal_strength must be weak, moderate, or strong")
        if self.continuous_noise not in {
            "gaussian",
            "student_t_df3",
            "student_t_df5",
            "centered_gamma_shape1_scale2",
        }:
            raise ValueError("unsupported continuous_noise")
        if not np.isfinite(self.expected_indegree) or self.expected_indegree <= 0:
            raise ValueError("expected_indegree must be finite and positive")
        if self.max_indegree < 1:
            raise ValueError("max_indegree must be at least one")
        if not 0.0 <= self.cross_type_minimum <= 1.0:
            raise ValueError("cross_type_minimum must lie in [0, 1]")
        if tuple(self.categorical_levels) != (3, 4):
            raise ValueError("the mixed generator fixes categorical_levels=(3, 4)")
        if self.variable_type_regime not in {
            "balanced_mixed",
            "continuous_dominant_mixed",
            "discrete_dominant_mixed",
        }:
            raise ValueError("unsupported variable_type_regime")
        if self.residual_sd <= 0 or not np.isfinite(self.residual_sd):
            raise ValueError("residual_sd must be finite and positive")
        if not 0.0 < self.min_category_marginal < 0.5:
            raise ValueError("min_category_marginal must lie in (0, 0.5)")
        if not 0.0 < self.min_binary_marginal < 0.5:
            raise ValueError("min_binary_marginal must lie in (0, 0.5)")
        if not 0.0 < self.min_median_entropy_fraction < 1.0:
            raise ValueError("min_median_entropy_fraction must lie in (0, 1)")
        if not 0.0 <= self.max_extreme_probability_fraction <= 0.25:
            raise ValueError("max_extreme_probability_fraction must lie in [0, 0.25]")
        if not isinstance(self.enforce_extreme_probability_fraction, bool):
            raise ValueError("enforce_extreme_probability_fraction must be boolean")
        low, high = self.target_binary_range
        if not 0.0 < low < high < 1.0:
            raise ValueError("target_binary_range must lie strictly inside (0, 1)")
        if self.categorical_dirichlet_concentration <= 0:
            raise ValueError("categorical_dirichlet_concentration must be positive")
        if self.hub_power <= 0 or self.cross_type_weight != 1.0:
            raise ValueError(
                "hub_power must be positive; the mixed generator fixes cross_type_weight=1 "
                "and enforces mixedness by rejection"
            )
        if self.max_graph_attempts < 1 or self.max_parameter_attempts < 1:
            raise ValueError("attempt limits must be positive")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        for name, value in (("experiment_id", self.experiment_id), ("cell_id", self.cell_id)):
            if not value or value != value.strip().lower() or "|" in value:
                raise ValueError(f"{name} must be a canonical lowercase seed token")
        if not 0 <= self.replicate_id < 10**6:
            raise ValueError("replicate_id must lie in [0, 10^6)")

    def protocol_deviations(self) -> tuple[str, ...]:
        """Return every deliberate deviation from the frozen generator."""

        expected_test = max(100, int(math.ceil(0.2 * self.n_train)))
        protocol_expected_indegree = (
            _E1_EXPECTED_INDEGREE if self.experiment_id in {"e1", "v1"} else 2.0
        )
        checks = (
            (
                not math.isclose(
                    self.expected_indegree,
                    protocol_expected_indegree,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "expected_indegree",
            ),
            (self.cross_type_minimum != 0.5, "cross_type_minimum"),
            (not self.require_all_cross_type_directions, "cross_type_direction_rule"),
            (
                self.variable_type_regime != "balanced_mixed",
                "variable_type_regime",
            ),
            (self.continuous_noise != "gaussian", "continuous_noise"),
            (self.fixed_topology_id is not None, "fixed_published_topology"),
            (self.residual_sd != 1.0, "residual_sd"),
            (self.pilot_size != 5_000, "calibration_size"),
            (self.min_category_marginal != 0.05, "minimum_category_marginal"),
            (
                self.max_extreme_probability_fraction != 0.01,
                "maximum_extreme_probability_fraction",
            ),
            (
                not self.enforce_extreme_probability_fraction,
                "extreme_probability_fraction_is_diagnostic_only",
            ),
            (self.target_binary_range != (0.25, 0.75), "binary_target_range"),
            (self.categorical_dirichlet_concentration != 4.0, "dirichlet_concentration"),
            (self.hub_power != 1.5, "hub_parent_degree_power"),
            (self.max_graph_attempts != 1_000, "maximum_graph_attempts"),
            (self.max_parameter_attempts != 100, "maximum_parameter_attempts"),
            (self.n_test is not None and self.n_test != expected_test, "test_size"),
        )
        return tuple(name for differs, name in checks if differs)


@dataclass(frozen=True)
class NodeMechanism:
    node: str
    variable_type: VariableType
    parents: tuple[str, ...]
    parameters: Dict[str, object]
    accepted_attempt: int


@dataclass
class MixedDAGDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    dag: nx.DiGraph
    variable_types: Dict[str, VariableType]
    cardinalities: Dict[str, int]
    topological_order: tuple[str, ...]
    mechanisms: Dict[str, NodeMechanism]
    config: MixedDAGConfig
    generation_diagnostics: Dict[str, object]
    signal_variant: Literal["weak", "strong"] | None = None
    base_manifest: Dict[str, object] | None = None

    def manifest(self) -> Dict[str, object]:
        deviations = self.config.protocol_deviations()
        if self.signal_variant is None:
            generator_name = (
                "ordersmc_protocol_v2_mixed_dag"
                if not deviations
                else "ordersmc_protocol_v2_pilot_variant"
            )
        else:
            generator_name = (
                "ordersmc_protocol_v2_e7_signal_variant"
                if not deviations
                else "ordersmc_protocol_v2_e7_signal_variant_pilot"
            )
        body: Dict[str, object] = {
            "manifest_schema": "ordersmc-mixed-dag-artifact-v2",
            "generator": generator_name,
            "protocol_compliant": not deviations,
            "protocol_deviations": list(deviations),
            "config": _manifest_config(self.config),
            "signal_variant": self.signal_variant,
            "topological_order": list(self.topological_order),
            "variable_types": dict(self.variable_types),
            "cardinalities": dict(self.cardinalities),
            "edges": [list(edge) for edge in sorted(self.dag.edges())],
            "mechanisms": {
                node: {
                    "node": mechanism.node,
                    "variable_type": mechanism.variable_type,
                    "parents": list(mechanism.parents),
                    "parameters": mechanism.parameters,
                    "accepted_attempt": mechanism.accepted_attempt,
                }
                for node, mechanism in self.mechanisms.items()
            },
            "diagnostics": dict(self.generation_diagnostics),
            "train_sha256": _frame_hash(self.train),
            "test_sha256": _frame_hash(self.test),
            "seed_namespaces": {
                **{
                    name: _seed_record(
                        self.config,
                        name,
                        int(
                            self.generation_diagnostics.get(
                                "accepted_graph_attempt",
                                self.generation_diagnostics.get(
                                    "base_accepted_graph_attempt", 0
                                ),
                            )
                        )
                        if name in {"types", "order", "graph"}
                        else 0,
                    ).__dict__
                    for name in ("types", "order", "graph", "train", "test", "candidate")
                },
                "mechanism_parameters": {
                    node: _seed_record(
                        self.config,
                        f"parameters-{node.lower()}",
                        mechanism.accepted_attempt,
                    ).__dict__
                    for node, mechanism in self.mechanisms.items()
                },
                "mechanism_calibration": {
                    node: _seed_record(
                        self.config,
                        f"calibration-{node.lower()}",
                        mechanism.accepted_attempt,
                    ).__dict__
                    for node, mechanism in self.mechanisms.items()
                },
            },
        }
        if self.signal_variant is not None:
            if self.base_manifest is None:
                raise RuntimeError("an E7 signal variant must retain its base manifest")
            base_manifest = dict(self.base_manifest)
            base_seeds = base_manifest.get("seed_namespaces")
            if not isinstance(base_seeds, Mapping):
                raise RuntimeError("the E7 base manifest has no seed namespace mapping")
            # The graph, type assignment, and generating order are immutable base
            # artifacts.  Only mechanisms, calibration, train/test, and candidate
            # streams use the e7-weak/e7-strong namespace.
            signal_seeds = body["seed_namespaces"]
            assert isinstance(signal_seeds, dict)
            for component in ("types", "order", "graph"):
                signal_seeds[component] = base_seeds[component]
            body["base_artifact"] = base_manifest
            body["base_artifact_fingerprint_sha256"] = base_manifest.get(
                "artifact_fingerprint_sha256"
            )
            body["reused_structure_fingerprint_sha256"] = _structure_fingerprint(
                self.dag,
                self.topological_order,
                self.variable_types,
                self.cardinalities,
            )
        body["serialized_artifacts"] = {
            "train.csv": _artifact_descriptor(
                _canonical_frame_bytes(self.train),
                media_type="text/csv",
                rows=len(self.train),
                columns=list(self.train.columns),
            ),
            "test.csv": _artifact_descriptor(
                _canonical_frame_bytes(self.test),
                media_type="text/csv",
                rows=len(self.test),
                columns=list(self.test.columns),
            ),
            "adjacency.csv": _artifact_descriptor(
                _canonical_adjacency_bytes(self.dag),
                media_type="text/csv",
                rows=self.dag.number_of_nodes(),
                columns=sorted(str(node) for node in self.dag.nodes()),
            ),
        }
        body["seed_record_inventory"] = _seed_record_inventory(
            body["seed_namespaces"], self.generation_diagnostics
        )
        body["artifact_fingerprint_sha256"] = _canonical_sha256(body)
        return body


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _manifest_config(config: MixedDAGConfig) -> Dict[str, object]:
    """Serialize configuration while preserving legacy default manifests.

    Artifact schema v2 predates ``max_extreme_probability_fraction`` and used
    its effective value of 0.01.  Omitting that default keeps old artifacts
    byte-replayable; amended protocols record every non-default value.
    """

    values = asdict(config)
    if config.fixed_topology_id is None:
        values.pop("fixed_topology_id")
    if config.max_extreme_probability_fraction == 0.01:
        values.pop("max_extreme_probability_fraction")
    if config.enforce_extreme_probability_fraction:
        values.pop("enforce_extreme_probability_fraction")
    if config.variable_type_regime == "balanced_mixed":
        values.pop("variable_type_regime")
    if config.continuous_noise == "gaussian":
        values.pop("continuous_noise")
    return values


def _structure_fingerprint(
    dag: nx.DiGraph,
    topological_order: Sequence[str],
    variable_types: Mapping[str, VariableType],
    cardinalities: Mapping[str, int],
) -> str:
    return _canonical_sha256(
        {
            "nodes": sorted(str(node) for node in dag.nodes()),
            "edges": [list(edge) for edge in sorted(dag.edges())],
            "topological_order": list(topological_order),
            "variable_types": dict(sorted(variable_types.items())),
            "cardinalities": dict(sorted(cardinalities.items())),
        }
    )


def _frame_hash(data: pd.DataFrame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    digest.update("\0".join(map(str, data.columns)).encode())
    digest.update("\0".join(map(str, data.dtypes)).encode())
    digest.update(f"{data.shape[0]}x{data.shape[1]}".encode())
    return digest.hexdigest()


def _canonical_frame_bytes(data: pd.DataFrame) -> bytes:
    """Serialize generated observations with a frozen, round-trip float format."""

    return data.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")


def _canonical_adjacency_bytes(dag: nx.DiGraph) -> bytes:
    nodes = sorted(str(node) for node in dag.nodes())
    frame = pd.DataFrame(
        nx.to_numpy_array(dag, nodelist=nodes, dtype=np.int8),
        index=nodes,
        columns=nodes,
    )
    return frame.to_csv(index=True, lineterminator="\n").encode("utf-8")


def _artifact_descriptor(
    payload: bytes,
    *,
    media_type: str,
    rows: int,
    columns: Sequence[str],
) -> Dict[str, object]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "media_type": media_type,
        "rows": int(rows),
        "columns": list(columns),
    }


def _seed_record_inventory(
    accepted: object,
    diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    if not isinstance(accepted, Mapping):
        raise RuntimeError("accepted seed namespaces must be a mapping")
    rejected_graph = []
    for rejection in diagnostics.get("graph_rejections", []):
        if isinstance(rejection, Mapping):
            rejected_graph.append(rejection.get("seed_records"))
    rejected_mechanisms: Dict[str, object] = {}
    raw_parameter_rejections = diagnostics.get("parameter_rejections", {})
    if isinstance(raw_parameter_rejections, Mapping):
        for node, rejections in raw_parameter_rejections.items():
            rejected_mechanisms[str(node)] = [
                rejection.get("seed_records")
                for rejection in rejections
                if isinstance(rejection, Mapping)
            ]
    rejected = {
        "graph": rejected_graph,
        "mechanisms": rejected_mechanisms,
    }

    def record_count(value: object) -> int:
        if isinstance(value, Mapping):
            if {"input_string", "seed", "sha256", "bit_generator", "numpy_version"} <= set(value):
                return 1
            return sum(record_count(child) for child in value.values())
        if isinstance(value, list):
            return sum(record_count(child) for child in value)
        return 0

    inventory_payload = {"accepted": dict(accepted), "rejected": rejected}
    return {
        **inventory_payload,
        "accepted_record_count": record_count(accepted),
        "rejected_record_count": record_count(rejected),
        "sha256": _canonical_sha256(inventory_payload),
    }


def _seed_record(config: MixedDAGConfig, component: str, attempt: int = 0) -> SeedRecord:
    return protocol_seed_record(
        config.master_seed,
        experiment=config.experiment_id,
        cell=config.cell_id,
        replicate=config.replicate_id,
        component=component,
        attempt=attempt,
    )


def _rng(config: MixedDAGConfig, component: str, attempt: int = 0) -> np.random.Generator:
    return _seed_record(config, component, attempt).generator()


def _draw_continuous_noise(
    rng: np.random.Generator,
    kind: ContinuousNoise | str,
    size: int,
    residual_sd: float,
) -> np.ndarray:
    """Draw centered unit-variance noise, scaled by ``residual_sd``.

    The Gaussian branch intentionally preserves the exact legacy RNG call.
    """

    if kind == "gaussian":
        return rng.normal(0.0, residual_sd, size)
    if kind == "student_t_df3":
        return residual_sd * rng.standard_t(3.0, size=size) / math.sqrt(3.0)
    if kind == "student_t_df5":
        return residual_sd * rng.standard_t(5.0, size=size) / math.sqrt(5.0 / 3.0)
    if kind == "centered_gamma_shape1_scale2":
        return residual_sd * (rng.gamma(shape=1.0, scale=2.0, size=size) - 2.0) / 2.0
    raise ValueError(f"unsupported continuous noise {kind!r}")


def _contrast_basis(cardinality: int) -> np.ndarray:
    basis = np.zeros((cardinality, cardinality - 1), dtype=float)
    for column in range(cardinality - 1):
        denominator = math.sqrt(float((column + 1) * (column + 2)))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def _softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _regime_type_counts(config: MixedDAGConfig) -> tuple[int, int, int]:
    """Return deterministic continuous, binary, and categorical node counts."""

    if config.variable_type_regime == "balanced_mixed":
        # Preserve the byte-for-byte legacy assignment for existing artifacts.
        continuous_count = config.d // 3
        binary_count = config.d // 3
        categorical_count = config.d - continuous_count - binary_count
    else:
        weights = (
            (0.60, 0.20, 0.20)
            if config.variable_type_regime == "continuous_dominant_mixed"
            else (0.20, 0.40, 0.40)
        )
        raw = np.asarray(weights, dtype=float) * config.d
        counts = np.floor(raw).astype(int)
        # Largest-remainder allocation is deterministic; the type order breaks
        # exact ties.  Every declared regime has all three types for d >= 8.
        for index in np.argsort(-(raw - counts), kind="stable")[: config.d - int(counts.sum())]:
            counts[int(index)] += 1
        continuous_count, binary_count, categorical_count = map(int, counts)
        if min(continuous_count, binary_count, categorical_count) < 1:
            raise ValueError("variable_type_regime requires at least one node of every type")
    return continuous_count, binary_count, categorical_count


def _draw_types(
    config: MixedDAGConfig, nodes: Sequence[str], attempt: int = 0
) -> tuple[Dict[str, VariableType], Dict[str, int]]:
    rng = _rng(config, "types", attempt)
    continuous_count, binary_count, categorical_count = _regime_type_counts(config)
    three_count = categorical_count // 2
    four_count = categorical_count // 2
    if categorical_count % 2:
        if int(rng.integers(0, 2)) == 0:
            three_count += 1
        else:
            four_count += 1
    typed_cards: list[tuple[VariableType, int | None]] = (
        [("con", None)] * continuous_count
        + [("bin", 2)] * binary_count
        + [("cat", 3)] * three_count
        + [("cat", 4)] * four_count
    )
    rng.shuffle(typed_cards)
    variable_types = {node: spec[0] for node, spec in zip(nodes, typed_cards)}
    cards: Dict[str, int] = {
        node: int(spec[1])
        for node, spec in zip(nodes, typed_cards)
        if spec[1] is not None
    }
    return variable_types, cards


def _cross_type_directions(
    edges: Sequence[tuple[str, str]], variable_types: Mapping[str, VariableType]
) -> set[tuple[VariableType, VariableType]]:
    return {
        (variable_types[parent], variable_types[child])
        for parent, child in edges
        if variable_types[parent] != variable_types[child]
    }


def _generate_graph(
    config: MixedDAGConfig,
    nodes: Sequence[str],
) -> tuple[
    nx.DiGraph,
    tuple[str, ...],
    Dict[str, VariableType],
    Dict[str, int],
    int,
    float,
    Dict[str, object],
    list[Dict[str, object]],
]:
    required_directions = {
        (parent_type, child_type)
        for parent_type in ("con", "bin", "cat")
        for child_type in ("con", "bin", "cat")
        if parent_type != child_type
    }
    rejection_log: list[Dict[str, object]] = []

    if config.fixed_topology_id is not None:
        fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "known_topologies_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        if fixture.get("schema") != "ordersmc-known-topologies-v1":
            raise RuntimeError("known-topology fixture has an unsupported schema")
        source = fixture["networks"][config.fixed_topology_id]
        source_nodes = sorted(
            {str(node) for edge in source["edges"] for node in edge}
        )
        if len(source_nodes) != config.d or len(nodes) != config.d:
            raise ValueError("fixed topology dimension disagrees with its plan")
        node_map = {source_node: str(nodes[index]) for index, source_node in enumerate(source_nodes)}
        edges = tuple(
            sorted((node_map[str(parent)], node_map[str(child)]) for parent, child in source["edges"])
        )
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        dag.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(dag):
            raise RuntimeError("fixed published topology is not acyclic")
        maximum_indegree = max(dict(dag.in_degree()).values(), default=0)
        if maximum_indegree > config.max_indegree:
            raise ValueError("fixed topology exceeds the configured maximum indegree")
        order = tuple(nx.lexicographical_topological_sort(dag, key=str))
        for attempt in range(config.max_graph_attempts):
            variable_types, cards = _draw_types(config, nodes, attempt)
            observed_directions = _cross_type_directions(edges, variable_types)
            cross_count = sum(
                variable_types[parent] != variable_types[child] for parent, child in edges
            )
            cross_fraction = cross_count / max(1, len(edges))
            if cross_fraction < config.cross_type_minimum:
                rejection_log.append(
                    {
                        "attempt": attempt,
                        "reason": "fixed_topology_cross_type_fraction_below_minimum",
                        "achieved": float(cross_fraction),
                        "required": float(config.cross_type_minimum),
                        "seed_records": {
                            component: asdict(_seed_record(config, component, attempt))
                            for component in ("types", "order", "graph")
                        },
                    }
                )
                continue
            if config.require_all_cross_type_directions and not required_directions <= observed_directions:
                rejection_log.append(
                    {
                        "attempt": attempt,
                        "reason": "fixed_topology_missing_cross_type_directions",
                        "missing": [
                            list(direction)
                            for direction in sorted(required_directions - observed_directions)
                        ],
                        "seed_records": {
                            component: asdict(_seed_record(config, component, attempt))
                            for component in ("types", "order", "graph")
                        },
                    }
                )
                continue
            graph_details: Dict[str, object] = {
                "fixed_topology_id": config.fixed_topology_id,
                "source_node_order": source_nodes,
                "source_to_generated_node": node_map,
                "source_bif_gz_sha256": source["bif_gz_sha256"],
                "edge_count": len(edges),
                "realized_average_indegree": len(edges) / config.d,
                "maximum_indegree": maximum_indegree,
            }
            return (
                dag,
                order,
                variable_types,
                cards,
                attempt,
                float(cross_fraction),
                graph_details,
                rejection_log,
            )
        raise GeneratorFailure(
            "unable to assign mixed types to the fixed topology under the frozen constraints",
            {"fixed_topology_id": config.fixed_topology_id},
        )

    if config.graph_family in {"order_blocks", "weak_bridges"}:
        if config.experiment_id != "e3":
            raise ValueError("module graph families are reserved for E3")
        from .stress_generators import E3ModuleTopologyConfig, generate_e3_module_topology

        topology = generate_e3_module_topology(
            E3ModuleTopologyConfig(
                d=config.d,
                family=config.graph_family,
                master_seed=config.master_seed,
                cell_id=config.cell_id,
                replicate_id=config.replicate_id,
                max_indegree=config.max_indegree,
                max_attempts=config.max_graph_attempts,
            )
        )
        variable_types, cards = _draw_types(config, nodes, topology.accepted_attempt)
        edges = tuple(sorted((str(u), str(v)) for u, v in topology.dag.edges()))
        cross_fraction = sum(
            variable_types[parent] != variable_types[child] for parent, child in edges
        ) / len(edges)
        observed_directions = _cross_type_directions(edges, variable_types)
        if cross_fraction < config.cross_type_minimum:
            raise GeneratorFailure(
                "E3 topology failed the frozen cross-type fraction",
                {"achieved": cross_fraction, "required": config.cross_type_minimum},
            )
        if config.require_all_cross_type_directions and config.d >= 20:
            if not required_directions <= observed_directions:
                raise GeneratorFailure(
                    "E3 topology missed a frozen cross-type direction",
                    {"missing": [list(row) for row in sorted(required_directions - observed_directions)]},
                )
        details: Dict[str, object] = {
            "topology_manifest": topology.manifest(),
            "weak_bridge_edges": [list(edge) for edge in topology.weak_bridge_edges],
            "moderate_within_module_edges": [
                list(edge) for edge in topology.moderate_within_module_edges
            ],
            "realized_average_indegree": len(edges) / config.d,
        }
        return (
            topology.dag.copy(),
            topology.topological_order,
            variable_types,
            cards,
            topology.accepted_attempt,
            float(cross_fraction),
            details,
            [dict(row) for row in topology.rejection_log],
        )

    def reject(attempt: int, reason: str, **details: object) -> None:
        rejection_log.append(
            {
                "attempt": attempt,
                "reason": reason,
                "seed_records": {
                    component: asdict(_seed_record(config, component, attempt))
                    for component in ("types", "order", "graph")
                },
                **details,
            }
        )

    for attempt in range(config.max_graph_attempts):
        variable_types, cards = _draw_types(config, nodes, attempt)
        order_rng = _rng(config, "order", attempt)
        graph_rng = _rng(config, "graph", attempt)
        order = tuple(str(node) for node in order_rng.permutation(nodes))
        candidates = [
            (order[left], order[right])
            for right in range(1, config.d)
            for left in range(right)
        ]
        graph_details: Dict[str, object]
        if config.graph_family == "random_order":
            # E[min(Binomial(r,q), K)] = sum_{k=1}^K P(Binomial(r,q)>=k).
            def realized_expectation(probability: float) -> float:
                return float(
                    sum(
                        sum(
                            float(binom.sf(k - 1, position, probability))
                            for k in range(1, config.max_indegree + 1)
                        )
                        for position in range(config.d)
                    )
                    / config.d
                )

            low, high = 0.0, 1.0
            attainable = realized_expectation(1.0)
            target = min(config.expected_indegree, attainable)
            for _ in range(80):
                midpoint = 0.5 * (low + high)
                if realized_expectation(midpoint) < target:
                    low = midpoint
                else:
                    high = midpoint
            edge_probability = 0.5 * (low + high)
            by_child: Dict[str, list[tuple[float, str, str]]] = {node: [] for node in nodes}
            proposed_count = 0
            for parent, child in candidates:
                edge_uniform = float(graph_rng.random())
                priority_uniform = float(graph_rng.random())
                if edge_uniform < edge_probability:
                    proposed_count += 1
                    by_child[child].append((priority_uniform, parent, child))
            selected = []
            for child in order:
                retained = sorted(by_child[child])[: config.max_indegree]
                selected.extend((parent, target_child) for _, parent, target_child in retained)
            graph_details = {
                "edge_probability": edge_probability,
                "expected_truncated_indegree": realized_expectation(edge_probability),
                "proposed_edge_count": proposed_count,
                "retained_edge_count": len(selected),
            }
        else:
            maximum_by_order = sum(
                min(config.max_indegree, position) for position in range(config.d)
            )
            target_edges = min(int(round(config.expected_indegree * config.d)), maximum_by_order)
            selected = []
            selected_set: set[tuple[str, str]] = set()
            indegree = {node: 0 for node in nodes}
            degree = {node: 0 for node in nodes}
            selection_sequence: list[list[object]] = []
            while len(selected) < target_edges:
                eligible = [
                    edge
                    for edge in candidates
                    if edge not in selected_set and indegree[edge[1]] < config.max_indegree
                ]
                if not eligible:
                    break
                eligible.sort()
                if config.graph_family == "modular":
                    order_position = {node: index for index, node in enumerate(order)}
                    module_count = min(5, max(2, config.d // 10))
                    module = {
                        node: min(
                            module_count - 1,
                            module_count * order_position[node] // config.d,
                        )
                        for node in nodes
                    }
                    weights = np.asarray(
                        [4.0 if module[parent] == module[child] else 1.0 for parent, child in eligible],
                        dtype=float,
                    )
                else:
                    weights = np.asarray(
                        [
                            (1.0 + degree[parent]) ** config.hub_power
                            * (1.0 + degree[child]) ** 0.5
                            for parent, child in eligible
                        ],
                        dtype=float,
                    )
                probabilities = weights / weights.sum()
                chosen_index = int(graph_rng.choice(len(eligible), p=probabilities))
                edge = eligible[chosen_index]
                selected.append(edge)
                selected_set.add(edge)
                indegree[edge[1]] += 1
                degree[edge[0]] += 1
                degree[edge[1]] += 1
                selection_sequence.append(
                    [len(selected), edge[0], edge[1], float(probabilities[chosen_index])]
                )
            if len(selected) != target_edges:
                reject(
                    attempt,
                    "hub_edge_target_unattainable",
                    selected_edges=len(selected),
                    target_edges=target_edges,
                )
                continue
            graph_details = {
                "target_edge_count": target_edges,
                "edge_selection_sequence": selection_sequence,
            }

        if config.graph_family == "empty":
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph_details.update(
                {
                    "target_edge_count": 0,
                    "retained_edge_count": 0,
                    "realized_average_indegree": 0.0,
                    "allowed_average_indegree_interval": [0.0, 0.0],
                }
            )
            return (
                graph,
                order,
                variable_types,
                cards,
                attempt,
                0.0,
                graph_details,
                rejection_log,
            )
        if not selected:
            reject(attempt, "empty_graph")
            continue
        cross_count = sum(
            variable_types[parent] != variable_types[child] for parent, child in selected
        )
        cross_fraction = cross_count / len(selected)
        if cross_fraction < config.cross_type_minimum:
            reject(
                attempt,
                "cross_type_fraction_below_minimum",
                achieved=float(cross_fraction),
                required=float(config.cross_type_minimum),
            )
            continue
        observed_directions = _cross_type_directions(selected, variable_types)
        if config.require_all_cross_type_directions:
            if config.d >= 20:
                if not required_directions <= observed_directions:
                    reject(
                        attempt,
                        "missing_cross_type_directions",
                        missing=[
                            list(direction)
                            for direction in sorted(required_directions - observed_directions)
                        ],
                    )
                    continue
            else:
                nonroot_child_types = {
                    variable_types[node]
                    for node in nodes
                    if any(child == node for _, child in selected)
                }
                cross_receiving_types = {child_type for _, child_type in observed_directions}
                if not nonroot_child_types <= cross_receiving_types:
                    reject(
                        attempt,
                        "nonroot_child_type_without_cross_type_parent",
                        missing_child_types=sorted(
                            nonroot_child_types - cross_receiving_types
                        ),
                    )
                    continue
        realized_indegree = len(selected) / config.d
        if config.experiment_id in {"e1", "v1"} and config.graph_family == "random_order":
            allowed_interval = _E1_RANDOM_ORDER_INDEGREE_INTERVAL
        else:
            allowed_interval = (
                (1.5, 2.4) if config.graph_family == "random_order" else (1.8, 2.2)
            )
        if not allowed_interval[0] <= realized_indegree <= allowed_interval[1]:
            reject(
                attempt,
                "average_indegree_outside_interval",
                achieved=float(realized_indegree),
                allowed=list(allowed_interval),
            )
            continue
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(selected)
        if not nx.is_directed_acyclic_graph(graph):  # construction invariant
            raise RuntimeError("ordered graph construction unexpectedly produced a cycle")
        graph_details["realized_average_indegree"] = realized_indegree
        graph_details["allowed_average_indegree_interval"] = list(allowed_interval)
        return (
            graph,
            order,
            variable_types,
            cards,
            attempt,
            cross_fraction,
            graph_details,
            rejection_log,
        )
    raise GeneratorFailure(
        f"unable to generate a graph satisfying the frozen constraints in "
        f"{config.max_graph_attempts} attempts",
        {"graph_rejections": rejection_log},
    )


def _feature_matrix(
    parents: Sequence[str],
    values: Mapping[str, np.ndarray],
    variable_types: Mapping[str, VariableType],
    cards: Mapping[str, int],
    continuous_stats: Mapping[str, Sequence[float]] | None = None,
) -> tuple[np.ndarray, Dict[str, tuple[int, int]], Dict[str, list[float]]]:
    columns: list[np.ndarray] = []
    slices: Dict[str, tuple[int, int]] = {}
    learned_stats: Dict[str, list[float]] = {}
    start = 0
    for parent in parents:
        kind = variable_types[parent]
        raw = np.asarray(values[parent])
        if kind == "con":
            if continuous_stats is None:
                mean = float(np.mean(raw))
                scale = float(np.std(raw, ddof=0))
                if not np.isfinite(scale) or scale < 1e-8:
                    raise RuntimeError(f"calibration SD for {parent} is below 1e-8")
            else:
                mean, scale = map(float, continuous_stats[parent])
            encoded = ((raw.astype(float) - mean) / scale)[:, None]
            learned_stats[parent] = [mean, scale]
        elif kind == "bin":
            if continuous_stats is None:
                probability = float(np.mean(raw))
            else:
                probability = float(continuous_stats[parent][0])
            variance = probability * (1.0 - probability)
            if variance < 1e-8:
                raise RuntimeError(f"calibration prevalence for {parent} is degenerate")
            encoded = ((raw.astype(float) - probability) / math.sqrt(variance))[:, None]
            learned_stats[parent] = [probability]
        else:
            cardinality = int(cards[parent])
            if continuous_stats is None:
                probabilities = np.bincount(raw.astype(int), minlength=cardinality) / len(raw)
            else:
                probabilities = np.asarray(continuous_stats[parent], dtype=float)
            if probabilities.shape != (cardinality,) or np.any(
                probabilities * (1.0 - probabilities) < 1e-8
            ):
                raise RuntimeError(f"calibration category proportions for {parent} are degenerate")
            integer_values = raw.astype(int)
            encoded = np.column_stack(
                [
                    ((integer_values == level).astype(float) - probabilities[level])
                    / math.sqrt(float(probabilities[level] * (1.0 - probabilities[level])))
                    for level in range(cardinality - 1)
                ]
            )
            learned_stats[parent] = probabilities.tolist()
        end = start + encoded.shape[1]
        slices[parent] = (start, end)
        columns.append(encoded)
        start = end
    row_count = len(next(iter(values.values()))) if values else 0
    matrix = np.column_stack(columns) if columns else np.empty((row_count, 0), dtype=float)
    return matrix, slices, learned_stats


def _draw_signed_coefficients(
    rng: np.random.Generator, count: int, signal_strength: SignalStrength
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=float)
    low, high = (0.5, 1.0)
    magnitudes = rng.uniform(low, high, size=count) / math.sqrt(float(count))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=count)
    return magnitudes * signs


def _draw_parent_block_vector(
    rng: np.random.Generator,
    parent_slices: Mapping[str, tuple[int, int]],
    feature_count: int,
) -> tuple[np.ndarray, Dict[str, Dict[str, object]]]:
    coefficients = np.zeros(feature_count, dtype=float)
    metadata: Dict[str, Dict[str, object]] = {}
    for parent in sorted(parent_slices):
        start, end = parent_slices[parent]
        dimension = end - start
        direction = rng.normal(size=dimension)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise RuntimeError("zero parent-block direction")
        direction /= norm
        magnitude = float(rng.uniform(0.5, 1.0))
        block = direction * magnitude / math.sqrt(float(dimension))
        coefficients[start:end] = block
        metadata[parent] = {
            "raw_direction": direction.tolist(),
            "raw_magnitude": magnitude,
            "block_dimension": dimension,
        }
    return coefficients, metadata


def _draw_parent_block_matrix(
    rng: np.random.Generator,
    parent_slices: Mapping[str, tuple[int, int]],
    rows: int,
    feature_count: int,
) -> tuple[np.ndarray, Dict[str, Dict[str, object]]]:
    coefficients = np.zeros((rows, feature_count), dtype=float)
    metadata: Dict[str, Dict[str, object]] = {}
    for parent in sorted(parent_slices):
        start, end = parent_slices[parent]
        dimension = rows * (end - start)
        direction = rng.normal(size=dimension)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise RuntimeError("zero categorical parent-block direction")
        direction /= norm
        magnitude = float(rng.uniform(0.5, 1.0))
        block = (direction * magnitude / math.sqrt(float(dimension))).reshape(rows, end - start)
        coefficients[:, start:end] = block
        metadata[parent] = {
            "raw_direction": direction.tolist(),
            "raw_magnitude": magnitude,
            "block_dimension": dimension,
        }
    return coefficients, metadata


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _binary_intercept(linear: np.ndarray, target: float) -> float:
    low, high = -40.0, 40.0

    def residual(intercept: float) -> float:
        return float(np.mean(expit(intercept + linear))) - target

    low_value = residual(low)
    high_value = residual(high)
    if low_value > 0.0 or high_value < 0.0:
        raise RuntimeError("binary intercept is not bracketed on [-40, 40]")
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        if residual(midpoint) < 0.0:
            low = midpoint
        else:
            high = midpoint
    return float(0.5 * (low + high))


def _calibrate_binary_signal(
    raw_predictor: np.ndarray,
    target_marginal: float,
    target_entropy: float,
) -> tuple[float, float, np.ndarray, float]:
    marginal_entropy = float(
        -(target_marginal * math.log(target_marginal)
          + (1.0 - target_marginal) * math.log(1.0 - target_marginal))
        / math.log(2.0)
    )
    if marginal_entropy < target_entropy + 0.02:
        raise RuntimeError("binary target marginal cannot support requested entropy")

    def evaluate(scale: float) -> tuple[float, np.ndarray, float]:
        linear = float(scale) * raw_predictor
        intercept = _binary_intercept(linear, target_marginal)
        probability = expit(intercept + linear)
        matrix = np.column_stack((1.0 - probability, probability))
        entropy = float(np.median(_entropy(matrix)) / math.log(2.0))
        return intercept, probability, entropy

    previous_scale = 0.0
    previous = evaluate(previous_scale)
    bracket: tuple[float, float] | None = None
    for step in range(1, 1001):
        scale = 0.02 * step
        current = evaluate(scale)
        if (previous[2] - target_entropy) * (current[2] - target_entropy) <= 0.0:
            bracket = (previous_scale, scale)
            break
        previous_scale, previous = scale, current
    if bracket is None:
        raise RuntimeError("binary entropy target is not bracketed on [0, 20]")
    low, high = bracket
    low_value = evaluate(low)[2] - target_entropy
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        mid_value = evaluate(midpoint)[2] - target_entropy
        if low_value * mid_value <= 0.0:
            high = midpoint
        else:
            low, low_value = midpoint, mid_value
    final_scale = 0.5 * (low + high)
    intercept, probability, achieved_entropy = evaluate(final_scale)
    return final_scale, intercept, probability, achieved_entropy


def _baseline_softmax(nonreference_logits: np.ndarray) -> np.ndarray:
    logits = np.column_stack(
        (nonreference_logits, np.zeros(nonreference_logits.shape[0], dtype=float))
    )
    return _softmax(logits)


def _softmax_intercept_newton(
    scaled_predictor: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cardinality = target.shape[0]
    intercept = (
        np.log(target[:-1] / target[-1])
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    for _ in range(200):
        probability = _baseline_softmax(scaled_predictor + intercept[None, :])
        full_residual = np.mean(probability, axis=0) - target
        residual = full_residual[:-1]
        residual_norm = float(np.max(np.abs(full_residual)))
        if residual_norm <= 1e-8:
            return intercept, probability
        first = probability[:, : cardinality - 1]
        jacobian = np.diag(np.mean(first, axis=0)) - (first.T @ first) / len(first)
        try:
            step = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("softmax intercept Jacobian is singular") from exc
        accepted = False
        step_scale = 1.0
        for _ in range(30):
            candidate = intercept - step_scale * step
            candidate_probability = _baseline_softmax(
                scaled_predictor + candidate[None, :]
            )
            candidate_residual = np.mean(candidate_probability, axis=0) - target
            if float(np.max(np.abs(candidate_residual))) < residual_norm:
                intercept = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            raise RuntimeError("softmax intercept line search failed")
    raise RuntimeError("softmax intercept calibration exceeded 200 Newton iterations")


def _calibrate_categorical_signal(
    raw_predictor: np.ndarray,
    target_marginal: np.ndarray,
    target_entropy: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    cardinality = target_marginal.shape[0]
    marginal_entropy = float(
        -np.sum(target_marginal * np.log(target_marginal)) / math.log(float(cardinality))
    )
    if marginal_entropy < target_entropy + 0.02:
        raise RuntimeError("categorical marginal cannot support requested entropy")

    def evaluate(
        scale: float, initial: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        intercept, probability = _softmax_intercept_newton(
            float(scale) * raw_predictor,
            target_marginal,
            initial,
        )
        entropy = float(
            np.median(_entropy(probability)) / math.log(float(cardinality))
        )
        return intercept, probability, entropy

    previous_scale = 0.0
    previous = evaluate(0.0)
    bracket: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for step in range(1, 1001):
        scale = 0.02 * step
        current = evaluate(scale, previous[0])
        if (previous[2] - target_entropy) * (current[2] - target_entropy) <= 0.0:
            bracket = (previous_scale, scale, previous[0], current[0])
            break
        previous_scale, previous = scale, current
    if bracket is None:
        raise RuntimeError("categorical entropy target is not bracketed on [0, 20]")
    low, high, low_intercept, high_intercept = bracket
    low_result = evaluate(low, low_intercept)
    low_value = low_result[2] - target_entropy
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        initial = 0.5 * (low_intercept + high_intercept)
        mid_result = evaluate(midpoint, initial)
        mid_value = mid_result[2] - target_entropy
        if low_value * mid_value <= 0.0:
            high, high_intercept = midpoint, mid_result[0]
        else:
            low, low_intercept, low_value = midpoint, mid_result[0], mid_value
    final_scale = 0.5 * (low + high)
    intercept, probability, achieved_entropy = evaluate(
        final_scale, 0.5 * (low_intercept + high_intercept)
    )
    return final_scale, intercept, probability, achieved_entropy


def _draw_categorical_target(
    rng: np.random.Generator, cardinality: int, config: MixedDAGConfig
) -> np.ndarray:
    target = rng.dirichlet(
        np.full(cardinality, config.categorical_dirichlet_concentration, dtype=float)
    )
    if float(np.min(target)) < config.min_category_marginal:
        raise RuntimeError("categorical target marginal below 0.05")
    return target


def _build_mechanisms(
    config: MixedDAGConfig,
    dag: nx.DiGraph,
    order: Sequence[str],
    variable_types: Mapping[str, VariableType],
    cards: Mapping[str, int],
) -> tuple[
    Dict[str, NodeMechanism],
    Dict[str, np.ndarray],
    Dict[str, list[Dict[str, object]]],
]:
    pilot: Dict[str, np.ndarray] = {}
    mechanisms: Dict[str, NodeMechanism] = {}
    rejection_log: Dict[str, list[Dict[str, object]]] = {}

    for node in order:
        kind = variable_types[node]
        parents = tuple(sorted(str(parent) for parent in dag.predecessors(node)))
        parent_values = {parent: pilot[parent] for parent in parents}
        features, parent_slices, continuous_stats = _feature_matrix(
            parents, parent_values, variable_types, cards
        ) if parents else (
            np.empty((config.pilot_size, 0), dtype=float),
            {},
            {},
        )

        accepted = False
        node_rejections: list[Dict[str, object]] = []
        rejection_log[node] = node_rejections

        def reject(reason: str, **details: object) -> None:
            node_rejections.append(
                {
                    "attempt": attempt,
                    "reason": reason,
                    "seed_records": {
                        "parameters": asdict(
                            _seed_record(config, f"parameters-{node.lower()}", attempt)
                        ),
                        "calibration": asdict(
                            _seed_record(config, f"calibration-{node.lower()}", attempt)
                        ),
                    },
                    **details,
                }
            )

        for attempt in range(config.max_parameter_attempts):
            parameter_rng = _rng(config, f"parameters-{node.lower()}", attempt)
            pilot_rng = _rng(config, f"calibration-{node.lower()}", attempt)
            if kind == "con":
                if not parents:
                    raw_coefficients = np.empty(0, dtype=float)
                    block_metadata: Dict[str, Dict[str, object]] = {}
                    target_r2 = 0.0
                    signal_scale = 0.0
                    coefficients = raw_coefficients
                    intercept = 0.0
                else:
                    raw_coefficients, block_metadata = _draw_parent_block_vector(
                        parameter_rng, parent_slices, features.shape[1]
                    )
                    raw_predictor = features @ raw_coefficients
                    predictor_variance = float(np.var(raw_predictor, ddof=0))
                    if predictor_variance < 1e-10:
                        reject(
                            "continuous_predictor_variance_below_minimum",
                            achieved=predictor_variance,
                        )
                        continue
                    target_r2 = float(parameter_rng.uniform(*_R2_RANGES[config.signal_strength]))
                    signal_scale = math.sqrt(
                        target_r2 / ((1.0 - target_r2) * predictor_variance)
                    )
                    coefficients = signal_scale * raw_coefficients
                    if float(np.max(np.abs(coefficients), initial=0.0)) > 10.0:
                        reject(
                            "continuous_coefficient_above_maximum",
                            maximum=float(np.max(np.abs(coefficients), initial=0.0)),
                        )
                        continue
                    intercept = float(-signal_scale * np.mean(raw_predictor))
                mean = intercept + features @ coefficients
                values = mean + _draw_continuous_noise(
                    pilot_rng,
                    config.continuous_noise,
                    config.pilot_size,
                    config.residual_sd,
                )
                parameters: Dict[str, object] = {
                    "intercept": intercept,
                    "coefficients": coefficients.tolist(),
                    "raw_coefficients": raw_coefficients.tolist(),
                    "raw_parent_blocks": block_metadata,
                    "signal_scale": signal_scale,
                    "target_r2": target_r2,
                    "achieved_calibration_r2": (
                        float(np.var(mean) / (np.var(mean) + config.residual_sd**2))
                        if parents
                        else 0.0
                    ),
                    "parent_feature_slices": {k: list(v) for k, v in parent_slices.items()},
                    "continuous_parent_stats": continuous_stats,
                    "residual_sd": float(config.residual_sd),
                }
                if config.continuous_noise != "gaussian":
                    parameters["continuous_noise"] = config.continuous_noise
                accepted = bool(np.all(np.isfinite(values)))
            elif kind == "bin":
                target = float(parameter_rng.uniform(*config.target_binary_range))
                if not parents:
                    raw_coefficients = np.empty(0, dtype=float)
                    block_metadata = {}
                    target_entropy = float(
                        -(target * math.log(target) + (1.0 - target) * math.log(1.0 - target))
                        / math.log(2.0)
                    )
                    signal_scale = 0.0
                    coefficients = raw_coefficients
                    intercept = float(math.log(target / (1.0 - target)))
                    probability = np.full(config.pilot_size, target, dtype=float)
                    entropy_fraction = target_entropy
                else:
                    raw_coefficients, block_metadata = _draw_parent_block_vector(
                        parameter_rng, parent_slices, features.shape[1]
                    )
                    raw_predictor = features @ raw_coefficients
                    target_entropy = float(
                        parameter_rng.uniform(*_ENTROPY_RANGES[config.signal_strength])
                    )
                    marginal_entropy = float(
                        -(target * math.log(target) + (1.0 - target) * math.log(1.0 - target))
                        / math.log(2.0)
                    )
                    if marginal_entropy < target_entropy + 0.02:
                        reject(
                            "binary_marginal_entropy_too_low_for_target",
                            marginal_entropy=marginal_entropy,
                            target_entropy=target_entropy,
                        )
                        continue
                    try:
                        (
                            signal_scale,
                            intercept,
                            probability,
                            entropy_fraction,
                        ) = _calibrate_binary_signal(
                            raw_predictor, target, target_entropy
                        )
                    except (RuntimeError, ValueError) as error:
                        reject("binary_calibration_failed", message=str(error))
                        continue
                    coefficients = signal_scale * raw_coefficients
                probability_matrix = np.column_stack((1.0 - probability, probability))
                marginal = np.mean(probability_matrix, axis=0)
                extreme_fraction = float(
                    np.mean((probability < 0.01) | (probability > 0.99))
                )
                accepted = bool(
                    np.min(marginal) >= config.min_binary_marginal
                    and (
                        not config.enforce_extreme_probability_fraction
                        or extreme_fraction <= config.max_extreme_probability_fraction
                    )
                )
                if accepted:
                    values = pilot_rng.binomial(1, probability, size=config.pilot_size).astype(np.int64)
                parameters = {
                    "intercept": intercept,
                    "coefficients": coefficients.tolist(),
                    "raw_coefficients": raw_coefficients.tolist(),
                    "raw_parent_blocks": block_metadata,
                    "signal_scale": signal_scale,
                    "parent_feature_slices": {k: list(v) for k, v in parent_slices.items()},
                    "continuous_parent_stats": continuous_stats,
                    "target_marginal": target,
                    "target_median_entropy_fraction": target_entropy,
                    "pilot_marginal": marginal.tolist(),
                    "pilot_median_entropy_fraction": entropy_fraction,
                    "pilot_extreme_probability_fraction": extreme_fraction,
                    "extreme_probability_fraction_enforced": bool(
                        config.enforce_extreme_probability_fraction
                    ),
                }
            else:
                cardinality = int(cards[node])
                try:
                    target = _draw_categorical_target(parameter_rng, cardinality, config)
                except RuntimeError as error:
                    reject("categorical_target_rejected", message=str(error))
                    continue
                if not parents:
                    raw_coefficients = np.empty((cardinality - 1, 0), dtype=float)
                    block_metadata = {}
                    target_entropy = float(
                        -np.sum(target * np.log(target)) / math.log(float(cardinality))
                    )
                    signal_scale = 0.0
                    coefficients = raw_coefficients
                    intercept = np.log(target[:-1] / target[-1])
                    probability = np.tile(target, (config.pilot_size, 1))
                    entropy_fraction = target_entropy
                else:
                    raw_coefficients, block_metadata = _draw_parent_block_matrix(
                        parameter_rng,
                        parent_slices,
                        cardinality - 1,
                        features.shape[1],
                    )
                    raw_predictor = features @ raw_coefficients.T
                    target_entropy = float(
                        parameter_rng.uniform(*_ENTROPY_RANGES[config.signal_strength])
                    )
                    marginal_entropy = float(
                        -np.sum(target * np.log(target)) / math.log(float(cardinality))
                    )
                    if marginal_entropy < target_entropy + 0.02:
                        reject(
                            "categorical_marginal_entropy_too_low_for_target",
                            marginal_entropy=marginal_entropy,
                            target_entropy=target_entropy,
                        )
                        continue
                    try:
                        (
                            signal_scale,
                            intercept,
                            probability,
                            entropy_fraction,
                        ) = _calibrate_categorical_signal(
                            raw_predictor, target, target_entropy
                        )
                    except RuntimeError as error:
                        reject("categorical_calibration_failed", message=str(error))
                        continue
                    coefficients = signal_scale * raw_coefficients
                marginal = np.mean(probability, axis=0)
                extreme_fraction = float(
                    np.mean(np.any(probability < 0.005, axis=1))
                )
                accepted = bool(
                    np.min(marginal) >= config.min_category_marginal
                    and (
                        not config.enforce_extreme_probability_fraction
                        or extreme_fraction <= config.max_extreme_probability_fraction
                    )
                )
                if accepted:
                    uniforms = pilot_rng.random(config.pilot_size)
                    values = np.sum(uniforms[:, None] > np.cumsum(probability, axis=1), axis=1)
                    values = np.minimum(values, cardinality - 1).astype(np.int64)
                parameters = {
                    "baseline_intercepts": intercept.tolist(),
                    "baseline_coefficients": coefficients.tolist(),
                    "raw_baseline_coefficients": raw_coefficients.tolist(),
                    "raw_parent_blocks": block_metadata,
                    "signal_scale": signal_scale,
                    "parent_feature_slices": {k: list(v) for k, v in parent_slices.items()},
                    "continuous_parent_stats": continuous_stats,
                    "target_marginal": target.tolist(),
                    "target_median_entropy_fraction": target_entropy,
                    "pilot_marginal": marginal.tolist(),
                    "pilot_median_entropy_fraction": entropy_fraction,
                    "pilot_extreme_probability_fraction": extreme_fraction,
                    "extreme_probability_fraction_enforced": bool(
                        config.enforce_extreme_probability_fraction
                    ),
                }

            if accepted:
                pilot[node] = np.asarray(values)
                mechanisms[node] = NodeMechanism(
                    node=node,
                    variable_type=kind,
                    parents=parents,
                    parameters=parameters,
                    accepted_attempt=attempt,
                )
                break
            if kind == "con":
                reject("continuous_calibration_values_nonfinite")
            elif kind == "bin":
                reject(
                    "binary_probability_acceptance_failed",
                    minimum_marginal=float(np.min(marginal)),
                    maximum_extreme_fraction=float(
                        config.max_extreme_probability_fraction
                    ),
                    extreme_fraction=float(
                        np.mean((probability < 0.01) | (probability > 0.99))
                    ),
                )
            else:
                reject(
                    "categorical_probability_acceptance_failed",
                    minimum_marginal=float(np.min(marginal)),
                    maximum_extreme_fraction=float(
                        config.max_extreme_probability_fraction
                    ),
                    extreme_fraction=float(
                        np.mean(np.any(probability < 0.005, axis=1))
                    ),
                )
        if not accepted:
            raise GeneratorFailure(
                f"failed to construct a non-degenerate mechanism for {node} after "
                f"{config.max_parameter_attempts} attempts",
                {
                    "node": node,
                    "parameter_rejections": rejection_log,
                    "accepted_mechanisms": {
                        accepted_node: {
                            "node": mechanism.node,
                            "variable_type": mechanism.variable_type,
                            "parents": list(mechanism.parents),
                            "parameters": mechanism.parameters,
                            "accepted_attempt": mechanism.accepted_attempt,
                            "seed_records": {
                                "parameters": asdict(
                                    _seed_record(
                                        config,
                                        f"parameters-{accepted_node.lower()}",
                                        mechanism.accepted_attempt,
                                    )
                                ),
                                "calibration": asdict(
                                    _seed_record(
                                        config,
                                        f"calibration-{accepted_node.lower()}",
                                        mechanism.accepted_attempt,
                                    )
                                ),
                            },
                        }
                        for accepted_node, mechanism in mechanisms.items()
                    },
                },
            )
    return mechanisms, pilot, rejection_log


def _sample_from_mechanisms(
    row_count: int,
    rng: np.random.Generator,
    order: Sequence[str],
    mechanisms: Mapping[str, NodeMechanism],
    variable_types: Mapping[str, VariableType],
    cards: Mapping[str, int],
) -> pd.DataFrame:
    values: Dict[str, np.ndarray] = {}
    for node in order:
        mechanism = mechanisms[node]
        parameters = mechanism.parameters
        parents = mechanism.parents
        if parents:
            features, _, _ = _feature_matrix(
                parents,
                {parent: values[parent] for parent in parents},
                variable_types,
                cards,
                continuous_stats=parameters["continuous_parent_stats"],
            )
        else:
            features = np.empty((row_count, 0), dtype=float)

        if mechanism.variable_type == "con":
            mean = float(parameters["intercept"]) + features @ np.asarray(
                parameters["coefficients"], dtype=float
            )
            values[node] = mean + _draw_continuous_noise(
                rng,
                str(parameters.get("continuous_noise", "gaussian")),
                row_count,
                float(parameters["residual_sd"]),
            )
        elif mechanism.variable_type == "bin":
            linear = float(parameters["intercept"]) + features @ np.asarray(
                parameters["coefficients"], dtype=float
            )
            values[node] = rng.binomial(1, expit(linear), size=row_count).astype(np.int64)
        else:
            intercept = np.asarray(parameters["baseline_intercepts"], dtype=float)
            coefficients = np.asarray(parameters["baseline_coefficients"], dtype=float)
            probability = _baseline_softmax(features @ coefficients.T + intercept[None, :])
            uniforms = rng.random(row_count)
            sampled = np.sum(uniforms[:, None] > np.cumsum(probability, axis=1), axis=1)
            values[node] = np.minimum(sampled, int(cards[node]) - 1).astype(np.int64)
    return pd.DataFrame({node: values[node] for node in sorted(values)})


def generate_mixed_dag_dataset(config: MixedDAGConfig) -> MixedDAGDataset:
    """Generate a graph, frozen mechanisms, training data, and independent test data."""

    nodes = tuple(f"X{index:03d}" for index in range(config.d))
    (
        dag,
        order,
        variable_types,
        cardinalities,
        graph_attempts,
        cross_fraction,
        graph_details,
        graph_rejections,
    ) = _generate_graph(config, nodes)
    try:
        mechanisms, _, parameter_rejections = _build_mechanisms(
            config, dag, order, variable_types, cardinalities
        )
    except GeneratorFailure as error:
        raise GeneratorFailure(
            str(error),
            {
                **error.diagnostics,
                "accepted_graph_attempt": graph_attempts,
                "graph_rejections": graph_rejections,
                "accepted_structure": {
                    "topological_order": list(order),
                    "variable_types": dict(variable_types),
                    "cardinalities": dict(cardinalities),
                    "edges": [list(edge) for edge in sorted(dag.edges())],
                    "seed_records": {
                        component: asdict(
                            _seed_record(config, component, graph_attempts)
                        )
                        for component in ("types", "order", "graph")
                    },
                },
            },
        ) from error
    train = _sample_from_mechanisms(
        config.n_train,
        _rng(config, "train"),
        order,
        mechanisms,
        variable_types,
        cardinalities,
    )
    test_size = (
        max(100, int(math.ceil(0.2 * config.n_train)))
        if config.n_test is None
        else int(config.n_test)
    )
    test = _sample_from_mechanisms(
        test_size,
        _rng(config, "test"),
        order,
        mechanisms,
        variable_types,
        cardinalities,
    )
    diagnostics: Dict[str, object] = {
        "accepted_graph_attempt": graph_attempts,
        "graph_attempts_used": graph_attempts + 1,
        "graph_attempts": graph_attempts + 1,
        "graph_rejections": graph_rejections,
        "edge_count": int(dag.number_of_edges()),
        "realized_indegree": float(dag.number_of_edges() / config.d),
        "maximum_indegree": max(dict(dag.in_degree()).values(), default=0),
        "cross_type_fraction": float(cross_fraction),
        "cross_type_directions": [
            list(direction)
            for direction in sorted(_cross_type_directions(list(dag.edges()), variable_types))
        ],
        "parameter_attempts": {
            node: mechanism.accepted_attempt + 1 for node, mechanism in mechanisms.items()
        },
        "accepted_parameter_attempt": {
            node: mechanism.accepted_attempt for node, mechanism in mechanisms.items()
        },
        "parameter_rejections": parameter_rejections,
        "graph_details": graph_details,
    }
    return MixedDAGDataset(
        train=train,
        test=test,
        dag=dag,
        variable_types=variable_types,
        cardinalities=cardinalities,
        topological_order=tuple(order),
        mechanisms=mechanisms,
        config=config,
        generation_diagnostics=diagnostics,
    )


def regenerate_e7_signal_variant(
    base: MixedDAGDataset,
    signal_strength: Literal["weak", "strong"],
) -> MixedDAGDataset:
    """Regenerate an E7 signal row while reusing the exact base structure.

    Protocol E7 does not redraw variable types, the generating order, or the
    graph.  This function therefore accepts the already materialized moderate
    signal artifact and redraws only node mechanisms/calibration responses and
    independent train/test samples under the ``e7-weak`` or ``e7-strong`` seed
    namespace.  The embedded base manifest makes the resulting artifact
    independently replayable.
    """

    if signal_strength not in {"weak", "strong"}:
        raise ValueError("E7 signal_strength must be 'weak' or 'strong'")
    if base.config.experiment_id != "e7":
        raise ValueError("E7 signal regeneration requires a base experiment_id of 'e7'")
    if base.config.signal_strength != "moderate":
        raise ValueError("E7 signal regeneration requires the moderate-signal base artifact")
    if base.signal_variant is not None:
        raise ValueError("an E7 signal variant cannot be regenerated from another variant")

    variant_config = replace(
        base.config,
        signal_strength=signal_strength,
        experiment_id=f"e7-{signal_strength}",
    )
    dag = base.dag.copy()
    order = tuple(base.topological_order)
    variable_types = dict(base.variable_types)
    cardinalities = dict(base.cardinalities)
    mechanisms, _, parameter_rejections = _build_mechanisms(
        variant_config,
        dag,
        order,
        variable_types,
        cardinalities,
    )
    train = _sample_from_mechanisms(
        variant_config.n_train,
        _rng(variant_config, "train"),
        order,
        mechanisms,
        variable_types,
        cardinalities,
    )
    test_size = (
        max(100, int(math.ceil(0.2 * variant_config.n_train)))
        if variant_config.n_test is None
        else int(variant_config.n_test)
    )
    test = _sample_from_mechanisms(
        test_size,
        _rng(variant_config, "test"),
        order,
        mechanisms,
        variable_types,
        cardinalities,
    )
    base_manifest = base.manifest()
    structure_fingerprint = _structure_fingerprint(
        dag, order, variable_types, cardinalities
    )
    diagnostics: Dict[str, object] = {
        "generation_mode": "e7_signal_regeneration",
        "signal_variant": signal_strength,
        "base_artifact_fingerprint_sha256": base_manifest[
            "artifact_fingerprint_sha256"
        ],
        "reused_structure_fingerprint_sha256": structure_fingerprint,
        "graph_reused": True,
        "type_assignment_reused": True,
        "topological_order_reused": True,
        "base_accepted_graph_attempt": base.generation_diagnostics[
            "accepted_graph_attempt"
        ],
        "edge_count": int(dag.number_of_edges()),
        "realized_indegree": float(dag.number_of_edges() / variant_config.d),
        "maximum_indegree": max(dict(dag.in_degree()).values(), default=0),
        "cross_type_fraction": float(
            sum(variable_types[parent] != variable_types[child] for parent, child in dag.edges())
            / max(1, dag.number_of_edges())
        ),
        "cross_type_directions": [
            list(direction)
            for direction in sorted(_cross_type_directions(list(dag.edges()), variable_types))
        ],
        "parameter_attempts": {
            node: mechanism.accepted_attempt + 1 for node, mechanism in mechanisms.items()
        },
        "accepted_parameter_attempt": {
            node: mechanism.accepted_attempt for node, mechanism in mechanisms.items()
        },
        "parameter_rejections": parameter_rejections,
    }
    return MixedDAGDataset(
        train=train,
        test=test,
        dag=dag,
        variable_types=variable_types,
        cardinalities=cardinalities,
        topological_order=order,
        mechanisms=mechanisms,
        config=variant_config,
        generation_diagnostics=diagnostics,
        signal_variant=signal_strength,
        base_manifest=base_manifest,
    )


def generate_e7_signal_variants(
    base: MixedDAGDataset,
) -> tuple[MixedDAGDataset, MixedDAGDataset]:
    """Return the prespecified weak and strong E7 rows for one base artifact."""

    return (
        regenerate_e7_signal_variant(base, "weak"),
        regenerate_e7_signal_variant(base, "strong"),
    )


def replay_mixed_dag_dataset(manifest: Mapping[str, object]) -> MixedDAGDataset:
    """Regenerate and compare the complete manifest, not selected hashes only."""

    config = validate_mixed_dag_manifest(manifest)
    generator_name = str(manifest["generator"])
    if generator_name in {
        "ordersmc_protocol_v2_e7_signal_variant",
        "ordersmc_protocol_v2_e7_signal_variant_pilot",
    }:
        raw_base = manifest.get("base_artifact")
        if not isinstance(raw_base, Mapping):
            raise ValueError("E7 signal manifest has no embedded base artifact")
        base = replay_mixed_dag_dataset(raw_base)
        raw_variant = manifest.get("signal_variant")
        if raw_variant not in {"weak", "strong"}:
            raise ValueError("E7 signal manifest has an invalid signal variant")
        regenerated = regenerate_e7_signal_variant(base, raw_variant)
    else:
        regenerated = generate_mixed_dag_dataset(config)

    expected = _json_normalize(dict(manifest))
    observed = _json_normalize(regenerated.manifest())
    if observed != expected:
        differing = sorted(
            key
            for key in set(expected) | set(observed)
            if expected.get(key) != observed.get(key)
        )
        raise RuntimeError(
            "regenerated mixed-DAG manifest differs in frozen fields: "
            + ", ".join(differing)
        )
    return regenerated


def validate_mixed_dag_manifest(manifest: Mapping[str, object]) -> MixedDAGConfig:
    """Validate the complete mixed-DAG manifest schema and seeds."""

    if not isinstance(manifest, Mapping):
        raise TypeError("generator manifest must be a mapping")
    generator_name = manifest.get("generator")
    ordinary_names = {
        "ordersmc_protocol_v2_mixed_dag",
        "ordersmc_protocol_v2_pilot_variant",
    }
    e7_names = {
        "ordersmc_protocol_v2_e7_signal_variant",
        "ordersmc_protocol_v2_e7_signal_variant_pilot",
    }
    if generator_name not in ordinary_names | e7_names:
        raise ValueError("manifest is not a supported OrderSMC mixed dataset")
    required = {
        "manifest_schema",
        "generator",
        "protocol_compliant",
        "protocol_deviations",
        "config",
        "signal_variant",
        "topological_order",
        "variable_types",
        "cardinalities",
        "edges",
        "mechanisms",
        "diagnostics",
        "train_sha256",
        "test_sha256",
        "seed_namespaces",
        "serialized_artifacts",
        "seed_record_inventory",
        "artifact_fingerprint_sha256",
    }
    if generator_name in e7_names:
        required |= {
            "base_artifact",
            "base_artifact_fingerprint_sha256",
            "reused_structure_fingerprint_sha256",
        }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        extra = sorted(set(manifest) - required)
        raise ValueError(f"generator manifest schema mismatch; missing={missing}, extra={extra}")
    if manifest["manifest_schema"] != "ordersmc-mixed-dag-artifact-v2":
        raise ValueError("unsupported mixed-DAG manifest schema")
    frozen_hash = manifest["artifact_fingerprint_sha256"]
    without_hash = dict(manifest)
    without_hash.pop("artifact_fingerprint_sha256")
    if not isinstance(frozen_hash, str) or frozen_hash != _canonical_sha256(without_hash):
        raise ValueError("mixed-DAG artifact fingerprint is invalid")

    raw_config = manifest["config"]
    if not isinstance(raw_config, Mapping):
        raise ValueError("generator manifest has no configuration mapping")
    config_values = dict(raw_config)
    for tuple_field in ("categorical_levels", "target_binary_range"):
        if tuple_field in config_values:
            config_values[tuple_field] = tuple(config_values[tuple_field])
    config = MixedDAGConfig(**config_values)
    deviations = list(config.protocol_deviations())
    if manifest["protocol_deviations"] != deviations:
        raise ValueError("protocol deviation list does not match the generator configuration")
    if bool(manifest["protocol_compliant"]) != (not deviations):
        raise ValueError("protocol compliance flag does not match the generator configuration")

    nodes = [f"X{index:03d}" for index in range(config.d)]
    variable_types = manifest["variable_types"]
    cardinalities = manifest["cardinalities"]
    order = manifest["topological_order"]
    edges = manifest["edges"]
    if not isinstance(variable_types, Mapping) or set(variable_types) != set(nodes):
        raise ValueError("variable-type schema does not match the configured nodes")
    if any(kind not in {"con", "bin", "cat"} for kind in variable_types.values()):
        raise ValueError("variable-type schema contains an unsupported type")
    continuous_count, binary_count, categorical_count = _regime_type_counts(config)
    expected_counts = {
        "con": continuous_count,
        "bin": binary_count,
        "cat": categorical_count,
    }
    observed_counts = {
        kind: sum(value == kind for value in variable_types.values())
        for kind in expected_counts
    }
    if observed_counts != expected_counts:
        raise ValueError("variable-type counts do not match the mixed-DAG protocol")
    if not isinstance(cardinalities, Mapping):
        raise ValueError("cardinalities must be a mapping")
    discrete_nodes = {node for node, kind in variable_types.items() if kind != "con"}
    if set(cardinalities) != discrete_nodes:
        raise ValueError("cardinality keys must be exactly the discrete variables")
    for node, cardinality in cardinalities.items():
        expected_values = {2} if variable_types[node] == "bin" else {3, 4}
        if isinstance(cardinality, bool) or cardinality not in expected_values:
            raise ValueError(f"invalid cardinality for {node}")
    if not isinstance(order, list) or len(order) != config.d or set(order) != set(nodes):
        raise ValueError("topological order is not a permutation of the configured nodes")
    if not isinstance(edges, list) or any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(node not in variable_types for node in edge)
        for edge in edges
    ):
        raise ValueError("edge list has an invalid schema")
    edge_tuples = [tuple(edge) for edge in edges]
    if edge_tuples != sorted(set(edge_tuples)):
        raise ValueError("edge list must be unique and lexicographically sorted")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_tuples)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("manifest edge list is cyclic")
    positions = {node: index for index, node in enumerate(order)}
    if any(positions[parent] >= positions[child] for parent, child in graph.edges()):
        raise ValueError("manifest order is not topological for its edge list")
    if max(dict(graph.in_degree()).values(), default=0) > config.max_indegree:
        raise ValueError("manifest graph exceeds the configured maximum indegree")

    mechanisms = manifest["mechanisms"]
    if not isinstance(mechanisms, Mapping) or set(mechanisms) != set(nodes):
        raise ValueError("mechanism records do not match the configured nodes")
    for node in nodes:
        mechanism = mechanisms[node]
        if not isinstance(mechanism, Mapping) or set(mechanism) != {
            "node",
            "variable_type",
            "parents",
            "parameters",
            "accepted_attempt",
        }:
            raise ValueError(f"mechanism record for {node} has an invalid schema")
        parents = sorted(str(parent) for parent in graph.predecessors(node))
        if (
            mechanism["node"] != node
            or mechanism["variable_type"] != variable_types[node]
            or mechanism["parents"] != parents
            or not isinstance(mechanism["parameters"], Mapping)
            or not mechanism["parameters"]
            or isinstance(mechanism["accepted_attempt"], bool)
            or not isinstance(mechanism["accepted_attempt"], int)
            or not 0 <= int(mechanism["accepted_attempt"]) < config.max_parameter_attempts
        ):
            raise ValueError(f"mechanism record for {node} is inconsistent")

    diagnostics = manifest["diagnostics"]
    seeds = manifest["seed_namespaces"]
    if not isinstance(diagnostics, Mapping) or not isinstance(seeds, Mapping):
        raise ValueError("diagnostics and seed namespaces must be mappings")
    expected_seed_keys = {
        "types",
        "order",
        "graph",
        "train",
        "test",
        "candidate",
        "mechanism_parameters",
        "mechanism_calibration",
    }
    if set(seeds) != expected_seed_keys:
        raise ValueError("accepted seed namespace schema is invalid")

    if generator_name in e7_names:
        raw_base = manifest["base_artifact"]
        if not isinstance(raw_base, Mapping):
            raise ValueError("E7 base artifact must be an embedded manifest")
        validate_mixed_dag_manifest(raw_base)
        if manifest["signal_variant"] not in {"weak", "strong"}:
            raise ValueError("E7 signal variant must be weak or strong")
        if config.experiment_id != f"e7-{manifest['signal_variant']}":
            raise ValueError("E7 configuration namespace does not match its signal variant")
        if manifest["base_artifact_fingerprint_sha256"] != raw_base.get(
            "artifact_fingerprint_sha256"
        ):
            raise ValueError("E7 base artifact fingerprint does not match its embedded manifest")
        base_seeds = raw_base["seed_namespaces"]
        for component in ("types", "order", "graph"):
            _require_seed_record(seeds[component], base_seeds[component], component)
        expected_structure = _structure_fingerprint(
            graph, order, variable_types, cardinalities
        )
        if manifest["reused_structure_fingerprint_sha256"] != expected_structure:
            raise ValueError("E7 reused-structure fingerprint is invalid")
        graph_attempt = int(diagnostics.get("base_accepted_graph_attempt", -1))
    else:
        if manifest["signal_variant"] is not None:
            raise ValueError("ordinary mixed-DAG manifests cannot name an E7 signal variant")
        graph_attempt = int(diagnostics.get("accepted_graph_attempt", -1))
        if not 0 <= graph_attempt < config.max_graph_attempts:
            raise ValueError("accepted graph attempt is invalid")
        for component in ("types", "order", "graph"):
            _require_seed_record(
                seeds[component], asdict(_seed_record(config, component, graph_attempt)), component
            )
        _validate_rejected_graph_seed_records(config, diagnostics, graph_attempt)
    for component in ("train", "test", "candidate"):
        _require_seed_record(seeds[component], asdict(_seed_record(config, component, 0)), component)
    _validate_mechanism_seed_records(config, mechanisms, diagnostics, seeds)

    inventory = manifest["seed_record_inventory"]
    expected_inventory = _seed_record_inventory(seeds, diagnostics)
    if _json_normalize(inventory) != _json_normalize(expected_inventory):
        raise ValueError("seed-record inventory is incomplete or inconsistent")
    artifacts = manifest["serialized_artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "train.csv",
        "test.csv",
        "adjacency.csv",
    }:
        raise ValueError("serialized-artifact schema is invalid")
    for filename, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "sha256",
            "bytes",
            "media_type",
            "rows",
            "columns",
        }:
            raise ValueError(f"serialized artifact {filename} has an invalid descriptor")
        if (
            not isinstance(descriptor["sha256"], str)
            or len(descriptor["sha256"]) != 64
            or not isinstance(descriptor["bytes"], int)
            or descriptor["bytes"] < 1
        ):
            raise ValueError(f"serialized artifact {filename} has an invalid hash or size")
    for name in ("train_sha256", "test_sha256"):
        if not isinstance(manifest[name], str) or len(manifest[name]) != 64:
            raise ValueError(f"{name} is not a SHA-256 digest")
    return config


def _require_seed_record(observed: object, expected: object, label: str) -> None:
    if _json_normalize(observed) != _json_normalize(expected):
        raise ValueError(f"seed record for {label} is invalid")


def _validate_rejected_graph_seed_records(
    config: MixedDAGConfig,
    diagnostics: Mapping[str, object],
    accepted_attempt: int,
) -> None:
    rejections = diagnostics.get("graph_rejections")
    if not isinstance(rejections, list) or len(rejections) != accepted_attempt:
        raise ValueError("graph rejection audit does not cover every rejected attempt")
    for attempt, rejection in enumerate(rejections):
        if not isinstance(rejection, Mapping) or rejection.get("attempt") != attempt:
            raise ValueError("graph rejection attempts must be complete and consecutive")
        records = rejection.get("seed_records")
        if not isinstance(records, Mapping) or set(records) != {"types", "order", "graph"}:
            raise ValueError("graph rejection seed schema is invalid")
        for component in ("types", "order", "graph"):
            _require_seed_record(
                records[component],
                asdict(_seed_record(config, component, attempt)),
                f"rejected graph {attempt}/{component}",
            )


def _validate_mechanism_seed_records(
    config: MixedDAGConfig,
    mechanisms: Mapping[str, object],
    diagnostics: Mapping[str, object],
    seeds: Mapping[str, object],
) -> None:
    parameters = seeds.get("mechanism_parameters")
    calibration = seeds.get("mechanism_calibration")
    rejections = diagnostics.get("parameter_rejections")
    attempts = diagnostics.get("accepted_parameter_attempt")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(calibration, Mapping)
        or not isinstance(rejections, Mapping)
        or not isinstance(attempts, Mapping)
        or set(parameters) != set(mechanisms)
        or set(calibration) != set(mechanisms)
        or set(rejections) != set(mechanisms)
        or set(attempts) != set(mechanisms)
    ):
        raise ValueError("mechanism seed/rejection audit does not cover every node")
    for node, mechanism in mechanisms.items():
        assert isinstance(mechanism, Mapping)
        accepted_attempt = int(mechanism["accepted_attempt"])
        if attempts[node] != accepted_attempt:
            raise ValueError(f"accepted mechanism attempt for {node} is inconsistent")
        _require_seed_record(
            parameters[node],
            asdict(_seed_record(config, f"parameters-{node.lower()}", accepted_attempt)),
            f"accepted parameters/{node}",
        )
        _require_seed_record(
            calibration[node],
            asdict(_seed_record(config, f"calibration-{node.lower()}", accepted_attempt)),
            f"accepted calibration/{node}",
        )
        node_rejections = rejections[node]
        if not isinstance(node_rejections, list) or len(node_rejections) != accepted_attempt:
            raise ValueError(f"mechanism rejection audit for {node} is incomplete")
        for attempt, rejection in enumerate(node_rejections):
            if not isinstance(rejection, Mapping) or rejection.get("attempt") != attempt:
                raise ValueError(f"mechanism rejection attempts for {node} are not consecutive")
            records = rejection.get("seed_records")
            if not isinstance(records, Mapping) or set(records) != {"parameters", "calibration"}:
                raise ValueError(f"mechanism rejection seed schema for {node} is invalid")
            _require_seed_record(
                records["parameters"],
                asdict(_seed_record(config, f"parameters-{node.lower()}", attempt)),
                f"rejected parameters/{node}/{attempt}",
            )
            _require_seed_record(
                records["calibration"],
                asdict(_seed_record(config, f"calibration-{node.lower()}", attempt)),
                f"rejected calibration/{node}/{attempt}",
            )


def write_mixed_dag_artifact(
    dataset: MixedDAGDataset,
    destination: str | "Path",
) -> Dict[str, object]:
    """Write a deterministic, content-addressed mixed-DAG artifact bundle."""

    from pathlib import Path

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    payloads = _mixed_dag_artifact_payloads(dataset)
    artifact_records = {
        filename: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for filename, payload in sorted(payloads.items())
    }
    audit: Dict[str, object] = {
        "schema": "ordersmc-mixed-dag-bundle-v1",
        "status": "complete",
        "artifacts": artifact_records,
        "bundle_sha256": _canonical_sha256(artifact_records),
        "manifest_payload_sha256": dataset.manifest()["artifact_fingerprint_sha256"],
    }
    for filename, payload in payloads.items():
        (target / filename).write_bytes(payload)
    (target / "artifact_audit.json").write_bytes(_pretty_json_bytes(audit))
    return audit


def replay_mixed_dag_artifact(source: str | "Path") -> MixedDAGDataset:
    """Verify raw bundle bytes, then perform complete semantic regeneration."""

    from pathlib import Path

    root = Path(source)
    try:
        audit = json.loads((root / "artifact_audit.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("mixed-DAG artifact audit is missing or invalid") from exc
    if not isinstance(audit, Mapping) or set(audit) != {
        "schema",
        "status",
        "artifacts",
        "bundle_sha256",
        "manifest_payload_sha256",
    }:
        raise ValueError("mixed-DAG artifact audit schema is invalid")
    if audit["schema"] != "ordersmc-mixed-dag-bundle-v1" or audit["status"] != "complete":
        raise ValueError("mixed-DAG artifact bundle is not complete")
    records = audit["artifacts"]
    required_files = {"train.csv", "test.csv", "adjacency.csv", "generator_manifest.json"}
    if not isinstance(records, Mapping) or set(records) != required_files:
        raise ValueError("mixed-DAG artifact file inventory is invalid")
    if audit["bundle_sha256"] != _canonical_sha256(records):
        raise ValueError("mixed-DAG artifact bundle content address is invalid")
    for filename, record in records.items():
        if not isinstance(record, Mapping) or set(record) != {"sha256", "bytes"}:
            raise ValueError(f"artifact record for {filename} is invalid")
        try:
            payload = (root / filename).read_bytes()
        except OSError as exc:
            raise ValueError(f"artifact {filename} is missing") from exc
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise RuntimeError(f"artifact {filename} does not match its frozen bytes")
    try:
        manifest = json.loads((root / "generator_manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("generator manifest is invalid JSON") from exc
    regenerated = replay_mixed_dag_dataset(manifest)
    expected_payloads = _mixed_dag_artifact_payloads(regenerated)
    for filename, expected in expected_payloads.items():
        if (root / filename).read_bytes() != expected:
            raise RuntimeError(f"artifact {filename} is not in canonical serialized form")
    expected_records = {
        filename: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        for filename, payload in sorted(expected_payloads.items())
    }
    expected_audit = {
        "schema": "ordersmc-mixed-dag-bundle-v1",
        "status": "complete",
        "artifacts": expected_records,
        "bundle_sha256": _canonical_sha256(expected_records),
        "manifest_payload_sha256": regenerated.manifest()["artifact_fingerprint_sha256"],
    }
    if _json_normalize(audit) != _json_normalize(expected_audit):
        raise RuntimeError("mixed-DAG artifact audit does not match regenerated content")
    return regenerated


def _mixed_dag_artifact_payloads(dataset: MixedDAGDataset) -> Dict[str, bytes]:
    return {
        "train.csv": _canonical_frame_bytes(dataset.train),
        "test.csv": _canonical_frame_bytes(dataset.test),
        "adjacency.csv": _canonical_adjacency_bytes(dataset.dag),
        "generator_manifest.json": _pretty_json_bytes(dataset.manifest()),
    }


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _json_normalize(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
