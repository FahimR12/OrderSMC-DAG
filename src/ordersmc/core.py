"""Order-based sequential Monte Carlo for Bayesian DAG structure learning."""

from __future__ import annotations

import math
import operator
import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .scoring import UnifiedFamilyScorer
from .config import DAGConfig


@dataclass
class OrderCandidateTable:
    nodes: List[str]
    parent_sets: List[np.ndarray]
    loglik: List[np.ndarray]
    logprior: List[np.ndarray]
    provenance: Dict[str, object] = field(default_factory=dict)
    build_diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def logevidence(self) -> List[np.ndarray]:
        """Local log marginal evidences (``loglik`` is the legacy field name)."""
        return self.loglik

    def __setstate__(self, state: Dict[str, object]) -> None:
        """Keep candidate tables pickled before provenance was added loadable."""
        self.__dict__.update(state)
        self.__dict__.setdefault("provenance", {})
        self.__dict__.setdefault("build_diagnostics", {})


@dataclass(frozen=True, eq=False, init=False)
class GraphSizeDistribution(Mapping[int, float]):
    """Normalized edge-count mass with inseparable target/support provenance."""

    probabilities: Tuple[Tuple[int, float], ...]
    target_label: str
    target_semantics: str
    support_id: str
    node_count: int
    max_indegree: int
    beta: float
    producer: str
    source_result_sha256: str | None = None
    distribution_sha256: str = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "GraphSizeDistribution is produced by an audited posterior or exact-feature result"
        )

    @classmethod
    def from_mapping(
        cls,
        probabilities: Mapping[int, float],
        *,
        target_label: str,
        target_semantics: str,
        support_id: str,
        node_count: int,
        max_indegree: int,
        beta: float,
        producer: str,
        source_result_sha256: str | None = None,
    ) -> "GraphSizeDistribution":
        del (
            probabilities,
            target_label,
            target_semantics,
            support_id,
            node_count,
            max_indegree,
            beta,
            producer,
            source_result_sha256,
        )
        raise TypeError(
            "caller-attested graph-size laws are not protocol eligible; use a computed result"
        )

    @classmethod
    def _from_computed(
        cls,
        probabilities: Mapping[int, float],
        *,
        target_label: str,
        target_semantics: str,
        support_id: str,
        node_count: int,
        max_indegree: int,
        beta: float,
        producer: str,
        source_result_sha256: str,
    ) -> "GraphSizeDistribution":
        prepared: list[tuple[int, float]] = []
        for count, mass in probabilities.items():
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise ValueError("edge counts must be integers")
            prepared.append((int(count), float(mass)))
        prepared_probabilities = tuple(sorted(prepared))
        payload = {
            "schema": "ordersmc-graph-size-distribution-v2",
            "probabilities": [
                {"edge_count": count, "probability_hex": float(mass).hex()}
                for count, mass in prepared_probabilities
            ],
            "target_label": target_label,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "node_count": node_count,
            "max_indegree": max_indegree,
            "beta_hex": float(beta).hex(),
            "producer": producer,
            "source_result_sha256": source_result_sha256,
        }
        distribution_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        instance = object.__new__(cls)
        values: Mapping[str, object] = {
            "probabilities": prepared_probabilities,
            "target_label": target_label,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "node_count": node_count,
            "max_indegree": max_indegree,
            "beta": float(beta),
            "producer": producer,
            "source_result_sha256": source_result_sha256,
            "distribution_sha256": distribution_sha256,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.validate_integrity()
        return instance

    def _identifier(self) -> str:
        payload = {
            "schema": "ordersmc-graph-size-distribution-v2",
            "probabilities": [
                {"edge_count": count, "probability_hex": float(mass).hex()}
                for count, mass in self.probabilities
            ],
            "target_label": self.target_label,
            "target_semantics": self.target_semantics,
            "support_id": self.support_id,
            "node_count": self.node_count,
            "max_indegree": self.max_indegree,
            "beta_hex": float(self.beta).hex(),
            "producer": self.producer,
            "source_result_sha256": self.source_result_sha256,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def validate_integrity(self) -> None:
        self._validate_fields()
        if (
            not isinstance(self.distribution_sha256, str)
            or len(self.distribution_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.distribution_sha256
            )
            or self.distribution_sha256 != self._identifier()
        ):
            raise ValueError("graph-size distribution failed content-hash verification")

    def _validate_fields(self) -> None:
        expected_semantics = {
            "ordersmc_weighted_order_rao_blackwellization": (
                "order",
                "candidate_restricted_order_modular",
            ),
            "exact_order_feature_dp": (
                "order",
                "candidate_restricted_order_modular",
            ),
            "exact_structure_feature_dp": (
                "structure",
                "candidate_restricted_structure_modular",
            ),
            "bidag_matched_order_empirical": (
                "order",
                "candidate_restricted_order_modular",
            ),
            "bidag_matched_partition_empirical": (
                "structure",
                "candidate_restricted_structure_modular",
            ),
        }
        if self.producer not in expected_semantics or (
            self.target_label,
            self.target_semantics,
        ) != expected_semantics[self.producer]:
            raise ValueError("graph-size producer and target semantics are invalid")
        if not isinstance(self.node_count, int) or isinstance(self.node_count, bool) or self.node_count < 1:
            raise ValueError("node_count must be a positive integer")
        if (
            not isinstance(self.max_indegree, int)
            or isinstance(self.max_indegree, bool)
            or not 0 <= self.max_indegree < self.node_count
        ):
            raise ValueError("max_indegree must be an integer in [0, node_count)")
        prefix = "candidate-sha256:"
        support_digest = self.support_id[len(prefix) :] if self.support_id.startswith(prefix) else ""
        if len(support_digest) != 64 or any(character not in "0123456789abcdef" for character in support_digest):
            raise ValueError("support_id must be a lowercase candidate SHA-256 identifier")
        if (
            not isinstance(self.source_result_sha256, str)
            or len(self.source_result_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_result_sha256)
        ):
            raise ValueError("source_result_sha256 must be a lowercase SHA-256 digest")
        beta = float(self.beta)
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be finite and lie in [0, 1]")
        cap = sum(min(rank, self.max_indegree) for rank in range(self.node_count))
        counts: set[int] = set()
        total = 0.0
        for count, mass in self.probabilities:
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= cap:
                raise ValueError(f"edge counts must be unique integers in [0, {cap}]")
            if count in counts:
                raise ValueError("graph-size probabilities contain a duplicate edge count")
            counts.add(count)
            if not math.isfinite(float(mass)) or float(mass) < 0.0:
                raise ValueError("graph-size probabilities must be finite and non-negative")
            total += float(mass)
        if not self.probabilities or not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("graph-size probabilities must sum to one")

    def __getitem__(self, count: int) -> float:
        for candidate, probability in self.probabilities:
            if candidate == count:
                return probability
        raise KeyError(count)

    def __iter__(self):
        return (count for count, _ in self.probabilities)

    def __len__(self) -> int:
        return len(self.probabilities)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphSizeDistribution):
            return (
                self.probabilities == other.probabilities
                and self.target_label == other.target_label
                and self.target_semantics == other.target_semantics
                and self.support_id == other.support_id
                and self.node_count == other.node_count
                and self.max_indegree == other.max_indegree
                and self.beta == other.beta
                and self.producer == other.producer
                and self.source_result_sha256 == other.source_result_sha256
                and self.distribution_sha256 == other.distribution_sha256
            )
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "ordersmc-graph-size-distribution-v2",
            "probabilities": [
                {"edge_count": count, "probability": probability}
                for count, probability in self.probabilities
            ],
            "target_label": self.target_label,
            "target_semantics": self.target_semantics,
            "support_id": self.support_id,
            "node_count": self.node_count,
            "max_indegree": self.max_indegree,
            "beta": self.beta,
            "producer": self.producer,
            "source_result_sha256": self.source_result_sha256,
            "distribution_sha256": self.distribution_sha256,
        }


@dataclass
class OrderSMCResult:
    orders: List[np.ndarray]
    weights: np.ndarray
    logw: np.ndarray
    best_order: np.ndarray
    best_edges: List[Tuple[str, str]]
    edge_probs: Dict[Tuple[str, str], float]
    history: List[Dict[str, object]]
    edge_mcse: Dict[Tuple[str, str], float] = field(default_factory=dict)
    effective_sample_size: float = 0.0
    run_info: Dict[str, object] = field(default_factory=dict)
    log_normalizing_constant_estimate: float = float("nan")
    log_prior_normalizing_constant_estimate: float = float("nan")
    log_evidence_ratio_estimate: float = float("nan")
    final_target_snapshots: List["OrderSMCSnapshot"] = field(default_factory=list)

    def best_dag(self):
        """Return the visited joint graph/order mode, NOT the marginal DAG MAP."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(self.best_order_names())
        graph.add_edges_from(self.best_edges)
        graph.graph["summary_semantics"] = "visited_joint_graph_order_mode_not_marginal_DAG_MAP"
        return graph

    def best_order_names(self) -> List[str]:
        nodes = self.run_info.get("nodes", [])
        return [str(nodes[int(i)]) for i in self.best_order] if nodes else []

    def posterior_dag(self, threshold: float = 0.7):
        from .graphs import dag_from_edge_probs

        nodes = self.run_info.get("nodes")
        if not nodes:
            nodes = sorted({node for edge in self.edge_probs for node in edge})
        return dag_from_edge_probs(self.edge_probs, nodes, threshold=threshold)

    def marginal_utility_dag(
        self,
        *,
        max_indegree: int | None = None,
        loss: str = "pair_state_hamming",
        time_limit_seconds: float | None = None,
        require_optimal: bool = True,
    ):
        """Return the common constrained representative DAG used by the protocol."""
        from .graphs import optimal_bayes_action_dag

        nodes = self.run_info.get("nodes")
        if not nodes:
            nodes = sorted({node for edge in self.edge_probs for node in edge})
        if max_indegree is None:
            recorded_max_indegree = self.run_info.get("candidate_max_indegree")
            if recorded_max_indegree is not None:
                max_indegree = int(recorded_max_indegree)
        return optimal_bayes_action_dag(
            self.edge_probs,
            nodes,
            max_indegree=max_indegree,
            loss=loss,
            time_limit_seconds=time_limit_seconds,
            require_optimal=require_optimal,
        )

    def marginal_utility_cpdag(self, **dag_options):
        """Convert :meth:`marginal_utility_dag` to its common CPDAG summary."""
        from .graphs import dag_to_cpdag

        return dag_to_cpdag(self.marginal_utility_dag(**dag_options))

    def _validate_posterior_table(self, table: "OrderCandidateTable") -> None:
        nodes = [str(node) for node in self.run_info.get("nodes", [])]
        if nodes and list(table.nodes) != nodes:
            raise ValueError("candidate table nodes do not match the SMC result")
        recorded = self.run_info.get("candidate_table_signature")
        if recorded is not None and str(recorded) != _candidate_table_signature(table):
            raise ValueError("candidate table content does not match the SMC result")

    def sample_posterior_dags(
        self,
        table: "OrderCandidateTable",
        *,
        draws: int,
        seed: int,
    ) -> List["PosteriorDAGSample"]:
        """Push the weighted order posterior through conditional parent draws.

        The returned draws have equal normalized weight. Each record preserves
        its source particle and order so downstream coverage/predictive code can
        audit the collapsed-state pushforward.
        """

        self._validate_posterior_table(table)
        return sample_weighted_posterior_dags(
            self.orders, self.weights, table, draws=draws, seed=seed, beta=1.0
        )

    def graph_size_distribution(
        self, table: "OrderCandidateTable"
    ) -> GraphSizeDistribution:
        """Return a provenance-bound Rao--Blackwellized law of ``|E(G)|``."""

        from .persistence import candidate_table_content_sha256

        self._validate_posterior_table(table)
        mass = weighted_graph_size_probs_for_orders(
            self.orders, self.weights, table, beta=1.0
        )
        normalized = np.asarray(self.weights, dtype=float)
        normalized = normalized / float(np.sum(normalized))
        return GraphSizeDistribution._from_computed(
            mass,
            target_label="order",
            target_semantics="candidate_restricted_order_modular",
            support_id=f"candidate-sha256:{candidate_table_content_sha256(table)}",
            node_count=len(table.nodes),
            max_indegree=candidate_max_indegree(table),
            beta=1.0,
            producer="ordersmc_weighted_order_rao_blackwellization",
            source_result_sha256=_weighted_order_source_signature(
                self.orders,
                normalized,
                _candidate_table_signature(table),
            ),
        )

    def edge_table(self):
        from .confidence import edge_confidence_table

        nodes = self.run_info.get("nodes")
        if not nodes:
            nodes = sorted({node for edge in self.edge_probs for node in edge})
        return edge_confidence_table(
            self.edge_probs,
            nodes,
            edge_mcse=self.edge_mcse,
            effective_sample_size=self.effective_sample_size,
        )

    def summary(self) -> Dict[str, object]:
        min_ess = min((float(row["ess"]) for row in self.history), default=self.effective_sample_size)
        acceptance = [
            float(row["acceptance_rate"])
            for row in self.history
            if "acceptance_rate" in row and np.isfinite(float(row["acceptance_rate"]))
        ]
        mean_acceptance = float(np.mean(acceptance)) if acceptance else float("nan")
        warnings = []
        initial_ess = self.run_info.get("initial_effective_sample_size")
        if self.orders and initial_ess is not None and float(initial_ess) < .1 * len(self.orders):
            warnings.append("Initial importance ESS is below 10%; use the uniform-start geometric bridge, not merely more likelihood temperatures.")
        if self.orders and min_ess < 0.1 * len(self.orders):
            warnings.append("Stage ESS fell below 10% of particles; consider more beta stages or particles.")
        if acceptance and mean_acceptance < 0.05:
            warnings.append("Mean MH acceptance was below 5%; order proposals may be mixing poorly.")
        if self.orders and self.effective_sample_size < 0.25 * len(self.orders):
            warnings.append("Final particle ESS was below 25%; posterior MC uncertainty may be understated.")
        root_ancestors = self.run_info.get("unique_root_ancestors")
        if self.orders and root_ancestors is not None and int(root_ancestors) < 0.25 * len(self.orders):
            warnings.append(
                "Fewer than 25% of initial particles remain as distinct root ancestors; "
                "the heuristic edge MCSE may be optimistic."
            )
        warnings.append(
            "Weight ESS and within-run edge MCSE do not establish convergence; "
            "validate posterior probabilities and graphs across independent seeds."
        )
        return {
            "particles": len(self.orders),
            "best_edges": len(self.best_edges),
            "posterior_edges_p50": int(sum(p >= 0.5 for p in self.edge_probs.values())),
            "effective_sample_size": float(self.effective_sample_size),
            "minimum_stage_ess": float(min_ess),
            "mean_acceptance_rate": mean_acceptance if np.isfinite(mean_acceptance) else None,
            "unique_root_ancestors": int(root_ancestors) if root_ancestors is not None else None,
            "genealogical_ess": self.run_info.get("genealogical_ess"),
            "initial_effective_sample_size": self.run_info.get("initial_effective_sample_size"),
            "edge_mcse_calibrated": False,
            "convergence_established": False,
            "best_edges_semantics": "visited_joint_graph_order_mode_not_marginal_DAG_MAP",
            "proposal_diagnostics": self.run_info.get("proposal_diagnostics"),
            "log_normalizing_constant_estimate": (
                float(self.log_normalizing_constant_estimate)
                if np.isfinite(self.log_normalizing_constant_estimate)
                else None
            ),
            "log_prior_normalizing_constant_estimate": (
                float(self.log_prior_normalizing_constant_estimate)
                if np.isfinite(self.log_prior_normalizing_constant_estimate)
                else None
            ),
            "log_evidence_ratio_estimate": (
                float(self.log_evidence_ratio_estimate)
                if np.isfinite(self.log_evidence_ratio_estimate)
                else None
            ),
            "final_target_snapshots": len(self.final_target_snapshots),
            "warnings": warnings,
        }

    def save(self, path, *, threshold: float = 0.7) -> None:
        from .persistence import save_result

        save_result(self, path, threshold=threshold)


@dataclass
class OrderSMCSnapshot:
    """Posterior summary recorded after the annealing path has reached beta=1.

    ``final_target_mh_sweeps`` counts *additional* full-particle MH sweeps after
    the ordinary beta=1 rejuvenation step.  Tempered distributions are never
    exposed through this snapshot interface as posterior estimates.
    """

    beta: float
    final_target_mh_sweeps: int
    edge_probs: Dict[Tuple[str, str], float]
    edge_mcse: Dict[Tuple[str, str], float]
    effective_sample_size: float
    unique_orders: int
    accepted_moves: int = 0
    proposed_moves: int = 0

    def __post_init__(self) -> None:
        if not math.isclose(float(self.beta), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("posterior snapshots are only valid at beta=1")


@dataclass(frozen=True)
class PosteriorDAGSample:
    """One auditable draw from the DAG pushforward of a weighted order result."""

    edges: Tuple[Tuple[str, str], ...]
    order: Tuple[int, ...]
    source_particle: int
    normalized_weight: float
    candidate_table_signature: str
    source_result_signature: str
    draw_seed: int
    draw_index: int
    source_particle_count: int
    target_label: str = "order"
    target_semantics: str = "candidate_restricted_order_modular_dag_pushforward"


@dataclass
class OrderSMCState:
    """Serializable local-SMC state captured at a completed stage boundary."""

    table_signature: str
    algorithm_config: Dict[str, object]
    orders: List[np.ndarray]
    weights: np.ndarray
    logw: np.ndarray
    cur_targets: np.ndarray
    current_beta: float
    next_beta_index: int
    fixed_betas: Optional[np.ndarray]
    lineage: np.ndarray
    history: List[Dict[str, object]]
    rng_state: Dict[str, object]
    log_normalizing_constant_estimate: float
    log_prior_normalizing_constant_estimate: float
    log_evidence_ratio_estimate: float
    best_order: np.ndarray
    best_edges: List[Tuple[str, str]]
    best_joint_score: float
    visited_order_states: int
    completed_tempering_stages: int
    final_target_mh_sweeps: int = 0
    final_target_snapshots: List[OrderSMCSnapshot] = field(default_factory=list)
    proposal_timing: List[Dict[str, object]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    target_evaluations: int = 0
    target_score_lookups: int = 0
    best_dag_score_lookups: int = 0
    posterior_summary_score_lookups: int = 0
    resume_validation_target_evaluations: int = 0
    resume_validation_score_lookups: int = 0


def _logsumexp(x: np.ndarray) -> float:
    if x.size == 0:
        return -math.inf
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def _normalize_logweights(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=float)
    if logw.ndim != 1:
        raise ValueError("log weights must be one-dimensional")
    if logw.size == 0:
        return np.zeros(0, dtype=float)
    if np.any(np.isnan(logw)) or np.any(np.isposinf(logw)):
        raise ValueError("log weights must not contain NaN or +inf")
    m = float(np.max(logw))
    if not np.isfinite(m):
        raise ValueError("at least one log weight must be finite")
    w = np.exp(logw - m)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("log weights could not be normalized")
    return w / s


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("resampling weights must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("resampling weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("resampling weights must have positive total mass")
    weights = weights / total
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _integer_argument(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return result


def _prepare_beta_schedule(betas: Optional[Sequence[float]]) -> np.ndarray:
    if betas is None:
        values = np.linspace(0.0, 1.0, 100, dtype=float)
    else:
        try:
            values = np.asarray(list(betas), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("betas must be a finite, one-dimensional sequence") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("betas must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)):
        raise ValueError("betas must contain only finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("betas must lie in the closed interval [0, 1]")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("betas must be strictly increasing")
    if values[0] > 0.0:
        values = np.concatenate(([0.0], values))
    if values[-1] < 1.0:
        values = np.concatenate((values, [1.0]))
    return values


def _validate_candidate_table(table: OrderCandidateTable) -> None:
    if not isinstance(table, OrderCandidateTable):
        raise TypeError("table must be an OrderCandidateTable")
    n = len(table.nodes)
    if n == 0:
        raise ValueError("candidate table must contain at least one node")
    if any(not isinstance(name, str) for name in table.nodes):
        raise ValueError("candidate-table node names must be strings")
    if len(set(table.nodes)) != n:
        raise ValueError("candidate-table node names must be unique")
    if not (len(table.parent_sets) == len(table.loglik) == len(table.logprior) == n):
        raise ValueError("candidate-table arrays must have one entry per node")
    for child_idx, (parent_sets, loglik, logprior) in enumerate(
        zip(table.parent_sets, table.loglik, table.logprior)
    ):
        parent_sets = np.asarray(parent_sets)
        loglik = np.asarray(loglik, dtype=float)
        logprior = np.asarray(logprior, dtype=float)
        if parent_sets.ndim != 2 or not np.issubdtype(parent_sets.dtype, np.integer):
            raise ValueError(f"parent sets for child {child_idx} must be a two-dimensional integer array")
        if loglik.ndim != 1 or logprior.ndim != 1:
            raise ValueError(f"scores for child {child_idx} must be one-dimensional")
        if parent_sets.shape[0] == 0 or parent_sets.shape[0] != loglik.size or loglik.size != logprior.size:
            raise ValueError(f"candidate rows and scores for child {child_idx} have inconsistent lengths")
        if np.any(parent_sets < -1) or np.any(parent_sets >= n):
            raise ValueError(f"parent sets for child {child_idx} contain an invalid node index")
        if np.any(parent_sets == child_idx):
            raise ValueError(f"parent sets for child {child_idx} contain the child itself")
        family_keys: set[Tuple[int, ...]] = set()
        for row in parent_sets:
            parents = row[row >= 0]
            if np.unique(parents).size != parents.size:
                raise ValueError(f"parent sets for child {child_idx} contain a duplicate parent")
            family_key = tuple(sorted(map(int, parents)))
            if family_key in family_keys:
                raise ValueError(f"parent sets for child {child_idx} contain a duplicate family")
            family_keys.add(family_key)
        if np.any(~np.isfinite(loglik)):
            raise ValueError(f"log-likelihood scores for child {child_idx} must be finite")
        if np.any(~np.isfinite(logprior)):
            raise ValueError(f"log-prior scores for child {child_idx} must be finite")
        empty = np.all(parent_sets < 0, axis=1)
        if not np.any(empty & np.isfinite(loglik) & np.isfinite(logprior)):
            raise ValueError(f"child {child_idx} needs a finite empty-parent candidate")


def _validate_smc_inputs(
    table: OrderCandidateTable,
    particles: object,
    betas: Optional[Sequence[float]],
    mh_steps: object,
    mh_steps_resample: Optional[object],
    resample_ess_frac: object,
    seed: object,
) -> Tuple[int, np.ndarray, int, int, float, int]:
    _validate_candidate_table(table)
    particle_count = _integer_argument("particles", particles, minimum=2)
    base_steps = _integer_argument("mh_steps", mh_steps, minimum=0)
    resample_steps = base_steps if mh_steps_resample is None else _integer_argument(
        "mh_steps_resample", mh_steps_resample, minimum=0
    )
    if isinstance(resample_ess_frac, (bool, np.bool_)):
        raise ValueError("resample_ess_frac must be a finite number in [0, 1]")
    try:
        ess_fraction = float(resample_ess_frac)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("resample_ess_frac must be a finite number in [0, 1]") from exc
    if not np.isfinite(ess_fraction) or not 0.0 <= ess_fraction <= 1.0:
        raise ValueError("resample_ess_frac must be a finite number in [0, 1]")
    random_seed = _integer_argument("seed", seed, minimum=0)
    return (
        particle_count,
        _prepare_beta_schedule(betas),
        base_steps,
        resample_steps,
        ess_fraction,
        random_seed,
    )


def _genealogy_summary(lineage: np.ndarray) -> Tuple[int, float]:
    lineage = np.asarray(lineage)
    if lineage.size == 0:
        return 0, 0.0
    _, counts = np.unique(lineage, return_counts=True)
    proportions = counts.astype(float) / float(np.sum(counts))
    return int(counts.size), float(1.0 / np.sum(proportions**2))


def _candidate_table_signature(table: OrderCandidateTable) -> str:
    """Content digest used to prevent resuming against a different score table."""
    digest = hashlib.sha256()
    digest.update(b"ordersmc-candidate-table-state-v1\0")
    for node in table.nodes:
        digest.update(str(node).encode("utf-8"))
        digest.update(b"\0")
    for arrays in (table.parent_sets, table.loglik, table.logprior):
        for value in arrays:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def candidate_max_indegree(table: OrderCandidateTable) -> int:
    """Return the largest retained parent-family size after table validation."""

    _validate_candidate_table(table)
    return max(
        (
            int(np.count_nonzero(parent_set >= 0))
            for child_sets in table.parent_sets
            for parent_set in child_sets
        ),
        default=0,
    )


_PROPOSAL_NAMES = ("swap", "adjacent_swap", "reverse")


def _normalize_proposal_weights(
    proposal_weights: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    raw: Mapping[str, float] = (
        {"swap": 0.75, "adjacent_swap": 0.25, "reverse": 0.0}
        if proposal_weights is None
        else proposal_weights
    )
    unknown = sorted(set(raw) - set(_PROPOSAL_NAMES))
    if unknown:
        raise ValueError(f"unknown order proposal types: {unknown}")
    values: Dict[str, float] = {}
    for name in _PROPOSAL_NAMES:
        value = raw.get(name, 0.0)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("proposal weights must be finite non-negative numbers")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("proposal weights must be finite non-negative numbers") from exc
        if not np.isfinite(number) or number < 0.0:
            raise ValueError("proposal weights must be finite non-negative numbers")
        values[name] = number
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("at least one proposal weight must be positive")
    return {name: values[name] / total for name in _PROPOSAL_NAMES}


def _prepare_snapshot_steps(steps: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if steps is None:
        return tuple()
    normalized = tuple(_integer_argument("final_target_snapshot_steps", x, minimum=0) for x in steps)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("final_target_snapshot_steps must be strictly increasing and unique")
    return normalized


def _conditional_ess_fraction(logw: np.ndarray, log_increment: np.ndarray) -> float:
    """Return conditional ESS divided by particle count.

    With normalized current weights W and incremental weights u, this is
    ``(sum W*u)^2 / sum W*u^2``.  It equals one for a zero-temperature
    increment irrespective of the current importance-weight ESS.
    """
    # Work directly on the log scale so extremely small, but finite, weights do
    # not underflow to zero before the CESS calculation.
    log_normalized = np.asarray(logw, dtype=float) - _logsumexp(np.asarray(logw, dtype=float))
    log_first = _logsumexp(log_normalized + log_increment)
    log_second = _logsumexp(log_normalized + 2.0 * log_increment)
    return float(np.clip(math.exp(2.0 * log_first - log_second), 0.0, 1.0))


def _adaptive_beta(
    orders: Sequence[np.ndarray],
    table: OrderCandidateTable,
    current_beta: float,
    cur_targets: np.ndarray,
    logw: np.ndarray,
    target_fraction: float,
    minimum_increment: float,
    target_evaluator: Optional[Callable[[np.ndarray, float], float]] = None,
) -> float:
    """Choose the largest next beta whose relative conditional ESS meets target."""

    def cess_at(beta: float) -> float:
        evaluate = target_evaluator
        if evaluate is None:
            evaluate = lambda order, value: order_log_target(order, table, value)
        targets = np.asarray(
            [evaluate(order, beta) for order in orders], dtype=np.float64
        )
        return _conditional_ess_fraction(logw, targets - cur_targets)

    if cess_at(1.0) >= target_fraction:
        return 1.0
    low = float(current_beta)
    high = 1.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        if cess_at(mid) >= target_fraction:
            low = mid
        else:
            high = mid
    chosen = max(
        low,
        float(current_beta) + minimum_increment,
        float(np.nextafter(float(current_beta), 1.0)),
    )
    return float(min(1.0, chosen))


def _validate_tempering_path(path: str) -> str:
    if path not in {"geometric", "collapsed_likelihood"}:
        raise ValueError("tempering_path must be 'geometric' or 'collapsed_likelihood'")
    return path


def _path_target_and_lookups(
    order: np.ndarray, table: OrderCandidateTable, beta: float, path: str
) -> Tuple[float, int]:
    """Two bridges with the same endpoint; never change conditional DAG scores.

    geometric: log gamma_t(order) = t * log gamma_1(order), gamma_0=1.
    collapsed_likelihood: temper the local evidences inside the family sums.
    """
    if path == "geometric":
        if beta == 0.0:
            return 0.0, 0
        value, lookups = _order_log_target_and_score_lookups(order, table, 1.0)
        return float(beta) * value, lookups
    return _order_log_target_and_score_lookups(order, table, beta)


def _geometric_beta(
    logw: np.ndarray, log_potential: np.ndarray, current_beta: float,
    target_fraction: float, minimum_increment: float,
) -> float:
    """Monotone CESS bisection for the geometric bridge (also used by MPI).

    The minimum increment is a fail-closed progress guard, NOT permission to
    override CESS and force a degenerating step. A failure requires a smaller
    guard or a different resource budget, not a silent jump to beta=1.
    """
    centered = np.asarray(log_potential, dtype=float) - float(np.max(log_potential))
    def cess(beta: float) -> float:
        return _conditional_ess_fraction(logw, (beta - current_beta) * centered)
    if cess(1.0) >= target_fraction:
        return 1.0
    low, high = float(current_beta), 1.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        if cess(mid) >= target_fraction:
            low = mid
        else:
            high = mid
    if low <= current_beta or low - current_beta < minimum_increment:
        raise RuntimeError(
            "CESS-preserving beta step is smaller than adaptive_min_beta_increment; "
            "lower the progress guard instead of forcing a particle-collapse step"
        )
    return low


def build_order_candidate_table_parts(
    scorer: UnifiedFamilyScorer,
    nodes: Sequence[str],
    cfg: DAGConfig,
    child_indices: Sequence[int],
    *,
    screen_k: int = 20,
    max_parents: int = 2,
    keep_per_child: Optional[int] = None,
    pair_rescue_anchor_k: int = 0,
    pair_rescue_k: int = 0,
    parent_pool_max: Optional[int] = None,
    triplet_rescue_pair_k: int = 0,
    triplet_rescue_k: int = 0,
    tail_edge_penalty_after: Optional[int] = None,
    tail_edge_penalty: Optional[float] = None,
    build_diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Build order-SMC candidate-table entries for a subset of children."""
    nodes = list(nodes)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    child_indices = [int(i) for i in child_indices]
    screen_k = max(1, int(screen_k))
    max_parents = max(0, int(max_parents))
    pair_rescue_anchor_k = max(0, int(pair_rescue_anchor_k))
    pair_rescue_k = max(0, int(pair_rescue_k))
    triplet_rescue_pair_k = max(0, int(triplet_rescue_pair_k))
    triplet_rescue_k = max(0, int(triplet_rescue_k))
    if tail_edge_penalty_after is not None:
        tail_edge_penalty_after = max(0, int(tail_edge_penalty_after))
    if tail_edge_penalty is not None:
        tail_edge_penalty = float(tail_edge_penalty)
    if keep_per_child is not None:
        keep_per_child = max(1, int(keep_per_child))
    if parent_pool_max is not None:
        parent_pool_max = max(screen_k, int(parent_pool_max))

    parent_sets: List[np.ndarray] = []
    logliks: List[np.ndarray] = []
    logpriors: List[np.ndarray] = []
    attempted_score_requests = 0
    unique_family_requests: set[Tuple[str, Tuple[str, ...]]] = set()
    retained_by_child: Dict[str, int] = {}
    excluded_nonestimable_families = 0

    def family_terms(child: str, parents: Tuple[str, ...]) -> Tuple[float, float]:
        nonlocal attempted_score_requests
        attempted_score_requests += 1
        unique_family_requests.add((child, tuple(sorted(parents))))
        res = scorer.score(child, parents)
        n_parents = len(parents)
        if tail_edge_penalty_after is None or tail_edge_penalty is None:
            edge_cost = cfg.edge_penalty * n_parents
        else:
            base_edges = min(n_parents, tail_edge_penalty_after)
            tail_edges = max(0, n_parents - tail_edge_penalty_after)
            edge_cost = cfg.edge_penalty * base_edges + tail_edge_penalty * tail_edges
        prior = -cfg.lam * int(res.k_params) - edge_cost
        return float(res.log_marginal), float(prior)

    def family_is_allowed(child: str, parents: Tuple[str, ...]) -> bool:
        if not hasattr(scorer, "family_allowed"):
            return True
        return bool(scorer.family_allowed(child, parents))

    for child_idx in child_indices:
        child = nodes[child_idx]
        eligible_parents = [
            parent
            for parent in nodes
            if parent != child
            and (
                not hasattr(scorer, "parent_allowed")
                or scorer.parent_allowed(child, parent)
            )
        ]
        empty_ll, empty_prior = family_terms(child, tuple())
        empty_obj = empty_ll + empty_prior

        single_scores: List[Tuple[float, str]] = []
        single_obj: Dict[str, float] = {}
        for p in eligible_parents:
            if not family_is_allowed(child, (p,)):
                excluded_nonestimable_families += 1
                continue
            ll, prior = family_terms(child, (p,))
            obj = ll + prior
            single_scores.append((obj - empty_obj, p))
            single_obj[p] = obj

        single_ranked = sorted(single_scores, reverse=True)
        screened = [p for _, p in single_ranked[:screen_k]]

        if max_parents >= 2 and pair_rescue_anchor_k > 0 and pair_rescue_k > 0:
            anchors = [p for _, p in single_ranked[:pair_rescue_anchor_k]]
            seen_pairs: set[Tuple[str, str]] = set()
            pair_scores: List[Tuple[float, float, Tuple[str, str]]] = []
            for anchor in anchors:
                for p in eligible_parents:
                    if p == anchor or p not in single_obj:
                        continue
                    pair = tuple(sorted((anchor, p)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if not family_is_allowed(child, pair):
                        excluded_nonestimable_families += 1
                        continue
                    ll, prior = family_terms(child, pair)
                    pair_obj = ll + prior
                    gain_over_components = pair_obj - max(empty_obj, single_obj[anchor], single_obj[p])
                    if gain_over_components > 0.0:
                        pair_scores.append((gain_over_components, pair_obj, pair))

            pool_limit = (
                parent_pool_max
                if parent_pool_max is not None
                else len(eligible_parents)
            )
            screened_set = set(screened)
            for _, _, pair in sorted(pair_scores, reverse=True)[:pair_rescue_k]:
                for p in pair:
                    if p not in screened_set and len(screened) < pool_limit:
                        screened.append(p)
                        screened_set.add(p)
                if len(screened) >= pool_limit:
                    break

        if max_parents >= 3 and triplet_rescue_pair_k > 0 and triplet_rescue_k > 0:
            pool_limit = (
                parent_pool_max
                if parent_pool_max is not None
                else len(eligible_parents)
            )
            screened_set = set(screened)
            anchor_pairs: List[Tuple[float, Tuple[str, str]]] = []
            for pair in combinations(screened, 2):
                pair = tuple(sorted(pair))
                if not family_is_allowed(child, pair):
                    excluded_nonestimable_families += 1
                    continue
                ll, prior = family_terms(child, pair)
                anchor_pairs.append((ll + prior, pair))

            triplet_scores: List[Tuple[float, float, Tuple[str, str, str]]] = []
            seen_triplets: set[Tuple[str, str, str]] = set()
            for pair_obj, pair in sorted(anchor_pairs, reverse=True)[:triplet_rescue_pair_k]:
                for p in eligible_parents:
                    if p in pair:
                        continue
                    triplet = tuple(sorted((pair[0], pair[1], p)))
                    if triplet in seen_triplets:
                        continue
                    seen_triplets.add(triplet)
                    if not family_is_allowed(child, triplet):
                        excluded_nonestimable_families += 1
                        continue
                    ll, prior = family_terms(child, triplet)
                    triplet_obj = ll + prior
                    gain_over_pair = triplet_obj - pair_obj
                    if gain_over_pair > 0.0:
                        triplet_scores.append((gain_over_pair, triplet_obj, triplet))

            for _, _, triplet in sorted(triplet_scores, reverse=True)[:triplet_rescue_k]:
                for p in triplet:
                    if p not in screened_set and len(screened) < pool_limit:
                        screened.append(p)
                        screened_set.add(p)
                if len(screened) >= pool_limit:
                    break

        scored: Dict[Tuple[str, ...], Tuple[float, float]] = {tuple(): (empty_ll, empty_prior)}
        for r in range(1, max_parents + 1):
            for ps in combinations(screened, r):
                ps = tuple(sorted(ps))
                if not family_is_allowed(child, ps):
                    excluded_nonestimable_families += 1
                    continue
                scored[ps] = family_terms(child, ps)

        ranked = sorted(scored.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)
        if keep_per_child is not None:
            ranked = ranked[:keep_per_child]
            if tuple() not in [ps for ps, _ in ranked]:
                ranked.append((tuple(), (empty_ll, empty_prior)))

        width = max(1, max((len(ps) for ps, _ in ranked), default=0))
        ps_arr = np.full((len(ranked), width), -1, dtype=np.int32)
        ll_arr = np.empty(len(ranked), dtype=np.float64)
        prior_arr = np.empty(len(ranked), dtype=np.float64)
        for i, (ps, (ll, prior)) in enumerate(ranked):
            for j, p in enumerate(ps):
                ps_arr[i, j] = node_to_idx[p]
            ll_arr[i] = ll
            prior_arr[i] = prior

        parent_sets.append(ps_arr)
        logliks.append(ll_arr)
        logpriors.append(prior_arr)
        retained_by_child[child] = len(ranked)

    if build_diagnostics is not None:
        build_diagnostics.clear()
        build_diagnostics.update(
            {
                "schema": "ordersmc-candidate-build-diagnostics-v1",
                "attempted_score_requests": int(attempted_score_requests),
                "unique_family_requests": int(len(unique_family_requests)),
                "retained_family_count": int(sum(retained_by_child.values())),
                "retained_families_by_child": dict(sorted(retained_by_child.items())),
                "fit_failures": 0,
                "excluded_nonestimable_family_requests": int(
                    excluded_nonestimable_families
                ),
                "attempted_count_semantics": (
                    "calls to UnifiedFamilyScorer.score, including repeated requests "
                    "made by screening/rescue/final enumeration"
                ),
            }
        )

    return child_indices, parent_sets, logliks, logpriors


def build_order_candidate_table(
    scorer: UnifiedFamilyScorer,
    nodes: Sequence[str],
    cfg: DAGConfig,
    *,
    screen_k: int = 20,
    max_parents: int = 2,
    keep_per_child: Optional[int] = None,
    pair_rescue_anchor_k: int = 0,
    pair_rescue_k: int = 0,
    parent_pool_max: Optional[int] = None,
    triplet_rescue_pair_k: int = 0,
    triplet_rescue_k: int = 0,
    tail_edge_penalty_after: Optional[int] = None,
    tail_edge_penalty: Optional[float] = None,
) -> OrderCandidateTable:
    """
    Precompute screened parent sets for order-SMC.

    Parent candidates are screened by single-parent local objective gain. Parent
    sets up to max_parents are then enumerated from the screened pool. Scores are
    stored as local log marginal evidence and structure-prior terms separately
    so SMC can temper only the data-evidence term. The serialized field remains
    named ``loglik`` for checkpoint compatibility.

    If pair_rescue_* is enabled, high-scoring two-parent families can add parents
    that look weak alone. This is meant for v-structures/interactions where a
    single-parent screen drops a true parent before order-SMC ever sees it.

    If triplet_rescue_* is enabled, high-scoring two-parent anchors can rescue a
    third candidate parent. tail_edge_penalty_* optionally makes the edge penalty
    non-linear, for example keeping the first two parents conservative while
    using a softer penalty for parent 3 and beyond.
    """
    nodes = list(nodes)
    build_diagnostics: Dict[str, object] = {}
    child_indices, parent_sets, logliks, logpriors = build_order_candidate_table_parts(
        scorer,
        nodes,
        cfg,
        range(len(nodes)),
        screen_k=screen_k,
        max_parents=max_parents,
        keep_per_child=keep_per_child,
        pair_rescue_anchor_k=pair_rescue_anchor_k,
        pair_rescue_k=pair_rescue_k,
        parent_pool_max=parent_pool_max,
        triplet_rescue_pair_k=triplet_rescue_pair_k,
        triplet_rescue_k=triplet_rescue_k,
        tail_edge_penalty_after=tail_edge_penalty_after,
        tail_edge_penalty=tail_edge_penalty,
        build_diagnostics=build_diagnostics,
    )
    assert child_indices == list(range(len(nodes)))
    return OrderCandidateTable(
        nodes=nodes,
        parent_sets=parent_sets,
        loglik=logliks,
        logprior=logpriors,
        build_diagnostics=build_diagnostics,
    )


def _order_log_target_and_score_lookups(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float,
) -> Tuple[float, int]:
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    total = 0.0
    score_lookups = 0
    for child_idx in range(n):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        vals = table.logprior[child_idx][ok] + float(beta) * table.loglik[child_idx][ok]
        score_lookups += int(np.count_nonzero(ok))
        total += _logsumexp(vals)
    return float(total), score_lookups


def order_log_target(order: np.ndarray, table: OrderCandidateTable, beta: float) -> float:
    return _order_log_target_and_score_lookups(order, table, beta)[0]


def _best_edges_score_and_lookups_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Tuple[List[Tuple[str, str]], float, int]:
    """Return the joint-score maximizing compatible DAG and its log score.

    Exact local-score ties are resolved independently of candidate-table row
    order.  For each tied parent set, its canonical representation is the
    sorted tuple of ``(parent_name, child_name)`` edges; the lexicographically
    smallest such tuple under Python's string ordering is selected.  Scores
    that are merely numerically close are not treated as ties.
    """
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    edges: List[Tuple[str, str]] = []
    joint_log_score = 0.0
    score_lookups = 0
    for child_idx, child in enumerate(table.nodes):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        idxs = np.flatnonzero(ok)
        if idxs.size == 0:
            raise ValueError(f"order has no compatible parent set for child {child!r}")
        score_lookups += int(idxs.size)
        vals = table.logprior[child_idx][idxs] + float(beta) * table.loglik[child_idx][idxs]
        best_local_score = float(np.max(vals))
        tied = idxs[vals == best_local_score]
        best_family_edges = min(
            tuple(
                sorted(
                    (table.nodes[int(pidx)], child)
                    for pidx in ps[int(candidate_idx)]
                    if pidx >= 0
                )
            )
            for candidate_idx in tied
        )
        joint_log_score += best_local_score
        edges.extend(best_family_edges)
    return sorted(edges), float(joint_log_score), score_lookups


def best_edges_and_score_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Tuple[List[Tuple[str, str]], float]:
    """Return the best compatible DAG and its score for one order."""

    edges, score, _ = _best_edges_score_and_lookups_for_order(order, table, beta)
    return edges, score


def best_edges_for_order(order: np.ndarray, table: OrderCandidateTable, beta: float = 1.0) -> List[Tuple[str, str]]:
    return best_edges_and_score_for_order(order, table, beta=beta)[0]


def sample_edges_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    rng: np.random.Generator,
    beta: float = 1.0,
) -> List[Tuple[str, str]]:
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    edges: List[Tuple[str, str]] = []
    for child_idx, child in enumerate(table.nodes):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        idxs = np.flatnonzero(ok)
        vals = table.logprior[child_idx][idxs] + float(beta) * table.loglik[child_idx][idxs]
        m = float(np.max(vals))
        probs = np.exp(vals - m)
        probs = probs / float(np.sum(probs))
        chosen = int(idxs[int(rng.choice(len(idxs), p=probs))])
        for pidx in ps[chosen]:
            if pidx >= 0:
                edges.append((table.nodes[int(pidx)], child))
    return sorted(edges)


def sample_weighted_posterior_dags(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    *,
    draws: int,
    seed: int,
    beta: float = 1.0,
) -> List[PosteriorDAGSample]:
    """Sample DAGs from weighted orders and exact conditional parent laws."""

    if not math.isclose(float(beta), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("posterior DAG samples are only valid at beta=1")
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if len(orders) != len(weights) or not orders:
        raise ValueError("orders and weights must be non-empty and aligned")
    _validate_candidate_table(table)
    expected_nodes = np.arange(len(table.nodes), dtype=np.int32)
    prepared_orders = [np.asarray(order, dtype=np.int32) for order in orders]
    if any(
        order.shape != expected_nodes.shape
        or not np.array_equal(np.sort(order), expected_nodes)
        for order in prepared_orders
    ):
        raise ValueError("orders must be permutations of the candidate-table nodes")
    normalized = np.asarray(weights, dtype=float)
    if normalized.ndim != 1 or np.any(~np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(normalized))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized = normalized / total
    candidate_signature = _candidate_table_signature(table)
    result_signature = _weighted_order_source_signature(
        prepared_orders, normalized, candidate_signature
    )
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    source_indices = rng.choice(len(orders), size=int(draws), replace=True, p=normalized)
    output: List[PosteriorDAGSample] = []
    equal_weight = 1.0 / int(draws)
    for draw_index, raw_index in enumerate(source_indices):
        index = int(raw_index)
        order = prepared_orders[index]
        output.append(
            PosteriorDAGSample(
                edges=tuple(sample_edges_for_order(order, table, rng, beta=beta)),
                order=tuple(map(int, order)),
                source_particle=index,
                normalized_weight=equal_weight,
                candidate_table_signature=candidate_signature,
                source_result_signature=result_signature,
                draw_seed=int(seed),
                draw_index=draw_index,
                source_particle_count=len(orders),
            )
        )
    return output


def _weighted_order_source_signature(
    orders: Sequence[np.ndarray],
    normalized_weights: np.ndarray,
    candidate_signature: str,
) -> str:
    """Hash the inferential inputs shared by collapsed posterior pushforwards."""

    signature = hashlib.sha256()
    signature.update(b"ordersmc-weighted-order-posterior-source-v1\0")
    signature.update(candidate_signature.encode("ascii"))
    signature.update(b"\0")
    for order in orders:
        signature.update(np.asarray(order, dtype="<i4").tobytes(order="C"))
    signature.update(
        np.asarray(normalized_weights, dtype="<f8").tobytes(order="C")
    )
    return signature.hexdigest()


def edge_probs_from_posterior_dag_samples(
    samples: Sequence[PosteriorDAGSample],
    table: OrderCandidateTable,
) -> Dict[Tuple[str, str], float]:
    """Summarize normalized sampled-parent DAG draws without Rao--Blackwellization."""

    if not samples:
        raise ValueError("posterior DAG samples must be non-empty")
    _validate_candidate_table(table)
    node_names = tuple(table.nodes)
    if len(node_names) < 1 or len(set(node_names)) != len(node_names):
        raise ValueError("nodes must be non-empty and unique")
    node_set = set(node_names)
    target_labels = {sample.target_label for sample in samples}
    target_semantics = {sample.target_semantics for sample in samples}
    if target_labels != {"order"} or target_semantics != {
        "candidate_restricted_order_modular_dag_pushforward"
    }:
        raise ValueError("posterior DAG samples do not share the expected target provenance")
    candidate_signature = _candidate_table_signature(table)
    if {sample.candidate_table_signature for sample in samples} != {
        candidate_signature
    }:
        raise ValueError("posterior DAG samples do not match the candidate table")
    if len({sample.source_result_signature for sample in samples}) != 1:
        raise ValueError("posterior DAG samples mix source-result signatures")
    if len({sample.draw_seed for sample in samples}) != 1:
        raise ValueError("posterior DAG samples mix draw seeds")
    if sorted(sample.draw_index for sample in samples) != list(range(len(samples))):
        raise ValueError("posterior DAG samples have invalid or duplicate draw indices")
    particle_counts = {sample.source_particle_count for sample in samples}
    if len(particle_counts) != 1 or next(iter(particle_counts)) < 1:
        raise ValueError("posterior DAG samples have inconsistent particle provenance")
    source_particle_count = next(iter(particle_counts))
    weights = np.asarray([sample.normalized_weight for sample in samples], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("posterior DAG sample weights must be finite and non-negative")
    total = float(np.sum(weights))
    if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("posterior DAG sample weights must sum to one")
    weights /= total
    mass = {
        (parent, child): 0.0
        for parent in node_names
        for child in node_names
        if parent != child
    }
    allowed_families: Dict[str, set[Tuple[str, ...]]] = {}
    for child_index, child in enumerate(node_names):
        allowed_families[child] = {
            tuple(
                sorted(
                    node_names[int(parent)]
                    for parent in row
                    if int(parent) >= 0
                )
            )
            for row in table.parent_sets[child_index]
        }
    for sample, weight in zip(samples, weights):
        if (
            sorted(sample.order) != list(range(len(node_names)))
            or sample.source_particle < 0
            or sample.source_particle >= source_particle_count
        ):
            raise ValueError("posterior DAG sample contains invalid source provenance")
        positions = {node_names[node]: index for index, node in enumerate(sample.order)}
        edge_set = set(sample.edges)
        if len(edge_set) != len(sample.edges):
            raise ValueError("posterior DAG sample contains duplicate edges")
        for parent, child in edge_set:
            if parent == child or parent not in node_set or child not in node_set:
                raise ValueError("posterior DAG sample contains an invalid edge")
            if (child, parent) in edge_set:
                raise ValueError("posterior DAG sample contains opposite directed edges")
            if positions[parent] >= positions[child]:
                raise ValueError("posterior DAG sample violates its source topological order")
            mass[(parent, child)] += float(weight)
        for child in node_names:
            family = tuple(sorted(parent for parent, target in edge_set if target == child))
            if family not in allowed_families[child]:
                raise ValueError(
                    "posterior DAG sample contains a family excluded by the candidate table"
                )
    return mass


def graph_size_probs_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> np.ndarray:
    """Exact conditional edge-count distribution for one order."""

    _validate_candidate_table(table)
    if isinstance(beta, (bool, np.bool_)):
        raise ValueError("beta must be finite and lie in [0, 1]")
    try:
        beta = float(beta)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("beta must be finite and lie in [0, 1]") from exc
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and lie in [0, 1]")
    return _graph_size_probs_for_order_validated_table(order, table, beta)


def _graph_size_probs_for_order_validated_table(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float,
) -> np.ndarray:
    """Conditional edge-count mass after one shared table validation.

    The public single-order entrypoint validates the candidate table before
    calling this helper.  Batched Rao--Blackwellization validates that same
    immutable table once and then calls the helper for every retained order.
    """

    n = len(table.nodes)
    order = np.asarray(order, dtype=np.int32)
    if order.shape != (n,) or not np.array_equal(
        np.sort(order), np.arange(n, dtype=np.int32)
    ):
        raise ValueError("order must be a permutation of the candidate-table nodes")
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    distribution = np.asarray([1.0], dtype=float)
    for child_idx in range(n):
        parent_sets = table.parent_sets[child_idx]
        compatible = np.all(
            (parent_sets < 0)
            | (pos[np.maximum(parent_sets, 0)] < pos[child_idx]),
            axis=1,
        )
        indices = np.flatnonzero(compatible)
        if indices.size == 0:
            raise ValueError("order has no compatible parent set for one child")
        values = (
            np.asarray(table.logprior[child_idx], dtype=float)[indices]
            + float(beta) * np.asarray(table.loglik[child_idx], dtype=float)[indices]
        )
        values -= float(np.max(values))
        probabilities = np.exp(values)
        probabilities /= float(np.sum(probabilities))
        sizes = np.count_nonzero(parent_sets[indices] >= 0, axis=1)
        local = np.zeros(int(np.max(sizes, initial=0)) + 1, dtype=float)
        for size, probability in zip(sizes, probabilities):
            local[int(size)] += float(probability)
        distribution = np.convolve(distribution, local)
    distribution /= float(np.sum(distribution))
    return distribution


def weighted_graph_size_probs_for_orders(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Dict[int, float]:
    """Rao--Blackwellize the graph-size posterior over weighted orders."""

    if len(orders) != len(weights) or not orders:
        raise ValueError("orders and weights must be non-empty and aligned")
    _validate_candidate_table(table)
    if isinstance(beta, (bool, np.bool_)):
        raise ValueError("beta must be finite and lie in [0, 1]")
    try:
        beta = float(beta)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("beta must be finite and lie in [0, 1]") from exc
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and lie in [0, 1]")
    # Copy even when ``weights`` is already a float ndarray.  Normalizing a
    # view in place would otherwise mutate the caller's SMC result and change
    # later summaries/replay digests merely by asking for graph-size mass.
    normalized = np.asarray(weights, dtype=float).copy()
    if normalized.ndim != 1 or np.any(~np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(normalized))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized /= total
    maximum_edges = sum(
        max((int(np.count_nonzero(row >= 0)) for row in child_sets), default=0)
        for child_sets in table.parent_sets
    )
    mass = np.zeros(maximum_edges + 1, dtype=float)
    cache: Dict[Tuple[int, ...], np.ndarray] = {}
    for order, weight in zip(orders, normalized):
        key = tuple(map(int, order))
        conditional = cache.get(key)
        if conditional is None:
            conditional = _graph_size_probs_for_order_validated_table(
                order, table, beta
            )
            cache[key] = conditional
        mass[: len(conditional)] += float(weight) * conditional
    mass /= float(np.sum(mass))
    # The allocation bound above is the sum of per-child family maxima.  It can
    # exceed the largest edge count of any acyclic graph (for example d=3 with
    # maximum indegree one), leaving exact trailing zeros.  Omit zero-mass
    # coefficients so consumers can safely zero-fill their own protocol grid.
    return {
        edge_count: float(probability)
        for edge_count, probability in enumerate(mass)
        if probability > 0.0
    }


def _edge_probs_and_lookups_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Tuple[Dict[Tuple[str, str], float], int]:
    """
    Exact conditional edge probabilities given an ordering.

    Parent-set choices are conditionally independent by child once the order is
    fixed, so we can analytically average over every compatible parent set
    instead of sampling one graph from the order.
    """
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    edge_mass: Dict[Tuple[str, str], float] = {}
    score_lookups = 0
    for child_idx, child in enumerate(table.nodes):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        idxs = np.flatnonzero(ok)
        score_lookups += int(idxs.size)
        vals = table.logprior[child_idx][idxs] + float(beta) * table.loglik[child_idx][idxs]
        m = float(np.max(vals))
        probs = np.exp(vals - m)
        probs = probs / float(np.sum(probs))
        for idx, prob in zip(idxs, probs):
            if prob <= 0.0:
                continue
            for pidx in ps[int(idx)]:
                if pidx >= 0:
                    edge = (table.nodes[int(pidx)], child)
                    edge_mass[edge] = edge_mass.get(edge, 0.0) + float(prob)
    return edge_mass, score_lookups


def edge_probs_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """Return exact conditional edge probabilities given an ordering."""

    return _edge_probs_and_lookups_for_order(order, table, beta)[0]


def weighted_edge_probs_for_orders(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    edge_mass, _, _ = weighted_edge_summary_for_orders(orders, weights, table, beta=beta)
    return edge_mass


def weighted_edge_summary_for_orders(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], float]:
    """Return Rao-Blackwellized edge means and heuristic particle MCSEs.

    The weighted variance uses the usual ``1 - sum(w**2)`` finite-sample
    correction after normalizing the weights.  Dividing that variance by the
    importance-weight ESS is appropriate for independent weighted particles,
    but SMC resampling ancestry and MH dependence are deliberately not modeled;
    callers should therefore treat these MCSEs as heuristic diagnostics.
    """
    edge_mass, edge_mcse, effective_sample_size, _ = (
        _weighted_edge_summary_and_lookups_for_orders(
            orders, weights, table, beta=beta
        )
    )
    return edge_mass, edge_mcse, effective_sample_size


def _weighted_edge_summary_and_lookups_for_orders(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Tuple[
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], float],
    float,
    int,
]:
    """Internal summary plus the number of local-family score rows read."""

    if len(orders) != len(weights):
        raise ValueError("orders and weights must have the same length")
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be a one-dimensional finite non-negative array")
    weight_total = float(np.sum(weight_array))
    if weight_array.size == 0 or weight_total <= 0.0:
        raise ValueError("weights must have positive total mass")
    weight_array = weight_array / weight_total

    edge_mass: Dict[Tuple[str, str], float] = {}
    edge_second_moment: Dict[Tuple[str, str], float] = {}
    # Resampling commonly creates duplicate orders. Their exact conditional
    # edge probabilities are identical, so calculate each distinct order once.
    conditional_cache: Dict[Tuple[int, ...], Dict[Tuple[str, str], float]] = {}
    score_lookups = 0
    for order, w in zip(orders, weight_array):
        if float(w) <= 0.0:
            continue
        order_key = tuple(map(int, order))
        conditional = conditional_cache.get(order_key)
        if conditional is None:
            conditional, local_lookups = _edge_probs_and_lookups_for_order(
                order, table, beta=beta
            )
            score_lookups += local_lookups
            conditional_cache[order_key] = conditional
        for edge, prob in conditional.items():
            edge_mass[edge] = edge_mass.get(edge, 0.0) + float(w) * prob
            edge_second_moment[edge] = edge_second_moment.get(edge, 0.0) + float(w) * prob * prob
    sum_squared_weights = float(np.sum(weight_array**2))
    effective_sample_size = float(1.0 / sum_squared_weights)
    variance_correction = 1.0 - sum_squared_weights
    if variance_correction <= np.finfo(float).eps:
        edge_mcse = {edge: float("nan") for edge in edge_mass}
    else:
        edge_mcse = {
            edge: math.sqrt(
                max(0.0, edge_second_moment[edge] - mean * mean)
                / variance_correction
                * sum_squared_weights
            )
            for edge, mean in edge_mass.items()
        }
    return edge_mass, edge_mcse, effective_sample_size, score_lookups


def _draw_proposal_name(
    rng: np.random.Generator, proposal_weights: Mapping[str, float]
) -> str:
    draw = float(rng.random())
    cumulative = 0.0
    for name in _PROPOSAL_NAMES:
        cumulative += float(proposal_weights[name])
        if draw < cumulative:
            return name
    return _PROPOSAL_NAMES[-1]


def _propose_order(
    order: np.ndarray,
    rng: np.random.Generator,
    proposal_name: str = "swap",
) -> np.ndarray:
    """Apply one of the symmetric order proposals used by the MH kernel."""
    new = order.copy()
    n = len(new)
    if n < 2:
        return new
    if proposal_name == "swap":
        i, j = rng.choice(n, size=2, replace=False)
        new[i], new[j] = new[j], new[i]
    elif proposal_name == "adjacent_swap":
        i = int(rng.integers(0, n - 1))
        new[i], new[i + 1] = new[i + 1], new[i]
    elif proposal_name == "reverse":
        i, j = sorted(map(int, rng.choice(n, size=2, replace=False)))
        new[i : j + 1] = new[i : j + 1][::-1]
    else:
        raise ValueError(f"unknown order proposal type {proposal_name!r}")
    return new


def order_smc(
    table: OrderCandidateTable,
    *,
    particles: int = 200,
    betas: Optional[Sequence[float]] = None,
    mh_steps: int = 20,
    mh_steps_resample: Optional[int] = None,
    resample_ess_frac: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
    proposal_weights: Optional[Mapping[str, float]] = None,
    resampling: str = "systematic",
    tempering_path: str = "collapsed_likelihood",
    adaptive_cess_target: Optional[float] = None,
    adaptive_min_beta_increment: float = 1e-8,
    adaptive_max_stages: int = 1000,
    final_target_snapshot_steps: Optional[Sequence[int]] = None,
    resume_state: Optional[OrderSMCState] = None,
    checkpoint_callback: Optional[Callable[[OrderSMCState], None]] = None,
) -> OrderSMCResult:
    particles, prepared_betas, mh_steps, mh_steps_resample, resample_ess_frac, seed = _validate_smc_inputs(
        table,
        particles,
        betas,
        mh_steps,
        mh_steps_resample,
        resample_ess_frac,
        seed,
    )
    if resampling not in {"systematic", "none"}:
        raise ValueError("resampling must be 'systematic' or 'none'")
    normalized_proposals = _normalize_proposal_weights(proposal_weights)
    tempering_path = _validate_tempering_path(tempering_path)
    snapshot_steps = _prepare_snapshot_steps(final_target_snapshot_steps)
    adaptive_stages = _integer_argument("adaptive_max_stages", adaptive_max_stages, minimum=1)
    if isinstance(adaptive_min_beta_increment, (bool, np.bool_)):
        raise ValueError("adaptive_min_beta_increment must be in (0, 1]")
    try:
        minimum_beta_increment = float(adaptive_min_beta_increment)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("adaptive_min_beta_increment must be in (0, 1]") from exc
    if not np.isfinite(minimum_beta_increment) or not 0.0 < minimum_beta_increment <= 1.0:
        raise ValueError("adaptive_min_beta_increment must be in (0, 1]")
    if adaptive_cess_target is None:
        fixed_betas: Optional[np.ndarray] = prepared_betas
        cess_target = None
    else:
        if betas is not None:
            raise ValueError("betas and adaptive_cess_target are mutually exclusive")
        if isinstance(adaptive_cess_target, (bool, np.bool_)):
            raise ValueError("adaptive_cess_target must be in (0, 1)")
        try:
            cess_target = float(adaptive_cess_target)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("adaptive_cess_target must be in (0, 1)") from exc
        if not np.isfinite(cess_target) or not 0.0 < cess_target < 1.0:
            raise ValueError("adaptive_cess_target must be in (0, 1)")
        fixed_betas = None

    algorithm_config: Dict[str, object] = {
        "particles": particles,
        "fixed_betas": None if fixed_betas is None else fixed_betas.tolist(),
        "mh_steps": mh_steps,
        "mh_steps_resample": mh_steps_resample,
        "resample_ess_frac": resample_ess_frac,
        "seed": seed,
        "proposal_weights": dict(normalized_proposals),
        "resampling": resampling,
        "tempering_path": tempering_path,
        "adaptive_cess_target": cess_target,
        "adaptive_min_beta_increment": minimum_beta_increment,
        "adaptive_max_stages": adaptive_stages,
        "final_target_snapshot_steps": list(snapshot_steps),
    }

    n = len(table.nodes)
    table_signature = _candidate_table_signature(table)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    segment_started = time.perf_counter()
    target_evaluations = 0
    target_score_lookups = 0
    best_dag_score_lookups = 0
    posterior_summary_score_lookups = 0
    resume_validation_target_evaluations = 0
    resume_validation_score_lookups = 0

    def counted_target(order: np.ndarray, beta: float) -> float:
        nonlocal target_evaluations, target_score_lookups
        value, lookups = _path_target_and_lookups(order, table, beta, tempering_path)
        target_evaluations += 1
        target_score_lookups += lookups
        return value

    if resume_state is None:
        orders = [rng.permutation(n).astype(np.int32) for _ in range(particles)]
        cur_targets = np.array(
            [counted_target(order, 0.0) for order in orders], dtype=np.float64
        )
        # q_0 is uniform over n! orders.  The common n! factor is omitted from
        # self-normalized particle weights but restored in the absolute Z_0
        # estimate below.
        logw = cur_targets.copy()
        initial_ess = float(1.0 / np.sum(_normalize_logweights(logw) ** 2))
        log_prior_normalizer = (
            math.lgamma(n + 1.0) + _logsumexp(logw) - math.log(float(particles))
        )
        log_normalizer = float(log_prior_normalizer)
        log_evidence_ratio = 0.0
        history: List[Dict[str, object]] = []
        lineage = np.arange(particles, dtype=np.int64)
        best_order: Optional[np.ndarray] = None
        best_edges: List[Tuple[str, str]] = []
        best_joint_score = -math.inf
        visited_order_states = len(orders)
        current_beta = 0.0
        next_beta_index = 1
        completed_tempering_stages = 0
        completed_final_sweeps = 0
        snapshots: List[OrderSMCSnapshot] = []
        proposal_timing: List[Dict[str, object]] = []
        elapsed_offset = 0.0
    else:
        if not isinstance(resume_state, OrderSMCState):
            raise TypeError("resume_state must be an OrderSMCState")
        if resume_state.table_signature != table_signature:
            raise ValueError("SMC checkpoint was created for a different candidate table")
        if resume_state.algorithm_config != algorithm_config:
            raise ValueError("SMC checkpoint settings do not match the requested sampler settings")
        if (resume_state.fixed_betas is None) != (fixed_betas is None):
            raise ValueError("SMC checkpoint tempering schedule is inconsistent")
        if fixed_betas is not None and not np.array_equal(
            np.asarray(resume_state.fixed_betas, dtype=float), fixed_betas
        ):
            raise ValueError("SMC checkpoint tempering schedule is inconsistent")
        if len(resume_state.orders) != particles:
            raise ValueError("SMC checkpoint particle count is inconsistent")
        orders = [np.asarray(order, dtype=np.int32).copy() for order in resume_state.orders]
        logw = np.asarray(resume_state.logw, dtype=np.float64).copy()
        stored_weights = np.asarray(resume_state.weights, dtype=np.float64)
        cur_targets = np.asarray(resume_state.cur_targets, dtype=np.float64).copy()
        lineage = np.asarray(resume_state.lineage, dtype=np.int64).copy()
        if not (logw.shape == cur_targets.shape == lineage.shape == (particles,)):
            raise ValueError("SMC checkpoint arrays have inconsistent dimensions")
        expected_nodes = np.arange(n, dtype=np.int32)
        if any(
            order.shape != (n,) or not np.array_equal(np.sort(order), expected_nodes)
            for order in orders
        ):
            raise ValueError("SMC checkpoint contains an invalid order permutation")
        if np.any(lineage < 0) or np.any(lineage >= particles):
            raise ValueError("SMC checkpoint lineage contains an invalid root ancestor")
        if stored_weights.shape != (particles,) or not np.allclose(
            stored_weights, _normalize_logweights(logw), atol=1e-15, rtol=1e-13
        ):
            raise ValueError("SMC checkpoint normalized weights are inconsistent with log weights")
        current_beta = float(resume_state.current_beta)
        if not np.isfinite(current_beta) or not 0.0 <= current_beta <= 1.0:
            raise ValueError("SMC checkpoint beta is invalid")
        next_beta_index = int(resume_state.next_beta_index)
        if fixed_betas is not None:
            if not 1 <= next_beta_index <= len(fixed_betas):
                raise ValueError("SMC checkpoint fixed-schedule position is invalid")
            if not math.isclose(
                current_beta,
                float(fixed_betas[next_beta_index - 1]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("SMC checkpoint beta does not match its fixed-schedule position")
        validation_targets_and_lookups = [
            _path_target_and_lookups(order, table, current_beta, tempering_path)
            for order in orders
        ]
        expected_targets = np.asarray(
            [value for value, _ in validation_targets_and_lookups], dtype=np.float64
        )
        if not np.allclose(cur_targets, expected_targets, atol=1e-10, rtol=1e-12):
            raise ValueError("SMC checkpoint targets do not match its orders and beta")
        history = copy.deepcopy(resume_state.history)
        initial_ess = float(history[0].get("initial_effective_sample_size", float("nan"))) if history else float("nan")
        log_normalizer = float(resume_state.log_normalizing_constant_estimate)
        log_prior_normalizer = float(resume_state.log_prior_normalizing_constant_estimate)
        log_evidence_ratio = float(resume_state.log_evidence_ratio_estimate)
        best_order = np.asarray(resume_state.best_order, dtype=np.int32).copy()
        best_edges = list(resume_state.best_edges)
        best_joint_score = float(resume_state.best_joint_score)
        visited_order_states = int(resume_state.visited_order_states)
        completed_tempering_stages = int(resume_state.completed_tempering_stages)
        completed_final_sweeps = int(resume_state.final_target_mh_sweeps)
        snapshots = copy.deepcopy(resume_state.final_target_snapshots)
        proposal_timing = copy.deepcopy(getattr(resume_state, "proposal_timing", []))
        elapsed_offset = float(getattr(resume_state, "elapsed_seconds", 0.0))
        if not np.isfinite(elapsed_offset) or elapsed_offset < 0.0:
            raise ValueError("SMC checkpoint elapsed timing is invalid")
        target_evaluations = int(getattr(resume_state, "target_evaluations", 0))
        target_score_lookups = int(getattr(resume_state, "target_score_lookups", 0))
        best_dag_score_lookups = int(
            getattr(resume_state, "best_dag_score_lookups", 0)
        )
        posterior_summary_score_lookups = int(
            getattr(resume_state, "posterior_summary_score_lookups", 0)
        )
        resume_validation_target_evaluations = int(
            getattr(resume_state, "resume_validation_target_evaluations", 0)
        ) + len(validation_targets_and_lookups)
        resume_validation_score_lookups = int(
            getattr(resume_state, "resume_validation_score_lookups", 0)
        ) + sum(lookups for _, lookups in validation_targets_and_lookups)
        if any(
            value < 0
            for value in (
                target_evaluations,
                target_score_lookups,
                best_dag_score_lookups,
                posterior_summary_score_lookups,
                resume_validation_target_evaluations,
                resume_validation_score_lookups,
            )
        ):
            raise ValueError("SMC checkpoint resource counters are invalid")
        rng.bit_generator.state = copy.deepcopy(resume_state.rng_state)

    best_cache: Dict[Tuple[int, ...], Tuple[List[Tuple[str, str]], float]] = {}

    def consider_for_best(order: np.ndarray) -> None:
        nonlocal best_order, best_edges, best_joint_score, best_dag_score_lookups
        order_key = tuple(map(int, order))
        candidate = best_cache.get(order_key)
        if candidate is None:
            candidate_edges, candidate_score, lookups = (
                _best_edges_score_and_lookups_for_order(order, table, beta=1.0)
            )
            candidate = (candidate_edges, candidate_score)
            best_dag_score_lookups += lookups
            if len(best_cache) < 4096:
                best_cache[order_key] = candidate
        candidate_edges, candidate_score = candidate
        candidate_tie_key = (tuple(sorted(candidate_edges)), order_key)
        current_tie_key = (
            tuple(sorted(best_edges)),
            tuple(map(int, best_order)) if best_order is not None else tuple(),
        )
        if candidate_score > best_joint_score or (
            candidate_score == best_joint_score and candidate_tie_key < current_tie_key
        ):
            best_order = order.copy()
            best_edges = candidate_edges
            best_joint_score = candidate_score

    if resume_state is None:
        for order in orders:
            consider_for_best(order)

    def make_state() -> OrderSMCState:
        if best_order is None:
            raise RuntimeError("SMC did not visit any order")
        return OrderSMCState(
            table_signature=table_signature,
            algorithm_config=copy.deepcopy(algorithm_config),
            orders=[order.copy() for order in orders],
            weights=_normalize_logweights(logw),
            logw=logw.copy(),
            cur_targets=cur_targets.copy(),
            current_beta=float(current_beta),
            next_beta_index=int(next_beta_index),
            fixed_betas=None if fixed_betas is None else fixed_betas.copy(),
            lineage=lineage.copy(),
            history=copy.deepcopy(history),
            rng_state=copy.deepcopy(rng.bit_generator.state),
            log_normalizing_constant_estimate=float(log_normalizer),
            log_prior_normalizing_constant_estimate=float(log_prior_normalizer),
            log_evidence_ratio_estimate=float(log_evidence_ratio),
            best_order=best_order.copy(),
            best_edges=list(best_edges),
            best_joint_score=float(best_joint_score),
            visited_order_states=int(visited_order_states),
            completed_tempering_stages=int(completed_tempering_stages),
            final_target_mh_sweeps=int(completed_final_sweeps),
            final_target_snapshots=copy.deepcopy(snapshots),
            proposal_timing=copy.deepcopy(proposal_timing),
            elapsed_seconds=float(elapsed_offset + time.perf_counter() - segment_started),
            target_evaluations=int(target_evaluations),
            target_score_lookups=int(target_score_lookups),
            best_dag_score_lookups=int(best_dag_score_lookups),
            posterior_summary_score_lookups=int(posterior_summary_score_lookups),
            resume_validation_target_evaluations=int(
                resume_validation_target_evaluations
            ),
            resume_validation_score_lookups=int(resume_validation_score_lookups),
        )

    def emit_checkpoint() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(make_state())

    def rejuvenate(
        steps: int,
        target_for: Callable[[np.ndarray], float],
    ) -> Tuple[int, int, Dict[str, int], Dict[str, int], Dict[str, float]]:
        nonlocal visited_order_states
        accepted_by_type = {name: 0 for name in _PROPOSAL_NAMES}
        proposed_by_type = {name: 0 for name in _PROPOSAL_NAMES}
        accepted = 0
        proposed = 0
        elapsed_by_type = {name: 0.0 for name in _PROPOSAL_NAMES}
        for particle_index, initial_order in enumerate(orders):
            order = initial_order
            target = float(cur_targets[particle_index])
            for _ in range(steps):
                proposal_name = _draw_proposal_name(rng, normalized_proposals)
                proposal_started = time.perf_counter()
                proposed += 1
                proposed_by_type[proposal_name] += 1
                prop = _propose_order(order, rng, proposal_name)
                prop_target = target_for(prop)
                if math.log(float(rng.random())) < min(0.0, prop_target - target):
                    accepted += 1
                    accepted_by_type[proposal_name] += 1
                    order = prop
                    target = prop_target
                    visited_order_states += 1
                    consider_for_best(order)
                elapsed_by_type[proposal_name] += time.perf_counter() - proposal_started
            orders[particle_index] = order
            cur_targets[particle_index] = target
        return accepted, proposed, accepted_by_type, proposed_by_type, elapsed_by_type

    def append_proposal_timing(
        *,
        phase: str,
        beta: float,
        started_elapsed_seconds: float,
        accepted_by_type: Mapping[str, int],
        proposed_by_type: Mapping[str, int],
        elapsed_by_type: Mapping[str, float],
    ) -> None:
        completed_elapsed_seconds = float(
            elapsed_offset + time.perf_counter() - segment_started
        )
        proposal_timing.append(
            {
                "phase": phase,
                "beta": float(beta),
                "started_elapsed_seconds": float(started_elapsed_seconds),
                "completed_elapsed_seconds": completed_elapsed_seconds,
                "proposal_elapsed_seconds": {
                    name: float(elapsed_by_type[name]) for name in _PROPOSAL_NAMES
                },
                "proposal_counts": {
                    name: int(proposed_by_type[name]) for name in _PROPOSAL_NAMES
                },
                "acceptance_counts": {
                    name: int(accepted_by_type[name]) for name in _PROPOSAL_NAMES
                },
            }
        )

    def add_proposal_diagnostics(
        row: Dict[str, object],
        accepted: int,
        proposed: int,
        accepted_by_type: Mapping[str, int],
        proposed_by_type: Mapping[str, int],
    ) -> None:
        row["acceptance_rate"] = float(accepted / proposed) if proposed else float("nan")
        row["accepted_moves"] = float(accepted)
        row["proposed_moves"] = float(proposed)
        for name in _PROPOSAL_NAMES:
            proposal_count = int(proposed_by_type[name])
            acceptance_count = int(accepted_by_type[name])
            row[f"{name}_proposed"] = float(proposal_count)
            row[f"{name}_accepted"] = float(acceptance_count)
            row[f"{name}_acceptance_rate"] = (
                float(acceptance_count / proposal_count) if proposal_count else float("nan")
            )

    while current_beta < 1.0 - 1e-15:
        stage_target_evaluations_start = target_evaluations
        stage_target_score_lookups_start = target_score_lookups
        if fixed_betas is None:
            if completed_tempering_stages >= adaptive_stages:
                raise RuntimeError("adaptive tempering exceeded adaptive_max_stages before beta=1")
            assert cess_target is not None
            if tempering_path == "geometric":
                potentials = np.asarray([counted_target(o, 1.0) for o in orders])
                beta = _geometric_beta(
                    logw, potentials, current_beta, cess_target, minimum_beta_increment
                )
            else:
                beta = _adaptive_beta(
                    orders, table, current_beta, cur_targets, logw, cess_target,
                    minimum_beta_increment, target_evaluator=counted_target,
                )
        else:
            if next_beta_index >= len(fixed_betas):
                raise RuntimeError("fixed beta schedule ended before beta=1")
            beta = float(fixed_betas[next_beta_index])

        target_cache: Dict[Tuple[int, ...], float] = {}

        def target_for(order: np.ndarray) -> float:
            order_key = tuple(map(int, order))
            value = target_cache.get(order_key)
            if value is None:
                value = counted_target(order, beta)
                target_cache[order_key] = value
            return value

        new_targets = np.array([target_for(o) for o in orders], dtype=np.float64)
        log_incremental_weights = new_targets - cur_targets
        conditional_ess_fraction = _conditional_ess_fraction(logw, log_incremental_weights)
        old_log_weight_sum = _logsumexp(logw)
        logw += log_incremental_weights
        log_normalizer_increment = _logsumexp(logw) - old_log_weight_sum
        log_normalizer += log_normalizer_increment
        log_evidence_ratio += log_normalizer_increment
        cur_targets = new_targets
        current_beta = beta
        completed_tempering_stages += 1
        if fixed_betas is not None:
            next_beta_index += 1

        weights = _normalize_logweights(logw)
        ess = float(1.0 / np.sum(weights**2))
        best_idx = int(np.argmax(cur_targets))
        history.append(
            {
                "phase": "tempering",
                "initial_effective_sample_size": initial_ess,
                "beta": beta,
                "ess": ess,
                "conditional_ess_fraction": conditional_ess_fraction,
                "log_normalizing_constant_increment": log_normalizer_increment,
                "log_normalizing_constant_estimate": log_normalizer,
                "log_evidence_ratio_estimate": log_evidence_ratio if tempering_path == "collapsed_likelihood" else None,
                "log_path_normalizer_ratio_estimate": log_evidence_ratio,
                "max_log_target": float(cur_targets[best_idx]),
                "mean_log_target": float(np.mean(cur_targets)),
                "resampled": 0.0,
            }
        )
        if verbose:
            print(
                f"[order-smc beta={beta:.4f}] ESS={ess:.1f}/{len(orders)} "
                f"max_log_target={cur_targets[best_idx]:.3f}",
                flush=True,
            )

        did_resample = (
            resampling == "systematic"
            and resample_ess_frac > 0.0
            and ess < float(resample_ess_frac) * len(orders)
        )
        if did_resample:
            anc = _systematic_resample(weights, rng)
            orders = [orders[int(i)].copy() for i in anc]
            cur_targets = cur_targets[anc].copy()
            lineage = lineage[anc].copy()
            logw = np.zeros(len(orders), dtype=np.float64)
            history[-1]["resampled"] = 1.0
            if verbose:
                print("  -> resampled orders", flush=True)

        steps = int(mh_steps_resample if did_resample else mh_steps)
        timing_started = float(elapsed_offset + time.perf_counter() - segment_started)
        accepted, proposed, accepted_by_type, proposed_by_type, elapsed_by_type = rejuvenate(
            steps, target_for
        )
        append_proposal_timing(
            phase="tempering",
            beta=beta,
            started_elapsed_seconds=timing_started,
            accepted_by_type=accepted_by_type,
            proposed_by_type=proposed_by_type,
            elapsed_by_type=elapsed_by_type,
        )
        add_proposal_diagnostics(
            history[-1], accepted, proposed, accepted_by_type, proposed_by_type
        )
        unique_roots, genealogical_ess = _genealogy_summary(lineage)
        history[-1]["unique_root_ancestors"] = float(unique_roots)
        history[-1]["genealogical_ess"] = float(genealogical_ess)
        history[-1]["unique_orders"] = float(len({tuple(map(int, order)) for order in orders}))
        history[-1]["target_evaluations"] = float(
            target_evaluations - stage_target_evaluations_start
        )
        history[-1]["target_score_lookups"] = float(
            target_score_lookups - stage_target_score_lookups_start
        )
        emit_checkpoint()

    if not math.isclose(current_beta, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("SMC terminated without reaching beta=1")

    existing_snapshot_steps = {snapshot.final_target_mh_sweeps for snapshot in snapshots}
    for requested_sweeps in snapshot_steps:
        if requested_sweeps in existing_snapshot_steps:
            continue
        if requested_sweeps < completed_final_sweeps:
            raise ValueError("SMC checkpoint is missing an earlier requested posterior snapshot")
        extra_sweeps = requested_sweeps - completed_final_sweeps
        stage_target_evaluations_start = target_evaluations
        stage_target_score_lookups_start = target_score_lookups
        accepted = 0
        proposed = 0
        accepted_by_type = {name: 0 for name in _PROPOSAL_NAMES}
        proposed_by_type = {name: 0 for name in _PROPOSAL_NAMES}
        elapsed_by_type = {name: 0.0 for name in _PROPOSAL_NAMES}
        timing_started = float(elapsed_offset + time.perf_counter() - segment_started)
        if extra_sweeps:
            target_cache: Dict[Tuple[int, ...], float] = {}

            def final_target_for(order: np.ndarray) -> float:
                order_key = tuple(map(int, order))
                value = target_cache.get(order_key)
                if value is None:
                    value = counted_target(order, 1.0)
                    target_cache[order_key] = value
                return value

            accepted, proposed, accepted_by_type, proposed_by_type, elapsed_by_type = rejuvenate(
                extra_sweeps, final_target_for
            )
        append_proposal_timing(
            phase="final_target",
            beta=1.0,
            started_elapsed_seconds=timing_started,
            accepted_by_type=accepted_by_type,
            proposed_by_type=proposed_by_type,
            elapsed_by_type=elapsed_by_type,
        )
        completed_final_sweeps = requested_sweeps
        weights = _normalize_logweights(logw)
        (
            snapshot_edges,
            snapshot_mcse,
            snapshot_ess,
            snapshot_score_lookups,
        ) = _weighted_edge_summary_and_lookups_for_orders(
            orders, weights, table, beta=1.0
        )
        posterior_summary_score_lookups += snapshot_score_lookups
        snapshot = OrderSMCSnapshot(
            beta=1.0,
            final_target_mh_sweeps=requested_sweeps,
            edge_probs=snapshot_edges,
            edge_mcse=snapshot_mcse,
            effective_sample_size=snapshot_ess,
            unique_orders=len({tuple(map(int, order)) for order in orders}),
            accepted_moves=accepted,
            proposed_moves=proposed,
        )
        snapshots.append(snapshot)
        existing_snapshot_steps.add(requested_sweeps)
        row: Dict[str, object] = {
            "phase": "final_target",
            "beta": 1.0,
            "final_target_mh_sweeps": float(requested_sweeps),
            "ess": float(snapshot_ess),
            "conditional_ess_fraction": 1.0,
            "log_normalizing_constant_increment": 0.0,
            "log_normalizing_constant_estimate": float(log_normalizer),
            "log_evidence_ratio_estimate": float(log_evidence_ratio) if tempering_path == "collapsed_likelihood" else None,
            "log_path_normalizer_ratio_estimate": float(log_evidence_ratio),
            "max_log_target": float(np.max(cur_targets)),
            "mean_log_target": float(np.mean(cur_targets)),
            "resampled": 0.0,
            "unique_orders": float(snapshot.unique_orders),
        }
        add_proposal_diagnostics(
            row, accepted, proposed, accepted_by_type, proposed_by_type
        )
        unique_roots, genealogical_ess = _genealogy_summary(lineage)
        row["unique_root_ancestors"] = float(unique_roots)
        row["genealogical_ess"] = float(genealogical_ess)
        row["target_evaluations"] = float(
            target_evaluations - stage_target_evaluations_start
        )
        row["target_score_lookups"] = float(
            target_score_lookups - stage_target_score_lookups_start
        )
        history.append(row)
        emit_checkpoint()

    weights = _normalize_logweights(logw)
    if best_order is None:
        raise RuntimeError("SMC did not visit any order")

    (
        edge_mass,
        edge_mcse,
        final_ess,
        final_summary_score_lookups,
    ) = _weighted_edge_summary_and_lookups_for_orders(
        orders, weights, table, beta=1.0
    )
    posterior_summary_score_lookups += final_summary_score_lookups
    final_unique_orders = len({tuple(map(int, order)) for order in orders})
    unique_roots, genealogical_ess = _genealogy_summary(lineage)
    proposal_diagnostics: Dict[str, Dict[str, object]] = {}
    for proposal_name in _PROPOSAL_NAMES:
        proposal_count = int(
            sum(float(row.get(f"{proposal_name}_proposed", 0.0)) for row in history)
        )
        acceptance_count = int(
            sum(float(row.get(f"{proposal_name}_accepted", 0.0)) for row in history)
        )
        proposal_diagnostics[proposal_name] = {
            "proposed": proposal_count,
            "accepted": acceptance_count,
            "acceptance_rate": (
                float(acceptance_count / proposal_count) if proposal_count else None
            ),
        }

    return OrderSMCResult(
        orders=orders,
        weights=weights,
        logw=logw,
        best_order=best_order,
        best_edges=best_edges,
        edge_probs=edge_mass,
        history=history,
        edge_mcse=edge_mcse,
        effective_sample_size=final_ess,
        log_normalizing_constant_estimate=float(log_normalizer),
        log_prior_normalizing_constant_estimate=(
            float(log_prior_normalizer) if tempering_path == "collapsed_likelihood" else float("nan")
        ),
        log_evidence_ratio_estimate=(
            float(log_evidence_ratio) if tempering_path == "collapsed_likelihood" else float("nan")
        ),
        final_target_snapshots=snapshots,
        run_info={
            "nodes": list(table.nodes),
            "candidate_table_signature": _candidate_table_signature(table),
            "bit_generator": "PCG64DXSM",
            "candidate_max_indegree": max(
                (
                    int(np.count_nonzero(parent_set >= 0))
                    for child_parent_sets in table.parent_sets
                    for parent_set in child_parent_sets
                ),
                default=0,
            ),
            "posterior_semantics": "candidate_restricted_order_modular",
            "target_label": "order",
            "order_target": "order-modular marginal over retained compatible parent sets",
            "initial_order_proposal": "uniform over permutations",
            "tempering_path": tempering_path,
            "initial_importance_weight": (
                "zero: geometric bridge starts uniformly" if tempering_path == "geometric"
                else "log_gamma_0 (uniform proposal constant omitted)"
            ),
            "initial_effective_sample_size": float(initial_ess),
            "log_path_normalizer_ratio_estimate": float(log_evidence_ratio),
            "log_path_initial_normalizer_estimate": float(log_prior_normalizer),
            "normalizing_constant_semantics": (
                "SMC estimate of log Z_1 for the candidate-restricted order-modular "
                "unnormalized target; the n! uniform-order proposal factor is included"
            ),
            "prior_normalizing_constant_semantics": (
                "unavailable on geometric path; its initial normalizer is d!, not structural prior Z_0"
                if tempering_path == "geometric" else
                "importance-sampling estimate of log Z_0 for the beta=0 structural baseline"
            ),
            "evidence_ratio_semantics": (
                "unavailable on geometric path; log_path_normalizer_ratio_estimate is log(Z_1/d!)"
                if tempering_path == "geometric" else
                "sum of SMC log normalizing-constant increments, estimating log(Z_1/Z_0); "
                "this is relative to the candidate-restricted beta=0 baseline"
            ),
            "log_normalizing_constant_estimate": float(log_normalizer),
            "log_prior_normalizing_constant_estimate": (
                float(log_prior_normalizer) if tempering_path == "collapsed_likelihood" else None
            ),
            "log_evidence_ratio_estimate": (
                float(log_evidence_ratio) if tempering_path == "collapsed_likelihood" else None
            ),
            "tempering_schedule": [0.0]
            + [
                float(row["beta"])
                for row in history
                if row.get("phase") == "tempering"
            ],
            "adaptive_cess_target": cess_target,
            "resampling_method": resampling,
            "resampling_ess_fraction": float(resample_ess_frac),
            "proposal_mixture": dict(normalized_proposals),
            "sampler_configuration": copy.deepcopy(algorithm_config),
            "proposal_diagnostics": proposal_diagnostics,
            "target_evaluations": int(target_evaluations),
            "target_score_lookups": int(target_score_lookups),
            "target_score_lookup_semantics": (
                "number of retained compatible local-family entries read by counted "
                "order-target evaluations; candidate building and posterior summarization excluded"
            ),
            "best_dag_score_lookups": int(best_dag_score_lookups),
            "posterior_summary_score_lookups": int(
                posterior_summary_score_lookups
            ),
            "resume_validation_target_evaluations": int(
                resume_validation_target_evaluations
            ),
            "resume_validation_score_lookups": int(
                resume_validation_score_lookups
            ),
            "physical_target_evaluations": int(
                target_evaluations + resume_validation_target_evaluations
            ),
            "total_score_lookups": int(
                target_score_lookups
                + best_dag_score_lookups
                + posterior_summary_score_lookups
                + resume_validation_score_lookups
            ),
            "total_score_lookup_semantics": (
                "all retained local-family entries read during target evaluation, "
                "resume validation, best-DAG tracking, and Rao-Blackwell posterior "
                "summaries; candidate construction is reported by its own artifact"
            ),
            "proposal_timing": proposal_timing,
            "elapsed_seconds": float(
                elapsed_offset + time.perf_counter() - segment_started
            ),
            "final_target_snapshot_steps": list(snapshot_steps),
            "mh_step_semantics": "one order proposal per particle, not a sweep over variables",
            "final_target_snapshot_semantics": (
                "all snapshots have beta=1; sweep counts are additional full-particle "
                "MH sweeps after the ordinary beta=1 rejuvenation"
            ),
            "best_dag_selection": (
                "highest joint log-score DAG among visited particle states; exact score "
                "ties between local parent sets use their lexicographically smallest "
                "canonical (parent_name, child_name) edge tuple, and ties between DAGs "
                "use lexicographic adjacency then lexicographic order"
            ),
            "best_dag_log_score": float(best_joint_score),
            "best_dag_score_beta": 1.0,
            "best_edges_semantics": "visited_joint_graph_order_mode_not_marginal_DAG_MAP",
            "visited_order_states": int(visited_order_states),
            "final_unique_orders": int(final_unique_orders),
            "final_unique_order_fraction": float(final_unique_orders / len(orders)),
            "unique_root_ancestors": int(unique_roots),
            "genealogical_ess": float(genealogical_ess),
            "edge_mcse_calibrated": False,
            "convergence_established": False,
            "edge_mcse_method": (
                "unbiased weighted between-order variance divided by importance-weight ESS; "
                "heuristic because SMC genealogy and MH dependence are ignored"
            ),
        },
    )
