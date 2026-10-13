"""Predeclared dataset-level metrics and seed estimands."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from typing import Dict, Iterable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
from ordersmc.core import (
    GraphSizeDistribution,
    OrderCandidateTable,
    PosteriorDAGSample,
    candidate_max_indegree,
    edge_probs_from_posterior_dag_samples,
)
from ordersmc.graphs import CPDAG, dag_to_cpdag
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from .prior_calibration import candidate_support_sha256

Edge = tuple[str, str]
PosteriorTarget = Literal["order", "structure"]

HPR_WEIGHT_ESS_ESTIMATOR = "kish_normalized_weights_v1"
HPR_ANCHOR_ESS_ESTIMATOR = "geyer_initial_positive_sequence_per_run_v1"
HPR_REGION_ESS_ESTIMATOR = "geyer_initial_positive_sequence_per_run_v1"
HPR_COVERAGE_MCSE_ESTIMATOR = "run_stratified_batch_means_v1"

_HPR_TARGET_SEMANTICS: Mapping[str, str] = {
    "order": "candidate_restricted_order_modular_dag_pushforward",
    "structure": "candidate_restricted_structure_modular_dag_posterior",
}


def _validate_candidate_support_id(value: str, name: str) -> str:
    prefix = "candidate-sha256:"
    digest = value[len(prefix) :] if isinstance(value, str) and value.startswith(prefix) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase candidate SHA-256 identifier")
    return value


def _validate_sha256(value: str, name: str, *, prefix: str = "") -> str:
    digest = value[len(prefix) :] if isinstance(value, str) and value.startswith(prefix) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        qualifier = f"{prefix}<digest>" if prefix else "a lowercase SHA-256 digest"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_target_sha256(table: OrderCandidateTable) -> str:
    """Hash the row-order-independent posterior target, excluding metadata."""

    children: list[dict[str, object]] = []
    for parent_sets, loglik, logprior in zip(
        table.parent_sets, table.loglik, table.logprior
    ):
        families = [
            {
                "parents": sorted(int(parent) for parent in row if int(parent) >= 0),
                "logevidence_hex": float(likelihood).hex(),
                "logprior_hex": float(prior).hex(),
            }
            for row, likelihood, prior in zip(parent_sets, loglik, logprior)
        ]
        families.sort(key=lambda family: tuple(family["parents"]))
        children.append({"families": families})
    return _canonical_json_sha256(
        {
            "schema": "ordersmc-candidate-posterior-target-v1",
            "nodes": list(table.nodes),
            "children": children,
            "beta_hex": float(1.0).hex(),
        }
    )


def _draw_identifier(
    *,
    edges: frozenset[Edge],
    normalized_weight: float,
    run_id: str,
    draw_index: int,
    target_semantics: str,
    support_id: str,
    candidate_target_id: str,
) -> str:
    return _canonical_json_sha256(
        {
            "schema": "ordersmc-posterior-dag-draw-id-v1",
            "edges": [list(edge) for edge in sorted(edges)],
            "normalized_weight_hex": float(normalized_weight).hex(),
            "run_id": run_id,
            "draw_index": draw_index,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "candidate_target_id": candidate_target_id,
        }
    )


def _bundle_identifier(
    *,
    draws: tuple[frozenset[Edge], ...],
    normalized_weights: tuple[float, ...],
    draw_ids: tuple[str, ...],
    run_ids: tuple[str, ...],
    nodes: tuple[str, ...],
    target_label: str,
    target_semantics: str,
    support_id: str,
    candidate_target_id: str,
    max_indegree: int,
    bundle_kind: str,
    producer: str,
    source_result_sha256: str | None,
) -> str:
    return _canonical_json_sha256(
        {
            "schema": "ordersmc-posterior-dag-draw-bundle-v1",
            "nodes": list(nodes),
            "target_label": target_label,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "candidate_target_id": candidate_target_id,
            "max_indegree": max_indegree,
            "beta_hex": float(1.0).hex(),
            "bundle_kind": bundle_kind,
            "producer": producer,
            "source_result_sha256": source_result_sha256,
            "draws": [
                {
                    "edges": [list(edge) for edge in sorted(edges)],
                    "normalized_weight_hex": float(weight).hex(),
                    "draw_id": draw_id,
                    "run_id": run_id,
                }
                for edges, weight, draw_id, run_id in zip(
                    draws, normalized_weights, draw_ids, run_ids
                )
            ],
        }
    )


@dataclass(frozen=True)
class SeedEstimandResult:
    mean_single_run_metrics: Dict[str, float]
    ensemble_metrics: Dict[str, float]
    per_seed_metrics: tuple[Dict[str, float], ...]
    ensemble_edge_probs: Dict[Edge, float]
    single_run_formula: str = "S^{-1} sum_s L(p_hat_s, Y)"
    ensemble_formula: str = "L(S^{-1} sum_s p_hat_s, Y)"
    ensemble_compute_multiplier: int = 1
    per_seed_representative_edges: tuple[tuple[Edge, ...], ...] = ()
    ensemble_representative_edges: tuple[Edge, ...] = ()


@dataclass(frozen=True)
class PairedBootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    resamples: int
    stratum_count: int
    one_sided_lower: float
    one_sided_upper: float


@dataclass(frozen=True)
class NoninferiorityResult:
    difference: float
    ci_lower: float
    ci_upper: float
    margin: float
    higher_is_better: bool
    established: bool


_EDGE_TARGET_SEMANTICS: Mapping[str, str] = {
    "order": "candidate_restricted_order_modular",
    "structure": "candidate_restricted_structure_modular",
}


def _edge_distribution_identifier(
    *,
    probabilities: tuple[tuple[Edge, float], ...],
    nodes: tuple[str, ...],
    target_label: str,
    target_semantics: str,
    support_id: str,
    candidate_target_id: str,
    max_indegree: int,
    beta: float,
    producer: str,
    reference_kind: str,
    log_ratio: float | None,
    source_result_sha256: str | None,
) -> str:
    return _canonical_json_sha256(
        {
            "schema": "ordersmc-posterior-edge-distribution-v2",
            "probabilities": [
                {
                    "parent": parent,
                    "child": child,
                    "probability_hex": float(probability).hex(),
                }
                for ((parent, child), probability) in probabilities
            ],
            "nodes": list(nodes),
            "target_label": target_label,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "candidate_target_id": candidate_target_id,
            "max_indegree": max_indegree,
            "beta_hex": float(beta).hex(),
            "producer": producer,
            "reference_kind": reference_kind,
            "log_ratio_hex": None if log_ratio is None else float(log_ratio).hex(),
            "source_result_sha256": source_result_sha256,
        }
    )


@dataclass(frozen=True, eq=False, init=False)
class PosteriorEdgeDistribution(Mapping[Edge, float]):
    """Complete directed PIPs inseparably bound to one candidate target.

    Approximate records are built from a candidate table. Exact records can be
    built only by :meth:`exact_from_candidate_table`, which runs the relevant
    exact DP rather than trusting caller-labelled probabilities.
    """

    probabilities: tuple[tuple[Edge, float], ...]
    nodes: tuple[str, ...]
    target_label: str
    target_semantics: str
    support_id: str
    candidate_target_id: str
    max_indegree: int
    beta: float
    producer: str
    reference_kind: str
    log_ratio: float | None = None
    source_result_sha256: str | None = None
    distribution_sha256: str = ""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "PosteriorEdgeDistribution must be constructed with "
            "from_candidate_table or exact_from_candidate_table"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        probabilities: tuple[tuple[Edge, float], ...],
        nodes: tuple[str, ...],
        target_label: str,
        target_semantics: str,
        support_id: str,
        candidate_target_id: str,
        max_indegree: int,
        beta: float,
        producer: str,
        reference_kind: str,
        log_ratio: float | None,
        source_result_sha256: str | None,
    ) -> "PosteriorEdgeDistribution":
        distribution_sha256 = _edge_distribution_identifier(
            probabilities=probabilities,
            nodes=nodes,
            target_label=target_label,
            target_semantics=target_semantics,
            support_id=support_id,
            candidate_target_id=candidate_target_id,
            max_indegree=max_indegree,
            beta=beta,
            producer=producer,
            reference_kind=reference_kind,
            log_ratio=log_ratio,
            source_result_sha256=source_result_sha256,
        )
        instance = object.__new__(cls)
        values: Mapping[str, object] = {
            "probabilities": probabilities,
            "nodes": nodes,
            "target_label": target_label,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "candidate_target_id": candidate_target_id,
            "max_indegree": max_indegree,
            "beta": beta,
            "producer": producer,
            "reference_kind": reference_kind,
            "log_ratio": log_ratio,
            "source_result_sha256": source_result_sha256,
            "distribution_sha256": distribution_sha256,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        _validate_edge_distribution_integrity(instance, "posterior edge")
        return instance

    @classmethod
    def from_candidate_table(
        cls,
        probabilities: Mapping[Edge, float],
        table: OrderCandidateTable,
        *,
        target_label: str,
        beta: float,
        producer: str,
        target_semantics: str | None = None,
        reference_kind: str = "approximate",
        log_ratio: float | None = None,
        source_result_sha256: str | None = None,
    ) -> "PosteriorEdgeDistribution":
        """Build a content-hashed approximate record on a real candidate table."""

        if not isinstance(table, OrderCandidateTable):
            raise TypeError("table must be an OrderCandidateTable")
        if target_label not in _EDGE_TARGET_SEMANTICS:
            raise ValueError("target_label must be 'order' or 'structure'")
        canonical_semantics = _EDGE_TARGET_SEMANTICS[target_label]
        if target_semantics is not None and target_semantics != canonical_semantics:
            raise ValueError("target_semantics does not match target_label")
        if reference_kind != "approximate":
            raise ValueError(
                "caller-supplied probabilities cannot be marked exact; use "
                "exact_from_candidate_table"
            )
        if isinstance(beta, (bool, np.bool_)):
            raise ValueError("beta must lie in [0, 1]")
        try:
            prepared_beta = float(beta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("beta must lie in [0, 1]") from exc
        if not math.isfinite(prepared_beta) or not 0.0 <= prepared_beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        nodes = tuple(str(node) for node in table.nodes)
        expected = {
            (parent, child)
            for parent in nodes
            for child in nodes
            if parent != child
        }
        observed = {(str(parent), str(child)) for parent, child in probabilities}
        if observed != expected or len(probabilities) != len(expected):
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                "posterior edge table must contain every directed non-self pair exactly once; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        allowed_edges = {
            (nodes[int(parent)], nodes[child_index])
            for child_index, parent_sets in enumerate(table.parent_sets)
            for row in parent_sets
            for parent in row
            if int(parent) >= 0
        }
        prepared = tuple(
            sorted(
                (
                    (str(parent), str(child)),
                    float(probabilities[(parent, child)]),
                )
                for parent, child in expected
            )
        )
        for edge, probability in prepared:
            if edge not in allowed_edges and not math.isclose(
                probability, 0.0, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(
                    f"posterior edge {edge!r} has nonzero mass outside candidate support"
                )
        return cls._from_validated(
            probabilities=prepared,
            nodes=nodes,
            target_label=target_label,
            target_semantics=canonical_semantics,
            support_id=f"candidate-sha256:{candidate_support_sha256(table)}",
            candidate_target_id=f"candidate-target-sha256:{_candidate_target_sha256(table)}",
            max_indegree=candidate_max_indegree(table),
            beta=prepared_beta,
            producer=producer,
            reference_kind="approximate",
            log_ratio=log_ratio,
            source_result_sha256=source_result_sha256,
        )

    @classmethod
    def exact_from_candidate_table(
        cls,
        table: OrderCandidateTable,
        *,
        target_label: PosteriorTarget,
        beta: float = 1.0,
        max_nodes: int = 12,
    ) -> "PosteriorEdgeDistribution":
        """Run the declared exact DP and bind its result to the candidate target."""

        if not isinstance(table, OrderCandidateTable):
            raise TypeError("table must be an OrderCandidateTable")
        if target_label == "order":
            from .exact_order import exact_order_dp

            result = exact_order_dp(
                table, beta=beta, max_nodes=max_nodes, compute_graph_size=False
            )
            producer = "exact_order_dp"
        elif target_label == "structure":
            from .exact_structure import exact_structure_dp

            result = exact_structure_dp(
                table, beta=beta, max_nodes=max_nodes, compute_graph_size=False
            )
            producer = "exact_structure_dp"
        else:
            raise ValueError("target_label must be 'order' or 'structure'")
        nodes = tuple(str(node) for node in table.nodes)
        complete = {
            (parent, child): float(result.edge_probs.get((parent, child), 0.0))
            for parent in nodes
            for child in nodes
            if parent != child
        }
        source_result_sha256 = _canonical_json_sha256(
            {
                "schema": "ordersmc-exact-edge-result-v1",
                "producer": producer,
                "candidate_target_sha256": _candidate_target_sha256(table),
                "beta_hex": float(result.beta).hex(),
                "log_normalizer_hex": float(result.log_normalizer).hex(),
                "log_prior_normalizer_hex": float(result.log_prior_normalizer).hex(),
                "log_evidence_ratio_hex": float(result.log_evidence_ratio).hex(),
                "probabilities": [
                    {
                        "parent": parent,
                        "child": child,
                        "probability_hex": float(probability).hex(),
                    }
                    for (parent, child), probability in sorted(complete.items())
                ],
            }
        )
        prepared = tuple(sorted(complete.items()))
        return cls._from_validated(
            probabilities=prepared,
            nodes=nodes,
            target_label=target_label,
            target_semantics=_EDGE_TARGET_SEMANTICS[target_label],
            support_id=f"candidate-sha256:{candidate_support_sha256(table)}",
            candidate_target_id=f"candidate-target-sha256:{_candidate_target_sha256(table)}",
            max_indegree=candidate_max_indegree(table),
            beta=float(result.beta),
            producer=producer,
            reference_kind="exact",
            log_ratio=float(result.log_evidence_ratio),
            source_result_sha256=source_result_sha256,
        )

    def _validate_fields(self) -> None:
        if len(self.nodes) < 2 or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("posterior edge nodes must be unique and contain at least two nodes")
        if (
            self.target_label not in _EDGE_TARGET_SEMANTICS
            or self.target_semantics != _EDGE_TARGET_SEMANTICS[self.target_label]
        ):
            raise ValueError("posterior edge target provenance is invalid")
        _validate_candidate_support_id(self.support_id, "support_id")
        _validate_sha256(
            self.candidate_target_id,
            "candidate_target_id",
            prefix="candidate-target-sha256:",
        )
        if (
            not isinstance(self.max_indegree, int)
            or isinstance(self.max_indegree, bool)
            or not 0 <= self.max_indegree < len(self.nodes)
        ):
            raise ValueError("posterior edge max_indegree is invalid")
        if not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
            raise ValueError("posterior edge beta must be finite and lie in [0, 1]")
        if (
            not isinstance(self.producer, str)
            or not self.producer
            or self.reference_kind not in {"approximate", "exact"}
        ):
            raise ValueError("posterior edge producer/reference_kind is invalid")
        if self.reference_kind == "exact" and self.producer not in {
            "exact_order_dp",
            "exact_structure_dp",
        }:
            raise ValueError("exact posterior edges require a recognized exact-DP producer")
        expected = {
            (parent, child)
            for parent in self.nodes
            for child in self.nodes
            if parent != child
        }
        seen: set[Edge] = set()
        for edge, probability in self.probabilities:
            if edge not in expected or edge in seen:
                raise ValueError("posterior edge probabilities have invalid or duplicate edges")
            seen.add(edge)
            if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
                raise ValueError("posterior edge probabilities must lie in [0, 1]")
        if seen != expected:
            raise ValueError("posterior edge probabilities must be complete")
        for left_index, left in enumerate(self.nodes):
            for right in self.nodes[left_index + 1 :]:
                if self[(left, right)] + self[(right, left)] > 1.0 + 1e-12:
                    raise ValueError("opposite directed-edge probabilities exceed one")
        if sum(probability for _, probability in self.probabilities) > (
            len(self.nodes) * self.max_indegree + 1e-12
        ):
            raise ValueError("expected edge count exceeds the indegree support bound")
        if self.log_ratio is not None and not math.isfinite(float(self.log_ratio)):
            raise ValueError("log_ratio must be finite when supplied")
        if self.source_result_sha256 is not None and (
            not isinstance(self.source_result_sha256, str)
            or len(self.source_result_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_result_sha256)
        ):
            raise ValueError("source_result_sha256 must be a lowercase SHA-256 digest")
        if self.reference_kind == "exact" and self.source_result_sha256 is None:
            raise ValueError("exact posterior edges require an exact-result SHA-256")

    def __getitem__(self, edge: Edge) -> float:
        for candidate, probability in self.probabilities:
            if candidate == edge:
                return probability
        raise KeyError(edge)

    def __iter__(self):
        return (edge for edge, _ in self.probabilities)

    def __len__(self) -> int:
        return len(self.probabilities)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PosteriorEdgeDistribution):
            return self.__dict__ == other.__dict__
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ordersmc-posterior-edge-distribution-v2",
            "probabilities": [
                {"parent": parent, "child": child, "probability": probability}
                for ((parent, child), probability) in self.probabilities
            ],
            "nodes": list(self.nodes),
            "target_label": self.target_label,
            "target_semantics": self.target_semantics,
            "support_id": self.support_id,
            "candidate_target_id": self.candidate_target_id,
            "max_indegree": self.max_indegree,
            "beta": self.beta,
            "producer": self.producer,
            "reference_kind": self.reference_kind,
            "log_ratio": self.log_ratio,
            "source_result_sha256": self.source_result_sha256,
            "distribution_sha256": self.distribution_sha256,
        }

    def pip_only_view(self) -> "PosteriorEdgeDistribution":
        """Return the same provenance-bound PIPs without evidence-ratio semantics.

        Some samplers estimate posterior PIPs but not a marginal-likelihood
        ratio.  Exact-target PIP error is then well defined, whereas a
        log-ratio comparison is not.  This explicit view avoids silently
        dropping an available exact log ratio inside the metric itself.
        """

        _validate_edge_distribution_integrity(self, "posterior edge")
        if self.log_ratio is None:
            return self
        return self._from_validated(
            probabilities=self.probabilities,
            nodes=self.nodes,
            target_label=self.target_label,
            target_semantics=self.target_semantics,
            support_id=self.support_id,
            candidate_target_id=self.candidate_target_id,
            max_indegree=self.max_indegree,
            beta=self.beta,
            producer=self.producer,
            reference_kind=self.reference_kind,
            log_ratio=None,
            source_result_sha256=self.source_result_sha256,
        )


def _validate_edge_distribution_integrity(
    distribution: PosteriorEdgeDistribution, name: str
) -> None:
    if not isinstance(distribution, PosteriorEdgeDistribution):
        raise TypeError(f"{name} must be a PosteriorEdgeDistribution")
    distribution._validate_fields()
    expected = _edge_distribution_identifier(
        probabilities=distribution.probabilities,
        nodes=distribution.nodes,
        target_label=distribution.target_label,
        target_semantics=distribution.target_semantics,
        support_id=distribution.support_id,
        candidate_target_id=distribution.candidate_target_id,
        max_indegree=distribution.max_indegree,
        beta=distribution.beta,
        producer=distribution.producer,
        reference_kind=distribution.reference_kind,
        log_ratio=distribution.log_ratio,
        source_result_sha256=distribution.source_result_sha256,
    )
    _validate_sha256(distribution.distribution_sha256, f"{name} distribution_sha256")
    if distribution.distribution_sha256 != expected:
        raise ValueError(f"{name} failed content-hash verification")


@dataclass(frozen=True, init=False)
class PosteriorDAGDrawBundle:
    """Content-addressed posterior DAG draws tied to one candidate target.

    Instances are constructed only through
    :func:`build_posterior_dag_draw_bundle` (or the checked OrderSMC adapter),
    so the HPR metric never accepts raw draws plus free provenance strings.
    """

    draws: tuple[frozenset[Edge], ...]
    normalized_weights: tuple[float, ...]
    draw_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    nodes: tuple[str, ...]
    target_label: str
    target_semantics: str
    support_id: str
    candidate_target_id: str
    max_indegree: int
    beta: float
    bundle_kind: str
    producer: str
    source_result_sha256: str | None
    bundle_sha256: str
    schema: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "PosteriorDAGDrawBundle must be constructed with "
            "build_posterior_dag_draw_bundle"
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        draws: tuple[frozenset[Edge], ...],
        normalized_weights: tuple[float, ...],
        draw_ids: tuple[str, ...],
        run_ids: tuple[str, ...],
        nodes: tuple[str, ...],
        target_label: str,
        target_semantics: str,
        support_id: str,
        candidate_target_id: str,
        max_indegree: int,
        bundle_kind: str,
        producer: str,
        source_result_sha256: str | None,
        bundle_sha256: str,
    ) -> "PosteriorDAGDrawBundle":
        instance = object.__new__(cls)
        values: Mapping[str, object] = {
            "draws": draws,
            "normalized_weights": normalized_weights,
            "draw_ids": draw_ids,
            "run_ids": run_ids,
            "nodes": nodes,
            "target_label": target_label,
            "target_semantics": target_semantics,
            "support_id": support_id,
            "candidate_target_id": candidate_target_id,
            "max_indegree": max_indegree,
            "beta": 1.0,
            "bundle_kind": bundle_kind,
            "producer": producer,
            "source_result_sha256": source_result_sha256,
            "bundle_sha256": bundle_sha256,
            "schema": "ordersmc-posterior-dag-draw-bundle-v1",
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance


def build_posterior_dag_draw_bundle(
    draws: Sequence[Iterable[Edge]],
    weights: Sequence[float],
    table: OrderCandidateTable,
    *,
    target_label: PosteriorTarget,
    run_ids: Sequence[str],
) -> PosteriorDAGDrawBundle:
    """Validate and content-address weighted DAG draws for HPR evaluation."""

    if not isinstance(table, OrderCandidateTable):
        raise TypeError("table must be an OrderCandidateTable")
    max_indegree = candidate_max_indegree(table)
    nodes = tuple(table.nodes)
    if (
        len(nodes) < 2
        or len(set(nodes)) != len(nodes)
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError("candidate-table nodes must contain at least two unique strings")
    if target_label not in _HPR_TARGET_SEMANTICS:
        raise ValueError("target_label must be 'order' or 'structure'")
    target_semantics = _HPR_TARGET_SEMANTICS[target_label]
    raw_draws = tuple(draws)
    raw_weights = tuple(weights)
    prepared_run_ids = tuple(run_ids)
    if not raw_draws or len(raw_draws) != len(raw_weights):
        raise ValueError("draws and weights must be non-empty and aligned")
    if len(prepared_run_ids) != len(raw_draws):
        raise ValueError("run_ids must align with every draw")
    for run_id in prepared_run_ids:
        _validate_sha256(run_id, "run_id")

    allowed_families: dict[str, set[frozenset[str]]] = {}
    for child_index, child in enumerate(nodes):
        allowed_families[child] = {
            frozenset(nodes[int(parent)] for parent in row if int(parent) >= 0)
            for row in table.parent_sets[child_index]
        }
    node_set = set(nodes)
    canonical_draws: list[frozenset[Edge]] = []
    for draw_index, raw_edges in enumerate(raw_draws):
        edge_list: list[Edge] = []
        for raw_edge in raw_edges:
            try:
                parent, child = raw_edge
            except (TypeError, ValueError) as exc:
                raise ValueError(f"draw {draw_index} contains a malformed edge") from exc
            if (
                not isinstance(parent, str)
                or not isinstance(child, str)
                or parent == child
                or parent not in node_set
                or child not in node_set
            ):
                raise ValueError(f"draw {draw_index} contains an invalid edge")
            edge_list.append((parent, child))
        edges = frozenset(edge_list)
        if len(edges) != len(edge_list):
            raise ValueError(f"draw {draw_index} contains a duplicate edge")
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f"draw {draw_index} is cyclic")
        for child in nodes:
            parents = frozenset(parent for parent, endpoint in edges if endpoint == child)
            if parents not in allowed_families[child]:
                raise ValueError(
                    f"draw {draw_index} uses a parent family outside the candidate support"
                )
        canonical_draws.append(edges)

    if any(isinstance(weight, (bool, np.bool_)) for weight in raw_weights):
        raise ValueError("weights must be finite non-negative numbers")
    probability = np.asarray(raw_weights, dtype=float)
    if probability.ndim != 1 or np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("weights must be finite non-negative numbers")
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    probability = probability / total
    normalized_weights = tuple(map(float, probability))
    prepared_draws = tuple(canonical_draws)
    support_id = f"candidate-sha256:{candidate_support_sha256(table)}"
    candidate_target_id = f"candidate-target-sha256:{_candidate_target_sha256(table)}"
    draw_ids = tuple(
        _draw_identifier(
            edges=edges,
            normalized_weight=weight,
            run_id=run_id,
            draw_index=draw_index,
            target_semantics=target_semantics,
            support_id=support_id,
            candidate_target_id=candidate_target_id,
        )
        for draw_index, (edges, weight, run_id) in enumerate(
            zip(prepared_draws, normalized_weights, prepared_run_ids)
        )
    )
    bundle_sha256 = _bundle_identifier(
        draws=prepared_draws,
        normalized_weights=normalized_weights,
        draw_ids=draw_ids,
        run_ids=prepared_run_ids,
        nodes=nodes,
        target_label=target_label,
        target_semantics=target_semantics,
        support_id=support_id,
        candidate_target_id=candidate_target_id,
        max_indegree=max_indegree,
        bundle_kind="sampled",
        producer="candidate_table_weighted_draws",
        source_result_sha256=None,
    )
    return PosteriorDAGDrawBundle._from_validated(
        draws=prepared_draws,
        normalized_weights=normalized_weights,
        draw_ids=draw_ids,
        run_ids=prepared_run_ids,
        nodes=nodes,
        target_label=target_label,
        target_semantics=target_semantics,
        support_id=support_id,
        candidate_target_id=candidate_target_id,
        max_indegree=max_indegree,
        bundle_kind="sampled",
        producer="candidate_table_weighted_draws",
        source_result_sha256=None,
        bundle_sha256=bundle_sha256,
    )


def _topological_order_count(
    nodes: tuple[str, ...], edges: frozenset[Edge]
) -> int:
    """Count linear extensions exactly by subset DP."""

    node_index = {node: index for index, node in enumerate(nodes)}
    parent_masks = [0] * len(nodes)
    for parent, child in edges:
        parent_masks[node_index[child]] |= 1 << node_index[parent]
    counts = [0] * (1 << len(nodes))
    counts[0] = 1
    full = len(counts) - 1
    for placed in range(full):
        if counts[placed] == 0:
            continue
        for node_index_, parent_mask in enumerate(parent_masks):
            bit = 1 << node_index_
            if placed & bit == 0 and parent_mask & ~placed == 0:
                counts[placed | bit] += counts[placed]
    return counts[full]


def exact_posterior_dag_bundle(
    table: OrderCandidateTable,
    *,
    target_label: PosteriorTarget,
    max_nodes: int = 12,
    max_family_combinations: int = 2_000_000,
) -> PosteriorDAGDrawBundle:
    """Enumerate and independently DP-check an exact beta-one DAG posterior.

    Structure-modular mass counts each candidate-supported DAG once. The
    order-modular DAG pushforward additionally multiplies each DAG's modular
    weight by its exact number of topological orders.
    """

    if not isinstance(table, OrderCandidateTable):
        raise TypeError("table must be an OrderCandidateTable")
    if (
        isinstance(max_family_combinations, bool)
        or not isinstance(max_family_combinations, int)
        or max_family_combinations < 1
    ):
        raise ValueError("max_family_combinations must be a positive integer")
    if target_label == "order":
        from .exact_order import exact_order_dp

        exact_result = exact_order_dp(
            table, beta=1.0, max_nodes=max_nodes, compute_graph_size=False
        )
        producer = "exact_order_candidate_dag_enumeration"
    elif target_label == "structure":
        from .exact_structure import exact_structure_dp

        exact_result = exact_structure_dp(
            table, beta=1.0, max_nodes=max_nodes, compute_graph_size=False
        )
        producer = "exact_structure_candidate_dag_enumeration"
    else:
        raise ValueError("target_label must be 'order' or 'structure'")

    family_count = math.prod(len(parent_sets) for parent_sets in table.parent_sets)
    if family_count > max_family_combinations:
        raise ValueError(
            "exact DAG enumeration exceeds max_family_combinations: "
            f"{family_count} > {max_family_combinations}"
        )
    nodes = tuple(table.nodes)
    draws: list[frozenset[Edge]] = []
    log_weights: list[float] = []
    family_ranges = [range(len(parent_sets)) for parent_sets in table.parent_sets]
    for family_indices in product(*family_ranges):
        edges: set[Edge] = set()
        log_weight = 0.0
        for child_index, family_index in enumerate(family_indices):
            log_weight += float(table.loglik[child_index][family_index])
            log_weight += float(table.logprior[child_index][family_index])
            for raw_parent in table.parent_sets[child_index][family_index]:
                parent_index = int(raw_parent)
                if parent_index >= 0:
                    edges.add((nodes[parent_index], nodes[child_index]))
        frozen_edges = frozenset(edges)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(frozen_edges)
        if not nx.is_directed_acyclic_graph(graph):
            continue
        if target_label == "order":
            extension_count = _topological_order_count(nodes, frozen_edges)
            if extension_count < 1:
                raise RuntimeError("acyclic graph has no topological order")
            log_weight += math.log(extension_count)
        draws.append(frozen_edges)
        log_weights.append(log_weight)
    if not draws:
        raise RuntimeError("candidate support contains no DAG")
    maximum_log_weight = max(log_weights)
    scaled = np.exp(np.asarray(log_weights, dtype=float) - maximum_log_weight)
    scaled_total = float(np.sum(scaled))
    probability = scaled / scaled_total
    enumerated_log_normalizer = maximum_log_weight + math.log(scaled_total)
    if not math.isclose(
        enumerated_log_normalizer,
        float(exact_result.log_normalizer),
        rel_tol=2e-11,
        abs_tol=2e-10,
    ):
        raise RuntimeError("exact DAG enumeration and exact DP normalizers disagree")
    enumerated_pips = {
        (parent, child): float(
            sum(
                weight
                for draw, weight in zip(draws, probability)
                if (parent, child) in draw
            )
        )
        for parent in nodes
        for child in nodes
        if parent != child
    }
    for edge, probability_ in enumerated_pips.items():
        if not math.isclose(
            probability_,
            float(exact_result.edge_probs.get(edge, 0.0)),
            rel_tol=2e-10,
            abs_tol=2e-11,
        ):
            raise RuntimeError("exact DAG enumeration and exact DP PIPs disagree")

    source_result_sha256 = _canonical_json_sha256(
        {
            "schema": "ordersmc-exact-posterior-dag-enumeration-v1",
            "producer": producer,
            "candidate_target_sha256": _candidate_target_sha256(table),
            "log_normalizer_hex": float(enumerated_log_normalizer).hex(),
            "draws": [
                {
                    "edges": [list(edge) for edge in sorted(draw)],
                    "probability_hex": float(weight).hex(),
                }
                for draw, weight in zip(draws, probability)
            ],
        }
    )
    sampled = build_posterior_dag_draw_bundle(
        draws,
        probability,
        table,
        target_label=target_label,
        run_ids=[source_result_sha256] * len(draws),
    )
    bundle_sha256 = _bundle_identifier(
        draws=sampled.draws,
        normalized_weights=sampled.normalized_weights,
        draw_ids=sampled.draw_ids,
        run_ids=sampled.run_ids,
        nodes=sampled.nodes,
        target_label=sampled.target_label,
        target_semantics=sampled.target_semantics,
        support_id=sampled.support_id,
        candidate_target_id=sampled.candidate_target_id,
        max_indegree=sampled.max_indegree,
        bundle_kind="exact",
        producer=producer,
        source_result_sha256=source_result_sha256,
    )
    result = PosteriorDAGDrawBundle._from_validated(
        draws=sampled.draws,
        normalized_weights=sampled.normalized_weights,
        draw_ids=sampled.draw_ids,
        run_ids=sampled.run_ids,
        nodes=sampled.nodes,
        target_label=sampled.target_label,
        target_semantics=sampled.target_semantics,
        support_id=sampled.support_id,
        candidate_target_id=sampled.candidate_target_id,
        max_indegree=sampled.max_indegree,
        bundle_kind="exact",
        producer=producer,
        source_result_sha256=source_result_sha256,
        bundle_sha256=bundle_sha256,
    )
    _validate_bundle_integrity(result, "exact posterior DAG")
    return result


def posterior_dag_bundle_from_ordersmc_samples(
    samples: Sequence[PosteriorDAGSample],
    table: OrderCandidateTable,
) -> PosteriorDAGDrawBundle:
    """Convert audited OrderSMC DAG-pushforward samples into an HPR bundle."""

    prepared = tuple(samples)
    if not prepared:
        raise ValueError("posterior DAG samples must be non-empty")
    # Bind the source/candidate signatures, seed, draw indices, particle
    # bounds, normalized weights, DAGs, and retained parent families first.
    edge_probs_from_posterior_dag_samples(prepared, table)
    return build_posterior_dag_draw_bundle(
        [sample.edges for sample in prepared],
        [sample.normalized_weight for sample in prepared],
        table,
        target_label="order",
        run_ids=[sample.source_result_signature for sample in prepared],
    )


@dataclass(frozen=True)
class ReferenceCoverageDiagnostics:
    definition_bundle_sha256: str
    evaluation_bundle_sha256: str
    definition_weight_ess: float
    evaluation_weight_ess: float
    anchor_min_ess: float
    region_indicator_ess: float
    heldout_coverage_mcse: float
    weight_ess_estimator: str = HPR_WEIGHT_ESS_ESTIMATOR
    anchor_ess_estimator: str = HPR_ANCHOR_ESS_ESTIMATOR
    region_indicator_ess_estimator: str = HPR_REGION_ESS_ESTIMATOR
    heldout_coverage_mcse_estimator: str = HPR_COVERAGE_MCSE_ESTIMATOR


@dataclass(frozen=True)
class HighPosteriorRegionResult:
    anchors: tuple[Edge, ...]
    regions: tuple[tuple[int, ...], ...]
    method_coverage: float
    reference_coverage: float
    coverage_deficit: float
    coverage_ratio: float
    definition_region_mass: float
    reference_is_exact: bool
    target_label: str
    target_semantics: str
    support_id: str
    candidate_target_id: str
    max_indegree: int
    method_bundle_sha256: str
    reference_definition_bundle_sha256: str
    reference_evaluation_bundle_sha256: str
    reference_uncertainty_mcse: float
    reference_diagnostics: ReferenceCoverageDiagnostics | None


@dataclass(frozen=True)
class ThresholdAttainmentPoint:
    time: float
    at_risk: int
    attained: int
    censored: int
    survival: float
    cumulative_attainment: float


def _validate_bundle_integrity(bundle: PosteriorDAGDrawBundle, name: str) -> None:
    if not isinstance(bundle, PosteriorDAGDrawBundle):
        raise TypeError(f"{name} must be a PosteriorDAGDrawBundle")
    count = len(bundle.draws)
    if count < 1 or not (
        len(bundle.normalized_weights)
        == len(bundle.draw_ids)
        == len(bundle.run_ids)
        == count
    ):
        raise ValueError(f"{name} bundle has unaligned draw records")
    if bundle.schema != "ordersmc-posterior-dag-draw-bundle-v1":
        raise ValueError(f"{name} bundle has an unsupported schema")
    if bundle.target_label not in _HPR_TARGET_SEMANTICS or bundle.target_semantics != (
        _HPR_TARGET_SEMANTICS[bundle.target_label]
    ):
        raise ValueError(f"{name} bundle has invalid target semantics")
    if not math.isclose(float(bundle.beta), 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{name} bundle is not a beta=1 posterior")
    if (
        len(bundle.nodes) < 2
        or len(set(bundle.nodes)) != len(bundle.nodes)
        or any(not isinstance(node, str) or not node for node in bundle.nodes)
    ):
        raise ValueError(f"{name} bundle has invalid nodes")
    if (
        not isinstance(bundle.max_indegree, int)
        or isinstance(bundle.max_indegree, bool)
        or not 0 <= bundle.max_indegree < len(bundle.nodes)
    ):
        raise ValueError(f"{name} bundle has invalid max_indegree")
    if bundle.bundle_kind not in {"sampled", "exact"}:
        raise ValueError(f"{name} bundle has invalid bundle_kind")
    if not isinstance(bundle.producer, str) or not bundle.producer:
        raise ValueError(f"{name} bundle has invalid producer")
    if bundle.bundle_kind == "exact":
        if bundle.producer not in {
            "exact_order_candidate_dag_enumeration",
            "exact_structure_candidate_dag_enumeration",
        }:
            raise ValueError(f"{name} exact bundle has an invalid producer")
        if bundle.source_result_sha256 is None:
            raise ValueError(f"{name} exact bundle lacks a source-result digest")
    if bundle.source_result_sha256 is not None:
        _validate_sha256(
            bundle.source_result_sha256, f"{name} source_result_sha256"
        )
    _validate_candidate_support_id(bundle.support_id, f"{name} support_id")
    _validate_sha256(
        bundle.candidate_target_id,
        f"{name} candidate_target_id",
        prefix="candidate-target-sha256:",
    )
    _validate_sha256(bundle.bundle_sha256, f"{name} bundle_sha256")
    for draw_id in bundle.draw_ids:
        _validate_sha256(draw_id, f"{name} draw_id")
    if len(set(bundle.draw_ids)) != count:
        raise ValueError(f"{name} bundle has duplicate draw IDs")
    for run_id in bundle.run_ids:
        _validate_sha256(run_id, f"{name} run_id")
    probability = np.asarray(bundle.normalized_weights, dtype=float)
    if (
        np.any(~np.isfinite(probability))
        or np.any(probability < 0.0)
        or not math.isclose(float(np.sum(probability)), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(f"{name} bundle has invalid normalized weights")
    expected_draw_ids = tuple(
        _draw_identifier(
            edges=edges,
            normalized_weight=weight,
            run_id=run_id,
            draw_index=draw_index,
            target_semantics=bundle.target_semantics,
            support_id=bundle.support_id,
            candidate_target_id=bundle.candidate_target_id,
        )
        for draw_index, (edges, weight, run_id) in enumerate(
            zip(bundle.draws, bundle.normalized_weights, bundle.run_ids)
        )
    )
    expected_bundle_sha256 = _bundle_identifier(
        draws=bundle.draws,
        normalized_weights=bundle.normalized_weights,
        draw_ids=expected_draw_ids,
        run_ids=bundle.run_ids,
        nodes=bundle.nodes,
        target_label=bundle.target_label,
        target_semantics=bundle.target_semantics,
        support_id=bundle.support_id,
        candidate_target_id=bundle.candidate_target_id,
        max_indegree=bundle.max_indegree,
        bundle_kind=bundle.bundle_kind,
        producer=bundle.producer,
        source_result_sha256=bundle.source_result_sha256,
    )
    if bundle.draw_ids != expected_draw_ids or bundle.bundle_sha256 != expected_bundle_sha256:
        raise ValueError(f"{name} bundle failed content-hash verification")


def _bundle_provenance(bundle: PosteriorDAGDrawBundle) -> tuple[object, ...]:
    return (
        bundle.nodes,
        bundle.target_label,
        bundle.target_semantics,
        bundle.support_id,
        bundle.candidate_target_id,
        bundle.max_indegree,
        bundle.beta,
    )


def high_posterior_region_coverage(
    method_bundle: PosteriorDAGDrawBundle,
    reference_definition_bundle: PosteriorDAGDrawBundle,
    reference_evaluation_bundle: PosteriorDAGDrawBundle,
    *,
    reference_is_exact: bool,
    diagnostics: ReferenceCoverageDiagnostics | None = None,
    maximum_anchors: int = 12,
    region_mass: float = 0.90,
) -> HighPosteriorRegionResult:
    """Compute Section 5.4 coverage from provenance-bearing DAG bundles."""

    _validate_bundle_integrity(method_bundle, "method")
    _validate_bundle_integrity(reference_definition_bundle, "reference definition")
    _validate_bundle_integrity(reference_evaluation_bundle, "reference evaluation")
    if not isinstance(reference_is_exact, (bool, np.bool_)):
        raise ValueError("reference_is_exact must be boolean")
    if isinstance(region_mass, (bool, np.bool_)):
        raise ValueError("confirmatory HPR requires q=min(12,d(d-1)) and mass 0.90")
    try:
        prepared_region_mass = float(region_mass)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "confirmatory HPR requires q=min(12,d(d-1)) and mass 0.90"
        ) from exc
    if maximum_anchors != 12 or not math.isclose(
        prepared_region_mass, 0.90, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("confirmatory HPR requires q=min(12,d(d-1)) and mass 0.90")
    provenance = _bundle_provenance(method_bundle)
    if (
        _bundle_provenance(reference_definition_bundle) != provenance
        or _bundle_provenance(reference_evaluation_bundle) != provenance
    ):
        raise ValueError("not_applicable_target_mismatch")

    ordered_nodes = method_bundle.nodes
    method = method_bundle.draws
    definition = reference_definition_bundle.draws
    evaluation = reference_evaluation_bundle.draws
    method_probability = np.asarray(method_bundle.normalized_weights, dtype=float)
    definition_probability = np.asarray(
        reference_definition_bundle.normalized_weights, dtype=float
    )
    evaluation_probability = np.asarray(
        reference_evaluation_bundle.normalized_weights, dtype=float
    )

    references_are_exact = (
        reference_definition_bundle.bundle_kind == "exact"
        and reference_evaluation_bundle.bundle_kind == "exact"
    )
    if bool(reference_is_exact) != references_are_exact:
        raise ValueError(
            "reference_is_exact must agree with provenance-verified reference bundles"
        )

    if references_are_exact:
        if diagnostics is not None:
            raise ValueError("exact references must not carry simulation diagnostics")
        if (
            reference_definition_bundle.source_result_sha256
            != reference_evaluation_bundle.source_result_sha256
        ):
            raise ValueError("exact reference bundles must share one exact-result digest")

        def aggregate_exact(
            draws_: Sequence[frozenset[Edge]], probabilities: np.ndarray
        ) -> dict[frozenset[Edge], float]:
            mass: dict[frozenset[Edge], float] = {}
            for draw, probability in zip(draws_, probabilities):
                if probability > 0.0:
                    mass[draw] = mass.get(draw, 0.0) + float(probability)
            return mass

        definition_exact = aggregate_exact(definition, definition_probability)
        evaluation_exact = aggregate_exact(evaluation, evaluation_probability)
        if set(definition_exact) != set(evaluation_exact) or any(
            not math.isclose(
                definition_exact[draw],
                evaluation_exact[draw],
                rel_tol=1e-12,
                abs_tol=1e-14,
            )
            for draw in definition_exact
        ):
            raise ValueError(
                "exact reference definition/evaluation distributions must be identical"
            )
    else:
        definition_ids = set(reference_definition_bundle.run_ids)
        evaluation_ids = set(reference_evaluation_bundle.run_ids)
        method_ids = set(method_bundle.run_ids)
        if definition_ids & evaluation_ids:
            raise ValueError(
                "not_applicable_reference_resolution: definition/evaluation run IDs "
                "must be disjoint"
            )
        if method_ids & (definition_ids | evaluation_ids):
            raise ValueError(
                "not_applicable_reference_resolution: evaluated-method and reference "
                "run IDs must be disjoint"
            )
        raise ValueError(
            "not_applicable_reference_resolution: simulation diagnostics are not "
            "computed by a trusted implementation"
        )

    all_edges = tuple(
        (parent, child)
        for parent in ordered_nodes
        for child in ordered_nodes
        if parent != child
    )
    pip = {
        edge: float(
            sum(
                weight
                for draw, weight in zip(definition, definition_probability)
                if edge in draw
            )
        )
        for edge in all_edges
    }

    def entropy(probability: float) -> float:
        if probability <= 0.0 or probability >= 1.0:
            return 0.0
        return float(
            -probability * math.log(probability)
            - (1.0 - probability) * math.log(1.0 - probability)
        )

    anchor_count = min(maximum_anchors, len(all_edges))
    anchors = tuple(
        sorted(all_edges, key=lambda edge: (-entropy(pip[edge]), edge))[:anchor_count]
    )

    def region(draw: frozenset[Edge]) -> tuple[int, ...]:
        return tuple(int(edge in draw) for edge in anchors)

    definition_mass: dict[tuple[int, ...], float] = {}
    for draw, weight in zip(definition, definition_probability):
        key = region(draw)
        definition_mass[key] = definition_mass.get(key, 0.0) + float(weight)
    ordered_regions = sorted(definition_mass, key=lambda key: (-definition_mass[key], key))
    selected: list[tuple[int, ...]] = []
    cumulative = 0.0
    for key in ordered_regions:
        selected.append(key)
        cumulative += definition_mass[key]
        if cumulative >= prepared_region_mass - 1e-15:
            break
    selected_set = set(selected)

    def coverage(draws_: Sequence[frozenset[Edge]], probabilities: np.ndarray) -> float:
        return float(
            sum(
                weight
                for draw, weight in zip(draws_, probabilities)
                if region(draw) in selected_set
            )
        )

    method_coverage = coverage(method, method_probability)
    reference_coverage = coverage(evaluation, evaluation_probability)
    if reference_coverage <= 0.0:
        raise ValueError("not_applicable_reference_resolution: zero held-out coverage")
    return HighPosteriorRegionResult(
        anchors=anchors,
        regions=tuple(selected),
        method_coverage=method_coverage,
        reference_coverage=reference_coverage,
        coverage_deficit=reference_coverage - method_coverage,
        coverage_ratio=method_coverage / reference_coverage,
        definition_region_mass=float(cumulative),
        reference_is_exact=references_are_exact,
        target_label=method_bundle.target_label,
        target_semantics=method_bundle.target_semantics,
        support_id=method_bundle.support_id,
        candidate_target_id=method_bundle.candidate_target_id,
        max_indegree=method_bundle.max_indegree,
        method_bundle_sha256=method_bundle.bundle_sha256,
        reference_definition_bundle_sha256=reference_definition_bundle.bundle_sha256,
        reference_evaluation_bundle_sha256=reference_evaluation_bundle.bundle_sha256,
        reference_uncertainty_mcse=(
            0.0
        ),
        reference_diagnostics=None,
    )


def threshold_attainment_curve(
    durations: Sequence[float],
    attained: Sequence[bool],
) -> tuple[ThresholdAttainmentPoint, ...]:
    """Kaplan--Meier-style cumulative threshold attainment with censoring."""

    times = np.asarray(durations, dtype=float)
    events = np.asarray(attained)
    if times.ndim != 1 or events.shape != times.shape or times.size == 0:
        raise ValueError("durations and attained must be non-empty aligned vectors")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("durations must be finite and non-negative")
    if events.dtype.kind != "b":
        if not all(isinstance(value, (bool, np.bool_)) for value in attained):
            raise ValueError("attained must contain booleans")
        events = events.astype(bool)
    at_risk = int(times.size)
    survival = 1.0
    output: list[ThresholdAttainmentPoint] = []
    for time in sorted(set(map(float, times))):
        mask = times == time
        event_count = int(np.sum(events[mask]))
        censor_count = int(np.sum(~events[mask]))
        if event_count:
            survival *= 1.0 - event_count / at_risk
        output.append(
            ThresholdAttainmentPoint(
                time=time,
                at_risk=at_risk,
                attained=event_count,
                censored=censor_count,
                survival=float(survival),
                cumulative_attainment=float(1.0 - survival),
            )
        )
        at_risk -= event_count + censor_count
    return tuple(output)


def graph_size_wasserstein(
    estimated: GraphSizeDistribution,
    reference: GraphSizeDistribution,
) -> Dict[str, object]:
    """Compute edge-count W1 only from provenance-bound common-target laws."""

    if not isinstance(estimated, GraphSizeDistribution) or not isinstance(
        reference, GraphSizeDistribution
    ):
        raise TypeError("graph-size distance requires GraphSizeDistribution records")
    estimated.validate_integrity()
    reference.validate_integrity()
    if (
        estimated.target_label != reference.target_label
        or estimated.target_semantics != reference.target_semantics
    ):
        raise ValueError("graph-size exact-target distance requires matching targets")
    if estimated.support_id != reference.support_id:
        raise ValueError("graph-size distance requires the same declared support")
    if (
        estimated.node_count != reference.node_count
        or estimated.max_indegree != reference.max_indegree
    ):
        raise ValueError("graph-size distance requires matching support bounds")
    if not math.isclose(estimated.beta, reference.beta, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("graph-size distance requires matching target temperatures")
    k_cap = sum(
        min(rank, estimated.max_indegree) for rank in range(estimated.node_count)
    )

    def normalized_vector(mass: Mapping[int, float], name: str) -> np.ndarray:
        vector = np.zeros(k_cap + 1, dtype=float)
        for raw_count, raw_probability in mass.items():
            if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)):
                raise ValueError(f"{name} edge counts must be integers")
            count = int(raw_count)
            probability = float(raw_probability)
            if not 0 <= count <= k_cap:
                raise ValueError(f"{name} edge count {count} lies outside [0, {k_cap}]")
            if not np.isfinite(probability) or probability < 0.0:
                raise ValueError(f"{name} masses must be finite and non-negative")
            vector[count] += probability
        total = float(np.sum(vector))
        if total <= 0.0 or not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError(f"{name} masses must sum to one")
        return vector / total

    estimated_vector = normalized_vector(estimated, "estimated")
    reference_vector = normalized_vector(reference, "reference")
    edge_units = float(
        np.sum(
            np.abs(
                np.cumsum(estimated_vector)[:-1]
                - np.cumsum(reference_vector)[:-1]
            )
        )
    )
    return {
        "target_label": estimated.target_label,
        "target_semantics": estimated.target_semantics,
        "support_id": estimated.support_id,
        "beta": estimated.beta,
        "estimated_producer": estimated.producer,
        "reference_producer": reference.producer,
        "k_cap": int(k_cap),
        "graph_size_W1_edges": edge_units,
        "graph_size_W1_normalized": edge_units / max(1, k_cap),
    }


def _edge_probability(edge_probs: Mapping[Edge, float], edge: Edge) -> float:
    probability = float(edge_probs.get(edge, 0.0))
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"invalid edge probability for {edge}: {probability}")
    return probability


def _complete_edge_probability_map(
    edge_probs: Mapping[Edge, float], nodes: Sequence[str], name: str
) -> dict[Edge, float]:
    if not isinstance(edge_probs, Mapping):
        raise TypeError(f"{name} must be a mapping")
    ordered_nodes = tuple(nodes)
    if (
        len(ordered_nodes) < 2
        or len(set(ordered_nodes)) != len(ordered_nodes)
        or any(not isinstance(node, str) or not node for node in ordered_nodes)
    ):
        raise ValueError("nodes must contain at least two unique non-empty strings")
    expected = {
        (parent, child)
        for parent in ordered_nodes
        for child in ordered_nodes
        if parent != child
    }
    observed: set[Edge] = set()
    prepared: dict[Edge, float] = {}
    for raw_edge, raw_probability in edge_probs.items():
        if (
            not isinstance(raw_edge, tuple)
            or len(raw_edge) != 2
            or not all(isinstance(node, str) for node in raw_edge)
        ):
            raise ValueError(f"{name} contains a malformed edge key")
        edge = (raw_edge[0], raw_edge[1])
        if edge in observed:
            raise ValueError(f"{name} contains a duplicate edge")
        observed.add(edge)
        if isinstance(raw_probability, (bool, np.bool_)):
            raise ValueError(f"{name} contains a boolean probability")
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid edge probability for {edge}: {probability}")
        prepared[edge] = probability
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{name} must contain every directed non-self pair exactly once; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for left_index, left in enumerate(ordered_nodes):
        for right in ordered_nodes[left_index + 1 :]:
            if prepared[(left, right)] + prepared[(right, left)] > 1.0 + 1e-10:
                raise ValueError(f"opposite edge PIPs sum above one for {(left, right)}")
    return prepared


def _lexicographic_average_precision(
    truth: np.ndarray, probability: np.ndarray, edge_ids: Sequence[object]
) -> float:
    """Stepwise AP with the protocol's deterministic edge tie-breaking."""

    positives = int(np.sum(truth))
    if positives == 0:
        return float("nan")
    order = sorted(
        range(len(truth)),
        key=lambda index: (-float(probability[index]), edge_ids[index]),
    )
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if int(truth[index]) == 1:
            hits += 1
            precision_sum += hits / rank
    return float(precision_sum / positives)


def _pair_state(graph: nx.DiGraph, left: str, right: str) -> int:
    forward = graph.has_edge(left, right)
    reverse = graph.has_edge(right, left)
    if forward and reverse:
        raise ValueError("graph contains both orientations of one adjacency")
    return 1 if forward else 2 if reverse else 0


def dag_pairwise_shd(estimated: nx.DiGraph, truth: nx.DiGraph, nodes: Sequence[str]) -> int:
    """Pair-state SHD where a reversal costs one operation."""

    return sum(
        _pair_state(estimated, nodes[left], nodes[right])
        != _pair_state(truth, nodes[left], nodes[right])
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    )


def _cpdag_pair_state(cpdag: CPDAG, left: str, right: str) -> int:
    """Encode none/undirected/forward/reverse for one unordered node pair."""

    if cpdag.is_undirected(left, right):
        return 1
    if cpdag.is_directed(left, right):
        return 2
    if cpdag.is_directed(right, left):
        return 3
    return 0


def cpdag_pairwise_shd(estimated: CPDAG, truth: CPDAG, nodes: Sequence[str]) -> int:
    """CPDAG SHD with one cost per adjacency or endpoint-mark disagreement.

    For a fixed unordered pair, changing no-edge to any adjacency costs one;
    changing undirected to directed or reversing a directed endpoint mark also
    costs one.  This is exactly a mismatch of the four pair states.
    """

    wanted_nodes = set(nodes)
    if set(estimated.nodes) != wanted_nodes or set(truth.nodes) != wanted_nodes:
        raise ValueError("both CPDAGs must contain exactly the supplied nodes")
    return sum(
        _cpdag_pair_state(estimated, nodes[left], nodes[right])
        != _cpdag_pair_state(truth, nodes[left], nodes[right])
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    )


def _set_recovery_metrics(
    estimated: set[object], truth: set[object], prefix: str
) -> Dict[str, float]:
    """Return DAGBagM-style recovery metrics for one declared edge universe."""

    true_positive = len(estimated & truth)
    false_positive = len(estimated - truth)
    false_negative = len(truth - estimated)
    selected = true_positive + false_positive
    if truth:
        true_positive_rate = true_positive / len(truth)
        f1 = 2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
    else:
        # A power or F1 value on an empty truth set is not scientifically
        # meaningful.  The false-edge count is the predeclared null endpoint.
        true_positive_rate = float("nan")
        f1 = float("nan")
    false_discovery_rate = false_positive / max(1, selected)
    return {
        f"{prefix}_true_edge_count": float(len(truth)),
        f"{prefix}_selected_edge_count": float(selected),
        f"{prefix}_true_positive_count": float(true_positive),
        f"{prefix}_false_positive_count": float(false_positive),
        f"{prefix}_false_negative_count": float(false_negative),
        f"{prefix}_tpr": float(true_positive_rate),
        f"{prefix}_power": float(true_positive_rate),
        f"{prefix}_fdr": float(false_discovery_rate),
        f"{prefix}_precision": float(1.0 - false_discovery_rate),
        f"{prefix}_f1": float(f1),
    }


def dagbagm_graph_recovery_metrics(
    estimated_dag: nx.DiGraph,
    true_dag: nx.DiGraph,
    nodes: Sequence[str],
    *,
    node_types: Mapping[str, str] | None = None,
) -> Dict[str, float]:
    """Compute paper-compatible TPR, FDR, and F1 for mixed-data DAG recovery.

    Skeleton edges are unordered. Directed edges require the exact orientation,
    so reversing a true arc creates one false positive and one false negative.
    If ``node_types`` is supplied, the output also contains the primary
    cross-type endpoint and every ordered parent-type/child-type stratum.
    """

    ordered_nodes = tuple(nodes)
    wanted_nodes = set(ordered_nodes)
    if (
        len(ordered_nodes) < 2
        or len(wanted_nodes) != len(ordered_nodes)
        or any(not isinstance(node, str) or not node for node in ordered_nodes)
    ):
        raise ValueError("nodes must contain at least two unique non-empty strings")
    for name, graph in (("estimated_dag", estimated_dag), ("true_dag", true_dag)):
        if (
            not isinstance(graph, nx.DiGraph)
            or isinstance(graph, nx.MultiDiGraph)
            or set(graph.nodes) != wanted_nodes
        ):
            raise ValueError(f"{name} must be a simple directed graph on exactly the nodes")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f"{name} must be acyclic")

    estimated_directed = set(estimated_dag.edges())
    true_directed = set(true_dag.edges())
    estimated_skeleton: set[object] = {
        frozenset(edge) for edge in estimated_directed
    }
    true_skeleton: set[object] = {frozenset(edge) for edge in true_directed}
    metrics = _set_recovery_metrics(estimated_skeleton, true_skeleton, "skeleton")
    metrics.update(
        _set_recovery_metrics(
            set(estimated_directed), set(true_directed), "directed"
        )
    )

    if node_types is None:
        return metrics
    if not isinstance(node_types, Mapping) or set(node_types) != wanted_nodes:
        raise ValueError("node_types must map exactly the supplied nodes")
    prepared_types: dict[str, str] = {}
    for node in ordered_nodes:
        label = node_types[node]
        if (
            not isinstance(label, str)
            or not label
            or not label[0].isalpha()
            or any(not (character.isalnum() or character == "_") for character in label)
        ):
            raise ValueError("node type labels must be non-empty alphanumeric identifiers")
        prepared_types[node] = label.lower()

    def edges_of_types(
        edges: set[tuple[str, str]], parent_type: str, child_type: str
    ) -> set[object]:
        return {
            edge
            for edge in edges
            if prepared_types[edge[0]] == parent_type
            and prepared_types[edge[1]] == child_type
        }

    estimated_cross: set[object] = {
        edge
        for edge in estimated_directed
        if prepared_types[edge[0]] != prepared_types[edge[1]]
    }
    true_cross: set[object] = {
        edge
        for edge in true_directed
        if prepared_types[edge[0]] != prepared_types[edge[1]]
    }
    metrics.update(
        _set_recovery_metrics(
            estimated_cross, true_cross, "directed_cross_type"
        )
    )
    type_labels = sorted(set(prepared_types.values()))
    for parent_type in type_labels:
        for child_type in type_labels:
            prefix = f"directed_{parent_type}_to_{child_type}"
            metrics.update(
                _set_recovery_metrics(
                    edges_of_types(estimated_directed, parent_type, child_type),
                    edges_of_types(true_directed, parent_type, child_type),
                    prefix,
                )
            )
    return metrics


def evaluate_edge_probabilities(
    edge_probs: Mapping[Edge, float],
    true_dag: nx.DiGraph,
    nodes: Sequence[str],
    *,
    representative_dag: nx.DiGraph | None = None,
    node_types: Mapping[str, str] | None = None,
) -> Dict[str, float]:
    """Compute all primary endpoints once for one dataset and one estimate."""

    ordered_nodes = tuple(nodes)
    edge_probs = _complete_edge_probability_map(
        edge_probs, ordered_nodes, "edge probabilities"
    )
    if not isinstance(true_dag, nx.DiGraph) or set(true_dag.nodes) != set(ordered_nodes):
        raise ValueError("true_dag must contain exactly the supplied nodes")
    if not nx.is_directed_acyclic_graph(true_dag):
        raise ValueError("true_dag must be acyclic")
    if representative_dag is not None and (
        not isinstance(representative_dag, nx.DiGraph)
        or set(representative_dag.nodes) != set(ordered_nodes)
        or not nx.is_directed_acyclic_graph(representative_dag)
    ):
        raise ValueError("representative_dag must be an acyclic graph on exactly the nodes")
    directed_truth: list[int] = []
    directed_probability: list[float] = []
    directed_ids: list[Edge] = []
    skeleton_truth: list[int] = []
    skeleton_probability: list[float] = []
    skeleton_ids: list[tuple[str, str]] = []
    for parent in ordered_nodes:
        for child in ordered_nodes:
            if parent == child:
                continue
            directed_truth.append(int(true_dag.has_edge(parent, child)))
            directed_probability.append(_edge_probability(edge_probs, (parent, child)))
            directed_ids.append((parent, child))
    for left_idx, left in enumerate(ordered_nodes):
        for right in ordered_nodes[left_idx + 1 :]:
            skeleton_truth.append(
                int(true_dag.has_edge(left, right) or true_dag.has_edge(right, left))
            )
            probability = _edge_probability(edge_probs, (left, right)) + _edge_probability(
                edge_probs, (right, left)
            )
            if probability > 1.0 + 1e-10:
                raise ValueError(f"opposite edge PIPs sum above one for {(left, right)}")
            skeleton_probability.append(min(1.0, probability))
            skeleton_ids.append((left, right))

    y_directed = np.asarray(directed_truth, dtype=int)
    p_directed = np.asarray(directed_probability, dtype=float)
    y_skeleton = np.asarray(skeleton_truth, dtype=int)
    p_skeleton = np.asarray(skeleton_probability, dtype=float)

    def component_brier(y: np.ndarray, p: np.ndarray, positive: bool) -> float:
        mask = y == int(positive)
        return float(np.mean((p[mask] - y[mask]) ** 2)) if np.any(mask) else float("nan")

    directed_ap = _lexicographic_average_precision(
        y_directed, p_directed, directed_ids
    )
    skeleton_ap = _lexicographic_average_precision(
        y_skeleton, p_skeleton, skeleton_ids
    )
    directed_auroc = (
        float("nan")
        if len(np.unique(y_directed)) < 2
        else float(roc_auc_score(y_directed, p_directed))
    )
    true_edges = list(true_dag.edges())
    reverse_mass = float(
        sum(_edge_probability(edge_probs, (child, parent)) for parent, child in true_edges)
    )
    metrics: Dict[str, float] = {
        "directed_auprc": directed_ap,
        "directed_auroc": directed_auroc,
        "directed_brier": float(np.mean((p_directed - y_directed) ** 2)),
        "skeleton_auprc": skeleton_ap,
        "skeleton_ap_lift": (
            float("nan")
            if float(np.mean(y_skeleton)) == 0.0
            else skeleton_ap / float(np.mean(y_skeleton))
        ),
        "skeleton_prevalence": float(np.mean(y_skeleton)),
        "skeleton_brier": float(np.mean((p_skeleton - y_skeleton) ** 2)),
        "skeleton_positive_brier": component_brier(y_skeleton, p_skeleton, True),
        "skeleton_negative_brier": component_brier(y_skeleton, p_skeleton, False),
        "reverse_edge_posterior_mass": reverse_mass,
        "reverse_edge_posterior_mass_per_true_edge": (
            float("nan") if not true_edges else reverse_mass / len(true_edges)
        ),
    }
    if representative_dag is not None:
        metrics.update(
            dagbagm_graph_recovery_metrics(
                representative_dag,
                true_dag,
                ordered_nodes,
                node_types=node_types,
            )
        )
        raw_shd = dag_pairwise_shd(representative_dag, true_dag, ordered_nodes)
        denominator = max(1, len(ordered_nodes) * (len(ordered_nodes) - 1) // 2)
        representative_cpdag = dag_to_cpdag(representative_dag)
        true_cpdag = dag_to_cpdag(true_dag)
        cpdag_shd = cpdag_pairwise_shd(
            representative_cpdag, true_cpdag, ordered_nodes
        )
        estimated_skeleton = {
            frozenset(edge) for edge in representative_dag.edges()
        }
        true_skeleton = {frozenset(edge) for edge in true_dag.edges()}
        skeleton_shd = len(estimated_skeleton.symmetric_difference(true_skeleton))
        metrics.update(
            {
                "directed_pair_shd": float(raw_shd),
                "directed_pair_shd_normalized": float(raw_shd / denominator),
                "skeleton_shd": float(skeleton_shd),
                "skeleton_shd_normalized": float(skeleton_shd / denominator),
                "cpdag_shd": float(cpdag_shd),
                "cpdag_shd_normalized": float(cpdag_shd / denominator),
            }
        )
    return metrics


def evaluate_seed_estimands(
    seed_edge_probs: Sequence[Mapping[Edge, float]],
    true_dag: nx.DiGraph,
    nodes: Sequence[str],
    *,
    representative_max_indegree: int | None = None,
    node_types: Mapping[str, str] | None = None,
) -> SeedEstimandResult:
    """Report single-run and multi-run-ensemble estimands without conflating them."""

    if not seed_edge_probs:
        raise ValueError("at least one seed result is required")
    prepared_seed_edge_probs = tuple(
        _complete_edge_probability_map(
            edge_probs, nodes, f"seed_edge_probs[{index}]"
        )
        for index, edge_probs in enumerate(seed_edge_probs)
    )
    representative_dags: list[nx.DiGraph] = []
    if representative_max_indegree is not None:
        from ordersmc.graphs import optimal_bayes_action_dag

        representative_dags = [
            optimal_bayes_action_dag(
                edge_probs, nodes, max_indegree=representative_max_indegree
            )
            for edge_probs in prepared_seed_edge_probs
        ]
    per_seed = tuple(
        evaluate_edge_probabilities(
            edge_probs,
            true_dag,
            nodes,
            representative_dag=(
                representative_dags[index] if representative_dags else None
            ),
            node_types=node_types,
        )
        for index, edge_probs in enumerate(prepared_seed_edge_probs)
    )
    metric_names = sorted(per_seed[0])
    mean_single = {
        metric: float(np.nanmean([values[metric] for values in per_seed]))
        for metric in metric_names
    }
    all_edges = [
        (parent, child)
        for parent in nodes
        for child in nodes
        if parent != child
    ]
    ensemble = {
        edge: float(
            np.mean([_edge_probability(seed, edge) for seed in prepared_seed_edge_probs])
        )
        for edge in all_edges
    }
    ensemble_representative = None
    if representative_max_indegree is not None:
        from ordersmc.graphs import optimal_bayes_action_dag

        ensemble_representative = optimal_bayes_action_dag(
            ensemble, nodes, max_indegree=representative_max_indegree
        )
    ensemble_metrics = evaluate_edge_probabilities(
        ensemble,
        true_dag,
        nodes,
        representative_dag=ensemble_representative,
        node_types=node_types,
    )
    return SeedEstimandResult(
        mean_single_run_metrics=mean_single,
        ensemble_metrics=ensemble_metrics,
        per_seed_metrics=per_seed,
        ensemble_edge_probs=ensemble,
        ensemble_compute_multiplier=len(prepared_seed_edge_probs),
        per_seed_representative_edges=tuple(
            tuple(sorted(graph.edges())) for graph in representative_dags
        ),
        ensemble_representative_edges=(
            ()
            if ensemble_representative is None
            else tuple(sorted(ensemble_representative.edges()))
        ),
    )


def evaluate_exact_target_error(
    estimated: PosteriorEdgeDistribution,
    exact: PosteriorEdgeDistribution,
    *,
    top_k: int = 20,
) -> Dict[str, object]:
    """Compare complete PIPs only on an identical provenance-bound target."""

    _validate_edge_distribution_integrity(estimated, "estimated")
    _validate_edge_distribution_integrity(exact, "exact")
    if exact.reference_kind != "exact":
        raise ValueError("the exact-target reference must be marked exact")
    if (
        estimated.target_label != exact.target_label
        or estimated.target_semantics != exact.target_semantics
    ):
        raise ValueError("exact-target error is undefined across different target labels")
    if estimated.support_id != exact.support_id:
        raise ValueError("exact-target error requires an identical candidate support")
    if estimated.candidate_target_id != exact.candidate_target_id:
        raise ValueError("exact-target error requires an identical candidate target")
    if estimated.max_indegree != exact.max_indegree:
        raise ValueError("exact-target error requires an identical indegree support")
    if not math.isclose(estimated.beta, exact.beta, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("exact-target error requires an identical target temperature")
    if estimated.nodes != exact.nodes:
        raise ValueError("exact-target error requires identical ordered node sets")
    nodes = estimated.nodes
    edges = [(parent, child) for parent in nodes for child in nodes if parent != child]
    estimated_vector = np.asarray([_edge_probability(estimated, edge) for edge in edges])
    exact_vector = np.asarray([_edge_probability(exact, edge) for edge in edges])
    difference = estimated_vector - exact_vector
    skeleton_estimated = []
    skeleton_exact = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            skeleton_estimated.append(
                _edge_probability(estimated, (left, right))
                + _edge_probability(estimated, (right, left))
            )
            skeleton_exact.append(
                _edge_probability(exact, (left, right))
                + _edge_probability(exact, (right, left))
            )
    top_estimated = sorted(
        range(len(edges)), key=lambda index: (-estimated_vector[index], edges[index])
    )[: min(top_k, len(edges))]
    top_exact = sorted(
        range(len(edges)), key=lambda index: (-exact_vector[index], edges[index])
    )[: min(top_k, len(edges))]
    union = sorted(set(top_estimated) | set(top_exact))
    correlation = (
        float("nan")
        if len(union) < 2
        else float(
            spearmanr(estimated_vector[union], exact_vector[union]).statistic
        )
    )
    result = {
        "target_label": estimated.target_label,
        "target_semantics": estimated.target_semantics,
        "support_id": estimated.support_id,
        "beta": estimated.beta,
        "estimated_producer": estimated.producer,
        "exact_producer": exact.producer,
        "directed_pip_rmse": float(np.sqrt(np.mean(difference**2))),
        "skeleton_pip_rmse": float(
            np.sqrt(
                np.mean(
                    (np.asarray(skeleton_estimated) - np.asarray(skeleton_exact)) ** 2
                )
            )
        ),
        "maximum_absolute_pip_error": float(np.max(np.abs(difference), initial=0.0)),
        "top_edge_spearman": correlation,
    }
    if (estimated.log_ratio is None) != (exact.log_ratio is None):
        raise ValueError("both distributions must carry log_ratio or neither may carry it")
    if estimated.log_ratio is not None and exact.log_ratio is not None:
        result["log_ratio_error"] = float(estimated.log_ratio - exact.log_ratio)
        result["absolute_log_ratio_error"] = abs(result["log_ratio_error"])
    return result


_E6_SCORE_CONTEXT_FIELDS = (
    "max_parents",
    "tail_edge_penalty_after",
    "tail_edge_penalty",
    "lam",
    "edge_penalty",
    "alpha",
    "tau",
    "glm_tau",
    "discrete_score",
    "clg_ig_shape",
    "clg_ig_scale",
    "standardize_continuous",
    "score_method",
    "dg_penalty_discount",
    "dg_ridge",
)


def _e6_analysis_context_sha256(table: OrderCandidateTable) -> str:
    provenance = getattr(table, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise ValueError("E6 tables require candidate provenance")
    data_fingerprint = provenance.get("data_fingerprint")
    if (
        not isinstance(data_fingerprint, str)
        or len(data_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in data_fingerprint)
    ):
        raise ValueError("E6 tables require a lowercase data fingerprint")
    if provenance.get("nodes") != list(table.nodes):
        raise ValueError("E6 table provenance nodes do not match")
    metadata = provenance.get("metadata")
    score_implementation = provenance.get("score_implementation")
    build_configuration = provenance.get("candidate_build_config")
    if not all(
        isinstance(value, Mapping)
        for value in (metadata, score_implementation, build_configuration)
    ):
        raise ValueError("E6 tables require metadata, score implementation, and build config")
    missing = [
        field for field in _E6_SCORE_CONTEXT_FIELDS if field not in build_configuration
    ]
    if missing:
        raise ValueError(f"E6 candidate build config lacks score fields: {missing}")
    return _canonical_json_sha256(
        {
            "schema": "ordersmc-e6-analysis-context-v1",
            "data_fingerprint": data_fingerprint,
            "nodes": list(table.nodes),
            "metadata": dict(metadata),
            "score_schema_version": provenance.get("score_schema_version"),
            "score_implementation": dict(score_implementation),
            "score_and_prior_configuration": {
                field: build_configuration[field]
                for field in _E6_SCORE_CONTEXT_FIELDS
            },
        }
    )


def _family_target_terms(
    table: OrderCandidateTable,
) -> tuple[dict[tuple[str, ...], tuple[str, str]], ...]:
    nodes = tuple(table.nodes)
    return tuple(
        {
            tuple(sorted(nodes[int(parent)] for parent in row if int(parent) >= 0)): (
                float(loglik).hex(),
                float(logprior).hex(),
            )
            for row, loglik, logprior in zip(parent_sets, child_loglik, child_logprior)
        }
        for parent_sets, child_loglik, child_logprior in zip(
            table.parent_sets, table.loglik, table.logprior
        )
    )


def decompose_screening_and_monte_carlo_error(
    estimated_restricted: PosteriorEdgeDistribution,
    exact_restricted: PosteriorEdgeDistribution,
    exact_complete: PosteriorEdgeDistribution,
    restricted_table: OrderCandidateTable,
    complete_table: OrderCandidateTable,
) -> Dict[str, object]:
    """Return the provenance-bound E6 identity and its non-additive cross-term."""

    for distribution, name in (
        (estimated_restricted, "estimated restricted"),
        (exact_restricted, "exact restricted"),
        (exact_complete, "exact complete"),
    ):
        _validate_edge_distribution_integrity(distribution, name)
    if exact_restricted.reference_kind != "exact" or exact_complete.reference_kind != "exact":
        raise ValueError("E6 restricted and complete references must be exact-DP results")
    if estimated_restricted.reference_kind != "approximate":
        raise ValueError("E6 estimated restricted PIPs must be an approximate result")
    if estimated_restricted.target_label != exact_restricted.target_label or (
        estimated_restricted.target_semantics != exact_restricted.target_semantics
    ):
        raise ValueError("E6 Monte Carlo terms must share one restricted target")
    if (
        estimated_restricted.support_id != exact_restricted.support_id
        or estimated_restricted.candidate_target_id
        != exact_restricted.candidate_target_id
    ):
        raise ValueError("E6 Monte Carlo terms must share one restricted candidate target")
    if exact_complete.target_label != exact_restricted.target_label or (
        exact_complete.target_semantics != exact_restricted.target_semantics
    ):
        raise ValueError("E6 complete and restricted references must use one target type")
    if not (
        math.isclose(estimated_restricted.beta, 1.0, abs_tol=1e-15, rel_tol=0.0)
        and math.isclose(exact_restricted.beta, 1.0, abs_tol=1e-15, rel_tol=0.0)
        and math.isclose(exact_complete.beta, 1.0, abs_tol=1e-15, rel_tol=0.0)
    ):
        raise ValueError("E6 decomposition requires beta-one posterior records")
    if tuple(restricted_table.nodes) != exact_restricted.nodes or (
        tuple(complete_table.nodes) != exact_complete.nodes
    ):
        raise ValueError("E6 records do not match their supplied candidate tables")
    expected_restricted_support = (
        f"candidate-sha256:{candidate_support_sha256(restricted_table)}"
    )
    expected_complete_support = f"candidate-sha256:{candidate_support_sha256(complete_table)}"
    expected_restricted_target = (
        f"candidate-target-sha256:{_candidate_target_sha256(restricted_table)}"
    )
    expected_complete_target = (
        f"candidate-target-sha256:{_candidate_target_sha256(complete_table)}"
    )
    if (
        exact_restricted.support_id != expected_restricted_support
        or exact_restricted.candidate_target_id != expected_restricted_target
        or exact_complete.support_id != expected_complete_support
        or exact_complete.candidate_target_id != expected_complete_target
    ):
        raise ValueError("E6 posterior records are not bound to the supplied candidate tables")
    if _e6_analysis_context_sha256(restricted_table) != _e6_analysis_context_sha256(
        complete_table
    ):
        raise ValueError("E6 restricted and complete tables do not share one dataset/score context")
    restricted_terms = _family_target_terms(restricted_table)
    complete_terms = _family_target_terms(complete_table)
    for child, (restricted_families, complete_families) in enumerate(
        zip(restricted_terms, complete_terms)
    ):
        if not set(restricted_families).issubset(complete_families):
            raise ValueError(f"E6 restricted support is not a subset for child {child}")
        if any(
            target_terms != complete_families[parents]
            for parents, target_terms in restricted_families.items()
        ):
            raise ValueError(f"E6 overlapping family scores differ for child {child}")

    nodes = exact_restricted.nodes
    edges = [(parent, child) for parent in nodes for child in nodes if parent != child]
    estimated_vector = np.asarray(
        [_edge_probability(estimated_restricted, edge) for edge in edges]
    )
    restricted_vector = np.asarray(
        [_edge_probability(exact_restricted, edge) for edge in edges]
    )
    complete_vector = np.asarray(
        [_edge_probability(exact_complete, edge) for edge in edges]
    )
    monte_carlo = estimated_vector - restricted_vector
    screening = restricted_vector - complete_vector
    total = estimated_vector - complete_vector
    cross_term = float(2.0 * np.mean(monte_carlo * screening))
    denominator = float(np.linalg.norm(monte_carlo) * np.linalg.norm(screening))
    return {
        "edges": edges,
        "analysis_context_sha256": _e6_analysis_context_sha256(restricted_table),
        "restricted_support_id": exact_restricted.support_id,
        "complete_support_id": exact_complete.support_id,
        "monte_carlo_vector": monte_carlo,
        "screening_vector": screening,
        "total_vector": total,
        "monte_carlo_rmse": float(np.sqrt(np.mean(monte_carlo**2))),
        "screening_rmse": float(np.sqrt(np.mean(screening**2))),
        "total_rmse": float(np.sqrt(np.mean(total**2))),
        "cross_term": cross_term,
        "cosine": (
            float("nan")
            if denominator == 0.0
            else float(np.dot(monte_carlo, screening) / denominator)
        ),
        "identity_max_absolute_error": float(
            np.max(np.abs(total - monte_carlo - screening), initial=0.0)
        ),
    }


def stratified_paired_bootstrap(
    differences: Sequence[float],
    strata: Sequence[object],
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Bootstrap complete paired datasets within cells, weighting cells equally."""

    values = np.asarray(differences, dtype=float)
    labels = list(strata)
    if values.ndim != 1 or len(labels) != values.size or values.size == 0:
        raise ValueError("differences and strata must be non-empty aligned vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite; handle timeouts before bootstrapping")
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    try:
        unique_strata = list(dict.fromkeys(labels))
    except TypeError as exc:
        raise ValueError("strata labels must be hashable") from exc
    stratum_indices = [
        np.asarray([index for index, observed in enumerate(labels) if observed == label])
        for label in unique_strata
    ]
    estimate = float(np.mean([np.mean(values[index]) for index in stratum_indices]))
    draws = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        cell_means = []
        for index in stratum_indices:
            sampled = rng.choice(index, size=len(index), replace=True)
            cell_means.append(float(np.mean(values[sampled])))
        draws[draw] = float(np.mean(cell_means))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    one_sided_lower, one_sided_upper = np.quantile(
        draws, [1.0 - confidence, confidence]
    )
    return PairedBootstrapResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        confidence=float(confidence),
        resamples=int(resamples),
        stratum_count=len(unique_strata),
        one_sided_lower=float(one_sided_lower),
        one_sided_upper=float(one_sided_upper),
    )


def assess_noninferiority(
    bootstrap: PairedBootstrapResult,
    *,
    margin: float,
    higher_is_better: bool,
) -> NoninferiorityResult:
    """Assess a predeclared paired noninferiority contrast ``new - reference``."""

    if margin < 0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")
    established = (
        bootstrap.one_sided_lower > -margin
        if higher_is_better
        else bootstrap.one_sided_upper < margin
    )
    return NoninferiorityResult(
        difference=bootstrap.estimate,
        ci_lower=bootstrap.one_sided_lower,
        ci_upper=bootstrap.one_sided_upper,
        margin=float(margin),
        higher_is_better=bool(higher_is_better),
        established=bool(established),
    )
