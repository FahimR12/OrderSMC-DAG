"""Deterministic target-specific structural-prior calibration references."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal, Mapping

import numpy as np

from ordersmc.core import OrderCandidateTable
from ordersmc.config import OrderSMCConfig, candidate_build_config

from .exact_order import exact_order_dp
from .exact_structure import exact_structure_dp

PriorTarget = Literal["order", "structure"]


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PriorCalibrationResult:
    target_label: PriorTarget
    lambda_target: float
    achieved_mean_indegree: float
    requested_mean_indegree: float
    absolute_error: float
    iterations: int
    bracket_lower: float
    bracket_upper: float
    final_bracket_lower: float
    final_bracket_upper: float
    tolerance: float
    maximum_iterations: int
    max_nodes: int
    method: str
    node_count: int
    candidate_max_indegree: int
    candidate_support_sha256: str
    schema: str = "ordersmc-protocol-v2-prior-calibration-v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def calibration_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PriorRetargetAudit:
    """Content-addressed proof that retargeting did not rescore any family."""

    target_label: PriorTarget
    lambda_target: float
    source_candidate_support_sha256: str
    target_candidate_support_sha256: str
    source_local_likelihood_sha256: str
    target_local_likelihood_sha256: str
    local_likelihood_arrays_bit_identical: bool
    source_candidate_build_config_sha256: str
    target_candidate_build_config_sha256: str
    calibration_sha256: str | None
    schema: str = "ordersmc-protocol-v2-prior-retarget-audit-v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def audit_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def candidate_support_sha256(table: OrderCandidateTable) -> str:
    """Hash only nodes and retained parent families, excluding opened data scores."""

    payload = {
        "schema": "ordersmc-candidate-support-v1",
        "nodes": list(table.nodes),
        "parent_sets": [
            sorted(
                [
                    sorted(int(parent) for parent in row if int(parent) >= 0)
                    for row in child_sets
                ]
            )
            for child_sets in table.parent_sets
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_local_likelihood_sha256(table: OrderCandidateTable) -> str:
    """Hash the exact scored-family arrays while excluding structural priors.

    Unlike :func:`candidate_support_sha256`, this identity is deliberately
    sensitive to candidate-row order, NumPy dtype, shape, and raw floating-point
    bits.  It is therefore suitable for proving that two target-specific tables
    came from one physical local-likelihood evaluation rather than a rescore.
    """

    if (
        len(table.parent_sets) != len(table.nodes)
        or len(table.loglik) != len(table.nodes)
    ):
        raise ValueError("candidate table dimensions do not match its nodes")

    children: list[dict[str, object]] = []
    for parent_sets, loglik in zip(table.parent_sets, table.loglik):
        parent_array = np.asarray(parent_sets)
        likelihood_array = np.asarray(loglik)
        if parent_array.ndim != 2 or likelihood_array.ndim != 1:
            raise ValueError("candidate parent sets/loglik arrays have invalid dimensions")
        if parent_array.shape[0] != likelihood_array.shape[0]:
            raise ValueError("candidate parent sets and loglik rows do not match")
        contiguous_parents = np.ascontiguousarray(parent_array)
        contiguous_likelihood = np.ascontiguousarray(likelihood_array)
        children.append(
            {
                "parent_sets_dtype": contiguous_parents.dtype.str,
                "parent_sets_shape": list(contiguous_parents.shape),
                "parent_sets_sha256": hashlib.sha256(
                    contiguous_parents.tobytes(order="C")
                ).hexdigest(),
                "loglik_dtype": contiguous_likelihood.dtype.str,
                "loglik_shape": list(contiguous_likelihood.shape),
                "loglik_sha256": hashlib.sha256(
                    contiguous_likelihood.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return _canonical_sha256(
        {
            "schema": "ordersmc-candidate-local-likelihood-identity-v1",
            "nodes": list(table.nodes),
            "children": children,
        }
    )


def _local_likelihood_arrays_bit_identical(
    source: OrderCandidateTable, target: OrderCandidateTable
) -> bool:
    if (
        list(source.nodes) != list(target.nodes)
        or len(source.loglik) != len(target.loglik)
    ):
        return False
    for source_values, target_values in zip(source.loglik, target.loglik):
        source_array = np.asarray(source_values)
        target_array = np.asarray(target_values)
        if (
            source_array.dtype != target_array.dtype
            or source_array.shape != target_array.shape
        ):
            return False
        if (
            np.ascontiguousarray(source_array).tobytes(order="C")
            != np.ascontiguousarray(target_array).tobytes(order="C")
        ):
            return False
    return True


def retarget_edge_count_prior(
    table: OrderCandidateTable,
    *,
    target_label: PriorTarget,
    lambda_target: float,
    target_config: OrderSMCConfig,
    calibration: PriorCalibrationResult | None = None,
) -> tuple[OrderCandidateTable, PriorRetargetAudit]:
    """Copy one scored table and replace only its structural-prior potential.

    The target configuration must use the protocol-v2 pure edge-count prior and
    must otherwise match the source candidate-building configuration.  The
    returned provenance is updated to the target configuration, so the model's
    ordinary ``validate_candidate_provenance(..., config=target_config)`` check
    remains authoritative.  The source table is never mutated.
    """

    if target_label not in {"order", "structure"}:
        raise ValueError("target_label must be 'order' or 'structure'")
    if not np.isfinite(lambda_target) or lambda_target < 0.0:
        raise ValueError("lambda_target must be finite and non-negative")
    if not isinstance(target_config, OrderSMCConfig):
        raise TypeError("target_config must be an OrderSMCConfig")
    if float(target_config.lam) != 0.0:
        raise ValueError("target_config must use the protocol-v2 setting lam=0")
    if float(target_config.edge_penalty).hex() != float(lambda_target).hex():
        raise ValueError("target_config.edge_penalty must equal lambda_target exactly")
    if (
        target_config.tail_edge_penalty_after is not None
        or target_config.tail_edge_penalty is not None
    ):
        raise ValueError("target_config must not use a tail edge penalty")

    source_provenance = getattr(table, "provenance", None)
    if not isinstance(source_provenance, Mapping) or not source_provenance:
        raise ValueError("source candidate table must have populated provenance")
    source_build = source_provenance.get("candidate_build_config")
    if not isinstance(source_build, Mapping):
        raise ValueError("source provenance has no candidate-building configuration")
    target_build = candidate_build_config(target_config)
    prior_fields = {
        "lam",
        "edge_penalty",
        "tail_edge_penalty_after",
        "tail_edge_penalty",
    }
    incompatible = [
        name
        for name in sorted((set(source_build) | set(target_build)) - prior_fields)
        if source_build.get(name) != target_build.get(name)
    ]
    if incompatible:
        details = ", ".join(
            f"{name}: source={source_build.get(name)!r}, target={target_build.get(name)!r}"
            for name in incompatible
        )
        raise ValueError(
            "target_config changes candidate support or likelihood settings; " + details
        )

    source_support_sha256 = candidate_support_sha256(table)
    source_local_likelihood_sha256 = candidate_local_likelihood_sha256(table)
    calibration_sha256: str | None = None
    if calibration is not None:
        if not isinstance(calibration, PriorCalibrationResult):
            raise TypeError("calibration must be a PriorCalibrationResult")
        if calibration.target_label != target_label:
            raise ValueError("calibration target_label does not match target_label")
        if float(calibration.lambda_target).hex() != float(lambda_target).hex():
            raise ValueError("calibration lambda_target does not match lambda_target exactly")
        if calibration.candidate_support_sha256 != source_support_sha256:
            raise ValueError("calibration candidate support does not match the source table")
        calibration_sha256 = calibration.calibration_sha256

    parent_sets = [np.asarray(value).copy() for value in table.parent_sets]
    loglik = [np.asarray(value).copy() for value in table.loglik]
    logprior = [
        -float(lambda_target)
        * np.count_nonzero(value >= 0, axis=1).astype(np.float64)
        for value in parent_sets
    ]
    target_provenance = copy.deepcopy(dict(source_provenance))
    target_provenance["candidate_build_config"] = copy.deepcopy(target_build)
    target = OrderCandidateTable(
        nodes=list(table.nodes),
        parent_sets=parent_sets,
        loglik=loglik,
        logprior=logprior,
        provenance=target_provenance,
        build_diagnostics=copy.deepcopy(getattr(table, "build_diagnostics", {})),
    )

    target_support_sha256 = candidate_support_sha256(target)
    target_local_likelihood_sha256 = candidate_local_likelihood_sha256(target)
    bit_identical = _local_likelihood_arrays_bit_identical(table, target)
    if source_support_sha256 != target_support_sha256:
        raise RuntimeError("retargeting unexpectedly changed candidate support")
    if (
        source_local_likelihood_sha256 != target_local_likelihood_sha256
        or not bit_identical
    ):
        raise RuntimeError("retargeting unexpectedly changed local likelihood arrays")

    audit = PriorRetargetAudit(
        target_label=target_label,
        lambda_target=float(lambda_target),
        source_candidate_support_sha256=source_support_sha256,
        target_candidate_support_sha256=target_support_sha256,
        source_local_likelihood_sha256=source_local_likelihood_sha256,
        target_local_likelihood_sha256=target_local_likelihood_sha256,
        local_likelihood_arrays_bit_identical=bit_identical,
        source_candidate_build_config_sha256=_canonical_sha256(dict(source_build)),
        target_candidate_build_config_sha256=_canonical_sha256(target_build),
        calibration_sha256=calibration_sha256,
    )
    target.provenance["prior_retarget_audit"] = {
        **audit.to_dict(),
        "audit_sha256": audit.audit_sha256,
    }
    return target, audit


def edge_count_prior_table(
    table: OrderCandidateTable, lambda_target: float
) -> OrderCandidateTable:
    """Replace all data/prior terms by the protocol's ``-lambda*|P|`` potential."""

    if not np.isfinite(lambda_target):
        raise ValueError("lambda_target must be finite")
    parent_sets = [np.asarray(value, dtype=np.int32).copy() for value in table.parent_sets]
    logprior = [
        -float(lambda_target) * np.count_nonzero(value >= 0, axis=1).astype(float)
        for value in parent_sets
    ]
    return OrderCandidateTable(
        nodes=list(table.nodes),
        parent_sets=parent_sets,
        loglik=[np.zeros(len(value), dtype=float) for value in parent_sets],
        logprior=logprior,
        provenance={
            "schema": "ordersmc-protocol-v2-prior-calibration-v1",
            "structural_potential": "-lambda_target*parent_count",
            "lambda_target": float(lambda_target),
            "source_candidate_provenance": dict(table.provenance),
        },
    )


def exact_prior_mean_indegree(
    table: OrderCandidateTable,
    *,
    target_label: PriorTarget,
    lambda_target: float,
    max_nodes: int = 12,
) -> float:
    """Evaluate prior mean indegree under one declared target exactly."""

    prior_table = edge_count_prior_table(table, lambda_target)
    if target_label == "order":
        result = exact_order_dp(
            prior_table, beta=0.0, max_nodes=max_nodes, compute_graph_size=False
        )
    elif target_label == "structure":
        result = exact_structure_dp(
            prior_table, beta=0.0, max_nodes=max_nodes, compute_graph_size=False
        )
    else:
        raise ValueError("target_label must be 'order' or 'structure'")
    return float(sum(result.edge_probs.values()) / len(table.nodes))


def calibrate_exact_edge_penalty(
    table: OrderCandidateTable,
    *,
    target_label: PriorTarget,
    requested_mean_indegree: float = 2.0,
    lower: float = 0.0,
    upper: float = 20.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    max_nodes: int = 12,
) -> PriorCalibrationResult:
    """Use deterministic bisection to match a target-specific prior density."""

    node_count = len(table.nodes)
    if node_count < 1:
        raise ValueError("candidate table must contain at least one node")
    if not np.isfinite(requested_mean_indegree) or requested_mean_indegree < 0.0:
        raise ValueError("requested_mean_indegree must be finite and non-negative")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("lower and upper must be finite with lower < upper")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    initial_lower = float(lower)
    initial_upper = float(upper)

    lower_mean = exact_prior_mean_indegree(
        table,
        target_label=target_label,
        lambda_target=lower,
        max_nodes=max_nodes,
    )
    upper_mean = exact_prior_mean_indegree(
        table,
        target_label=target_label,
        lambda_target=upper,
        max_nodes=max_nodes,
    )
    if lower_mean < upper_mean - 1e-12:
        raise RuntimeError("prior mean indegree is not decreasing in lambda")
    if not upper_mean - tolerance <= requested_mean_indegree <= lower_mean + tolerance:
        raise ValueError(
            "requested mean indegree is outside the candidate support over the bracket"
        )

    midpoint = 0.5 * (lower + upper)
    achieved = exact_prior_mean_indegree(
        table,
        target_label=target_label,
        lambda_target=midpoint,
        max_nodes=max_nodes,
    )
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        midpoint = 0.5 * (lower + upper)
        achieved = exact_prior_mean_indegree(
            table,
            target_label=target_label,
            lambda_target=midpoint,
            max_nodes=max_nodes,
        )
        if abs(achieved - requested_mean_indegree) <= tolerance:
            break
        if achieved > requested_mean_indegree:
            lower = midpoint
        else:
            upper = midpoint
    else:
        raise RuntimeError("prior calibration did not meet tolerance")

    candidate_max_indegree = max(
        (
            int(np.count_nonzero(parent_set >= 0))
            for child_sets in table.parent_sets
            for parent_set in child_sets
        ),
        default=0,
    )
    return PriorCalibrationResult(
        target_label=target_label,
        lambda_target=float(midpoint),
        achieved_mean_indegree=float(achieved),
        requested_mean_indegree=float(requested_mean_indegree),
        absolute_error=abs(float(achieved - requested_mean_indegree)),
        iterations=iterations,
        bracket_lower=initial_lower,
        bracket_upper=initial_upper,
        final_bracket_lower=float(lower),
        final_bracket_upper=float(upper),
        tolerance=float(tolerance),
        maximum_iterations=int(max_iterations),
        max_nodes=int(max_nodes),
        method="exact deterministic bisection",
        node_count=node_count,
        candidate_max_indegree=candidate_max_indegree,
        candidate_support_sha256=candidate_support_sha256(table),
    )
