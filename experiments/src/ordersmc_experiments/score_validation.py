"""E0a: independent tiny-family checks of every native local score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import gammaln, logsumexp

from ordersmc.scoring import UnifiedFamilyScorer


@dataclass(frozen=True)
class ScoreValidationRecord:
    family: str
    implementation_log_score: float
    reference_log_score: float
    absolute_error: float
    tolerance: float
    passed: bool
    reference_method: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def manual_bdeu_log_marginal(
    child: Sequence[int],
    child_cardinality: int,
    parent_configuration: Sequence[int] | None,
    parent_configuration_count: int,
    *,
    alpha: float,
) -> float:
    y = np.asarray(child, dtype=int)
    contexts = (
        np.zeros(len(y), dtype=int)
        if parent_configuration is None
        else np.asarray(parent_configuration, dtype=int)
    )
    if contexts.shape != y.shape:
        raise ValueError("child and parent_configuration must align")
    q = int(parent_configuration_count)
    r = int(child_cardinality)
    counts = np.zeros((q, r), dtype=int)
    for context, value in zip(contexts, y):
        counts[int(context), int(value)] += 1
    alpha_context = float(alpha) / q
    alpha_cell = float(alpha) / (q * r)
    total = 0.0
    for context in range(q):
        count = int(np.sum(counts[context]))
        total += gammaln(alpha_context) - gammaln(alpha_context + count)
        total += float(
            np.sum(gammaln(alpha_cell + counts[context]) - gammaln(alpha_cell))
        )
    return float(total)


def independent_nig_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    *,
    tau: float,
    shape: float,
    scale: float,
) -> float:
    """Independent determinant/Schur-complement form of the NIG integral."""

    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    n, dimension = design.shape
    precision = design.T @ design + np.eye(dimension) / tau**2
    sign, logdet_precision = np.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("posterior precision is not positive definite")
    posterior_mean = np.linalg.solve(precision, design.T @ y)
    schur = float(y @ y - (design.T @ y) @ posterior_mean)
    schur = max(0.0, schur)
    posterior_shape = shape + n / 2.0
    posterior_scale = scale + schur / 2.0
    return float(
        -n / 2.0 * math.log(2.0 * math.pi)
        -0.5 * (logdet_precision + dimension * math.log(tau**2))
        +shape * math.log(scale)
        -posterior_shape * math.log(posterior_scale)
        +gammaln(posterior_shape)
        -gammaln(shape)
    )


def _normal_gauss_hermite_log_integral(
    log_likelihood,
    dimension: int,
    *,
    tau: float,
    quadrature_order: int,
) -> float:
    if dimension < 1 or dimension > 5:
        raise ValueError("Gauss-Hermite validation is limited to 1--5 dimensions")
    nodes, weights = hermgauss(quadrature_order)
    terms = []
    normalization = -0.5 * dimension * math.log(math.pi)
    for index_tuple in product(range(quadrature_order), repeat=dimension):
        coefficient = math.sqrt(2.0) * tau * nodes[list(index_tuple)]
        log_weight = float(np.sum(np.log(weights[list(index_tuple)]))) + normalization
        terms.append(log_weight + float(log_likelihood(coefficient)))
    return float(logsumexp(terms))


def gauss_hermite_logit_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    *,
    tau: float,
    quadrature_order: int = 30,
) -> float:
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)

    def log_likelihood(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    return _normal_gauss_hermite_log_integral(
        log_likelihood,
        design.shape[1],
        tau=tau,
        quadrature_order=quadrature_order,
    )


def _contrast_basis(cardinality: int) -> np.ndarray:
    basis = np.zeros((cardinality, cardinality - 1), dtype=float)
    for column in range(cardinality - 1):
        denominator = math.sqrt(float((column + 1) * (column + 2)))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def gauss_hermite_softmax_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    cardinality: int,
    *,
    tau: float,
    quadrature_order: int = 12,
) -> float:
    y = np.asarray(y, dtype=int)
    design = np.asarray(design, dtype=float)
    contrasts = cardinality - 1
    dimension = contrasts * design.shape[1]
    basis = _contrast_basis(cardinality)

    def log_likelihood(flat: np.ndarray) -> float:
        theta = flat.reshape(contrasts, design.shape[1])
        logits = (design @ theta.T) @ basis.T
        return float(np.sum(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1)))

    return _normal_gauss_hermite_log_integral(
        log_likelihood,
        dimension,
        tau=tau,
        quadrature_order=quadrature_order,
    )


def run_e0a_score_validation() -> list[ScoreValidationRecord]:
    """Run deterministic reference checks used by experiment E0a."""

    data = pd.DataFrame(
        {
            "disc_parent": np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=int),
            "binary": np.asarray([0, 1, 1, 1, 0, 0, 1, 0], dtype=int),
            "category": np.asarray([0, 1, 2, 1, 0, 2, 1, 0], dtype=int),
            "continuous_parent": np.asarray([-1.4, -0.9, -0.2, 0.1, 0.5, 0.8, 1.1, 1.7]),
            "continuous_child": np.asarray([-0.8, -0.1, 0.2, 0.6, 0.5, 1.2, 1.4, 2.0]),
        }
    )
    vtypes = {
        "disc_parent": "bin",
        "binary": "bin",
        "category": "cat",
        "continuous_parent": "con",
        "continuous_child": "con",
    }
    cards = {"disc_parent": 2, "binary": 2, "category": 3}
    alpha = 1.0
    tau = 1.0
    scorer = UnifiedFamilyScorer(
        data,
        vtypes,
        cards,
        alpha=alpha,
        tau=3.0,
        glm_tau=tau,
        clg_ig_shape=2.0,
        clg_ig_scale=1.0,
        standardize_continuous=True,
    )
    records: list[ScoreValidationRecord] = []

    def add(family: str, implementation: float, reference: float, tolerance: float, method: str):
        error = abs(float(implementation) - float(reference))
        records.append(
            ScoreValidationRecord(
                family=family,
                implementation_log_score=float(implementation),
                reference_log_score=float(reference),
                absolute_error=error,
                tolerance=float(tolerance),
                passed=error <= tolerance,
                reference_method=method,
            )
        )

    bdeu = scorer.score("binary", ["disc_parent"]).log_marginal
    manual_bdeu = manual_bdeu_log_marginal(
        data["binary"], 2, data["disc_parent"], 2, alpha=alpha
    )
    add("bdeu", bdeu, manual_bdeu, 1e-12, "independent count-table formula")

    clg = scorer.score("continuous_child", ["continuous_parent"]).log_marginal
    clg_design = np.column_stack(
        (np.ones(len(data)), scorer.Xf["continuous_parent"])
    )
    manual_clg = independent_nig_log_marginal(
        scorer.Xf["continuous_child"],
        clg_design,
        tau=3.0,
        shape=2.0,
        scale=1.0,
    )
    add("clg_nig", clg, manual_clg, 1e-10, "independent determinant NIG formula")

    logit = scorer.score("binary", ["continuous_parent"]).log_marginal
    logit_design = np.column_stack(
        (np.ones(len(data)), scorer.Xf["continuous_parent"])
    )
    exact_logit = gauss_hermite_logit_log_marginal(
        data["binary"].to_numpy(), logit_design, tau=tau, quadrature_order=30
    )
    add("binary_logit_laplace", logit, exact_logit, 0.05, "30-point tensor Gauss-Hermite")

    softmax = scorer.score("category", ["continuous_parent"]).log_marginal
    softmax_design = np.column_stack(
        (np.ones(len(data)), scorer.Xf["continuous_parent"])
    )
    exact_softmax = gauss_hermite_softmax_log_marginal(
        data["category"].to_numpy(),
        softmax_design,
        3,
        tau=tau,
        quadrature_order=12,
    )
    add(
        "multinomial_softmax_laplace",
        softmax,
        exact_softmax,
        0.15,
        "12-point four-dimensional tensor Gauss-Hermite",
    )
    return records


# Protocol-v2 acceptance implementation.  Keeping this import at the public
# module boundary preserves the original import path and the small compatibility
# helpers while moving the substantially larger acceptance suite to its own file.
from .score_validation_v2 import (  # noqa: E402,F401
    BranchBoundaryAudit,
    DerivativeAudit,
    ErrorPolicyRecord,
    FIXTURE_MATRIX_SCHEMA,
    FIXTURE_MATRIX_SHA256,
    FixtureMatrixAudit,
    ImportanceSamplingReference,
    QuadratureReference,
    ScoreValidationRecord,
    adaptive_finite_cube_logit_log_marginal,
    adaptive_softmax_log_marginal,
    audit_fixture_matrix,
    audit_logit_derivatives,
    audit_softmax_derivatives,
    canonical_sha256,
    default_fixture_matrix_path,
    gauss_hermite_logit_log_marginal,
    gauss_hermite_softmax_log_marginal,
    independent_nig_log_marginal,
    load_fixture_matrix,
    manual_bdeu_log_marginal,
    run_branch_boundary_audit,
    run_e0a_full_validation,
    run_e0a_score_validation,
    run_error_policy_checks,
    run_invariance_checks,
    run_structural_fixture_matrix,
    scalar_quadrature_nig_log_marginal,
    softmax_multivariate_t_importance_reference,
)
