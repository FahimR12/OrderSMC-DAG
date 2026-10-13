"""Protocol-v2 E0a acceptance checks with independent numerical references.

Failed Laplace fixtures remain failures, and a numerical reference that does
not meet its own error requirement is explicitly inconclusive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import cubature, quad
from scipy.special import gammaln, logsumexp

from ordersmc.scoring import UnifiedFamilyScorer


ValidationStatus = Literal["pass", "fail", "inconclusive"]
FIXTURE_MATRIX_SCHEMA = "ordersmc-e0a-fixture-matrix-v1"
FIXTURE_MATRIX_SHA256 = "e048f220757c0bc5278459b921eb3ce9cb3dcc5cc6770d16bd50ad9bc990e5ff"


@dataclass(frozen=True)
class ScoreValidationRecord:
    family: str
    implementation_log_score: float
    reference_log_score: float
    absolute_error: float
    tolerance: float
    passed: bool
    reference_method: str
    fixture_id: str = ""
    check: str = "score_accuracy"
    status: ValidationStatus = "pass"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = "pass" if self.passed else "fail"
        if self.status not in {"pass", "fail", "inconclusive"}:
            raise ValueError(f"invalid validation status {self.status!r}")
        if self.status != "inconclusive" and self.status != expected:
            raise ValueError("passed and status disagree")
        for name in (
            "implementation_log_score",
            "reference_log_score",
            "absolute_error",
            "tolerance",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.absolute_error < 0.0 or self.tolerance < 0.0:
            raise ValueError("error and tolerance must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuadratureReference:
    log_marginal: float
    reported_absolute_log_error: float
    status: str
    evaluations_or_subdivisions: int
    method: str
    refinement_history: tuple[tuple[int, float], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ImportanceSamplingReference:
    log_marginal: float
    effective_sample_size: float
    effective_sample_fraction: float
    log_mcse: float
    draws: int
    degrees_of_freedom: float
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DerivativeAudit:
    fixture_id: str
    family: str
    gradient_fd_max_error: float
    hessian_fd_max_error: float
    mode_gradient_inf_norm: float
    largest_log_posterior_hessian_eigenvalue: float
    optimizer_converged: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorPolicyRecord:
    fixture_id: str
    pathology: str
    expected_policy: str
    observed_policy: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FixtureMatrixAudit:
    matrix_sha256: str
    expanded_fixture_count: int
    resolved_fixture_hashes: Mapping[str, str]
    sample_sizes: tuple[int, ...]
    child_types: tuple[str, ...]
    parent_cells: tuple[tuple[str, int], ...]
    pathologies: tuple[str, ...]
    transformations: tuple[str, ...]
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BranchBoundaryAudit:
    fixture_id: str
    replicates: int
    seed: int
    finite_rate: float
    mean_log_bayes_factor_difference: float
    sd_log_bayes_factor_difference: float
    quantiles: Mapping[str, float]
    first_five_pairs: tuple[tuple[float, float], ...]
    bdeu_formula: str
    logit_formula: str
    duplicated_external_penalty: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _MaterializedFixture:
    fixture_id: str
    data: pd.DataFrame
    vtypes: Mapping[str, str]
    cards: Mapping[str, int]
    child: str
    parents: tuple[str, ...]
    expected_family: str
    data_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def default_fixture_matrix_path() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "e0a_fixture_matrix_v1.json"


def load_fixture_matrix(
    path: str | Path | None = None, *, verify_hash: bool = True
) -> dict[str, object]:
    source = default_fixture_matrix_path() if path is None else Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != FIXTURE_MATRIX_SCHEMA:
        raise ValueError(f"{source} is not a {FIXTURE_MATRIX_SCHEMA} document")
    observed = canonical_sha256(value)
    if verify_hash and observed != FIXTURE_MATRIX_SHA256:
        raise ValueError(
            f"fixture matrix hash mismatch: expected {FIXTURE_MATRIX_SHA256}, observed {observed}"
        )
    return value


def _column_payload(data: pd.DataFrame) -> dict[str, list[int | str]]:
    payload: dict[str, list[int | str]] = {}
    for name in sorted(data.columns):
        values = data[name].to_numpy()
        if values.dtype.kind in "iu":
            payload[name] = [int(value) for value in values]
        else:
            payload[name] = [float(value).hex() for value in values]
    return payload


def _base_static_data(n: int) -> pd.DataFrame:
    if n not in {8, 20, 100}:
        raise ValueError("static E0a matrix supports n in {8,20,100}")
    index = np.arange(n, dtype=int)
    d1 = index % 2
    d2 = (index // 2) % 3
    x1 = np.linspace(-1.7, 1.7, n) + 0.05 * ((index % 3) - 1)
    x2 = np.sin(2.0 * np.pi * (index + 0.25) / n) + 0.03 * ((index % 4) - 1.5)
    residual = np.asarray([-0.31, 0.12, 0.27, -0.08, 0.19, -0.24, 0.06, -0.15])
    return pd.DataFrame(
        {
            "d1": d1,
            "d2": d2,
            "x1": x1,
            "x2": x2,
            "yc": 0.45 * x1 - 0.30 * x2 + 0.20 * d1 + residual[index % 8],
            "yb": np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=int)[index % 8],
            "y3": np.asarray([0, 1, 2, 1, 0, 2, 1, 0], dtype=int)[index % 8],
            "y4": np.asarray([0, 1, 2, 3, 1, 0, 3, 2], dtype=int)[index % 8],
        }
    )


def _materialize_fixture(
    child_type: str, parent_block: str, parent_count: int, n: int
) -> _MaterializedFixture:
    data = _base_static_data(n)
    child_by_type = {
        "continuous": "yc",
        "binary": "yb",
        "categorical3": "y3",
        "categorical4": "y4",
    }
    parents_by_cell = {
        ("none", 0): (),
        ("discrete_only", 1): ("d1",),
        ("discrete_only", 2): ("d1", "d2"),
        ("continuous_only", 1): ("x1",),
        ("continuous_only", 2): ("x1", "x2"),
        ("mixed", 2): ("d1", "x1"),
    }
    try:
        child = child_by_type[child_type]
        parents = parents_by_cell[(parent_block, int(parent_count))]
    except KeyError as error:
        raise ValueError("unsupported fixture matrix cell") from error
    selected = data.loc[:, [*parents, child]].copy()
    vtypes = {
        name: (
            "con"
            if name in {"x1", "x2", "yc"}
            else "bin"
            if name in {"d1", "yb"}
            else "cat"
        )
        for name in selected.columns
    }
    cards = {
        name: (2 if name in {"d1", "yb"} else 3 if name in {"d2", "y3"} else 4)
        for name in selected.columns
        if vtypes[name] != "con"
    }
    has_continuous_parent = any(vtypes[parent] == "con" for parent in parents)
    if child_type == "continuous":
        expected_family = "clg"
    elif child_type == "binary" and has_continuous_parent:
        expected_family = "logit"
    elif child_type.startswith("categorical") and has_continuous_parent:
        expected_family = "softmax"
    else:
        expected_family = "disc_bdeu"
    fixture_id = f"{child_type}-{parent_block}-{parent_count}p-n{n}"
    return _MaterializedFixture(
        fixture_id=fixture_id,
        data=selected,
        vtypes=vtypes,
        cards=cards,
        child=child,
        parents=parents,
        expected_family=expected_family,
        data_sha256=canonical_sha256(_column_payload(selected)),
    )


def audit_fixture_matrix(path: str | Path | None = None) -> FixtureMatrixAudit:
    matrix = load_fixture_matrix(path)
    axes = matrix["matrix_axes"]
    if not isinstance(axes, dict):
        raise ValueError("matrix_axes must be an object")
    sample_sizes = tuple(int(value) for value in axes["sample_sizes"])
    child_types = tuple(str(value) for value in axes["child_types"])
    parent_cells = tuple(
        (str(value["parent_block"]), int(value["parent_count"]))
        for value in axes["parent_cells"]
    )
    hashes: dict[str, str] = {}
    for n in sample_sizes:
        for child_type in child_types:
            for block, count in parent_cells:
                fixture = _materialize_fixture(child_type, block, count, n)
                if fixture.fixture_id in hashes:
                    raise ValueError(f"duplicate expanded fixture {fixture.fixture_id}")
                hashes[fixture.fixture_id] = fixture.data_sha256
    pathologies = tuple(
        str(value["pathology"]) for value in matrix["pathology_fixtures"]
    )
    transformations = tuple(
        str(value["transformation_id"]) for value in matrix["transformations"]
    )
    complete = (
        set(sample_sizes) == {8, 20, 100}
        and set(child_types)
        == {"continuous", "binary", "categorical3", "categorical4"}
        and set(parent_cells)
        == {
            ("none", 0),
            ("discrete_only", 1),
            ("discrete_only", 2),
            ("continuous_only", 1),
            ("continuous_only", 2),
            ("mixed", 2),
        }
        and {
            "separation",
            "nearly_singular",
            "rare_category",
            "empty_category",
            "constant_predictor",
            "collinearity",
            "singular_dg_covariance",
            "nonfinite_input",
        }.issubset(pathologies)
        and {
            "row-reversal",
            "fixed-row-permutation",
            "parent-column-reversal",
            "category3-label-cycle",
            "category4-label-permutation",
            "softmax-contrast-rotation",
        }.issubset(transformations)
    )
    return FixtureMatrixAudit(
        matrix_sha256=canonical_sha256(matrix),
        expanded_fixture_count=len(hashes),
        resolved_fixture_hashes=hashes,
        sample_sizes=sample_sizes,
        child_types=child_types,
        parent_cells=parent_cells,
        pathologies=pathologies,
        transformations=transformations,
        complete=complete,
    )


def manual_bdeu_log_marginal(
    child: Sequence[int],
    child_cardinality: int,
    parent_configuration: Sequence[int] | None,
    parent_configuration_count: int,
    *,
    alpha: float,
) -> float:
    """Closed-form Dirichlet--multinomial calculation including all constants."""
    y = np.asarray(child, dtype=int)
    contexts = (
        np.zeros(len(y), dtype=int)
        if parent_configuration is None
        else np.asarray(parent_configuration, dtype=int)
    )
    if contexts.shape != y.shape:
        raise ValueError("child and parent_configuration must align")
    q, r = int(parent_configuration_count), int(child_cardinality)
    if q < 1 or r < 2 or alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("invalid BDeu dimensions or alpha")
    if y.size and (y.min() < 0 or y.max() >= r):
        raise ValueError("child value outside declared cardinality")
    if contexts.size and (contexts.min() < 0 or contexts.max() >= q):
        raise ValueError("parent context outside declared configuration count")
    counts = np.zeros((q, r), dtype=int)
    np.add.at(counts, (contexts, y), 1)
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
    y, design = np.asarray(y, dtype=float), np.asarray(design, dtype=float)
    n, dimension = design.shape
    precision = design.T @ design + np.eye(dimension) / tau**2
    sign, logdet_precision = np.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("posterior precision is not positive definite")
    posterior_mean = np.linalg.solve(precision, design.T @ y)
    schur = float(y @ y - (design.T @ y) @ posterior_mean)
    if schur < -1e-10 * max(1.0, float(y @ y)):
        raise ArithmeticError("negative Schur complement beyond roundoff")
    schur = max(0.0, schur)
    posterior_shape = shape + n / 2.0
    posterior_scale = scale + schur / 2.0
    return float(
        -n / 2.0 * math.log(2.0 * math.pi)
        - 0.5 * (logdet_precision + dimension * math.log(tau**2))
        + shape * math.log(scale)
        - posterior_shape * math.log(posterior_scale)
        + gammaln(posterior_shape)
        - gammaln(shape)
    )


def scalar_quadrature_nig_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    *,
    tau: float,
    shape: float,
    scale: float,
) -> QuadratureReference:
    """Integrate sigma^2 numerically after an independent Gaussian integral."""
    y, design = np.asarray(y, dtype=float), np.asarray(design, dtype=float)
    n = int(len(y))
    covariance_factor = np.eye(n) + tau**2 * (design @ design.T)
    sign, logdet = np.linalg.slogdet(covariance_factor)
    if sign <= 0:
        raise ValueError("integrated Gaussian covariance is not positive definite")
    quadratic = float(y @ np.linalg.solve(covariance_factor, y))
    posterior_shape = shape + 0.5 * n
    posterior_scale = scale + 0.5 * quadratic
    constant = (
        -0.5 * n * math.log(2.0 * math.pi)
        - 0.5 * logdet
        + shape * math.log(scale)
        - gammaln(shape)
    )
    log_variance_mode = math.log(posterior_scale / posterior_shape)

    def log_integrand(log_variance: float) -> float:
        if log_variance < -745.0:
            return -math.inf
        return float(
            constant
            - posterior_shape * log_variance
            - posterior_scale * math.exp(-log_variance)
        )

    shift = log_integrand(log_variance_mode)

    def scaled_integrand(log_variance: float) -> float:
        value = log_integrand(float(log_variance)) - shift
        return 0.0 if value < -745.0 else math.exp(value)

    estimate, error = quad(
        scaled_integrand,
        -math.inf,
        math.inf,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=400,
    )
    if estimate <= 0.0 or not math.isfinite(estimate):
        raise ArithmeticError("NIG scalar quadrature returned a non-positive integral")
    relative = float(error / estimate)
    return QuadratureReference(
        log_marginal=float(shift + math.log(estimate)),
        reported_absolute_log_error=float(math.log1p(relative)),
        status="converged" if relative < 1e-10 else "not_converged",
        evaluations_or_subdivisions=0,
        method="adaptive scalar quadrature over log(sigma^2) after Gaussian integration",
    )


def _logit_log_likelihood(beta: np.ndarray, y: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta = np.atleast_2d(np.asarray(beta, dtype=float))
    eta = beta @ np.asarray(design, dtype=float).T
    return eta @ np.asarray(y, dtype=float) - np.sum(np.logaddexp(0.0, eta), axis=1)


def adaptive_finite_cube_logit_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    *,
    tau: float,
    requested_log_error: float = 1e-8,
) -> QuadratureReference:
    """Integrate a <=3-coefficient logit model over its full finite-cube image."""
    y, design = np.asarray(y, dtype=float), np.asarray(design, dtype=float)
    dimension = int(design.shape[1])
    if dimension < 1 or dimension > 3:
        raise ValueError("finite-cube logit reference supports 1--3 coefficients")
    if design.shape[0] != len(y):
        raise ValueError("design and outcome do not align")
    shift = float(-len(y) * math.log(2.0))
    jacobian_constant = -0.5 * math.log(2.0 * math.pi) + math.log(math.pi / 2.0)

    def integrand(points: np.ndarray) -> np.ndarray:
        angles = 0.5 * math.pi * points
        standardized_beta = np.tan(angles)
        beta = tau * standardized_beta
        log_likelihood = _logit_log_likelihood(beta, y, design)
        log_prior_times_jacobian = np.sum(
            -0.5 * standardized_beta**2
            + jacobian_constant
            - 2.0 * np.log(np.cos(angles)),
            axis=1,
        )
        log_value = log_likelihood - shift + log_prior_times_jacobian
        if not np.isfinite(log_value).all():
            raise FloatingPointError("non-finite transformed logit cubature integrand")
        return np.exp(log_value)

    result = cubature(
        integrand,
        -np.ones(dimension),
        np.ones(dimension),
        rule="gk21",
        rtol=min(2e-10, requested_log_error / 5.0),
        atol=1e-13,
        max_subdivisions=10_000,
    )
    estimate, absolute_error = float(result.estimate), float(result.error)
    if estimate <= 0.0 or not math.isfinite(estimate + absolute_error):
        raise ArithmeticError("logit cubature returned an invalid integral")
    log_error = (
        math.inf
        if absolute_error >= estimate
        else max(
            math.log1p(absolute_error / estimate),
            -math.log1p(-absolute_error / estimate),
        )
    )
    return QuadratureReference(
        log_marginal=float(shift + math.log(estimate)),
        reported_absolute_log_error=float(log_error),
        status=str(result.status),
        evaluations_or_subdivisions=int(result.subdivisions),
        method="SciPy adaptive Gauss-Kronrod cubature after full-domain tangent transform",
    )


def gauss_hermite_logit_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    *,
    tau: float,
    quadrature_order: int = 30,
) -> float:
    """Compatibility diagnostic; protocol acceptance uses finite-cube cubature."""
    y, design = np.asarray(y, dtype=float), np.asarray(design, dtype=float)
    dimension = int(design.shape[1])
    nodes, weights = hermgauss(quadrature_order)
    indices = np.indices((quadrature_order,) * dimension).reshape(dimension, -1).T
    beta = math.sqrt(2.0) * tau * nodes[indices]
    log_weight = np.sum(np.log(weights[indices]), axis=1) - 0.5 * dimension * math.log(
        math.pi
    )
    return float(logsumexp(log_weight + _logit_log_likelihood(beta, y, design)))


def _contrast_basis(cardinality: int) -> np.ndarray:
    basis = np.zeros((cardinality, cardinality - 1), dtype=float)
    for column in range(cardinality - 1):
        denominator = math.sqrt(float((column + 1) * (column + 2)))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def _softmax_log_likelihood_batch(
    flat_coefficients: np.ndarray,
    y: np.ndarray,
    design: np.ndarray,
    cardinality: int,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    basis = _contrast_basis(cardinality) if basis is None else np.asarray(basis, dtype=float)
    contrasts = cardinality - 1
    theta = np.asarray(flat_coefficients, dtype=float).reshape(
        -1, contrasts, design.shape[1]
    )
    logits = np.einsum("nd,mrd,kr->mnk", design, theta, basis, optimize=True)
    rows = np.arange(len(y))
    return np.sum(logits[:, rows, y], axis=1) - np.sum(logsumexp(logits, axis=2), axis=1)


def _softmax_tensor_gh_order(
    y: np.ndarray,
    design: np.ndarray,
    cardinality: int,
    *,
    tau: float,
    order: int,
) -> float:
    dimension = int((cardinality - 1) * design.shape[1])
    if dimension > 4:
        raise ValueError("softmax cubature is restricted to at most four coefficients")
    nodes, weights = hermgauss(order)
    nodes = math.sqrt(2.0) * tau * nodes
    log_weights = np.log(weights)
    normalization = -0.5 * dimension * math.log(math.pi)
    if dimension < 4:
        indices = np.indices((order,) * dimension).reshape(dimension, -1).T
        values = nodes[indices]
        terms = np.sum(log_weights[indices], axis=1) + _softmax_log_likelihood_batch(
            values, y, design, cardinality
        )
        return float(normalization + logsumexp(terms))

    # Four dimensions are evaluated in O(order^2) bounded-memory blocks.
    tail_grid = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    tail_log_weights = (log_weights[:, None] + log_weights[None, :]).reshape(-1)
    block_integrals: list[float] = []
    for first in range(order):
        for second in range(order):
            values = np.column_stack(
                (
                    np.full(order * order, nodes[first]),
                    np.full(order * order, nodes[second]),
                    tail_grid,
                )
            )
            log_likelihood = _softmax_log_likelihood_batch(
                values, y, design, cardinality
            )
            block_integrals.append(
                float(
                    logsumexp(
                        log_likelihood
                        + log_weights[first]
                        + log_weights[second]
                        + tail_log_weights
                    )
                )
            )
    return float(normalization + logsumexp(block_integrals))


def adaptive_softmax_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    cardinality: int,
    *,
    tau: float,
    requested_log_error: float = 1e-8,
    orders: Sequence[int] = (32, 40, 48, 56, 64, 72),
) -> QuadratureReference:
    """Order-adaptive tensor Gauss--Hermite cubature on the full domain."""
    dimension = int((cardinality - 1) * design.shape[1])
    if dimension > 4:
        raise ValueError("softmax reference supports at most four coefficients")
    history: list[tuple[int, float]] = []
    previous: float | None = None
    reported_error = math.inf
    for order in orders:
        estimate = _softmax_tensor_gh_order(
            np.asarray(y, dtype=int),
            np.asarray(design, dtype=float),
            int(cardinality),
            tau=float(tau),
            order=int(order),
        )
        history.append((int(order), estimate))
        if previous is not None:
            reported_error = abs(estimate - previous)
            if reported_error < requested_log_error:
                return QuadratureReference(
                    log_marginal=estimate,
                    reported_absolute_log_error=reported_error,
                    status="converged",
                    evaluations_or_subdivisions=int(order**dimension),
                    method="order-adaptive tensor Gauss-Hermite cubature",
                    refinement_history=tuple(history),
                )
        previous = estimate
    return QuadratureReference(
        log_marginal=history[-1][1],
        reported_absolute_log_error=reported_error,
        status="not_converged",
        evaluations_or_subdivisions=int(history[-1][0] ** dimension),
        method="order-adaptive tensor Gauss-Hermite cubature",
        refinement_history=tuple(history),
    )


def gauss_hermite_softmax_log_marginal(
    y: np.ndarray,
    design: np.ndarray,
    cardinality: int,
    *,
    tau: float,
    quadrature_order: int = 12,
) -> float:
    """Fixed-order compatibility wrapper around the independent cubature."""
    return _softmax_tensor_gh_order(
        np.asarray(y, dtype=int),
        np.asarray(design, dtype=float),
        int(cardinality),
        tau=float(tau),
        order=int(quadrature_order),
    )


def softmax_multivariate_t_importance_reference(
    y: np.ndarray,
    design: np.ndarray,
    cardinality: int,
    *,
    tau: float,
    mode: np.ndarray,
    negative_log_posterior_hessian: np.ndarray,
    draws: int = 200_000,
    degrees_of_freedom: float = 7.0,
    seed: int = 92731,
) -> ImportanceSamplingReference:
    """Deterministic-batch fitted multivariate-t importance reference."""
    y, design = np.asarray(y, dtype=int), np.asarray(design, dtype=float)
    location = np.asarray(mode, dtype=float).reshape(-1)
    hessian = np.asarray(negative_log_posterior_hessian, dtype=float)
    dimension = len(location)
    covariance = np.linalg.inv(hessian)
    shape = covariance * (degrees_of_freedom - 2.0) / degrees_of_freedom
    chol = np.linalg.cholesky(shape)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    normal = rng.normal(size=(draws, dimension))
    chi_square = rng.chisquare(degrees_of_freedom, size=draws)
    coefficients = location + (normal @ chol.T) / np.sqrt(
        chi_square / degrees_of_freedom
    )[:, None]
    log_likelihood = _softmax_log_likelihood_batch(
        coefficients, y, design, cardinality
    )
    log_prior = -0.5 * (
        np.sum(coefficients**2, axis=1) / tau**2
        + dimension * math.log(2.0 * math.pi * tau**2)
    )
    whitened = np.linalg.solve(chol, (coefficients - location).T).T
    mahalanobis = np.sum(whitened**2, axis=1)
    log_proposal = (
        gammaln((degrees_of_freedom + dimension) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * dimension * math.log(degrees_of_freedom * math.pi)
        - np.sum(np.log(np.diag(chol)))
        - 0.5
        * (degrees_of_freedom + dimension)
        * np.log1p(mahalanobis / degrees_of_freedom)
    )
    log_weight = log_likelihood + log_prior - log_proposal
    log_marginal = float(logsumexp(log_weight) - math.log(draws))
    scaled_weight = np.exp(log_weight - np.max(log_weight))
    sum_weight = float(np.sum(scaled_weight))
    ess = float(sum_weight**2 / np.sum(scaled_weight**2))
    relative_mcse = float(
        math.sqrt(float(np.var(scaled_weight, ddof=1)) / draws)
        / float(np.mean(scaled_weight))
    )
    return ImportanceSamplingReference(
        log_marginal=log_marginal,
        effective_sample_size=ess,
        effective_sample_fraction=ess / draws,
        log_mcse=relative_mcse,
        draws=int(draws),
        degrees_of_freedom=float(degrees_of_freedom),
        seed=int(seed),
    )


def _central_gradient(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    gradient = np.empty_like(point)
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = step
        gradient[index] = (function(point + offset) - function(point - offset)) / (2.0 * step)
    return gradient


def _central_jacobian(
    function: Callable[[np.ndarray], np.ndarray], point: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    value = np.asarray(function(point), dtype=float)
    jacobian = np.empty((len(value), len(point)), dtype=float)
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = step
        jacobian[:, index] = (function(point + offset) - function(point - offset)) / (
            2.0 * step
        )
    return jacobian


def _logit_log_joint_derivatives(
    beta: np.ndarray, y: np.ndarray, design: np.ndarray, tau: float
) -> tuple[float, np.ndarray, np.ndarray]:
    beta = np.asarray(beta, dtype=float)
    eta = design @ beta
    probability = np.empty_like(eta)
    nonnegative = eta >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-eta[nonnegative]))
    exponential = np.exp(eta[~nonnegative])
    probability[~nonnegative] = exponential / (1.0 + exponential)
    precision = 1.0 / tau**2
    value = float(
        np.sum(y * eta - np.logaddexp(0.0, eta))
        - 0.5 * precision * np.dot(beta, beta)
        - 0.5 * len(beta) * math.log(2.0 * math.pi * tau**2)
    )
    gradient = design.T @ (y - probability) - precision * beta
    weights = probability * (1.0 - probability)
    hessian = -(design.T * weights) @ design - precision * np.eye(len(beta))
    return value, gradient, hessian


def audit_logit_derivatives(
    scorer: UnifiedFamilyScorer, child: str, parents: Sequence[str], fixture_id: str
) -> DerivativeAudit:
    continuous = sorted(parent for parent in parents if scorer.vtypes[parent] == "con")
    if any(scorer.vtypes[parent] != "con" for parent in parents):
        raise ValueError("derivative audit expects a single, unpartitioned context")
    y = scorer.Xi[child].astype(float)
    design = np.column_stack([np.ones(len(y)), *[scorer.Xf[name] for name in continuous]])
    mode, _, _ = scorer.logit._fit_map(y, design)
    value = lambda beta: _logit_log_joint_derivatives(
        beta, y, design, scorer.logit.tau
    )[0]
    gradient = lambda beta: _logit_log_joint_derivatives(
        beta, y, design, scorer.logit.tau
    )[1]
    _, analytic_gradient, analytic_hessian = _logit_log_joint_derivatives(
        mode, y, design, scorer.logit.tau
    )
    gradient_error = float(np.max(np.abs(_central_gradient(value, mode) - analytic_gradient)))
    hessian_error = float(
        np.max(np.abs(_central_jacobian(gradient, mode) - analytic_hessian))
    )
    largest_eigenvalue = float(np.max(np.linalg.eigvalsh(analytic_hessian)))
    mode_norm = float(np.linalg.norm(analytic_gradient, ord=np.inf))
    converged = mode_norm <= 1e-7 * (1.0 + len(y))
    return DerivativeAudit(
        fixture_id=fixture_id,
        family="logit",
        gradient_fd_max_error=gradient_error,
        hessian_fd_max_error=hessian_error,
        mode_gradient_inf_norm=mode_norm,
        largest_log_posterior_hessian_eigenvalue=largest_eigenvalue,
        optimizer_converged=converged,
        passed=(
            gradient_error < 1e-6
            and hessian_error < 1e-5
            and largest_eigenvalue < 0.0
            and converged
        ),
    )


def _softmax_log_joint_derivatives(
    flat: np.ndarray,
    y: np.ndarray,
    design: np.ndarray,
    tau: float,
    basis: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    flat = np.asarray(flat, dtype=float)
    contrasts = basis.shape[1]
    theta = flat.reshape(contrasts, design.shape[1])
    logits = (design @ theta.T) @ basis.T
    probability = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    indicators = np.zeros_like(probability)
    indicators[np.arange(len(y)), y] = 1.0
    precision = 1.0 / tau**2
    value = float(
        np.sum(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1))
        - 0.5 * precision * np.dot(flat, flat)
        - 0.5 * len(flat) * math.log(2.0 * math.pi * tau**2)
    )
    gradient = ((basis.T @ (indicators - probability).T) @ design).reshape(-1)
    gradient -= precision * flat
    projected = probability @ basis
    dimension = design.shape[1]
    negative_hessian = np.empty((len(flat), len(flat)))
    for first in range(contrasts):
        for second in range(contrasts):
            weight = (
                probability @ (basis[:, first] * basis[:, second])
                - projected[:, first] * projected[:, second]
            )
            block = (design.T * weight) @ design
            if first == second:
                block = block + precision * np.eye(dimension)
            negative_hessian[
                first * dimension : (first + 1) * dimension,
                second * dimension : (second + 1) * dimension,
            ] = block
    hessian = -0.5 * (negative_hessian + negative_hessian.T)
    return value, gradient, hessian


def audit_softmax_derivatives(
    scorer: UnifiedFamilyScorer, child: str, parents: Sequence[str], fixture_id: str
) -> DerivativeAudit:
    continuous = sorted(parent for parent in parents if scorer.vtypes[parent] == "con")
    if any(scorer.vtypes[parent] != "con" for parent in parents):
        raise ValueError("derivative audit expects a single, unpartitioned context")
    y = scorer.Xi[child].astype(int)
    design = np.column_stack([np.ones(len(y)), *[scorer.Xf[name] for name in continuous]])
    cardinality = int(scorer.cards[child])
    basis = _contrast_basis(cardinality)
    mode, _, _ = scorer.softmax._fit_map(y.astype(float), design, cardinality)
    flat_mode = mode.reshape(-1)
    value = lambda flat: _softmax_log_joint_derivatives(
        flat, y, design, scorer.softmax.tau, basis
    )[0]
    gradient = lambda flat: _softmax_log_joint_derivatives(
        flat, y, design, scorer.softmax.tau, basis
    )[1]
    _, analytic_gradient, analytic_hessian = _softmax_log_joint_derivatives(
        flat_mode, y, design, scorer.softmax.tau, basis
    )
    gradient_error = float(
        np.max(np.abs(_central_gradient(value, flat_mode) - analytic_gradient))
    )
    hessian_error = float(
        np.max(np.abs(_central_jacobian(gradient, flat_mode) - analytic_hessian))
    )
    largest_eigenvalue = float(np.max(np.linalg.eigvalsh(analytic_hessian)))
    mode_norm = float(np.linalg.norm(analytic_gradient, ord=np.inf))
    converged = mode_norm <= 1e-7 * (1.0 + len(y))
    return DerivativeAudit(
        fixture_id=fixture_id,
        family="softmax",
        gradient_fd_max_error=gradient_error,
        hessian_fd_max_error=hessian_error,
        mode_gradient_inf_norm=mode_norm,
        largest_log_posterior_hessian_eigenvalue=largest_eigenvalue,
        optimizer_converged=converged,
        passed=(
            gradient_error < 1e-6
            and hessian_error < 1e-5
            and largest_eigenvalue < 0.0
            and converged
        ),
    )


def run_structural_fixture_matrix() -> list[ScoreValidationRecord]:
    """Resolve and execute every one of the 72 branch/sample-size cells."""
    audit = audit_fixture_matrix()
    records: list[ScoreValidationRecord] = []
    for n in audit.sample_sizes:
        for child_type in audit.child_types:
            for block, count in audit.parent_cells:
                fixture = _materialize_fixture(child_type, block, count, n)
                scorer = UnifiedFamilyScorer(
                    fixture.data,
                    fixture.vtypes,
                    fixture.cards,
                    alpha=1.0,
                    tau=3.0,
                    glm_tau=1.0,
                    clg_ig_shape=2.0,
                    clg_ig_scale=1.0,
                    standardize_continuous=True,
                )
                result = scorer.score(fixture.child, fixture.parents)
                finite = math.isfinite(result.log_marginal)
                passed = finite and result.family == fixture.expected_family
                records.append(
                    ScoreValidationRecord(
                        family=result.family,
                        implementation_log_score=float(result.log_marginal),
                        reference_log_score=float(result.log_marginal),
                        absolute_error=0.0,
                        tolerance=0.0,
                        passed=passed,
                        reference_method="declared branch-dispatch and finite-score contract",
                        fixture_id=fixture.fixture_id,
                        check="branch_dispatch_and_finite_policy",
                        status="pass" if passed else "fail",
                        diagnostics={
                            "expected_family": fixture.expected_family,
                            "data_sha256": fixture.data_sha256,
                            "n": n,
                            "child_type": child_type,
                            "parent_block": block,
                            "parent_count": count,
                        },
                    )
                )
    return records


def _legacy_data() -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    data = pd.DataFrame(
        {
            "disc_parent": np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=int),
            "binary": np.asarray([0, 1, 1, 1, 0, 0, 1, 0], dtype=int),
            "category": np.asarray([0, 1, 2, 1, 0, 2, 1, 0], dtype=int),
            "continuous_parent": np.asarray(
                [-1.4, -0.9, -0.2, 0.1, 0.5, 0.8, 1.1, 1.7]
            ),
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
    return data, vtypes, cards


def _score_record(
    family: str,
    fixture_id: str,
    implementation: float,
    reference: float,
    tolerance: float,
    method: str,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    check: str = "score_accuracy",
    conclusive: bool = True,
) -> ScoreValidationRecord:
    error = abs(float(implementation) - float(reference))
    passed = conclusive and error <= tolerance
    status: ValidationStatus = "pass" if passed else "fail" if conclusive else "inconclusive"
    return ScoreValidationRecord(
        family=family,
        implementation_log_score=float(implementation),
        reference_log_score=float(reference),
        absolute_error=error,
        tolerance=float(tolerance),
        passed=passed,
        reference_method=method,
        fixture_id=fixture_id,
        check=check,
        status=status,
        diagnostics={} if diagnostics is None else diagnostics,
    )


@lru_cache(maxsize=1)
def _reference_anchor_bundle() -> tuple[
    tuple[ScoreValidationRecord, ...],
    tuple[DerivativeAudit, ...],
    Mapping[str, Any],
]:
    data, vtypes, cards = _legacy_data()
    scorer = UnifiedFamilyScorer(
        data,
        vtypes,
        cards,
        alpha=1.0,
        tau=3.0,
        glm_tau=1.0,
        clg_ig_shape=2.0,
        clg_ig_scale=1.0,
        standardize_continuous=True,
    )
    records: list[ScoreValidationRecord] = []

    implementation = scorer.score("binary", ["disc_parent"]).log_marginal
    reference = manual_bdeu_log_marginal(
        data["binary"], 2, data["disc_parent"], 2, alpha=1.0
    )
    records.append(
        _score_record(
            "bdeu",
            "bdeu-binary-one-discrete-n8",
            implementation,
            reference,
            1e-10,
            "independent integer count-table Dirichlet-multinomial formula",
        )
    )

    fixture = _materialize_fixture("categorical3", "discrete_only", 2, 20)
    fixture_scorer = UnifiedFamilyScorer(
        fixture.data, fixture.vtypes, fixture.cards, alpha=1.0
    )
    contexts = fixture.data["d1"].to_numpy() + 2 * fixture.data["d2"].to_numpy()
    records.append(
        _score_record(
            "bdeu_cat3_two_parent",
            "bdeu-cat3-two-discrete-n20",
            fixture_scorer.score("y3", ["d1", "d2"]).log_marginal,
            manual_bdeu_log_marginal(
                fixture.data["y3"], 3, contexts, 6, alpha=1.0
            ),
            1e-10,
            "independent six-context integer count-table formula",
        )
    )

    cat4_fixture = _materialize_fixture("categorical4", "discrete_only", 2, 100)
    cat4_scorer = UnifiedFamilyScorer(
        cat4_fixture.data, cat4_fixture.vtypes, cat4_fixture.cards, alpha=1.0
    )
    cat4_contexts = (
        cat4_fixture.data["d1"].to_numpy()
        + 2 * cat4_fixture.data["d2"].to_numpy()
    )
    records.append(
        _score_record(
            "bdeu_cat4_two_parent",
            "bdeu-cat4-two-discrete-n100",
            cat4_scorer.score("y4", ["d1", "d2"]).log_marginal,
            manual_bdeu_log_marginal(
                cat4_fixture.data["y4"], 4, cat4_contexts, 6, alpha=1.0
            ),
            1e-10,
            "independent six-context four-cell Dirichlet-multinomial formula",
        )
    )

    clg_design = np.column_stack(
        (np.ones(len(data)), scorer.Xf["continuous_parent"])
    )
    clg_y = scorer.Xf["continuous_child"]
    clg_implementation = scorer.score(
        "continuous_child", ["continuous_parent"]
    ).log_marginal
    clg_reference = independent_nig_log_marginal(
        clg_y, clg_design, tau=3.0, shape=2.0, scale=1.0
    )
    records.append(
        _score_record(
            "clg_nig",
            "nig-continuous-one-continuous-n8",
            clg_implementation,
            clg_reference,
            1e-10,
            "independent determinant and Schur-complement NIG formula",
        )
    )

    no_parent_fixture = _materialize_fixture("continuous", "none", 0, 8)
    no_parent_scorer = UnifiedFamilyScorer(
        no_parent_fixture.data,
        no_parent_fixture.vtypes,
        no_parent_fixture.cards,
        tau=3.0,
        clg_ig_shape=2.0,
        clg_ig_scale=1.0,
    )
    no_parent_y = no_parent_scorer.Xf["yc"]
    no_parent_design = np.ones((len(no_parent_y), 1))
    records.append(
        _score_record(
            "clg_nig_no_parent",
            "nig-continuous-no-parent-n8",
            no_parent_scorer.score("yc", []).log_marginal,
            independent_nig_log_marginal(
                no_parent_y,
                no_parent_design,
                tau=3.0,
                shape=2.0,
                scale=1.0,
            ),
            1e-10,
            "independent intercept-only determinant and Schur-complement NIG formula",
        )
    )

    mixed_fixture = _materialize_fixture("continuous", "mixed", 2, 20)
    mixed_scorer = UnifiedFamilyScorer(
        mixed_fixture.data,
        mixed_fixture.vtypes,
        mixed_fixture.cards,
        tau=3.0,
        clg_ig_shape=2.0,
        clg_ig_scale=1.0,
    )
    mixed_reference = 0.0
    for context in (0, 1):
        indices = np.flatnonzero(mixed_scorer.Xi["d1"] == context)
        context_design = np.column_stack(
            (np.ones(len(indices)), mixed_scorer.Xf["x1"][indices])
        )
        mixed_reference += independent_nig_log_marginal(
            mixed_scorer.Xf["yc"][indices],
            context_design,
            tau=3.0,
            shape=2.0,
            scale=1.0,
        )
    records.append(
        _score_record(
            "clg_nig_mixed_contexts",
            "nig-continuous-mixed-n20",
            mixed_scorer.score("yc", ["d1", "x1"]).log_marginal,
            mixed_reference,
            1e-10,
            "sum of two independently indexed determinant/Schur NIG context integrals",
        )
    )

    clg_bic_scorer = UnifiedFamilyScorer(
        mixed_fixture.data,
        mixed_fixture.vtypes,
        mixed_fixture.cards,
        score_method="clg-bic",
        standardize_continuous=True,
    )
    clg_bic_log_likelihood = 0.0
    for context in (0, 1):
        indices = np.flatnonzero(clg_bic_scorer.Xi["d1"] == context)
        design = np.column_stack(
            (np.ones(len(indices)), clg_bic_scorer.Xf["x1"][indices])
        )
        response = clg_bic_scorer.Xf["yc"][indices]
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        rss = float(np.sum((response - design @ coefficients) ** 2))
        variance = rss / len(indices)
        clg_bic_log_likelihood += -0.5 * len(indices) * (
            math.log(2.0 * math.pi * variance) + 1.0
        )
    clg_bic_reference = clg_bic_log_likelihood - 0.5 * 6 * math.log(
        len(mixed_fixture.data)
    )
    records.append(
        _score_record(
            "clg_bic",
            "clg-bic-continuous-mixed-n20",
            clg_bic_scorer.score("yc", ["d1", "x1"]).log_marginal,
            clg_bic_reference,
            1e-10,
            "independent context-wise OLS maximum-likelihood and BIC formula",
            diagnostics={
                "continuous_parents": 1,
                "discrete_contexts": 2,
                "parameter_count": 6,
                "variance_denominator": "context_size",
            },
        )
    )

    discrete_bic_scorer = UnifiedFamilyScorer(
        data,
        vtypes,
        cards,
        score_method="clg-bic",
    )
    binary = data["binary"].to_numpy(dtype=int)
    discrete_context = data["disc_parent"].to_numpy(dtype=int)
    multinomial_log_likelihood = 0.0
    for context in (0, 1):
        selected = binary[discrete_context == context]
        counts = np.bincount(selected, minlength=2)
        positive = counts > 0
        multinomial_log_likelihood += float(
            np.sum(counts[positive] * np.log(counts[positive] / len(selected)))
        )
    discrete_bic_reference = multinomial_log_likelihood - 0.5 * 2 * math.log(
        len(data)
    )
    records.append(
        _score_record(
            "clg_bic",
            "clg-bic-binary-one-discrete-n8",
            discrete_bic_scorer.score("binary", ["disc_parent"]).log_marginal,
            discrete_bic_reference,
            1e-12,
            "independent multinomial maximum-likelihood and BIC formula",
            diagnostics={"parent_contexts": 2, "parameter_count": 2},
        )
    )
    scalar_nig = scalar_quadrature_nig_log_marginal(
        clg_y, clg_design, tau=3.0, shape=2.0, scale=1.0
    )
    records.append(
        _score_record(
            "clg_nig_scalar_quadrature",
            "nig-continuous-one-continuous-n8",
            clg_implementation,
            scalar_nig.log_marginal,
            1e-10,
            scalar_nig.method,
            diagnostics={
                "quadrature_status": scalar_nig.status,
                "reported_absolute_log_error": scalar_nig.reported_absolute_log_error,
            },
            conclusive=(
                scalar_nig.status == "converged"
                and scalar_nig.reported_absolute_log_error < 1e-10
            ),
        )
    )

    logit_design = np.column_stack((np.ones(len(data)), scorer.Xf["continuous_parent"]))
    logit_reference = adaptive_finite_cube_logit_log_marginal(
        data["binary"].to_numpy(), logit_design, tau=1.0
    )
    records.append(
        _score_record(
            "binary_logit_laplace",
            "logit-binary-one-continuous-n8",
            scorer.score("binary", ["continuous_parent"]).log_marginal,
            logit_reference.log_marginal,
            0.10,
            logit_reference.method,
            diagnostics={
                "quadrature_status": logit_reference.status,
                "reported_absolute_log_error": logit_reference.reported_absolute_log_error,
                "subdivisions": logit_reference.evaluations_or_subdivisions,
                "conditioning_label_frozen_in_fixture": "well_conditioned",
            },
            conclusive=(
                logit_reference.status == "converged"
                and logit_reference.reported_absolute_log_error < 1e-8
            ),
        )
    )

    logit_three_fixture = _materialize_fixture("binary", "continuous_only", 2, 20)
    logit_three_scorer = UnifiedFamilyScorer(
        logit_three_fixture.data,
        logit_three_fixture.vtypes,
        logit_three_fixture.cards,
        glm_tau=1.0,
    )
    logit_three_design = np.column_stack(
        (
            np.ones(20),
            logit_three_scorer.Xf["x1"],
            logit_three_scorer.Xf["x2"],
        )
    )
    logit_three_reference = adaptive_finite_cube_logit_log_marginal(
        logit_three_scorer.Xi["yb"], logit_three_design, tau=1.0
    )
    records.append(
        _score_record(
            "binary_logit_3coef_laplace",
            "logit-binary-two-continuous-n20",
            logit_three_scorer.score("yb", ["x1", "x2"]).log_marginal,
            logit_three_reference.log_marginal,
            0.10,
            logit_three_reference.method,
            diagnostics={
                "quadrature_status": logit_three_reference.status,
                "reported_absolute_log_error": logit_three_reference.reported_absolute_log_error,
                "subdivisions": logit_three_reference.evaluations_or_subdivisions,
                "free_coefficients": 3,
                "conditioning_label_frozen_in_fixture": "well_conditioned",
            },
            conclusive=(
                logit_three_reference.status == "converged"
                and logit_three_reference.reported_absolute_log_error < 1e-8
            ),
        )
    )

    softmax_design = np.column_stack(
        (np.ones(len(data)), scorer.Xf["continuous_parent"])
    )
    softmax_y = data["category"].to_numpy(dtype=int)
    softmax_reference = adaptive_softmax_log_marginal(
        softmax_y, softmax_design, 3, tau=1.0
    )
    mode, _, _ = scorer.softmax._fit_map(softmax_y.astype(float), softmax_design, 3)
    basis = _contrast_basis(3)
    _, _, log_hessian = _softmax_log_joint_derivatives(
        mode.reshape(-1), softmax_y, softmax_design, 1.0, basis
    )
    importance = softmax_multivariate_t_importance_reference(
        softmax_y,
        softmax_design,
        3,
        tau=1.0,
        mode=mode,
        negative_log_posterior_hessian=-log_hessian,
    )
    references_agree = (
        softmax_reference.status == "converged"
        and softmax_reference.reported_absolute_log_error < 1e-8
        and importance.log_mcse < 0.01
        and abs(softmax_reference.log_marginal - importance.log_marginal) <= 0.03
    )
    records.append(
        _score_record(
            "multinomial_softmax_laplace",
            "softmax-cat3-one-continuous-n8",
            scorer.score("category", ["continuous_parent"]).log_marginal,
            softmax_reference.log_marginal,
            0.20,
            "adaptive Gauss-Hermite cubature cross-checked by multivariate-t importance sampling",
            diagnostics={
                "quadrature_status": softmax_reference.status,
                "reported_absolute_log_error": softmax_reference.reported_absolute_log_error,
                "quadrature_history": softmax_reference.refinement_history,
                "importance_log_marginal": importance.log_marginal,
                "reference_disagreement": abs(
                    softmax_reference.log_marginal - importance.log_marginal
                ),
                "importance_draws": importance.draws,
                "importance_effective_sample_size": importance.effective_sample_size,
                "importance_log_mcse": importance.log_mcse,
                "conditioning_label_frozen_in_fixture": "difficult_small_n",
            },
            conclusive=references_agree,
        )
    )
    records.append(
        _score_record(
            "softmax_reference_agreement",
            "softmax-cat3-one-continuous-n8",
            softmax_reference.log_marginal,
            importance.log_marginal,
            0.03,
            "independent adaptive cubature versus fitted multivariate-t importance sampling",
            diagnostics=importance.to_dict(),
            check="independent_reference_agreement",
            conclusive=(
                softmax_reference.status == "converged"
                and softmax_reference.reported_absolute_log_error < 1e-8
                and importance.log_mcse < 0.01
            ),
        )
    )
    derivatives = (
        audit_logit_derivatives(
            scorer,
            "binary",
            ["continuous_parent"],
            "logit-binary-one-continuous-n8",
        ),
        audit_softmax_derivatives(
            scorer,
            "category",
            ["continuous_parent"],
            "softmax-cat3-one-continuous-n8",
        ),
        audit_logit_derivatives(
            logit_three_scorer,
            "yb",
            ["x1", "x2"],
            "logit-binary-two-continuous-n20",
        ),
    )
    references = {
        "logit_finite_cube": logit_reference.to_dict(),
        "logit_three_coefficient_finite_cube": logit_three_reference.to_dict(),
        "softmax_adaptive_cubature": softmax_reference.to_dict(),
        "softmax_importance_sampling": importance.to_dict(),
        "nig_scalar_quadrature": scalar_nig.to_dict(),
    }
    return tuple(records), derivatives, references


def run_invariance_checks() -> list[ScoreValidationRecord]:
    data, vtypes, cards = _legacy_data()
    base = UnifiedFamilyScorer(data, vtypes, cards, tau=3.0, glm_tau=1.0)
    permuted_rows = np.r_[np.arange(0, len(data), 2), np.arange(1, len(data), 2)]
    transformed_scorers = {
        "row-reversal": UnifiedFamilyScorer(
            data.iloc[::-1].reset_index(drop=True), vtypes, cards, tau=3.0, glm_tau=1.0
        ),
        "fixed-row-permutation": UnifiedFamilyScorer(
            data.iloc[permuted_rows].reset_index(drop=True),
            vtypes,
            cards,
            tau=3.0,
            glm_tau=1.0,
        ),
    }
    records: list[ScoreValidationRecord] = []
    for label, child, parents in (
        ("bdeu", "binary", ["disc_parent"]),
        ("clg", "continuous_child", ["continuous_parent"]),
        ("logit", "binary", ["continuous_parent"]),
        ("softmax", "category", ["continuous_parent"]),
    ):
        original = base.score(child, parents).log_marginal
        for transformation, transformed in transformed_scorers.items():
            records.append(
                _score_record(
                    f"{label}_{transformation}",
                    f"{label}-{transformation}",
                    original,
                    transformed.score(child, parents).log_marginal,
                    1e-9,
                    "same score after a fixed static row transformation",
                    check="row_order_invariance",
                )
            )

    two_parent = _materialize_fixture("binary", "continuous_only", 2, 20)
    two_scorer = UnifiedFamilyScorer(
        two_parent.data, two_parent.vtypes, two_parent.cards, glm_tau=1.0
    )
    records.append(
        _score_record(
            "logit_parent_column_reversal",
            "logit-binary-two-continuous-n20",
            two_scorer.score(two_parent.child, ["x1", "x2"]).log_marginal,
            two_scorer.score(two_parent.child, ["x2", "x1"]).log_marginal,
            1e-12,
            "same family after parent-list reversal",
            check="column_order_invariance",
        )
    )

    for cardinality, mapping in (
        (3, np.asarray([1, 2, 0], dtype=int)),
        (4, np.asarray([2, 0, 3, 1], dtype=int)),
    ):
        fixture = _materialize_fixture(
            f"categorical{cardinality}", "continuous_only", 1, 20
        )
        child = fixture.child
        relabelled = fixture.data.copy()
        relabelled[child] = mapping[relabelled[child].to_numpy(dtype=int)]
        first = UnifiedFamilyScorer(
            fixture.data, fixture.vtypes, fixture.cards, glm_tau=1.0
        ).score(child, fixture.parents)
        second = UnifiedFamilyScorer(
            relabelled, fixture.vtypes, fixture.cards, glm_tau=1.0
        ).score(child, fixture.parents)
        records.append(
            _score_record(
                f"softmax_category{cardinality}_label_permutation",
                f"softmax-category{cardinality}-label-permutation-n20",
                first.log_marginal,
                second.log_marginal,
                1e-8,
                "isotropic Helmert-coordinate score after fixed category relabelling",
                check="category_label_invariance",
            )
        )

    y = base.Xi["category"].astype(int)
    design = np.column_stack((np.ones(len(data)), base.Xf["continuous_parent"]))
    basis = _contrast_basis(3)
    rotation = np.asarray([[0.6, -0.8], [0.8, 0.6]])
    rotated_basis = basis @ rotation
    mode, _, _ = base.softmax._fit_map(y.astype(float), design, 3)
    rotated_mode = rotation.T @ mode
    value, _, hessian = _softmax_log_joint_derivatives(
        mode.reshape(-1), y, design, 1.0, basis
    )
    rotated_value, _, rotated_hessian = _softmax_log_joint_derivatives(
        rotated_mode.reshape(-1), y, design, 1.0, rotated_basis
    )
    dimension = mode.size
    log_score = value - 0.5 * np.linalg.slogdet(-hessian)[1] + 0.5 * dimension * math.log(
        2.0 * math.pi
    )
    rotated_log_score = (
        rotated_value
        - 0.5 * np.linalg.slogdet(-rotated_hessian)[1]
        + 0.5 * dimension * math.log(2.0 * math.pi)
    )
    records.append(
        _score_record(
            "softmax_orthogonal_contrast_rotation",
            "softmax-cat3-contrast-rotation-n8",
            log_score,
            rotated_log_score,
            1e-10,
            "independently evaluated Laplace formula in two orthogonal contrast bases",
            check="contrast_basis_invariance",
        )
    )
    return records


def run_error_policy_checks() -> list[ErrorPolicyRecord]:
    records: list[ErrorPolicyRecord] = []
    context = np.repeat([0, 1], 10)
    x = np.linspace(-2.0, 2.0, 20)
    separation = pd.DataFrame({"d": context, "x": x, "y": context.copy()})
    value = UnifiedFamilyScorer(
        separation,
        {"d": "bin", "x": "con", "y": "bin"},
        {"d": 2, "y": 2},
    ).score("y", ["d", "x"]).log_marginal
    records.append(
        ErrorPolicyRecord(
            "binary-separated-n20",
            "separation",
            "finite_proper_prior_score",
            "finite_score" if math.isfinite(value) else "nonfinite_score",
            math.isfinite(value),
            f"log_score={value:.17g}",
        )
    )

    nearly = pd.DataFrame(
        {
            "x1": x,
            "x2": x + 1e-10 * np.asarray([(-1) ** i for i in range(20)]),
            "y": 0.4 * x + np.asarray([0.1, -0.1] * 10),
        }
    )
    value = UnifiedFamilyScorer(
        nearly, {"x1": "con", "x2": "con", "y": "con"}, {}
    ).score("y", ["x1", "x2"]).log_marginal
    records.append(
        ErrorPolicyRecord(
            "continuous-nearly-singular-n20",
            "nearly_singular",
            "finite_proper_prior_score",
            "finite_score" if math.isfinite(value) else "nonfinite_score",
            math.isfinite(value),
            f"log_score={value:.17g}",
        )
    )

    rare = _base_static_data(100).loc[:, ["x1", "y4"]].copy()
    rare["y4"] = 0
    rare.loc[[24, 49], "y4"] = 1
    rare.loc[[74], "y4"] = 2
    rare.loc[[99], "y4"] = 3
    value = UnifiedFamilyScorer(
        rare, {"x1": "con", "y4": "cat"}, {"y4": 4}
    ).score("y4", ["x1"]).log_marginal
    records.append(
        ErrorPolicyRecord(
            "categorical4-rare-category-n100",
            "rare_category",
            "finite_proper_prior_score",
            "finite_score" if math.isfinite(value) else "nonfinite_score",
            math.isfinite(value),
            f"log_score={value:.17g}",
        )
    )

    empty = _base_static_data(20).loc[:, ["x1", "y4"]].copy()
    empty["y4"] = empty["y4"].replace({3: 2})
    value = UnifiedFamilyScorer(
        empty, {"x1": "con", "y4": "cat"}, {"y4": 4}
    ).score("y4", ["x1"]).log_marginal
    records.append(
        ErrorPolicyRecord(
            "categorical4-empty-category-n20",
            "empty_category",
            "finite_proper_prior_score",
            "finite_score" if math.isfinite(value) else "nonfinite_score",
            math.isfinite(value),
            f"log_score={value:.17g}",
        )
    )

    def expect_value_error(
        fixture_id: str,
        pathology: str,
        action: Callable[[], object],
        message_fragment: str,
    ) -> None:
        try:
            action()
        except ValueError as error:
            matched = message_fragment in str(error)
            records.append(
                ErrorPolicyRecord(
                    fixture_id,
                    pathology,
                    "ValueError",
                    "ValueError",
                    matched,
                    str(error),
                )
            )
        else:
            records.append(
                ErrorPolicyRecord(
                    fixture_id,
                    pathology,
                    "ValueError",
                    "ordinary_score_or_no_error",
                    False,
                    "the invalid fixture did not raise",
                )
            )

    expect_value_error(
        "continuous-constant-predictor-n8",
        "constant_predictor",
        lambda: UnifiedFamilyScorer(
            pd.DataFrame({"x": np.ones(8), "y": np.arange(8) % 2}),
            {"x": "con", "y": "bin"},
            {"y": 2},
        ),
        "zero variance",
    )

    collinear = pd.DataFrame({"x1": x, "x2": 2.0 * x, "y": np.arange(20) % 2})
    value = UnifiedFamilyScorer(
        collinear,
        {"x1": "con", "x2": "con", "y": "bin"},
        {"y": 2},
    ).score("y", ["x1", "x2"]).log_marginal
    records.append(
        ErrorPolicyRecord(
            "continuous-collinear-parents-n20",
            "collinearity",
            "finite_proper_prior_score",
            "finite_score" if math.isfinite(value) else "nonfinite_score",
            math.isfinite(value),
            f"log_score={value:.17g}",
        )
    )

    expect_value_error(
        "dg-singular-covariance-n20",
        "singular_dg_covariance",
        lambda: UnifiedFamilyScorer(
            pd.DataFrame({"x": x, "duplicate": x.copy()}),
            {"x": "con", "duplicate": "con"},
            {},
            score_method="dg",
        ).score("duplicate", ["x"]),
        "DG covariance is singular",
    )
    nonfinite = _base_static_data(8).loc[:, ["x1", "yb"]].copy()
    nonfinite.loc[3, "x1"] = np.inf
    expect_value_error(
        "nonfinite-continuous-n8",
        "nonfinite_input",
        lambda: UnifiedFamilyScorer(
            nonfinite, {"x1": "con", "yb": "bin"}, {"yb": 2}
        ),
        "non-finite",
    )
    return records


@lru_cache(maxsize=1)
def run_branch_boundary_audit() -> BranchBoundaryAudit:
    matrix = load_fixture_matrix()
    specification = matrix["branch_boundary_audit"]
    if not isinstance(specification, dict):
        raise ValueError("branch_boundary_audit must be an object")
    replicates, seed = int(specification["replicates"]), int(specification["seed"])
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    pairs: list[tuple[float, float]] = []
    n = 20
    context = np.arange(n) % 2
    alpha_cell = 1.0 / (2 * 2)
    for _ in range(replicates):
        probabilities = rng.beta(alpha_cell, alpha_cell, size=2)
        child = rng.binomial(1, probabilities[context])
        null_parent = rng.normal(size=n)
        frame = pd.DataFrame({"d": context, "x_null": null_parent, "y": child})
        scorer = UnifiedFamilyScorer(
            frame,
            {"d": "bin", "x_null": "con", "y": "bin"},
            {"d": 2, "y": 2},
            alpha=1.0,
            glm_tau=1.0,
        )
        bdeu = scorer.score("y", ["d"])
        logit = scorer.score("y", ["d", "x_null"])
        if bdeu.family != "disc_bdeu" or logit.family != "logit":
            raise AssertionError("branch-boundary fixture dispatched unexpectedly")
        pairs.append((bdeu.log_marginal, logit.log_marginal))
    values = np.asarray([right - left for left, right in pairs], dtype=float)
    finite = np.isfinite(values)
    quantile_values = np.quantile(values[finite], [0.05, 0.5, 0.95])
    return BranchBoundaryAudit(
        fixture_id=str(specification["fixture_id"]),
        replicates=replicates,
        seed=seed,
        finite_rate=float(np.mean(finite)),
        mean_log_bayes_factor_difference=float(np.mean(values[finite])),
        sd_log_bayes_factor_difference=float(np.std(values[finite], ddof=1)),
        quantiles={
            "q05": float(quantile_values[0]),
            "q50": float(quantile_values[1]),
            "q95": float(quantile_values[2]),
        },
        first_five_pairs=tuple((float(left), float(right)) for left, right in pairs[:5]),
        bdeu_formula="normalized Dirichlet-multinomial marginal; no external BIC penalty",
        logit_formula="normalized Normal-prior Laplace marginal; no external BIC penalty",
        duplicated_external_penalty=False,
        passed=bool(np.all(finite)),
    )


def run_e0a_score_validation() -> list[ScoreValidationRecord]:
    """Run the independent score-reference anchors used by the E0a CLI."""
    records, _, _ = _reference_anchor_bundle()
    return list(records)


def _reference_coverage_report(expanded_structural_cells: int) -> dict[str, object]:
    """Fail-closed coverage decision for the independent-reference matrix."""
    matrix = load_fixture_matrix()
    raw_anchors = matrix.get("reference_anchors")
    if not isinstance(raw_anchors, list):
        raise ValueError("fixture matrix reference_anchors must be a list")
    anchor_ids = tuple(
        str(anchor["fixture_id"])
        for anchor in raw_anchors
        if isinstance(anchor, dict) and "fixture_id" in anchor
    )
    independently_validated = len(set(anchor_ids))
    anchors_well_formed = len(anchor_ids) == len(raw_anchors) == independently_validated
    required_prefixes = {
        "bdeu-binary",
        "bdeu-cat3",
        "bdeu-cat4",
        "nig-continuous-no-parent",
        "nig-continuous-one-continuous",
        "nig-continuous-mixed",
        "logit-binary-one-continuous",
        "logit-binary-two-continuous",
        "softmax-cat3-one-continuous",
    }
    covered_prefixes = {
        prefix
        for prefix in required_prefixes
        if any(fixture_id.startswith(prefix) for fixture_id in anchor_ids)
    }
    complete = anchors_well_formed and covered_prefixes == required_prefixes
    return {
        "status": "pass" if complete else "fail",
        "complete": complete,
        "blocker_code": None if complete else "incomplete_independent_reference_coverage",
        "blocker": (
            None
            if complete
            else (
                "Independent references do not cover every required mathematical branch; "
                f"missing {sorted(required_prefixes - covered_prefixes)}."
            )
        ),
        "expanded_structural_cells": int(expanded_structural_cells),
        "independently_validated_fixture_count": independently_validated,
        "independently_validated_fixture_ids": list(anchor_ids),
        "anchor_ids_unique_and_well_formed": anchors_well_formed,
        "independent_reference_anchors": independently_validated,
        "logit_free_coefficient_counts": [2, 3],
        "softmax_free_coefficient_counts": [4],
        "all_expanded_cells_executed": int(expanded_structural_cells) == 72,
        "required_reference_branches": sorted(required_prefixes),
        "covered_reference_branches": sorted(covered_prefixes),
        "interpretation": (
            "All 72 static cells are hashed and executed for branch and finite-value "
            "policy. Independent formula or integration references cover every distinct "
            "mathematical score branch rather than duplicating the same integration over "
            "every sample-size and parent-count cell."
        ),
    }


def run_e0a_full_validation() -> dict[str, object]:
    """Return the complete fixture-level E0a report without hiding failures."""
    score_records, derivative_audits, numerical_references = _reference_anchor_bundle()
    structural = run_structural_fixture_matrix()
    invariance = run_invariance_checks()
    policies = run_error_policy_checks()
    boundary = run_branch_boundary_audit()
    reference_coverage = _reference_coverage_report(len(structural))
    statuses: list[ValidationStatus] = [record.status for record in score_records]
    statuses.extend(record.status for record in structural)
    statuses.extend(record.status for record in invariance)
    statuses.extend("pass" if record.passed else "fail" for record in derivative_audits)
    statuses.extend("pass" if record.passed else "fail" for record in policies)
    statuses.append("pass" if boundary.passed else "fail")
    statuses.append(str(reference_coverage["status"]))
    overall: ValidationStatus = (
        "fail"
        if "fail" in statuses
        else "inconclusive"
        if "inconclusive" in statuses
        else "pass"
    )
    return {
        "schema": "ordersmc-e0a-validation-report-v1",
        "status": overall,
        "all_passed": overall == "pass",
        "fixture_matrix": audit_fixture_matrix().to_dict(),
        "score_accuracy_records": [record.to_dict() for record in score_records],
        "structural_matrix_records": [record.to_dict() for record in structural],
        "derivative_audits": [record.to_dict() for record in derivative_audits],
        "invariance_records": [record.to_dict() for record in invariance],
        "error_policy_records": [record.to_dict() for record in policies],
        "branch_boundary_audit": boundary.to_dict(),
        "numerical_references": numerical_references,
        "reference_coverage": reference_coverage,
        "known_gate_rule": (
            "Any failed branch-specific accuracy fixture, inconclusive numerical reference, "
            "incomplete mathematical-branch coverage, or failed exhaustive structural cell "
            "blocks G0."
        ),
    }
