"""Local family scores for discrete, continuous, and mixed Bayesian networks."""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
from scipy.special import gammaln, logsumexp, ndtri
from scipy.stats import rankdata

try:
    import pandas as pd  # optional, only needed for input df type
except Exception:  # pragma: no cover
    pd = None  # type: ignore


VarType = Literal["bin", "cat", "con"]
FamilyType = Literal[
    "disc_bds",
    "disc_bdeu",
    "clg",
    "clg_bic",
    "logit",
    "softmax",
    "dg",
]


class UnscorableFamilyError(ValueError):
    """A local family is outside the estimable support of the selected score."""


def rank_normal_scores(values: np.ndarray) -> np.ndarray:
    """Training-sample normal scores Phi^-1((midrank - 0.5) / n).

    Applied once per continuous column, independently of every proposed graph.
    This is a plug-in transformation, not an integrated copula rank likelihood.
    Ties have deterministic average ranks; no jitter or inference seed is used.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("rank-normal scores require a one-dimensional sample of size >= 2")
    if not np.isfinite(values).all():
        raise ValueError("rank-normal scores require finite observations")
    if np.all(values == values[0]):
        raise ValueError("rank-normal scores require nonzero variance")
    return ndtri((rankdata(values, method="average") - 0.5) / values.size)


@dataclass(frozen=True)
class FamilyScoreResult:
    family: FamilyType
    log_marginal: float
    k_params: int
    q_total: Optional[int] = None
    q_obs: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


def _prod(ints: Sequence[int]) -> int:
    out = 1
    for x in ints:
        out *= int(x)
    return int(out)


def _to_int(x) -> np.ndarray:
    # x can be np array or pd series
    arr = np.asarray(x)
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int64, copy=False)
    else:
        arr = arr.astype(np.int64, copy=False)
    return arr


def _to_float(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind != "f":
        arr = arr.astype(np.float64, copy=False)
    else:
        arr = arr.astype(np.float64, copy=False)
    return arr


def _validated_discrete_to_int(x, *, name: str, kind: VarType,
                               card: int) -> np.ndarray:
    """Validate the raw values before any potentially lossy integer cast."""
    raw = np.asarray(x)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional column.")
    try:
        numeric = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is discrete but contains non-numeric values.") from exc
    if numeric.size and not np.isfinite(numeric).all():
        raise ValueError(f"{name} is discrete but contains missing or non-finite values.")
    if numeric.size and not np.equal(numeric, np.trunc(numeric)).all():
        raise ValueError(f"{name} is discrete but contains non-integer values.")

    if kind == "bin" and int(card) != 2:
        raise ValueError(f"{name} is bin but has cardinality {card}; expected 2.")
    upper = 2 if kind == "bin" else int(card)
    if upper < 2:
        raise ValueError(f"{name} has invalid cardinality {upper}; expected at least 2.")
    if numeric.size and (numeric.min() < 0.0 or numeric.max() >= float(upper)):
        if kind == "bin":
            raise ValueError(f"{name} is bin but has values outside {{0,1}}.")
        raise ValueError(f"{name} is cat but values outside [0,{upper - 1}].")
    return numeric.astype(np.int64, copy=False)


def _mixed_radix_index(parent_cols: List[np.ndarray], parent_cards: List[int], n: int) -> Tuple[np.ndarray, int]:
    """
    Map discrete parent configurations to indices 0..q_total-1 in mixed radix.
    Returns (idx, q_total).
    """
    if len(parent_cols) == 0:
        return np.zeros(n, dtype=np.int64), 1

    idx = np.zeros(n, dtype=np.int64)
    mul = 1
    q_total = 1
    for col, K in zip(parent_cols, parent_cards):
        K = int(K)
        idx += col * mul
        mul *= K
        q_total *= K
    return idx, int(q_total)


def _group_slices(inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given inv in {0..q_obs-1}, return (order, starts, ends) where:
      order sorts inv,
      inv[order][starts[i]:ends[i]] is one group.
    """
    if inv.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(inv, kind="mergesort")  # stable
    invs = inv[order]
    cuts = np.flatnonzero(np.diff(invs)) + 1
    starts = np.concatenate(([0], cuts)).astype(np.int64)
    ends = np.concatenate((cuts, [inv.size])).astype(np.int64)
    return order.astype(np.int64), starts, ends


def _disc_counts_compact(
    y: np.ndarray,
    parent_cols: List[np.ndarray],
    parent_cards: List[int],
    r: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
      counts: (q_obs, r) counts for observed parent configs only
      q_total: product of parent_cards (or 1)
      q_obs: number of observed configs (0 if n==0)
    Memory-safe even if q_total is huge (we never allocate q_total*r).
    """
    y = y.astype(np.int64, copy=False)
    n = int(y.shape[0])

    if n == 0:
        return np.zeros((0, r), dtype=np.int64), _prod(parent_cards) if parent_cards else 1, 0

    idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)

    # compact observed configs
    uniq, inv = np.unique(idx, return_inverse=True)
    q_obs = int(uniq.shape[0])

    flat = inv.astype(np.int64) * int(r) + y
    counts = np.bincount(flat, minlength=q_obs * int(r)).reshape(q_obs, int(r)).astype(np.int64, copy=False)
    return counts, int(q_total), int(q_obs)


def _bdeu_log_marginal_from_counts(
    counts: np.ndarray,
    *,
    alpha: float,
    r: int,
    q_total: int,
    q_obs: int,
    sparse: bool,
) -> float:
    """
    Vectorized BDeu / "sparse BDeu" over *observed* contexts only.

    For standard BDeu: alpha_c = alpha / q_total
    For sparse-BDeu:   alpha_c = alpha / q_obs

    Unobserved contexts contribute exactly zero to the BDeu sum, so we do not need
    to iterate/allocate over q_total.
    """
    if q_obs <= 0:
        return 0.0

    alpha = float(alpha)
    r = int(r)
    q_total = int(q_total)
    q_obs = int(q_obs)

    alpha_c = alpha / (q_obs if sparse else q_total)
    alpha_sc = alpha_c / r

    Nc = counts.sum(axis=1).astype(np.float64, copy=False)  # (q_obs,)
    # sum over observed contexts
    term_ctx = np.sum(gammaln(alpha_c) - gammaln(alpha_c + Nc))
    term_cells = np.sum(gammaln(alpha_sc + counts) - gammaln(alpha_sc))
    return float(term_ctx + term_cells)


class DiscreteSparseBDeuScorer:
    """
    Discrete local score using a sparse-BDeu allocation:
      alpha spread across observed parent configurations q_obs (Nc>0).
    Parameter count uses q_total*(r-1).
    """

    def __init__(self, data_int: Dict[str, np.ndarray], cards: Dict[str, int], alpha: float = 1.0):
        self.Xi = data_int
        self.cards = dict(cards)
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and strictly positive.")

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        r = int(self.cards[child])
        y = self.Xi[child]

        parent_cols = [self.Xi[p] for p in parents]
        parent_cards = [int(self.cards[p]) for p in parents]

        counts, q_total, q_obs = _disc_counts_compact(y, parent_cols, parent_cards, r)
        log_m = _bdeu_log_marginal_from_counts(counts, alpha=self.alpha, r=r, q_total=q_total, q_obs=q_obs, sparse=True)

        return FamilyScoreResult(
            family="disc_bds",
            log_marginal=float(log_m),
            k_params=int(q_total * (r - 1)),
            q_total=int(q_total),
            q_obs=int(q_obs),
        )


class DiscreteBDeuScorer:
    """Standard BDeu: alpha is spread across ALL q_total parent configs."""

    def __init__(self, data_int: Dict[str, np.ndarray], cards: Dict[str, int], alpha: float = 1.0):
        self.Xi = data_int
        self.cards = dict(cards)
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and strictly positive.")

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        r = int(self.cards[child])
        y = self.Xi[child]

        parent_cols = [self.Xi[p] for p in parents]
        parent_cards = [int(self.cards[p]) for p in parents]

        counts, q_total, q_obs = _disc_counts_compact(y, parent_cols, parent_cards, r)
        log_m = _bdeu_log_marginal_from_counts(counts, alpha=self.alpha, r=r, q_total=q_total, q_obs=q_obs, sparse=False)

        return FamilyScoreResult(
            family="disc_bdeu",
            log_marginal=float(log_m),
            k_params=int(q_total * (r - 1)),
            q_total=int(q_total),
            q_obs=int(q_obs),
        )


class ConditionalGaussianBICScorer:
    """Restricted conditional-Gaussian BIC local score.

    This is the standard heterogeneous conditional-Gaussian model class: a
    discrete child may have only discrete parents, while a continuous child
    may have discrete and continuous parents.  Continuous regressions are fit
    separately in each discrete-parent configuration.  The score is the
    maximized local log likelihood minus ``0.5 * k * log(n)`` and is therefore
    a BIC generalized-posterior weight, not a marginal likelihood.
    """

    def __init__(
        self,
        data_int: Dict[str, np.ndarray],
        data_float: Dict[str, np.ndarray],
        vtypes: Dict[str, VarType],
        cards: Dict[str, int],
    ):
        self.Xi = data_int
        self.Xf = data_float
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        lengths = {len(values) for values in [*self.Xi.values(), *self.Xf.values()]}
        if len(lengths) != 1:
            raise ValueError("all variables must contain the same number of observations")
        self.n = int(next(iter(lengths))) if lengths else 0
        if self.n < 2:
            raise ValueError("clg-bic requires at least two observations")

    def parent_allowed(self, child: str, parent: str) -> bool:
        """Return whether an edge belongs to the restricted CLG model class."""
        child_discrete = self.vtypes[child] in ("bin", "cat")
        parent_continuous = self.vtypes[parent] == "con"
        return not (child_discrete and parent_continuous)

    def _discrete_score(self, child: str, parents: Tuple[str, ...]) -> FamilyScoreResult:
        r = int(self.cards[child])
        parent_cols = [self.Xi[parent] for parent in parents]
        parent_cards = [int(self.cards[parent]) for parent in parents]
        counts, q_total, q_obs = _disc_counts_compact(
            self.Xi[child], parent_cols, parent_cards, r
        )
        row_totals = counts.sum(axis=1, keepdims=True)
        positive = counts > 0
        log_likelihood = float(
            np.sum(
                counts[positive]
                * (
                    np.log(counts[positive])
                    - np.broadcast_to(np.log(row_totals), counts.shape)[positive]
                )
            )
        )
        k_params = int(q_total * (r - 1))
        score = log_likelihood - 0.5 * k_params * np.log(float(self.n))
        return FamilyScoreResult(
            family="clg_bic",
            log_marginal=float(score),
            k_params=k_params,
            q_total=q_total,
            q_obs=q_obs,
            details={
                "component": "multinomial",
                "maximized_log_likelihood": log_likelihood,
                "bic_penalty": float(0.5 * k_params * np.log(float(self.n))),
            },
        )

    def _continuous_score(self, child: str, parents: Tuple[str, ...]) -> FamilyScoreResult:
        discrete_parents = [p for p in parents if self.vtypes[p] in ("bin", "cat")]
        continuous_parents = [p for p in parents if self.vtypes[p] == "con"]
        y = self.Xf[child]
        coefficient_count = len(continuous_parents) + 1

        if discrete_parents:
            parent_cols = [self.Xi[parent] for parent in discrete_parents]
            parent_cards = [int(self.cards[parent]) for parent in discrete_parents]
            context_index, q_total = _mixed_radix_index(parent_cols, parent_cards, self.n)
            _, inverse = np.unique(context_index, return_inverse=True)
            order, starts, ends = _group_slices(inverse.astype(np.int64, copy=False))
        else:
            q_total = 1
            order = np.arange(self.n, dtype=np.int64)
            starts = np.asarray([0], dtype=np.int64)
            ends = np.asarray([self.n], dtype=np.int64)

        log_likelihood = 0.0
        for start, end in zip(starts, ends):
            indices = order[start:end]
            context_size = int(indices.size)
            if context_size <= coefficient_count:
                raise UnscorableFamilyError(
                    f"clg-bic cannot fit {child!r}: a discrete-parent context has "
                    f"{context_size} observations for {coefficient_count} coefficients"
                )
            design = np.ones((context_size, coefficient_count), dtype=np.float64)
            for column, parent in enumerate(continuous_parents, start=1):
                design[:, column] = self.Xf[parent][indices]
            if np.linalg.matrix_rank(design) < coefficient_count:
                raise UnscorableFamilyError(
                    f"clg-bic cannot fit {child!r}: a local regression design is rank deficient"
                )
            response = y[indices]
            coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
            residual = response - design @ coefficients
            rss = float(np.dot(residual, residual))
            tolerance = (
                np.finfo(np.float64).eps
                * max(1.0, float(np.dot(response, response)))
                * max(1, context_size)
            )
            if not np.isfinite(rss) or rss <= tolerance:
                raise UnscorableFamilyError(
                    f"clg-bic cannot fit {child!r}: a local residual variance is numerically zero"
                )
            variance = rss / context_size
            log_likelihood += -0.5 * context_size * (
                np.log(2.0 * np.pi * variance) + 1.0
            )

        q_obs = int(len(starts))
        k_params = int(q_total * (coefficient_count + 1))
        bic_penalty = 0.5 * k_params * np.log(float(self.n))
        return FamilyScoreResult(
            family="clg_bic",
            log_marginal=float(log_likelihood - bic_penalty),
            k_params=k_params,
            q_total=int(q_total),
            q_obs=q_obs,
            details={
                "component": "context_specific_gaussian_regression",
                "maximized_log_likelihood": float(log_likelihood),
                "bic_penalty": float(bic_penalty),
                "variance_denominator": "context_size",
            },
        )

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        invalid = [parent for parent in parents if not self.parent_allowed(child, parent)]
        if invalid:
            raise UnscorableFamilyError(
                "restricted clg-bic forbids continuous parents of a discrete child; "
                f"child={child!r}, invalid_parents={invalid!r}"
            )
        if self.vtypes[child] in ("bin", "cat"):
            return self._discrete_score(child, parents)
        return self._continuous_score(child, parents)


class CLGScorer:
    """
    Continuous child with (possibly) discrete + continuous parents:
      - partition by discrete parent configurations;
      - integrate a Gaussian regression independently in every observed context;
      - use a proper Normal--Inverse-Gamma prior, so even singleton and
        perfectly fitted contexts have a finite score.

    In a context, with ``d`` regression coefficients (including the intercept),

      y | beta, sigma2 ~ N(Z beta, sigma2 I)
      beta | sigma2    ~ N(0, sigma2 * tau^2 I)
      sigma2           ~ InvGamma(ig_shape, ig_scale).

    Unobserved discrete contexts integrate to one.  They therefore add zero to
    the log marginal, but they still count in ``k_params`` because the model has
    parameters for all ``q_total`` possible contexts.
    """

    def __init__(self, data_int: Dict[str, np.ndarray], data_float: Dict[str, np.ndarray],
                 vtypes: Dict[str, VarType], cards: Dict[str, int], *,
                 tau: float = 1.0, ig_shape: float = 2.0,
                 ig_scale: float = 1.0):
        self.Xi = data_int
        self.Xf = data_float
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        self.tau = float(tau)
        self.ig_shape = float(ig_shape)
        self.ig_scale = float(ig_scale)
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and strictly positive.")
        if not np.isfinite(self.ig_shape) or self.ig_shape <= 0.0:
            raise ValueError("ig_shape must be finite and strictly positive.")
        if not np.isfinite(self.ig_scale) or self.ig_scale <= 0.0:
            raise ValueError("ig_scale must be finite and strictly positive.")

    def _nig_log_marginal(self, y: np.ndarray, Z: np.ndarray) -> float:
        """Exact Normal--Inverse-Gamma regression log marginal likelihood."""
        n = int(y.shape[0])
        if n <= 0:
            return 0.0

        d = int(Z.shape[1])
        prior_precision = 1.0 / (self.tau * self.tau)
        precision_n = Z.T @ Z
        precision_n[np.diag_indices(d)] += prior_precision

        try:
            chol = np.linalg.cholesky(precision_n)
        except np.linalg.LinAlgError as exc:  # the proper prior makes this PD
            raise RuntimeError("CLG posterior precision is not positive definite.") from exc

        zty = Z.T @ y
        mean_n = np.linalg.solve(chol.T, np.linalg.solve(chol, zty))
        # y'y - m_n' Lambda_n m_n is non-negative analytically.  Evaluating it
        # as a residual plus the prior quadratic avoids cancellation for nearly
        # perfect fits.
        residual = y - Z @ mean_n
        quadratic = float(np.dot(residual, residual) + prior_precision * np.dot(mean_n, mean_n))

        shape_n = self.ig_shape + 0.5 * n
        scale_n = self.ig_scale + 0.5 * quadratic
        logdet_precision_n = float(2.0 * np.sum(np.log(np.diag(chol))))
        logdet_v0 = float(d * np.log(self.tau * self.tau))

        return float(
            -0.5 * n * np.log(2.0 * np.pi)
            -0.5 * (logdet_precision_n + logdet_v0)
            + self.ig_shape * np.log(self.ig_scale)
            - shape_n * np.log(scale_n)
            + gammaln(shape_n)
            - gammaln(self.ig_shape)
        )

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        y = self.Xf[child]
        n = int(y.shape[0])

        Pd = [p for p in parents if self.vtypes[p] in ("bin", "cat")]
        Pc = [p for p in parents if self.vtypes[p] == "con"]

        p = int(len(Pc))
        k_per_context = (p + 1) + 1  # coefficients (incl intercept) + variance

        def design(indices: Optional[np.ndarray] = None) -> np.ndarray:
            rows = n if indices is None else int(indices.shape[0])
            Z = np.empty((rows, p + 1), dtype=np.float64)
            Z[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Z[:, j] = self.Xf[name] if indices is None else self.Xf[name][indices]
            return Z

        # no discrete parents: single regression
        if len(Pd) == 0:
            log_m = self._nig_log_marginal(y, design())
            q_obs = 1 if n > 0 else 0
            return FamilyScoreResult(
                "clg",
                float(log_m),
                k_params=int(k_per_context),
                q_total=1,
                q_obs=q_obs,
                details={
                    "contexts": q_obs,
                    "q_total_disc": 1,
                    "q_obs_disc": q_obs,
                    "ig_shape": self.ig_shape,
                    "ig_scale": self.ig_scale,
                    "tau": self.tau,
                },
            )

        # discrete contexts (observed only)
        parent_cols = [self.Xi[pn] for pn in Pd]
        parent_cards = [int(self.cards[pn]) for pn in Pd]
        idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)
        uniq, inv = np.unique(idx, return_inverse=True)
        q_obs = int(uniq.shape[0])

        order, starts, ends = _group_slices(inv.astype(np.int64, copy=False))

        log_m_sum = 0.0
        contexts = 0

        for a, b in zip(starts, ends):
            inds = order[a:b]
            n_c = int(inds.shape[0])
            if n_c == 0:
                continue
            contexts += 1
            yc = y[inds]

            log_m_sum += self._nig_log_marginal(yc, design(inds))

        return FamilyScoreResult(
            "clg",
            float(log_m_sum),
            k_params=int(q_total * k_per_context),
            q_total=int(q_total),
            q_obs=int(q_obs),
            details={
                "contexts": int(contexts),
                "q_total_disc": int(q_total),
                "q_obs_disc": int(q_obs),
                "ig_shape": self.ig_shape,
                "ig_scale": self.ig_scale,
                "tau": self.tau,
            },
        )


class LogitScorer:
    """
    Binary child with continuous parents (+ optional discrete contexts),
    Laplace approximation around MAP (IRLS/Newton).
    """

    def __init__(self, data_int: Dict[str, np.ndarray], data_float: Dict[str, np.ndarray],
                 vtypes: Dict[str, VarType], cards: Dict[str, int], tau: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-9):
        self.Xi = data_int
        self.Xf = data_float
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        self.tau = float(tau)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and strictly positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be strictly positive.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and strictly positive.")

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def _fit_map(self, y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Returns (beta_hat, logdet(H_post), loglik(beta_hat) + logprior(beta_hat)).
        Every coefficient, including the intercept, has the proper prior
        N(0, tau^2).  Newton steps are backtracked against the exact, stably
        evaluated negative log posterior.
        """
        n, p = Z.shape
        beta = np.zeros(p, dtype=np.float64)
        prior_precision = 1.0 / (self.tau * self.tau)
        eye = np.eye(p, dtype=np.float64)

        def objective(value: np.ndarray) -> float:
            eta = Z @ value
            return float(
                np.sum(np.logaddexp(0.0, eta) - y * eta)
                + 0.5 * prior_precision * np.dot(value, value)
            )

        converged = False
        for _iteration in range(1, self.max_iter + 1):
            eta = Z @ beta
            prob = self._sigmoid(eta)
            W = prob * (1.0 - prob)

            g = Z.T @ (prob - y) + prior_precision * beta
            if float(np.linalg.norm(g, ord=np.inf)) <= self.tol * (1.0 + n):
                converged = True
                break
            H = (Z.T * W) @ Z
            H[np.diag_indices(p)] += prior_precision

            damping = 0.0
            for _attempt in range(12):
                try:
                    step = np.linalg.solve(H + damping * eye, g)
                except np.linalg.LinAlgError:
                    damping = 1e-8 if damping == 0.0 else damping * 10.0
                    continue

                directional = float(np.dot(g, step))
                if not np.isfinite(directional) or directional <= 0.0:
                    damping = 1e-8 if damping == 0.0 else damping * 10.0
                    continue

                old_objective = objective(beta)
                step_scale = 1.0
                accepted = False
                for _backtrack in range(40):
                    candidate = beta - step_scale * step
                    candidate_objective = objective(candidate)
                    if np.isfinite(candidate_objective) and candidate_objective <= (
                        old_objective - 1e-4 * step_scale * directional
                    ):
                        accepted = True
                        break
                    step_scale *= 0.5
                if accepted:
                    break
                damping = 1e-8 if damping == 0.0 else damping * 10.0
            else:
                raise RuntimeError("Binary-logit Newton step or line search failed.")

            beta = candidate
            if float(np.linalg.norm(step_scale * step)) <= self.tol * (
                1.0 + float(np.linalg.norm(beta))
            ):
                converged = True
                break

        if not converged:
            eta = Z @ beta
            prob = self._sigmoid(eta)
            gradient = Z.T @ (prob - y) + prior_precision * beta
            converged = float(np.linalg.norm(gradient, ord=np.inf)) <= 10.0 * self.tol * (1.0 + n)
        if not converged:
            raise RuntimeError(f"Binary-logit MAP did not converge in {self.max_iter} iterations.")

        eta = Z @ beta
        prob = self._sigmoid(eta)
        loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        logprior = float(
            -0.5 * (
                prior_precision * np.dot(beta, beta)
                + p * np.log(2.0 * np.pi * self.tau * self.tau)
            )
        )

        W = prob * (1.0 - prob)
        H = (Z.T * W) @ Z
        H[np.diag_indices(p)] += prior_precision
        try:
            chol = np.linalg.cholesky(H)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Binary-logit posterior Hessian is not positive definite.") from exc
        logdet = float(2.0 * np.sum(np.log(np.diag(chol))))

        return beta, float(logdet), float(loglik + logprior)

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)

        Pd = [p for p in parents if self.vtypes[p] in ("bin", "cat")]
        Pc = [p for p in parents if self.vtypes[p] == "con"]

        y = self.Xi[child].astype(np.float64, copy=False)
        if y.size and (y.min() < 0 or y.max() > 1):
            raise ValueError(f"{child} is binary but has values outside {{0,1}}.")

        if len(Pc) == 0:
            raise ValueError("LogitScorer requires at least one continuous parent.")

        n = int(y.shape[0])
        p = int(len(Pc))
        Kdim = p + 1  # intercept + slopes

        # no discrete contexts -> one fit
        if len(Pd) == 0:
            Z = np.empty((n, Kdim), dtype=np.float64)
            Z[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Z[:, j] = self.Xf[name]

            _, logdet, ll_plus_prior = self._fit_map(y, Z)
            log_marg = ll_plus_prior - 0.5 * logdet + 0.5 * Kdim * np.log(2.0 * np.pi)
            q_obs = 1 if n > 0 else 0
            return FamilyScoreResult(
                "logit", float(log_marg), k_params=int(Kdim), q_total=1, q_obs=q_obs,
                details={"contexts": q_obs, "q_total_disc": 1, "q_obs_disc": q_obs, "tau": self.tau},
            )

        # observed discrete contexts only
        parent_cols = [self.Xi[pn] for pn in Pd]
        parent_cards = [int(self.cards[pn]) for pn in Pd]
        idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)
        uniq, inv = np.unique(idx, return_inverse=True)
        q_obs = int(uniq.shape[0])

        order, starts, ends = _group_slices(inv.astype(np.int64, copy=False))

        total = 0.0
        contexts = 0

        for a, b in zip(starts, ends):
            inds = order[a:b]
            n_c = int(inds.shape[0])
            if n_c == 0:
                continue
            contexts += 1

            yc = y[inds]
            Zc = np.empty((n_c, Kdim), dtype=np.float64)
            Zc[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Zc[:, j] = self.Xf[name][inds]

            _, logdet, ll_plus_prior = self._fit_map(yc, Zc)
            total += ll_plus_prior - 0.5 * logdet + 0.5 * Kdim * np.log(2.0 * np.pi)

        return FamilyScoreResult(
            "logit",
            float(total),
            k_params=int(q_total * Kdim),
            q_total=int(q_total),
            q_obs=int(q_obs),
            details={
                "contexts": int(contexts), "q_total_disc": int(q_total),
                "q_obs_disc": int(q_obs), "tau": self.tau,
            },
        )


class SoftmaxScorer:
    """
    Categorical child with continuous parents (+ optional discrete contexts).
    Laplace approximation around the MAP for multinomial logistic regression.

    Class coefficients are represented in an orthonormal contrast basis for the
    sum-to-zero subspace.  The resulting proper isotropic Normal prior treats all
    labels symmetrically, unlike a regularized baseline-class parametrization.
    """

    def __init__(self, data_int: Dict[str, np.ndarray], data_float: Dict[str, np.ndarray],
                 vtypes: Dict[str, VarType], cards: Dict[str, int], tau: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-9):
        self.Xi = data_int
        self.Xf = data_float
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        self.tau = float(tau)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and strictly positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be strictly positive.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and strictly positive.")

    @staticmethod
    def _softmax(A: np.ndarray) -> np.ndarray:
        A = A - A.max(axis=1, keepdims=True)
        E = np.exp(A)
        return E / E.sum(axis=1, keepdims=True)

    @staticmethod
    def _contrast_basis(K: int) -> np.ndarray:
        """Return a K x (K-1) orthonormal Helmert basis for 1-perp."""
        C = np.zeros((K, K - 1), dtype=np.float64)
        for j in range(K - 1):
            denom = np.sqrt(float((j + 1) * (j + 2)))
            C[:j + 1, j] = 1.0 / denom
            C[j + 1, j] = -(j + 1) / denom
        return C

    @staticmethod
    def _hessian(P: np.ndarray, Z: np.ndarray, C: np.ndarray,
                 prior_precision: float) -> np.ndarray:
        """Negative-log-posterior Hessian in contrast coordinates."""
        R = int(C.shape[1])
        d = int(Z.shape[1])
        dim = R * d
        H = np.empty((dim, dim), dtype=np.float64)
        PC = P @ C
        for a in range(R):
            for b in range(R):
                w = P @ (C[:, a] * C[:, b]) - PC[:, a] * PC[:, b]
                block = (Z.T * w) @ Z
                if a == b:
                    block = block.copy()
                    block[np.diag_indices(d)] += prior_precision
                H[a*d:(a+1)*d, b*d:(b+1)*d] = block
        # Remove only floating-point block asymmetry; the analytic Hessian is symmetric.
        return 0.5 * (H + H.T)

    def _fit_map(self, y: np.ndarray, Z: np.ndarray, K: int) -> Tuple[np.ndarray, float, float]:
        """
        Fit identifiable contrast coefficients Theta of shape ((K-1), d).
        Entries, including intercepts, have N(0, tau^2) priors for K>=3.
        For K=2 the contrast variance is tau^2/2, making the log-odds
        prior identical to LogitScorer under the same declared tau.
        """
        n, d = Z.shape
        R = K - 1
        Theta = np.zeros((R, d), dtype=np.float64)
        C = self._contrast_basis(K)
        # For K=2 the log-odds coefficient is sqrt(2) times its Helmert
        # contrast. Match LogitScorer's N(0, tau^2) log-odds prior exactly.
        # K>=3 keeps the existing label-exchangeable contrast prior.
        prior_variance = self.tau * self.tau / (2.0 if K == 2 else 1.0)
        prior_precision = 1.0 / prior_variance

        Y = np.zeros((n, K), dtype=np.float64)
        Y[np.arange(n), y.astype(int)] = 1.0

        dim = R * d
        eye = np.eye(dim, dtype=np.float64)

        def logits_for(value: np.ndarray) -> np.ndarray:
            return (Z @ value.T) @ C.T

        def objective(value: np.ndarray) -> float:
            logits = logits_for(value)
            loglik = float(np.sum(logits[np.arange(n), y.astype(int)] - logsumexp(logits, axis=1)))
            return float(-loglik + 0.5 * prior_precision * np.sum(value * value))

        converged = False
        for _iteration in range(1, self.max_iter + 1):
            logits = logits_for(Theta)
            P = self._softmax(logits)

            G = (C.T @ (P - Y).T) @ Z + prior_precision * Theta
            g = G.reshape(-1)
            if float(np.linalg.norm(g, ord=np.inf)) <= self.tol * (1.0 + n):
                converged = True
                break
            H = self._hessian(P, Z, C, prior_precision)

            damping = 0.0
            for _attempt in range(12):
                try:
                    step = np.linalg.solve(H + damping * eye, g)
                except np.linalg.LinAlgError:
                    damping = 1e-8 if damping == 0.0 else damping * 10.0
                    continue

                directional = float(np.dot(g, step))
                if not np.isfinite(directional) or directional <= 0.0:
                    damping = 1e-8 if damping == 0.0 else damping * 10.0
                    continue

                old_objective = objective(Theta)
                step_scale = 1.0
                accepted = False
                for _backtrack in range(40):
                    candidate = (Theta.reshape(-1) - step_scale * step).reshape(R, d)
                    candidate_objective = objective(candidate)
                    if np.isfinite(candidate_objective) and candidate_objective <= (
                        old_objective - 1e-4 * step_scale * directional
                    ):
                        accepted = True
                        break
                    step_scale *= 0.5
                if accepted:
                    break
                damping = 1e-8 if damping == 0.0 else damping * 10.0
            else:
                raise RuntimeError("Softmax Newton step or line search failed.")

            Theta = candidate
            if float(np.linalg.norm(step_scale * step)) <= self.tol * (
                1.0 + float(np.linalg.norm(Theta))
            ):
                converged = True
                break

        if not converged:
            logits = logits_for(Theta)
            P = self._softmax(logits)
            gradient = (C.T @ (P - Y).T) @ Z + prior_precision * Theta
            converged = float(np.linalg.norm(gradient, ord=np.inf)) <= 10.0 * self.tol * (1.0 + n)
        if not converged:
            raise RuntimeError(f"Softmax MAP did not converge in {self.max_iter} iterations.")

        logits = logits_for(Theta)
        P = self._softmax(logits)
        loglik = float(np.sum(logits[np.arange(n), y.astype(int)] - logsumexp(logits, axis=1)))
        logprior = float(
            -0.5 * (
                prior_precision * np.sum(Theta * Theta)
                + dim * np.log(2.0 * np.pi * prior_variance)
            )
        )

        H = self._hessian(P, Z, C, prior_precision)
        try:
            chol = np.linalg.cholesky(H)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Softmax posterior Hessian is not positive definite.") from exc
        logdet = float(2.0 * np.sum(np.log(np.diag(chol))))

        return Theta, float(logdet), float(loglik + logprior)

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        K = int(self.cards[child])
        if K < 2:
            raise ValueError(f"{child} categorical cardinality must be at least 2.")
        y = self.Xi[child].astype(np.int64, copy=False)
        if y.size and (y.min() < 0 or y.max() >= K):
            raise ValueError(f"{child} categorical but values outside [0,{K-1}].")

        Pd = [p for p in parents if self.vtypes[p] in ("bin", "cat")]
        Pc = [p for p in parents if self.vtypes[p] == "con"]
        if len(Pc) == 0:
            raise ValueError("SoftmaxScorer requires at least one continuous parent.")

        n = int(y.shape[0])
        p = int(len(Pc))
        Km1 = K - 1
        Kdim = int(Km1 * (p + 1))

        if len(Pd) == 0:
            Z = np.empty((n, p + 1), dtype=np.float64)
            Z[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Z[:, j] = self.Xf[name]
            _, logdet, ll_plus_prior = self._fit_map(y.astype(np.float64, copy=False), Z, K)
            log_marg = ll_plus_prior - 0.5 * logdet + 0.5 * Kdim * np.log(2.0 * np.pi)
            q_obs = 1 if n > 0 else 0
            return FamilyScoreResult(
                "softmax", float(log_marg), k_params=int(Kdim), q_total=1, q_obs=q_obs,
                details={"contexts": q_obs, "q_total_disc": 1, "q_obs_disc": q_obs, "tau": self.tau},
            )

        parent_cols = [self.Xi[pn] for pn in Pd]
        parent_cards = [int(self.cards[pn]) for pn in Pd]
        idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)
        uniq, inv = np.unique(idx, return_inverse=True)
        q_obs = int(uniq.shape[0])

        order, starts, ends = _group_slices(inv.astype(np.int64, copy=False))

        total = 0.0
        contexts = 0

        y_f = y.astype(np.float64, copy=False)

        for a, b in zip(starts, ends):
            inds = order[a:b]
            n_c = int(inds.shape[0])
            if n_c == 0:
                continue
            contexts += 1

            Zc = np.empty((n_c, p + 1), dtype=np.float64)
            Zc[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Zc[:, j] = self.Xf[name][inds]

            _, logdet, ll_plus_prior = self._fit_map(y_f[inds], Zc, K)
            total += ll_plus_prior - 0.5 * logdet + 0.5 * Kdim * np.log(2.0 * np.pi)

        return FamilyScoreResult(
            "softmax",
            float(total),
            k_params=int(q_total * Kdim),
            q_total=int(q_total),
            q_obs=int(q_obs),
            details={
                "contexts": int(contexts), "q_total_disc": int(q_total),
                "q_obs_disc": int(q_obs), "tau": self.tau,
            },
        )


class DegenerateGaussianScorer:
    """Score every local family with one degenerate-Gaussian BIC model.

    A continuous variable occupies one embedded dimension and a categorical
    variable with ``K`` levels occupies ``K - 1`` one-hot dimensions.  The
    local score is the joint Gaussian log likelihood minus the parent-only log
    likelihood and the model-dependent BIC penalty from Andrews, Ramsey, and
    Cooper (2019).  Using this scorer for *all* child/parent type combinations
    is what makes the resulting DAG score score-equivalent.

    ``ridge`` is zero by default so the implemented score is the published
    score and remains invariant to full-rank recodings of a categorical block.
    A positive ridge can stabilize singular families while retaining score
    equivalence (the score remains a difference of set functions), but exact
    coding/scale invariance is then no longer guaranteed.
    """

    def __init__(
        self,
        data_int: Dict[str, np.ndarray],
        data_float: Dict[str, np.ndarray],
        vtypes: Dict[str, VarType],
        cards: Dict[str, int],
        *,
        penalty_discount: float = 1.0,
        ridge: float = 0.0,
    ) -> None:
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        for option_name, option_value in (
            ("dg_penalty_discount", penalty_discount),
            ("dg_ridge", ridge),
        ):
            if isinstance(option_value, (bool, np.bool_)) or not isinstance(
                option_value, numbers.Real
            ):
                raise ValueError(f"{option_name} must be a finite number.")
        self.penalty_discount = float(penalty_discount)
        self.ridge = float(ridge)
        if not np.isfinite(self.penalty_discount) or self.penalty_discount <= 0.0:
            raise ValueError("dg_penalty_discount must be finite and strictly positive.")
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("dg_ridge must be finite and non-negative.")

        lengths = {
            int((data_float if kind == "con" else data_int)[name].shape[0])
            for name, kind in self.vtypes.items()
        }
        if len(lengths) != 1:
            raise ValueError("All columns must contain the same number of observations.")
        self.n = lengths.pop()
        if self.n < 2:
            raise ValueError("The DG score requires at least two observations.")

        blocks: List[np.ndarray] = []
        self.block_indices: Dict[str, np.ndarray] = {}
        self.block_dims: Dict[str, int] = {}
        offset = 0
        for name, kind in self.vtypes.items():
            if kind == "con":
                block = np.asarray(data_float[name], dtype=np.float64).reshape(self.n, 1)
            else:
                card = int(self.cards[name])
                values = np.asarray(data_int[name], dtype=np.int64)
                observed = np.bincount(values, minlength=card)
                if observed.shape[0] != card or np.any(observed == 0):
                    raise ValueError(
                        f"{name}: the DG score requires every declared category to be observed."
                    )
                # Drop one indicator to remove the deterministic one-hot sum.
                block = (values[:, None] == np.arange(card - 1)[None, :]).astype(
                    np.float64, copy=False
                )
            dim = int(block.shape[1])
            self.block_indices[name] = np.arange(offset, offset + dim, dtype=np.int64)
            self.block_dims[name] = dim
            offset += dim
            blocks.append(block)

        embedded = np.concatenate(blocks, axis=1)
        centered = embedded - np.mean(embedded, axis=0, keepdims=True)
        self.covariance = (centered.T @ centered) / float(self.n)
        if self.ridge > 0.0:
            self.covariance = self.covariance + self.ridge * np.eye(
                self.covariance.shape[0], dtype=np.float64
            )
        self._joint_cache: Dict[frozenset[str], float] = {frozenset(): 0.0}

    def _joint_log_likelihood(self, variables: Sequence[str]) -> float:
        key = frozenset(variables)
        cached = self._joint_cache.get(key)
        if cached is not None:
            return cached
        ordered = [name for name in self.vtypes if name in key]
        indices = np.concatenate([self.block_indices[name] for name in ordered])
        covariance = self.covariance[np.ix_(indices, indices)]
        # Cholesky is not a sufficient singularity test: round-off can turn a
        # mathematically zero eigenvalue into a tiny positive pivot.  Reject
        # eigenvalues at the standard scale-relative numerical-rank tolerance
        # before computing a log determinant, otherwise deterministic families
        # can receive arbitrarily large finite scores.
        eigenvalues = np.linalg.eigvalsh(covariance)
        spectral_scale = float(np.max(np.abs(eigenvalues)))
        rank_tolerance = (
            np.finfo(np.float64).eps * max(1, int(indices.size)) * spectral_scale
        )
        if (
            not np.isfinite(eigenvalues).all()
            or spectral_scale <= 0.0
            or float(np.min(eigenvalues)) <= rank_tolerance
        ):
            names = ", ".join(ordered)
            raise ValueError(
                "The DG covariance is singular for family variables "
                f"[{names}]. Increase the sample size, reduce category cardinality/"
                "max_parents, choose score_method='native', or explicitly set a "
                "small positive dg_ridge."
            )
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            names = ", ".join(ordered)
            raise ValueError(
                "The DG covariance is singular for family variables "
                f"[{names}]. Increase the sample size, reduce category cardinality/"
                "max_parents, choose score_method='native', or explicitly set a "
                "small positive dg_ridge."
            ) from exc
        logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
        dim = int(indices.size)
        value = -0.5 * self.n * (
            dim * np.log(2.0 * np.pi * np.e) + logdet
        )
        self._joint_cache[key] = float(value)
        return float(value)

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(sorted(parents))
        if child not in self.vtypes:
            raise ValueError(f"Unknown child variable: {child!r}")
        if child in parents:
            raise ValueError("A variable cannot be its own parent.")
        if len(set(parents)) != len(parents):
            raise ValueError("Parent names must be unique.")
        unknown = [name for name in parents if name not in self.vtypes]
        if unknown:
            raise ValueError(f"Unknown parent variables: {unknown}")

        child_dim = int(self.block_dims[child])
        parent_dim = int(sum(self.block_dims[name] for name in parents))
        joint_ll = self._joint_log_likelihood((child, *parents))
        parent_ll = self._joint_log_likelihood(parents)
        bic_penalty = (
            0.5
            * self.penalty_discount
            * child_dim
            * parent_dim
            * np.log(float(self.n))
        )
        score = joint_ll - parent_ll - bic_penalty
        # Only the parent-dependent cross-covariance parameters matter for DAG
        # comparison; child mean/covariance terms are constant across families.
        varying_params = child_dim * parent_dim
        return FamilyScoreResult(
            family="dg",
            log_marginal=float(score),
            k_params=int(varying_params),
            q_total=None,
            q_obs=None,
            details={
                "child_dimension": child_dim,
                "parent_dimension": parent_dim,
                "joint_log_likelihood": float(joint_ll),
                "parent_log_likelihood": float(parent_ll),
                "bic_penalty": float(bic_penalty),
                "penalty_discount": self.penalty_discount,
                "ridge": self.ridge,
            },
        )


class UnifiedFamilyScorer:
    """
    One entry-point: score(child, parents).
    Fast version: converts df->numpy once and never touches pandas in hot loops.
    """

    def __init__(
        self,
        df,
        vtypes: Dict[str, VarType],
        cards: Dict[str, int],
        *,
        alpha: float = 1.0,
        tau: float = 1.0,
        glm_tau: Optional[float] = None,
        clg_ig_shape: float = 2.0,
        clg_ig_scale: float = 1.0,
        standardize_continuous: bool = True,
        score_method: Literal["native", "rank-native", "dg", "clg-bic"] = "native",
        dg_penalty_discount: float = 1.0,
        dg_ridge: float = 0.0,
        use_sparse_bdeu: bool = False,
        validate: bool = True,  # validate discrete value ranges once at init
        cache_scores: bool = True,
        max_cache_size: Optional[int] = None,
    ):
        for option_name, option_value in (
            ("standardize_continuous", standardize_continuous),
            ("use_sparse_bdeu", use_sparse_bdeu),
            ("validate", validate),
            ("cache_scores", cache_scores),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise ValueError(f"{option_name} must be a boolean.")
        if max_cache_size is not None and (
            isinstance(max_cache_size, (bool, np.bool_))
            or not isinstance(max_cache_size, numbers.Integral)
            or int(max_cache_size) < 0
        ):
            raise ValueError("max_cache_size must be a non-negative integer or None.")
        if glm_tau is not None and (
            isinstance(glm_tau, (bool, np.bool_)) or not isinstance(glm_tau, numbers.Real)
        ):
            raise ValueError("glm_tau must be a finite number or None.")

        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        if not isinstance(score_method, str) or score_method not in {
            "native",
            "rank-native",
            "dg",
            "clg-bic",
        }:
            raise ValueError("score_method must be 'native', 'rank-native', 'dg', or 'clg-bic'.")
        self.score_method = score_method
        self.standardize_continuous = bool(standardize_continuous)
        self.continuous_stats: Dict[str, Dict[str, float]] = {}
        self.use_sparse_bdeu = bool(use_sparse_bdeu)
        self.cache_scores = bool(cache_scores)
        self.max_cache_size = None if max_cache_size is None else int(max_cache_size)
        self._score_cache: Dict[Tuple[str, Tuple[str, ...]], FamilyScoreResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # ---- Extract numpy columns once ----
        cols_int: Dict[str, np.ndarray] = {}
        cols_float: Dict[str, np.ndarray] = {}

        is_dataframe = pd is not None and isinstance(df, pd.DataFrame)
        for name, t in self.vtypes.items():
            if t not in ("bin", "cat", "con"):
                raise ValueError(f"Unknown vtype for {name}: {t}")
            col = df[name].to_numpy(copy=False) if is_dataframe else df[name]
            raw = np.asarray(col)
            if raw.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional column.")

            if t in ("bin", "cat"):
                if validate:
                    cols_int[name] = _validated_discrete_to_int(
                        raw, name=name, kind=t, card=int(self.cards.get(name, 0))
                    )
                else:
                    cols_int[name] = _to_int(raw)
                continue

            continuous = _to_float(raw)
            if continuous.size and not np.isfinite(continuous).all():
                raise ValueError(f"{name} is continuous but contains missing or non-finite values.")
            if self.score_method == "rank-native":
                continuous = rank_normal_scores(continuous)
            mean = float(np.mean(continuous)) if continuous.size else 0.0
            scale = float(np.std(continuous, ddof=0)) if continuous.size else 1.0
            if validate and continuous.size and (not np.isfinite(scale) or scale <= 0.0):
                raise ValueError(f"{name} is continuous but has zero variance.")
            safe_scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
            self.continuous_stats[name] = {"mean": mean, "scale": safe_scale}
            if self.standardize_continuous:
                cols_float[name] = (continuous - mean) / safe_scale
            else:
                cols_float[name] = continuous.astype(np.float64, copy=False)

        self.Xi = cols_int
        self.Xf = cols_float

        # ---- One-time validation (optional, but recommended) ----
        if validate:
            for name, t in self.vtypes.items():
                if t == "bin":
                    x = self.Xi[name]
                    if x.size and (x.min() < 0 or x.max() > 1):
                        raise ValueError(f"{name} is bin but has values outside {{0,1}}.")
                elif t == "cat":
                    K = int(self.cards[name])
                    x = self.Xi[name]
                    if x.size and (x.min() < 0 or x.max() >= K):
                        raise ValueError(f"{name} is cat but values outside [0,{K-1}].")
                elif t == "con":
                    x = self.Xf[name]
                    if x.size and not np.isfinite(x).all():
                        raise ValueError(f"{name} is continuous but contains non-finite values.")
                else:
                    raise ValueError(f"Unknown vtype for {name}: {t}")

        # ---- Sub-scorers ----
        self.disc_sparse = DiscreteSparseBDeuScorer(self.Xi, self.cards, alpha=alpha)
        self.disc_std = DiscreteBDeuScorer(self.Xi, self.cards, alpha=alpha)
        self.clg = CLGScorer(
            self.Xi,
            self.Xf,
            self.vtypes,
            self.cards,
            tau=tau,
            ig_shape=clg_ig_shape,
            ig_scale=clg_ig_scale,
        )
        self.clg_bic = (
            ConditionalGaussianBICScorer(
                self.Xi,
                self.Xf,
                self.vtypes,
                self.cards,
            )
            if self.score_method == "clg-bic"
            else None
        )
        effective_glm_tau = tau if glm_tau is None else glm_tau
        self.logit = LogitScorer(
            self.Xi, self.Xf, self.vtypes, self.cards, tau=effective_glm_tau
        )
        self.softmax = SoftmaxScorer(
            self.Xi, self.Xf, self.vtypes, self.cards, tau=effective_glm_tau
        )
        self.dg = (
            DegenerateGaussianScorer(
                self.Xi,
                self.Xf,
                self.vtypes,
                self.cards,
                penalty_discount=dg_penalty_discount,
                ridge=dg_ridge,
            )
            if self.score_method == "dg"
            else None
        )

    def parent_allowed(self, child: str, parent: str) -> bool:
        """Return whether a candidate edge is valid under the selected score."""
        if self.clg_bic is not None:
            return self.clg_bic.parent_allowed(child, parent)
        return True

    def family_allowed(self, child: str, parents: Sequence[str]) -> bool:
        """Return whether a family is structurally valid and locally estimable."""
        if self.clg_bic is None:
            return True
        try:
            self.score(child, parents)
        except UnscorableFamilyError:
            return False
        return True

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(sorted(parents))
        key = (child, parents)
        if self.cache_scores:
            cached = self._score_cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        if self.dg is not None:
            res = self.dg.score(child, parents)
            if self.cache_scores:
                if self.max_cache_size is None or len(self._score_cache) < self.max_cache_size:
                    self._score_cache[key] = res
            return res

        if self.clg_bic is not None:
            res = self.clg_bic.score(child, parents)
            if self.cache_scores:
                if self.max_cache_size is None or len(self._score_cache) < self.max_cache_size:
                    self._score_cache[key] = res
            return res

        t_child = self.vtypes[child]

        Pd = [p for p in parents if self.vtypes[p] in ("bin", "cat")]
        Pc = [p for p in parents if self.vtypes[p] == "con"]

        if t_child == "con":
            res = self.clg.score(child, parents)
        elif t_child == "bin":
            if len(Pc) > 0:
                res = self.logit.score(child, parents)
            else:
                res = self.disc_sparse.score(child, Pd) if self.use_sparse_bdeu else self.disc_std.score(child, Pd)
        elif t_child == "cat":
            if len(Pc) > 0:
                res = self.softmax.score(child, parents)
            else:
                res = self.disc_sparse.score(child, Pd) if self.use_sparse_bdeu else self.disc_std.score(child, Pd)
        else:
            raise ValueError(f"Unknown child type: {t_child}")

        if self.cache_scores:
            if self.max_cache_size is None or len(self._score_cache) < self.max_cache_size:
                self._score_cache[key] = res
        return res

    def clear_score_cache(self) -> None:
        self._score_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def score_cache_info(self) -> Dict[str, int]:
        return {
            "size": int(len(self._score_cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
        }
