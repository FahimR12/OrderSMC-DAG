"""Local family scores for discrete, continuous, and mixed Bayesian networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
from scipy.special import gammaln

try:
    import pandas as pd  # optional, only needed for input df type
except Exception:  # pragma: no cover
    pd = None  # type: ignore


VarType = Literal["bin", "cat", "con"]
FamilyType = Literal["disc_bds", "disc_bdeu", "clg", "logit", "softmax"]


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


class CLGScorer:
    """
    Continuous child with (possibly) discrete + continuous parents:
      - partition by discrete parent configs
      - OLS within each observed context, sum maximized Gaussian log-likelihoods
      - k_params sums per observed context
    """

    def __init__(self, data_int: Dict[str, np.ndarray], data_float: Dict[str, np.ndarray],
                 vtypes: Dict[str, VarType], cards: Dict[str, int]):
        self.Xi = data_int
        self.Xf = data_float
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)

    @staticmethod
    def _ols_loglik(y: np.ndarray, Z: np.ndarray) -> float:
        n = int(y.shape[0])
        if n <= 0:
            return 0.0
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        rss = float(np.dot(resid, resid))
        rss = max(rss, 1e-12)
        sigma2 = rss / n
        return float(-0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2)))

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        y = self.Xf[child]
        n = int(y.shape[0])

        Pd = [p for p in parents if self.vtypes[p] in ("bin", "cat")]
        Pc = [p for p in parents if self.vtypes[p] == "con"]

        p = int(len(Pc))
        k_per_context = (p + 1) + 1  # coefficients (incl intercept) + variance

        # no discrete parents: single regression
        if len(Pd) == 0:
            Z = np.empty((n, p + 1), dtype=np.float64)
            Z[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Z[:, j] = self.Xf[name]
            ll = self._ols_loglik(y, Z)
            return FamilyScoreResult("clg", float(ll), k_params=int(k_per_context), details={"contexts": 1})

        # discrete contexts (observed only)
        parent_cols = [self.Xi[pn] for pn in Pd]
        parent_cards = [int(self.cards[pn]) for pn in Pd]
        idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)
        uniq, inv = np.unique(idx, return_inverse=True)
        q_obs = int(uniq.shape[0])

        order, starts, ends = _group_slices(inv.astype(np.int64, copy=False))

        ll_sum = 0.0
        k_sum = 0
        contexts = 0

        for a, b in zip(starts, ends):
            inds = order[a:b]
            n_c = int(inds.shape[0])
            if n_c == 0:
                continue
            contexts += 1
            yc = y[inds]

            Zc = np.empty((n_c, p + 1), dtype=np.float64)
            Zc[:, 0] = 1.0
            for j, name in enumerate(Pc, start=1):
                Zc[:, j] = self.Xf[name][inds]

            ll_sum += self._ols_loglik(yc, Zc)
            k_sum += k_per_context

        return FamilyScoreResult(
            "clg",
            float(ll_sum),
            k_params=int(k_sum),
            details={"contexts": int(contexts), "q_total_disc": int(q_total), "q_obs_disc": int(q_obs)},
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
        Prior: slopes ~ N(0, tau^2 I), intercept unregularized (but we add tiny jitter for stability).
        """
        n, p = Z.shape
        beta = np.zeros(p, dtype=np.float64)

        ridge = np.zeros(p, dtype=np.float64)
        ridge[1:] = 1.0 / (self.tau ** 2)

        I = np.eye(p, dtype=np.float64)

        for _ in range(self.max_iter):
            eta = Z @ beta
            prob = self._sigmoid(eta)
            W = prob * (1.0 - prob) + 1e-12

            g = Z.T @ (prob - y) + ridge * beta
            H = (Z.T * W) @ Z
            H[np.diag_indices(p)] += ridge
            H[np.diag_indices(p)] += 1e-12  # jitter to help PD

            damp = 1e-6
            solved = False
            for _k in range(8):
                try:
                    step = np.linalg.solve(H + damp * I, g)
                    solved = True
                    break
                except np.linalg.LinAlgError:
                    damp *= 10.0
            if not solved:
                step = np.linalg.lstsq(H + damp * I, g, rcond=None)[0]

            beta_new = beta - step
            if float(np.linalg.norm(step)) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        eta = Z @ beta
        prob = self._sigmoid(eta)
        loglik = float(np.sum(y * np.log(prob + 1e-12) + (1.0 - y) * np.log(1.0 - prob + 1e-12)))

        b = beta[1:]
        p_slopes = int(b.shape[0])
        logprior = float(
            -0.5 * (np.dot(b, b) / (self.tau ** 2) + p_slopes * np.log(2.0 * np.pi * (self.tau ** 2)))
        )

        W = prob * (1.0 - prob) + 1e-12
        H = (Z.T * W) @ Z
        H[np.diag_indices(p)] += ridge
        H[np.diag_indices(p)] += 1e-12

        # logdet via cholesky with fallback jitter
        jitter = 1e-10
        for _ in range(6):
            try:
                L = np.linalg.cholesky(H + jitter * I)
                logdet = float(2.0 * np.sum(np.log(np.diag(L))))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            sign, ld = np.linalg.slogdet(H + jitter * I)
            logdet = float(ld)

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
            return FamilyScoreResult("logit", float(log_marg), k_params=int(Kdim), details={"contexts": 1})

        # observed discrete contexts only
        parent_cols = [self.Xi[pn] for pn in Pd]
        parent_cards = [int(self.cards[pn]) for pn in Pd]
        idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)
        uniq, inv = np.unique(idx, return_inverse=True)
        q_obs = int(uniq.shape[0])

        order, starts, ends = _group_slices(inv.astype(np.int64, copy=False))

        total = 0.0
        k_sum = 0
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
            k_sum += Kdim

        return FamilyScoreResult(
            "logit",
            float(total),
            k_params=int(k_sum),
            details={"contexts": int(contexts), "q_total_disc": int(q_total), "q_obs_disc": int(q_obs)},
        )


class SoftmaxScorer:
    """
    Categorical child with continuous parents (+ optional discrete contexts).
    Laplace approx around MAP for multinomial logistic regression (baseline class).
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

    @staticmethod
    def _softmax(A: np.ndarray) -> np.ndarray:
        A = A - A.max(axis=1, keepdims=True)
        E = np.exp(A)
        return E / E.sum(axis=1, keepdims=True)

    def _fit_map(self, y: np.ndarray, Z: np.ndarray, K: int) -> Tuple[np.ndarray, float, float]:
        """
        y in {0..K-1}, baseline is class K-1 fixed to 0.
        Parameters: B for classes 0..K-2, each has p features (incl intercept).
        Prior: slopes regularized, intercepts unregularized (tiny jitter for stability).
        """
        n, p = Z.shape
        Km1 = K - 1
        B = np.zeros((Km1, p), dtype=np.float64)

        ridge = np.zeros_like(B)
        ridge[:, 1:] = 1.0 / (self.tau ** 2)

        Y = np.zeros((n, K), dtype=np.float64)
        Y[np.arange(n), y.astype(int)] = 1.0

        dim = Km1 * p
        I = np.eye(dim, dtype=np.float64)

        for _ in range(self.max_iter):
            logits = np.zeros((n, K), dtype=np.float64)
            logits[:, :Km1] = Z @ B.T
            P = self._softmax(logits)

            G = np.zeros_like(B)
            for k in range(Km1):
                G[k] = Z.T @ (P[:, k] - Y[:, k]) + ridge[k] * B[k]
            g = G.reshape(-1)

            H = np.zeros((dim, dim), dtype=np.float64)
            for a in range(Km1):
                Za = Z
                Pa = P[:, a]
                for b in range(Km1):
                    w = Pa * ((1.0 if a == b else 0.0) - P[:, b])
                    Hab = (Za.T * w) @ Za
                    if a == b:
                        Hab = Hab.copy()
                        Hab[np.diag_indices(p)] += ridge[a]
                        Hab[np.diag_indices(p)] += 1e-12
                    H[a*p:(a+1)*p, b*p:(b+1)*p] = Hab

            damp = 1e-6
            solved = False
            for _k in range(8):
                try:
                    step = np.linalg.solve(H + damp * I, g)
                    solved = True
                    break
                except np.linalg.LinAlgError:
                    damp *= 10.0
            if not solved:
                step = np.linalg.lstsq(H + damp * I, g, rcond=None)[0]

            B_new = (B.reshape(-1) - step).reshape(Km1, p)
            if float(np.linalg.norm(step)) < self.tol:
                B = B_new
                break
            B = B_new

        logits = np.zeros((n, K), dtype=np.float64)
        logits[:, :Km1] = Z @ B.T
        P = self._softmax(logits)
        loglik = float(np.sum(np.log(P[np.arange(n), y.astype(int)] + 1e-12)))

        b_slopes = B[:, 1:].reshape(-1)
        p_slopes = int(b_slopes.shape[0])
        logprior = float(
            -0.5 * (np.dot(b_slopes, b_slopes) / (self.tau ** 2) + p_slopes * np.log(2.0 * np.pi * (self.tau ** 2)))
        )

        # logdet Hessian at mode
        dim = Km1 * p
        H = np.zeros((dim, dim), dtype=np.float64)
        for a in range(Km1):
            Pa = P[:, a]
            for b in range(Km1):
                w = Pa * ((1.0 if a == b else 0.0) - P[:, b])
                Hab = (Z.T * w) @ Z
                if a == b:
                    Hab = Hab.copy()
                    diag = np.zeros(p, dtype=np.float64)
                    diag[1:] = 1.0 / (self.tau ** 2)
                    Hab[np.diag_indices(p)] += diag
                    Hab[np.diag_indices(p)] += 1e-12
                H[a*p:(a+1)*p, b*p:(b+1)*p] = Hab

        # jittered cholesky
        jitter = 1e-10
        for _ in range(6):
            try:
                L = np.linalg.cholesky(H + jitter * np.eye(dim, dtype=np.float64))
                logdet = float(2.0 * np.sum(np.log(np.diag(L))))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            sign, ld = np.linalg.slogdet(H + jitter * np.eye(dim, dtype=np.float64))
            logdet = float(ld)

        return B, float(logdet), float(loglik + logprior)

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(parents)
        K = int(self.cards[child])
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
            return FamilyScoreResult("softmax", float(log_marg), k_params=int(Kdim), details={"contexts": 1})

        parent_cols = [self.Xi[pn] for pn in Pd]
        parent_cards = [int(self.cards[pn]) for pn in Pd]
        idx, q_total = _mixed_radix_index(parent_cols, parent_cards, n)
        uniq, inv = np.unique(idx, return_inverse=True)
        q_obs = int(uniq.shape[0])

        order, starts, ends = _group_slices(inv.astype(np.int64, copy=False))

        total = 0.0
        k_sum = 0
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
            k_sum += Kdim

        return FamilyScoreResult(
            "softmax",
            float(total),
            k_params=int(k_sum),
            details={"contexts": int(contexts), "q_total_disc": int(q_total), "q_obs_disc": int(q_obs)},
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
        use_sparse_bdeu: bool = True,
        validate: bool = True,  # validate discrete value ranges once at init
        cache_scores: bool = True,
        max_cache_size: Optional[int] = None,
    ):
        self.vtypes = dict(vtypes)
        self.cards = dict(cards)
        self.use_sparse_bdeu = bool(use_sparse_bdeu)
        self.cache_scores = bool(cache_scores)
        self.max_cache_size = None if max_cache_size is None else int(max_cache_size)
        self._score_cache: Dict[Tuple[str, Tuple[str, ...]], FamilyScoreResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # ---- Extract numpy columns once ----
        cols_int: Dict[str, np.ndarray] = {}
        cols_float: Dict[str, np.ndarray] = {}

        # If df is pandas, use to_numpy() where possible
        if pd is not None and isinstance(df, pd.DataFrame):
            for name, t in self.vtypes.items():
                if t in ("bin", "cat"):
                    cols_int[name] = _to_int(df[name].to_numpy(copy=False))
                else:
                    cols_float[name] = _to_float(df[name].to_numpy(copy=False))
        else:
            # assume dict-like / np arrays
            for name, t in self.vtypes.items():
                col = df[name]
                if t in ("bin", "cat"):
                    cols_int[name] = _to_int(col)
                else:
                    cols_float[name] = _to_float(col)

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
                    # no range check
                    pass
                else:
                    raise ValueError(f"Unknown vtype for {name}: {t}")

        # ---- Sub-scorers ----
        self.disc_sparse = DiscreteSparseBDeuScorer(self.Xi, self.cards, alpha=alpha)
        self.disc_std = DiscreteBDeuScorer(self.Xi, self.cards, alpha=alpha)
        self.clg = CLGScorer(self.Xi, self.Xf, self.vtypes, self.cards)
        self.logit = LogitScorer(self.Xi, self.Xf, self.vtypes, self.cards, tau=tau)
        self.softmax = SoftmaxScorer(self.Xi, self.Xf, self.vtypes, self.cards, tau=tau)

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        parents = tuple(sorted(parents))
        key = (child, parents)
        if self.cache_scores:
            cached = self._score_cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

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
