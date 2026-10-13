"""Configuration objects shared by candidate construction and sampling."""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Literal, Mapping, Optional, Tuple


# Candidate tables contain already-computed family scores and screened parent
# sets.  Changing any of these fields therefore requires rebuilding the table.
CANDIDATE_BUILD_FIELDS = (
    "screen_k",
    "max_parents",
    "keep_per_child",
    "pair_rescue_anchor_k",
    "pair_rescue_k",
    "parent_pool_max",
    "triplet_rescue_pair_k",
    "triplet_rescue_k",
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

# These settings only control SMC or the downstream graph summary and can be
# changed safely while reusing a candidate table.
SAMPLING_FIELDS = (
    "particles",
    "num_betas",
    "mh_steps",
    "mh_steps_resample",
    "resample_ess_frac",
    "resampling",
    "tempering_path",
    "adaptive_tempering",
    "adaptive_cess_target",
    "adaptive_min_beta_increment",
    "adaptive_max_stages",
    "proposal_swap_weight",
    "proposal_adjacent_swap_weight",
    "proposal_reverse_weight",
    "final_target_snapshot_steps",
    "seed",
    "posterior_threshold",
)


@dataclass(frozen=True)
class DAGConfig:
    """Complexity prior and indegree limit for candidate DAGs."""

    lam: float = 0.0
    edge_penalty: float = 0.0
    max_parents: int | None = None

    def __post_init__(self) -> None:
        _finite_nonnegative("lam", self.lam)
        _finite_nonnegative("edge_penalty", self.edge_penalty)
        if self.max_parents is not None:
            _integer_at_least("max_parents", self.max_parents, 0)


@dataclass(frozen=True)
class OrderSMCConfig:
    """Complete candidate-building and SMC configuration."""

    particles: int = 400
    num_betas: int = 160
    mh_steps: int = 20
    mh_steps_resample: Optional[int] = 50
    resample_ess_frac: float = 0.5
    resampling: Literal["systematic", "none"] = "systematic"
    tempering_path: Literal["geometric", "collapsed_likelihood"] = "geometric"
    adaptive_tempering: bool = True
    adaptive_cess_target: float = 0.9
    adaptive_min_beta_increment: float = 1e-8
    adaptive_max_stages: int = 1000
    proposal_swap_weight: float = 0.75
    proposal_adjacent_swap_weight: float = 0.25
    proposal_reverse_weight: float = 0.0
    final_target_snapshot_steps: Tuple[int, ...] = ()
    screen_k: int = 16
    max_parents: int = 2
    keep_per_child: Optional[int] = None
    pair_rescue_anchor_k: int = 8
    pair_rescue_k: int = 20
    parent_pool_max: Optional[int] = 24
    triplet_rescue_pair_k: int = 0
    triplet_rescue_k: int = 0
    tail_edge_penalty_after: Optional[int] = None
    tail_edge_penalty: Optional[float] = None
    lam: float = 0.75
    edge_penalty: float = 8.0
    alpha: float = 1.0
    tau: float = 3.0
    glm_tau: float = 1.0
    discrete_score: Literal["bdeu", "bds"] = "bdeu"
    clg_ig_shape: float = 2.0
    clg_ig_scale: float = 1.0
    standardize_continuous: bool = True
    score_method: Literal["native", "rank-native", "dg", "clg-bic"] = "native"
    dg_penalty_discount: float = 1.0
    dg_ridge: float = 0.0
    seed: int = 0
    posterior_threshold: float = 0.7

    def __post_init__(self) -> None:
        _integer_at_least("particles", self.particles, 2)
        _integer_at_least("num_betas", self.num_betas, 2)
        _integer_at_least("mh_steps", self.mh_steps, 0)
        if self.mh_steps_resample is not None:
            _integer_at_least("mh_steps_resample", self.mh_steps_resample, 0)
        _finite_interval("resample_ess_frac", self.resample_ess_frac, 0.0, 1.0)
        if self.resampling not in {"systematic", "none"}:
            raise ValueError("resampling must be 'systematic' or 'none'")
        if self.tempering_path not in {"geometric", "collapsed_likelihood"}:
            raise ValueError("tempering_path must be 'geometric' or 'collapsed_likelihood'")
        if not isinstance(self.adaptive_tempering, bool):
            raise ValueError("adaptive_tempering must be a boolean")
        _finite_interval(
            "adaptive_cess_target", self.adaptive_cess_target, 0.0, 1.0, lower_open=True
        )
        if float(self.adaptive_cess_target) >= 1.0:
            raise ValueError("adaptive_cess_target must be in (0, 1)")
        _finite_interval(
            "adaptive_min_beta_increment",
            self.adaptive_min_beta_increment,
            0.0,
            1.0,
            lower_open=True,
        )
        _integer_at_least("adaptive_max_stages", self.adaptive_max_stages, 1)
        _finite_nonnegative("proposal_swap_weight", self.proposal_swap_weight)
        _finite_nonnegative("proposal_adjacent_swap_weight", self.proposal_adjacent_swap_weight)
        _finite_nonnegative("proposal_reverse_weight", self.proposal_reverse_weight)
        if (
            float(self.proposal_swap_weight)
            + float(self.proposal_adjacent_swap_weight)
            + float(self.proposal_reverse_weight)
            <= 0.0
        ):
            raise ValueError("at least one order-proposal weight must be positive")
        try:
            snapshot_steps = tuple(self.final_target_snapshot_steps)
        except TypeError as exc:
            raise ValueError("final_target_snapshot_steps must be a sequence of integers") from exc
        for step in snapshot_steps:
            _integer_at_least("final_target_snapshot_steps", step, 0)
        if tuple(sorted(set(snapshot_steps))) != snapshot_steps:
            raise ValueError("final_target_snapshot_steps must be strictly increasing and unique")
        object.__setattr__(self, "final_target_snapshot_steps", snapshot_steps)
        _integer_at_least("screen_k", self.screen_k, 1)
        _integer_at_least("max_parents", self.max_parents, 0)
        if self.keep_per_child is not None:
            _integer_at_least("keep_per_child", self.keep_per_child, 1)
        _integer_at_least("pair_rescue_anchor_k", self.pair_rescue_anchor_k, 0)
        _integer_at_least("pair_rescue_k", self.pair_rescue_k, 0)
        _integer_at_least("triplet_rescue_pair_k", self.triplet_rescue_pair_k, 0)
        _integer_at_least("triplet_rescue_k", self.triplet_rescue_k, 0)
        if self.parent_pool_max is not None:
            _integer_at_least("parent_pool_max", self.parent_pool_max, 1)
        if self.parent_pool_max is not None and self.parent_pool_max < self.screen_k:
            raise ValueError("parent_pool_max cannot be smaller than screen_k")
        if self.tail_edge_penalty_after is not None:
            _integer_at_least("tail_edge_penalty_after", self.tail_edge_penalty_after, 0)
        if (self.tail_edge_penalty_after is None) != (self.tail_edge_penalty is None):
            raise ValueError(
                "tail_edge_penalty_after and tail_edge_penalty must be supplied together"
            )
        if self.tail_edge_penalty is not None:
            _finite_nonnegative("tail_edge_penalty", self.tail_edge_penalty)
        _finite_nonnegative("lam", self.lam)
        _finite_nonnegative("edge_penalty", self.edge_penalty)
        _finite_positive("alpha", self.alpha)
        _finite_positive("tau", self.tau)
        _finite_positive("glm_tau", self.glm_tau)
        if self.discrete_score not in {"bdeu", "bds"}:
            raise ValueError("discrete_score must be 'bdeu' or 'bds'")
        _finite_positive("clg_ig_shape", self.clg_ig_shape)
        _finite_positive("clg_ig_scale", self.clg_ig_scale)
        if not isinstance(self.standardize_continuous, bool):
            raise ValueError("standardize_continuous must be a boolean")
        if not isinstance(self.score_method, str) or self.score_method not in {
            "native",
            "rank-native",
            "dg",
            "clg-bic",
        }:
            raise ValueError("score_method must be 'native', 'rank-native', 'dg', or 'clg-bic'")
        _finite_positive("dg_penalty_discount", self.dg_penalty_discount)
        _finite_nonnegative("dg_ridge", self.dg_ridge)
        _integer_at_least("seed", self.seed, 0)
        _finite_interval("posterior_threshold", self.posterior_threshold, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def candidate_build_dict(self) -> Dict[str, Any]:
        """Settings frozen into a constructed candidate table."""
        values = self.to_dict()
        return {name: values[name] for name in CANDIDATE_BUILD_FIELDS}

    def sampling_dict(self) -> Dict[str, Any]:
        """Settings that may change without rebuilding candidate scores."""
        values = self.to_dict()
        return {name: values[name] for name in SAMPLING_FIELDS}

    def proposal_weights(self) -> Dict[str, float]:
        """Normalized mixture probabilities for the symmetric order proposals."""
        raw = {
            "swap": float(self.proposal_swap_weight),
            "adjacent_swap": float(self.proposal_adjacent_swap_weight),
            "reverse": float(self.proposal_reverse_weight),
        }
        total = sum(raw.values())
        return {name: value / total for name, value in raw.items()}

    def with_overrides(self, **overrides: Any) -> "OrderSMCConfig":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OrderSMCConfig":
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown OrderSMC configuration keys: {unknown}")
        return cls(**dict(values))

    @classmethod
    def protocol_v2(
        cls,
        *,
        lambda_target: float,
        max_parents: int = 3,
        **overrides: Any,
    ) -> "OrderSMCConfig":
        """Construct the paper protocol's pure edge-count structural prior.

        The historical general-purpose defaults combine a parameter-count
        penalty (``lam``) with an edge penalty. Protocol v2 instead declares
        exactly ``-lambda_target * |P|``. This constructor prevents accidental
        double penalization by forcing ``lam=0`` and mapping the calibrated
        target-specific value to ``edge_penalty``.
        """

        if "lam" in overrides or "edge_penalty" in overrides:
            raise ValueError(
                "protocol_v2 fixes lam=0 and edge_penalty=lambda_target; "
                "pass lambda_target only"
            )
        return cls(
            lam=0.0,
            edge_penalty=lambda_target,
            max_parents=max_parents,
            **overrides,
        )

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "OrderSMCConfig":
        presets: Dict[str, Dict[str, Any]] = {
            "quick": {
                "particles": 80,
                "num_betas": 40,
                "mh_steps": 5,
                "mh_steps_resample": 10,
                "screen_k": 12,
                "pair_rescue_anchor_k": 6,
                "pair_rescue_k": 12,
                "parent_pool_max": 18,
            },
            "standard": {},
            "thorough": {
                "particles": 1000,
                "num_betas": 300,
                "mh_steps": 40,
                "mh_steps_resample": 80,
                "screen_k": 20,
                "pair_rescue_anchor_k": 10,
                "pair_rescue_k": 30,
                "parent_pool_max": 30,
            },
        }
        key = name.strip().lower()
        if key not in presets:
            raise ValueError(f"Unknown preset {name!r}; choose quick, standard, or thorough")
        return cls(**{**presets[key], **{k: v for k, v in overrides.items() if v is not None}})


def candidate_build_config(config: OrderSMCConfig) -> Dict[str, Any]:
    """Return the reproducibility-critical candidate-building settings."""
    return config.candidate_build_dict()


def sampling_config(config: OrderSMCConfig) -> Dict[str, Any]:
    """Return settings that affect SMC and posterior summarization only."""
    return config.sampling_dict()


def _integer_at_least(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _as_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _finite_positive(name: str, value: Any) -> None:
    if _as_finite(name, value) <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _finite_nonnegative(name: str, value: Any) -> None:
    if _as_finite(name, value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _finite_interval(
    name: str,
    value: Any,
    lower: float,
    upper: float,
    *,
    lower_open: bool = False,
) -> None:
    number = _as_finite(name, value)
    lower_ok = number > lower if lower_open else number >= lower
    if not lower_ok or number > upper:
        opening = "(" if lower_open else "["
        raise ValueError(f"{name} must be in {opening}{lower}, {upper}]")
