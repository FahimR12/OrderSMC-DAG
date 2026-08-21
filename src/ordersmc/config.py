"""Configuration objects shared by candidate construction and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class DAGConfig:
    """Complexity prior and indegree limit for candidate DAGs."""

    lam: float = 0.0
    edge_penalty: float = 0.0
    max_parents: int | None = None


@dataclass(frozen=True)
class OrderSMCConfig:
    """Complete candidate-building and SMC configuration."""

    particles: int = 400
    num_betas: int = 160
    mh_steps: int = 20
    mh_steps_resample: Optional[int] = 50
    resample_ess_frac: float = 0.5
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
    seed: int = 0
    posterior_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.particles < 2:
            raise ValueError("particles must be at least 2")
        if self.num_betas < 2:
            raise ValueError("num_betas must be at least 2")
        if self.mh_steps < 0 or (self.mh_steps_resample is not None and self.mh_steps_resample < 0):
            raise ValueError("MH step counts cannot be negative")
        if not 0.0 < self.resample_ess_frac <= 1.0:
            raise ValueError("resample_ess_frac must be in (0, 1]")
        if self.screen_k < 1:
            raise ValueError("screen_k must be positive")
        if self.max_parents < 0:
            raise ValueError("max_parents cannot be negative")
        if self.keep_per_child is not None and self.keep_per_child < 1:
            raise ValueError("keep_per_child must be positive when supplied")
        if self.parent_pool_max is not None and self.parent_pool_max < self.screen_k:
            raise ValueError("parent_pool_max cannot be smaller than screen_k")
        if self.alpha <= 0.0 or self.tau <= 0.0:
            raise ValueError("alpha and tau must be positive")
        if not 0.0 <= self.posterior_threshold <= 1.0:
            raise ValueError("posterior_threshold must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "OrderSMCConfig":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OrderSMCConfig":
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown OrderSMC configuration keys: {unknown}")
        return cls(**dict(values))

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
