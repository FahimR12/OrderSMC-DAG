"""Stable, namespaced seed derivation for confirmatory experiments."""

from __future__ import annotations

import hashlib
import numbers
import re
from dataclasses import dataclass

import numpy as np


_TOKEN = re.compile(r"[a-z0-9][a-z0-9._/-]*\Z", flags=re.ASCII)
_SEED_INPUT = re.compile(
    r"ordersmc-protocol-v2\|[0-9]{20}\|"
    r"[a-z0-9][a-z0-9._/-]*\|[a-z0-9][a-z0-9._/-]*\|"
    r"[0-9]{6}\|[a-z0-9][a-z0-9._/-]*\|[0-9]{6}\Z",
    flags=re.ASCII,
)


@dataclass(frozen=True)
class SeedRecord:
    input_string: str
    seed: int
    sha256: str
    bit_generator: str = "PCG64DXSM"
    numpy_version: str = np.__version__

    def _validate(self) -> None:
        if (
            not isinstance(self.input_string, str)
            or _SEED_INPUT.fullmatch(self.input_string) is None
        ):
            raise ValueError(
                "seed input_string does not match the stable protocol-v2 schema"
            )
        digest = hashlib.sha256(self.input_string.encode("utf-8")).digest()
        if self.sha256 != digest.hex():
            raise ValueError("seed SHA-256 does not match input_string")
        expected_seed = int.from_bytes(digest[:8], "big", signed=False)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, numbers.Integral)
            or int(self.seed) != expected_seed
        ):
            raise ValueError("seed integer does not match the SHA-256 prefix")
        if self.bit_generator != "PCG64DXSM":
            raise ValueError("protocol seed records require PCG64DXSM")
        if self.numpy_version != np.__version__:
            raise ValueError(
                "seed record NumPy version does not match the executing environment"
            )

    def __post_init__(self) -> None:
        self._validate()

    def generator(self) -> np.random.Generator:
        self._validate()
        return np.random.Generator(np.random.PCG64DXSM(self.seed))


def protocol_seed_record(
    master_seed: int,
    *,
    experiment: str,
    cell: str,
    replicate: int,
    component: str,
    attempt: int = 0,
) -> SeedRecord:
    """Create a stable namespaced seed record.

    Integer widths are part of the frozen schema: master seeds use 20 decimal
    digits and replicate/attempt identifiers use six. Experiment, cell, and
    component identifiers must already be canonical lowercase tokens.
    """

    if isinstance(master_seed, bool) or not isinstance(master_seed, numbers.Integral):
        raise TypeError("master_seed must be an integer")
    if not 0 <= int(master_seed) < 10**20:
        raise ValueError("master_seed must lie in [0, 10^20)")
    for name, value in (("replicate", replicate), ("attempt", attempt)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= int(value) < 10**6:
            raise ValueError(f"{name} must lie in [0, 10^6)")
    for name, value in (
        ("experiment", experiment),
        ("cell", cell),
        ("component", component),
    ):
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip().lower()
            or not value.isascii()
            or _TOKEN.fullmatch(value) is None
        ):
            raise ValueError(f"{name} must be a non-empty canonical lowercase token")
        if "|" in value:
            raise ValueError(f"{name} may not contain the seed-field separator '|'")
    input_string = (
        f"ordersmc-protocol-v2|{int(master_seed):020d}|{experiment}|{cell}|"
        f"{int(replicate):06d}|{component}|{int(attempt):06d}"
    )
    digest = hashlib.sha256(input_string.encode("utf-8")).digest()
    return SeedRecord(
        input_string=input_string,
        seed=int.from_bytes(digest[:8], "big", signed=False),
        sha256=digest.hex(),
    )


def derive_seed(master_seed: int, namespace: str, index: int = 0) -> int:
    """Derive a stable unsigned 64-bit seed without Python's salted hash.

    Separate namespaces ensure that changing, for example, the sampler seed
    list does not silently alter graph generation or coefficient draws.
    """

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    canonical_namespace = namespace.strip().lower().replace("_", "-")
    return protocol_seed_record(
        master_seed,
        experiment="adhoc",
        cell="default",
        replicate=0,
        component=canonical_namespace,
        attempt=index,
    ).seed
