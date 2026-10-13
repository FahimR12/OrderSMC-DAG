"""Machine-readable run records required before confirmatory computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
from typing import Any, Mapping, Sequence

import pandas as pd


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    digest.update("\0".join(map(str, data.columns)).encode())
    digest.update("\0".join(map(str, data.dtypes)).encode())
    digest.update(f"{data.shape[0]}x{data.shape[1]}".encode())
    return digest.hexdigest()


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass
class RunManifest:
    experiment_id: str
    dataset_id: str
    method: str
    dataset_seed: int
    sampler_seed: int
    configuration: Mapping[str, Any]
    model_version: str
    model_source_sha256: str
    dataset_sha256: str
    schema: str = "ordersmc-run-record-v2"
    protocol_version: str = "2.0"
    run_id: str | None = None
    cell_id: str | None = None
    replicate_id: int | None = None
    fold_id: str | None = None
    target_label: str | None = None
    target_semantics: str | None = None
    master_seed: int | None = None
    seed_records: Mapping[str, Any] = field(default_factory=dict)
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    score_record: Mapping[str, Any] = field(default_factory=dict)
    candidate_record: Mapping[str, Any] = field(default_factory=dict)
    resource_record: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_record: Mapping[str, Any] = field(default_factory=dict)
    output_record: Mapping[str, Any] = field(default_factory=dict)
    source_commit: str | None = None
    checkpoint_path: str | None = None
    candidate_table_sha256: str | None = None
    backend: str = "local"
    mpi_ranks: int = 1
    status: str = "created"
    failure_type: str | None = None
    failure_message: str | None = None
    failure_traceback: str | None = None
    failure_traceback_sha256: str | None = None
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_utc: str | None = None
    wall_seconds: float | None = None
    cpu_seconds: float | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("experiment_id", "dataset_id", "method", "model_version"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.dataset_seed < 0 or self.sampler_seed < 0 or self.mpi_ranks < 1:
            raise ValueError("seeds must be non-negative and mpi_ranks positive")
        if self.master_seed is not None and self.master_seed < 0:
            raise ValueError("master_seed must be non-negative when supplied")
        if self.replicate_id is not None and self.replicate_id < 0:
            raise ValueError("replicate_id must be non-negative when supplied")
        for name in ("model_source_sha256", "dataset_sha256"):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        dataset_id: str,
        method: str,
        dataset_seed: int,
        sampler_seed: int,
        configuration: Mapping[str, Any],
        model_version: str,
        model_source_sha256: str,
        dataset_sha256: str,
        candidate_table_sha256: str | None = None,
        backend: str = "local",
        mpi_ranks: int = 1,
        protocol_version: str = "2.0",
        run_id: str | None = None,
        cell_id: str | None = None,
        replicate_id: int | None = None,
        fold_id: str | None = None,
        target_label: str | None = None,
        target_semantics: str | None = None,
        master_seed: int | None = None,
        seed_records: Mapping[str, Any] | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        score_record: Mapping[str, Any] | None = None,
        candidate_record: Mapping[str, Any] | None = None,
        resource_record: Mapping[str, Any] | None = None,
        source_commit: str | None = None,
    ) -> "RunManifest":
        thread_variables = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "packages": _package_versions(
                ["numpy", "scipy", "pandas", "networkx", "scikit-learn", "ordersmc-dag"]
            ),
            "thread_environment": thread_variables,
        }
        return cls(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            method=method,
            dataset_seed=int(dataset_seed),
            sampler_seed=int(sampler_seed),
            configuration=dict(configuration),
            model_version=model_version,
            model_source_sha256=model_source_sha256,
            dataset_sha256=dataset_sha256,
            protocol_version=protocol_version,
            run_id=run_id,
            cell_id=cell_id,
            replicate_id=replicate_id,
            fold_id=fold_id,
            target_label=target_label,
            target_semantics=target_semantics,
            master_seed=master_seed,
            seed_records={} if seed_records is None else dict(seed_records),
            artifact_hashes={} if artifact_hashes is None else dict(artifact_hashes),
            score_record={} if score_record is None else dict(score_record),
            candidate_record={} if candidate_record is None else dict(candidate_record),
            resource_record={} if resource_record is None else dict(resource_record),
            source_commit=source_commit,
            candidate_table_sha256=candidate_table_sha256,
            backend=backend,
            mpi_ranks=int(mpi_ranks),
            environment=environment,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_for_execution(self) -> None:
        """Reject an underspecified record before scientific code starts."""

        missing = []
        for name in (
            "run_id",
            "cell_id",
            "target_label",
            "target_semantics",
            "source_commit",
        ):
            if not getattr(self, name):
                missing.append(name)
        if self.replicate_id is None:
            missing.append("replicate_id")
        if self.master_seed is None:
            missing.append("master_seed")
        for name in (
            "configuration",
            "seed_records",
            "artifact_hashes",
            "score_record",
            "candidate_record",
            "resource_record",
        ):
            if not getattr(self, name):
                missing.append(name)
        if missing:
            raise ValueError(
                "run manifest is not execution-ready; missing: "
                + ", ".join(missing)
            )
        allowed_targets = {"order", "structure", "native_external", "point_estimate"}
        if self.target_label not in allowed_targets:
            raise ValueError(
                "target_label must be one of: " + ", ".join(sorted(allowed_targets))
            )

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
