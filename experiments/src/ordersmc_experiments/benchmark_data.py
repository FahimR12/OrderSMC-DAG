"""Reproducible data adapters for recognized mixed-network benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import networkx as nx
import pandas as pd

from .bnlearn_adapter import bnlearn_version, rscript_path


CLGAUSSIAN_MODEL = "[A][B][C][H][D|A:H][F|B:C][E|B:D][G|A:D:E:F]"
CLGAUSSIAN_EDGES = (
    ("A", "D"),
    ("A", "G"),
    ("B", "E"),
    ("B", "F"),
    ("C", "F"),
    ("D", "E"),
    ("D", "G"),
    ("E", "G"),
    ("F", "G"),
    ("H", "D"),
)
CLGAUSSIAN_TYPES = {
    "A": "bin",
    "B": "cat",
    "C": "cat",
    "D": "con",
    "E": "con",
    "F": "bin",
    "G": "con",
    "H": "con",
}
CLGAUSSIAN_CARDINALITIES = {"A": 2, "B": 3, "C": 4, "F": 2}
GENERATOR_SCRIPT = Path(__file__).resolve().parents[2] / "adapters" / "generate_clgaussian.R"


@dataclass(frozen=True)
class BenchmarkDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    dag: nx.DiGraph
    variable_types: dict[str, str]
    cardinalities: dict[str, int]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _r_seed(seed: int) -> int:
    return int(seed) % 2_147_483_646 + 1


def write_clgaussian_artifact(
    destination: Path,
    *,
    n_train: int,
    dataset_seed: int,
    dataset_id: str,
) -> BenchmarkDataset:
    """Fit the published bnlearn test DAG once, then simulate an independent replicate."""

    destination.mkdir(parents=True, exist_ok=False)
    n_test = max(100, (int(n_train) + 4) // 5)
    train_seed = _r_seed(dataset_seed)
    test_seed = _r_seed(dataset_seed + 1_000_003)
    executable = rscript_path()
    completed = subprocess.run(
        [
            str(executable),
            "--vanilla",
            str(GENERATOR_SCRIPT),
            str(int(n_train)),
            str(n_test),
            str(train_seed),
            str(test_seed),
            str(destination / "train.csv"),
            str(destination / "test.csv"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"clgaussian.test generation failed: {message}")
    train = pd.read_csv(destination / "train.csv")
    test = pd.read_csv(destination / "test.csv")
    nodes = list(CLGAUSSIAN_TYPES)
    if list(train.columns) != nodes or list(test.columns) != nodes:
        raise RuntimeError("clgaussian.test generator returned unexpected columns")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(CLGAUSSIAN_EDGES)
    adjacency = nx.to_pandas_adjacency(graph, nodelist=nodes, dtype=int)
    adjacency.to_csv(destination / "adjacency.csv", lineterminator="\n")
    manifest: dict[str, Any] = {
        "schema": "ordersmc-clgaussian-benchmark-v1",
        "dataset_id": dataset_id,
        "source": "bnlearn::clgaussian.test",
        "source_model": CLGAUSSIAN_MODEL,
        "bnlearn_version": bnlearn_version(executable),
        "n_train": int(n_train),
        "n_test": n_test,
        "dataset_seed": int(dataset_seed),
        "r_train_seed": train_seed,
        "r_test_seed": test_seed,
        "variable_types": CLGAUSSIAN_TYPES,
        "cardinalities": CLGAUSSIAN_CARDINALITIES,
        "edges": [list(edge) for edge in CLGAUSSIAN_EDGES],
        "artifacts": {
            filename: _sha256(destination / filename)
            for filename in ("train.csv", "test.csv", "adjacency.csv")
        },
    }
    manifest["artifact_fingerprint_sha256"] = _canonical_sha(manifest)
    (destination / "generator_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BenchmarkDataset(
        train=train,
        test=test,
        dag=graph,
        variable_types=dict(CLGAUSSIAN_TYPES),
        cardinalities=dict(CLGAUSSIAN_CARDINALITIES),
        manifest=manifest,
    )


def load_clgaussian_artifact(source: Path) -> BenchmarkDataset:
    manifest = json.loads((source / "generator_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "ordersmc-clgaussian-benchmark-v1":
        raise ValueError("not a supported clgaussian.test artifact")
    claimed = manifest.get("artifact_fingerprint_sha256")
    body = dict(manifest)
    body.pop("artifact_fingerprint_sha256", None)
    if claimed != _canonical_sha(body):
        raise RuntimeError("clgaussian.test manifest fingerprint mismatch")
    for filename, digest in manifest["artifacts"].items():
        if _sha256(source / filename) != digest:
            raise RuntimeError(f"clgaussian.test artifact hash mismatch: {filename}")
    train = pd.read_csv(source / "train.csv")
    test = pd.read_csv(source / "test.csv")
    graph = nx.DiGraph()
    graph.add_nodes_from(CLGAUSSIAN_TYPES)
    graph.add_edges_from(tuple(tuple(edge) for edge in manifest["edges"]))
    return BenchmarkDataset(
        train=train,
        test=test,
        dag=graph,
        variable_types=dict(manifest["variable_types"]),
        cardinalities={key: int(value) for key, value in manifest["cardinalities"].items()},
        manifest=manifest,
    )


def load_experiment_artifact(source: Path):
    manifest = json.loads((source / "generator_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") == "ordersmc-clgaussian-benchmark-v1":
        return load_clgaussian_artifact(source)
    if manifest.get("schema") == "ordersmc-metadata-calibrated-dag-v1":
        from .metadata_profiles import load_metadata_dag_artifact

        return load_metadata_dag_artifact(source)
    from .generator import replay_mixed_dag_artifact

    return replay_mixed_dag_artifact(source)
