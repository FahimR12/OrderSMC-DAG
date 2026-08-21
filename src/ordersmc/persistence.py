"""Portable configuration, candidate-table, and result persistence."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .config import OrderSMCConfig
from .core import OrderCandidateTable, OrderSMCResult

FORMAT_VERSION = 1


def load_mapping(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            values = json.load(handle)
        elif source.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML support requires PyYAML; install ordersmc-dag with its normal dependencies") from exc
            values = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported file type {source.suffix!r}; use JSON or YAML")
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a mapping at its top level")
    return values


def save_mapping(values: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        if destination.suffix.lower() == ".json":
            json.dump(values, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        elif destination.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML support requires PyYAML; install ordersmc-dag with its normal dependencies") from exc
            yaml.safe_dump(dict(values), handle, sort_keys=False)
        else:
            raise ValueError(f"Unsupported file type {destination.suffix!r}; use JSON or YAML")


def load_config(path: str | Path | None = None, *, preset: str = "standard") -> OrderSMCConfig:
    base = OrderSMCConfig.preset(preset)
    if path is None:
        return base
    values = load_mapping(path)
    if "ordersmc" in values:
        values = values["ordersmc"]
    return OrderSMCConfig.from_mapping({**base.to_dict(), **values})


def save_candidate_table(table: OrderCandidateTable, path: str | Path) -> Path:
    """Save an expensive candidate table for later SMC runs."""
    destination = Path(path)
    if destination.suffix not in {".gz", ".gzip"}:
        destination = destination.with_suffix(destination.suffix + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "ordersmc-candidate-table", "version": FORMAT_VERSION, "table": table}
    with gzip.open(destination, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_candidate_table(path: str | Path) -> OrderCandidateTable:
    """Load a table created by :func:`save_candidate_table`.

    Candidate files use pickle internally and therefore must only be loaded from
    trusted sources.
    """
    with gzip.open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != "ordersmc-candidate-table":
        raise ValueError("Not an OrderSMC-DAG candidate table")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported candidate-table version {payload.get('version')}")
    table = payload.get("table")
    if not isinstance(table, OrderCandidateTable):
        raise ValueError("Candidate-table payload is invalid")
    if not (len(table.nodes) == len(table.parent_sets) == len(table.loglik) == len(table.logprior)):
        raise ValueError("Candidate-table arrays have inconsistent lengths")
    return table


def save_result(result: OrderSMCResult, path: str | Path, *, threshold: float = 0.7) -> None:
    """Write a complete, inspectable analysis result directory."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    nodes = list(result.run_info.get("nodes", []))

    pd.DataFrame(result.best_edges, columns=["Parent", "Child"]).to_csv(
        destination / "best_edges.csv", index=False
    )
    result.edge_table().to_csv(destination / "edge_confidence.csv", index=False)
    pd.DataFrame(result.history).to_csv(destination / "smc_diagnostics.csv", index=False)
    pd.DataFrame(sorted(result.posterior_dag(threshold).edges()), columns=["Parent", "Child"]).to_csv(
        destination / "posterior_dag.csv", index=False
    )
    np.save(destination / "orders.npy", np.asarray(result.orders, dtype=np.int32))
    np.save(destination / "weights.npy", result.weights)
    np.save(destination / "log_weights.npy", result.logw)
    np.save(destination / "best_order.npy", result.best_order)
    save_mapping(result.summary(), destination / "summary.json")
    save_mapping(
        {**result.run_info, "posterior_threshold": threshold, "format_version": FORMAT_VERSION},
        destination / "run.json",
    )


def load_edge_probability_map(path: str | Path) -> Dict[tuple[str, str], float]:
    source = Path(path)
    if source.is_dir():
        source = source / "edge_confidence.csv"
    frame = pd.read_csv(source)
    probability_column = "PosteriorProbability" if "PosteriorProbability" in frame else "MeanPosterior"
    return {
        (str(row.Parent), str(row.Child)): float(getattr(row, probability_column))
        for row in frame.itertuples(index=False)
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
