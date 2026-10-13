"""Portable configuration, candidate-table, and result persistence."""

from __future__ import annotations

import gzip
import hashlib
from itertools import combinations
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .config import OrderSMCConfig
from .core import (
    OrderCandidateTable,
    OrderSMCResult,
    OrderSMCState,
    _candidate_table_signature,
    _validate_candidate_table as _validate_core_candidate_table,
)

CANDIDATE_FORMAT_VERSION = 2
LEGACY_CANDIDATE_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 2
SMC_CHECKPOINT_FORMAT_VERSION = 1
# Backward-compatible name used by external candidate-table tooling.
FORMAT_VERSION = CANDIDATE_FORMAT_VERSION

_CANONICAL_CANDIDATE_MAGIC = b"ordersmc-candidate-table-canonical-v2\n"
_SERIALIZED_CANDIDATE_MAGIC = b"ordersmc-candidate-table-serialized-v2\n"


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
    """Save a candidate table using the deterministic canonical v2 format.

    The gzip header has a frozen timestamp and no source filename.  Therefore
    two saves of the same canonical table are byte-identical, in addition to
    sharing the content address returned by
    :func:`candidate_table_content_sha256`.
    """
    _validate_candidate_table(table)
    if not getattr(table, "provenance", None):
        warnings.warn(
            "Saving a candidate table without provenance; safe data/config reuse cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
    destination = Path(path)
    if destination.suffix not in {".gz", ".gzip"}:
        destination = destination.with_suffix(destination.suffix + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized_candidate_table_bytes(table)
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as handle:
            handle.write(payload)
    return destination


def load_candidate_table(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> OrderCandidateTable:
    """Load a table created by :func:`save_candidate_table`.

    Canonical v2 files contain JSON and are safe to inspect.  Legacy v1 files
    used pickle and must only be loaded from trusted sources.
    """
    with gzip.open(Path(path), "rb") as handle:
        raw_payload = handle.read()
    if raw_payload.startswith(_SERIALIZED_CANDIDATE_MAGIC):
        table = _table_from_serialized_bytes(raw_payload)
        if _serialized_candidate_table_bytes(table) != raw_payload:
            raise ValueError("Candidate-table v2 payload is not canonically encoded")
    elif raw_payload.startswith(_CANONICAL_CANDIDATE_MAGIC):
        table = _table_from_canonical_bytes(raw_payload)
        # Reject semantically equivalent but noncanonical encodings: artifact
        # byte parity is part of the protocol acceptance criterion.
        if canonical_candidate_table_bytes(table) != raw_payload:
            raise ValueError("Candidate-table v2 payload is not canonically encoded")
    else:
        try:
            payload = pickle.loads(raw_payload)
        except Exception as exc:
            raise ValueError("Not an OrderSMC-DAG candidate table") from exc
        if not isinstance(payload, dict) or payload.get("format") != "ordersmc-candidate-table":
            raise ValueError("Not an OrderSMC-DAG candidate table")
        if payload.get("version") != LEGACY_CANDIDATE_FORMAT_VERSION:
            raise ValueError(f"Unsupported candidate-table version {payload.get('version')}")
        table = payload.get("table")
        if not isinstance(table, OrderCandidateTable):
            raise ValueError("Candidate-table payload is invalid")
    # Pickles written before provenance was introduced do not receive a
    # dataclass default during unpickling. Normalize them here so the explicit
    # API provenance policy can decide whether reuse is acceptable.
    if not hasattr(table, "provenance"):
        table.provenance = {}
    _validate_candidate_table(table)
    if expected_sha256 is not None:
        actual = candidate_table_content_sha256(table)
        if actual != expected_sha256:
            raise ValueError(
                "Candidate-table content address mismatch: "
                f"expected {expected_sha256}, observed {actual}"
            )
    return table


def canonical_candidate_table_bytes(table: OrderCandidateTable) -> bytes:
    """Return the portable canonical bytes used to content-address a table.

    Parent indices within a family and candidate families within a child are
    sorted before encoding.  Thus an irrelevant candidate-row permutation does
    not change the hash.  Floating-point scores use ``float.hex`` so every
    finite IEEE-754 binary64 value round-trips exactly.
    """

    _validate_candidate_table(table)
    children: list[dict[str, object]] = []
    for parent_sets, loglik, logprior in zip(
        table.parent_sets, table.loglik, table.logprior
    ):
        families = []
        for row, likelihood, prior in zip(parent_sets, loglik, logprior):
            parents = sorted(int(parent) for parent in row if int(parent) >= 0)
            families.append(
                {
                    "parents": parents,
                    "logevidence_hex": float(likelihood).hex(),
                    "logprior_hex": float(prior).hex(),
                }
            )
        families.sort(key=lambda family: tuple(family["parents"]))
        children.append({"families": families})
    payload = {
        "format": "ordersmc-candidate-table",
        "version": CANDIDATE_FORMAT_VERSION,
        "nodes": list(table.nodes),
        "provenance": dict(getattr(table, "provenance", {})),
        "children": children,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate-table provenance is not canonically JSON serializable") from exc
    return _CANONICAL_CANDIDATE_MAGIC + encoded + b"\n"


def _serialized_candidate_table_bytes(table: OrderCandidateTable) -> bytes:
    """Encode canonical content plus the API-visible candidate row order."""

    canonical = json.loads(
        canonical_candidate_table_bytes(table)[len(_CANONICAL_CANDIDATE_MAGIC) :]
    )
    row_orders: list[list[int]] = []
    for parent_sets in table.parent_sets:
        original = [
            tuple(sorted(int(parent) for parent in row if int(parent) >= 0))
            for row in parent_sets
        ]
        canonical_order = sorted(original)
        positions = {family: index for index, family in enumerate(canonical_order)}
        row_orders.append([positions[family] for family in original])
    wrapper = {"canonical": canonical, "row_orders": row_orders}
    encoded = json.dumps(
        wrapper,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return _SERIALIZED_CANDIDATE_MAGIC + encoded + b"\n"


def candidate_table_content_sha256(table: OrderCandidateTable) -> str:
    """Return the canonical, row-order-independent candidate-table address."""

    return hashlib.sha256(canonical_candidate_table_bytes(table)).hexdigest()


def candidate_table_artifact_record(
    table: OrderCandidateTable,
    path: str | Path | None = None,
) -> dict[str, object]:
    """Build a machine-readable content/serialization record for a table."""

    canonical = canonical_candidate_table_bytes(table)
    record: dict[str, object] = {
        "schema": "ordersmc-candidate-table-artifact-v1",
        "format_version": CANDIDATE_FORMAT_VERSION,
        "canonical_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "canonical_content_bytes": len(canonical),
        "node_count": len(table.nodes),
        "family_counts": [int(len(families)) for families in table.parent_sets],
    }
    if path is not None:
        source = Path(path)
        # Fail closed if the supplied file is not the table being described.
        load_candidate_table(
            source,
            expected_sha256=str(record["canonical_content_sha256"]),
        )
        serialized = source.read_bytes()
        record.update(
            {
                "filename": source.name,
                "serialized_sha256": hashlib.sha256(serialized).hexdigest(),
                "serialized_bytes": len(serialized),
            }
        )
    return record


def validate_complete_candidate_pool(
    table: OrderCandidateTable,
    *,
    max_parents: int,
) -> dict[str, object]:
    """Validate exact enumeration of every allowed parent family per child.

    The empty family and every subset of the other nodes through
    ``max_parents`` must occur exactly once.  The returned acceptance record is
    suitable for inclusion beside the table's content address.
    """

    _validate_candidate_table(table)
    if isinstance(max_parents, (bool, np.bool_)) or not isinstance(
        max_parents, (int, np.integer)
    ):
        raise TypeError("max_parents must be an integer")
    parent_limit = int(max_parents)
    if parent_limit < 0:
        raise ValueError("max_parents must be non-negative")
    per_child: dict[str, int] = {}
    provenance = getattr(table, "provenance", {})
    build_config = provenance.get("candidate_build_config", {})
    restricted_clg = (
        isinstance(build_config, Mapping)
        and build_config.get("score_method") == "clg-bic"
    )
    variable_metadata: Mapping[str, object] = {}
    if restricted_clg:
        metadata = provenance.get("metadata")
        variables = metadata.get("variables") if isinstance(metadata, Mapping) else None
        if not isinstance(variables, Mapping):
            raise ValueError(
                "clg-bic candidate completeness requires variable types in provenance"
            )
        variable_metadata = variables
    all_nodes = range(len(table.nodes))
    for child_index, (child, encoded) in enumerate(zip(table.nodes, table.parent_sets)):
        if restricted_clg:
            child_spec = variable_metadata.get(child)
            if not isinstance(child_spec, Mapping) or child_spec.get("type") not in {
                "bin",
                "cat",
                "con",
            }:
                raise ValueError(f"missing or invalid clg-bic metadata for child {child!r}")
            child_discrete = child_spec["type"] in {"bin", "cat"}
            eligible = []
            for node in all_nodes:
                if node == child_index:
                    continue
                parent = table.nodes[node]
                parent_spec = variable_metadata.get(parent)
                if not isinstance(parent_spec, Mapping) or parent_spec.get("type") not in {
                    "bin",
                    "cat",
                    "con",
                }:
                    raise ValueError(
                        f"missing or invalid clg-bic metadata for parent {parent!r}"
                    )
                if child_discrete and parent_spec["type"] == "con":
                    continue
                eligible.append(node)
        else:
            eligible = [node for node in all_nodes if node != child_index]
        expected = {
            family
            for size in range(min(parent_limit, len(eligible)) + 1)
            for family in combinations(eligible, size)
        }
        observed = {
            tuple(sorted(int(parent) for parent in row if int(parent) >= 0))
            for row in encoded
        }
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"candidate pool for child {child!r} is incomplete; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        per_child[str(child)] = len(observed)
    return {
        "schema": "ordersmc-complete-candidate-pool-acceptance-v1",
        "accepted": True,
        "max_parents": parent_limit,
        "node_count": len(table.nodes),
        "families_per_child": per_child,
        "model_restriction": (
            "restricted-conditional-gaussian" if restricted_clg else "none"
        ),
        "candidate_table_sha256": candidate_table_content_sha256(table),
    }


def _table_from_canonical_bytes(payload: bytes) -> OrderCandidateTable:
    try:
        decoded = json.loads(
            payload[len(_CANONICAL_CANDIDATE_MAGIC) :].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Candidate-table v2 JSON payload is invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Candidate-table v2 payload must be a JSON object")
    required = {"format", "version", "nodes", "provenance", "children"}
    if set(decoded) != required:
        raise ValueError("Candidate-table v2 payload has an invalid top-level schema")
    if decoded["format"] != "ordersmc-candidate-table" or decoded["version"] != 2:
        raise ValueError("Candidate-table v2 format/version is invalid")
    nodes = decoded["nodes"]
    provenance = decoded["provenance"]
    children = decoded["children"]
    if not isinstance(nodes, list) or not isinstance(provenance, dict) or not isinstance(children, list):
        raise ValueError("Candidate-table v2 nodes, provenance, or children are invalid")
    if len(children) != len(nodes):
        raise ValueError("Candidate-table v2 has the wrong number of child records")
    parent_sets: list[np.ndarray] = []
    loglik: list[np.ndarray] = []
    logprior: list[np.ndarray] = []
    for child in children:
        if not isinstance(child, dict) or set(child) != {"families"}:
            raise ValueError("Candidate-table v2 child record is invalid")
        families = child["families"]
        if not isinstance(families, list) or not families:
            raise ValueError("Candidate-table v2 child must contain candidate families")
        parsed: list[tuple[tuple[int, ...], float, float]] = []
        for family in families:
            if not isinstance(family, dict) or set(family) != {
                "parents",
                "logevidence_hex",
                "logprior_hex",
            }:
                raise ValueError("Candidate-table v2 family record is invalid")
            parents = family["parents"]
            if not isinstance(parents, list) or any(
                isinstance(parent, bool) or not isinstance(parent, int)
                for parent in parents
            ):
                raise ValueError("Candidate-table v2 parent indices are invalid")
            try:
                likelihood = float.fromhex(family["logevidence_hex"])
                prior = float.fromhex(family["logprior_hex"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Candidate-table v2 score encoding is invalid") from exc
            parsed.append((tuple(parents), likelihood, prior))
        width = max(1, max(len(parents) for parents, _, _ in parsed))
        encoded = np.full((len(parsed), width), -1, dtype=np.int32)
        for row_index, (parents, _, _) in enumerate(parsed):
            encoded[row_index, : len(parents)] = parents
        parent_sets.append(encoded)
        loglik.append(np.asarray([value for _, value, _ in parsed], dtype=np.float64))
        logprior.append(np.asarray([value for _, _, value in parsed], dtype=np.float64))
    table = OrderCandidateTable(
        nodes=list(nodes),
        parent_sets=parent_sets,
        loglik=loglik,
        logprior=logprior,
        provenance=dict(provenance),
    )
    _validate_candidate_table(table)
    return table


def _table_from_serialized_bytes(payload: bytes) -> OrderCandidateTable:
    try:
        wrapper = json.loads(
            payload[len(_SERIALIZED_CANDIDATE_MAGIC) :].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Candidate-table serialized v2 JSON payload is invalid") from exc
    if not isinstance(wrapper, dict) or set(wrapper) != {"canonical", "row_orders"}:
        raise ValueError("Candidate-table serialized v2 wrapper is invalid")
    canonical_object = wrapper["canonical"]
    row_orders = wrapper["row_orders"]
    if not isinstance(canonical_object, dict) or not isinstance(row_orders, list):
        raise ValueError("Candidate-table serialized v2 content is invalid")
    canonical_payload = (
        _CANONICAL_CANDIDATE_MAGIC
        + json.dumps(
            canonical_object,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    table = _table_from_canonical_bytes(canonical_payload)
    if len(row_orders) != len(table.nodes):
        raise ValueError("Candidate-table serialized v2 row-order count is invalid")
    for child, row_order in enumerate(row_orders):
        family_count = len(table.parent_sets[child])
        if (
            not isinstance(row_order, list)
            or any(isinstance(index, bool) or not isinstance(index, int) for index in row_order)
            or sorted(row_order) != list(range(family_count))
        ):
            raise ValueError("Candidate-table serialized v2 row order is not a permutation")
        permutation = np.asarray(row_order, dtype=np.int64)
        table.parent_sets[child] = table.parent_sets[child][permutation]
        table.loglik[child] = table.loglik[child][permutation]
        table.logprior[child] = table.logprior[child][permutation]
    _validate_candidate_table(table)
    return table


def _validate_candidate_table(table: OrderCandidateTable) -> None:
    # Persistence and sampling deliberately share one structural validator so a
    # checkpoint cannot be saved successfully and then fail only when resumed.
    _validate_core_candidate_table(table)

    provenance = getattr(table, "provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("Candidate-table provenance must be a mapping")
    if provenance:
        required = {
            "schema_version",
            "ordersmc_version",
            "score_schema_version",
            "data_fingerprint",
            "nodes",
            "metadata",
            "candidate_build_config",
        }
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(f"Candidate-table provenance is missing fields: {missing}")
        if list(provenance["nodes"]) != list(table.nodes):
            raise ValueError("Candidate-table provenance nodes do not match the table")
        fingerprint = provenance["data_fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Candidate-table data fingerprint is invalid")
        if not isinstance(provenance["metadata"], Mapping):
            raise ValueError("Candidate-table provenance metadata must be a mapping")
        if not isinstance(provenance["candidate_build_config"], Mapping):
            raise ValueError("Candidate-table candidate_build_config must be a mapping")


def inferential_replay_record(result: OrderSMCResult) -> dict[str, object]:
    """Return a timing-independent digest of a completed scientific result.

    Wall-clock and timestamp fields remain in the ordinary run metadata.  This
    parallel record covers the stochastic/inferential state only, allowing two
    same-seed runs to be compared byte-for-byte even when their runtimes differ.
    """

    if not isinstance(result, OrderSMCResult):
        raise TypeError("result must be an OrderSMCResult")

    def array_record(value: object, dtype: np.dtype) -> dict[str, object]:
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        little_endian = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        return {
            "dtype": little_endian.dtype.str,
            "shape": list(little_endian.shape),
            "sha256": hashlib.sha256(little_endian.tobytes(order="C")).hexdigest(),
        }

    def probability_rows(values: Mapping[tuple[str, str], float]) -> list[dict[str, str]]:
        return [
            {
                "parent": str(parent),
                "child": str(child),
                "value_hex": float(value).hex(),
            }
            for (parent, child), value in sorted(values.items())
        ]

    timing_tokens = ("elapsed", "timestamp", "started_utc", "completed_utc", "wall_seconds", "cpu_seconds")

    def strip_timing(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): strip_timing(child)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
                if not any(token in str(key).lower() for token in timing_tokens)
            }
        if isinstance(value, (list, tuple)):
            return [strip_timing(child) for child in value]
        if isinstance(value, (float, np.floating)):
            return {"float64_hex": float(value).hex()}
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, np.ndarray):
            return strip_timing(value.tolist())
        raise TypeError(f"unsupported inferential replay value {type(value).__name__}")

    context_keys = (
        "nodes",
        "bit_generator",
        "candidate_max_indegree",
        "posterior_semantics",
        "target_label",
        "order_target",
        "tempering_path",
        "initial_effective_sample_size",
        "log_path_normalizer_ratio_estimate",
        "log_path_initial_normalizer_estimate",
        "best_edges_semantics",
        "edge_mcse_calibrated",
        "convergence_established",
        "candidate_score_schema_version",
        "mpi_ranks",
        "rng_lineage",
        "tempering_schedule",
        "adaptive_cess_target",
        "resampling_method",
        "resampling_ess_fraction",
        "proposal_mixture",
        "proposal_diagnostics",
        "final_target_snapshot_steps",
        "best_dag_log_score",
        "visited_order_states",
        "final_unique_orders",
        "final_unique_order_fraction",
        "unique_root_ancestors",
        "genealogical_ess",
        "target_evaluations",
        "target_score_lookups",
        "config",
        "candidate_build_config",
        "sampling_config",
        "data_fingerprint",
    )
    run_context = {
        key: result.run_info[key]
        for key in context_keys
        if key in result.run_info
    }
    snapshots = [
        {
            "beta_hex": float(snapshot.beta).hex(),
            "final_target_mh_sweeps": int(snapshot.final_target_mh_sweeps),
            "edge_probabilities": probability_rows(snapshot.edge_probs),
            "edge_mcse": probability_rows(snapshot.edge_mcse),
            "effective_sample_size_hex": float(snapshot.effective_sample_size).hex(),
            "unique_orders": int(snapshot.unique_orders),
            "accepted_moves": int(snapshot.accepted_moves),
            "proposed_moves": int(snapshot.proposed_moves),
        }
        for snapshot in result.final_target_snapshots
    ]
    record: dict[str, object] = {
        "schema": "ordersmc-inferential-replay-v1",
        "orders": array_record(np.asarray(result.orders), np.dtype("<i8")),
        "weights": array_record(result.weights, np.dtype("<f8")),
        "log_weights": array_record(result.logw, np.dtype("<f8")),
        "best_order": array_record(result.best_order, np.dtype("<i8")),
        "best_edges": [list(edge) for edge in sorted(result.best_edges)],
        "edge_probabilities": probability_rows(result.edge_probs),
        "edge_mcse": probability_rows(result.edge_mcse),
        "effective_sample_size_hex": float(result.effective_sample_size).hex(),
        "history": strip_timing(result.history),
        "proposal_events": strip_timing(result.run_info.get("proposal_timing", [])),
        "snapshots": snapshots,
        "log_normalizing_constant_hex": float(
            result.log_normalizing_constant_estimate
        ).hex(),
        "log_prior_normalizing_constant_hex": float(
            result.log_prior_normalizing_constant_estimate
        ).hex(),
        "log_evidence_ratio_hex": float(result.log_evidence_ratio_estimate).hex(),
        "candidate_table_signature": result.run_info.get("candidate_table_signature"),
        "candidate_table_sha256": result.run_info.get("candidate_table_sha256"),
        "run_context": strip_timing(run_context),
    }
    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    record["inferential_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


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
    snapshot_rows = []
    for snapshot_index, snapshot in enumerate(result.final_target_snapshots):
        for (parent, child), probability in sorted(snapshot.edge_probs.items()):
            snapshot_rows.append(
                {
                    "Snapshot": snapshot_index,
                    "Beta": 1.0,
                    "FinalTargetMHSweeps": snapshot.final_target_mh_sweeps,
                    "Parent": parent,
                    "Child": child,
                    "PosteriorProbability": probability,
                    "MonteCarloSE": snapshot.edge_mcse.get((parent, child), float("nan")),
                    "MonteCarloSECalibrated": False,
                    "EffectiveSampleSize": snapshot.effective_sample_size,
                    "UniqueOrders": snapshot.unique_orders,
                    "AcceptedMoves": snapshot.accepted_moves,
                    "ProposedMoves": snapshot.proposed_moves,
                }
            )
    if result.final_target_snapshots:
        pd.DataFrame(
            snapshot_rows,
            columns=[
                "Snapshot",
                "Beta",
                "FinalTargetMHSweeps",
                "Parent",
                "Child",
                "PosteriorProbability",
                "MonteCarloSE",
                "MonteCarloSECalibrated",
                "EffectiveSampleSize",
                "UniqueOrders",
                "AcceptedMoves",
                "ProposedMoves",
            ],
        ).to_csv(destination / "posterior_snapshots.csv", index=False)
    save_mapping(result.summary(), destination / "summary.json")
    save_mapping(inferential_replay_record(result), destination / "inferential_replay.json")
    save_mapping(
        {
            **result.run_info,
            "posterior_threshold": threshold,
            "format_version": RESULT_FORMAT_VERSION,
        },
        destination / "run.json",
    )


def save_smc_checkpoint(state: OrderSMCState, path: str | Path) -> Path:
    """Atomically save a resumable local-SMC stage-boundary state.

    The file contains pickle data and must only be loaded from a trusted source.
    Candidate scores are not duplicated; :func:`load_smc_checkpoint` verifies
    the supplied table against the state's content digest before resumption.
    """
    if not isinstance(state, OrderSMCState):
        raise TypeError("state must be an OrderSMCState")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = {
        "format": "ordersmc-local-smc-checkpoint",
        "version": SMC_CHECKPOINT_FORMAT_VERSION,
        "state": state,
    }
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_smc_checkpoint(
    path: str | Path,
    *,
    table: OrderCandidateTable | None = None,
) -> OrderSMCState:
    """Load a trusted local-SMC checkpoint and optionally verify its score table."""
    with gzip.open(Path(path), "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != "ordersmc-local-smc-checkpoint":
        raise ValueError("Not an OrderSMC-DAG local SMC checkpoint")
    if payload.get("version") != SMC_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported SMC-checkpoint version {payload.get('version')}")
    state = payload.get("state")
    if not isinstance(state, OrderSMCState):
        raise ValueError("SMC-checkpoint payload is invalid")
    if table is not None:
        _validate_candidate_table(table)
        if state.table_signature != _candidate_table_signature(table):
            raise ValueError("SMC checkpoint was created for a different candidate table")
    return state


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
