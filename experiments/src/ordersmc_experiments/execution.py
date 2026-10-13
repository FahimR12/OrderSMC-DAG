"""Resumable E1--E4 execution entry points for local and Slurm jobs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import site
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_info

from ordersmc import OrderSMCConfig, build_candidates, sample
from ordersmc.core import OrderCandidateTable
from ordersmc.persistence import (
    candidate_table_content_sha256,
    load_candidate_table,
    save_candidate_table,
)

from .adapter_validation import validate_algorithm_adapters
from .benchmark_data import load_experiment_artifact, write_clgaussian_artifact
from .bnlearn_adapter import run_bnlearn
from .design import EXPERIMENT_IDS, build_experiment_plan
from .exact_order import exact_order_dp
from .generator import (
    MixedDAGConfig,
    generate_mixed_dag_dataset,
    write_mixed_dag_artifact,
)
from .metadata_profiles import (
    MetadataDAGConfig,
    generate_metadata_dag_dataset,
    metadata_config_for_spec,
    write_metadata_dag_artifact,
)
from .order_mcmc import OrderMCMCConfig, order_mcmc
from .score_validation import run_e0a_full_validation
from .structure_mcmc import (
    PartitionMCMCConfig,
    REVMCMCConfig,
    partition_mcmc,
    rev_mcmc,
)


SCHEMA = "ordersmc-execution-v3"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "execution_v3.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _complete(path: Path, identity: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return value.get("status") == "complete" and value.get("identity_sha256") == identity


def _execution_config(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    schema = value.get("schema")
    if schema not in {
        "ordersmc-execution-config-v3",
        "ordersmc-execution-config-v4",
        "ordersmc-execution-config-v5",
        "ordersmc-execution-config-v6",
        "ordersmc-execution-config-v7",
        "ordersmc-execution-config-v8",
        "ordersmc-execution-config-v9",
    }:
        raise ValueError("unsupported execution configuration")
    if value.get("status") == "confirmatory-frozen":
        freeze_name = value.get("freeze_record")
        if not isinstance(freeze_name, str) or Path(freeze_name).name != freeze_name:
            raise ValueError("confirmatory configuration has no safe freeze record")
        freeze_path = path.parent / freeze_name
        freeze = _load_json(freeze_path)
        expected_freeze_schema = {
            "ordersmc-execution-config-v3": "ordersmc-pilot-freeze-v3",
            "ordersmc-execution-config-v4": "ordersmc-mixed-paper-freeze-v1",
            "ordersmc-execution-config-v5": "ordersmc-mixed-paper-freeze-v2",
            "ordersmc-execution-config-v6": "ordersmc-budget-freeze-v1",
            "ordersmc-execution-config-v7": "ordersmc-v3-resource-freeze-v1",
        }[str(schema)]
        if (
            freeze.get("schema") != expected_freeze_schema
            or freeze.get("status") != "confirmatory-frozen"
            or freeze.get("execution_config") != path.name
            or freeze.get("execution_config_sha256") != _sha(value)
            or freeze.get("execution_config_raw_sha256")
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("execution configuration disagrees with its pilot freeze")
    return value


def _validate_plan_execution_binding(
    plan: Mapping[str, object],
    config_path: Path,
    execution: Mapping[str, object],
    profile: str,
) -> None:
    """Enforce predeclared confirmatory identifiers at every execution entrypoint."""

    if (
        execution.get("schema") != "ordersmc-execution-config-v7"
        or execution.get("status") != "confirmatory-frozen"
    ):
        return
    if plan.get("experiment") != "v3" or profile != "full":
        raise ValueError("the frozen V3 configuration is restricted to full V3")
    freeze = _load_json(config_path.parent / str(execution["freeze_record"]))
    expected_seed = int(freeze["confirmatory_master_seed"])
    if int(plan.get("master_seed", -1)) != expected_seed:
        raise ValueError(
            f"full V3 master seed must equal the frozen value {expected_seed}"
        )


def _source_sha256() -> str:
    digest = hashlib.sha256()
    roots = (
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "experiments" / "src",
        REPOSITORY_ROOT / "experiments" / "adapters",
        REPOSITORY_ROOT / "experiments" / "fixtures",
        REPOSITORY_ROOT / "scripts",
        REPOSITORY_ROOT / "experiments" / "configs",
        REPOSITORY_ROOT / "summary_statistics_multiopmic_data_pathway_molecule_level",
    )
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validated_preflight(output_root: Path) -> dict[str, Any]:
    _attest_environment()
    path = output_root / "validation" / "preflight.json"
    report = _load_json(path)
    if report.get("status") != "complete" or not report.get("all_passed"):
        raise RuntimeError("the score and adapter preflight is not complete")
    if report.get("source_sha256") != _source_sha256():
        raise RuntimeError("source changed after preflight; rerun preflight before execution")
    return report


def _attest_environment() -> dict[str, object]:
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    incorrect = {name: os.environ.get(name) for name in thread_variables if os.environ.get(name) != "1"}
    if incorrect:
        raise RuntimeError(f"numerical thread environment is not locked to one: {incorrect}")
    if sys.flags.hash_randomization != 0 or os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 must be set before Python starts")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or site.ENABLE_USER_SITE:
        raise RuntimeError("PYTHONNOUSERSITE=1 must disable user-site packages")
    pools = threadpool_info()
    non_single = [row for row in pools if int(row.get("num_threads", 1)) != 1]
    if non_single:
        raise RuntimeError(f"loaded numerical thread pools are not single-threaded: {non_single}")
    return {
        "hash_randomization": int(sys.flags.hash_randomization),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "thread_environment": {name: os.environ[name] for name in thread_variables},
        "loaded_thread_pools": pools,
    }


def _plan(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    claimed = value.get("plan_sha256")
    body = dict(value)
    body.pop("plan_sha256", None)
    if claimed != _sha(body):
        raise ValueError("plan SHA-256 does not match its content")
    return value


def _profile_config(
    dataset: Mapping[str, object], execution: Mapping[str, object], profile: str, seed: int
) -> OrderSMCConfig:
    e1 = dataset["experiment"] in {"e1", "v1"}
    exact_v3 = dataset["experiment"] == "v3" and int(dataset["dimension"]) <= 8
    candidate = dict(execution["candidate"])
    if e1 or exact_v3:
        candidate.update(
            {
                "max_parents": int(dataset["max_indegree"]),
                "screen_k": int(dataset["dimension"]) - 1,
                "parent_pool_max": None,
                "pair_rescue_anchor_k": 0,
                "pair_rescue_k": 0,
            }
        )
    elif dataset["experiment"] == "v3":
        candidate["max_parents"] = int(dataset["max_indegree"])
        if int(dataset["max_indegree"]) >= 4:
            policy = dict(
                execution.get(
                    "v3_high_indegree_candidate",
                    {"screen_k": 10, "parent_pool_max": 10},
                )
            )
            candidate["screen_k"] = int(policy["screen_k"])
            candidate["parent_pool_max"] = int(policy["parent_pool_max"])
    values = {**candidate, **dict(execution["ordersmc"]), "seed": int(seed)}
    # Historical execution JSONs predate the v0.7 path. Do not reinterpret a
    # frozen fixed-temperature protocol through newly changed API defaults.
    values.setdefault("tempering_path", "collapsed_likelihood")
    values.setdefault("adaptive_tempering", False)
    model_class_scores = execution.get("model_class_scores", {})
    if not isinstance(model_class_scores, Mapping):
        raise ValueError("model_class_scores must be a mapping when supplied")
    data_kind = str(dataset.get("data_kind", ""))
    if data_kind in model_class_scores:
        values["score_method"] = str(model_class_scores[data_kind])
    metadata_policy = execution.get("metadata_score_policy")
    if metadata_policy is not None and data_kind == "metadata_calibrated":
        genuinely_cg = (dataset.get("causal_regime") == "cg_compatible"
                        and dataset.get("data_variant") == "gaussian")
        values["score_method"] = str(metadata_policy[
            "gaussian_cg_compatible" if genuinely_cg else "default"])
        if values["score_method"] == "clg-bic" and not genuinely_cg:
            raise ValueError("clg-bic requires Gaussian mechanisms AND CG-compatible directions")
    if profile == "smoke":
        smoke = execution["smoke"]
        values.update(
            {
                "particles": smoke["particles"],
                "num_betas": smoke["num_betas"],
                "mh_steps": smoke["mh_steps"],
                "mh_steps_resample": smoke["mh_steps_resample"],
            }
        )
    return OrderSMCConfig.protocol_v2(
        lambda_target=float(execution["prior"]["order_lambda"]), **values
    )


def _structure_table(
    table: OrderCandidateTable, order_lambda: float, structure_lambda: float
) -> OrderCandidateTable:
    adjustment = float(order_lambda) - float(structure_lambda)
    priors = []
    for parent_sets, logprior in zip(table.parent_sets, table.logprior):
        sizes = np.count_nonzero(np.asarray(parent_sets) >= 0, axis=1)
        priors.append(np.asarray(logprior, dtype=float) + adjustment * sizes)
    provenance = dict(getattr(table, "provenance", {}))
    provenance["retargeting"] = {
        "from": "order-modular",
        "to": "structure-modular",
        "order_lambda": float(order_lambda),
        "structure_lambda": float(structure_lambda),
        "local_logevidence_unchanged": True,
    }
    return OrderCandidateTable(
        nodes=list(table.nodes),
        parent_sets=[np.asarray(row).copy() for row in table.parent_sets],
        loglik=[np.asarray(row).copy() for row in table.loglik],
        logprior=priors,
        provenance=provenance,
        build_diagnostics=dict(getattr(table, "build_diagnostics", {})),
    )


def _dataset_paths(root: Path, dataset_id: str) -> tuple[Path, Path, Path]:
    directory = root / "datasets" / dataset_id
    return directory, directory / "candidate.json.gz", directory / "prepared.json"


def _dataset_identity(
    plan: Mapping[str, object],
    spec: Mapping[str, object],
    execution: Mapping[str, object],
    profile: str,
    preflight_report: Mapping[str, object],
) -> str:
    """Return the immutable identity shared by preparation and its audit."""

    return _sha(
        {
            "schema": SCHEMA,
            "kind": "dataset",
            "plan_sha256": plan["plan_sha256"],
            "spec": spec,
            "execution_config_sha256": _sha(execution),
            "profile": profile,
            "preflight_identity_sha256": preflight_report["identity_sha256"],
        }
    )


def _seed(run: Mapping[str, object]) -> int:
    record = run.get("method_seed")
    return 0 if record is None else int(record["seed"])


def preflight(output_root: Path) -> dict[str, object]:
    environment_attestation = _attest_environment()
    score_report = run_e0a_full_validation()
    adapters = validate_algorithm_adapters()
    report: dict[str, object] = {
        "schema": "ordersmc-preflight-v3",
        "created_utc": _utc(),
        "score_validation": score_report,
        "adapter_validation": adapters,
        "source_sha256": _source_sha256(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                package: metadata.version(package)
                for package in ("numpy", "scipy", "pandas", "networkx", "scikit-learn", "mpi4py")
                if _package_available(package)
            },
            "attestation": environment_attestation,
        },
        "all_passed": bool(score_report["all_passed"]) and bool(adapters["all_passed"]),
    }
    report["status"] = "complete" if report["all_passed"] else "failed"
    report["identity_sha256"] = _sha(
        {key: value for key, value in report.items() if key not in {"created_utc", "identity_sha256"}}
    )
    _atomic_json(output_root / "validation" / "preflight.json", report)
    if not report["all_passed"]:
        raise RuntimeError("score or adapter preflight failed")
    return report


def _package_available(package: str) -> bool:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        return False
    return True


def write_plan(
    experiment: str, master_seed: int, destination: Path, profile: str = "full"
) -> None:
    value = build_experiment_plan(
        experiment, master_seed=master_seed, profile=profile
    ).to_dict()
    if destination.exists():
        # JSON serialization normalizes tuples to arrays/lists. Compare the
        # canonical JSON identity rather than Python container types so an
        # unchanged plan is genuinely resumable across launcher invocations.
        if _sha(_load_json(destination)) != _sha(value):
            raise FileExistsError(f"existing plan differs: {destination}")
        return
    _atomic_json(destination, value)


def prepare_dataset(
    plan_path: Path,
    index: int,
    output_root: Path,
    config_path: Path,
    profile: str,
) -> None:
    plan = _plan(plan_path)
    execution = _execution_config(config_path)
    _validate_plan_execution_binding(plan, config_path, execution, profile)
    preflight_report = _validated_preflight(output_root)
    datasets = plan["datasets"]
    if execution.get("metadata_calibration_gate", {}).get("required", False):
        from .metadata_calibration_v2 import validate_gate
        validate_gate(output_root, config_path, profile)
    if not 0 <= index < len(datasets):
        raise IndexError("dataset index outside plan")
    spec = datasets[index]
    identity = _dataset_identity(plan, spec, execution, profile, preflight_report)
    directory, candidate_path, completed_path = _dataset_paths(
        output_root, str(spec["dataset_id"])
    )
    if _complete(completed_path, identity):
        return
    if directory.exists():
        raise RuntimeError(f"incomplete dataset directory requires inspection: {directory}")
    staging = directory.with_name(directory.name + f".staging-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging directory already exists: {staging}")

    signal = str(
        spec.get(
            "signal",
            "weak" if spec["graph_family"] == "weak_bridges" else "moderate",
        )
    )
    if signal == "mixed-difficulty":
        signal = "weak" if spec["graph_family"] == "weak_bridges" else "moderate"
    if signal == "none":
        signal = "moderate"
    pilot_size = (
        int(execution["smoke"]["pilot_size"])
        if profile == "smoke"
        else (
            int(dict(execution.get("generator", {})).get("pilot_size", 2_000))
            if spec.get("data_kind") == "metadata_calibrated"
            else 5_000
        )
    )
    started = time.perf_counter()
    if spec.get("data_kind") == "native_mixed_conditional_gaussian":
        dataset = write_clgaussian_artifact(
            staging,
            n_train=int(spec["n_train"]),
            dataset_seed=int(spec["dataset_seed"]["seed"]),
            dataset_id=str(spec["dataset_id"]),
        )
    elif spec.get("data_kind") == "metadata_calibrated":
        generator_settings = dict(execution.get("generator", {}))
        metadata_config = metadata_config_for_spec(spec, plan["master_seed"], generator_settings, pilot_size)
        dataset = generate_metadata_dag_dataset(metadata_config)
        write_metadata_dag_artifact(dataset, staging)
    else:
        config = MixedDAGConfig(
            d=int(spec["dimension"]),
            n_train=int(spec["n_train"]),
            graph_family=str(spec["graph_family"]),
            fixed_topology_id=spec.get("fixed_topology_id"),
            expected_indegree=float(spec["expected_indegree"]),
            max_indegree=int(spec["max_indegree"]),
            signal_strength=signal,
            continuous_noise=str(spec.get("continuous_noise", "gaussian")),
            variable_type_regime=str(spec.get("variable_type_regime", "balanced_mixed")),
            pilot_size=pilot_size,
            max_extreme_probability_fraction=float(
                dict(execution.get("generator", {})).get(
                    "max_extreme_probability_fraction", 0.01
                )
            ),
            enforce_extreme_probability_fraction=bool(
                dict(execution.get("generator", {})).get(
                    "enforce_extreme_probability_fraction", True
                )
            ),
            master_seed=int(plan["master_seed"]),
            experiment_id=str(spec["experiment"]),
            cell_id=str(
                spec.get(
                    "cell_id", f"d{int(spec['dimension']):03d}-{spec['graph_family']}"
                )
            ),
            replicate_id=int(spec["replicate"]),
        )
        dataset = generate_mixed_dag_dataset(config)
        write_mixed_dag_artifact(dataset, staging)
    candidate_config = _profile_config(spec, execution, profile, seed=0)
    candidate_started = time.perf_counter()
    table = build_candidates(
        dataset.train,
        dataset.variable_types,
        dataset.cardinalities,
        config=candidate_config,
        backend="local",
        verbose=False,
    )
    candidate_seconds = time.perf_counter() - candidate_started
    save_candidate_table(table, staging / candidate_path.name)
    true_edges = set(dataset.dag.edges())
    retained = {
        (table.nodes[int(parent)], table.nodes[child])
        for child, rows in enumerate(table.parent_sets)
        for row in rows
        for parent in row
        if int(parent) >= 0
    }
    complete_families = 0
    for child in table.nodes:
        parents = tuple(sorted(dataset.dag.predecessors(child)))
        child_index = table.nodes.index(child)
        families = {
            tuple(sorted(table.nodes[int(p)] for p in row if int(p) >= 0))
            for row in table.parent_sets[child_index]
        }
        complete_families += int(parents in families)
    audit = {
        "schema": "ordersmc-prepared-dataset-v3",
        "status": "complete",
        "identity_sha256": identity,
        "dataset_id": spec["dataset_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile": profile,
        "execution_config_sha256": _sha(execution),
        "preflight_identity_sha256": preflight_report["identity_sha256"],
        "source_sha256": preflight_report["source_sha256"],
        "candidate_table_sha256": candidate_table_content_sha256(table),
        "candidate_build_seconds": candidate_seconds,
        "total_prepare_seconds": time.perf_counter() - started,
        "true_directed_parent_recall": len(true_edges & retained) / max(1, len(true_edges)),
        "complete_family_recall": complete_families / len(table.nodes),
        "created_utc": _utc(),
    }
    if spec.get("data_kind") == "metadata_calibrated":
        audit["metadata_calibration"] = dataset.manifest["calibration"]
    _atomic_json(staging / completed_path.name, audit)
    os.replace(staging, directory)


def audit_prepared_datasets(
    plan_path: Path,
    output_root: Path,
    config_path: Path,
    profile: str,
    *,
    minimum_recall: float = 0.95,
) -> dict[str, object]:
    """Fail closed unless every candidate table is intact and sufficiently inclusive."""

    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must lie in [0, 1]")
    plan = _plan(plan_path)
    execution = _execution_config(config_path)
    _validate_plan_execution_binding(plan, config_path, execution, profile)
    preflight_report = _validated_preflight(output_root)
    rows: list[dict[str, object]] = []
    for spec in plan["datasets"]:
        dataset_id = str(spec["dataset_id"])
        _, candidate_path, prepared_path = _dataset_paths(output_root, dataset_id)
        required_recall = (
            1.0
            if plan["experiment"] in {"v1", "v3", "v3-clg"}
            and int(spec["dimension"]) <= 8
            else float(minimum_recall)
        )
        row: dict[str, object] = {
            "dataset_id": dataset_id,
            "cell_id": str(spec.get("cell_id", "")),
            "dimension": int(spec["dimension"]),
            "required_recall": required_recall,
            "status": "failed",
        }
        try:
            identity = _dataset_identity(
                plan, spec, execution, profile, preflight_report
            )
            if not _complete(prepared_path, identity):
                raise RuntimeError("prepared record is missing, incomplete, or stale")
            prepared = _load_json(prepared_path)
            table = load_candidate_table(
                candidate_path,
                expected_sha256=str(prepared["candidate_table_sha256"]),
            )
            if candidate_table_content_sha256(table) != prepared["candidate_table_sha256"]:
                raise RuntimeError("candidate table content hash changed after loading")
            edge_recall = float(prepared["true_directed_parent_recall"])
            family_recall = float(prepared["complete_family_recall"])
            row.update(
                {
                    "status": "complete",
                    "candidate_table_sha256": prepared["candidate_table_sha256"],
                    "true_directed_parent_recall": edge_recall,
                    "complete_family_recall": family_recall,
                    "candidate_build_seconds": float(
                        prepared["candidate_build_seconds"]
                    ),
                    "total_prepare_seconds": float(prepared["total_prepare_seconds"]),
                    "artifact_integrity_passed": True,
                    "individual_descriptive_threshold_met": (
                        edge_recall >= required_recall
                        and family_recall >= required_recall
                    ),
                }
            )
        except Exception as error:
            row.update(
                {
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "artifact_integrity_passed": False,
                    "individual_descriptive_threshold_met": False,
                }
            )
        rows.append(row)

    durations = np.asarray(
        [
            float(row["candidate_build_seconds"])
            for row in rows
            if "candidate_build_seconds" in row
        ],
        dtype=float,
    )
    failed_artifacts = [row for row in rows if not row["artifact_integrity_passed"]]
    cells: list[dict[str, object]] = []
    for cell_id in sorted({str(row["cell_id"]) for row in rows}):
        group = [row for row in rows if row["cell_id"] == cell_id]
        complete_group = [row for row in group if row["artifact_integrity_passed"]]
        exact_cell = (
            plan["experiment"] in {"v1", "v3", "v3-clg"}
            and int(group[0]["dimension"]) <= 8
        )
        mean_edge_recall = (
            float(np.mean([row["true_directed_parent_recall"] for row in complete_group]))
            if complete_group
            else None
        )
        mean_family_recall = (
            float(np.mean([row["complete_family_recall"] for row in complete_group]))
            if complete_group
            else None
        )
        if exact_cell:
            recall_passed = bool(
                len(complete_group) == len(group)
                and all(
                    float(row["true_directed_parent_recall"]) == 1.0
                    and float(row["complete_family_recall"]) == 1.0
                    for row in complete_group
                )
            )
            gate_basis = "every_dataset_exact_recall"
            threshold = 1.0
        else:
            recall_passed = bool(
                len(complete_group) == len(group)
                and mean_family_recall is not None
                and mean_family_recall >= minimum_recall
            )
            gate_basis = "cell_mean_complete_family_recall"
            threshold = float(minimum_recall)
        cells.append(
            {
                "cell_id": cell_id,
                "dimension": int(group[0]["dimension"]),
                "dataset_count": len(group),
                "complete_artifact_count": len(complete_group),
                "gate_basis": gate_basis,
                "required_recall": threshold,
                "mean_true_directed_parent_recall": mean_edge_recall,
                "mean_complete_family_recall": mean_family_recall,
                "minimum_true_directed_parent_recall": min(
                    (float(row["true_directed_parent_recall"]) for row in complete_group),
                    default=None,
                ),
                "minimum_complete_family_recall": min(
                    (float(row["complete_family_recall"]) for row in complete_group),
                    default=None,
                ),
                "recall_gate_passed": recall_passed,
            }
        )
    failed_cells = [row for row in cells if not row["recall_gate_passed"]]
    report: dict[str, object] = {
        "schema": "ordersmc-prepared-audit-v1",
        "status": "complete" if not failed_artifacts and not failed_cells else "failed",
        "created_utc": _utc(),
        "plan_sha256": plan["plan_sha256"],
        "execution_config_sha256": _sha(execution),
        "preflight_identity_sha256": preflight_report["identity_sha256"],
        "source_sha256": preflight_report["source_sha256"],
        "profile": profile,
        "minimum_recall": float(minimum_recall),
        "exact_panel_required_recall": 1.0,
        "planned_datasets": len(plan["datasets"]),
        "passed_datasets": len(rows) - len(failed_artifacts),
        "failed_datasets": len(failed_artifacts),
        "planned_cells": len(cells),
        "passed_cells": len(cells) - len(failed_cells),
        "failed_cells": len(failed_cells),
        "nonexact_gate_basis": "cell_mean_complete_family_recall",
        "minimum_true_directed_parent_recall": min(
            (float(row["true_directed_parent_recall"]) for row in rows if "true_directed_parent_recall" in row),
            default=None,
        ),
        "minimum_complete_family_recall": min(
            (float(row["complete_family_recall"]) for row in rows if "complete_family_recall" in row),
            default=None,
        ),
        "candidate_build_seconds": {
            "mean": float(np.mean(durations)) if len(durations) else None,
            "p95": float(np.quantile(durations, 0.95)) if len(durations) else None,
            "maximum": float(np.max(durations)) if len(durations) else None,
        },
        "cells": cells,
        "datasets": rows,
    }
    _atomic_json(output_root / "validation" / "prepared-audit.json", report)
    if failed_artifacts or failed_cells:
        raise RuntimeError(
            "prepared-dataset audit failed for "
            f"{len(failed_artifacts)}/{len(rows)} artifacts and "
            f"{len(failed_cells)}/{len(cells)} recall cells"
        )
    return report


def _all_edges(nodes: Sequence[str], values: Mapping[tuple[str, str], float]) -> list[dict[str, object]]:
    return [
        {
            "parent": parent,
            "child": child,
            # Numerical normalization can leave harmless round-off immediately
            # outside [0, 1]. Persist a valid probability, not that artefact.
            "probability": float(
                np.clip(float(values.get((parent, child), 0.0)), 0.0, 1.0)
            ),
        }
        for parent in nodes
        for child in nodes
        if parent != child
    ]


def _smc_diagnostics(result: object, config: OrderSMCConfig) -> dict[str, object]:
    """Serialize the diagnostics needed to audit one scientific SMC result."""

    run_info = dict(result.run_info)
    keys = (
        "visited_order_states",
        "final_unique_orders",
        "final_unique_order_fraction",
        "unique_root_ancestors",
        "genealogical_ess",
        "edge_mcse_method",
        "rng_lineage",
        "proposal_diagnostics",
        "target_evaluations",
        "target_score_lookups",
        "total_score_lookups",
        "initial_effective_sample_size",
        "tempering_path",
        "proposal_mixture",
        "tempering_schedule",
        "log_path_normalizer_ratio_estimate",
        "best_edges_semantics",
        "edge_mcse_calibrated",
        "convergence_established",
    )
    edge_mcse = []
    for parent in run_info.get("nodes", []):
        for child in run_info.get("nodes", []):
            if parent == child:
                continue
            value = float(result.edge_mcse.get((parent, child), float("nan")))
            edge_mcse.append(
                {
                    "parent": parent,
                    "child": child,
                    "mcse": value if math.isfinite(value) and value >= 0.0 else None,
                }
            )
    return {
        "effective_sample_size": result.effective_sample_size,
        "log_evidence_ratio": result.log_evidence_ratio_estimate,
        "history": result.history,
        "sampling_config": config.sampling_dict(),
        "summary": result.summary(),
        "edge_mcse": edge_mcse,
        **{key: run_info[key] for key in keys if key in run_info},
    }


def _run_identity(
    plan: Mapping[str, object],
    run: Mapping[str, object],
    prepared: Mapping[str, object],
    execution: Mapping[str, object],
    profile: str,
    preflight_report: Mapping[str, object],
) -> str:
    return _sha(
        {
            "schema": SCHEMA,
            "kind": "run",
            "plan_sha256": plan["plan_sha256"],
            "run": run,
            "candidate_table_sha256": prepared["candidate_table_sha256"],
            "execution_config_sha256": _sha(execution),
            "profile": profile,
            "preflight_identity_sha256": preflight_report["identity_sha256"],
        }
    )


def _metrics(
    nodes: Sequence[str],
    true_edges: set[tuple[str, str]],
    probabilities: Mapping[tuple[str, str], float],
    threshold: float,
    variable_types: Mapping[str, str] | None = None,
) -> dict[str, float]:
    pairs = [(u, v) for u in nodes for v in nodes if u != v]
    y = np.asarray([int(edge in true_edges) for edge in pairs], dtype=int)
    p = np.asarray([float(probabilities.get(edge, 0.0)) for edge in pairs], dtype=float)
    directed_auprc = float(average_precision_score(y, p))
    directed_brier = float(np.mean((p - y) ** 2))
    directed_predicted = p >= threshold
    directed_tp = int(np.sum(directed_predicted & (y == 1)))
    directed_fp = int(np.sum(directed_predicted & (y == 0)))
    directed_fn = int(np.sum(~directed_predicted & (y == 1)))
    skeleton_true = []
    skeleton_prob = []
    expected_shd = 0.0
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            p_lr = float(probabilities.get((left, right), 0.0))
            p_rl = float(probabilities.get((right, left), 0.0))
            skeleton_prob.append(min(1.0, p_lr + p_rl))
            skeleton_true.append(int((left, right) in true_edges or (right, left) in true_edges))
            if (left, right) in true_edges:
                expected_shd += 1.0 - p_lr
            elif (right, left) in true_edges:
                expected_shd += 1.0 - p_rl
            else:
                expected_shd += p_lr + p_rl
    sk_y = np.asarray(skeleton_true, dtype=int)
    sk_p = np.asarray(skeleton_prob, dtype=float)
    predicted = sk_p >= threshold
    tp = int(np.sum(predicted & (sk_y == 1)))
    fp = int(np.sum(predicted & (sk_y == 0)))
    fn = int(np.sum(~predicted & (sk_y == 1)))
    f1 = 0.0 if 2 * tp + fp + fn == 0 else 2.0 * tp / (2 * tp + fp + fn)
    result = {
        "directed_auprc": directed_auprc,
        "directed_brier": directed_brier,
        "directed_precision": directed_tp / max(1, directed_tp + directed_fp),
        "directed_recall": (
            directed_tp / (directed_tp + directed_fn)
            if directed_tp + directed_fn else float("nan")
        ),
        "directed_f1": (
            0.0
            if 2 * directed_tp + directed_fp + directed_fn == 0
            else 2.0 * directed_tp / (2 * directed_tp + directed_fp + directed_fn)
        ),
        "skeleton_auprc": float(average_precision_score(sk_y, sk_p)),
        "skeleton_brier": float(np.mean((sk_p - sk_y) ** 2)),
        "skeleton_precision": tp / max(1, tp + fp),
        "skeleton_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "skeleton_f1": float(f1),
        "posterior_expected_pair_state_shd": float(expected_shd),
    }
    if "PD" in nodes:
        pd_pairs = [(u, v) for u, v in pairs if u == "PD" or v == "PD"]
        pd_y = np.asarray([int(edge in true_edges) for edge in pd_pairs], dtype=int)
        pd_p = np.asarray([float(probabilities.get(edge, 0.0)) for edge in pd_pairs])
        pd_selected = pd_p >= threshold
        pd_tp = int(np.sum(pd_selected & (pd_y == 1)))
        pd_fp = int(np.sum(pd_selected & (pd_y == 0)))
        pd_fn = int(np.sum(~pd_selected & (pd_y == 1)))
        result.update(
            {
                "pd_directed_precision": pd_tp / max(1, pd_tp + pd_fp),
                "pd_directed_recall": (
                    pd_tp / (pd_tp + pd_fn) if pd_tp + pd_fn else float("nan")
                ),
                "pd_selected_false_directed_edges": float(pd_fp),
            }
        )
    if variable_types is not None:
        cross = [
            index
            for index, (parent, child) in enumerate(pairs)
            if variable_types[parent] != variable_types[child]
        ]
        if cross:
            cy, cp = y[cross], p[cross]
            selected = cp >= threshold
            ctp = int(np.sum(selected & (cy == 1)))
            cfp = int(np.sum(selected & (cy == 0)))
            cfn = int(np.sum(~selected & (cy == 1)))
            result.update(
                {
                    "cross_type_directed_precision": ctp / max(1, ctp + cfp),
                    "cross_type_directed_recall": (
                        ctp / (ctp + cfn) if ctp + cfn else float("nan")
                    ),
                    "cross_type_directed_f1": (
                        0.0
                        if 2 * ctp + cfp + cfn == 0
                        else 2.0 * ctp / (2 * ctp + cfp + cfn)
                    ),
                }
            )
    return result


def _full_mcmc_controls(
    method: str,
    run: Mapping[str, object],
    execution: Mapping[str, object],
) -> tuple[int, float, float]:
    """Return thinning, declared total budget, and sampler-only budget."""
    mcmc = execution["mcmc"]
    order_method = method in {"order-mcmc", "order-modular-reference"}
    thin = int(mcmc["order_thin"] if order_method else mcmc["structure_thin"])
    wall_seconds = run.get("wall_seconds")
    declared = float(
        execution["budgets_seconds"]["reference"]
        if str(run["role"]) == "reference"
        else wall_seconds or execution["budgets_seconds"]["e2"]
    )
    sampler = declared
    if order_method and str(run["role"]) != "reference":
        sampler *= float(mcmc["order_sampling_wall_fraction"])
    return thin, declared, sampler


def _run_result(
    run: Mapping[str, object],
    spec: Mapping[str, object],
    table: OrderCandidateTable,
    execution: Mapping[str, object],
    profile: str,
    dataset: object,
    *,
    mpi: bool,
) -> tuple[Mapping[tuple[str, str], float], Sequence[tuple[str, str]], dict[str, object]] | None:
    method = str(run["method"])
    seed = _seed(run)
    smc_config = _profile_config(spec, execution, profile, seed)
    wall_seconds = run.get("wall_seconds")
    if (
        profile in {"full", "pilot"}
        and spec["experiment"] == "e3"
        and method == "ordersmc"
        and wall_seconds is not None
    ):
        dimension = str(int(spec["dimension"]))
        if "e3_ordersmc_ladder_by_dimension" in execution:
            ladder = execution["e3_ordersmc_ladder_by_dimension"][dimension][
                str(int(wall_seconds))
            ]
        else:
            ladder = execution["e3_ordersmc_ladder"][str(int(wall_seconds))]
        smc_config = smc_config.with_overrides(**dict(ladder))
    if (
        profile in {"full", "pilot"}
        and spec["experiment"] in {"e2", "v2"}
        and method == "ordersmc"
    ):
        ladder = execution.get("e2_ordersmc_by_dimension")
        if ladder is not None:
            smc_config = smc_config.with_overrides(
                **dict(ladder[str(int(spec["dimension"]))])
            )
    if profile in {"full", "pilot"} and spec["experiment"] == "e4" and method == "ordersmc":
        scaling = execution.get("e4_ordersmc")
        if scaling is not None:
            smc_config = smc_config.with_overrides(**dict(scaling))
    if method == "exact-order-dp":
        result = exact_order_dp(table, max_nodes=12)
        return result.edge_probs, (), {
            "log_normalizer": result.log_normalizer,
            "log_evidence_ratio": result.log_evidence_ratio,
            "subset_count": result.subset_count,
        }
    if method == "ordersmc":
        result = sample(
            table,
            config=smc_config,
            backend="mpi" if mpi else "local",
            parent_seed_record=run.get("method_seed") if mpi else None,
            verbose=False,
            provenance_policy="require",
        )
        if mpi:
            if result.rank != 0:
                return None
            result = result.to_local_result()
        return result.edge_probs, result.best_edges, _smc_diagnostics(
            result, smc_config
        )

    if method.startswith("bnlearn-"):
        algorithm = "tabu" if "-tabu-" in method else "hc"
        scoring_mode = "native-bic-cg" if method.endswith("-bic-cg") else "shared-mixed-score"
        result = run_bnlearn(
            dataset.train,
            dataset.variable_types,
            table,
            algorithm=algorithm,
            scoring_mode=scoring_mode,
            seed=seed,
            timeout_seconds=int(wall_seconds or 300),
        )
        return result.edge_probs, result.edges, {
            "algorithm": result.algorithm,
            "scoring_mode": result.scoring_mode,
            "network_score": result.score,
            "bnlearn_version": result.bnlearn_version,
            "rscript": result.rscript,
            "deterministic_point_estimate": True,
        }

    smoke = execution["smoke"]
    mcmc = execution["mcmc"]
    if profile == "smoke":
        iterations = int(smoke["mcmc_iterations"])
        burnin = int(smoke["mcmc_burnin"])
        thin = int(smoke["mcmc_thin"])
        budget = smoke["wall_seconds"]
        declared_budget = budget
    else:
        iterations = int(mcmc["maximum_iterations"])
        burnin = max(1, iterations // 4)
        thin, declared_budget, budget = _full_mcmc_controls(
            method, run, execution
        )
    if method in {"order-mcmc", "order-modular-reference"}:
        result = order_mcmc(
            table,
            OrderMCMCConfig(
                iterations=iterations,
                burnin=burnin,
                thin=thin,
                seed=seed,
                wall_seconds=budget,
                wall_burnin_fraction=float(mcmc["burnin_fraction"]),
            ),
        )
    else:
        structure = _structure_table(
            table,
            float(execution["prior"]["order_lambda"]),
            float(execution["prior"]["structure_lambda"]),
        )
        if method == "partition-mcmc":
            result = partition_mcmc(
                structure,
                PartitionMCMCConfig(
                    iterations=iterations,
                    burnin=burnin,
                    thin=thin,
                    seed=seed,
                    wall_seconds=budget,
                    wall_burnin_fraction=float(mcmc["burnin_fraction"]),
                ),
            )
        elif method in {"rev-mcmc", "structure-modular-reference"}:
            result = rev_mcmc(
                structure,
                REVMCMCConfig(
                    iterations=iterations,
                    burnin=burnin,
                    thin=thin,
                    rev_probability=float(mcmc["rev_probability"]),
                    seed=seed,
                    wall_seconds=budget,
                    wall_burnin_fraction=float(mcmc["burnin_fraction"]),
                ),
            )
        else:
            raise ValueError(f"no executable adapter for method {method}")
    return result.edge_probs, result.best_edges, {
        "acceptance_rate": result.acceptance_rate,
        "acceptance_by_proposal": result.acceptance_by_proposal,
        "effective_sample_size": result.effective_sample_size,
        "retained_samples": result.retained_samples,
        "stored_samples": result.stored_samples,
        "completed_iterations": result.completed_iterations,
        "sampler_runtime_seconds": result.runtime_seconds,
        "declared_total_budget_seconds": declared_budget,
        "sampler_budget_seconds": budget,
    }


def run_row(
    plan_path: Path,
    index: int,
    output_root: Path,
    config_path: Path,
    profile: str,
) -> None:
    plan = _plan(plan_path)
    execution = _execution_config(config_path)
    _validate_plan_execution_binding(plan, config_path, execution, profile)
    preflight_report = _validated_preflight(output_root)
    runs = plan["runs"]
    if not 0 <= index < len(runs):
        raise IndexError("run index outside plan")
    run = runs[index]
    spec = next(row for row in plan["datasets"] if row["dataset_id"] == run["dataset_id"])
    if execution.get("metadata_calibration_gate", {}).get("required", False):
        from .metadata_calibration_v2 import validate_gate
        validate_gate(output_root, config_path, profile, prepared=True)
    _, candidate_path, prepared_path = _dataset_paths(output_root, str(run["dataset_id"]))
    if not prepared_path.is_file():
        raise RuntimeError(f"dataset has not been prepared: {run['dataset_id']}")
    prepared = _load_json(prepared_path)
    identity = _run_identity(
        plan, run, prepared, execution, profile, preflight_report
    )
    output = output_root / "runs" / f"{run['run_id']}.json"
    if _complete(output, identity):
        return
    mpi = int(run.get("mpi_ranks", 1)) > 1
    rank = 0
    if mpi:
        from mpi4py import MPI

        rank = int(MPI.COMM_WORLD.Get_rank())
        if MPI.COMM_WORLD.Get_size() != int(run["mpi_ranks"]):
            raise RuntimeError("MPI communicator size does not match the planned rank count")
    table = load_candidate_table(candidate_path, expected_sha256=prepared["candidate_table_sha256"])
    dataset = load_experiment_artifact(output_root / "datasets" / str(run["dataset_id"]))
    started = time.perf_counter()
    try:
        result = _run_result(run, spec, table, execution, profile, dataset, mpi=mpi)
    except Exception as error:
        runtime = time.perf_counter() - started
        if rank == 0:
            _atomic_json(
                output,
                {
                    "schema": "ordersmc-run-result-v3",
                    "status": "failed",
                    "identity_sha256": identity,
                    "run": run,
                    "dataset": spec,
                    "plan_sha256": plan["plan_sha256"],
                    "profile": profile,
                    "candidate_table_sha256": prepared["candidate_table_sha256"],
                    "runtime_seconds": runtime,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "completed_utc": _utc(),
                },
            )
        raise
    runtime = time.perf_counter() - started
    if result is None:
        return
    probabilities, best_edges, diagnostics = result
    true_edges = set((str(u), str(v)) for u, v in dataset.dag.edges())
    threshold = float(execution["ordersmc"]["posterior_threshold"])
    record: dict[str, object] = {
        "schema": "ordersmc-run-result-v3",
        "status": "complete",
        "identity_sha256": identity,
        "run": run,
        "dataset": spec,
        "plan_sha256": plan["plan_sha256"],
        "execution_config_sha256": _sha(execution),
        "preflight_identity_sha256": preflight_report["identity_sha256"],
        "source_sha256": preflight_report["source_sha256"],
        "profile": profile,
        "candidate_table_sha256": prepared["candidate_table_sha256"],
        "target": run["target"],
        "posterior_threshold": threshold,
        "edge_probabilities": _all_edges(table.nodes, probabilities),
        "best_edges": [list(edge) for edge in sorted(best_edges)],
        "metrics": _metrics(
            table.nodes,
            true_edges,
            probabilities,
            threshold,
            dataset.variable_types,
        ),
        "runtime_seconds": runtime,
        "budget_seconds": run.get("wall_seconds"),
        "budget_overrun": bool(
            run.get("wall_seconds") is not None and runtime > 1.05 * float(run["wall_seconds"])
        ),
        "diagnostics": diagnostics,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "mpi_ranks": int(run.get("mpi_ranks", 1)),
            "mpi_transport": {
                "slurm_mpi_type": os.environ.get("ORDERSMC_SLURM_MPI_TYPE", "pmi2"),
                "ucx_tls": os.environ.get("UCX_TLS"),
            },
            "thread_limits": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "completed_utc": _utc(),
    }
    if rank == 0:
        _atomic_json(output, record)


def _compressed_indices(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current == previous + 1:
            previous = current
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = current
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _sharded_indices(
    indices: Sequence[int], shard_size: int
) -> list[tuple[int, str]]:
    """Map global plan indices to bounded zero-based Slurm array shards."""

    if isinstance(shard_size, bool) or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    groups: dict[int, list[int]] = {}
    for raw in sorted(set(map(int, indices))):
        if raw < 0:
            raise ValueError("indices must be non-negative")
        offset = (raw // shard_size) * shard_size
        groups.setdefault(offset, []).append(raw - offset)
    return [
        (offset, _compressed_indices(relative))
        for offset, relative in sorted(groups.items())
    ]


def missing_indices(
    plan_path: Path,
    output_root: Path,
    config_path: Path,
    profile: str,
    kind: str,
    mpi_ranks: int | None = None,
) -> list[int]:
    """Return only work whose immutable completion record is absent or stale."""

    plan = _plan(plan_path)
    execution = _execution_config(config_path)
    _validate_plan_execution_binding(plan, config_path, execution, profile)
    preflight_report = _validated_preflight(output_root)
    missing: list[int] = []
    if kind == "datasets":
        for index, spec in enumerate(plan["datasets"]):
            _, candidate_path, prepared_path = _dataset_paths(
                output_root, str(spec["dataset_id"])
            )
            identity = _dataset_identity(
                plan, spec, execution, profile, preflight_report
            )
            if not candidate_path.is_file() or not _complete(prepared_path, identity):
                missing.append(index)
        return missing
    if kind != "runs":
        raise ValueError("kind must be datasets or runs")
    for index, run in enumerate(plan["runs"]):
        if mpi_ranks is not None and int(run.get("mpi_ranks", 1)) != mpi_ranks:
            continue
        _, candidate_path, prepared_path = _dataset_paths(
            output_root, str(run["dataset_id"])
        )
        if not candidate_path.is_file() or not prepared_path.is_file():
            missing.append(index)
            continue
        prepared = _load_json(prepared_path)
        identity = _run_identity(
            plan, run, prepared, execution, profile, preflight_report
        )
        result_path = output_root / "runs" / f"{run['run_id']}.json"
        if not _complete(result_path, identity):
            missing.append(index)
    return missing


def _probability_map(row: Mapping[str, object]) -> dict[tuple[str, str], float]:
    return {
        (str(edge["parent"]), str(edge["child"])): float(edge["probability"])
        for edge in row["edge_probabilities"]
    }


def aggregate(plan_path: Path, output_root: Path) -> None:
    plan = _plan(plan_path)
    records: list[dict[str, Any]] = []
    raw: dict[str, dict[str, Any]] = {}
    for run in plan["runs"]:
        path = output_root / "runs" / f"{run['run_id']}.json"
        if not path.is_file():
            records.append(
                {
                    "run_id": run["run_id"],
                    "dataset_id": run["dataset_id"],
                    "method": run["method"],
                    "target": run["target"],
                    "status": "missing",
                }
            )
            continue
        result = _load_json(path)
        raw[str(run["run_id"])] = result
        if result.get("status") != "complete":
            records.append(
                {
                    "run_id": run["run_id"],
                    "dataset_id": run["dataset_id"],
                    "method": run["method"],
                    "target": run["target"],
                    "role": run["role"],
                    "status": result.get("status", "failed"),
                    "failure_type": result.get("failure_type"),
                    "failure_message": result.get("failure_message"),
                }
            )
            continue
        flat = {
            "run_id": run["run_id"],
            "dataset_id": run["dataset_id"],
            "method": run["method"],
            "target": run["target"],
            "role": run["role"],
            "dimension": result["dataset"]["dimension"],
            "seed_index": run["seed_index"],
            "wall_seconds": run["wall_seconds"],
            "mpi_ranks": run["mpi_ranks"],
            "status": result["status"],
            "runtime_seconds": result["runtime_seconds"],
            **result["metrics"],
        }
        for field in (
            "cell_id",
            "role",
            "metadata_profile",
            "data_variant",
            "causal_regime",
            "negative_control",
            "sample_multiplier",
            "pairing_id",
            "n_train",
        ):
            if field in result["dataset"]:
                flat[f"dataset_{field}" if field == "role" else field] = result["dataset"][field]
        records.append(flat)
    if plan["experiment"] in {"e1", "v1", "v3", "v3-clg"}:
        references = {
            row["dataset_id"]: _probability_map(raw[row["run_id"]])
            for row in plan["runs"]
            if row["method"] == "exact-order-dp"
            and row["run_id"] in raw
            and raw[row["run_id"]].get("status") == "complete"
        }
        for flat in records:
            if flat.get("status") != "complete" or flat["dataset_id"] not in references:
                continue
            current = _probability_map(raw[flat["run_id"]])
            reference = references[flat["dataset_id"]]
            keys = sorted(reference)
            flat["pip_rmse_to_exact_order"] = math.sqrt(
                sum((current.get(key, 0.0) - reference[key]) ** 2 for key in keys) / len(keys)
            )
    if plan["experiment"] == "e3":
        reference_runs: dict[tuple[str, str], list[dict[tuple[str, str], float]]] = {}
        for run in plan["runs"]:
            if run["role"] == "reference" and run["run_id"] in raw and raw[run["run_id"]].get("status") == "complete":
                reference_runs.setdefault((run["dataset_id"], run["target"]), []).append(
                    _probability_map(raw[run["run_id"]])
                )
        references: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        for key, maps in reference_runs.items():
            edges = sorted(maps[0])
            references[key] = {
                edge: float(np.mean([values[edge] for values in maps])) for edge in edges
            }
        for flat in records:
            key = (flat.get("dataset_id"), flat.get("target"))
            if flat.get("status") != "complete" or flat.get("role") == "reference" or key not in references:
                continue
            current = _probability_map(raw[flat["run_id"]])
            reference = references[key]
            flat["pip_rmse_to_independent_target_reference"] = math.sqrt(
                sum((current.get(edge, 0.0) - probability) ** 2 for edge, probability in reference.items())
                / len(reference)
            )

    for flat in records:
        if flat.get("status") != "complete":
            continue
        prepared_path = output_root / "datasets" / str(flat["dataset_id"]) / "prepared.json"
        if prepared_path.is_file():
            prepared = _load_json(prepared_path)
            flat["candidate_build_seconds"] = float(prepared["candidate_build_seconds"])
            flat["end_to_end_seconds"] = float(prepared["candidate_build_seconds"]) + float(flat["runtime_seconds"])
    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in records for key in row})
    with (analysis / "run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    complete = [row for row in records if row.get("status") == "complete"]
    metric_names = sorted(
        {
            key
            for row in complete
            for key, value in row.items()
            if isinstance(value, (int, float))
            and key
            not in {"seed_index", "dimension", "wall_seconds", "mpi_ranks"}
        }
    )
    summary: list[dict[str, object]] = []
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    # Keep heterogeneous metadata arms/sample sizes/controls separate. Existing
    # non-metadata experiments retain their historical grouping keys.
    stratum_fields = (
        "dataset_role", "cell_id", "metadata_profile", "data_variant",
        "causal_regime", "negative_control", "n_train", "sample_multiplier",
    ) if plan["experiment"] == "s2" else ()
    for row in complete:
        key = (row.get("dimension"), row.get("method"), row.get("wall_seconds"), row.get("mpi_ranks"))
        key += tuple(row.get(field) for field in stratum_fields)
        groups.setdefault(key, []).append(row)
    bootstrap_rng = np.random.Generator(np.random.PCG64DXSM(int(plan["master_seed"])))
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        for metric in metric_names:
            by_dataset: dict[str, list[float]] = {}
            defined_runs = 0
            for row in rows:
                if row.get(metric) is not None and np.isfinite(float(row[metric])):
                    by_dataset.setdefault(str(row["dataset_id"]), []).append(float(row[metric]))
                    defined_runs += 1
            values = np.asarray(
                [np.mean(seed_values) for seed_values in by_dataset.values()], dtype=float
            )
            if len(values):
                if len(values) == 1:
                    ci_lower = ci_upper = float(values[0])
                else:
                    draws = bootstrap_rng.integers(0, len(values), size=(2_000, len(values)))
                    boot = np.mean(values[draws], axis=1)
                    ci_lower, ci_upper = map(float, np.quantile(boot, [0.025, 0.975]))
                summary.append(
                    {
                        "dimension": key[0],
                        "method": key[1],
                        "wall_seconds": key[2],
                        "mpi_ranks": key[3],
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                        "n_datasets": len(values),
                        "seed_aggregation": "mean-within-dataset-before-bootstrap",
                        **dict(zip(stratum_fields, key[4:])),
                        "n_total_datasets": len({row["dataset_id"] for row in rows}),
                        "n_defined_runs": defined_runs,
                        "n_undefined_runs": len(rows)-defined_runs,
                    }
                )
            else:
                # Undefined recall on an empty truth graph is not zero recall.
                # Keep an explicit blank-valued row and its denominator.
                summary.append({
                    "dimension": key[0], "method": key[1], "wall_seconds": key[2],
                    "mpi_ranks": key[3], "metric": metric, "mean": None,
                    "ci_lower": None, "ci_upper": None, "standard_error": None,
                    "n_datasets": 0, "seed_aggregation": "mean-within-dataset-before-bootstrap",
                    **dict(zip(stratum_fields, key[4:])),
                    "n_total_datasets": len({row["dataset_id"] for row in rows}),
                    "n_defined_runs": 0, "n_undefined_runs": len(rows),
                })
    with (analysis / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["dimension", "method", "wall_seconds", "mpi_ranks", "metric", "mean", "ci_lower", "ci_upper", "standard_error", "n_datasets", "seed_aggregation"]
        columns += list(stratum_fields) + ["n_total_datasets", "n_defined_runs", "n_undefined_runs"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    completion_rows: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in plan["runs"]}):
        planned = [row for row in plan["runs"] if row["method"] == method]
        completed_count = sum(
            1
            for row in planned
            if row["run_id"] in raw and raw[row["run_id"]].get("status") == "complete"
        )
        completion_rows.append(
            {
                "method": method,
                "planned_runs": len(planned),
                "completed_runs": completed_count,
                "completion_rate": completed_count / len(planned),
            }
        )
    with (analysis / "completion_by_method.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "planned_runs", "completed_runs", "completion_rate"],
        )
        writer.writeheader()
        writer.writerows(completion_rows)

    if plan["experiment"] == "e4":
        scaling_rows: list[dict[str, object]] = []
        for dimension in sorted({int(row["dimension"]) for row in complete}):
            medians: dict[int, float] = {}
            for ranks in sorted({int(row["mpi_ranks"]) for row in complete if int(row["dimension"]) == dimension}):
                dataset_medians = []
                for dataset_id in sorted({str(row["dataset_id"]) for row in complete if int(row["dimension"]) == dimension}):
                    values = [
                        float(row["runtime_seconds"])
                        for row in complete
                        if int(row["dimension"]) == dimension
                        and int(row["mpi_ranks"]) == ranks
                        and row["dataset_id"] == dataset_id
                    ]
                    if values:
                        dataset_medians.append(float(np.median(values)))
                if dataset_medians:
                    medians[ranks] = float(np.median(dataset_medians))
            baseline = medians.get(1)
            if baseline is not None:
                for ranks, runtime in sorted(medians.items()):
                    speedup = baseline / runtime
                    scaling_rows.append(
                        {
                            "dimension": dimension,
                            "mpi_ranks": ranks,
                            "median_inference_seconds": runtime,
                            "speedup": speedup,
                            "parallel_efficiency": speedup / ranks,
                            "region": "useful" if ranks <= 32 else "saturation",
                        }
                    )
        with (analysis / "scaling.csv").open("w", newline="", encoding="utf-8") as handle:
            columns = ["dimension", "mpi_ranks", "median_inference_seconds", "speedup", "parallel_efficiency", "region"]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(scaling_rows)
    _atomic_json(
        analysis / "completion.json",
        {
            "schema": "ordersmc-analysis-completion-v3",
            "status": "complete" if len(complete) == len(plan["runs"]) else "incomplete",
            "planned_runs": len(plan["runs"]),
            "completed_runs": len(complete),
            "missing_or_failed_runs": len(plan["runs"]) - len(complete),
            "created_utc": _utc(),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ordersmc_experiments.execution")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("preflight")
    command.add_argument("--output-root", type=Path, required=True)
    command = commands.add_parser("write-plan")
    command.add_argument("experiment", choices=EXPERIMENT_IDS)
    command.add_argument("--master-seed", type=int, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--profile", choices=["full", "pilot", "smoke"], default="full")
    command = commands.add_parser("count")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--kind", choices=["datasets", "runs"], required=True)
    command.add_argument("--mpi-ranks", type=int)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--index", type=int, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--profile", choices=["full", "pilot", "smoke"], default="full")
    command = commands.add_parser("audit-prepared")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--profile", choices=["full", "pilot", "smoke"], default="full")
    command.add_argument("--minimum-recall", type=float, default=0.95)
    for name in ("missing", "missing-shards"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument(
            "--profile", choices=["full", "pilot", "smoke"], default="full"
        )
        command.add_argument("--kind", choices=["datasets", "runs"], required=True)
        command.add_argument("--mpi-ranks", type=int)
        if name == "missing-shards":
            command.add_argument("--shard-size", type=int, default=1000)
    command = commands.add_parser("aggregate")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        preflight(args.output_root)
    elif args.command == "write-plan":
        write_plan(args.experiment, args.master_seed, args.output, args.profile)
    elif args.command == "count":
        rows = _plan(args.plan)[args.kind]
        if args.mpi_ranks is not None:
            indices = [str(index) for index, row in enumerate(rows) if int(row.get("mpi_ranks", 1)) == args.mpi_ranks]
            print(",".join(indices))
        else:
            print(len(rows))
    elif args.command == "prepare":
        prepare_dataset(args.plan, args.index, args.output_root, args.config, args.profile)
    elif args.command == "run":
        run_row(args.plan, args.index, args.output_root, args.config, args.profile)
    elif args.command == "audit-prepared":
        audit_prepared_datasets(
            args.plan,
            args.output_root,
            args.config,
            args.profile,
            minimum_recall=args.minimum_recall,
        )
    elif args.command == "missing":
        print(
            _compressed_indices(
                missing_indices(
                    args.plan,
                    args.output_root,
                    args.config,
                    args.profile,
                    args.kind,
                    args.mpi_ranks,
                )
            )
        )
    elif args.command == "missing-shards":
        missing = missing_indices(
            args.plan,
            args.output_root,
            args.config,
            args.profile,
            args.kind,
            args.mpi_ranks,
        )
        for offset, expression in _sharded_indices(missing, args.shard_size):
            print(f"{offset}\t{expression}")
    elif args.command == "aggregate":
        aggregate(args.plan, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
