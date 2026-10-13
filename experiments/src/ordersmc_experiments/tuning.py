"""Prespecified E1 pilot tuning on completed exact-reference datasets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ordersmc import sample
from ordersmc.persistence import candidate_table_content_sha256, load_candidate_table

from .execution import (
    DEFAULT_CONFIG,
    _execution_config,
    _json_safe,
    _load_json,
    _profile_config,
    _source_sha256,
    _validated_preflight,
)
from .order_mcmc import OrderMCMCConfig, order_mcmc


TUNING_SCHEMA = "ordersmc-e1-pilot-tuning-v1"
SMC_VARIANTS: Mapping[str, Mapping[str, int]] = {
    "smc-384x120x12": {
        "particles": 384,
        "num_betas": 120,
        "mh_steps": 12,
        "mh_steps_resample": 30,
    },
    "smc-512x100x10": {
        "particles": 512,
        "num_betas": 100,
        "mh_steps": 10,
        "mh_steps_resample": 25,
    },
    "smc-768x080x08": {
        "particles": 768,
        "num_betas": 80,
        "mh_steps": 8,
        "mh_steps_resample": 20,
    },
}
MCMC_VARIANTS: Mapping[str, Mapping[str, int]] = {
    "mcmc-thin050-chain270": {"thin": 50, "wall_seconds": 270},
    "mcmc-thin100-chain285": {"thin": 100, "wall_seconds": 285},
}


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _edge_map(record: Mapping[str, object]) -> dict[tuple[str, str], float]:
    return {
        (str(row["parent"]), str(row["child"])): float(row["probability"])
        for row in record["edge_probabilities"]
    }


def build_tuning_plan(source_root: Path) -> dict[str, object]:
    completion = _load_json(source_root / "analysis" / "completion.json")
    if completion.get("completed_runs") != completion.get("planned_runs"):
        raise RuntimeError("source E1 pilot is incomplete")
    source_plan = _load_json(source_root / "plan.json")
    selected = [
        row
        for row in source_plan["datasets"]
        if int(row["replicate"]) in {0, 1, 2}
    ]
    if len(selected) != 6 or {int(row["dimension"]) for row in selected} != {8, 12}:
        raise RuntimeError("expected three predeclared E1 datasets per dimension")
    source_runs = source_plan["runs"]
    rows: list[dict[str, object]] = []
    for dataset in selected:
        seeds = [
            run
            for run in source_runs
            if run["dataset_id"] == dataset["dataset_id"]
            and run["method"] == "ordersmc"
            and int(run["seed_index"]) in {0, 1}
        ]
        if len(seeds) != 2:
            raise RuntimeError("source plan lacks the two tuning seeds")
        for source_seed in seeds:
            for variant, settings in SMC_VARIANTS.items():
                rows.append(
                    {
                        "run_id": f"{dataset['dataset_id']}-{variant}-s{int(source_seed['seed_index']):02d}",
                        "dataset_id": dataset["dataset_id"],
                        "dimension": dataset["dimension"],
                        "method": "ordersmc",
                        "variant": variant,
                        "settings": dict(settings),
                        "seed_index": source_seed["seed_index"],
                        "method_seed": source_seed["method_seed"],
                    }
                )
            for variant, settings in MCMC_VARIANTS.items():
                rows.append(
                    {
                        "run_id": f"{dataset['dataset_id']}-{variant}-s{int(source_seed['seed_index']):02d}",
                        "dataset_id": dataset["dataset_id"],
                        "dimension": dataset["dimension"],
                        "method": "order-mcmc",
                        "variant": variant,
                        "settings": dict(settings),
                        "seed_index": source_seed["seed_index"],
                        "method_seed": source_seed["method_seed"],
                    }
                )
    body: dict[str, object] = {
        "schema": TUNING_SCHEMA,
        "source_root": str(source_root.resolve()),
        "source_plan_sha256": source_plan["plan_sha256"],
        "source_completion": completion,
        "selection": {
            "datasets": "replicates 0,1,2 at each of d=8,12; selected without outcomes",
            "seeds": [0, 1],
            "ordersmc_eligibility": {
                "mean_pip_rmse_max": 0.012,
                "p95_pip_rmse_max": 0.020,
                "mean_runtime_seconds_max": 300.0,
                "choice": "lowest pooled mean PIP RMSE, then runtime",
            },
            "order_mcmc_eligibility": {
                "mean_pip_rmse_max": 0.005,
                "p95_pip_rmse_max": 0.010,
                "mean_runtime_seconds_max": 305.0,
                "choice": "lowest runtime, then PIP RMSE",
            },
        },
        "runs": rows,
        "run_count": len(rows),
    }
    body["plan_sha256"] = _canonical_sha(body)
    return body


def write_tuning_plan(source_root: Path, destination: Path) -> None:
    value = build_tuning_plan(source_root)
    if destination.exists():
        if _canonical_sha(_load_json(destination)) != _canonical_sha(value):
            raise FileExistsError(f"existing tuning plan differs: {destination}")
        return
    _atomic_json(destination, value)


def _validated_plan(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    claimed = value.get("plan_sha256")
    body = dict(value)
    body.pop("plan_sha256", None)
    if value.get("schema") != TUNING_SCHEMA or claimed != _canonical_sha(body):
        raise ValueError("invalid E1 tuning plan")
    return value


def run_tuning_row(
    plan_path: Path, index: int, output_root: Path, config_path: Path
) -> None:
    plan = _validated_plan(plan_path)
    preflight = _validated_preflight(output_root)
    execution = _execution_config(config_path)
    if not 0 <= index < len(plan["runs"]):
        raise IndexError("tuning row index outside plan")
    row = plan["runs"][index]
    source_root = Path(plan["source_root"])
    dataset_root = source_root / "datasets" / str(row["dataset_id"])
    prepared = _load_json(dataset_root / "prepared.json")
    table = load_candidate_table(
        dataset_root / "candidate.json.gz",
        expected_sha256=str(prepared["candidate_table_sha256"]),
    )
    exact_candidates = list(
        (source_root / "runs").glob(f"{row['dataset_id']}-exact-order.json")
    )
    if len(exact_candidates) != 1:
        raise RuntimeError("source exact-order reference is missing")
    exact = _edge_map(_load_json(exact_candidates[0]))
    identity = _canonical_sha(
        {
            "plan_sha256": plan["plan_sha256"],
            "row": row,
            "candidate_table_sha256": candidate_table_content_sha256(table),
            "execution_config_sha256": _canonical_sha(execution),
            "preflight_identity_sha256": preflight["identity_sha256"],
        }
    )
    destination = output_root / "runs" / f"{row['run_id']}.json"
    if destination.is_file():
        existing = _load_json(destination)
        if existing.get("status") == "complete" and existing.get("identity_sha256") == identity:
            return
    seed = int(row["method_seed"]["seed"])
    dataset_spec = next(
        item for item in _load_json(source_root / "plan.json")["datasets"]
        if item["dataset_id"] == row["dataset_id"]
    )
    started = time.perf_counter()
    try:
        if row["method"] == "ordersmc":
            config = _profile_config(dataset_spec, execution, "full", seed).with_overrides(
                **dict(row["settings"])
            )
            result = sample(
                table,
                config=config,
                backend="local",
                verbose=False,
                provenance_policy="require",
            )
            probabilities = result.edge_probs
            diagnostics = {
                "effective_sample_size": result.effective_sample_size,
                "history": result.history,
            }
        else:
            settings = row["settings"]
            result = order_mcmc(
                table,
                OrderMCMCConfig(
                    iterations=1_000_000_000,
                    burnin=250_000_000,
                    thin=int(settings["thin"]),
                    seed=seed,
                    wall_seconds=float(settings["wall_seconds"]),
                    wall_burnin_fraction=0.25,
                ),
            )
            probabilities = result.edge_probs
            diagnostics = {
                "effective_sample_size": result.effective_sample_size,
                "acceptance_rate": result.acceptance_rate,
                "stored_samples": result.stored_samples,
                "completed_iterations": result.completed_iterations,
                "sampler_runtime_seconds": result.runtime_seconds,
            }
        runtime = time.perf_counter() - started
        edges = sorted(exact)
        rmse = math.sqrt(
            sum((float(probabilities.get(edge, 0.0)) - exact[edge]) ** 2 for edge in edges)
            / len(edges)
        )
        record: dict[str, object] = {
            "schema": "ordersmc-e1-pilot-tuning-result-v1",
            "status": "complete",
            "identity_sha256": identity,
            "plan_sha256": plan["plan_sha256"],
            "row": row,
            "candidate_table_sha256": candidate_table_content_sha256(table),
            "source_sha256": _source_sha256(),
            "preflight_identity_sha256": preflight["identity_sha256"],
            "pip_rmse_to_exact_order": rmse,
            "runtime_seconds": runtime,
            "diagnostics": diagnostics,
            "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    except Exception as error:
        record = {
            "schema": "ordersmc-e1-pilot-tuning-result-v1",
            "status": "failed",
            "identity_sha256": identity,
            "plan_sha256": plan["plan_sha256"],
            "row": row,
            "runtime_seconds": time.perf_counter() - started,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }
        _atomic_json(destination, record)
        raise
    _atomic_json(destination, record)


def _recommend(summary: list[dict[str, object]], method: str, rule: Mapping[str, object]) -> dict[str, object]:
    pooled: list[dict[str, object]] = []
    for variant in sorted({str(row["variant"]) for row in summary if row["method"] == method}):
        rows = [row for row in summary if row["method"] == method and row["variant"] == variant]
        mean_rmse = float(np.mean([row["mean_pip_rmse"] for row in rows]))
        p95_rmse = float(max(row["p95_pip_rmse"] for row in rows))
        mean_runtime = float(np.mean([row["mean_runtime_seconds"] for row in rows]))
        eligible = (
            mean_rmse <= float(rule["mean_pip_rmse_max"])
            and p95_rmse <= float(rule["p95_pip_rmse_max"])
            and mean_runtime <= float(rule["mean_runtime_seconds_max"])
        )
        pooled.append(
            {
                "variant": variant,
                "mean_pip_rmse": mean_rmse,
                "worst_dimension_p95_pip_rmse": p95_rmse,
                "mean_runtime_seconds": mean_runtime,
                "eligible": eligible,
            }
        )
    eligible = [row for row in pooled if row["eligible"]]
    if method == "ordersmc":
        eligible.sort(key=lambda row: (row["mean_pip_rmse"], row["mean_runtime_seconds"], row["variant"]))
    else:
        eligible.sort(key=lambda row: (row["mean_runtime_seconds"], row["mean_pip_rmse"], row["variant"]))
    return {"method": method, "candidates": pooled, "selected": eligible[0]["variant"] if eligible else None}


def aggregate_tuning(plan_path: Path, output_root: Path) -> None:
    plan = _validated_plan(plan_path)
    records = []
    for row in plan["runs"]:
        path = output_root / "runs" / f"{row['run_id']}.json"
        if path.is_file():
            records.append(_load_json(path))
    complete = [row for row in records if row.get("status") == "complete"]
    summary: list[dict[str, object]] = []
    for method in ("ordersmc", "order-mcmc"):
        for variant in sorted({str(row["row"]["variant"]) for row in complete if row["row"]["method"] == method}):
            for dimension in (8, 12):
                rows = [
                    row for row in complete
                    if row["row"]["method"] == method
                    and row["row"]["variant"] == variant
                    and int(row["row"]["dimension"]) == dimension
                ]
                if rows:
                    rmses = np.asarray([float(row["pip_rmse_to_exact_order"]) for row in rows])
                    runtimes = np.asarray([float(row["runtime_seconds"]) for row in rows])
                    summary.append(
                        {
                            "method": method,
                            "variant": variant,
                            "dimension": dimension,
                            "mean_pip_rmse": float(np.mean(rmses)),
                            "p95_pip_rmse": float(np.quantile(rmses, 0.95)),
                            "mean_runtime_seconds": float(np.mean(runtimes)),
                            "runs": len(rows),
                        }
                    )
    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["method", "variant", "dimension", "mean_pip_rmse", "p95_pip_rmse", "mean_runtime_seconds", "runs"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    recommendation = {
        "schema": "ordersmc-e1-pilot-tuning-recommendation-v1",
        "status": "complete" if len(complete) == len(plan["runs"]) else "incomplete",
        "planned_runs": len(plan["runs"]),
        "completed_runs": len(complete),
        "failed_or_missing_runs": len(plan["runs"]) - len(complete),
        "ordersmc": _recommend(summary, "ordersmc", plan["selection"]["ordersmc_eligibility"]),
        "order_mcmc": _recommend(summary, "order-mcmc", plan["selection"]["order_mcmc_eligibility"]),
    }
    _atomic_json(analysis / "recommendation.json", recommendation)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ordersmc_experiments.tuning")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("plan")
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("count")
    command.add_argument("--plan", type=Path, required=True)
    command = commands.add_parser("run")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command = commands.add_parser("aggregate")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        write_tuning_plan(args.source_root, args.output)
    elif args.command == "count":
        print(len(_validated_plan(args.plan)["runs"]))
    elif args.command == "run":
        run_tuning_row(args.plan, args.index, args.output_root, args.config)
    elif args.command == "aggregate":
        aggregate_tuning(args.plan, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
