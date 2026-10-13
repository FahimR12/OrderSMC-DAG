"""Small command-line interface for Version 3 experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .adapter_validation import validate_algorithm_adapters
from .design import EXPERIMENT_IDS, build_experiment_plan
from .generator import (
    GeneratorFailure,
    MixedDAGConfig,
    generate_mixed_dag_dataset,
    replay_mixed_dag_artifact,
    write_mixed_dag_artifact,
)
from .registry import method_registry
from .score_validation import run_e0a_full_validation, run_e0a_score_validation


def _write_json(value: object, destination: Path | None) -> None:
    normalized = json.loads(json.dumps(value, allow_nan=False))
    payload = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination is None:
        print(payload, end="")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output {destination}")
    destination.write_text(payload, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ordersmc-experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    registry = commands.add_parser("registry", help="print the honest comparison-method registry")
    registry.add_argument("--output", type=Path)

    validate = commands.add_parser("validate-scores", aliases=["e0a"], help="run G0 score checks")
    validate.add_argument("--output", type=Path)
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--full", action="store_true")

    adapters = commands.add_parser(
        "validate-adapters", help="run exact tiny-state adapter acceptance checks"
    )
    adapters.add_argument("--output", type=Path)
    adapters.add_argument("--strict", action="store_true")

    plan = commands.add_parser("plan", help="build a deterministic Version 3 cohort plan")
    plan.add_argument("experiment", choices=EXPERIMENT_IDS)
    plan.add_argument("--master-seed", type=int, required=True)
    plan.add_argument("--output", type=Path)

    generate = commands.add_parser("generate", help="generate one replayable mixed-DAG dataset")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--d", type=int, default=20)
    generate.add_argument("--n-train", type=int, required=True)
    generate.add_argument("--n-test", type=int)
    generate.add_argument("--family", choices=["random_order", "hub"], default="random_order")
    generate.add_argument("--expected-indegree", type=float)
    generate.add_argument("--signal", choices=["weak", "moderate", "strong"], default="moderate")
    generate.add_argument("--max-indegree", type=int, default=3)
    generate.add_argument("--pilot-size", type=int, default=5_000)
    generate.add_argument("--max-graph-attempts", type=int, default=1_000)
    generate.add_argument("--max-parameter-attempts", type=int, default=100)
    generate.add_argument("--master-seed", type=int, required=True)
    generate.add_argument("--experiment-id", default="synthetic")
    generate.add_argument("--cell-id", default="default")
    generate.add_argument("--replicate-id", type=int, default=0)

    replay = commands.add_parser("replay-dataset", help="verify and regenerate a dataset bundle")
    replay.add_argument("source", type=Path)
    replay.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "registry":
        _write_json(method_registry(), args.output)
        return 0
    if args.command in {"validate-scores", "e0a"}:
        if args.full:
            report = run_e0a_full_validation()
        else:
            fixtures = [record.to_dict() for record in run_e0a_score_validation()]
            report = {"schema": "ordersmc-g0-score-smoke-v3", "fixtures": fixtures}
            report["all_passed"] = all(record["passed"] for record in fixtures)
            report["status"] = "pass" if report["all_passed"] else "fail"
        _write_json(report, args.output)
        return 1 if args.strict and not report["all_passed"] else 0
    if args.command == "validate-adapters":
        report = validate_algorithm_adapters()
        _write_json(report, args.output)
        return 1 if args.strict and not report["all_passed"] else 0
    if args.command == "plan":
        _write_json(
            build_experiment_plan(args.experiment, master_seed=args.master_seed).to_dict(),
            args.output,
        )
        return 0
    if args.command == "replay-dataset":
        dataset = replay_mixed_dag_artifact(args.source)
        result = {
            "schema": "ordersmc-dataset-replay-v3",
            "status": "pass",
            "dataset_fingerprint": dataset.manifest()["artifact_fingerprint_sha256"],
        }
        _write_json(result, args.output)
        return 0
    if args.command == "generate":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output directory {args.output}")
        request = {
            "d": args.d,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "graph_family": args.family,
            "expected_indegree": (
                1.5
                if args.expected_indegree is None and args.experiment_id == "e1"
                else 2.0 if args.expected_indegree is None else args.expected_indegree
            ),
            "signal_strength": args.signal,
            "max_indegree": args.max_indegree,
            "pilot_size": args.pilot_size,
            "max_graph_attempts": args.max_graph_attempts,
            "max_parameter_attempts": args.max_parameter_attempts,
            "master_seed": args.master_seed,
            "experiment_id": args.experiment_id,
            "cell_id": args.cell_id,
            "replicate_id": args.replicate_id,
        }
        try:
            dataset = generate_mixed_dag_dataset(MixedDAGConfig(**request))
        except (GeneratorFailure, RuntimeError, ValueError) as error:
            args.output.mkdir(parents=True, exist_ok=False)
            failure = {
                "schema": "ordersmc-mixed-dag-generation-failure-v1",
                "status": "failed",
                "request": request,
                "request_sha256": hashlib.sha256(
                    json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "diagnostics": dict(error.diagnostics) if isinstance(error, GeneratorFailure) else {},
            }
            _write_json(failure, args.output / "generation_failure.json")
            return 1
        write_mixed_dag_artifact(dataset, args.output)
        return 0
    raise RuntimeError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
