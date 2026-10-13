"""High-particle MPI diagnostic plans derived from the executable V2 pilot.

This deliberately creates a separate, explicitly diagnostic plan. It does
not mutate the frozen single-core pilot or present an MPI run as an equal-core
comparison against the existing MCMC baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .design import build_experiment_plan


def _sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_v2_mpi_diagnostic_plan(
    *,
    master_seed: int,
    profile: str = "pilot",
    mpi_ranks: int = 64,
    method_seeds: int | None = None,
) -> dict[str, Any]:
    """Return the V2 datasets with only high-throughput MPI OrderSMC rows."""

    if isinstance(mpi_ranks, bool) or not isinstance(mpi_ranks, int) or mpi_ranks < 2:
        raise ValueError("mpi_ranks must be an integer of at least two")
    if profile not in {"smoke", "pilot"}:
        raise ValueError("the executable V2 MPI diagnostic supports smoke or pilot")

    base = build_experiment_plan("v2", master_seed=master_seed, profile=profile).to_dict()
    available_seeds = 1 + max(
        int(row["seed_index"])
        for row in base["runs"]
        if row["method"] == "ordersmc"
    )
    selected_seeds = available_seeds if method_seeds is None else int(method_seeds)
    if not 1 <= selected_seeds <= available_seeds:
        raise ValueError(
            f"method_seeds must lie in [1, {available_seeds}] for profile {profile}"
        )

    runs: list[dict[str, object]] = []
    for source in base["runs"]:
        if source["method"] != "ordersmc" or int(source["seed_index"]) >= selected_seeds:
            continue
        row = dict(source)
        row["mpi_ranks"] = mpi_ranks
        row["role"] = "mpi-diagnostic"
        row["run_id"] = f"{source['run_id']}-mpi-r{mpi_ranks:03d}"
        runs.append(row)

    body = dict(base)
    body.pop("plan_sha256", None)
    body["runs"] = runs
    body["run_count"] = len(runs)
    settings = dict(body["settings"])
    settings["diagnostic_execution"] = {
        "purpose": "fast-first-high-particle-ordersmc-diagnostics",
        "comparison_status": "resource-unmatched-diagnostic-only",
        "source_plan": "v2-pilot",
        "methods": ["ordersmc"],
        "mpi_ranks_per_run": mpi_ranks,
        "method_seeds": selected_seeds,
    }
    body["settings"] = settings
    body["plan_sha256"] = _sha(body)
    return body


def write_plan(
    destination: Path,
    *,
    master_seed: int,
    profile: str,
    mpi_ranks: int,
    method_seeds: int | None,
) -> None:
    value = build_v2_mpi_diagnostic_plan(
        master_seed=master_seed,
        profile=profile,
        mpi_ranks=mpi_ranks,
        method_seeds=method_seeds,
    )
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping) or _sha(existing) != _sha(value):
            raise FileExistsError(f"existing plan differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ordersmc_experiments.mpi_diagnostics"
    )
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--mpi-ranks", type=int, default=64)
    parser.add_argument("--method-seeds", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    write_plan(
        args.output,
        master_seed=args.master_seed,
        profile=args.profile,
        mpi_ranks=args.mpi_ranks,
        method_seeds=args.method_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
