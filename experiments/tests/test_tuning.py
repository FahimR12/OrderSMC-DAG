from __future__ import annotations

import json
from pathlib import Path

from ordersmc_experiments.tuning import (
    MCMC_VARIANTS,
    SMC_VARIANTS,
    build_tuning_plan,
)


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "analysis").mkdir(parents=True)
    (root / "analysis" / "completion.json").write_text(
        json.dumps({"planned_runs": 180, "completed_runs": 180}), encoding="utf-8"
    )
    datasets = [
        {"dataset_id": f"e1-d{dimension:02d}-r{replicate:03d}", "dimension": dimension, "replicate": replicate}
        for dimension in (8, 12)
        for replicate in range(10)
    ]
    runs = [
        {
            "dataset_id": dataset["dataset_id"],
            "method": "ordersmc",
            "seed_index": seed_index,
            "method_seed": {"seed": 1000 + seed_index},
        }
        for dataset in datasets
        for seed_index in range(4)
    ]
    (root / "plan.json").write_text(
        json.dumps({"plan_sha256": "source-plan", "datasets": datasets, "runs": runs}),
        encoding="utf-8",
    )
    return root


def test_tuning_plan_is_deterministic_and_prespecified(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = build_tuning_plan(source)
    second = build_tuning_plan(source)
    assert first == second
    assert first["run_count"] == 60
    assert {row["seed_index"] for row in first["runs"]} == {0, 1}
    assert {row["dimension"] for row in first["runs"]} == {8, 12}
    assert {
        int(row["dataset_id"].split("-r")[-1]) for row in first["runs"]
    } == {0, 1, 2}
    assert {row["variant"] for row in first["runs"] if row["method"] == "ordersmc"} == set(SMC_VARIANTS)
    assert {row["variant"] for row in first["runs"] if row["method"] == "order-mcmc"} == set(MCMC_VARIANTS)
