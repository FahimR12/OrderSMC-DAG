from __future__ import annotations

import csv
import json
from pathlib import Path

from ordersmc_experiments.paper_metrics import (
    PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE,
    analyse,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _result(run: dict[str, object], probabilities: list[float]) -> dict[str, object]:
    edges = [("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")]
    return {
        "status": "complete",
        "run": run,
        "dataset": {
            "dimension": 3,
            "n_train": 100,
            "benchmark_n_train": 100,
            "max_indegree": 1,
            "cell_id": "tiny",
            "role": "primary",
        },
        "profile": "smoke",
        "posterior_threshold": 0.5,
        "runtime_seconds": 1.0,
        "metrics": {},
        "edge_probabilities": [
            {"parent": parent, "child": child, "probability": probability}
            for (parent, child), probability in zip(edges, probabilities)
        ],
    }


def test_analysis_averages_pips_before_graph_reduction_and_marks_smoke_gate(
    tmp_path: Path,
) -> None:
    dataset_id = "tiny-rep000"
    exact_run = {
        "run_id": "tiny-exact",
        "dataset_id": dataset_id,
        "method": "exact-order-dp",
        "target": "order-modular",
        "role": "reference",
        "seed_index": None,
        "wall_seconds": None,
    }
    seed_runs = [
        {
            "run_id": f"tiny-smc-{seed}",
            "dataset_id": dataset_id,
            "method": "ordersmc",
            "target": "order-modular",
            "role": "evaluation",
            "seed_index": seed,
            "wall_seconds": 300,
        }
        for seed in range(2)
    ]
    _write(
        tmp_path / "plan.json",
        {
            "master_seed": 7,
            "settings": {"profile": "smoke"},
            "runs": [exact_run, *seed_runs],
        },
    )
    _write(
        tmp_path / "analysis" / "completion.json",
        {
            "planned_runs": 3,
            "completed_runs": 3,
            "missing_or_failed_runs": 0,
        },
    )
    _write(
        tmp_path / "datasets" / dataset_id / "generator_manifest.json",
        {
            "variable_types": {"A": "continuous", "B": "binary", "C": "continuous"},
            "edges": [["A", "B"]],
            "config": {"max_indegree": 1},
        },
    )
    _write(
        tmp_path / "datasets" / dataset_id / "prepared.json",
        {
            "candidate_build_seconds": 2.0,
            "true_directed_parent_recall": 1.0,
            "complete_family_recall": 1.0,
        },
    )
    _write(
        tmp_path / "runs" / "tiny-exact.json",
        _result(exact_run, [0.8, 0.1, 0.05, 0.1, 0.05, 0.1]),
    )
    _write(
        tmp_path / "runs" / "tiny-smc-0.json",
        _result(seed_runs[0], [0.7, 0.2, 0.1, 0.1, 0.05, 0.1]),
    )
    _write(
        tmp_path / "runs" / "tiny-smc-1.json",
        _result(seed_runs[1], [0.9, 0.0, 0.0, 0.1, 0.05, 0.1]),
    )

    analyse(tmp_path)

    with (tmp_path / "analysis" / "seed_averaged_pips.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    edge = next(row for row in rows if row["parent"] == "A" and row["child"] == "B")
    assert float(edge["mean_pip"]) == 0.8
    assert edge["selected_in_seed_averaged_bayes_action"] == "True"
    with (tmp_path / "analysis" / "exact_pip_gate.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        gate = next(row for row in csv.DictReader(handle) if row["method"] == "ordersmc")
    assert gate["gate_status"] == "underpowered_smoke"
    assert gate["passes_ordersmc_gate"] == "not_evaluable"

    policy = json.loads(
        (tmp_path / "analysis" / "paper_metrics_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["milp_objective_audit_tolerance"] == (
        PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE
    )
    with (tmp_path / "analysis" / "seed_averaged_dagbagm_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        averaged = next(csv.DictReader(handle))
    assert float(averaged["bayes_action_milp_objective_audit_tolerance"]) == (
        PAPER_MILP_OBJECTIVE_AUDIT_TOLERANCE
    )
    assert averaged["bayes_action_audit_status"] == "certified"
