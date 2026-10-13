from __future__ import annotations

from ordersmc_experiments.design import build_experiment_plan
from ordersmc_experiments.mpi_diagnostics import build_v2_mpi_diagnostic_plan


def test_v2_mpi_diagnostic_is_ordersmc_only_and_preserves_seeds() -> None:
    base = build_experiment_plan("v2", master_seed=20260903, profile="pilot").to_dict()
    diagnostic = build_v2_mpi_diagnostic_plan(
        master_seed=20260903,
        profile="pilot",
        mpi_ranks=64,
        method_seeds=2,
    )

    expected = [row for row in base["runs"] if row["method"] == "ordersmc"]
    assert diagnostic["datasets"] == base["datasets"]
    assert diagnostic["run_count"] == len(expected) == 96
    assert {row["method"] for row in diagnostic["runs"]} == {"ordersmc"}
    assert {row["mpi_ranks"] for row in diagnostic["runs"]} == {64}
    assert {row["role"] for row in diagnostic["runs"]} == {"mpi-diagnostic"}
    assert [row["method_seed"] for row in diagnostic["runs"]] == [
        row["method_seed"] for row in expected
    ]
    assert diagnostic["settings"]["diagnostic_execution"]["comparison_status"] == (
        "resource-unmatched-diagnostic-only"
    )
