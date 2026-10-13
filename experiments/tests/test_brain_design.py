from __future__ import annotations

from ordersmc_experiments.brain_design import build_design_manifest
from ordersmc_experiments.design import build_experiment_plan


def test_g1_keeps_mixed_capabilities_and_installation_controls_distinct() -> None:
    plan = build_design_manifest("g1", "smoke")
    assert plan["cell_count"] == 11
    assert plan["method_count"] == 11
    assert plan["run_count"] == 121
    assert sum(row["role"] == "required-mixed-capability" for row in plan["cells"]) == 9
    assert sum(row["role"] == "installation-control-not-scientific" for row in plan["cells"]) == 2


def test_v1_full_design_is_lean_and_exact() -> None:
    plan = build_design_manifest("v1", "full")
    assert plan["cell_count"] == 9
    assert plan["dataset_count"] == 90
    assert plan["exact_run_count"] == 90
    assert plan["stochastic_run_count"] == 720
    assert plan["run_count"] == 810
    assert len({row["cell_id"] for row in plan["cells"]}) == 9
    assert {row["variable_type_regime"] for row in plan["cells"]} == {
        "balanced_mixed",
        "continuous_dominant_mixed",
        "discrete_dominant_mixed",
    }


def test_v2_adds_paper_aligned_robustness_without_crossing_every_type() -> None:
    plan = build_design_manifest("v2", "full")
    assert plan["cell_count"] == 16
    assert plan["dataset_count"] == 320
    assert plan["run_count"] == 5120
    assert sum(row["role"] == "type-regime-sensitivity" for row in plan["cells"]) == 2
    assert sum(row["role"] == "mixed-null-negative-control" for row in plan["cells"]) == 1
    assert {row["continuous_noise"] for row in plan["cells"]} == {
        "gaussian",
        "student_t_df3",
        "student_t_df5",
        "centered_gamma_shape1_scale2",
    }
    assert all("mixed" in str(row["variable_type_regime"]) for row in plan["cells"])


def test_design_declares_dagbagm_aligned_metrics() -> None:
    from ordersmc_experiments.brain_design import load_design

    metrics = load_design()["evaluation_metrics"]
    expected = ["tpr", "fdr", "f1"]
    assert metrics["headline_graph_recovery"]["skeleton"] == expected
    assert metrics["headline_graph_recovery"]["directed_cross_type"] == expected
    assert metrics["definitions"]["directed_reversal"] == (
        "one_false_positive_plus_one_false_negative"
    )
    assert metrics["mixed_type_strata"]["ordered_cross_type_directions"] == 6


def test_v3_keeps_native_and_semisynthetic_benchmarks_separate() -> None:
    plan = build_design_manifest("v3", "full")
    assert plan["cell_count"] == 14
    assert plan["dataset_count"] == 280
    assert plan["exact_run_count"] == 100
    assert plan["stochastic_run_count"] == 4480
    assert plan["comparator_run_count"] == 680
    assert plan["run_count"] == 5260
    roles = {row["role"] for row in plan["cells"]}
    assert roles == {
        "restricted-native-mixed-benchmark",
        "topology-preserving-semi-synthetic-mixed",
    }


def test_design_manifests_are_deterministic_and_nonexecutable() -> None:
    first = build_design_manifest("v1", "smoke")
    second = build_design_manifest("v1", "smoke")
    assert first == second
    assert first["status"] == "design-only-not-executable"
    assert first["master_seed"] == 20260903


def test_v3_executable_pilot_includes_native_and_shared_score_bnlearn_controls() -> None:
    plan = build_experiment_plan("v3", master_seed=20260903, profile="pilot")
    assert len(plan.datasets) == 42
    assert len(plan.runs) == 453
    methods = {row.method for row in plan.runs}
    assert {
        "ordersmc",
        "order-mcmc",
        "bnlearn-hc-bic-cg",
        "bnlearn-tabu-bic-cg",
        "bnlearn-hc-shared-score",
        "bnlearn-tabu-shared-score",
    } <= methods
    assert all(
        row.dataset_id.startswith("v3-clgaussian.test-")
        for row in plan.runs
        if row.method.endswith("-bic-cg")
    )
    assert {row.mpi_ranks for row in plan.runs if row.method == "ordersmc"} == {64}


def test_v3_smoke_caps_rows_but_records_the_declared_benchmark_size() -> None:
    plan = build_experiment_plan("v3", master_seed=20260903, profile="smoke")
    assert all(row.n_train <= max(100, 5 * row.dimension) for row in plan.datasets)
    assert max(row.benchmark_n_train for row in plan.datasets) == 20_000
    assert any(row.n_train < row.benchmark_n_train for row in plan.datasets)


def test_v3_clg_corrective_plan_is_held_out_and_restricted_to_native_clg() -> None:
    plan = build_experiment_plan("v3-clg", master_seed=20260917, profile="full")
    assert plan.experiment == "v3-clg"
    assert plan.master_seed == 20260917
    assert len(plan.datasets) == 60
    assert len(plan.runs) == 1260
    assert all(
        row.data_kind == "native_mixed_conditional_gaussian"
        for row in plan.datasets
    )
    assert plan.settings["corrective_validation"] is True
    assert plan.settings["score_selection_basis"] == (
        "statistical-model-class-not-dataset-name"
    )
    smoke = build_experiment_plan("v3-clg", master_seed=20260916, profile="smoke")
    assert [row.n_train for row in smoke.datasets] == [250, 1_000, 5_000]
    assert all(row.n_train == row.benchmark_n_train for row in smoke.datasets)
