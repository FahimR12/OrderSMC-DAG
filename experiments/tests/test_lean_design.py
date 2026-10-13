from __future__ import annotations

import pytest

from ordersmc_experiments.design import build_experiment_plan


def test_e1_is_complete_pool_same_target_validation() -> None:
    plan = build_experiment_plan("e1", master_seed=20260814)
    assert len(plan.datasets) == 20
    assert len(plan.runs) == 180
    assert {row.target for row in plan.runs} == {"order-modular"}
    assert {
        row.wall_seconds for row in plan.runs if row.method in {"ordersmc", "order-mcmc"}
    } == {300}
    assert plan.settings["candidate_policy"]["e1"] == "complete-parent-sets-through-indegree-2"


def test_e2_keeps_same_target_and_end_to_end_panels_separate() -> None:
    plan = build_experiment_plan("e2", master_seed=20260814)
    assert len(plan.datasets) == 60
    assert len(plan.runs) == 960
    assert plan.settings["comparison_panels"]["same_target"] == ["ordersmc", "order-mcmc"]
    assert set(plan.settings["comparison_panels"]["end_to_end"]) == {
        "ordersmc",
        "partition-mcmc",
        "rev-mcmc",
    }


def test_brain_v1_profiles_are_executable_and_seed_disjoint_by_composition() -> None:
    smoke = build_experiment_plan("v1", master_seed=20260903, profile="smoke")
    pilot = build_experiment_plan("v1", master_seed=20260903, profile="pilot")
    full = build_experiment_plan("v1", master_seed=20260904, profile="full")
    assert (len(smoke.datasets), len(smoke.runs)) == (9, 27)
    assert (len(pilot.datasets), len(pilot.runs)) == (27, 135)
    assert (len(full.datasets), len(full.runs)) == (90, 810)
    assert {row.variable_type_regime for row in full.datasets} == {
        "balanced_mixed",
        "continuous_dominant_mixed",
        "discrete_dominant_mixed",
    }
    dataset_seed_hashes = [row.dataset_seed["sha256"] for row in full.datasets]
    assert len(dataset_seed_hashes) == len(set(dataset_seed_hashes))
    assert all(row.max_indegree == 2 for row in full.datasets)


def test_brain_v2_pilot_is_all_mixed_and_full_is_profile_blocked() -> None:
    smoke = build_experiment_plan("v2", master_seed=20260903, profile="smoke")
    pilot = build_experiment_plan("v2", master_seed=20260903, profile="pilot")
    assert (len(smoke.datasets), len(smoke.runs)) == (16, 64)
    assert (len(pilot.datasets), len(pilot.runs)) == (48, 384)
    assert {row.variable_type_regime for row in pilot.datasets} == {
        "balanced_mixed",
        "continuous_dominant_mixed",
        "discrete_dominant_mixed",
    }
    assert {row.continuous_noise for row in pilot.datasets} == {
        "gaussian",
        "student_t_df3",
        "student_t_df5",
        "centered_gamma_shape1_scale2",
    }
    assert all("mixed" in row.variable_type_regime for row in pilot.datasets)
    with pytest.raises(ValueError, match="brain_data_profile"):
        build_experiment_plan("v2", master_seed=20260904, profile="full")


def test_e3_references_are_target_specific_and_seed_disjoint() -> None:
    plan = build_experiment_plan("e3", master_seed=20260814)
    assert len(plan.runs) == 1840
    references = [row for row in plan.runs if row.role == "reference"]
    assert {row.target for row in references} == {"order-modular", "structure-modular"}
    assert all(row.method.endswith("-reference") for row in references)
    evaluation_seeds = {
        row.method_seed["sha256"]
        for row in plan.runs
        if row.role == "evaluation" and row.method_seed is not None
    }
    reference_seeds = {row.method_seed["sha256"] for row in references if row.method_seed is not None}
    assert evaluation_seeds.isdisjoint(reference_seeds)
    evaluation = [row for row in plan.runs if row.role == "evaluation"]
    assert {row.wall_seconds for row in evaluation} == {30, 60, 180, 300}
    by_prefix = {}
    for row in evaluation:
        key = (row.dataset_id, row.method, row.seed_index)
        by_prefix.setdefault(key, set()).add(row.method_seed["sha256"])
    assert all(len(seeds) == 1 for seeds in by_prefix.values())


def test_e4_is_fixed_work_strong_scaling_and_includes_rank_eight() -> None:
    plan = build_experiment_plan("e4", master_seed=20260814)
    assert len(plan.datasets) == 10
    assert len(plan.runs) == 270
    assert plan.settings["particles"] == 512
    assert plan.settings["mpi_ranks"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    grouped = {}
    for row in plan.runs:
        grouped.setdefault((row.dataset_id, row.timing_repeat), set()).add(
            row.method_seed["sha256"]
        )
    assert all(len(seeds) == 1 for seeds in grouped.values())


def test_plans_are_replay_deterministic() -> None:
    first = build_experiment_plan("e3", master_seed=123).to_json()
    second = build_experiment_plan("e3", master_seed=123).to_json()
    third = build_experiment_plan("e3", master_seed=124).to_json()
    assert first == second
    assert first != third
