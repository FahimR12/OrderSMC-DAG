from __future__ import annotations

import hashlib
import numpy as np
import pytest

from ordersmc_experiments.reproducibility import (
    assess_internal_mcse_calibration,
    assess_rank_layout_consistency,
)


def _fixture(shift: float = 0.0):
    nodes = {f"d{index}": ("a", "b", "c") for index in range(4)}
    truths = {dataset: [("a", "b")] for dataset in nodes}
    reference = {}
    comparison = {}
    seed_ids = [hashlib.sha256(f"seed-{index}".encode()).hexdigest() for index in range(20)]
    for dataset_index, dataset in enumerate(nodes):
        base = 0.72 + 0.01 * dataset_index
        complete_reference = {
            ("a", "b"): base,
            ("a", "c"): 0.0,
            ("b", "a"): 0.03,
            ("b", "c"): 0.20,
            ("c", "a"): 0.0,
            ("c", "b"): 0.0,
        }
        complete_comparison = {
                ("a", "b"): base + shift,
                ("a", "c"): 0.0,
                ("b", "a"): 0.03,
                ("b", "c"): 0.20,
                ("c", "a"): 0.0,
                ("c", "b"): 0.0,
        }
        reference[dataset] = {
            seed_id: dict(complete_reference) for seed_id in seed_ids
        }
        comparison[dataset] = {
            seed_id: dict(complete_comparison) for seed_id in seed_ids
        }
    return {"rank1": reference, "rank4": comparison}, nodes, truths


def test_rank_layout_equivalence_uses_whole_paired_datasets_and_is_deterministic() -> None:
    runs, nodes, truths = _fixture()
    first = assess_rank_layout_consistency(
        runs,
        nodes,
        truths,
        reference_layout="rank1",
        comparison_layout="rank4",
        resamples=2_000,
        seed=91,
    )
    second = assess_rank_layout_consistency(
        runs,
        nodes,
        truths,
        reference_layout="rank1",
        comparison_layout="rank4",
        resamples=2_000,
        seed=91,
    )
    assert first == second
    assert first.equivalent
    assert first.global_pip_rms_ci_upper == 0.0
    assert first.skeleton_brier_ci_lower == first.skeleton_brier_ci_upper == 0.0
    assert first.minimum_seed_count_per_layout == 20


def test_rank_layout_equivalence_fails_when_pip_margin_is_exceeded() -> None:
    runs, nodes, truths = _fixture(shift=0.05)
    result = assess_rank_layout_consistency(
        runs,
        nodes,
        truths,
        reference_layout="rank1",
        comparison_layout="rank4",
        resamples=1_000,
        seed=7,
    )
    assert result.global_pip_rms_ci_lower > 0.01
    assert not result.equivalent


def test_mcse_calibration_reduces_edges_within_dataset_strata_first() -> None:
    nodes = {"d0": ("a", "b", "c"), "d1": ("a", "b", "c")}
    truths = {dataset: [("a", "b")] for dataset in nodes}
    reference_row = {
        ("a", "b"): 0.75,
        ("a", "c"): 0.0,
        ("b", "a"): 0.02,
        ("b", "c"): 0.25,
        ("c", "a"): 0.0,
        ("c", "b"): 0.0,
    }
    reference = {dataset: dict(reference_row) for dataset in nodes}
    probability_runs = {}
    mcse_runs = {}
    offsets = np.asarray([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
    empirical = float(np.std(offsets, ddof=1))
    seed_ids = [
        hashlib.sha256(f"mcse-seed-{index}".encode()).hexdigest()
        for index in range(len(offsets))
    ]
    for dataset in nodes:
        probability_runs[dataset] = {
            seed_id: {
                ("a", "b"): 0.75 + float(offset),
                ("a", "c"): 0.0,
                ("b", "a"): 0.02,
                ("b", "c"): 0.25 - float(offset),
                ("c", "a"): 0.0,
                ("c", "b"): 0.0,
            }
            for seed_id, offset in zip(seed_ids, offsets)
        }
        mcse_runs[dataset] = {
            seed_id: {
                ("a", "b"): empirical,
                ("b", "a"): 0.0,
                ("b", "c"): empirical,
                ("a", "c"): 0.0,
                ("c", "a"): 0.0,
                ("c", "b"): 0.0,
            }
            for seed_id in seed_ids
        }
    result = assess_internal_mcse_calibration(
        probability_runs,
        mcse_runs,
        reference,
        truths,
        nodes,
        target_label="order",
        support_ids_by_dataset={dataset: "candidate-table-v1" for dataset in nodes},
    )
    assert result.dataset_count == 2
    assert result.minimum_seed_count == 7
    assert result.median_cell_ratio == pytest.approx(1.0)
    assert result.p90_cell_ratio == pytest.approx(1.0)
    assert {cell.dataset_id for cell in result.cells} == {"d0", "d1"}
    assert all(cell.truth_status in {"true_edge", "nonedge"} for cell in result.cells)


def test_mcse_audit_flags_positive_mcse_when_empirical_seed_sd_is_zero() -> None:
    nodes = {"d0": ("a", "b")}
    edges = (("a", "b"), ("b", "a"))
    seed_ids = [hashlib.sha256(f"flat-{index}".encode()).hexdigest() for index in range(3)]
    probabilities = {
        "d0": {seed_id: {edges[0]: 0.5, edges[1]: 0.1} for seed_id in seed_ids}
    }
    mcse = {
        "d0": {seed_id: {edges[0]: 0.02, edges[1]: 0.0} for seed_id in seed_ids}
    }
    result = assess_internal_mcse_calibration(
        probabilities,
        mcse,
        {"d0": {edges[0]: 0.5, edges[1]: 0.1}},
        {"d0": [edges[0]]},
        nodes,
        target_label="order",
        support_ids_by_dataset={"d0": "candidate-table-v1"},
    )
    assert result.calibration_status == "infinite_ratio_positive_mcse_zero_empirical_sd"
    assert result.median_cell_ratio is None
    assert any(cell.positive_mcse_zero_sd_count for cell in result.cells)
