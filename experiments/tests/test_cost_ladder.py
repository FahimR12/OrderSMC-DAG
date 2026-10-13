from __future__ import annotations

from copy import deepcopy
import time

import pytest
import numpy as np

from ordersmc.core import OrderCandidateTable, OrderSMCResult, order_smc

from ordersmc_experiments.cost_ladder import (
    CompletedRunRecord,
    CostLadderRung,
    PilotCostObservation,
    accuracy_at_time,
    audit_confirmatory_resource_order,
    completed_run_from_ordersmc_result,
    freeze_cost_ladder,
    validate_completed_run,
    validate_frozen_cost_ladder,
)


def _ladder():
    rungs = [
        CostLadderRung("small", 100, 8, 2),
        CostLadderRung("medium", 200, 12, 4),
        CostLadderRung("large", 400, 16, 8),
    ]
    observations = [
        PilotCostObservation("small", 100, 80),
        PilotCostObservation("small", 110, 90),
        PilotCostObservation("medium", 220, 200),
        PilotCostObservation("medium", 240, 210),
        PilotCostObservation("large", 500, 450),
        PilotCostObservation("large", 520, 470),
    ]
    return freeze_cost_ladder(rungs, observations)


def _record(
    rung_id: str,
    *,
    endpoint: float | None,
    target: int,
    proposals: int,
    completion: float | None,
    status: str = "success",
    beta: float | None = 1.0,
) -> CompletedRunRecord:
    ladder = _ladder()
    rung = next(rung for rung in ladder.rungs if rung.rung_id == rung_id)
    return CompletedRunRecord(
        rung_id=rung_id,
        ladder_content_sha256=ladder.content_sha256,
        particles=rung.particles,
        tempering_steps=rung.tempering_steps,
        mh_steps=rung.mh_steps,
        status=status,
        beta=beta,
        final_target_value=endpoint,
        target_evaluations=target,
        proposal_evaluations=proposals,
        score_lookups=2 * target,
        inference_completion_seconds=completion,
        inference_attempt_wall_seconds=() if completion is None else (completion,),
        resumed_from_checkpoint=False,
        candidate_build_seconds=5.0,
        wall_clock_cap_seconds=60.0,
    )


def test_pilot_freeze_is_resource_only_deterministic_and_strict() -> None:
    first = _ladder()
    second = _ladder()
    assert first == second
    assert len(first.content_sha256) == 64
    assert first.to_json() == second.to_json()
    validate_frozen_cost_ladder(first)

    with pytest.raises(ValueError, match="proposal counts must increase strictly"):
        freeze_cost_ladder(
            [CostLadderRung("a", 10, 2, 1), CostLadderRung("b", 20, 3, 1)],
            [
                PilotCostObservation("a", 10, 10),
                PilotCostObservation("b", 20, 10),
            ],
        )


def test_only_beta_one_success_is_a_scientific_output_and_failures_are_censored() -> None:
    ladder = _ladder()
    valid = _record("small", endpoint=0.04, target=100, proposals=80, completion=10.0)
    validate_completed_run(ladder, valid)
    assert valid.end_to_end_completion_seconds == pytest.approx(15.0)

    tempered = _record(
        "small", endpoint=0.04, target=100, proposals=80, completion=10.0, beta=0.9
    )
    with pytest.raises(ValueError, match="finish at beta=1"):
        validate_completed_run(ladder, tempered)

    timeout = _record(
        "medium",
        endpoint=None,
        target=150,
        proposals=140,
        completion=None,
        status="timeout",
        beta=0.8,
    )
    validate_completed_run(ladder, timeout)
    assert timeout.right_censored_seconds == pytest.approx(65.0)

    early_oom = _record(
        "medium",
        endpoint=None,
        target=80,
        proposals=70,
        completion=12.0,
        status="oom",
        beta=0.4,
    )
    validate_completed_run(ladder, early_oom)
    assert early_oom.right_censored_seconds == pytest.approx(17.0)

    imputed = _record(
        "medium",
        endpoint=99.0,
        target=150,
        proposals=140,
        completion=None,
        status="timeout",
        beta=0.8,
    )
    with pytest.raises(ValueError, match="must not contain an imputed endpoint"):
        validate_completed_run(ladder, imputed)


def test_accuracy_carries_forward_only_after_actual_final_target_completion() -> None:
    ladder = _ladder()
    records = [
        _record("small", endpoint=0.08, target=100, proposals=80, completion=10.0),
        _record(
            "medium",
            endpoint=None,
            target=210,
            proposals=190,
            completion=None,
            status="timeout",
            beta=0.7,
        ),
        _record("large", endpoint=0.03, target=500, proposals=450, completion=40.0),
    ]
    assert accuracy_at_time(ladder, records, [10.0, 15.0, 30.0, 45.0]) == (
        None,
        0.08,
        0.08,
        0.03,
    )


def test_confirmatory_resource_inversion_is_reported_without_reordering() -> None:
    ladder = _ladder()
    records = [
        _record("small", endpoint=0.08, target=100, proposals=80, completion=10.0),
        _record("medium", endpoint=0.05, target=250, proposals=220, completion=20.0),
        _record("large", endpoint=0.03, target=230, proposals=400, completion=40.0),
    ]
    audit = audit_confirmatory_resource_order(ladder, records)
    assert audit.frozen_rung_ids == ("small", "medium", "large")
    assert not audit.cost_monotone
    assert audit.description_label == "frozen_rung_sequence_with_resource_inversion"


def test_production_result_binding_uses_measured_final_target_resources() -> None:
    ladder = _ladder()
    table = OrderCandidateTable(
        nodes=["a", "b"],
        parent_sets=[
            np.asarray([[-1], [1]], dtype=np.int32),
            np.asarray([[-1], [0]], dtype=np.int32),
        ],
        loglik=[np.asarray([0.0, 0.2]), np.asarray([0.0, 0.3])],
        logprior=[np.zeros(2), np.zeros(2)],
    )
    timing_started = time.perf_counter()
    result = order_smc(
        table,
        particles=100,
        betas=np.linspace(0.0, 1.0, 9),
        mh_steps=2,
        seed=5,
        verbose=False,
    )
    attempt_wall_seconds = time.perf_counter() - timing_started
    record = completed_run_from_ordersmc_result(
        ladder,
        rung_id="small",
        result=result,
        final_target_value=0.02,
        candidate_build_seconds=2.0,
        wall_clock_cap_seconds=60.0,
        inference_attempt_wall_seconds=[attempt_wall_seconds],
    )
    assert record.beta == 1.0
    assert record.proposal_evaluations == 100 * 8 * 2
    assert record.target_evaluations == result.run_info["physical_target_evaluations"]
    assert record.score_lookups == result.run_info["total_score_lookups"]
    assert record.inference_completion_seconds == pytest.approx(attempt_wall_seconds)
    assert record.end_to_end_completion_seconds == pytest.approx(
        2.0 + attempt_wall_seconds
    )

    with pytest.raises(ValueError, match="external wall timing"):
        completed_run_from_ordersmc_result(
            ladder,
            rung_id="small",
            result=result,
            final_target_value=0.02,
            candidate_build_seconds=2.0,
            wall_clock_cap_seconds=60.0,
        )

    with pytest.raises(ValueError, match=r"frozen \(N,T,M\) rung"):
        completed_run_from_ordersmc_result(
            ladder,
            rung_id="medium",
            result=result,
            final_target_value=0.02,
            candidate_build_seconds=2.0,
            wall_clock_cap_seconds=60.0,
            inference_attempt_wall_seconds=[attempt_wall_seconds],
        )


def test_resumed_result_requires_complete_attempt_timing_and_counts_physical_work() -> None:
    ladder = _ladder()
    table = OrderCandidateTable(
        nodes=["a", "b"],
        parent_sets=[
            np.asarray([[-1], [1]], dtype=np.int32),
            np.asarray([[-1], [0]], dtype=np.int32),
        ],
        loglik=[np.asarray([0.0, 0.2]), np.asarray([0.0, 0.3])],
        logprior=[np.zeros(2), np.zeros(2)],
    )
    result = order_smc(
        table,
        particles=100,
        betas=np.linspace(0.0, 1.0, 9),
        mh_steps=2,
        seed=8,
        verbose=False,
    )
    resumed = deepcopy(result)
    logical = int(resumed.run_info["target_evaluations"])
    resumed.run_info["resume_validation_target_evaluations"] = 100
    resumed.run_info["physical_target_evaluations"] = logical + 100
    internal = float(resumed.run_info["elapsed_seconds"])

    with pytest.raises(ValueError, match="original and every resume"):
        completed_run_from_ordersmc_result(
            ladder,
            rung_id="small",
            result=resumed,
            final_target_value=0.02,
            candidate_build_seconds=2.0,
            wall_clock_cap_seconds=60.0,
            inference_attempt_wall_seconds=[internal + 0.1],
        )

    record = completed_run_from_ordersmc_result(
        ladder,
        rung_id="small",
        result=resumed,
        final_target_value=0.02,
        candidate_build_seconds=2.0,
        wall_clock_cap_seconds=60.0,
        inference_attempt_wall_seconds=[0.25, internal + 0.1],
    )
    assert record.resumed_from_checkpoint
    assert record.target_evaluations == logical + 100
    assert record.inference_completion_seconds == pytest.approx(internal + 0.35)

    inconsistent = deepcopy(resumed)
    inconsistent.run_info["physical_target_evaluations"] += 1
    with pytest.raises(ValueError, match="physical target-evaluation counter"):
        completed_run_from_ordersmc_result(
            ladder,
            rung_id="small",
            result=inconsistent,
            final_target_value=0.02,
            candidate_build_seconds=2.0,
            wall_clock_cap_seconds=60.0,
            inference_attempt_wall_seconds=[0.25, internal + 0.1],
        )
