"""Frozen, final-target cost ladders for protocol time comparisons.

The helpers in this module deliberately separate pilot ordering from confirmatory
accuracy.  A ladder is ordered once using resource counts only; confirmatory
resource inversions are reported without reordering or deleting a rung.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from statistics import median
from typing import Iterable, Mapping, Sequence

from ordersmc.core import OrderSMCResult


@dataclass(frozen=True)
class CostLadderRung:
    """One independent completed-run configuration ``(N, T, M)``."""

    rung_id: str
    particles: int
    tempering_steps: int
    mh_steps: int

    def __post_init__(self) -> None:
        if not self.rung_id or not self.rung_id.strip():
            raise ValueError("rung_id must be non-empty")
        for name in ("particles", "tempering_steps", "mh_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.particles == 0 or self.tempering_steps == 0:
            raise ValueError("particles and tempering_steps must be positive")


@dataclass(frozen=True)
class PilotCostObservation:
    rung_id: str
    target_evaluations: int
    proposal_evaluations: int

    def __post_init__(self) -> None:
        if not self.rung_id:
            raise ValueError("rung_id must be non-empty")
        for name in ("target_evaluations", "proposal_evaluations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class FrozenCostLadder:
    rungs: tuple[CostLadderRung, ...]
    pilot_median_target_evaluations: tuple[float, ...]
    pilot_median_proposal_evaluations: tuple[float, ...]
    content_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema": "ordersmc-protocol-v2-final-target-cost-ladder-v1",
            "rungs": [asdict(rung) for rung in self.rungs],
            "pilot_median_target_evaluations": list(
                self.pilot_median_target_evaluations
            ),
            "pilot_median_proposal_evaluations": list(
                self.pilot_median_proposal_evaluations
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "content_sha256": self.content_sha256}

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"


@dataclass(frozen=True)
class CompletedRunRecord:
    """Observed outcome for one frozen rung on one dataset/method seed.

    ``final_target_value`` is an arbitrary scalar endpoint, such as PIP RMSE.
    It is absent for timeout/OOM/error rows.  Such failures remain available as
    right-censored observations rather than being imputed.

    ``target_evaluations`` is the physical count, including checkpoint-resume
    validation.  ``inference_attempt_wall_seconds`` contains externally timed
    wall durations for every attempt, including interrupted attempts; its sum
    is ``inference_completion_seconds``.  This prevents the sampler's
    accumulated internal segment timer from being mislabelled end-to-end time.
    """

    rung_id: str
    ladder_content_sha256: str
    particles: int
    tempering_steps: int
    mh_steps: int
    status: str
    beta: float | None
    final_target_value: float | None
    target_evaluations: int
    proposal_evaluations: int
    score_lookups: int
    inference_completion_seconds: float | None
    inference_attempt_wall_seconds: tuple[float, ...]
    resumed_from_checkpoint: bool
    candidate_build_seconds: float
    wall_clock_cap_seconds: float

    @property
    def completed_final_target(self) -> bool:
        return (
            self.status == "success"
            and self.beta is not None
            and math.isclose(self.beta, 1.0, rel_tol=0.0, abs_tol=1e-12)
        )

    @property
    def end_to_end_completion_seconds(self) -> float | None:
        if self.inference_completion_seconds is None:
            return None
        return self.candidate_build_seconds + self.inference_completion_seconds

    @property
    def right_censored_seconds(self) -> float:
        completion = self.end_to_end_completion_seconds
        if self.completed_final_target and completion is not None:
            return completion
        if completion is not None:
            return completion
        return self.candidate_build_seconds + self.wall_clock_cap_seconds


@dataclass(frozen=True)
class ConfirmatoryLadderAudit:
    frozen_rung_ids: tuple[str, ...]
    observed_median_target_evaluations: tuple[float, ...]
    observed_median_proposal_evaluations: tuple[float, ...]
    cost_monotone: bool
    description_label: str


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_cost_ladder(
    rungs: Sequence[CostLadderRung],
    pilot_observations: Iterable[PilotCostObservation],
) -> FrozenCostLadder:
    """Freeze a pilot-ordered ladder without consulting accuracy outcomes.

    Both pilot-median target-evaluation and proposal counts must increase
    strictly at every adjacent rung.  Requiring both is the conservative,
    auditable interpretation of the protocol's resource-count ordering rule.
    """

    frozen_rungs = tuple(rungs)
    if len(frozen_rungs) < 2:
        raise ValueError("a cost ladder requires at least two rungs")
    rung_ids = tuple(rung.rung_id for rung in frozen_rungs)
    if len(set(rung_ids)) != len(rung_ids):
        raise ValueError("cost-ladder rung IDs must be unique")
    for left, right in zip(frozen_rungs, frozen_rungs[1:]):
        left_config = (left.particles, left.tempering_steps, left.mh_steps)
        right_config = (right.particles, right.tempering_steps, right.mh_steps)
        if not all(new >= old for old, new in zip(left_config, right_config)) or (
            right_config == left_config
        ):
            raise ValueError(
                "cost-ladder (particles, tempering_steps, mh_steps) must increase "
                "componentwise with at least one strict increase"
            )

    grouped: dict[str, list[PilotCostObservation]] = {rung_id: [] for rung_id in rung_ids}
    for observation in pilot_observations:
        if observation.rung_id not in grouped:
            raise ValueError(f"pilot observation uses unknown rung {observation.rung_id!r}")
        grouped[observation.rung_id].append(observation)
    missing = [rung_id for rung_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"missing pilot cost observations for rungs: {missing}")

    target_medians = tuple(
        float(median(row.target_evaluations for row in grouped[rung_id]))
        for rung_id in rung_ids
    )
    proposal_medians = tuple(
        float(median(row.proposal_evaluations for row in grouped[rung_id]))
        for rung_id in rung_ids
    )
    if not all(right > left for left, right in zip(target_medians, target_medians[1:])):
        raise ValueError("pilot-median target-evaluation counts must increase strictly")
    if not all(right > left for left, right in zip(proposal_medians, proposal_medians[1:])):
        raise ValueError("pilot-median proposal counts must increase strictly")

    payload = {
        "schema": "ordersmc-protocol-v2-final-target-cost-ladder-v1",
        "rungs": [asdict(rung) for rung in frozen_rungs],
        "pilot_median_target_evaluations": list(target_medians),
        "pilot_median_proposal_evaluations": list(proposal_medians),
    }
    return FrozenCostLadder(
        rungs=frozen_rungs,
        pilot_median_target_evaluations=target_medians,
        pilot_median_proposal_evaluations=proposal_medians,
        content_sha256=_canonical_hash(payload),
    )


def validate_frozen_cost_ladder(ladder: FrozenCostLadder) -> None:
    """Fail closed if a frozen ladder or its resource ordering was altered."""

    if not isinstance(ladder, FrozenCostLadder):
        raise TypeError("ladder must be a FrozenCostLadder")
    if _canonical_hash(ladder.content_payload()) != ladder.content_sha256:
        raise ValueError("frozen cost-ladder content hash mismatch")
    if len(ladder.rungs) < 2 or len({rung.rung_id for rung in ladder.rungs}) != len(
        ladder.rungs
    ):
        raise ValueError("frozen cost ladder has invalid or duplicate rungs")
    for left, right in zip(ladder.rungs, ladder.rungs[1:]):
        left_config = (left.particles, left.tempering_steps, left.mh_steps)
        right_config = (right.particles, right.tempering_steps, right.mh_steps)
        if not all(new >= old for old, new in zip(left_config, right_config)) or (
            right_config == left_config
        ):
            raise ValueError("frozen cost-ladder configurations are not increasing")
    if not (
        len(ladder.rungs)
        == len(ladder.pilot_median_target_evaluations)
        == len(ladder.pilot_median_proposal_evaluations)
    ):
        raise ValueError("frozen cost-ladder medians are not aligned with its rungs")
    for values, name in (
        (ladder.pilot_median_target_evaluations, "target-evaluation"),
        (ladder.pilot_median_proposal_evaluations, "proposal"),
    ):
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"frozen pilot-median {name} counts are invalid")
        if not all(right > left for left, right in zip(values, values[1:])):
            raise ValueError(f"frozen pilot-median {name} counts are not strictly increasing")


def validate_completed_run(
    ladder: FrozenCostLadder,
    record: CompletedRunRecord,
) -> None:
    """Reject scientific outputs that are not completed final-target runs."""

    validate_frozen_cost_ladder(ladder)
    rungs_by_id = {rung.rung_id: rung for rung in ladder.rungs}
    if record.rung_id not in rungs_by_id:
        raise ValueError(f"run uses unknown frozen rung {record.rung_id!r}")
    if record.ladder_content_sha256 != ladder.content_sha256:
        raise ValueError("run is not bound to this frozen cost ladder")
    frozen_rung = rungs_by_id[record.rung_id]
    observed_configuration = (
        record.particles,
        record.tempering_steps,
        record.mh_steps,
    )
    expected_configuration = (
        frozen_rung.particles,
        frozen_rung.tempering_steps,
        frozen_rung.mh_steps,
    )
    if observed_configuration != expected_configuration:
        raise ValueError("run resources do not match its frozen (N,T,M) rung")
    if record.status not in {"success", "timeout", "oom", "error"}:
        raise ValueError("status must be success, timeout, oom, or error")
    for name in (
        "particles",
        "tempering_steps",
        "mh_steps",
        "target_evaluations",
        "proposal_evaluations",
        "score_lookups",
    ):
        value = getattr(record, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("candidate_build_seconds", "wall_clock_cap_seconds"):
        value = getattr(record, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if record.wall_clock_cap_seconds <= 0.0:
        raise ValueError("wall_clock_cap_seconds must be positive")
    if not isinstance(record.resumed_from_checkpoint, bool):
        raise ValueError("resumed_from_checkpoint must be boolean")
    attempts = record.inference_attempt_wall_seconds
    if not isinstance(attempts, tuple) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in attempts
    ):
        raise ValueError(
            "inference_attempt_wall_seconds must be a tuple of finite "
            "non-negative measurements"
        )
    if record.inference_completion_seconds is None:
        if attempts:
            raise ValueError(
                "attempt wall times require a recorded inference completion/failure time"
            )
        if record.resumed_from_checkpoint:
            raise ValueError(
                "a resumed row requires complete timing for every inference attempt"
            )
    else:
        if not attempts:
            raise ValueError(
                "a recorded inference time requires externally measured attempt walls"
            )
        if record.resumed_from_checkpoint and len(attempts) < 2:
            raise ValueError(
                "a resumed row requires timing for the original and resumed attempts"
            )
        if not record.resumed_from_checkpoint and len(attempts) != 1:
            raise ValueError("a non-resumed row must contain exactly one attempt wall time")
        if not math.isclose(
            float(sum(attempts)),
            record.inference_completion_seconds,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "inference completion time must equal all measured attempt wall times"
            )

    if record.status == "success":
        if record.beta is None or not math.isclose(record.beta, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("a scientific cost-ladder output must finish at beta=1")
        if record.final_target_value is None or not math.isfinite(record.final_target_value):
            raise ValueError("a successful final-target run requires a finite endpoint")
        if (
            record.inference_completion_seconds is None
            or not math.isfinite(record.inference_completion_seconds)
            or record.inference_completion_seconds < 0.0
        ):
            raise ValueError("a successful run requires a finite completion time")
        if record.inference_completion_seconds > record.wall_clock_cap_seconds + 1e-9:
            raise ValueError("successful inference completion exceeds its wall-clock cap")
    else:
        if record.final_target_value is not None:
            raise ValueError("failed/censored runs must not contain an imputed endpoint")
        if record.beta is not None:
            if not math.isfinite(record.beta) or not 0.0 <= record.beta < 1.0:
                raise ValueError("a failed row beta must be finite in [0, 1)")
        if record.inference_completion_seconds is not None:
            if (
                not math.isfinite(record.inference_completion_seconds)
                or record.inference_completion_seconds < 0.0
            ):
                raise ValueError("failure time must be finite and non-negative when recorded")
            if record.inference_completion_seconds > record.wall_clock_cap_seconds + 1e-9:
                raise ValueError("recorded failure time exceeds its wall-clock cap")


def completed_run_from_ordersmc_result(
    ladder: FrozenCostLadder,
    *,
    rung_id: str,
    result: OrderSMCResult,
    final_target_value: float,
    candidate_build_seconds: float,
    wall_clock_cap_seconds: float,
    inference_attempt_wall_seconds: Sequence[float] | None = None,
) -> CompletedRunRecord:
    """Bind a successful production result to one final-target ladder row.

    Attempt-wall measurements must be taken outside ``order_smc`` and wrap
    every original/resumed invocation.  The internal ``elapsed_seconds`` field
    is only a sum of persisted sampler segments and therefore is not accepted
    as end-to-end timing evidence.
    """

    if not isinstance(result, OrderSMCResult):
        raise TypeError("result must be an OrderSMCResult")
    schedule = result.run_info.get("tempering_schedule")
    if (
        not isinstance(schedule, Sequence)
        or not schedule
        or not math.isclose(float(schedule[-1]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("OrderSMC result did not complete the beta=1 target")
    proposal_diagnostics = result.run_info.get("proposal_diagnostics")
    if not isinstance(proposal_diagnostics, Mapping):
        raise ValueError("OrderSMC result is missing proposal diagnostics")
    proposal_evaluations = 0
    for value in proposal_diagnostics.values():
        if not isinstance(value, Mapping) or "proposed" not in value:
            raise ValueError("OrderSMC proposal diagnostics are malformed")
        proposal_evaluations += int(value["proposed"])
    rungs_by_id = {rung.rung_id: rung for rung in ladder.rungs}
    if rung_id not in rungs_by_id:
        raise ValueError(f"run uses unknown frozen rung {rung_id!r}")
    rung = rungs_by_id[rung_id]
    sampler_configuration = result.run_info.get("sampler_configuration")
    if not isinstance(sampler_configuration, Mapping):
        raise ValueError("OrderSMC result is missing its sampler configuration")
    fixed_betas = sampler_configuration.get("fixed_betas")
    if not isinstance(fixed_betas, Sequence) or isinstance(fixed_betas, (str, bytes)):
        raise ValueError("cost-ladder runs require a frozen fixed beta schedule")
    observed_tempering_steps = len(fixed_betas) - 1
    try:
        frozen_schedule = [float(value) for value in fixed_betas]
        reported_schedule = [float(value) for value in schedule]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OrderSMC beta schedule is malformed") from exc
    if frozen_schedule != reported_schedule:
        raise ValueError("reported tempering schedule differs from sampler configuration")
    try:
        observed_particles = int(sampler_configuration["particles"])
        observed_mh_steps = int(sampler_configuration["mh_steps"])
        observed_resample_mh_steps = int(
            sampler_configuration["mh_steps_resample"]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OrderSMC sampler configuration is incomplete") from exc
    if len(result.orders) != observed_particles:
        raise ValueError("OrderSMC particle state disagrees with its sampler configuration")
    observed_ntm = (
        observed_particles,
        observed_tempering_steps,
        observed_mh_steps,
    )
    if observed_ntm != (rung.particles, rung.tempering_steps, rung.mh_steps):
        raise ValueError("OrderSMC result does not match the frozen (N,T,M) rung")
    if observed_resample_mh_steps != rung.mh_steps:
        raise ValueError("resampling-specific MH work differs from the frozen M rung")
    continuation_steps = sampler_configuration.get("final_target_snapshot_steps", [])
    if not isinstance(continuation_steps, Sequence) or any(
        int(value) != 0 for value in continuation_steps
    ):
        raise ValueError(
            "post-target continuation must be analysed separately from a completed-run rung"
        )
    expected_proposals = (
        rung.particles * rung.tempering_steps * rung.mh_steps
    )
    if proposal_evaluations != expected_proposals:
        raise ValueError("proposal counters do not match the frozen (N,T,M) work")
    try:
        logical_target_evaluations = int(result.run_info["target_evaluations"])
        resume_target_evaluations = int(
            result.run_info["resume_validation_target_evaluations"]
        )
        target_evaluations = int(result.run_info["physical_target_evaluations"])
        score_lookups = int(result.run_info["total_score_lookups"])
        internal_elapsed_seconds = float(result.run_info["elapsed_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OrderSMC result is missing exact resource counters") from exc
    if any(
        value < 0
        for value in (
            logical_target_evaluations,
            resume_target_evaluations,
            target_evaluations,
        )
    ):
        raise ValueError("OrderSMC target-evaluation counters must be non-negative")
    if target_evaluations != logical_target_evaluations + resume_target_evaluations:
        raise ValueError("OrderSMC physical target-evaluation counter is inconsistent")
    if resume_target_evaluations % rung.particles != 0:
        raise ValueError("OrderSMC resume-validation counter is not particle aligned")
    resume_count = resume_target_evaluations // rung.particles
    if inference_attempt_wall_seconds is None:
        raise ValueError(
            "external wall timing for every inference attempt is required"
        )
    if isinstance(inference_attempt_wall_seconds, (str, bytes)):
        raise ValueError("inference attempt wall timings must be a numeric sequence")
    try:
        raw_attempt_wall_seconds = tuple(inference_attempt_wall_seconds)
    except TypeError as exc:
        raise ValueError(
            "inference attempt wall timings must be a numeric sequence"
        ) from exc
    if any(isinstance(value, bool) for value in raw_attempt_wall_seconds):
        raise ValueError("inference attempt wall timings cannot be boolean")
    try:
        attempt_wall_seconds = tuple(
            float(value) for value in raw_attempt_wall_seconds
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "inference attempt wall timings must be a numeric sequence"
        ) from exc
    if len(attempt_wall_seconds) != resume_count + 1:
        raise ValueError(
            "inference attempt wall timings do not cover the original and every resume"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in attempt_wall_seconds):
        raise ValueError(
            "inference attempt wall timings must be finite and non-negative"
        )
    completion_seconds = float(sum(attempt_wall_seconds))
    if not math.isfinite(internal_elapsed_seconds) or internal_elapsed_seconds < 0.0:
        raise ValueError("OrderSMC internal elapsed timing is invalid")
    if completion_seconds + 1e-9 < internal_elapsed_seconds:
        raise ValueError(
            "external attempt-wall timing cannot be shorter than sampler segment timing"
        )
    record = CompletedRunRecord(
        rung_id=rung_id,
        ladder_content_sha256=ladder.content_sha256,
        particles=rung.particles,
        tempering_steps=rung.tempering_steps,
        mh_steps=rung.mh_steps,
        status="success",
        beta=1.0,
        final_target_value=float(final_target_value),
        target_evaluations=target_evaluations,
        proposal_evaluations=proposal_evaluations,
        score_lookups=score_lookups,
        inference_completion_seconds=completion_seconds,
        inference_attempt_wall_seconds=attempt_wall_seconds,
        resumed_from_checkpoint=resume_count > 0,
        candidate_build_seconds=float(candidate_build_seconds),
        wall_clock_cap_seconds=float(wall_clock_cap_seconds),
    )
    validate_completed_run(ladder, record)
    return record


def audit_confirmatory_resource_order(
    ladder: FrozenCostLadder,
    records: Iterable[CompletedRunRecord],
) -> ConfirmatoryLadderAudit:
    """Report resource inversions while retaining the pilot-frozen rung order."""

    rung_ids = tuple(rung.rung_id for rung in ladder.rungs)
    grouped: dict[str, list[CompletedRunRecord]] = {rung_id: [] for rung_id in rung_ids}
    for record in records:
        validate_completed_run(ladder, record)
        grouped[record.rung_id].append(record)
    missing = [rung_id for rung_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"no confirmatory records for frozen rungs: {missing}")
    target_medians = tuple(
        float(median(row.target_evaluations for row in grouped[rung_id]))
        for rung_id in rung_ids
    )
    proposal_medians = tuple(
        float(median(row.proposal_evaluations for row in grouped[rung_id]))
        for rung_id in rung_ids
    )
    monotone = all(
        right > left for left, right in zip(target_medians, target_medians[1:])
    ) and all(right > left for left, right in zip(proposal_medians, proposal_medians[1:]))
    return ConfirmatoryLadderAudit(
        frozen_rung_ids=rung_ids,
        observed_median_target_evaluations=target_medians,
        observed_median_proposal_evaluations=proposal_medians,
        cost_monotone=monotone,
        description_label="cost_monotone_ladder" if monotone else "frozen_rung_sequence_with_resource_inversion",
    )


def accuracy_at_time(
    ladder: FrozenCostLadder,
    records: Iterable[CompletedRunRecord],
    checkpoints_seconds: Sequence[float],
) -> tuple[float | None, ...]:
    """Carry forward only the latest valid beta-one completed-run endpoint."""

    validated = []
    for record in records:
        validate_completed_run(ladder, record)
        validated.append(record)
    successful = sorted(
        (
            (record.end_to_end_completion_seconds, record.final_target_value)
            for record in validated
            if record.completed_final_target
        ),
        key=lambda item: float(item[0]),
    )
    result: list[float | None] = []
    for checkpoint in checkpoints_seconds:
        if not math.isfinite(checkpoint) or checkpoint < 0.0:
            raise ValueError("checkpoints must be finite and non-negative")
        available = [value for completed, value in successful if float(completed) <= checkpoint]
        result.append(None if not available else float(available[-1]))
    return tuple(result)
