import copy

import pandas as pd
import pytest

from ordersmc_experiments import catchup, effort_match, effort_submit, gate_repair
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.readiness import canonical_sha
from test_catchup import tiny_plan


def save_plan(root, plan):
    plan = {k: v for k, v in plan.items() if k != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha(plan)
    _atomic_json(root / "plan.json", plan)
    return catchup.read(root / "plan.json")


def paired_plan(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    plan = tiny_plan(source)
    for fit in plan["fits"]:
        fit["config"]["final_target_snapshot_steps"] = [0]
        fit["role"] = "evaluation"
    plan = save_plan(source, plan)
    catchup.preflight(source)
    catchup.run(source, 0)
    target = tmp_path / "effort"
    paired = copy.deepcopy(plan)
    paired["source_root"] = str(source)
    paired["config"].update(primary_arm="adaptive_matched_effort", concurrency=2,
                            source_plan_sha256=plan["plan_sha256"])
    paired["comparisons"] = []
    for fit in paired["fits"]:
        base_fit = next(f for f in plan["fits"] if f["fit_id"] == fit["fit_id"])
        adaptive, path = catchup.fit_record(source, plan, base_fit)
        particles = fit["config"]["particles"]
        before = effort_match.proposal_count(adaptive["history"])
        legacy = catchup.located(fit["original"])
        legacy["diagnostics"] = dict(history=[dict(proposed_moves=before + 3 * particles)])
        legacy_path = source / (fit["fit_id"] + "-legacy.json")
        _atomic_json(legacy_path, legacy)
        fit.update(original=catchup.locator(legacy_path),
                   adaptive_source=catchup.artifact(path),
                   adaptive_weights=catchup.artifact(path.parent / "weights.npy"),
                   effort=effort_match.effort_steps(legacy["diagnostics"]["history"], adaptive["history"], particles))
        fit["config"]["final_target_snapshot_steps"] = [0, 3]
        paired["comparisons"].append(dict(dataset_key="tiny", method="order-mcmc",
            seed_index=fit["seed_index"], original=catchup.locator(legacy_path)))
    return target, save_plan(target, paired)


def test_effort_rule_uses_proposals_only_and_rejects_unmatchable_counts():
    legacy = [dict(proposed_moves=48, directed_f1=.1), dict(proposed_moves=24)]
    adaptive = [dict(proposed_moves=24, directed_f1=.99)]
    assert effort_match.effort_steps(legacy, adaptive, 24) == dict(
        legacy_proposals=72, adaptive_proposals=24, extra_steps=2)
    with pytest.raises(ValueError, match="exceeds"):
        effort_match.effort_steps(adaptive, legacy, 24)
    with pytest.raises(ValueError, match="whole proposals"):
        effort_match.effort_steps([dict(proposed_moves=73)], adaptive, 24)
    for invalid in (float("inf"), -1, .5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            effort_match.proposal_count([dict(proposed_moves=invalid)])


def test_actual_paired_run_exactly_replays_and_reports_all_arms(tmp_path):
    root, plan = paired_plan(tmp_path)
    effort_match.preflight(root)
    with pytest.raises(RuntimeError, match="unverified paired"):
        effort_match.aggregate(root)
    effort_match.run(root, 0)
    audit = catchup.read(root / "validation/replay/fit0.json")
    assert audit["snapshot_zero_exact_replay"] and audit["weights_unchanged"]
    assert audit["extra_steps"] == 3
    saved = (root / "runs/fit0/result.json").read_bytes()
    effort_match.run(root, 0)
    assert (root / "runs/fit0/result.json").read_bytes() == saved
    effort_match.aggregate(root)
    frame = pd.read_csv(root / "analysis/per_run_metrics.csv")
    assert set(frame.method) == {"adaptive_matched_effort", "adaptive_original_budget", "legacy", "order-mcmc"}
    assert len(frame) == 8
    completion = catchup.read(root / "analysis/completion.json")
    assert completion["all_replays_verified"]
    assert completion["primary_accuracy_passed"] is None  # Only one synthetic dataset.
    assert not completion["automatic_default_promotion"]
    record_path = root / "runs/fit0/result.json"
    record = catchup.read(record_path)
    record["snapshots"][0]["edge_probabilities"][0]["probability"] += .001
    _atomic_json(record_path, record)
    with pytest.raises(RuntimeError, match="exactly replay"):
        effort_match.validate_replay(root, plan, plan["fits"][0])
    with pytest.raises(RuntimeError, match="unverified paired"):
        effort_match.aggregate(root)


def test_effort_submission_resumes_and_waits_for_original_report(tmp_path):
    root, plan = paired_plan(tmp_path)
    source = tmp_path / "source"
    _atomic_json(source / "submission.json", dict(jobs={"metadata-report": "999"}))
    first = []
    def fails_second(args, **kwargs):
        first.append(args)
        if len(first) == 2:
            raise RuntimeError("submission limit")
        return "1234\n"
    with pytest.raises(RuntimeError, match="submission limit"):
        effort_submit.submit(root, "start", sbatch=fails_second)
    assert catchup.read(root / "submission.json")["jobs"] == {"preflight": "1234"}
    calls = []
    def accepts(args, **kwargs):
        calls.append(args)
        return str(1234 + len(calls)) + "\n"
    effort_submit.submit(root, "start", sbatch=accepts)
    assert len(calls) == 2
    assert "--dependency=afterany:1234" in calls[0]
    assert "--dependency=afterany:999" in calls[1]
    with pytest.raises(FileNotFoundError):
        effort_submit.submit(root, "launch", sbatch=accepts)
    assert len(calls) == 2
    effort_match.preflight(root)
    effort_submit.submit(root, "launch", sbatch=accepts)
    assert "--array=0-0%2" in calls[2]
    assert "--dependency=afterany:1237" in calls[3]
    effort_submit.submit(root, "start", sbatch=accepts)
    effort_submit.submit(root, "launch", sbatch=accepts)
    assert len(calls) == 4


def test_report_repair_archives_original_and_fails_closed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    plan = tiny_plan(source)
    catchup.preflight(source)
    catchup.run(source, 0)
    catchup.aggregate(source, "metadata")
    root = tmp_path / "reporting"
    reporting = dict(source_files={}, config=dict(source_plan_sha256=plan["plan_sha256"]))
    save_plan(root, reporting)
    gate = source / "analysis/metadata/exact_rmse_flags.json"
    _atomic_json(gate, dict(old_per_seed_report=True))
    previous = gate.read_bytes()
    corrected = gate_repair.repair(source, "metadata", root)
    archive = source / "reporting_corrections/dataset-gate-v1/metadata/original_per_seed_gate.json"
    assert archive.read_bytes() == previous
    assert catchup.read(gate) == corrected
    assert corrected["all_passed"] is None
    assert gate_repair.repair(source, "metadata", root) == corrected
    assert archive.read_bytes() == previous
    manifest = catchup.read(archive.with_name("manifest.json"))
    assert not manifest["fit_files_changed"] and not manifest["frozen_source_changed"]
    metrics = source / "analysis/metadata/per_run_metrics.csv"
    frame = pd.read_csv(metrics)
    frame.iloc[:-1].to_csv(metrics, index=False)
    with pytest.raises(RuntimeError, match="inventory|input or correction"):
        gate_repair.repair(source, "metadata", root)
