from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ordersmc_experiments import metadata_calibration_v2 as audit
from ordersmc_experiments import metadata_profiles as generator
from ordersmc_experiments.design import build_experiment_plan
from ordersmc_experiments.execution import _execution_config, _profile_config, _sha


def config(**values):
    return generator.MetadataDAGConfig(
        profile_id="protein_pathway", dimension=8, n_train=20, n_test=20,
        pilot_size=200, calibration_scales=(.5, 1.), calibration_repeats=2,
        master_seed=91, pairing_id="finite-calibration-test", **values)


def test_finite_calibration_uses_metadata_n_and_independent_named_streams(monkeypatch):
    calls = []
    original = generator._sample

    def record(n, split, *args):
        calls.append((n, split))
        return original(n, split, *args)

    monkeypatch.setattr(generator, "_sample", record)
    dataset = generator.generate_metadata_dag_dataset(config(calibration_method="finite-sample-v2"))
    assert calls == [(222, "calibration-evaluate-000"), (222, "calibration-evaluate-001")] * 2 + [
        (20, "train"), (20, "test")]
    assert dataset.manifest["calibration"]["evaluation_sample_size"] == 222
    assert dataset.manifest["calibration"]["selection_uses_graph_recovery"] is False


def test_finite_calibration_preserves_paired_mechanisms_and_sample_prefixes():
    cfg = config(calibration_method="finite-sample-v2")
    first = generator.generate_metadata_dag_dataset(cfg)
    larger = generator.generate_metadata_dag_dataset(replace(cfg, n_train=40, n_test=40))
    gaussian = generator.generate_metadata_dag_dataset(replace(cfg, data_variant="gaussian"))
    assert first.manifest["mechanisms"] == larger.manifest["mechanisms"]
    assert first.manifest["calibration"]["scale_grid"] == larger.manifest["calibration"]["scale_grid"]
    assert first.manifest["calibration"]["scale_grid"] == gaussian.manifest["calibration"]["scale_grid"]
    assert first.manifest["edges"] == gaussian.manifest["edges"]
    pd.testing.assert_frame_equal(first.train, larger.train.iloc[:20])
    pd.testing.assert_frame_equal(first.test, larger.test.iloc[:20])


def test_legacy_generator_does_not_add_new_manifest_fields():
    data = generator.generate_metadata_dag_dataset(config())
    assert data.manifest["generator"] == "independent-dag-aggregate-metadata-calibration-v1"
    assert "calibration_method" not in data.manifest["config"]
    assert "calibration_repeats" not in data.manifest["config"]
    assert "covariate_effect_policy" not in data.manifest["config"]
    assert "evaluation_sample_size" not in data.manifest["calibration"]


def test_calibration_tolerances_keep_relative_and_absolute_units():
    targets = {key: 1. for key in audit.TOLERANCES}
    targets["effective_rank"] = 40.
    draws = [dict(targets) for _ in range(50)]
    for row in draws:
        row["effective_rank"] = 51.
        row["absolute_correlation_q95"] = 1.09
    summary = {row["metric"]: row for row in audit.summarize_draws(draws, targets)}
    assert summary["effective_rank"]["scaled_difference"] == pytest.approx(.275)
    assert summary["effective_rank"]["systematic_tolerance_miss"]
    assert summary["absolute_correlation_q95"]["systematic_tolerance_miss"]
    assert not summary["pd_prevalence"]["systematic_tolerance_miss"]


def test_nonfinite_validation_cannot_pass():
    targets = {key: 1. for key in audit.TOLERANCES}
    draws = [dict(targets), dict(targets)]
    draws[0]["effective_rank"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        audit.summarize_draws(draws, targets)


def test_covariate_policy_preserves_edges_and_only_scales_direct_exogenous_effects():
    cfg = replace(config(), dimension=20, calibration_scales=(1.,))
    legacy = generator.generate_metadata_dag_dataset(cfg)
    revised = generator.generate_metadata_dag_dataset(replace(cfg, covariate_effect_policy="metadata-ratio-v1"))
    assert legacy.manifest["edges"] == revised.manifest["edges"]
    target = revised.manifest["calibration"]["target"]
    for node, mechanism in legacy.manifest["mechanisms"].items():
        for parent, coefficient in mechanism["coefficients"].items():
            expected = coefficient
            if node not in {"age_z", "female", "PD"} and parent in {"age_z", "female"}:
                expected *= .17/.475 * target[f"mean_abs_correlation_with_{parent}"] / target["mean_abs_correlation_with_PD"]
            assert revised.manifest["mechanisms"][node]["coefficients"][parent] == pytest.approx(expected)


def test_pilot_uses_frozen_model_settings_without_changing_legacy_score_policy():
    directory = Path(__file__).resolve().parents[1] / "configs"
    execution = _execution_config(directory / "execution_section2_pilot_v3.json")
    accepted = json.loads((directory / execution["accepted_model_freeze"]).read_text())
    plan = build_experiment_plan("s2", master_seed=20260923, profile="pilot")
    for dataset in plan.datasets:
        runtime = _profile_config(asdict(dataset), execution, "pilot", 1)
        expected = "clg-bic" if dataset.role == "m4_cg_compatible" else "rank-native"
        assert runtime.score_method == expected
        for key, value in accepted["primary_ordersmc"].items():
            assert getattr(runtime, key) == value
    assert execution["prior"] == accepted["prior"]
    assert execution["metadata_calibration_gate"]["validation_repeats"] == 50


def test_prepared_gate_binds_the_exact_audited_primary_generator(tmp_path, monkeypatch):
    plan = {"datasets": [{"role": "m1_matched", "dataset_id": "d0"}]}
    identity = {"plan_sha256": "test", "source_sha256": "source"}
    execution = {"metadata_calibration_gate": {"required": True}}
    monkeypatch.setattr(audit, "binding", lambda *args: (plan, execution, identity))
    body = {"mechanisms": {"x": {"coefficient": 1.}}}
    report = tmp_path / "calibration_validation/rep000.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"generator_manifest_sha256": _sha(body)}))
    gate = tmp_path / "validation/calibration-gate.json"
    gate.parent.mkdir()
    gate.write_text(json.dumps({"status": "passed", "identity": identity,
                               "source_artifacts": {"rep000.json": audit.sha(report)}}))
    dataset = tmp_path / "datasets/d0/generator_manifest.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps({**body, "artifacts": {}, "artifact_fingerprint_sha256": "ignored"}))
    audit.validate_gate(tmp_path, tmp_path / "config.json", "pilot", prepared=True)
    dataset.write_text(json.dumps({"mechanisms": {}}))
    with pytest.raises(RuntimeError, match="audited mechanism"):
        audit.validate_gate(tmp_path, tmp_path / "config.json", "pilot", prepared=True)
    report.write_text("{}")
    with pytest.raises(RuntimeError, match="changed after audit"):
        audit.validate_gate(tmp_path, tmp_path / "config.json", "pilot")


def test_failed_or_stale_calibration_gate_cannot_launch(tmp_path, monkeypatch):
    identity = {"source_sha256": "new"}
    monkeypatch.setattr(audit, "binding", lambda *args: (
        {}, {"metadata_calibration_gate": {"required": True}}, identity))
    path = tmp_path / "validation/calibration-gate.json"
    path.parent.mkdir()
    for status, claimed in [("failed", identity), ("passed", {"source_sha256": "old"})]:
        path.write_text(json.dumps({"status": status, "identity": claimed}))
        with pytest.raises(RuntimeError, match="missing, failed or stale"):
            audit.validate_gate(tmp_path, tmp_path / "config.json", "pilot")
