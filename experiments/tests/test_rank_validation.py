from pathlib import Path

import pytest

from ordersmc_experiments import rank_validation as validation
from ordersmc_experiments.readiness_audits import read, write


ROOT = Path(__file__).resolve().parents[2]


def test_fixed_design_is_paired_without_prior_or_smc_retuning():
    spec = read(ROOT / "experiments/configs/rank_validation_v1.json")
    base = read(ROOT / "experiments/configs/section2_readiness_v1.json")
    arms = validation.arm_configs(spec, base)
    assert len(arms) == 6
    assert sum(a["config"]["replicates"] for a in arms) == 36
    for i in range(0, len(arms), 2):
        left, right = [a["config"].copy() for a in arms[i:i+2]]
        assert left.pop("score_method") == "native"
        assert right.pop("score_method") == "rank-native"
        assert left == right
        assert left["edge_prior_lambda"] == 1.
        assert left["ordersmc"] == base["ordersmc"]
    assert sum(a["config"]["replicates"] * a["config"]["method_seeds"] * 6 for a in arms) == 768


def test_freeze_idempotent_and_changed_settings_rejected(tmp_path):
    path = ROOT / "experiments/configs/rank_validation_v1.json"
    first = validation.freeze(tmp_path, path, smoke=True)
    assert validation.freeze(tmp_path, path, smoke=True) == first
    assert validation.load_suite(tmp_path) == first
    with pytest.raises(FileExistsError):
        validation.freeze(tmp_path, path, smoke=False)


def test_suite_hash_tampering_rejected(tmp_path):
    validation.freeze(tmp_path, ROOT / "experiments/configs/rank_validation_v1.json", smoke=True)
    path = tmp_path / "suite.json"
    suite = read(path)
    suite["tasks"][0]["index"] = 100
    write(path, suite)
    with pytest.raises(RuntimeError, match="suite hash mismatch"):
        validation.load_suite(tmp_path)


def test_missing_fits_are_not_reported_as_complete(tmp_path):
    validation.freeze(tmp_path, ROOT / "experiments/configs/rank_validation_v1.json", smoke=True)
    with pytest.raises(RuntimeError, match="conditions missing"):
        validation.aggregate(tmp_path)
    report = read(tmp_path / "analysis/completion.json")
    assert report["status"] == "incomplete"
    assert len(report["missing"]) == 12
    assert not (tmp_path / "analysis/summary.csv").exists()


def test_agreement_distinguishes_direction_from_skeleton():
    assert validation.agreement([{("a", "b")}, {("b", "a")}]) == dict(mean_jaccard=0., all_identical=False)
    assert validation.agreement([set(), set()]) == dict(mean_jaccard=1., all_identical=True)
