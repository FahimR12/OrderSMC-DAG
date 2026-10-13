from dataclasses import asdict

import numpy as np

from ordersmc import OrderSMCConfig
from ordersmc.persistence import load_candidate_table
from ordersmc_experiments import prior_holdout
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.metadata_profiles import MetadataDAGConfig
from ordersmc_experiments.readiness import canonical_sha
from ordersmc_experiments.readiness_audits import read


def test_holdout_changes_only_random_draw_identity():
    original = asdict(MetadataDAGConfig(profile_id="protein_pathway", master_seed=9))
    changed = prior_holdout.heldout_generator_config(original, 100)
    assert {key for key in original if original[key] != changed[key]} == {
        "master_seed", "experiment_id", "replicate_id"}
    assert changed["master_seed"] != original["master_seed"]


def test_real_holdout_preparation_preserves_shared_likelihood_and_resumes(tmp_path):
    generator = MetadataDAGConfig(profile_id="protein_pathway", dimension=8, n_train=40, n_test=30,
        pilot_size=200, calibration_scales=(.5, 1.), calibration_repeats=2,
        calibration_method="finite-sample-v2", covariate_effect_policy="metadata-ratio-v1",
        master_seed=2026092102, replicate_id=100)
    cfg = OrderSMCConfig.protocol_v2(lambda_target=1., score_method="rank-native", max_parents=2,
        screen_k=7, parent_pool_max=None, pair_rescue_anchor_k=0, pair_rescue_k=0)
    spec = dict(key="test-heldout", kind="metadata-signal", replicate=100,
        generator_config=asdict(generator), score_config=cfg.to_dict())
    plan = dict(cases=[spec], source_files={}); plan["plan_sha256"] = canonical_sha(plan)
    _atomic_json(tmp_path / "plan.json", plan)
    _atomic_json(tmp_path / "validation/preflight.json", dict(all_passed=True, plan_sha256=plan["plan_sha256"]))
    prior_holdout.run(tmp_path, 0)
    record_path = tmp_path / "prepared/test-heldout.json"
    original = record_path.read_bytes(); record = read(record_path)
    first, second = [load_candidate_table(c["candidate"]["path"]) for c in record["cases"]]
    assert all(np.array_equal(a, b) for a, b in zip(first.loglik, second.loglik))
    assert all(c["prior_audit"]["local_likelihood_arrays_bit_identical"] for c in record["cases"])
    assert record["no_recovery_based_selection"]
    assert record["metadata_calibration"]["selection_uses_graph_recovery"] is False
    prior_holdout.run(tmp_path, 0)
    assert record_path.read_bytes() == original
