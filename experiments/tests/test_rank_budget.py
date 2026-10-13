from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pytest

from ordersmc import OrderSMCConfig
from ordersmc_experiments import rank_budget as budget
from ordersmc_experiments.readiness import canonical_sha, summarize_posterior
from ordersmc_experiments.readiness_audits import read, write, sha

ROOT = Path(__file__).resolve().parents[2]


def fixture_plan(tmp_path, seeds=2):
    config = read(ROOT / "experiments/configs/rank_budget_v1.json")
    config.update(method_seeds=seeds, seeds_per_task=2)
    base = OrderSMCConfig.protocol_v2(lambda_target=1., score_method="rank-native", max_parents=2,
                                    **config["budgets"]["baseline"])
    dataset = dict(dataset_id="gaussian-rep000", noise="gaussian", replicate=0, n_train=5000,
                   base_config=base.to_dict(), candidate_sha256="test-fixture")
    reference = dict(variable_types={"a": "con", "b": "bin"}, truth_edges=[["a", "b"]],
                     smc_runs=[{"previous_study_run": "must not enter the twenty-seed summaries"}],
                     references={"exact-order": dict(representative_edges=[["a", "b"]],
                         edge_probabilities=[dict(parent="a", child="b", probability=.8),
                                             dict(parent="b", child="a", probability=.2)])})
    path = tmp_path / "datasets/gaussian-rep000/reference.json"
    path.parent.mkdir(parents=True)
    write(path, reference)
    plan = dict(config=config, datasets=[dataset], source_files={},
                input_files={str(path.relative_to(tmp_path)): sha(path)})
    plan["tasks"] = budget.task_list(plan["datasets"], config)
    plan["plan_sha256"] = canonical_sha(plan)
    write(tmp_path / "plan.json", plan)
    return plan, dataset


def save_run(tmp_path, plan, dataset, arm, index):
    graph = nx.DiGraph([("a", "b")])
    summary = summarize_posterior({("a", "b"): .8, ("b", "a"): .2}, graph, ["a", "b"], 2)
    record = dict(status="complete", plan_sha256=plan["plan_sha256"], dataset_id=dataset["dataset_id"],
                  budget=arm, seed_index=index, seed=asdict(budget.seed_record(plan, dataset, index)),
                  config=budget.sampling_config(plan, dataset, arm, index).to_dict(),
                  candidate_sha256=dataset["candidate_sha256"], mpi_ranks=plan["config"]["mpi_ranks"],
                  own_target_pip_rmse=0., **summary)
    path = budget.run_path(tmp_path, dataset, arm, index)
    path.parent.mkdir(exist_ok=True)
    write(path, record)
    return path


def test_frozen_budget_has_320_fits_and_only_sampling_changes(tmp_path):
    plan, dataset = fixture_plan(tmp_path, seeds=20)
    config = read(ROOT / "experiments/configs/rank_budget_v1.json")
    data = [dict(dataset, dataset_id=f"{noise}-{rep}", noise=noise, replicate=rep)
            for noise in config["noise_families"] for rep in config["replicates"]]
    tasks = budget.task_list(data, config)
    assert len(tasks) == 64
    assert sum(len(t["seed_indices"]) for t in tasks) == 320
    for d in data:
        for index in range(20):
            left = budget.sampling_config(plan, d, "baseline", index)
            right = budget.sampling_config(plan, d, "extended", index)
            assert left.candidate_build_dict() == right.candidate_build_dict()
            assert left.seed == right.seed
            assert right.particles == 4 * left.particles
            assert right.mh_steps == 2 * left.mh_steps
            assert right.mh_steps_resample == 2 * left.mh_steps_resample
            assert {k for k in left.to_dict() if left.to_dict()[k] != right.to_dict()[k]} == {
                "particles", "mh_steps", "mh_steps_resample"}
    assert len({budget.seed_record(plan, d, i).seed for d in data for i in range(20)}) == 160


def test_plan_tampering_rejected(tmp_path):
    plan, _ = fixture_plan(tmp_path)
    plan["config"]["method_seeds"] = 3
    write(tmp_path / "plan.json", plan)
    with pytest.raises(RuntimeError, match="plan hash"):
        budget.checked_plan(tmp_path)


def test_source_change_rejected(tmp_path):
    plan, _ = fixture_plan(tmp_path)
    plan["source_files"] = {"src/ordersmc/config.py": "bad"}
    plan["plan_sha256"] = canonical_sha({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(tmp_path / "plan.json", plan)
    with pytest.raises(RuntimeError, match="source mismatch"):
        budget.checked_plan(tmp_path, sources=True)


@pytest.mark.parametrize("field,value", [("seed_index", 1), ("budget", "extended"), ("mpi_ranks", 1),
                                         ("candidate_sha256", "changed"), ("status", "failed")])
def test_incompatible_existing_result_rejected(tmp_path, field, value):
    plan, dataset = fixture_plan(tmp_path)
    record = read(save_run(tmp_path, plan, dataset, "baseline", 0))
    record[field] = value
    with pytest.raises(RuntimeError, match="identity/configuration"):
        budget.validate_run(record, plan, dataset, "baseline", 0)


def test_missing_runs_are_explicit_and_complete_runs_aggregate(tmp_path):
    plan, dataset = fixture_plan(tmp_path)
    save_run(tmp_path, plan, dataset, "baseline", 0)
    with pytest.raises(RuntimeError, match="3 runs missing"):
        budget.aggregate(tmp_path)
    assert read(tmp_path / "analysis/completion.json")["status"] == "incomplete"
    assert not (tmp_path / "analysis/stability.csv").exists()
    for arm, index in [("baseline", 1), ("extended", 0), ("extended", 1)]:
        save_run(tmp_path, plan, dataset, arm, index)
    budget.aggregate(tmp_path)
    assert read(tmp_path / "analysis/completion.json")["completed_runs"] == 4
    assert read(tmp_path / "analysis/completion.json")["status"] == "complete"
    import pandas as pd
    stability = pd.read_csv(tmp_path / "analysis/stability.csv")
    assert stability.directed_all_identical.all()
    assert (stability.unique_directed_graphs == 1).all()
    assert (stability.exact_representative_match_fraction == 1).all()


def test_inconsistent_saved_pip_error_is_rejected(tmp_path):
    plan, dataset = fixture_plan(tmp_path)
    path = save_run(tmp_path, plan, dataset, "baseline", 0)
    record = read(path)
    record["own_target_pip_rmse"] = .3
    write(path, record)
    with pytest.raises(RuntimeError, match="reference error"):
        budget.aggregate(tmp_path)


def test_sparse_sampler_mapping_is_completed_with_zeros():
    assert budget.complete_probabilities({("a", "b"): 1.}, ["a", "b"]) == {
        ("a", "b"): 1., ("b", "a"): 0.}
    assert budget.complete_probabilities({}, ["a", "b"]) == {
        ("a", "b"): 0., ("b", "a"): 0.}


def test_sparse_mapping_rejects_unknown_edges_and_invalid_probabilities():
    with pytest.raises(RuntimeError, match="unknown node or self edge"):
        budget.complete_probabilities({("a", "c"): .2}, ["a", "b"])
    with pytest.raises(ArithmeticError):
        budget.complete_probabilities({("a", "b"): 1.1}, ["a", "b"])
