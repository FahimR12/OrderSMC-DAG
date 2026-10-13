from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ordersmc import OrderSMCConfig, build_candidates
from ordersmc.persistence import save_candidate_table, candidate_table_content_sha256
from ordersmc_experiments import catchup, catchup_submit
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.exact_order import exact_order_dp
from ordersmc_experiments.readiness import canonical_sha
from ordersmc_experiments.seeds import protocol_seed_record


def tiny_plan(tmp_path):
    cfg = OrderSMCConfig(max_parents=1, screen_k=1, parent_pool_max=1,
                         particles=24, mh_steps=1, mh_steps_resample=2,
                         final_target_snapshot_steps=(0, 2))
    data = pd.DataFrame(dict(x=np.linspace(-2, 2, 30), y=np.arange(30) % 2))
    table = build_candidates(data, dict(x="con", y="bin"), dict(x=0, y=2), config=cfg, verbose=False)
    candidate = save_candidate_table(table, tmp_path / "candidate.json.gz")
    exact = exact_order_dp(table)
    oldpath = tmp_path / "old.json"
    _atomic_json(oldpath, dict(edge_probabilities=catchup._all_edges(table.nodes, exact.edge_probs)))
    d = dict(key="tiny", cohort="metadata", dataset_id="tiny", candidate=catchup.artifact(candidate),
             candidate_sha256=candidate_table_content_sha256(table), nodes=table.nodes, max_parents=1,
             truth_edges=[["x", "y"]], inputs=[], score="native", spec=dict(cell_id="tiny", n_train=30, replicate=0),
             references={"exact-order": catchup.locator(oldpath)})
    fits=[]
    for index in range(2):
        seed=asdict(protocol_seed_record(231, experiment="catchup-test", cell="tiny", replicate=0, component=f"s{index}"))
        fits.append(dict(fit_id=f"fit{index}", dataset_key="tiny", cohort="metadata", budget="baseline",
            role="evaluation" if index == 0 else "reproducibility-extension", seed_index=index,
            seed_record=seed, config=cfg.with_overrides(seed=seed["seed"]).to_dict(), mpi_ranks=1,
            original=catchup.locator(oldpath)))
    plan=dict(source_files={}, evidence=[], datasets=[d], fits=fits, comparisons=[],
              config=dict(exact_rmse_flags=dict(mean=.012,p95=.02), mpi_concurrency_per_cohort=2, serial_concurrency=2),
              tasks=[dict(cohort="metadata",mpi_ranks=1,fit_ids=[f["fit_id"] for f in fits])])
    plan["plan_sha256"]=canonical_sha(plan)
    # Normalize tuple serialization, as happens in a real frozen plan.
    _atomic_json(tmp_path / "plan.json",plan)
    return catchup.read(tmp_path / "plan.json")


def test_end_to_end_catchup_is_nonoverwriting_and_keeps_seed_roles(tmp_path):
    plan=tiny_plan(tmp_path)
    original=(tmp_path / "old.json").read_bytes()
    with pytest.raises(RuntimeError, match="preflight.json"):
        catchup.run(tmp_path,0)
    catchup.preflight(tmp_path)
    with pytest.raises(RuntimeError,match="absent/invalid"):
        catchup.aggregate(tmp_path,"metadata")
    assert not (tmp_path / "analysis/metadata/per_run_metrics.csv").exists()
    catchup.run(tmp_path,0)
    saved=(tmp_path / "runs/fit0/result.json").read_bytes()
    catchup.run(tmp_path,0)
    assert (tmp_path / "runs/fit0/result.json").read_bytes()==saved
    catchup.aggregate(tmp_path,"metadata")
    pooled=pd.read_csv(tmp_path / "analysis/metadata/mean_pip_graph_metrics.csv")
    assert (pooled.evaluation_seeds==1).all()
    stable=pd.read_csv(tmp_path / "analysis/metadata/seed_stability.csv")
    assert (stable.seeds==2).all()
    assert (tmp_path / "old.json").read_bytes()==original
    assert catchup.fit_record(tmp_path,plan,plan["fits"][0])[0]["status"]=="complete"
    catchup_submit.completion(tmp_path)


def test_corrected_config_does_not_change_scores_or_original_budget():
    original=OrderSMCConfig(particles=4096,mh_steps=10,mh_steps_resample=25,seed=123)
    config=dict(bridge=dict(tempering_path="geometric",adaptive_tempering=True,adaptive_max_stages=2000,
                adaptive_min_beta_increment=1e-8,adaptive_cess_target=.9),
                metadata_snapshot_steps=[0,250,1000],primary_snapshot_steps=0)
    updated=catchup.corrected_config(original,config,True)
    assert updated.candidate_build_dict()==original.candidate_build_dict()
    assert (updated.particles,updated.mh_steps,updated.mh_steps_resample,updated.seed)==(4096,10,25,123)
    assert updated.final_target_snapshot_steps==(0,250,1000)


def test_submission_records_partial_acceptance_and_resumes_without_duplicates(tmp_path,monkeypatch):
    tiny_plan(tmp_path)
    calls=[]
    def sbatch(args,**kwargs):
        calls.append(args)
        if len(calls)==2:
            raise RuntimeError("temporary submission limit")
        return "1234\n"
    with pytest.raises(RuntimeError,match="submission limit"):
        catchup_submit.submit(tmp_path,"start",sbatch=sbatch)
    assert catchup.read(tmp_path / "submission.json")["jobs"]=={"preflight":"1234"}
    second=[]
    def retry(args,**kwargs):
        second.append(args)
        return "1235\n"
    catchup_submit.submit(tmp_path,"start",sbatch=retry)
    assert len(second)==1
    assert "--dependency=afterany:1234" in second[0]
    with pytest.raises(FileNotFoundError):
        catchup_submit.submit(tmp_path,"launch",sbatch=retry)
    assert len(second)==1


def test_tampered_plan_and_input_fail_closed(tmp_path):
    tiny_plan(tmp_path)
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(RuntimeError,match="input changed"):
        catchup.preflight(tmp_path)
    plan=catchup.read(tmp_path / "plan.json")
    plan["fits"][0]["seed_index"]=999
    _atomic_json(tmp_path / "plan.json",plan)
    with pytest.raises(RuntimeError,match="plan hash"):
        catchup.checked_plan(tmp_path)


def test_inventory_counts_cohorts_not_task_grouping_tuples(tmp_path):
    plan=tiny_plan(tmp_path)
    catchup.write_inventory(tmp_path,plan)
    inventory=catchup.read(tmp_path / "inventory.json")
    assert inventory["cohorts"]==[dict(cohort="metadata",datasets=1,corrected_fits=2,new_fits=2,array_tasks=1)]
