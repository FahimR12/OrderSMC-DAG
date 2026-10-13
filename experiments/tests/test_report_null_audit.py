import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_t

from ordersmc import OrderSMCConfig, build_candidates
from ordersmc.scoring import CLGScorer
from ordersmc.persistence import save_candidate_table, candidate_table_content_sha256
from ordersmc_experiments import catchup, null_pilot, reporting_repair, audit_submit
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.exact_order import exact_order_dp
from ordersmc_experiments.prior_calibration import retarget_edge_count_prior
from ordersmc_experiments.readiness import canonical_sha
from ordersmc_experiments.seeds import protocol_seed_record
from ordersmc_experiments.null_score_audit import full_support_prior_edges
from test_catchup import tiny_plan


@pytest.mark.parametrize("parents",[0,1,2])
def test_nig_score_matches_independent_multivariate_t_density(parents):
    rng=np.random.default_rng(20260921);n=15
    y=rng.normal(size=n);z=np.column_stack([np.ones(n),rng.normal(size=(n,parents))])
    scorer=CLGScorer({}, {}, {}, {}, tau=3.,ig_shape=2.,ig_scale=1.)
    actual=scorer._nig_log_marginal(y,z)
    expected=multivariate_t.logpdf(y,loc=np.zeros(n),shape=.5*(np.eye(n)+9.*z@z.T),df=4.)
    assert actual==pytest.approx(expected,abs=1e-10)


def test_fresh_null_signal_generator_and_global_prior_rule():
    null,types,cards,edges=null_pilot.fresh_data(0,False)
    signal,_,_,truth=null_pilot.fresh_data(0,True)
    assert null.shape==(222,8) and not edges and len(truth)==7
    assert set(types.values())=={"con","bin","cat"}
    assert null.equals(null_pilot.fresh_data(0,False)[0])
    assert not signal.equals(null)
    for d in (8,20,196):
        penalty=null_pilot.dimension_penalty(d)
        assert 1/(1+np.exp(penalty))==pytest.approx(1/d)


def test_real_small_prior_retarget_and_sampler(tmp_path):
    data,types,cards,truth=null_pilot.fresh_data(0,False,n=50)
    cfg=OrderSMCConfig.protocol_v2(lambda_target=1.,score_method="rank-native",max_parents=1,
        screen_k=7,parent_pool_max=None,particles=24,mh_steps=1,mh_steps_resample=2,final_target_snapshot_steps=(0,2))
    source=build_candidates(data,types,cards,config=cfg,verbose=False)
    cfg=cfg.with_overrides(edge_penalty=null_pilot.dimension_penalty(8))
    table,audit=retarget_edge_count_prior(source,target_label="order",lambda_target=cfg.edge_penalty,target_config=cfg)
    assert audit.local_likelihood_arrays_bit_identical
    assert all(np.array_equal(a,b) for a,b in zip(source.loglik,table.loglik))
    path=save_candidate_table(table,tmp_path/"candidate.json.gz")
    exact=exact_order_dp(table,compute_graph_size=False)
    prior=exact_order_dp(table,beta=0.,compute_graph_size=False)
    assert sum(prior.edge_probs.values())==pytest.approx(full_support_prior_edges(8,cfg.edge_penalty,k=1),abs=1e-10)
    ref=tmp_path/"exact.json";_atomic_json(ref,dict(edge_probabilities=catchup._all_edges(table.nodes,exact.edge_probs)))
    case=dict(case_id="tiny",config=cfg.to_dict(),candidate=catchup.artifact(path),
        candidate_sha256=candidate_table_content_sha256(table),exact=catchup.artifact(ref),arm="dimension",penalty=cfg.edge_penalty,
        dataset=dict(key="tiny",kind="fresh-null",nodes=table.nodes,truth_edges=truth,replicate=0))
    task=dict(task_id="tiny-s0",case_id="tiny",seed_index=0,mpi_ranks=1,
        seed_record=asdict(protocol_seed_record(22,experiment="audit-test",cell="tiny",replicate=0,component="fit")))
    plan=dict(cases=[case],tasks=[task],source_files={});plan["plan_sha256"]=canonical_sha(plan)
    _atomic_json(tmp_path/"plan.json",plan)
    _atomic_json(tmp_path/"validation/preflight.json",dict(all_passed=True,plan_sha256=plan["plan_sha256"]))
    null_pilot.run(tmp_path,0);null_pilot.aggregate(tmp_path)
    result=catchup.read(tmp_path/"runs/tiny-s0/result.json")
    assert all(s["selected_edges"]==s["false_edges"] and s["true_edges"]==0 for s in result["snapshots"])
    assert all(s["own_target_pip_rmse"]>=0 for s in result["snapshots"])
    saved=(tmp_path/"runs/tiny-s0/result.json").read_bytes();null_pilot.run(tmp_path,0)
    assert (tmp_path/"runs/tiny-s0/result.json").read_bytes()==saved


def test_new_report_edition_does_not_overwrite_original_artifacts(tmp_path):
    plan=tiny_plan(tmp_path);catchup.preflight(tmp_path);catchup.run(tmp_path,0)
    catchup.aggregate(tmp_path,"metadata")
    original=(tmp_path/"analysis/metadata/per_run_metrics.csv").read_bytes()
    fit=(tmp_path/"runs/fit0/result.json").read_bytes()
    destination=tmp_path/"new-edition/metadata"
    catchup.aggregate_saved_fits(tmp_path,"metadata",plan,destination)
    assert (destination/"per_run_metrics.csv").read_bytes()==original
    assert (tmp_path/"analysis/metadata/per_run_metrics.csv").read_bytes()==original
    assert (tmp_path/"runs/fit0/result.json").read_bytes()==fit
    decisions=catchup.read(destination/"representative_graphs.json")["decisions"]
    assert decisions and all(d["diagnostics"]["milp_objective_scale_exponent"]==20 for d in decisions)


def test_submission_resumes_and_requires_preflight(tmp_path):
    plan=dict(source_files={},cohorts=["v1"]);plan["plan_sha256"]=canonical_sha(plan)
    _atomic_json(tmp_path/"plan.json",plan);calls=[]
    def sbatch(args,**kwargs):
        calls.append(args)
        if len(calls)==2:raise RuntimeError("limit")
        return "123\n"
    with pytest.raises(RuntimeError,match="limit"):audit_submit.submit(tmp_path,"reporting_repair","start",sbatch=sbatch)
    assert catchup.read(tmp_path/"submission.json")["jobs"]=={"preflight":"123"}
    follow=[]
    def resume(args,**kwargs):follow.append(args);return "124\n"
    audit_submit.submit(tmp_path,"reporting_repair","start",sbatch=resume)
    assert len(follow)==1 and "--dependency=afterany:123" in follow[0]
    with pytest.raises(FileNotFoundError):audit_submit.submit(tmp_path,"reporting_repair","launch",sbatch=resume)
