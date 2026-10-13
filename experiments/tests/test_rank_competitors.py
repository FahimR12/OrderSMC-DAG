from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pytest

from ordersmc_experiments.rank_competitors import task_list,seed_record,fit_identity,checked_plan,fit_path,MCMC
from ordersmc_experiments.readiness import canonical_sha


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/rank_competitors_v1.json").read_text())


def datasets(cfg):
    return [dict(dataset_id=f"{noise}-{rep}-{n}",noise=noise,replicate=rep,n_train=n,
        seed_count=cfg['seeds_high_n'] if n==5000 else cfg['seeds_low_n'],candidate_sha256='same-target')
        for noise in cfg['noise_families'] for rep in cfg['replicates'] for n in cfg['sample_sizes']]


def test_complete_competitor_design_and_no_duplicate_fits():
    cfg=config(); data=datasets(cfg);tasks=task_list(data,cfg)
    records=[(t['dataset_id'],t['method'],t['budget'],s) for t in tasks for s in t['seed_indices']]
    assert len(data)==24
    assert len(records)==len(set(records))==1392
    assert len([t for t in tasks if t['phase']=='primary'])==240
    assert len([t for t in tasks if t['phase']=='extended'])==96
    assert sum(r[1] in MCMC for r in records)==1344
    assert all(t['method'] in MCMC and t['dataset_id'].endswith('5000') for t in tasks if t['phase']=='extended')


def test_seed_identity_is_fixed_across_budgets_not_shared_across_methods_or_data():
    cfg=config(); plan={'config':cfg,'plan_sha256':'plan'};data=datasets(cfg)
    seeds=[seed_record(plan,d,m,i).seed for d in data for m in cfg['methods'] for i in range(d['seed_count'])]
    assert len(set(seeds))==len(seeds)
    left=fit_identity(plan,data[0],'order-mcmc','baseline',0)
    right=fit_identity(plan,data[0],'order-mcmc','extended',0)
    assert left['seed']==right['seed']
    assert left['candidate_sha256']==right['candidate_sha256']
    assert left['budget']!=right['budget']


def test_plan_source_guard_fails_closed(tmp_path,monkeypatch):
    import ordersmc_experiments.rank_competitors as runner
    source=tmp_path/'source.py'; source.write_text('original')
    monkeypatch.setattr(runner,'REPO',tmp_path)
    from ordersmc_experiments.readiness_audits import sha
    plan={'source_files':{'source.py':sha(source)}}
    plan['plan_sha256']=canonical_sha(plan)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    assert checked_plan(tmp_path)==plan
    source.write_text('changed')
    with pytest.raises(RuntimeError,match='source changed'):checked_plan(tmp_path)
    plan['source_files']={}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(RuntimeError,match='plan hash'):checked_plan(tmp_path)


def test_accepted_source_ids_are_exactly_bound():
    cfg=config()
    accepted=json.loads((Path(__file__).resolve().parents[1]/'configs/rank_acceptance_freeze_v1.json').read_text())
    assert cfg['source_suite_sha256']==accepted['rank_suite_sha256']
    assert cfg['source_budget_plan_sha256']==accepted['budget_plan_sha256']


def test_sparse_seed_probabilities_are_completed_without_changing_inputs():
    from copy import deepcopy
    from ordersmc_experiments.rank_competitors import reporting_records
    from ordersmc_experiments.readiness_audits import pips, rmse
    records=[{'edge_probabilities':[dict(parent='a',child='b',probability=.6)]},
             {'edge_probabilities':[dict(parent='b',child='c',probability=.8)]},
             {'edge_probabilities':[]}]
    before=deepcopy(records)
    fixed=reporting_records(records,['a','b','c'])
    assert records==before
    assert all(len(pips(r))==6 for r in fixed)
    assert pips(fixed[0])['b','c']==0
    assert pips(fixed[1])['b','c']==.8
    assert np.mean([pips(r)['b','c'] for r in fixed])==pytest.approx(.8/3)
    assert rmse(pips(fixed[0]),pips(fixed[1]))==pytest.approx(np.sqrt((.6**2+.8**2)/6))
    assert reporting_records(list(reversed(records)),['a','b','c'])==list(reversed(fixed))


def test_aggregate_accepts_disjoint_sparse_smc_records(tmp_path,monkeypatch):
    import ordersmc_experiments.rank_competitors as runner
    from ordersmc_experiments.adapter_validation import adapter_fixture_table
    from ordersmc_experiments.readiness_audits import read
    import pandas as pd
    table=adapter_fixture_table()
    case={'truth_edges':[['a','b'],['b','c']], 'variable_types':dict(a='con',b='con',c='con')}
    common=dict(representative_edges=[], own_target_pip_rmse=0.)
    records=[{**common,'edge_probabilities':[dict(parent='a',child='b',probability=.6)]},
             {**common,'edge_probabilities':[dict(parent='b',child='c',probability=.8)]}]
    dataset=dict(dataset_id='fixture',noise='gaussian',replicate=0,n_train=250,seed_count=2,max_indegree=2)
    plan=dict(datasets=[dataset],tasks=[],config=config(),plan_sha256='fixture')
    monkeypatch.setattr(runner,'checked_plan',lambda *a,**kw:plan)
    monkeypatch.setattr(runner,'load_dataset',lambda *a:(table,case))
    monkeypatch.setattr(runner,'smc_records',lambda *a:records)
    result=runner.aggregate(tmp_path)
    assert result['status']=='complete'
    assert result['reused_smc_fits']==2
    assert len(pd.read_csv(tmp_path/'analysis/mean_pip_graph_metrics.csv'))==1
    stability=pd.read_csv(tmp_path/'analysis/stability.csv')
    assert stability.iloc[0].mean_pairwise_pip_rmse==pytest.approx(np.sqrt(1/6))
