from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ordersmc_experiments import scaling_quality as quality, timing_pilot, timing_submit
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.readiness import canonical_sha
from ordersmc_experiments.readiness_audits import read
from test_timing_pilot import tiny_timing_plan, config


def settings():
    path=Path(__file__).resolve().parents[1]/"configs/scaling_quality_v1.json"
    return {**config(), **read(path)}


def test_followup_inventory_and_independent_seed_edition():
    cfg=settings(); datasets=[dict(key=f"d{d}",nodes=list(range(d))) for d in (12,56,196)]
    tasks,bundles,refs=timing_pilot.make_tasks(cfg,datasets)
    assert len(tasks)==243 and len(bundles)==9 and not refs
    assert len({t["task_id"] for t in tasks})==243
    assert sum(t["purpose"]=="fixed-work-scaling" for t in tasks)==36
    assert sum(t["method"]=="order-mcmc" for t in tasks)==189
    assert all(t["arm"]=="dimension" for t in tasks)
    assert sorted(i for b in bundles for i in b["tasks"])==list(range(243))
    assert quality.make_tasks(cfg,datasets)==(tasks,bundles,refs)
    assert min(t["wall_seconds"] for t in tasks if t["method"]=="order-mcmc")==.5
    assert all(t["particles"]==512 for t in tasks if t["purpose"]=="fixed-work-scaling")
    assert timing_pilot.seed_record(cfg,"d12",0,"evaluation-smc") != timing_pilot.seed_record(config(),"d12",0,"evaluation-smc")


def test_pair_state_shd_counts_reversal_once():
    assert quality.pair_shd({("a","b")},{("b","a")})==1
    assert quality.pair_shd(set(),{("a","b")})==1
    assert quality.pair_shd({("a","b"), ("b","c")},{("b","a"), ("b","c")})==1
    assert quality.pair_shd({("a","b")},{("a","b")})==0


def crossing_frame():
    rows=[]
    for method,budget,seconds in (("MCMC",480,500),("MCMC",.5,1),("SMC fixed",512,2)):
        for seed in range(3):
            rows.append(dict(dataset_key="tiny",arm="dimension",variant=method,budget=budget,ranks=64,
                seed_index=seed,inference_seconds=seconds,directed_f1=.8,skeleton_f1=.9,shd_per_true_edge=.1))
    return pd.DataFrame(rows)


def test_recovery_crossing_uses_short_mcmc_and_all_seeds():
    frame=crossing_frame(); cfg=settings()
    result=quality.recovery_crossings(frame,cfg).iloc[0]
    assert result.speedup==.5  # The method may lose: never assume SMC wins.
    assert result.mcmc_budget==.5
    frame.loc[(frame.variant=="SMC fixed") & (frame.seed_index==0),"directed_f1"]=.7
    assert quality.recovery_crossings(frame,cfg).iloc[0].status=="not_established_on_tested_grid"
    frame=crossing_frame().iloc[:-1]
    assert quality.recovery_crossings(frame,cfg).iloc[0].status=="not_established_on_tested_grid"
    frame=crossing_frame().drop(index=0)
    assert quality.recovery_crossings(frame,cfg).empty  # incomplete long benchmark


def test_scaling_rejects_changed_work_and_hardware():
    rows=[dict(dataset_key="tiny",arm="dimension",variant="SMC fixed",d=2,seed_index=0,
        ranks=r,particles=512,inference_seconds=64/r,host="node1",cpu_model="cpu",proposed_moves=158720,
        stages=31,directed_f1=.8,skeleton_f1=.9,representative_shd=1,shd_per_true_edge=.1) for r in (1,8,64)]
    frame=pd.DataFrame(rows)
    assert list(quality.scaling_table(frame).speedup)==[1,8,64]
    frame.loc[1,"host"]="node2"
    with pytest.raises(RuntimeError,match="host"): quality.scaling_table(frame)
    frame.loc[1,"host"]="node1";frame.loc[1,"proposed_moves"]+=1
    with pytest.raises(RuntimeError,match="proposed_moves"): quality.scaling_table(frame)


def write_attempt(root,plan,index,status="completed"):
    task=plan["tasks"][index]
    _atomic_json(root/"attempts"/(task["task_id"]+".json"),dict(task=task,status=status,
        plan_sha256=plan["plan_sha256"],launcher_wall_seconds=.5,reserved_core_hours=.001))


def test_real_workers_truth_report_does_not_require_reference_gate(tmp_path):
    from mpi4py import MPI
    plan=tiny_timing_plan(tmp_path)
    for i in range(2):
        timing_pilot.run(tmp_path,i,MPI.COMM_SELF);write_attempt(tmp_path,plan,i)
    _atomic_json(tmp_path/"bundles/0.json",dict(status="attempted_all",plan_sha256=plan["plan_sha256"]))
    quality.aggregate(tmp_path)
    done=read(tmp_path/"analysis/completion.json")
    assert done["succeeded"]==2 and done["status"]=="complete"
    assert not done["posterior_accuracy_claim_for_large_networks"]
    assert len(list((tmp_path/"analysis/figures").glob("*.pdf")))==7
    metrics=pd.read_csv(tmp_path/"analysis/per_run_metrics.csv")
    assert metrics.exact_pip_rmse.notna().all()
    assert metrics.long_mcmc_agreement_rmse.isna().all()
    with pytest.raises(FileExistsError): quality.aggregate(tmp_path)


def test_partial_is_explicit_and_failed_attempts_are_preserved(tmp_path):
    from mpi4py import MPI
    plan=tiny_timing_plan(tmp_path)
    timing_pilot.run(tmp_path,0,MPI.COMM_SELF);write_attempt(tmp_path,plan,0)
    with pytest.raises(RuntimeError,match="incomplete bundle"): quality.collect(tmp_path)
    _,frame,_,failed,missing,bundles,_=quality.collect(tmp_path,True)
    assert len(frame)==1 and not failed and len(missing)==1 and bundles==[0]
    write_attempt(tmp_path,plan,1,"failed_or_time_limited")
    _atomic_json(tmp_path/"bundles/0.json",dict(status="attempted_all",plan_sha256=plan["plan_sha256"]))
    _,frame,_,failed,missing,bundles,_=quality.collect(tmp_path)
    assert len(frame)==1 and len(failed)==1 and not missing and not bundles


def test_scheduler_supports_no_reference_jobs(tmp_path):
    plan=tiny_timing_plan(tmp_path);plan["references"]=[]
    plan["config"].update(job_prefix="smc-scaling",bundle_hours=8)
    plan.pop("plan_sha256");plan["plan_sha256"]=canonical_sha(plan)
    _atomic_json(tmp_path/"plan.json",plan)
    _atomic_json(tmp_path/"validation/preflight.json",dict(all_passed=True,plan_sha256=plan["plan_sha256"]))
    prep=read(tmp_path/"prepared/tiny/prepared.json");prep["plan_sha256"]=plan["plan_sha256"]
    _atomic_json(tmp_path/"prepared/tiny/prepared.json",prep)
    calls=[]
    def sbatch(args,**kwargs): calls.append(args); return str(100+len(calls))
    result=timing_submit.submit(tmp_path,"launch",sbatch=sbatch)
    assert set(result["jobs"])=={"bundles","report"}
    assert "--time=08:00:00" in calls[0] and "--job-name=smc-scaling-bundles" in calls[0]
    assert "--dependency=afterany:101" in calls[1]
