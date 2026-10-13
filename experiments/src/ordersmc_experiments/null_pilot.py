"""Fixed, two-prior null/signal diagnostic; not a default-selection procedure."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import itertools
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from ordersmc import OrderSMCConfig, build_candidates, sample, validate_candidate_provenance
from ordersmc.persistence import load_candidate_table, save_candidate_table, candidate_table_content_sha256
from . import catchup, reporting_repair
from .execution import _atomic_json
from .exact_order import exact_order_dp
from .prior_calibration import retarget_edge_count_prior
from .rank_budget import complete_probabilities
from .readiness import root_call, summarize_posterior
from .readiness_audits import read, pips, rmse, sha
from .sampler_corrections import artifact, checked_input
from .seeds import protocol_seed_record

MASTER_SEED=20260921


def dimension_penalty(d):
    if isinstance(d,bool) or not isinstance(d,int) or d<2:raise ValueError("dimension must be an integer >=2")
    return math.log(d-1)  # Bernoulli edge odds 1/(d-1), i.e. probability 1/d before constraints.


def fresh_data(replicate, signal, n=222):
    rng=np.random.Generator(np.random.PCG64DXSM(MASTER_SEED+replicate))
    kinds=("con","bin","cat","con","bin","con","cat","con")
    values={};truth=[]
    for j,kind in enumerate(kinds):
        name=f"x{j}";parent=f"x{j-1}"
        predictor=np.zeros(n)
        if signal and j:
            raw=values[parent].astype(float)
            # Population-scale coding; no normalization using graph recovery.
            predictor=raw-(1. if kinds[j-1]=="cat" else .5 if kinds[j-1]=="bin" else 0.)
            truth.append((parent,name))
        if kind=="con":values[name]=1.5*predictor+rng.normal(size=n)
        elif kind=="bin":values[name]=rng.binomial(1,expit(1.5*predictor))
        else:
            prob=softmax(np.column_stack((-1.5*predictor,np.zeros(n),1.5*predictor)),axis=1)
            values[name]=(rng.random(n)[:,None]>np.cumsum(prob,axis=1)).sum(axis=1)
    types={f"x{j}":kind for j,kind in enumerate(kinds)}
    cards={node:2 if kind=="bin" else 3 if kind=="cat" else 0 for node,kind in types.items()}
    return pd.DataFrame(values),types,cards,truth


def prepare(root):
    plan=catchup.checked_source(reporting_repair.SOURCE/"plan.json")
    if root.exists():raise FileExistsError("use a new diagnostic root")
    datasets=[]
    for d in plan["datasets"]:
        if d["cohort"]!="metadata" or d["spec"]["replicate"] not in (0,1,2):continue
        role=d["spec"]["role"];null=d["spec"]["negative_control"]=="empty"
        if role!="m1_matched" and not null:continue
        table=load_candidate_table(checked_input(d["candidate"]),expected_sha256=d["candidate_sha256"])
        datasets.append(dict(key=d["dataset_id"],kind="metadata-null" if null else "metadata-signal",
            replicate=d["spec"]["replicate"],nodes=d["nodes"],truth_edges=d["truth_edges"],
            candidate=d["candidate"],candidate_sha256=d["candidate_sha256"],source_inputs=d["inputs"],mpi_ranks=64))
    if len(datasets)!=6:raise RuntimeError("expected all three null and first three positive metadata replicates")
    for signal in (False,True):
        for rep in range(3):
            data,types,cards,truth=fresh_data(rep,signal)
            key=f"fresh-{'signal' if signal else 'null'}-r{rep}"
            cfg=OrderSMCConfig.protocol_v2(lambda_target=1.,score_method="rank-native",max_parents=2,
                screen_k=7,parent_pool_max=None,pair_rescue_anchor_k=0,pair_rescue_k=0)
            table=build_candidates(data,types,cards,config=cfg,verbose=False)
            folder=root/"data"/key;folder.mkdir(parents=True)
            data.to_csv(folder/"data.csv",index=False)
            path=save_candidate_table(table,folder/"candidate.json.gz")
            datasets.append(dict(key=key,kind="fresh-signal" if signal else "fresh-null",replicate=rep,
                nodes=table.nodes,truth_edges=truth,candidate=artifact(path),
                candidate_sha256=candidate_table_content_sha256(table),source_inputs=[artifact(folder/"data.csv")],mpi_ranks=1))
    cases=[];tasks=[]
    for d in datasets:
        source=load_candidate_table(checked_input(d["candidate"]),expected_sha256=d["candidate_sha256"])
        base=OrderSMCConfig.from_mapping(source.provenance["candidate_build_config"])
        for arm,penalty in (("fixed",1.),("dimension",dimension_penalty(len(d["nodes"])))):
            cfg=base.with_overrides(lam=0.,edge_penalty=penalty,particles=1024 if d["mpi_ranks"]>1 else 512,
                mh_steps=10,mh_steps_resample=25,tempering_path="geometric",adaptive_tempering=True,
                adaptive_cess_target=.9,adaptive_min_beta_increment=1e-8,adaptive_max_stages=2000,
                final_target_snapshot_steps=(0,250))
            table,audit=retarget_edge_count_prior(source,target_label="order",lambda_target=penalty,target_config=cfg)
            validate_candidate_provenance(table,config=cfg,missing_policy="require")
            case_id=d["key"]+"--"+arm;folder=root/"cases"/case_id
            path=save_candidate_table(table,folder/"candidate.json.gz")
            case=dict(case_id=case_id,dataset=d,arm=arm,penalty=penalty,config=cfg.to_dict(),
                candidate=artifact(path),candidate_sha256=candidate_table_content_sha256(table),prior_audit=audit.to_dict())
            if len(d["nodes"])<=8:
                exact=exact_order_dp(table,compute_graph_size=False)
                prior=exact_order_dp(table,beta=0.,compute_graph_size=False)
                ref=folder/"exact.json"
                _atomic_json(ref,dict(edge_probabilities=catchup._all_edges(table.nodes,exact.edge_probs),
                    exact_prior_expected_edges=sum(prior.edge_probs.values())))
                case["exact"]=artifact(ref)
            cases.append(case)
            for seed_index in range(4):
                seed=asdict(protocol_seed_record(MASTER_SEED,experiment="null-prior-pilot",cell=d["key"],
                    replicate=d["replicate"],component=f"inference-{seed_index}"))
                tasks.append(dict(task_id=f"{case_id}--s{seed_index}",case_id=case_id,seed_index=seed_index,
                    seed_record=seed,mpi_ranks=d["mpi_ranks"]))
    reporting_repair.snapshot(root,dict(schema="ordersmc-null-prior-diagnostic-v1",cases=cases,tasks=tasks,
        source_plan=artifact(reporting_repair.SOURCE/"plan.json"),seeds=4,model_default_changed=False,
        selection="all three null and first three signal metadata replicates; three fresh null/signal pairs",
        prior_rule="fixed lambda=1 versus lambda(d)=log(d-1); no recovery-dependent tuning",
        support_policy="fixed candidate support, bit-identical likelihoods; not a rebuilt-screening comparison",
        purpose="development sensitivity; any adoption requires independent validation and matched comparator reruns"))


def preflight(root):
    plan=catchup.checked_plan(root);checked_input(plan["source_plan"])
    for case in plan["cases"]:
        d=case["dataset"];checked_input(d["candidate"])
        for a in d["source_inputs"]:checked_input(a)
        table=load_candidate_table(checked_input(case["candidate"]),expected_sha256=case["candidate_sha256"])
        validate_candidate_provenance(table,config=OrderSMCConfig.from_mapping(case["config"]),missing_policy="require")
        if "exact" in case:checked_input(case["exact"])
        if not case["prior_audit"]["local_likelihood_arrays_bit_identical"]:raise RuntimeError("likelihood changed")
    _atomic_json(root/"validation/preflight.json",dict(all_passed=True,plan_sha256=plan["plan_sha256"]))


def run(root,index,comm=None):
    plan=root_call(comm,lambda:catchup.checked_plan(root));task=plan["tasks"][index]
    if (comm.size if comm is not None else 1)!=task["mpi_ranks"]:raise RuntimeError("MPI layout mismatch")
    acceptance=root_call(comm,lambda:read(root/"validation/preflight.json"))
    if not acceptance["all_passed"] or acceptance["plan_sha256"]!=plan["plan_sha256"]:raise RuntimeError("preflight failed")
    case=next(c for c in plan["cases"] if c["case_id"]==task["case_id"]);d=case["dataset"]
    dest=root/"runs"/task["task_id"]
    def existing():
        if not (dest/"result.json").exists():return False
        record=read(dest/"result.json")
        if record.get("status")!="complete" or record["plan_sha256"]!=plan["plan_sha256"] or record["task"]!=task:
            raise RuntimeError("stale diagnostic fit")
        return True
    if root_call(comm,existing):return
    table=root_call(comm,lambda:load_candidate_table(checked_input(case["candidate"]),expected_sha256=case["candidate_sha256"]))
    cfg=OrderSMCConfig.from_mapping(case["config"]).with_overrides(seed=task["seed_record"]["seed"])
    result=sample(table,config=cfg,backend="mpi" if comm is not None else "local",comm=comm,
        parent_seed_record=task["seed_record"] if comm is not None else None,verbose=False,provenance_policy="require")
    def save():
        local=result.to_local_result() if comm is not None else result
        truth=nx.DiGraph();truth.add_nodes_from(d["nodes"]);truth.add_edges_from(d["truth_edges"])
        exact=pips(read(checked_input(case["exact"]))) if "exact" in case else None
        snapshots=[]
        for s in local.final_target_snapshots:
            probs=complete_probabilities(s.edge_probs,d["nodes"]);row=summarize_posterior(probs,truth,d["nodes"],cfg.max_parents)
            selected=set(map(tuple,row["representative_edges"]));actual=set(truth.edges())
            row.update(final_target_steps=s.final_target_mh_sweeps,selected_edges=len(selected),
                false_edges=len(selected-actual),true_edges=len(selected&actual),
                posterior_expected_edges=sum(probs.values()),own_target_pip_rmse=rmse(probs,exact) if exact else None)
            snapshots.append(row)
        local.save(dest)
        _atomic_json(dest/"result.json",dict(status="complete",plan_sha256=plan["plan_sha256"],task=task,
            candidate_sha256=case["candidate_sha256"],snapshots=snapshots,history=local.history))
        print("complete",task["task_id"],flush=True)
    root_call(comm,save)


def aggregate(root):
    plan=catchup.checked_plan(root);rows=[];hashes={};missing=[];groups=defaultdict(list)
    for task in plan["tasks"]:
        case=next(c for c in plan["cases"] if c["case_id"]==task["case_id"]);d=case["dataset"]
        path=root/"runs"/task["task_id"]/"result.json"
        if not path.exists():missing.append(task["task_id"]);continue
        r=read(path)
        if r.get("status")!="complete" or r["task"]!=task or r["plan_sha256"]!=plan["plan_sha256"] or r["candidate_sha256"]!=case["candidate_sha256"]:
            raise RuntimeError("invalid diagnostic artifact")
        if {s["final_target_steps"] for s in r["snapshots"]}!=set(case["config"]["final_target_snapshot_steps"]):
            raise RuntimeError("missing diagnostic posterior snapshot")
        hashes[task["task_id"]]=sha(path)
        for s in r["snapshots"]:
            groups[(case["case_id"],s["final_target_steps"])].append(s)
            rows.append(dict(dataset_key=d["key"],kind=d["kind"],replicate=d["replicate"],arm=case["arm"],
                penalty=case["penalty"],seed_index=task["seed_index"],**{k:v for k,v in s.items()
                if k not in {"representative_edges","edge_probabilities","representative_graph_diagnostics"}}))
    if missing:
        _atomic_json(root/"analysis/completion.json",dict(status="incomplete",missing=missing))
        raise RuntimeError("diagnostic incomplete; no partial scientific comparison")
    frame=pd.DataFrame(rows);out=root/"analysis";out.mkdir(exist_ok=True)
    frame.to_csv(out/"per_run_metrics.csv",index=False)
    metrics=["directed_f1","skeleton_f1","directed_auprc","false_edges","true_edges","selected_edges",
             "posterior_expected_edges","own_target_pip_rmse"]
    means=frame.groupby(["dataset_key","kind","arm","final_target_steps"],as_index=False)[metrics].mean()
    means.to_csv(out/"dataset_metrics.csv",index=False)
    means.groupby(["kind","arm","final_target_steps"],as_index=False)[metrics].mean().to_csv(out/"summary.csv",index=False)
    stability=[]
    for (case_id,steps),snapshots in groups.items():
        case=next(c for c in plan["cases"] if c["case_id"]==case_id)
        edges=[set(map(tuple,s["representative_edges"])) for s in snapshots]
        maps=[pips(s) for s in snapshots];pairs=list(itertools.combinations(range(len(maps)),2))
        stability.append(dict(case_id=case_id,dataset_key=case["dataset"]["key"],kind=case["dataset"]["kind"],
            arm=case["arm"],final_target_steps=steps,seeds=len(snapshots),
            mean_directed_jaccard=float(np.mean([len(edges[a]&edges[b])/len(edges[a]|edges[b])
                if edges[a]|edges[b] else 1. for a,b in pairs])) if pairs else None,
            mean_pairwise_pip_rmse=float(np.mean([rmse(maps[a],maps[b]) for a,b in pairs])) if pairs else None))
    pd.DataFrame(stability).to_csv(out/"seed_stability.csv",index=False)
    _atomic_json(out/"artifact_hashes.json",hashes)
    _atomic_json(out/"completion.json",dict(status="complete",plan_sha256=plan["plan_sha256"],
        completed=len(plan["tasks"]),automatic_default_promotion=False,confirmatory=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","preflight","run","aggregate"])
    p.add_argument("--output",type=Path,required=True);p.add_argument("--task",type=int);p.add_argument("--mpi",action="store_true")
    a=p.parse_args()
    if a.action=="run":
        comm=None
        if a.mpi:
            from mpi4py import MPI
            comm=MPI.COMM_WORLD
        run(a.output,a.task,comm)
    else:globals()[a.action](a.output)


if __name__=="__main__":main()
