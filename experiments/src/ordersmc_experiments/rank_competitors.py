"""Matched competitors on immutable rank-native targets; no data regeneration."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import itertools
from pathlib import Path
import shutil
import time

import networkx as nx
import numpy as np
import pandas as pd

from ordersmc import validate_candidate_provenance
from ordersmc.persistence import load_candidate_table
from .adapter_validation import validate_algorithm_adapters
from .bnlearn_adapter import run_bnlearn, validation_report
from .exact_order import exact_order_dp
from .exact_partition_log import exact_partition_log_dp
from .execution import _atomic_json
from .order_mcmc import OrderMCMCConfig, order_mcmc
from .structure_mcmc import PartitionMCMCConfig, REVMCMCConfig, partition_mcmc, rev_mcmc
from .rank_budget import complete_probabilities
from .rank_validation import load_suite, stability
from .readiness import canonical_sha, score_config, summarize_posterior
from .readiness_audits import read, sha, pips, rmse
from .seeds import protocol_seed_record

REPO = Path(__file__).resolve().parents[3]
MCMC = ("order-mcmc", "rev-mcmc", "partition-mcmc")
METRICS = ("directed_f1", "skeleton_f1", "directed_auprc", "directed_brier",
           "skeleton_auprc", "skeleton_brier", "posterior_expected_pair_state_shd",
           "own_target_pip_rmse")


def checked_plan(root, sources=True, source_root=None):
    plan = read(root / "plan.json")
    if canonical_sha({k:v for k,v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise RuntimeError("competitor plan hash mismatch")
    if sources:
        for path, digest in plan["source_files"].items():
            if sha((REPO if source_root is None else Path(source_root)) / path) != digest:
                raise RuntimeError(f"competitor source changed: {path}")
    return plan


def task_list(datasets, config):
    tasks = []
    for d in datasets:
        for method in config["methods"]:
            budgets = ["baseline", "extended"] if method in MCMC and d["n_train"] == config["extended_only_n"] else ["baseline"]
            for budget in budgets:
                count = d["seed_count"] if method in MCMC else 1
                for start in range(0, count, config["seeds_per_task"]):
                    tasks.append(dict(dataset_id=d["dataset_id"], method=method, budget=budget,
                        phase="extended" if budget == "extended" else "primary",
                        seed_indices=list(range(start, min(start + config["seeds_per_task"], count)))))
    return tasks


def freeze(source, budget_source, output, config_path, smoke=False):
    if output.exists():
        raise FileExistsError("use a new competitor root; frozen inputs are never overwritten")
    config = read(config_path)
    suite = load_suite(source)
    source_completion = read(source / "analysis/completion.json")
    budget_plan = read(budget_source / "plan.json")
    budget_completion = read(budget_source / "analysis/completion.json")
    if source_completion["status"] != "complete" or source_completion["missing"]:
        raise RuntimeError("incomplete rank-validation source")
    if suite["suite_sha256"] != config["source_suite_sha256"]:
        raise RuntimeError("rank source is not the accepted cohort")
    if (canonical_sha({k:v for k,v in budget_plan.items() if k != "plan_sha256"}) != budget_plan["plan_sha256"]
            or budget_plan["plan_sha256"] != config["source_budget_plan_sha256"]
            or budget_completion["plan_sha256"] != budget_plan["plan_sha256"]
            or budget_completion["status"] != "complete" or budget_completion["missing"]):
        raise RuntimeError("budget source is not the completed accepted cohort")
    if smoke:
        config.update(noise_families=["gaussian"], replicates=[0], sample_sizes=[250,5000],
                      seeds_low_n=2, seeds_high_n=2, seeds_per_task=2,
                      status="implementation-smoke-not-scientific-results")
        config["mcmc_budgets_seconds"] = dict(baseline=1, extended=2)
        config["mcmc"].update(order_thin=1, structure_thin=1)
    artifacts = read(source / "analysis/artifact_hashes.json")
    budget_hashes = read(budget_source / "analysis/run_hashes.json")
    pending, datasets = [], []
    for noise, rep, n in itertools.product(config["noise_families"], config["replicates"], config["sample_sizes"]):
        folder = source / f"fresh-{noise}-rank-native"
        arm = read(folder / "plan.json")
        for relative, expected in arm["source_files"].items():
            if relative.startswith("src/ordersmc/") and sha(REPO / relative) != expected:
                raise RuntimeError("core model changed since accepted SMC fits")
        path = folder / "cases" / f"rep{rep:03d}" / f"unrestricted_native-n{n:05d}.json"
        case = read(path)
        candidate = path.with_name(path.stem + "-candidate.json.gz")
        paired = path.with_name(path.name.replace("unrestricted_native", "paired_nongaussian"))
        paired_candidate = candidate.with_name(candidate.name.replace("unrestricted_native", "paired_nongaussian"))
        csv = path.parent / "unrestricted_native.csv"
        for p in (path,candidate,paired,paired_candidate,csv):
            if sha(p) != artifacts[str(p.relative_to(source))]:
                raise RuntimeError(f"changed source artifact: {p}")
        if case["status"] != "complete" or case["plan_sha256"] != arm["plan_sha256"]:
            raise RuntimeError("invalid source case")
        table = load_candidate_table(candidate, expected_sha256=case["candidate_sha256"])
        pair_table = load_candidate_table(paired_candidate, expected_sha256=read(paired)["candidate_sha256"])
        if table.nodes != pair_table.nodes or any(not np.array_equal(a,b)
                for field in ("parent_sets", "loglik", "logprior")
                for a,b in zip(getattr(table,field),getattr(pair_table,field))):
            raise RuntimeError("paired transformed target is not identical")
        validate_candidate_provenance(table, config=score_config(arm["config"]), missing_policy="require")
        for key, function in [("exact-order", exact_order_dp), ("exact-structure", exact_partition_log_dp)]:
            oracle = function(table, compute_graph_size=False) if key == "exact-order" else function(table)
            if rmse(oracle.edge_probs, pips(case["references"][key])) > 1e-8:
                raise RuntimeError(f"exact reference mismatch: {key}")
        dataset_id = f"{noise.replace('_','-')}-rep{rep:03d}-n{n:05d}"
        count = config["seeds_high_n"] if n == config["extended_only_n"] else config["seeds_low_n"]
        d = dict(dataset_id=dataset_id, noise=noise, replicate=rep, n_train=n,
                 seed_count=count, candidate_sha256=case["candidate_sha256"], dimension=len(table.nodes),
                 max_indegree=arm["config"]["maximum_indegree"], score="rank-native",
                 paired_variants=["unrestricted_native", "paired_nongaussian"],
                 source_case=str(path.relative_to(source)))
        datasets.append(d)
        for src,name in [(path,"reference.json"),(candidate,"candidate.json.gz"),(csv,"data.csv")]:
            pending.append((src,f"datasets/{dataset_id}/{name}"))
        if n == config["extended_only_n"]:
            old_id = f"{noise.replace('_','-')}-rep{rep:03d}"
            old = next(x for x in budget_plan["datasets"] if x["dataset_id"] == old_id)
            if old["candidate_sha256"] != d["candidate_sha256"]:
                raise RuntimeError("budget SMC used a different candidate target")
            for budget,index in itertools.product(["baseline","extended"],range(count)):
                record_path=budget_source/"runs"/f"{old_id}-{budget}-s{index:02d}.json"
                if sha(record_path) != budget_hashes[str(record_path.relative_to(budget_source))]:
                    raise RuntimeError("changed completed SMC budget fit")
                pending.append((record_path,f"datasets/{dataset_id}/smc-{budget}-s{index:02d}.json"))
    folders=("src", "experiments/src", "experiments/adapters", "experiments/configs", "scripts")
    source_files={str(p.relative_to(REPO)):sha(p) for folder in folders
                  for p in sorted((REPO/folder).rglob('*')) if p.is_file() and "__pycache__" not in p.parts}
    plan=dict(schema="ordersmc-rank-competitor-plan-v1",config=config,datasets=datasets,
              source_root=str(source.resolve()),budget_source_root=str(budget_source.resolve()),
              tasks=task_list(datasets,config),source_files=source_files,
              input_files={dst:sha(src) for src,dst in pending})
    plan["plan_sha256"]=canonical_sha(plan)
    for src,relative in pending:
        destination=output/relative; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,destination)
    for relative,digest in source_files.items():
        destination=output/"code_snapshot"/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,destination)
        if sha(destination)!=digest: raise RuntimeError("source changed during freeze")
    _atomic_json(output/"plan.json",plan)
    return plan


def seed_record(plan,dataset,method,index):
    return protocol_seed_record(plan["config"]["master_seed"],experiment="rank-competitors",
        cell=f"{dataset['noise'].replace('_','-')}-n{dataset['n_train']:05d}",
        replicate=dataset["replicate"],component=f"{method}-{index:02d}")


def fit_path(root,dataset,method,budget,index):
    return root/"runs"/f"{dataset['dataset_id']}-{method}-{budget}-s{index:02d}.json"


def fit_identity(plan,dataset,method,budget,index):
    return dict(plan_sha256=plan["plan_sha256"],dataset_id=dataset["dataset_id"],method=method,
                budget=budget,seed_index=index,seed=asdict(seed_record(plan,dataset,method,index)),
                candidate_sha256=dataset["candidate_sha256"])


def load_dataset(root,plan,dataset):
    folder=root/"datasets"/dataset["dataset_id"]
    for name in ("reference.json","candidate.json.gz","data.csv"):
        path=folder/name
        if sha(path)!=plan["input_files"][str(path.relative_to(root))]:
            raise RuntimeError(f"frozen input changed: {path}")
    return load_candidate_table(folder/"candidate.json.gz",expected_sha256=dataset["candidate_sha256"]),read(folder/"reference.json")


def preflight(root):
    plan=checked_plan(root)
    report=dict(plan_sha256=plan["plan_sha256"],adapters=validate_algorithm_adapters(),bnlearn=validation_report())
    report["all_passed"]=report["adapters"]["all_passed"] and report["bnlearn"]["all_passed"]
    _atomic_json(root/"validation/preflight.json",report)
    if not report["all_passed"]: raise RuntimeError("competitor adapter preflight failed")


def run_task(root,index):
    plan=checked_plan(root)
    acceptance=read(root/"validation/preflight.json")
    if not acceptance["all_passed"] or acceptance["plan_sha256"]!=plan["plan_sha256"]:
        raise RuntimeError("competitor preflight missing/failed/stale")
    task=plan["tasks"][index]
    dataset=next(d for d in plan["datasets"] if d["dataset_id"]==task["dataset_id"])
    table,case=load_dataset(root,plan,dataset)
    method,budget=task["method"],task["budget"]
    truth=nx.DiGraph();truth.add_nodes_from(table.nodes);truth.add_edges_from(case["truth_edges"])
    failures=[]
    for index in task["seed_indices"]:
        destination=fit_path(root,dataset,method,budget,index)
        identity=fit_identity(plan,dataset,method,budget,index)
        if destination.exists():
            old=read(destination)
            if canonical_sha({k:old.get(k) for k in identity})!=canonical_sha(identity):
                raise RuntimeError("existing competitor run has different identity")
            if old["status"]!="complete": failures.append(str(destination))
            continue
        started=time.perf_counter()
        try:
            seed=seed_record(plan,dataset,method,index).seed
            if method in MCMC:
                settings=plan["config"]["mcmc"]
                options={k:settings[k] for k in ("iterations","burnin","wall_burnin_fraction")}
                options.update(seed=seed,wall_seconds=plan["config"]["mcmc_budgets_seconds"][budget],
                    thin=settings["order_thin" if method=="order-mcmc" else "structure_thin"])
                config_type,algorithm={"order-mcmc":(OrderMCMCConfig,order_mcmc),
                    "rev-mcmc":(REVMCMCConfig,rev_mcmc),
                    "partition-mcmc":(PartitionMCMCConfig,partition_mcmc)}[method]
                cfg=config_type(**options); result=algorithm(table,cfg)
                probs=complete_probabilities(result.edge_probs,table.nodes)
                target="exact-order" if method=="order-mcmc" else "exact-structure"
                diagnostics={k:getattr(result,k) for k in ("acceptance_rate","acceptance_by_proposal",
                    "retained_samples","stored_samples","effective_sample_size","completed_iterations")}
                diagnostics.update(sampler_config=asdict(cfg),target=target)
                error=rmse(probs,pips(case["references"][target]))
            else:
                data=pd.read_csv(root/"datasets"/dataset["dataset_id"]/"data.csv").iloc[:dataset["n_train"]]
                result=run_bnlearn(data,case["variable_types"],table,
                    algorithm="hc" if method=="bnlearn-hc-shared-score" else "tabu",
                    scoring_mode="shared-mixed-score",seed=seed)
                graph=nx.DiGraph();graph.add_nodes_from(table.nodes);graph.add_edges_from(result.edges)
                if not nx.is_directed_acyclic_graph(graph): raise RuntimeError("baseline graph is cyclic")
                probs=complete_probabilities({e:1. for e in result.edges},table.nodes)
                diagnostics=asdict(result);diagnostics["target"]="point-estimate-shared-structure-score"
                error=None
            summary=summarize_posterior(probs,truth,table.nodes,dataset["max_indegree"])
            record=dict(schema="ordersmc-rank-competitor-run-v1",status="complete",**identity,
                cores=1,diagnostics=diagnostics,fit_wall_seconds=time.perf_counter()-started,
                own_target_pip_rmse=error,**summary)
            if method not in MCMC and set(map(tuple,summary["representative_edges"]))!=set(result.edges):
                raise RuntimeError("point graph changed in summary construction")
        except Exception as error:
            record=dict(schema="ordersmc-rank-competitor-run-v1",status="failed",**identity,
                failure_type=type(error).__name__,failure_message=str(error),fit_wall_seconds=time.perf_counter()-started)
            failures.append(str(destination))
        _atomic_json(destination,record)
        print(f"{record['status']}: {destination.name}",flush=True)
    if failures: raise RuntimeError(f"{len(failures)} competitor fits failed; all saved, none dropped")


def smc_records(root,plan,dataset,case,budget):
    if dataset["n_train"]!=plan["config"]["extended_only_n"]:
        return case["smc_runs"][:dataset["seed_count"]]
    records=[]
    for index in range(dataset["seed_count"]):
        path=root/"datasets"/dataset["dataset_id"]/f"smc-{budget}-s{index:02d}.json"
        if sha(path)!=plan["input_files"][str(path.relative_to(root))]: raise RuntimeError("reused SMC fit changed")
        records.append(read(path))
    return records


def reporting_records(records, nodes):
    """Expand the sampler's omitted zeros without modifying saved fit records.

    Use all d(d-1) edges for mean PIPs AND between-seed RMSE. Taking just the
    first seed's support both loses other seeds' edges and changes denominators.
    """
    return [{**record, "edge_probabilities": [
        dict(parent=u, child=v, probability=p)
        for (u,v),p in complete_probabilities(pips(record), nodes).items()
    ]} for record in records]


def aggregate(root, *, source_root=None):
    plan=checked_plan(root, source_root=source_root);analysis=root/"analysis";analysis.mkdir(exist_ok=True)
    rows,stable,pooled,missing,failed,hashes=[],[],[],[],[],{}
    for dataset in plan["datasets"]:
        table,case=load_dataset(root,plan,dataset)
        common={k:dataset[k] for k in ("dataset_id","noise","replicate","n_train")}
        groups={}
        for task in [t for t in plan["tasks"] if t["dataset_id"]==dataset["dataset_id"]]:
            key=task["method"],task["budget"]
            groups.setdefault(key,[])
            for index in task["seed_indices"]:
                path=fit_path(root,dataset,*key,index)
                if not path.exists(): missing.append(str(path.relative_to(root)));continue
                record=read(path);identity=fit_identity(plan,dataset,*key,index)
                if canonical_sha({k:record.get(k) for k in identity})!=canonical_sha(identity):
                    raise RuntimeError("competitor aggregation identity mismatch")
                hashes[str(path.relative_to(root))]=sha(path)
                if record["status"]!="complete":
                    failed.append(dict(path=str(path.relative_to(root)),reason=record.get("failure_message")));continue
                probabilities=complete_probabilities(pips(record),table.nodes)
                if len(probabilities)!=len(pips(record)): raise RuntimeError("incomplete edge inventory")
                if key[0] in MCMC:
                    target="exact-order" if key[0]=="order-mcmc" else "exact-structure"
                    if not np.isclose(record["own_target_pip_rmse"],rmse(probabilities,pips(case["references"][target])),atol=1e-12,rtol=0):
                        raise RuntimeError("saved exact-target error differs")
                groups[key].append(record)
        for budget in (["baseline","extended"] if dataset["n_train"]==plan["config"]["extended_only_n"] else ["baseline"]):
            groups["ordersmc",budget]=smc_records(root,plan,dataset,case,budget)
        for (method,budget),records in groups.items():
            records=reporting_records(records, table.nodes)
            expected=1 if method.startswith("bnlearn") else dataset["seed_count"]
            for index,r in enumerate(records):
                diagnostics=r.get("diagnostics",{})
                rows.append(dict(**common,method=method,budget=budget,seed_index=r.get("seed_index",index),
                    reused=method=="ordersmc",**{k:r.get(k) for k in METRICS},
                    acceptance_rate=diagnostics.get("acceptance_rate"),effective_sample_size=diagnostics.get("effective_sample_size"),
                    stored_samples=diagnostics.get("stored_samples"),group_complete=len(records)==expected))
            if len(records)!=expected: continue
            if expected>1:
                counts=Counter(tuple(sorted(map(tuple,r["representative_edges"]))) for r in records)
                errors=np.asarray([r["own_target_pip_rmse"] for r in records])
                stable.append(dict(**common,method=method,budget=budget,seed_count=len(records),
                    unique_directed_graphs=len(counts),modal_graph_fraction=max(counts.values())/len(records),
                    mean_own_target_pip_rmse=errors.mean(),p95_own_target_pip_rmse=np.quantile(errors,.95),
                    sampling_flag=bool(errors.mean()>plan["config"]["rmse_flags"]["mean"] or np.quantile(errors,.95)>plan["config"]["rmse_flags"]["p95"]),
                    **stability({**case,"smc_runs":records},plan["config"]["orientation_display_threshold"])))
            truth=nx.DiGraph();truth.add_nodes_from(table.nodes);truth.add_edges_from(case["truth_edges"])
            means={e:float(np.mean([pips(r)[e] for r in records])) for e in pips(records[0])}
            summary=summarize_posterior(means,truth,table.nodes,dataset["max_indegree"])
            pooled.append(dict(**common,method=method,budget=budget,seed_count=expected,
                               **{k:summary.get(k) for k in METRICS if k!="own_target_pip_rmse"}))
    frame=pd.DataFrame(rows)
    frame.to_csv(analysis/"runs.csv",index=False)
    pd.DataFrame(stable).to_csv(analysis/"stability.csv",index=False)
    pd.DataFrame(pooled).to_csv(analysis/"mean_pip_graph_metrics.csv",index=False)
    if len(frame):
        complete=frame[frame.group_complete]
        means=complete.groupby(["dataset_id","noise","replicate","n_train","method","budget"],as_index=False)[list(METRICS)].mean()
        means.to_csv(analysis/"dataset_means.csv",index=False)
        means.groupby(["noise","n_train","method","budget"],as_index=False).agg(
            datasets=("dataset_id","nunique"),**{k:(k,"mean") for k in METRICS}).to_csv(analysis/"summary.csv",index=False)
        comparisons=[]
        for (noise,n,budget),group in means.groupby(["noise","n_train","budget"]):
            smc=group[group.method=="ordersmc"].set_index("dataset_id")
            for method,competitor in group[group.method!="ordersmc"].groupby("method"):
                competitor=competitor.set_index("dataset_id")
                if set(smc.index)!=set(competitor.index): continue
                for metric in METRICS:
                    if metric=="own_target_pip_rmse": continue  # different targets/point graphs
                    delta=(smc[metric]-competitor[metric]).dropna().to_numpy()
                    if not len(delta): continue
                    rng=np.random.Generator(np.random.PCG64DXSM(20260925))
                    boots=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
                    comparisons.append(dict(noise=noise,n_train=n,budget=budget,competitor=method,
                        metric=metric,datasets=len(delta),mean_smc_minus_competitor=delta.mean(),
                        ci_lower=np.quantile(boots,.025),ci_upper=np.quantile(boots,.975)))
        pd.DataFrame(comparisons).to_csv(analysis/"paired_comparisons.csv",index=False)
    _atomic_json(analysis/"run_hashes.json",hashes)
    result=dict(status="complete" if not missing and not failed else "incomplete",plan_sha256=plan["plan_sha256"],
        planned_new_fits=sum(len(t["seed_indices"]) for t in plan["tasks"]),
        completed_new_fits=sum(r["method"]!="ordersmc" for r in rows),reused_smc_fits=sum(r["method"]=="ordersmc" for r in rows),
        missing=missing,failed=failed,summary_policy="partial groups explicitly flagged; compare only matched complete groups; exploratory, not confirmation")
    _atomic_json(analysis/"completion.json",result)
    print(result["status"],result["completed_new_fits"],"new fits",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["freeze","preflight","run","aggregate"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source",type=Path,default=REPO/"results/rank-validation/20260917-v1")
    p.add_argument("--budget-source",type=Path,default=REPO/"results/rank-budget/20260917-pilot-v1")
    p.add_argument("--config",type=Path,default=REPO/"experiments/configs/rank_competitors_v1.json")
    p.add_argument("--source-root",type=Path,help="Original frozen inference source to verify during reporting only")
    p.add_argument("--task",type=int,default=0);p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.action=="freeze": print(freeze(a.source,a.budget_source,a.output,a.config,a.smoke)["plan_sha256"])
    elif a.action=="preflight": preflight(a.output)
    elif a.action=="run": run_task(a.output,a.task)
    else:
        result=aggregate(a.output, source_root=a.source_root)
        if result["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()
