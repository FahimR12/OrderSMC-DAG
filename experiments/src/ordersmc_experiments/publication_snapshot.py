"""Build a dated, auditable presentation over corrected results; no inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .execution import _atomic_json
from .readiness_audits import read
from .scaling_quality import digest
from .timing_report import reference_check

REPO=Path(__file__).resolve().parents[3]
LABELS={"ordersmc-v07":"OrderSMC v0.7", "order-mcmc":"Order MCMC",
        "bnlearn-hc-shared-score":"bnlearn HC (shared)","bnlearn-tabu-shared-score":"bnlearn Tabu (shared)",
        "bnlearn-hc-bic-cg":"bnlearn HC (native CG)","bnlearn-tabu-bic-cg":"bnlearn Tabu (native CG)",
        "partition-mcmc":"Partition MCMC", "rev-mcmc":"REV MCMC",
        "adaptive_original_budget":"Adaptive SMC: original effort",
        "adaptive_matched_effort":"Adaptive SMC: matched effort", "legacy":"Legacy SMC"}


def plot_ranges(center, low, high):
    """Clip only ulp-sized negative plotting lengths, never alter observations."""
    values=np.array([np.asarray(center)-np.asarray(low),np.asarray(high)-np.asarray(center)])
    if np.min(values)<-1e-12:raise ValueError("center outside reported range")
    return np.maximum(values,0.)


def build(output):
    output=output.resolve()
    if (output/"presentation.pdf").exists(): raise FileExistsError("use a new dated output folder")
    for directory in ("figures","tables","sources","reproduce"):
        (output/directory).mkdir(parents=True,exist_ok=True)
    inputs=[]
    def copy(path,relative):
        path=Path(path); destination=output/"sources"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        value=digest(path)
        if digest(destination)!=value:raise RuntimeError("copy changed")
        inputs.append(dict(source=str(path.resolve()),copy=str(destination.relative_to(output)),sha256=value))
        return destination
    def table(path,relative):return pd.read_csv(copy(path,relative))
    def complete(path,relative):
        value=read(copy(path,relative))
        if value["status"]!="complete":raise RuntimeError(f"source not complete: {path}")
        return value
    repaired=REPO/"results/reporting-null-audit/v1/reporting_repair"
    complete(repaired/"analysis/completion.json","corrected/completion.json")
    copy(repaired/"plan.json","corrected/plan.json")
    recovery,stability={},{}
    for cohort in ("v1","v3_mixed","v3_clg","metadata","rank_budget","rank_validation"):
        folder=repaired/"analysis"/cohort
        complete(folder/"completion.json",f"corrected/{cohort}/completion.json")
        for filename in ("mean_pip_graph_metrics.csv","seed_stability.csv","dataset_mean_metrics.csv",
                         "cell_summary.csv","exact_rmse_flags.json","interpretation.json","reporting_edition.json",
                         "artifact_hashes.json"):
            copy(folder/filename,f"corrected/{cohort}/{filename}")
        recovery[cohort]=pd.read_csv(folder/"mean_pip_graph_metrics.csv")
        stability[cohort]=pd.read_csv(folder/"seed_stability.csv")
    method_order=["ordersmc-v07","order-mcmc","bnlearn-tabu-shared-score","bnlearn-hc-shared-score",
                  "rev-mcmc","partition-mcmc","bnlearn-tabu-bic-cg","bnlearn-hc-bic-cg"]
    rows=[];stable=[]
    for cohort in ("v3_mixed","v3_clg","metadata"):
        f=recovery[cohort];s=stability[cohort]
        f=f[(f.final_target_steps==0)&f.method.isin(method_order)]
        s=s[(s.final_target_steps==0)&s.method.isin(method_order)]
        if cohort=="metadata":
            f=f[f.cell_id.str.startswith("m1_matched")];s=s[s.cell_id.str.startswith("m1_matched")]
        expected={"v3_mixed":220,"v3_clg":60,"metadata":5}[cohort]
        for method,group in f.groupby("method"):
            if group.dataset_key.nunique()!=expected or len(group)!=expected:raise RuntimeError("unequal recovery cohort")
            rows.append(dict(cohort=cohort,method=method,datasets=expected,
                graph_summary="Bayes-action graph of mean evaluation-seed PIPs",
                mean_directed_f1=group.directed_f1.mean(),mean_skeleton_f1=group.skeleton_f1.mean(),
                mean_directed_auprc=group.directed_auprc.mean(),
                mean_expected_pair_state_shd=group.posterior_expected_pair_state_shd.mean()))
        for method,group in s.groupby("method"):
            if group.dataset_key.nunique()!=expected:raise RuntimeError("unequal stability cohort")
            stable.append(dict(cohort=cohort,method=method,datasets=expected,seeds=int(group.seeds.min()),
                mean_directed_jaccard=group.mean_directed_jaccard.mean(),
                mean_skeleton_jaccard=group.mean_skeleton_jaccard.mean(),
                mean_pairwise_pip_rmse=group.mean_pairwise_pip_rmse.mean()))
    recovery_summary=pd.DataFrame(rows);stability_summary=pd.DataFrame(stable)
    recovery_summary.to_csv(output/"tables/cohort_recovery.csv",index=False)
    stability_summary.to_csv(output/"tables/cohort_reproducibility.csv",index=False)
    effort=REPO/"results/sampler-corrections/v07-v1-effort-match-v1/analysis"
    complete(effort/"completion.json","exact_validation/completion.json")
    for fn in ("dataset_metrics.csv","cell_summary.csv","seed_stability.csv","exact_rmse_flags.json","paired_dataset_contrasts.csv"):
        copy(effort/fn,f"exact_validation/{fn}")
    exact=pd.read_csv(effort/"dataset_metrics.csv");gate=read(effort/"exact_rmse_flags.json")
    exact_rows=[]
    for method,group in exact.groupby("method"):
        flags=gate["arms"][method]["flags"]
        exact_rows.append(dict(method=method,datasets=group.dataset_key.nunique(),mean_pip_rmse=group.own_target_pip_rmse.mean(),
            cells_passed=sum(f["passed"] for f in flags),cells=len(flags),worst_cell_mean=max(f["mean"] for f in flags),
            worst_cell_dataset_p95=max(f["p95"] for f in flags)))
    pd.DataFrame(exact_rows).to_csv(output/"tables/exact_validation.csv",index=False)
    prior=REPO/"results/prior-holdout/v1/fits/analysis"
    complete(prior/"completion.json","prior_holdout/completion.json")
    for fn in ("dataset_metrics.csv","seed_stability.csv","summary.csv"):
        copy(prior/fn,f"prior_holdout/{fn}")
    prior_data=pd.read_csv(prior/"dataset_metrics.csv")
    prior_summary=pd.read_csv(prior/"summary.csv")
    prior_summary.to_csv(output/"tables/prior_holdout.csv",index=False)
    orientation=REPO/"results/orientation-audit/v1/analysis"
    complete(orientation/"completion.json","orientation_audit/completion.json")
    for fn in ("summary.csv","dataset_metrics.csv","edge_directions.csv"):
        copy(orientation/fn,f"orientation_audit/{fn}")
    for path,name in ((REPO/"results/sampler-corrections/v07-catchup-v1/README.md","corrected_protocol.md"),
                      (effort.parent/"README.md","exact_protocol.md"),
                      (REPO/"results/prior-holdout/v1/README.md","prior_protocol_historical_status.md"),
                      (REPO/"docs/SCALING_QUALITY.md","scaling_protocol.md"),
                      (REPO/"results/orientation-audit/v1/README.md","orientation_audit_historical_summary.md")):
        copy(path,f"protocols/{name}")
    # Retain approximate-reference failures, independently of truth recovery.
    pilot=REPO/"results/timing-pilot/v2";plan=read(pilot/"plan.json");reference_rows=[]
    for d in plan["datasets"]:
        if len(d["nodes"])<=12:continue
        for arm in ("dimension","fixed"):
            records=[]
            for t in plan["references"]:
                if (t["dataset_key"],t["arm"])!=(d["key"],arm):continue
                path=pilot/"references"/t["task_id"]/"result.json";r=read(path)
                if r["status"]!="complete" or r["task"]!=t or r["plan_sha256"]!=plan["plan_sha256"]:raise RuntimeError("invalid reference")
                records.append(r);inputs.append(dict(source=str(path),sha256=digest(path),copy=None))
            _,status=reference_check(records,plan["config"],d["nodes"])
            reference_rows.append(dict(dataset_key=d["key"],d=len(d["nodes"]),arm=arm,**status))
    _atomic_json(output/"tables/mcmc_reference_checks.json",dict(checks=reference_rows))
    timing=[];inventories=[]
    for folder,title,particles in (("completed_128_particle_pilot","128-particle pilot",128),
                                    ("partial_512_particle_followup","512-particle follow-up",512)):
        path=output/"timing"/folder;done=read(path/"completion.json")
        inventories.append(dict(experiment=title,**done))
        scaling=pd.read_csv(path/"scaling.csv");scaling["edition"]=title
        timing.append(scaling)
    scaling=pd.concat(timing,ignore_index=True)
    scale_summary=scaling.groupby(["edition","d","particles","ranks"],as_index=False).agg(
        repetitions=("seed_index","size"),median_seconds=("inference_seconds","median"),
        median_speedup=("speedup","median"),minimum_speedup=("speedup","min"),maximum_speedup=("speedup","max"))
    scale_summary.to_csv(output/"tables/scaling_headlines.csv",index=False)
    _atomic_json(output/"tables/timing_inventory.json",inventories)
    queue={}
    commands={"queue":["squeue","-u","psedrou","-o","%.22i %.32j %.3t %.10M %.35R"],
              "pending_detail":["scontrol","show","job","10662452_5"],
              "start_estimate":["squeue","--start","-j","10662452"],
              "node_states":["sinfo","-p","nodes","-o","%P %D %t %C"]}
    for name,cmd in commands.items():
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=20,check=False)
            queue[name]=dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except (OSError,subprocess.TimeoutExpired) as exc:queue[name]=dict(error=str(exc))
    _atomic_json(output/"queue_snapshot.json",dict(captured_utc=datetime.now(timezone.utc).isoformat(),checks=queue,jobs_changed=False))
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    with PdfPages(output/"presentation.pdf") as pdf:
        def save(fig,name):
            fig.tight_layout(rect=(0,.065,1,.91))
            for suffix in ("pdf","png"):fig.savefig(output/"figures"/f"{name}.{suffix}",dpi=180)
            pdf.savefig(fig);plt.close(fig)
        def text_slide(title,paragraphs):
            fig=plt.figure(figsize=(13.3,7.5));fig.text(.06,.9,title,fontsize=23,weight="bold")
            y=.79
            for heading,body in paragraphs:
                fig.text(.06,y,heading,fontsize=15,weight="bold");y-=.047
                wrapped=textwrap.fill(body,width=113)
                fig.text(.06,y,wrapped,fontsize=12,va="top",linespacing=1.5)
                y-=.031*(wrapped.count("\n")+1)+.06
            pdf.savefig(fig);plt.close(fig)
        text_slide("OrderSMC: evidence available on 25 September 2026",[
            ("Main computational question","How much does fixed-work SMC accelerate with more cores, and when does it achieve recovery comparable to MCMC sooner?"),
            ("Completed evidence","Corrected mixed benchmark (220 datasets), restricted mixed/CLG benchmark (60), exact validation (90), metadata M1 synthetic pilot (5), and independent prior check (3 null + 3 signal)."),
            ("Timing status","Older pilot: 177 successful runs + 3 serial timeouts, all attempts accounted for. New 512-particle follow-up: 135/243 successful; 108 remain queued. No 128/256-core experiment has run."),
            ("Interpretation","Corrected development/validation evidence, not a new untouched confirmatory study. No participant-level living-brain analysis. Larger-network MCMC reference checks still fail.")])
        fig,axes=plt.subplots(1,3,figsize=(13.3,6.5))
        for ax,d in zip(axes,(12,56,196)):
            for edition,color in (("128-particle pilot","#1768ac"),("512-particle follow-up","#d46b08")):
                g=scale_summary[(scale_summary.d==d)&(scale_summary.edition==edition)].sort_values("ranks")
                if g.empty:continue
                n=int(g.repetitions.min());label=f"{edition} (n={n} seeds)"
                ax.errorbar(g.ranks.to_numpy(),g.median_speedup.to_numpy(),
                    yerr=np.array([g.median_speedup-g.minimum_speedup,g.maximum_speedup-g.median_speedup]),
                    marker="o",capsize=3,color=color,label=label)
            ax.plot([1,8,32,64],[1,8,32,64],":",color="grey",label="Ideal")
            ax.set(title=f"{d} variables",xlabel="Active cores",ylabel="Speedup over one core",ylim=(0,68))
            ax.set_xticks([1,8,32,64]);ax.grid(alpha=.2);ax.legend(fontsize=8,loc="upper left")
        fig.suptitle("SMC strong scaling: fixed work within each curve",fontsize=19)
        fig.text(.06,.025,"128-particle pilot complete; 512-particle follow-up PARTIAL. Separate workloads, never pooled. Bars: seed min/max, not population CIs.",fontsize=10)
        save(fig,"01_parallel_scaling")
        fig,axes=plt.subplots(1,3,figsize=(13.3,6.7))
        shown=["ordersmc-v07","order-mcmc","bnlearn-tabu-shared-score","bnlearn-hc-shared-score"]
        for ax,cohort,title in zip(axes,("v3_mixed","v3_clg","metadata"),("Unrestricted mixed: 220 datasets","Restricted mixed / CLG: 60","Metadata M1 synthetic: 5")):
            g=recovery_summary[recovery_summary.cohort==cohort].set_index("method").loc[shown]
            y=np.arange(len(g));ax.barh(y-.17,g.mean_directed_f1,.32,label="Directed F1",color="#1768ac")
            ax.barh(y+.17,g.mean_skeleton_f1,.32,label="Skeleton F1",color="#82b7a8")
            ax.set_yticks(y,[LABELS[m] for m in shown],fontsize=9);ax.invert_yaxis();ax.set(xlim=(0,1.02),title=title,xlabel="Dataset mean F1")
            ax.legend(fontsize=9,loc="lower left");ax.grid(axis="x",alpha=.2)
        fig.suptitle("Corrected recovery: similar to MCMC, not uniform dominance",fontsize=19)
        fig.text(.04,.025,"Graphs from mean evaluation-seed PIPs; baseline snapshot 0; original cohort priors. Not equal-compute comparisons. No biological observations.",fontsize=9.5)
        save(fig,"02_corrected_graph_recovery")
        fig,axes=plt.subplots(1,3,figsize=(13.3,6.5))
        for ax,cohort,title in zip(axes,("v3_mixed","v3_clg","metadata"),("Unrestricted mixed: 220 datasets","Restricted mixed / CLG: 60","Metadata M1 synthetic: 5")):
            g=stability_summary[(stability_summary.cohort==cohort)&stability_summary.method.isin(["ordersmc-v07","order-mcmc"])].set_index("method").loc[["ordersmc-v07","order-mcmc"]]
            x=np.arange(2);ax.bar(x-.17,g.mean_directed_jaccard,.32,label="Directed graph",color="#1768ac")
            ax.bar(x+.17,g.mean_skeleton_jaccard,.32,label="Skeleton",color="#82b7a8")
            ax.set_xticks(x,[f"{LABELS[m]}\n{int(g.loc[m,'seeds'])} seeds" for m in g.index],fontsize=10)
            ax.set(ylim=(0,1.02),ylabel="Mean pairwise Jaccard",title=title);ax.grid(axis="y",alpha=.2)
            ax.legend(loc="lower left",fontsize=9)
        fig.suptitle("Reproducibility depends on dataset and budget",fontsize=19)
        fig.text(.05,.025,"Same-data seed reproducibility, then averaged over datasets. Metadata uses 8 SMC versus 4 MCMC seeds; repeatability does not establish correctness.",fontsize=9.5)
        save(fig,"03_corrected_reproducibility")
        fig,axes=plt.subplots(1,2,figsize=(13.3,6.5))
        methods=["adaptive_original_budget","adaptive_matched_effort","order-mcmc"]
        for m in methods:
            flags=gate["arms"][m]["flags"];means=[f["mean"] for f in flags];p95=[f["p95"] for f in flags]
            axes[0].plot(range(1,10),means,"o-",label=LABELS[m]);axes[1].plot(range(1,10),p95,"o-",label=LABELS[m])
        for ax,limit,title in zip(axes,(.012,.020),("Cell mean dataset RMSE","Cell 95th percentile dataset RMSE")):
            ax.axhline(limit,color="grey",linestyle="--",label="Frozen limit");ax.set(title=title,xlabel="Experimental cell (mapping in source tables)",ylabel="PIP RMSE",ylim=(0,None))
            ax.set_xticks(range(1,10));ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.suptitle("Exact posterior validation: effort-matched SMC passes 9/9 cells",fontsize=19)
        fig.text(.05,.025,"90 datasets, four inference seeds per dataset. Original adaptive budget passes 8/9. Matched effort is a separate development arm, not equal wall time.",fontsize=9.5)
        save(fig,"04_exact_posterior_validation")
        fig,axes=plt.subplots(1,2,figsize=(13.3,6.5))
        for ax,kind,metric,title in ((axes[0],"metadata-null","false_edges","Null controls: selected false edges"),
                                  (axes[1],"metadata-signal","directed_f1","Positive controls: directed F1")):
            g=prior_data[(prior_data.kind==kind)&(prior_data.final_target_steps==0)]
            for x,arm in enumerate(("fixed","dimension")):
                vals=g[g.arm==arm][metric].to_numpy();ax.bar(x,float(np.mean(vals)),color=["#999999","#1768ac"][x],alpha=.75)
                ax.scatter(x+np.linspace(-.09,.09,len(vals)),vals,color="black",s=25,zorder=3)
            ax.set_xticks([0,1],["lambda = 1","lambda = log(d - 1)"]);ax.set(title=title,ylabel=metric.replace("_"," "))
            if metric=="directed_f1":ax.set_ylim(0,1)
            ax.grid(axis="y",alpha=.2)
        fig.suptitle("Independent prior check: 196 variables, 222 observations",fontsize=19)
        fig.text(.05,.025,"Three fresh datasets per condition, four paired seeds; dots are dataset means. Snapshot 0 primary. Small replication, not a default-model promotion.",fontsize=9.5)
        save(fig,"05_independent_prior_check")
        fig,axes=plt.subplots(2,3,figsize=(13.3,7.5))
        old=pd.read_csv(output/"timing/completed_128_particle_pilot/per_run_metrics.csv")
        new=pd.read_csv(output/"timing/partial_512_particle_followup/per_run_metrics.csv")
        for col,d in enumerate((12,56,196)):
            use=new if d==12 else old
            label="New short-budget grid (3 seeds)" if d==12 else "Older grid (3 seeds)"
            f=use[(use.d==d)&(use.arm=="dimension")&(use.ranks==64)&(use.variant.isin(["SMC adaptive","MCMC"]))]
            for variant,g in f.groupby("variant"):
                for row,metric in enumerate(("directed_f1","skeleton_f1")):
                    p=g.groupby("budget").agg(t=("inference_seconds","median"),m=(metric,"mean"),lo=(metric,"min"),hi=(metric,"max")).sort_values("t")
                    axes[row,col].errorbar(p.t.to_numpy(),p.m.to_numpy(),yerr=plot_ranges(p.m,p.lo,p.hi),marker="o",capsize=3,label=variant)
                    axes[row,col].set(xscale="log",xlabel="Inference seconds",ylabel=metric.replace("_"," "))
                    axes[row,col].grid(alpha=.2)
            axes[0,col].set_title(f"{d} variables\n{label}",fontsize=12);axes[0,col].legend(fontsize=9)
        fig.suptitle("Quality versus time at 64 cores: empirical trade-offs",fontsize=19)
        fig.text(.04,.025,"Separate panels/editions, not pooled. Old MCMC grid starts at 30 sampling seconds; new starts at 0.5. No universal faster-than-MCMC conclusion.",fontsize=9.5)
        save(fig,"06_recovery_versus_time")
        text_slide("What the paper can—and cannot—claim yet",[
            ("Supported descriptive findings","Substantial within-SMC MPI acceleration; competitive recovery and high reproducibility on the broad mixed benchmark; computable exact-target validation with the documented effort policy."),
            ("Not established universally","SMC is not faster than MCMC at comparable recovery in every case. Short MCMC budgets can remove an apparent advantage. Larger-network MCMC convergence checks failed; probability agreement is not posterior accuracy."),
            ("Still needed","Finish the 108 queued timing runs, inspect all failures and quality curves, then expand independently selected scaling datasets. A 128/256-core extension has only passed scheduler feasibility, not MPI validation or inference."),
            ("Biological scope","These are synthetic and metadata-based synthetic results. No living-brain biological discovery, causal identification guarantee or conference/journal acceptance claim follows from this package.")])
    shutil.copy2(__file__,output/"reproduce/publication_snapshot.py")
    for name in ("scaling_quality.py","timing_report.py"):
        shutil.copy2(Path(__file__).with_name(name),output/"reproduce"/name)
    for path in (output/"timing").rglob("artifact_hashes.json"):
        inputs.append(dict(source=str(path),sha256=digest(path),copy=str(path.relative_to(output))))
    _atomic_json(output/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=inputs,
        build_code_sha256=digest(__file__),model_or_fits_changed=False,queued_jobs_changed=False,
        new_inference_submitted=False,publication_ready=False,
        note="Presentation-ready figures over current corrected evidence; partial experiments explicitly retained."))
    files={str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file() and p.name!="checksums.json"}
    _atomic_json(output/"checksums.json",files)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    build(parser.parse_args().output)


if __name__=="__main__":main()
