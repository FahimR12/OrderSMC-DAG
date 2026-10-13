"""Resumable submission of numerical reporting repair and null-prior pilot."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
from . import catchup
from .execution import _atomic_json, _compressed_indices
from .readiness_audits import read


def submit(root, workflow, action, *, sbatch=subprocess.check_output):
    root=root.resolve();plan=catchup.checked_plan(root)
    if workflow not in {"reporting_repair","null_pilot","orientation_audit","prior_holdout"} or action not in {"start","launch"}:raise ValueError("invalid workflow/action")
    (root/"logs").mkdir(exist_ok=True)
    with (root/".submission.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        path=root/"submission.json";record=read(path) if path.exists() else dict(plan_sha256=plan["plan_sha256"],jobs={})
        if record["plan_sha256"]!=plan["plan_sha256"]:raise RuntimeError("stale submission")
        env=dict(os.environ,OUTPUT_ROOT=str(root),ORDERSMC_REPO_ROOT=str(root/"code_snapshot"),AUDIT_WORKFLOW=workflow,UCX_TLS="self,sm")
        def job(key,step,extra=(),ranks=1):
            if key in record["jobs"]:return record["jobs"][key]
            args=["sbatch","--parsable",*shlex.split(os.environ.get("ORDERSMC_SBATCH_ARGS","")),
                f"--job-name=audit-{workflow}-{key}",f"--output={root}/logs/%x-%A_%a.out",
                "--nodes=1",f"--ntasks={ranks}","--cpus-per-task=1","--mem=32G" if ranks>1 else "--mem=16G",
                "--time=04:00:00",*extra,f"--export=ALL,AUDIT_STEP={step}",str(root/"code_snapshot/scripts/slurm/report_null_audit.sh")]
            value=sbatch(args,env=env,text=True).strip().split(';')[0]
            if not value.isdigit():raise RuntimeError("invalid scheduler response")
            record["jobs"][key]=value;record.setdefault("commands",{})[key]=args;_atomic_json(path,record)
            print("submitted",key,value,flush=True);return value
        if action=="start":
            pre=job("preflight","preflight",ranks=1 if workflow=="orientation_audit" else 4)
            job("launch","launch",[f"--dependency=afterany:{pre}"])
        else:
            check=read(root/"validation/preflight.json")
            if not check["all_passed"] or check["plan_sha256"]!=plan["plan_sha256"]:raise RuntimeError("preflight failed")
            if workflow=="reporting_repair":
                arr=job("reports","run",[f"--array=0-{len(plan['cohorts'])-1}%2"])
                job("complete","complete",[f"--dependency=afterany:{arr}"])
            elif workflow in {"orientation_audit","prior_holdout"}:
                arr=job("cases","run",[f"--array=0-{len(plan['cases'])-1}%6"])
                job("aggregate","aggregate",[f"--dependency=afterany:{arr}"])
            else:
                arrays=[]
                for ranks in sorted({t["mpi_ranks"] for t in plan["tasks"]}):
                    indices=[i for i,t in enumerate(plan["tasks"]) if t["mpi_ranks"]==ranks]
                    arrays.append(job(f"fits-{ranks}","run",[f"--array={_compressed_indices(indices)}%{4 if ranks>1 else 12}"],ranks=ranks))
                job("aggregate","aggregate",[f"--dependency=afterany:{':'.join(arrays)}"])
        return record


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["start","launch"])
    p.add_argument("--workflow",choices=["reporting_repair","null_pilot","orientation_audit","prior_holdout"],required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();submit(a.output,a.workflow,a.action)


if __name__=="__main__":main()
