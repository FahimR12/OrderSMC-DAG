"""Resumable submissions for the effort diagnostic and reporting-only repairs."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess

from . import catchup
from .execution import _atomic_json
from .readiness_audits import read


def submit(root, action, *, sbatch=subprocess.check_output):
    root=root.resolve();plan=catchup.checked_plan(root)
    (root/"logs").mkdir(exist_ok=True)
    with (root/".submission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/"submission.json"
        record=read(path) if path.exists() else dict(plan_sha256=plan["plan_sha256"],jobs={})
        if record["plan_sha256"] != plan["plan_sha256"]:raise RuntimeError("submission plan mismatch")
        env=dict(os.environ,OUTPUT_ROOT=str(root),ORDERSMC_REPO_ROOT=str(root/"code_snapshot"))
        def job(key, step, options=(), extra=""):
            if key in record["jobs"]:return record["jobs"][key]
            command=["sbatch","--parsable",*shlex.split(os.environ.get("ORDERSMC_SBATCH_ARGS","")),
                f"--job-name=effort-{key}",f"--output={root}/logs/%x-%A_%a.out",
                "--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=8G","--time=04:00:00",
                *options,f"--export=ALL,EFFORT_ACTION={step}{extra}",str(root/"code_snapshot/scripts/slurm/effort_match.sh")]
            value=sbatch(command,env=env,text=True).strip().split(';')[0]
            if not value.isdigit():raise RuntimeError(f"invalid scheduler response: {value}")
            record["jobs"][key]=value;record.setdefault("commands",{})[key]=command
            _atomic_json(path,record);print(f"submitted {key}={value}",flush=True);return value
        if action=="start":
            preflight=job("preflight","preflight")
            job("launch","launch",[f"--dependency=afterany:{preflight}"])
            # These repairs do not depend on the new fits. Each waits for its
            # original report, so frozen jobs cannot subsequently overwrite it.
            source=Path(plan["source_root"]);old_jobs=read(source/"submission.json")["jobs"]
            source_plan=catchup.checked_source(source/"plan.json")
            for cohort in sorted({d["cohort"] for d in source_plan["datasets"]}):
                complete=source/"analysis"/cohort/"completion.json"
                already=(complete.exists() and read(complete).get("status")=="complete"
                         and (complete.parent/"interpretation.json").exists())
                dependencies=[] if already else [f"--dependency=afterany:{old_jobs[cohort+'-report']}"]
                job(f"gate-{cohort}","repair",dependencies,extra=f",GATE_COHORT={cohort}")
        else:
            acceptance=read(root/"validation/preflight.json")
            if not acceptance.get("all_passed") or acceptance["plan_sha256"]!=plan["plan_sha256"]:
                raise RuntimeError("preflight failed; no effort fits submitted")
            array=job("fits","run",[f"--array=0-{len(plan['tasks'])-1}%{plan['config']['concurrency']}"])
            job("report","aggregate",[f"--dependency=afterany:{array}"])
        return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["start","launch"]);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();submit(args.output,args.action)


if __name__=="__main__":main()
