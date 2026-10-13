"""Submit bounded competitor phases, recording every successful submission."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

from .rank_competitors import checked_plan, aggregate
from .readiness_audits import read
from .execution import _atomic_json


def submit(root,phase,concurrency):
    if not 1 <= concurrency <= 64: raise ValueError("concurrency must be in [1,64]")
    plan=checked_plan(root)
    acceptance=read(root/"validation/preflight.json")
    if not acceptance["all_passed"] or acceptance["plan_sha256"]!=plan["plan_sha256"]:
        raise RuntimeError("preflight failed; no fits submitted")
    record=read(root/"submission.json")
    if phase in record: raise RuntimeError("phase was already submitted; inspect saved job IDs")
    if phase=="extended": aggregate(root)  # primary failures remain recorded, not hidden
    indices=[str(i) for i,t in enumerate(plan["tasks"]) if t["phase"]==phase]
    worker=Path(os.environ["ORDERSMC_REPO_ROOT"])/"scripts/slurm/rank_competitors.sh"
    extra=shlex.split(os.environ.get("ORDERSMC_SBATCH_ARGS",""))
    def sbatch(key,*options):
        result=subprocess.run(["sbatch","--parsable",*extra,*options,str(worker)],check=True,text=True,capture_output=True)
        job=result.stdout.strip().split(';')[0]
        record[key]=job;_atomic_json(root/"submission.json",record)
        print(key,job,flush=True)
        return job
    if not indices: raise RuntimeError("empty competitor phase")
    job=sbatch(phase,"--job-name=ordersmc-competitors",f"--array={','.join(indices)}%{concurrency}",
               "--export=ALL,COMPETITOR_ACTION=run","--time=02:00:00" if phase=="extended" else "--time=00:45:00")
    action="submit-extended" if phase=="primary" else "aggregate"
    sbatch(f"{phase}_controller",f"--job-name={'ordersmc-comp-next' if phase=='primary' else 'ordersmc-comp-summary'}",
           f"--dependency=afterany:{job}",f"--export=ALL,COMPETITOR_ACTION={action}","--time=00:20:00")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--phase",choices=["primary","extended"],required=True)
    p.add_argument("--concurrency",type=int,default=24)
    a=p.parse_args();submit(a.root,a.phase,a.concurrency)
