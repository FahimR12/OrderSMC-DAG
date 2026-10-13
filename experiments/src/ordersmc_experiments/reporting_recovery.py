"""Audited, reporting-only overlays; original inference snapshots remain intact."""
from __future__ import annotations

import argparse
import ast
import difflib
from pathlib import Path
import shutil
import subprocess
import os
import shlex

from .execution import _atomic_json, _utc
from .readiness import canonical_sha
from .readiness_audits import read, sha

REPO = Path(__file__).resolve().parents[3]
EDITION = "20260918-v1"


def reporting_only_diff(original, patched, allowed_functions):
    """Fail if anything outside declared reporting entry points was changed."""
    def remaining(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in allowed_functions)]
        return ast.dump(tree, include_attributes=False)
    if remaining(original) != remaining(patched):
        raise RuntimeError("patch changes code outside the permitted reporting functions")


def prepare(root, kind):
    recovery = root / "reporting_recovery" / EDITION
    if recovery.exists():
        raise FileExistsError("reporting recovery already exists; do not overwrite its provenance")
    plan = read(root / "plan.json")
    original = root / "code_snapshot"
    frozen = plan["source_files"] if kind == "rank" else read(root/"source_freeze.json")["files"]
    for relative, digest in frozen.items():
        if sha(original/relative) != digest:
            raise RuntimeError(f"original frozen source changed: {relative}")
    module = "rank_competitors" if kind == "rank" else "execution"
    relative = f"experiments/src/ordersmc_experiments/{module}.py"
    old, new = (original/relative).read_text(), (REPO/relative).read_text()
    allowed = {"aggregate", "reporting_records", "checked_plan", "main"} if kind == "rank" else {"aggregate"}
    reporting_only_diff(old, new, allowed)
    inputs = {str(p.relative_to(root)):sha(p) for folder in ("runs", "datasets", "validation")
              for p in sorted((root/folder).rglob('*')) if p.is_file()}
    inputs["plan.json"] = sha(root/"plan.json")
    for name in ("source_freeze.json",):
        if (root/name).exists(): inputs[name]=sha(root/name)
    snapshot = recovery / "code_snapshot"
    for name in frozen:
        destination = snapshot/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original/name,destination)
    shutil.copy2(REPO/relative,snapshot/relative)
    additions = ("experiments/src/ordersmc_experiments/reporting_recovery.py", "scripts/slurm/recovered_report.sh")
    for name in additions:
        (snapshot/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,snapshot/name)
    if (root/"analysis").exists(): shutil.copytree(root/"analysis",recovery/"pre_repair_analysis")
    manifest = dict(schema="ordersmc-reporting-recovery-v1",kind=kind,created_utc=_utc(),
        root=str(root.resolve()),plan_sha256=plan["plan_sha256"],frozen_source_files=frozen,
        preserved_input_files=inputs,changed_functions=sorted(allowed),
        reporting_files={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},
        diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                        fromfile=f"original/{relative}",tofile=f"reporting/{relative}")))
    manifest["manifest_sha256"]=canonical_sha(manifest)
    _atomic_json(recovery/"manifest.json",manifest)
    print(recovery,flush=True)
    return recovery


def verify(root):
    recovery=root/"reporting_recovery"/EDITION
    manifest=read(recovery/"manifest.json")
    if canonical_sha({k:v for k,v in manifest.items() if k!="manifest_sha256"})!=manifest["manifest_sha256"]:
        raise RuntimeError("reporting manifest hash mismatch")
    if root.resolve()!=Path(manifest["root"]).resolve():
        raise RuntimeError("reporting manifest belongs to another result tree")
    for base, files in ((root/"code_snapshot",manifest["frozen_source_files"]),
                        (recovery/"code_snapshot",manifest["reporting_files"]),
                        (root,manifest["preserved_input_files"])):
        for relative,digest in files.items():
            if sha(base/relative)!=digest: raise RuntimeError(f"preserved artifact changed: {base/relative}")
    return recovery,manifest


def report(root, require_complete=False):
    recovery,manifest=verify(root)
    if REPO.resolve()!=(recovery/"code_snapshot").resolve():
        raise RuntimeError("run reporting from the verified overlay, not the live checkout")
    def state(stage,status="running",**extra):
        _atomic_json(recovery/"state.json",dict(stage=stage,status=status,updated_utc=_utc(),
            manifest_sha256=manifest["manifest_sha256"],**extra))
    state("aggregate")
    try:
        if manifest["kind"]=="rank":
            from .rank_competitors import aggregate
            completion=aggregate(root,source_root=root/"code_snapshot")
        else:
            from .execution import aggregate
            from . import paper_metrics,section2_analysis
            aggregate(root/"plan.json",root)
            completion=read(root/"analysis/completion.json")
            if completion["missing_or_failed_runs"]:
                raise RuntimeError("metadata fits incomplete; do not publish complete-cohort tables")
            state("paper_metrics")
            paper_metrics.analyse(root)
            state("section2_analysis")
            section2_analysis.analyse(root)
        verify(root)
        _atomic_json(root/"analysis/reporting_provenance.json",dict(
            manifest=str((recovery/"manifest.json").resolve()),manifest_sha256=manifest["manifest_sha256"],
            original_fits_and_source_preserved=True,reporting_completed_utc=_utc()))
        state("finished",status=completion["status"])
        if require_complete and completion["status"]!="complete":
            raise RuntimeError("comparison still contains missing/failed fits; inspect completion.json")
    except Exception as error:
        state("reporting_failed",status="failed",error_type=type(error).__name__,error=str(error))
        raise


def resume_extended(root, concurrency=24):
    """Submit untouched frozen workers; the final report uses the verified overlay."""
    recovery,manifest=verify(root)
    if manifest["kind"]!="rank": raise ValueError("extended stage is rank-comparison only")
    from .rank_competitors import checked_plan
    plan=checked_plan(root,source_root=root/"code_snapshot")
    preflight=read(root/"validation/preflight.json")
    if not preflight["all_passed"] or preflight["plan_sha256"]!=plan["plan_sha256"]:
        raise RuntimeError("original adapter preflight not passed")
    if not 1<=concurrency<=64: raise ValueError("concurrency must be in [1,64]")
    completion=read(root/"analysis/completion.json")
    expected=sum(len(t["seed_indices"]) for t in plan["tasks"] if t["phase"]=="primary")
    if (completion["plan_sha256"]!=plan["plan_sha256"] or completion["failed"]
            or completion["completed_new_fits"]<expected):
        raise RuntimeError("inspect incomplete/failed primary fits before resuming")
    record=read(root/"submission.json")
    if "extended_controller" in record: raise RuntimeError("extended stage and controller already submitted")
    env=dict(os.environ,OUTPUT_ROOT=str(root.resolve()),ORDERSMC_REPO_ROOT=str((root/"code_snapshot").resolve()),
             REPORTING_REPO_ROOT=str((recovery/"code_snapshot").resolve()))
    extra=shlex.split(env.get("ORDERSMC_SBATCH_ARGS",""))
    logs=recovery/"logs";logs.mkdir(exist_ok=True)
    def sbatch(key,worker,*args):
        process=subprocess.run(["sbatch","--parsable",*extra,*args,str(worker)],check=True,
            text=True,capture_output=True,env=env,cwd=root/"code_snapshot")
        job=process.stdout.strip().split(';')[0]
        record[key]=job;_atomic_json(root/"submission.json",record)
        print(key,job,flush=True);return job
    if "extended" not in record:
        active=subprocess.check_output(["squeue","--noheader","--user",str(os.getuid()),
            "--name=ordersmc-competitors,ordersmc-comp-next,ordersmc-comp-summary","--format=%A"],text=True)
        if active.strip(): raise RuntimeError("competitor jobs still active; refusing duplicate submission")
        indices=[str(i) for i,t in enumerate(plan["tasks"]) if t["phase"]=="extended"]
        sbatch("extended",root/"code_snapshot/scripts/slurm/rank_competitors.sh",
               "--job-name=ordersmc-competitors",f"--array={','.join(indices)}%{concurrency}",
               "--export=ALL,COMPETITOR_ACTION=run","--time=02:00:00")
    sbatch("extended_controller",recovery/"code_snapshot/scripts/slurm/recovered_report.sh",
           "--job-name=ordersmc-comp-summary",f"--dependency=afterany:{record['extended']}",
           "--time=00:30:00",f"--output={logs}/%x-%j.out")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","report","resume-extended","verify"])
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--kind",choices=["rank","metadata"])
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--concurrency",type=int,default=24)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=="prepare":
        if args.kind is None: parser.error("prepare requires --kind")
        prepare(root,args.kind)
    elif args.action=="report": report(root,args.require_complete)
    elif args.action=="resume-extended": resume_extended(root,args.concurrency)
    else: verify(root);print("original artifacts and reporting overlay verified")


if __name__=="__main__": main()
