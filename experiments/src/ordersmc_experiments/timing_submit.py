"""Resumable small-inventory scheduler for the controlled timing pilot."""
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
from .sampler_corrections import checked_input


def submit(root, action, *, sbatch=subprocess.check_output):
    root = root.resolve(); plan = catchup.checked_plan(root)
    if action not in {"start", "prepare", "launch"}: raise ValueError("unknown submission stage")
    (root / "logs").mkdir(exist_ok=True)
    with (root / ".submission.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = root / "submission.json"
        record = read(path) if path.exists() else dict(plan_sha256=plan["plan_sha256"], jobs={})
        if record["plan_sha256"] != plan["plan_sha256"]: raise RuntimeError("stale submission")
        env = dict(os.environ, OUTPUT_ROOT=str(root), ORDERSMC_REPO_ROOT=str(root / "code_snapshot"), UCX_TLS="self,sm")
        def job(key, step, *, extra=(), ranks=1, hours=4):
            if key in record["jobs"]: return record["jobs"][key]
            args = ["sbatch", "--parsable", *shlex.split(os.environ.get("ORDERSMC_SBATCH_ARGS", "")),
                f"--job-name={plan['config'].get('job_prefix', 'timing')}-{key}", f"--output={root}/logs/%x-%A_%a.out", "--nodes=1",
                f"--ntasks={ranks}", "--cpus-per-task=1", "--hint=nomultithread", "--mem=64G",
                f"--time={hours:02d}:00:00", *extra, f"--export=ALL,TIMING_STEP={step}",
                str(root / "code_snapshot/scripts/slurm/timing_pilot.sh")]
            value = sbatch(args, env=env, text=True).strip().split(";")[0]
            if not value.isdigit(): raise RuntimeError("invalid scheduler job ID")
            record["jobs"][key] = value; record.setdefault("commands", {})[key] = args
            _atomic_json(path, record); print("submitted", key, value, flush=True)
            return value
        if action == "start":
            pre = job("preflight", "preflight", ranks=4)
            job("prepare-controller", "submit-prepare", extra=[f"--dependency=afterany:{pre}"])
        else:
            if read(root / "validation/preflight.json") != dict(all_passed=True, plan_sha256=plan["plan_sha256"]):
                raise RuntimeError("benchmark preflight failed")
            if action == "prepare":
                prep = job("prepare", "prepare", extra=[f"--array=0-{len(plan['datasets'])-1}%3"])
                job("launch-controller", "submit-launch", extra=[f"--dependency=afterany:{prep}"])
            else:
                for d in plan["datasets"]:
                    p = read(root / "prepared" / d["key"] / "prepared.json")
                    if p["plan_sha256"] != plan["plan_sha256"] or p["status"] != "complete":
                        raise RuntimeError("candidate preparation failed")
                    for case in p["cases"].values():
                        checked_input(case["candidate"])
                        if "exact" in case: checked_input(case["exact"])
                cfg = plan["config"]
                dependencies = []
                if plan["references"]:
                    dependencies.append(job("references", "reference", ranks=cfg["reference_ranks"],
                        extra=[f"--array=0-{len(plan['references'])-1}%{cfg['reference_concurrency']}"]))
                bundles = job("bundles", "bundle", ranks=max(cfg["ranks"]), hours=cfg.get("bundle_hours", 24),
                    extra=["--exclusive", f"--array=0-{len(plan['bundles'])-1}%{cfg['bundle_concurrency']}"])
                dependencies.append(bundles)
                job("report", "aggregate", extra=["--dependency=afterany:" + ":".join(dependencies)])
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "prepare", "launch"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); submit(args.output, args.action)


if __name__ == "__main__": main()
