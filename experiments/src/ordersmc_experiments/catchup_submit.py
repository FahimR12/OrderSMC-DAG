"""Resumable, bounded Slurm submissions for the frozen catch-up plan."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import shlex
import subprocess

from .catchup import checked_plan
from .execution import _atomic_json, _compressed_indices
from .readiness_audits import read


def submit(root, action, *, sbatch=subprocess.check_output):
    root = root.resolve()
    plan = checked_plan(root)
    frozen = root / "code_snapshot"
    (root / "logs").mkdir(exist_ok=True)
    with (root / ".submission.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = root / "submission.json"
        record = read(path) if path.exists() else dict(plan_sha256=plan["plan_sha256"], jobs={})
        if record["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("submission belongs to a different plan")
        env = dict(os.environ, OUTPUT_ROOT=str(root), ORDERSMC_REPO_ROOT=str(frozen), UCX_TLS="self,sm")

        def job(key, task_action, options, extra=""):
            if key in record["jobs"]:
                return record["jobs"][key]
            args = ["sbatch", "--parsable", *shlex.split(os.environ.get("ORDERSMC_SBATCH_ARGS", "")),
                    f"--job-name=v07-{key}", f"--output={root}/logs/%x-%A_%a.out",
                    f"--export=ALL,CATCHUP_ACTION={task_action}{extra}", *options,
                    str(frozen / "scripts/slurm/catchup.sh")]
            raw = sbatch(args, env=env, text=True).strip().split(";")[0]
            if not raw.isdigit():
                raise RuntimeError(f"unexpected scheduler response: {raw}")
            record["jobs"][key] = raw
            record.setdefault("commands", {})[key] = args
            _atomic_json(path, record)  # Save EVERY accepted job before attempting another.
            print(f"submitted {key}={raw}", flush=True)
            return raw

        serial = ["--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=8G", "--time=02:00:00"]
        if action == "start":
            preflight = job("preflight", "preflight", [*serial, "--ntasks=4"])
            job("launch", "launch", [*serial, f"--dependency=afterany:{preflight}"])
            return record
        acceptance = read(root / "validation/preflight.json")
        if not acceptance.get("all_passed") or acceptance["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("preflight did not pass; no scientific arrays were submitted")
        indices = defaultdict(list)
        for index, task in enumerate(plan["tasks"]):
            indices[task["cohort"], task["mpi_ranks"]].append(index)
        # Four seeds per small/medium task keep total queued array elements
        # comfortably below this account's 2,500 submit-job limit.
        reports = []
        for (cohort, ranks), selected in sorted(indices.items()):
            cap = plan["config"]["mpi_concurrency_per_cohort"] if ranks > 1 else plan["config"]["serial_concurrency"]
            array = job(cohort, "run", [f"--array={_compressed_indices(selected)}%{cap}",
                "--nodes=1", f"--ntasks={ranks}", "--cpus-per-task=1",
                "--mem=64G" if ranks > 1 else "--mem=8G", "--time=12:00:00"])
            dependencies = [array]
            if cohort == "metadata":
                pilot_root = Path(next(f["external"]["root"] for f in plan["fits"] if "external" in f))
                pilot_jobs = read(pilot_root / "submission.json")
                dependencies.append(str(pilot_jobs["report"]))
            reports.append(job(f"{cohort}-report", "aggregate",
                [*serial, "--mem=24G", f"--dependency=afterany:{':'.join(dependencies)}"],
                extra=f",CATCHUP_COHORT={cohort}"))
        job("completion", "completion", [*serial, f"--dependency=afterany:{':'.join(reports)}"])
        record["all_cohorts_submitted"] = True
        _atomic_json(path, record)
        return record


def completion(root):
    plan = checked_plan(root)
    reports = {}
    for cohort in sorted({d["cohort"] for d in plan["datasets"]}):
        path = root / "analysis" / cohort / "completion.json"
        report = read(path) if path.exists() else dict(status="missing-report")
        if report.get("plan_sha256") != plan["plan_sha256"]:
            report = dict(status="missing-or-stale-report")
        # A fit-completeness record can precede a failed reporting operation.
        interpretation = path.with_name("interpretation.json")
        if report.get("status") == "complete" and not interpretation.exists():
            report = dict(status="reporting-incomplete", fits=report)
        reports[cohort] = report
    complete = all(r.get("status") == "complete" for r in reports.values())
    _atomic_json(root / "analysis/completion.json", dict(plan_sha256=plan["plan_sha256"],
        status="complete" if complete else "incomplete", cohorts=reports))
    if not complete:
        raise RuntimeError("catch-up incomplete; inspect per-cohort reports and logs, no failures dropped")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "launch", "completion"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.action == "completion":
        completion(args.output)
    else:
        submit(args.output, args.action)


if __name__ == "__main__":
    main()
