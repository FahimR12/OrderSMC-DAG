"""Archive and replace derived gate reports without touching fits or frozen code."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from . import catchup
from .accuracy_gate import dataset_rmse_gate
from .execution import _atomic_json
from .readiness_audits import read, sha

EDITION = "dataset-gate-v1"


def repair(root, cohort, reporting_root):
    reporting_plan = catchup.checked_plan(reporting_root)
    plan = catchup.checked_source(root/"plan.json")
    if plan["plan_sha256"] != reporting_plan["config"]["source_plan_sha256"]:
        raise RuntimeError("report correction refers to a different source plan")
    for relative, digest in plan["source_files"].items():
        if sha(root/"code_snapshot"/relative) != digest:
            raise RuntimeError("original source snapshot changed")
    directory = root/"analysis"/cohort
    completion = read(directory/"completion.json")
    if completion.get("status") != "complete" or completion.get("plan_sha256") != plan["plan_sha256"] or completion.get("missing"):
        raise RuntimeError("cannot correct an incomplete cohort report")
    if not (directory/"interpretation.json").is_file():
        raise RuntimeError("original reporting has not finished")
    path = directory/"per_run_metrics.csv"; csv_hash = sha(path)
    frame = pd.read_csv(path)
    selected = frame[frame.method == "ordersmc-v07"]
    expected = {(f["dataset_key"], f["budget"], steps, f["seed_index"])
                for f in plan["fits"] if f["cohort"] == cohort for steps in f["config"]["final_target_snapshot_steps"]}
    actual = set(selected[["dataset_key","budget","final_target_steps","seed_index"]].itertuples(index=False,name=None))
    if actual != expected or len(actual) != len(selected):
        raise RuntimeError("report lacks the complete unique frozen seed/snapshot inventory")
    corrected = dataset_rmse_gate(frame, plan["config"]["exact_rmse_flags"])
    overlay = root/"reporting_corrections"/EDITION/cohort
    destination = directory/"exact_rmse_flags.json"
    manifest_path = overlay/"manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["per_run_metrics_sha256"] != csv_hash or manifest["reporting_plan_sha256"] != reporting_plan["plan_sha256"]:
            raise RuntimeError("report input or correction source changed; use a new edition")
    else:
        overlay.mkdir(parents=True,exist_ok=True)
        original_hash = sha(destination)
        shutil.copy2(destination, overlay/"original_per_seed_gate.json")
        manifest = dict(source_plan_sha256=plan["plan_sha256"], cohort=cohort,
            reporting_plan_sha256=reporting_plan["plan_sha256"], reporting_root=str(reporting_root.resolve()),
            per_run_metrics_sha256=csv_hash, original_gate_sha256=original_hash,
            fit_files_changed=False, frozen_source_changed=False,
            change="quantiles over dataset-averaged RMSE; retain per-seed percentile separately")
        _atomic_json(manifest_path,manifest)
    if sha(overlay/"original_per_seed_gate.json") != manifest["original_gate_sha256"] or sha(path) != csv_hash:
        raise RuntimeError("input or archived report changed")
    _atomic_json(overlay/"exact_rmse_flags.json", corrected)
    _atomic_json(destination,corrected)
    _atomic_json(directory/"accuracy_gate_correction.json", dict(
        manifest=str(manifest_path.resolve()), report_sha256=sha(destination), statistical_unit="dataset",
        fit_files_changed=False, original_gate_archived=True))
    print(f"corrected {cohort}: {sum(f['passed'] is True for f in corrected['flags'])}/{len(corrected['flags'])} evaluable-or-listed conditions",flush=True)
    return corrected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path);p.add_argument("--cohort",required=True)
    p.add_argument("--reporting-root",required=True,type=Path)
    args=p.parse_args();repair(args.root,args.cohort,args.reporting_root)


if __name__=="__main__":main()
