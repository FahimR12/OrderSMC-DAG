"""New reporting edition over unchanged, hashed catch-up posteriors."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from . import catchup
from .execution import _atomic_json
from .readiness import canonical_sha
from .readiness_audits import read, sha
from .sampler_corrections import artifact, checked_input

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "results/sampler-corrections/v07-catchup-v1"


def snapshot(root, payload):
    if (root/"plan.json").exists():
        raise FileExistsError("freeze requires a new plan")
    folders=("src","experiments/src","experiments/configs","experiments/adapters",
             "experiments/fixtures","experiments/validation","scripts","tests","experiments/tests","docs",
             "summary_statistics_multiopmic_data_pathway_molecule_level")
    files={str(p.relative_to(REPO)):sha(p) for folder in folders for p in sorted((REPO/folder).rglob("*"))
           if p.is_file() and "__pycache__" not in p.parts}
    for name in ("pyproject.toml","README.md","LICENSE"):files[name]=sha(REPO/name)
    payload={**payload,"source_files":files};payload["plan_sha256"]=canonical_sha(payload)
    for name,digest in files.items():
        dest=root/"code_snapshot"/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,dest)
        if sha(dest)!=digest:raise RuntimeError("source changed while freezing")
    _atomic_json(root/"plan.json",payload)
    return payload


def source_plan(plan):
    path=checked_input(plan["source_plan"]);source=path.parent
    original=catchup.checked_source(path)
    for relative,digest in original["source_files"].items():
        if sha(source/"code_snapshot"/relative)!=digest:
            raise RuntimeError("original source freeze was changed")
        if relative.startswith("src/ordersmc/") and relative!="src/ordersmc/graphs.py" and sha(REPO/relative)!=digest:
            raise RuntimeError("reporting repair cannot change scores, priors or sampler")
    return source,original


def freeze(root):
    plan=catchup.checked_source(SOURCE/"plan.json")
    records=[]
    for fit in plan["fits"]:
        _,path=catchup.fit_record(SOURCE,plan,fit);records.append(artifact(path))
    payload=dict(schema="ordersmc-reporting-repair-v1",source_plan=artifact(SOURCE/"plan.json"),
        fitted_artifacts=records,cohorts=sorted({d["cohort"] for d in plan["datasets"]}),
        model_changes=False,posterior_changes=False,objective_scale_exponent=20,
        audit_tolerance_unchanged=1e-9,automatic_default_promotion=False)
    snapshot(root,payload)


def preflight(root):
    plan=catchup.checked_plan(root);source,original=source_plan(plan)
    for record in plan["fitted_artifacts"]:checked_input(record)
    for d in original["datasets"]:checked_input(d["candidate"])
    for c in original["comparisons"]:checked_input(c["original"])
    _atomic_json(root/"validation/preflight.json",dict(all_passed=True,plan_sha256=plan["plan_sha256"],
        fits=len(plan["fitted_artifacts"]),source_plan_sha256=original["plan_sha256"]))


def run(root,index):
    plan=catchup.checked_plan(root);source,original=source_plan(plan)
    acceptance=read(root/"validation/preflight.json")
    if not acceptance["all_passed"] or acceptance["plan_sha256"]!=plan["plan_sha256"]:
        raise RuntimeError("missing reporting preflight")
    cohort=plan["cohorts"][index]
    selected={f["fit_id"] for f in original["fits"] if f["cohort"]==cohort and "external" not in f}
    for record in plan["fitted_artifacts"]:
        if Path(record["path"]).parent.name in selected or cohort=="metadata":checked_input(record)
    destination=root/"analysis"/cohort
    catchup.aggregate_saved_fits(source,cohort,original,destination)
    old=source/"analysis"/cohort/"per_run_metrics.csv"
    if old.exists():
        before=pd.read_csv(old);after=pd.read_csv(destination/"per_run_metrics.csv")
        keys=["dataset_key","method","budget","final_target_steps","seed_index"]
        a=before.set_index(keys).sort_index();b=after.set_index(keys).sort_index()
        if not a.index.equals(b.index):raise RuntimeError("repair changed fit inventory")
        delta={m:float((a[m]-b[m]).abs().max()) for m in catchup.METRICS}
        _atomic_json(destination/"reporting_metric_changes.json",dict(maximum_absolute_change=delta,
            prior_report_sha256=sha(old),old_results_preserved=True))
    _atomic_json(destination/"reporting_edition.json",dict(reporting_plan_sha256=plan["plan_sha256"],
        original_plan_sha256=original["plan_sha256"],posterior_fits_unchanged=True,objective_scale_exponent=20))


def complete(root):
    plan=catchup.checked_plan(root);source,_=source_plan(plan)
    for record in plan["fitted_artifacts"]:checked_input(record)
    for cohort in plan["cohorts"]:
        folder=root/"analysis"/cohort
        edition=read(folder/"reporting_edition.json")
        if edition["reporting_plan_sha256"]!=plan["plan_sha256"]:raise RuntimeError("stale report")
        if read(folder/"completion.json")["status"]!="complete":raise RuntimeError("incomplete report")
    _atomic_json(root/"analysis/completion.json",dict(status="complete",plan_sha256=plan["plan_sha256"],
        cohorts=plan["cohorts"],fitted_artifacts_unchanged=len(plan["fitted_artifacts"]),
        source_root=str(source),publication_ready=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["freeze","preflight","run","complete"])
    p.add_argument("--output",type=Path,required=True);p.add_argument("--task",type=int)
    a=p.parse_args()
    if a.action=="run":run(a.output,a.task)
    else:globals()[a.action](a.output)


if __name__=="__main__":main()
