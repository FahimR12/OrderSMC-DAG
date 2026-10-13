"""Frozen independent replication of the dimension-prior diagnostic.

Generate three new null and three new signal datasets; never select replicates
by recovery or fidelity. Use the existing null_pilot worker/report contract.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ordersmc import OrderSMCConfig, build_candidates, validate_candidate_provenance
from ordersmc.persistence import save_candidate_table, load_candidate_table, candidate_table_content_sha256
from . import audit_submit, catchup, null_pilot, reporting_repair
from .execution import _atomic_json
from .metadata_profiles import (
    MetadataDAGConfig, generate_metadata_dag_dataset, write_metadata_dag_artifact, load_metadata_dag_artifact,
)
from .prior_calibration import retarget_edge_count_prior
from .readiness_audits import read
from .sampler_corrections import artifact, checked_input
from .seeds import protocol_seed_record

REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "results/reporting-null-audit/v1/null_pilot"
MASTER_SEED = 2026092102


def heldout_generator_config(original, replicate):
    return dict(original, master_seed=MASTER_SEED, experiment_id="prior-independent-validation",
                replicate_id=replicate)


def prepare(root):
    if root.exists(): raise FileExistsError("independent replication requires a new root")
    original = catchup.checked_source(PILOT / "plan.json")
    cases = []
    for kind in ("metadata-null", "metadata-signal"):
        source = next(c for c in original["cases"] if c["dataset"]["kind"] == kind and c["arm"] == "fixed")
        manifest = next(a for a in source["dataset"]["source_inputs"] if Path(a["path"]).name == "generator_manifest.json")
        config = read(checked_input(manifest))["config"]
        for replicate in range(100, 103):
            cases.append(dict(key=f"{kind}-heldout-r{replicate}", kind=kind, replicate=replicate,
                generator_config=heldout_generator_config(config, replicate),
                source_manifest=manifest, score_config=source["config"]))
    reporting_repair.snapshot(root, dict(schema="ordersmc-prior-independent-validation-v1", cases=cases,
        source_plan=artifact(PILOT / "plan.json"),
        design="three new null + three new signal datasets; 196 variables; n=222; two priors; four paired seeds",
        prior_rule="fixed lambda=1 versus unchanged lambda(d)=log(d-1)",
        generator_policy="unchanged metadata generator/calibration settings; new seed namespace and replicates",
        selection_policy="include every planned dataset regardless of recovery or metadata fidelity",
        confirmatory=False, model_default_changed=False,
        inference=dict(particles=1024, mpi_ranks=64, final_target_snapshot_steps=[0, 250]),
        limitations="small independent replication, not final publication evidence or a new-prior competitor ranking"))


def preflight(root):
    plan = catchup.checked_plan(root); checked_input(plan["source_plan"])
    for case in plan["cases"]:
        checked_input(case["source_manifest"])
        if case["generator_config"]["master_seed"] == read(checked_input(case["source_manifest"]))["config"]["master_seed"]:
            raise RuntimeError("held-out generator reused development seed")
    _atomic_json(root / "validation/preflight.json", dict(all_passed=True, plan_sha256=plan["plan_sha256"]))


def run(root, index):
    plan = catchup.checked_plan(root); spec = plan["cases"][index]
    if read(root / "validation/preflight.json") != dict(all_passed=True, plan_sha256=plan["plan_sha256"]):
        raise RuntimeError("preflight failed")
    record_path = root / "prepared" / (spec["key"] + ".json")
    if record_path.exists():
        record = read(record_path)
        if record["plan_sha256"] != plan["plan_sha256"]: raise RuntimeError("stale prepared dataset")
        for c in record["cases"]: checked_input(c["candidate"])
        return
    destination = root / "datasets" / spec["key"]
    if destination.exists():
        data = load_metadata_dag_artifact(destination)
        if data.manifest["config"] != spec["generator_config"]: raise RuntimeError("generator config mismatch")
    else:
        data = generate_metadata_dag_dataset(MetadataDAGConfig(**spec["generator_config"]))
        write_metadata_dag_artifact(data, destination)
    # Re-read saved binary/CSV inputs before scoring, for reproducible reruns.
    data = load_metadata_dag_artifact(destination)
    cfg = OrderSMCConfig.from_mapping(spec["score_config"])
    path = destination / "candidate.json.gz"
    if path.exists():
        table = load_candidate_table(path)
        validate_candidate_provenance(table, config=cfg, missing_policy="require")
    else:
        table = build_candidates(data.train, data.variable_types, data.cardinalities, config=cfg, verbose=False)
        save_candidate_table(table, path)
    dataset = dict(key=spec["key"], kind=spec["kind"], replicate=spec["replicate"],
        nodes=table.nodes, truth_edges=sorted(map(list, data.dag.edges())), mpi_ranks=64,
        candidate=artifact(path), candidate_sha256=candidate_table_content_sha256(table),
        source_inputs=[artifact(destination / name) for name in ("generator_manifest.json", "train.csv")])
    cases = []
    for arm, penalty in (("fixed", 1.), ("dimension", null_pilot.dimension_penalty(len(table.nodes)))):
        model = cfg.with_overrides(lam=0., edge_penalty=penalty)
        retargeted, audit = retarget_edge_count_prior(table, target_label="order", lambda_target=penalty, target_config=model)
        target_path = save_candidate_table(retargeted, destination / f"candidate-{arm}.json.gz")
        cases.append(dict(case_id=spec["key"] + "--" + arm, dataset=dataset, arm=arm, penalty=penalty,
            config=model.to_dict(), candidate=artifact(target_path),
            candidate_sha256=candidate_table_content_sha256(retargeted), prior_audit=audit.to_dict()))
    _atomic_json(record_path, dict(status="complete", plan_sha256=plan["plan_sha256"], cases=cases,
        metadata_calibration=data.manifest["calibration"], no_recovery_based_selection=True))
    print("prepared", spec["key"], flush=True)


def aggregate(root):
    """Finish preparation and submit the preflight-gated inference workflow."""
    plan = catchup.checked_plan(root); cases, tasks = [], []
    for spec in plan["cases"]:
        record = read(root / "prepared" / (spec["key"] + ".json"))
        if record["status"] != "complete" or record["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("not all independent datasets prepared")
        for c in record["cases"]:
            checked_input(c["candidate"]); cases.append(c)
            for seed_index in range(4):
                seed = asdict(protocol_seed_record(MASTER_SEED, experiment="prior-independent-validation",
                    cell=spec["key"], replicate=spec["replicate"], component=f"inference-{seed_index}"))
                tasks.append(dict(task_id=c["case_id"] + f"--s{seed_index}", case_id=c["case_id"],
                    seed_index=seed_index, seed_record=seed, mpi_ranks=64))
    if len(cases) != 12 or len(tasks) != 48: raise RuntimeError("held-out inventory mismatch")
    fits = root / "fits"
    if not (fits / "plan.json").exists():
        reporting_repair.snapshot(fits, dict(schema="ordersmc-null-prior-independent-v1", cases=cases, tasks=tasks,
            source_plan=artifact(root / "plan.json"), seeds=4, model_default_changed=False, confirmatory=False,
            selection="all six independent datasets", prior_rule=plan["prior_rule"],
            support_policy="fixed within-dataset support, bit-identical likelihoods across priors",
            purpose="independent small replication of fixed prior rule; no automatic promotion"))
    else:
        existing = catchup.checked_plan(fits)
        if existing["cases"] != cases or existing["tasks"] != tasks: raise RuntimeError("stale held-out fit plan")
    submitted = audit_submit.submit(fits, "null_pilot", "start")
    _atomic_json(root / "analysis/preparation_completion.json", dict(status="prepared_and_submitted",
        datasets=6, planned_fits=48, jobs=submitted["jobs"], results=str(fits / "analysis"),
        scientific_results_complete=False, plan_sha256=plan["plan_sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "preflight", "run", "aggregate"])
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.action == "run": run(args.output, args.task)
    else: globals()[args.action](args.output)


if __name__ == "__main__": main()
