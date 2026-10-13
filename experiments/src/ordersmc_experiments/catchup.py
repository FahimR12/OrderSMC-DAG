"""Immutable, same-data v0.7 catch-up fits and complete-cohort reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import shutil
import time

import networkx as nx
import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig, sample, validate_candidate_provenance
from ordersmc.persistence import load_candidate_table
from .execution import _all_edges, _atomic_json, _json_safe, _profile_config
from .readiness import canonical_sha, root_call, score_config, summarize_posterior
from .readiness_audits import read, sha, pips, rmse
from .sampler_corrections import artifact, checked_input
from .rank_budget import complete_probabilities
from .accuracy_gate import dataset_rmse_gate

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "experiments/configs/catchup_v07_v1.json"
METRICS = ("directed_f1", "skeleton_f1", "directed_auprc", "skeleton_auprc",
           "directed_brier", "skeleton_brier", "posterior_expected_pair_state_shd",
           "own_target_pip_rmse")


def checked_plan(root, sources=True):
    plan = read(root / "plan.json")
    if canonical_sha({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise RuntimeError("catch-up plan hash mismatch")
    if sources:
        for relative, digest in plan["source_files"].items():
            if sha(REPO / relative) != digest:
                raise RuntimeError(f"run from the frozen source: {relative}")
    return plan


def checked_source(path, key="plan_sha256"):
    value = read(path)
    if canonical_sha({k: v for k, v in value.items() if k != key}) != value[key]:
        raise RuntimeError(f"source plan identity mismatch: {path}")
    return value


def locator(path, *keys):
    return {**artifact(path), "keys": list(keys)}


def located(record):
    value = read(checked_input(record))
    for key in record.get("keys", []):
        value = value[key]
    return value


def corrected_config(base, config, metadata=False):
    return base.with_overrides(**config["bridge"], final_target_snapshot_steps=tuple(
        config["metadata_snapshot_steps"] if metadata else [config["primary_snapshot_steps"]]))


def freeze(output, config_path=DEFAULT_CONFIG):
    if output.exists():
        raise FileExistsError("use a new catch-up output root; old artifacts are never overwritten")
    config = read(config_path)
    datasets, fits, comparisons, evidence = [], [], [], []
    pilot_root = REPO / config["correction_pilot"]
    pilot = checked_source(pilot_root / "plan.json")
    if pilot["plan_sha256"] != config["correction_pilot_plan_sha256"]:
        raise RuntimeError("unexpected correction pilot")
    pilot_tasks = {(t["dataset_id"], t["seed_index"]): t for t in pilot["tasks"]
                   if t["arm"] == "geometric_adaptive"}
    execution_configs = {canonical_sha(read(p)): p for p in (REPO / "experiments/configs").glob("*.json")}

    def dataset_record(cohort, label, spec, candidate, digest, truth, extra_inputs=()):
        table = load_candidate_table(candidate, expected_sha256=digest)
        base = OrderSMCConfig.from_mapping(table.provenance["candidate_build_config"])
        # Fail closed on K=2 categorical or ambiguous old tables: such a cohort
        # needs rebuilding and ALL shared-score methods, not a sampler-only rerun.
        validate_candidate_provenance(table, config=base, missing_policy="require")
        key = f"{cohort}--{label}"
        d = dict(key=key, cohort=cohort, dataset_id=label, spec=spec,
                 candidate=artifact(candidate), candidate_sha256=digest,
                 nodes=table.nodes, max_parents=base.max_parents, truth_edges=truth,
                 score=base.score_method, inputs=[artifact(p) for p in extra_inputs],
                 references={})
        datasets.append(d)
        return d, table, base

    def add_fit(d, old, original, cfg, ranks, index, budget="baseline", role="evaluation", external=None):
        seed = old.get("seed", old.get("run", {}).get("method_seed"))
        if not isinstance(seed, dict) or cfg.seed != seed["seed"]:
            raise RuntimeError("inference seed mismatch")
        fit = dict(fit_id=f"{d['key']}--{budget}--s{index:02d}", dataset_key=d["key"],
                   cohort=d["cohort"], budget=budget, role=role, seed_index=index,
                   seed_record=seed, config=cfg.to_dict(), mpi_ranks=ranks, original=original)
        if external:
            fit["external"] = external
        fits.append(fit)

    for cohort, relative in config["source_cohorts"].items():
        source = REPO / relative
        plan = checked_source(source / "plan.json")
        evidence.append(artifact(source / "plan.json"))
        completion = read(source / "analysis/completion.json")
        if completion.get("status") != "complete" or completion.get("missing_or_failed_runs", 0):
            raise RuntimeError(f"incomplete original cohort: {cohort}")
        for spec in plan["datasets"]:
            if cohort == "v3_mixed" and spec.get("data_kind") == "native_mixed_conditional_gaussian":
                continue
            folder = source / "datasets" / spec["dataset_id"]
            prepared = read(folder / "prepared.json")
            manifest = read(folder / "generator_manifest.json")
            d, table, base = dataset_record(cohort, spec["dataset_id"], spec,
                folder / "candidate.json.gz", prepared["candidate_table_sha256"], manifest["edges"],
                [folder / x for x in ("prepared.json", "generator_manifest.json", "train.csv")])
            for r in plan["runs"]:
                if r["dataset_id"] != spec["dataset_id"]:
                    continue
                path = source / "runs" / f"{r['run_id']}.json"
                old = read(path)
                if (old.get("status") != "complete" or old["plan_sha256"] != plan["plan_sha256"]
                        or old["candidate_table_sha256"] != d["candidate_sha256"] or old["run"] != r):
                    raise RuntimeError(f"source fit identity mismatch: {path}")
                if r["method"] == "exact-order-dp":
                    d["references"]["exact-order"] = locator(path)
                if r["method"] != "ordersmc":
                    comparisons.append(dict(dataset_key=d["key"], method=r["method"], budget="baseline",
                        role=r["role"], seed_index=r["seed_index"], target=r["target"], original=locator(path)))
                    continue
                sampling = old["diagnostics"].get("sampling_config")
                if sampling is None:
                    cfgpath = execution_configs.get(old["execution_config_sha256"])
                    if cfgpath is None:
                        raise RuntimeError("cannot reconstruct the original SMC budget")
                    cfg = _profile_config(spec, read(cfgpath), old["profile"], r["method_seed"]["seed"])
                    evidence.append(artifact(cfgpath))
                else:
                    cfg = OrderSMCConfig.from_mapping({**base.to_dict(), **sampling})
                cfg = corrected_config(cfg, config, cohort == "metadata")
                validate_candidate_provenance(table, config=cfg, missing_policy="require")
                external = None
                pt = pilot_tasks.get((spec["dataset_id"], r["seed_index"])) if cohort == "metadata" else None
                if pt is not None:
                    if canonical_sha(pt["config"]) != canonical_sha(cfg.to_dict()) or pt["seed_record"] != r["method_seed"]:
                        raise RuntimeError("pilot configuration differs; cannot reuse its fit")
                    if pilot["inputs"][spec["dataset_id"]]["candidate_table_sha256"] != d["candidate_sha256"]:
                        raise RuntimeError("pilot target differs")
                    external = dict(root=str(pilot_root.resolve()), plan_sha256=pilot["plan_sha256"], task=pt)
                add_fit(d, old, locator(path), cfg, int(r["mpi_ranks"]), r["seed_index"],
                        role=r["role"], external=external)
        print(f"Inventoried {cohort}", flush=True)

    rank_root = REPO / config["rank_validation"]
    suite = checked_source(rank_root / "suite.json", "suite_sha256")
    evidence.append(artifact(rank_root / "suite.json"))
    # Retain every fresh native/rank and paired measurement condition, including
    # algebraically redundant rank transforms; these are not independent data.
    for arm in sorted(rank_root.glob("fresh-*/plan.json")):
        plan = checked_source(arm)
        evidence.append(artifact(arm))
        for path in sorted((arm.parent / "cases").glob("rep*/*.json")):
            if path.name == "generator_manifest.json":
                continue
            case = read(path)
            if case.get("status") != "complete" or case["plan_sha256"] != plan["plan_sha256"]:
                raise RuntimeError(f"invalid rank case: {path}")
            label = f"{arm.parent.name}-{path.parent.name}-{path.stem}"
            spec = dict(n_train=case["n_train"], replicate=case["replicate"],
                        noise=arm.parent.name.removeprefix("fresh-").removesuffix("-rank-native").removesuffix("-native"),
                        variant=case["variant"], cell_id=f"{arm.parent.name}-{path.stem}")
            d, table, base = dataset_record("rank_validation", label, spec,
                path.with_name(path.stem + "-candidate.json.gz"), case["candidate_sha256"],
                case["truth_edges"], [path.parent / f"{case['variant']}.csv", path])
            d["references"] = {key: locator(path, "references", key) for key in case["references"]}
            for i, old in enumerate(case["smc_runs"]):
                cfg = corrected_config(score_config(plan["config"], old["seed"]["seed"]), config)
                validate_candidate_provenance(table, config=cfg, missing_policy="require")
                add_fit(d, old, locator(path, "smc_runs", i), cfg, case["mpi_ranks"], old["seed_index"])
    evidence.append(artifact(rank_root / "analysis/artifact_hashes.json"))
    print("Inventoried native/rank score comparison", flush=True)

    budget_root = REPO / config["rank_budget"]
    budget_plan = checked_source(budget_root / "plan.json")
    evidence.append(artifact(budget_root / "plan.json"))
    for spec in budget_plan["datasets"]:
        folder = budget_root / "datasets" / spec["dataset_id"]
        path = folder / "reference.json"
        case = read(path)
        d, table, base = dataset_record("rank_budget", spec["dataset_id"],
            {**spec, "cell_id": f"{spec['noise']}-n{spec['n_train']}"}, folder / "candidate.json.gz",
            spec["candidate_sha256"], case["truth_edges"], [path, folder / "data.csv"])
        d["references"] = {key: locator(path, "references", key) for key in case["references"]}
        for budget, index in itertools.product(budget_plan["config"]["budgets"], range(budget_plan["config"]["method_seeds"])):
            path = budget_root / "runs" / f"{spec['dataset_id']}-{budget}-s{index:02d}.json"
            old = read(path)
            if old.get("status") != "complete" or old["plan_sha256"] != budget_plan["plan_sha256"] or old["candidate_sha256"] != d["candidate_sha256"]:
                raise RuntimeError("invalid old budget fit")
            cfg = corrected_config(OrderSMCConfig.from_mapping(old["config"]), config)
            validate_candidate_provenance(table, config=cfg, missing_policy="require")
            add_fit(d, old, locator(path), cfg, old["mpi_ranks"], index, budget=budget)

    competitor_root = REPO / config["rank_competitors"]
    competitor_plan = checked_source(competitor_root / "plan.json")
    evidence.append(artifact(competitor_root / "plan.json"))
    for spec in competitor_plan["datasets"]:
        if spec["n_train"] == 5000:
            key = f"rank_budget--{spec['noise'].replace('_', '-')}-rep{spec['replicate']:03d}"
        else:
            key = (f"rank_validation--fresh-{spec['noise']}-rank-native-rep{spec['replicate']:03d}"
                   f"-unrestricted_native-n{spec['n_train']:05d}")
        d = next(x for x in datasets if x["key"] == key)
        if d["candidate_sha256"] != spec["candidate_sha256"]:
            raise RuntimeError("competitors do not share the corrected SMC target")
        for path in sorted((competitor_root / "runs").glob(f"{spec['dataset_id']}-*.json")):
            old = read(path)
            if old.get("status") != "complete" or old["plan_sha256"] != competitor_plan["plan_sha256"] or old["candidate_sha256"] != d["candidate_sha256"]:
                raise RuntimeError("invalid completed competitor fit")
            comparisons.append(dict(dataset_key=key, method=old["method"], budget=old["budget"], role="evaluation",
                seed_index=old["seed_index"], original=locator(path),
                target="structure-modular" if old["method"] in {"rev-mcmc", "partition-mcmc"} else "order-modular"))
    print("Inventoried budget study and matched competitors", flush=True)

    groups = defaultdict(list)
    for fit in fits:
        if "external" not in fit:
            groups[fit["cohort"], fit["dataset_key"], fit["budget"], fit["mpi_ranks"]].append(fit["fit_id"])
    tasks = []
    for (cohort, key, budget, ranks), ids in groups.items():
        batch = 1 if cohort == "metadata" else config["seeds_per_task"]
        for start in range(0, len(ids), batch):
            tasks.append(dict(cohort=cohort, mpi_ranks=ranks, fit_ids=ids[start:start + batch]))
    files = {str(p.relative_to(REPO)): sha(p)
             for folder in ("src", "experiments/src", "experiments/configs", "experiments/adapters",
                            "experiments/fixtures", "experiments/validation", "scripts", "tests", "experiments/tests", "docs",
                            "summary_statistics_multiopmic_data_pathway_molecule_level")
             for p in sorted((REPO / folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    for name in ("README.md", "pyproject.toml"):
        files[name] = sha(REPO / name)
    plan = dict(schema="ordersmc-v07-catchup-plan-v1", config=config, source_files=files,
                evidence=list({r["path"]: r for r in evidence}.values()), datasets=datasets, fits=fits,
                comparisons=comparisons, tasks=tasks)
    plan["plan_sha256"] = canonical_sha(plan)
    for relative, digest in files.items():
        dest = output / "code_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)
        if sha(dest) != digest:
            raise RuntimeError("source changed while freezing")
    _atomic_json(output / "plan.json", plan)
    write_inventory(output, plan)
    shutil.copy2(REPO / "docs/CATCHUP_V07.md", output / "README.md")
    return plan


def write_inventory(root, plan):
    """Derived, non-scientific counts; do not confuse task grouping tuples with cohorts."""
    counts = [dict(cohort=c, datasets=sum(d["cohort"] == c for d in plan["datasets"]),
                   corrected_fits=sum(f["cohort"] == c for f in plan["fits"]),
                   new_fits=sum(f["cohort"] == c and "external" not in f for f in plan["fits"]),
                   array_tasks=sum(t["cohort"] == c for t in plan["tasks"]))
              for c in sorted({d["cohort"] for d in plan["datasets"]})]
    _atomic_json(root / "inventory.json", dict(plan_sha256=plan["plan_sha256"], cohorts=counts,
        reused_comparator_or_reference_records=len(plan["comparisons"]),
        status="frozen-work-inventory-see-submission-json-for-live-status"))


def fit_record(root, plan, fit):
    if "external" in fit:
        ext = fit["external"]
        path = Path(ext["root"]) / "runs" / ext["task"]["task_id"] / "result.json"
        value = read(path)
        if value.get("status") != "complete" or value["plan_sha256"] != ext["plan_sha256"] or value["task"] != ext["task"]:
            raise RuntimeError("reused pilot fit identity mismatch")
    else:
        path = root / "runs" / fit["fit_id"] / "result.json"
        value = read(path)
        if value.get("status") != "complete" or value["plan_sha256"] != plan["plan_sha256"] or value["fit"] != fit:
            raise RuntimeError(f"catch-up fit identity mismatch: {path}")
    d = next(d for d in plan["datasets"] if d["key"] == fit["dataset_key"])
    if value["candidate_table_sha256"] != d["candidate_sha256"]:
        raise RuntimeError("result used a different target")
    if {s["final_target_steps"] for s in value["snapshots"]} != set(fit["config"]["final_target_snapshot_steps"]):
        raise RuntimeError("missing or extra posterior snapshot")
    return value, path


def run(root, index, comm=None):
    plan = root_call(comm, lambda: checked_plan(root))
    task = plan["tasks"][index]
    if (comm.size if comm is not None else 1) != task["mpi_ranks"]:
        raise RuntimeError("incorrect MPI rank count")
    acceptance = root_call(comm, lambda: read(root / "validation/preflight.json"))
    if acceptance.get("plan_sha256") != plan["plan_sha256"] or not acceptance.get("all_passed"):
        raise RuntimeError("preflight absent, failed or stale")
    for fit_id in task["fit_ids"]:
        fit = next(f for f in plan["fits"] if f["fit_id"] == fit_id)
        dest = root / "runs" / fit_id
        def existing():
            if not (dest / "result.json").exists():
                return False
            fit_record(root, plan, fit)
            return True
        if root_call(comm, existing):
            continue
        d = next(d for d in plan["datasets"] if d["key"] == fit["dataset_key"])
        def prepare():
            for record in d["inputs"]:
                checked_input(record)
            checked_input(fit["original"])
            return load_candidate_table(checked_input(d["candidate"]), expected_sha256=d["candidate_sha256"])
        table = root_call(comm, prepare)
        cfg = OrderSMCConfig.from_mapping(fit["config"])
        started = time.perf_counter()
        result = sample(table, config=cfg, backend="mpi" if comm is not None else "local", comm=comm,
                        parent_seed_record=fit["seed_record"] if comm is not None else None,
                        provenance_policy="require", verbose=False)
        def save():
            local = result.to_local_result() if comm is not None else result
            truth = nx.DiGraph(); truth.add_nodes_from(d["nodes"]); truth.add_edges_from(d["truth_edges"])
            exact = pips(located(d["references"]["exact-order"])) if "exact-order" in d["references"] else None
            snapshots = []
            for s in local.final_target_snapshots:
                probabilities = complete_probabilities(s.edge_probs, d["nodes"])
                row = summarize_posterior(probabilities, truth, d["nodes"], d["max_parents"])
                row.update(final_target_steps=s.final_target_mh_sweeps,
                           own_target_pip_rmse=rmse(probabilities, exact) if exact is not None else None)
                snapshots.append(row)
            local.save(dest)
            _atomic_json(dest / "result.json", dict(status="complete", plan_sha256=plan["plan_sha256"],
                fit=fit, candidate_table_sha256=d["candidate_sha256"], snapshots=snapshots,
                runtime_seconds=time.perf_counter()-started, summary=local.summary(),
                history=local.history, run_info=local.run_info))
            print(f"complete: {fit_id}", flush=True)
        root_call(comm, save)


def preflight(root):
    from .exact_order import exact_order_dp
    from .readiness_audits import rmse
    plan = checked_plan(root)
    for record in plan["evidence"]:
        checked_input(record)
    seen = set()
    for d in plan["datasets"]:
        table = load_candidate_table(checked_input(d["candidate"]), expected_sha256=d["candidate_sha256"])
        base = OrderSMCConfig.from_mapping(table.provenance["candidate_build_config"])
        validate_candidate_provenance(table, config=base, missing_policy="require")
        for record in d["inputs"]:
            checked_input(record)
        # Recheck an exact reference for each scoring/dimension stratum without
        # redoing unchanged d=12 dynamic programs for all datasets.
        stratum = (d["cohort"], len(table.nodes), base.score_method)
        if stratum not in seen and "exact-order" in d["references"]:
            reference = exact_order_dp(table, max_nodes=12, compute_graph_size=False)
            if rmse(reference.edge_probs, pips(located(d["references"]["exact-order"]))) > 1e-8:
                raise RuntimeError("stored exact posterior differs from the reused target")
            seen.add(stratum)
    for row in plan["comparisons"]:
        checked_input(row["original"])
    _atomic_json(root / "validation/preflight.json", dict(plan_sha256=plan["plan_sha256"],
        all_passed=True, datasets=len(plan["datasets"]), exact_strata_checked=[list(x) for x in sorted(seen)],
        note="Artifact/target integrity and exact-reference checks, not a convergence certificate."))


def aggregate(root, cohort):
    plan = checked_plan(root)
    return aggregate_saved_fits(root, cohort, plan, root / "analysis" / cohort)


def aggregate_saved_fits(root, cohort, plan, analysis):
    """Report verified records into a caller-selected edition.

    The caller must validate the source plan and reporting source identity.
    This separation permits numerical reporting repairs without editing the
    immutable source freeze or resampling any fitted posterior.
    """
    selected = [f for f in plan["fits"] if f["cohort"] == cohort]
    missing, hashes = [], {}
    # Check completeness before loading large posterior records into summaries.
    for fit in selected:
        try:
            _, path = fit_record(root, plan, fit)
            hashes[str(path)] = sha(path)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            missing.append(dict(fit_id=fit["fit_id"], reason=str(error)))
    _atomic_json(analysis / "completion.json", dict(plan_sha256=plan["plan_sha256"],
        status="incomplete" if missing else "complete", planned=len(selected), completed=len(selected)-len(missing), missing=missing))
    if missing:
        raise RuntimeError(f"{len(missing)} absent/invalid fits; no partial-cohort scientific summary")
    rows, stable, pooled, decisions = [], [], [], []
    for d in [d for d in plan["datasets"] if d["cohort"] == cohort]:
        truth = nx.DiGraph(); truth.add_nodes_from(d["nodes"]); truth.add_edges_from(d["truth_edges"])
        groups = defaultdict(list)
        common = dict(dataset_key=d["key"], dataset_id=d["dataset_id"], cohort=cohort,
                      cell_id=d["spec"].get("cell_id", d["dataset_id"]), score=d["score"],
                      n_train=d["spec"].get("n_train"), replicate=d["spec"].get("replicate"))
        for fit in [f for f in selected if f["dataset_key"] == d["key"]]:
            value, _ = fit_record(root, plan, fit)
            for snapshot in value["snapshots"]:
                groups["ordersmc-v07", fit["budget"], snapshot["final_target_steps"]].append(
                    (fit["seed_index"], fit["role"], snapshot))
            groups["ordersmc-original", fit["budget"], 0].append(
                (fit["seed_index"], fit["role"], located(fit["original"])))
        for record in [r for r in plan["comparisons"] if r["dataset_key"] == d["key"]]:
            old = located(record["original"])
            hashes[record["original"]["path"]] = record["original"]["sha256"]
            groups[record["method"], record["budget"], 0].append((record["seed_index"], record["role"], old))
        for (method, budget, steps), records in groups.items():
            maps = [complete_probabilities(pips(r), d["nodes"]) for _, _, r in records]
            summaries = [summarize_posterior(m, truth, d["nodes"], d["max_parents"]) for m in maps]
            target = ("point-graph-search" if method.startswith("bnlearn") else
                      "structure-modular" if method in {"partition-mcmc", "rev-mcmc"} else "order-modular")
            exact_key = "exact-structure" if target == "structure-modular" else "exact-order"
            exact = pips(located(d["references"][exact_key])) if exact_key in d["references"] and not method.startswith("bnlearn") else None
            for (index, role, record), m, summary in zip(records, maps, summaries):
                decisions.append(dict(dataset_key=d["key"], method=method, budget=budget,
                    final_target_steps=steps, seed_index=index, kind="individual",
                    edges=summary["representative_edges"], diagnostics=summary["representative_graph_diagnostics"]))
                rows.append(dict(**common, method=method, budget=budget, final_target_steps=steps,
                    seed_index=index, role=role, target=target,
                    **{k: summary.get(k) for k in METRICS if k != "own_target_pip_rmse"},
                    own_target_pip_rmse=rmse(m, exact) if exact is not None else None))
            # All seeds enter reproducibility. Only evaluation seeds enter the
            # paper's mean-PIP graph; M1 extra seeds are never silently pooled.
            included = [i for i, (_, role, _) in enumerate(records) if role != "reproducibility-extension"]
            if included:
                mean = {e: float(np.mean([maps[i][e] for i in included])) for e in maps[0]}
                summary = summarize_posterior(mean, truth, d["nodes"], d["max_parents"])
                decisions.append(dict(dataset_key=d["key"], method=method, budget=budget,
                    final_target_steps=steps, kind="mean_pip", evaluation_seeds=len(included),
                    edges=summary["representative_edges"], diagnostics=summary["representative_graph_diagnostics"]))
                summary["own_target_pip_rmse"] = rmse(mean, exact) if exact is not None else None
                pooled.append(dict(**common, method=method, budget=budget, final_target_steps=steps,
                    evaluation_seeds=len(included), **{k: summary.get(k) for k in METRICS}))
            if len(records) > 1:
                edges = [set(map(tuple, r["representative_edges"])) for r in summaries]
                skeletons = [{tuple(sorted(e)) for e in row} for row in edges]
                def jaccard(sets):
                    return float(np.mean([len(a & b)/len(a | b) if a | b else 1. for a, b in itertools.combinations(sets, 2)]))
                vectors = np.array([[m[e] for e in maps[0]] for m in maps])
                stable.append(dict(**common, method=method, budget=budget, final_target_steps=steps,
                    seeds=len(records), mean_directed_jaccard=jaccard(edges), mean_skeleton_jaccard=jaccard(skeletons),
                    all_directed_graphs_identical=all(e == edges[0] for e in edges),
                    mean_pairwise_pip_rmse=float(np.mean([np.sqrt(np.mean((a-b)**2)) for a,b in itertools.combinations(vectors,2)])),
                    maximum_edge_seed_sd=float(np.std(vectors, axis=0, ddof=1).max()),
                    maximum_edge_seed_mean_se=float(np.std(vectors, axis=0, ddof=1).max()/np.sqrt(len(records)))))
    pd.DataFrame(rows).to_csv(analysis / "per_run_metrics.csv", index=False)
    pd.DataFrame(pooled).to_csv(analysis / "mean_pip_graph_metrics.csv", index=False)
    pd.DataFrame(stable).to_csv(analysis / "seed_stability.csv", index=False)
    frame = pd.DataFrame(rows)
    ds = frame[frame.role != "reproducibility-extension"].groupby(
        ["dataset_key", "cell_id", "method", "budget", "final_target_steps"], as_index=False)[list(METRICS)].mean()
    ds.to_csv(analysis / "dataset_mean_metrics.csv", index=False)
    ds.groupby(["cell_id", "method", "budget", "final_target_steps"], as_index=False)[list(METRICS)].mean().to_csv(analysis / "cell_summary.csv", index=False)
    _atomic_json(analysis / "exact_rmse_flags.json", dataset_rmse_gate(frame, plan["config"]["exact_rmse_flags"]))
    _atomic_json(analysis / "artifact_hashes.json", hashes)
    _atomic_json(analysis / "representative_graphs.json", dict(decisions=decisions))
    _atomic_json(analysis / "interpretation.json", dict(primary_snapshot_steps=0, convergence_established=False,
        seed_sd_is_not_bias_estimate=True, historical_comparators_reused=True, equal_compute=False,
        metadata_evaluation_seeds="0-3; seeds 4-7 are reproducibility extension only",
        prior_scores_data_and_particle_budgets_unchanged=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "preflight", "run", "aggregate"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--task", type=int)
    p.add_argument("--cohort")
    p.add_argument("--mpi", action="store_true")
    args = p.parse_args()
    if args.action == "freeze":
        plan = freeze(args.output)
        print(json.dumps(dict(fits=len(plan["fits"]), tasks=len(plan["tasks"]), plan_sha256=plan["plan_sha256"])))
    elif args.action == "preflight":
        preflight(args.output)
    elif args.action == "aggregate":
        aggregate(args.output, args.cohort)
    else:
        comm = None
        if args.mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        run(args.output, args.task, comm)


if __name__ == "__main__":
    main()
