"""Frozen, resource-matched posterior-accuracy/timing pilot (not confirmatory)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

from ordersmc import OrderSMCConfig, build_candidates, sample, validate_candidate_provenance
from ordersmc.core import edge_probs_for_order, order_log_target
from ordersmc.persistence import load_candidate_table, save_candidate_table, candidate_table_content_sha256
from . import catchup, reporting_repair
from .exact_order import exact_order_dp
from .execution import _atomic_json
from .order_mcmc import OrderMCMCConfig, order_mcmc
from .prior_calibration import retarget_edge_count_prior, candidate_support_sha256
from .readiness import root_call, summarize_posterior
from .readiness_audits import pips, read, sha, rmse
from .sampler_corrections import artifact, checked_input
from .seeds import protocol_seed_record

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "experiments/configs/timing_pilot_v1.json"


def seed_record(config, dataset, seed_index, component):
    return asdict(protocol_seed_record(config["master_seed"], experiment="timing-pilot-v1",
        cell=dataset, replicate=seed_index, component=component))


def hardware():
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown")
    return dict(host=socket.gethostname(), cpu_model=model, platform=platform.platform(),
        python=sys.version, numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0)),
        slurm_job=os.environ.get("SLURM_JOB_ID"), allocated_cpus=os.environ.get("SLURM_CPUS_ON_NODE"),
        physical_cpu_count=os.cpu_count(), load_average=list(os.getloadavg()))


def make_tasks(config, datasets):
    if config.get("task_design") == "scaling-quality-v1":
        from .scaling_quality import make_tasks as quality_tasks
        return quality_tasks(config, datasets)
    tasks, bundles, references = [], [], []
    for d in datasets:
        for arm in ("dimension", "fixed"):
            if len(d["nodes"]) > 12:
                for group in range(config["reference_groups"]):
                    references.append(dict(task_id=f"{d['key']}--{arm}--reference-g{group}",
                        dataset_key=d["key"], arm=arm, method="order-mcmc", ranks=config["reference_ranks"],
                        wall_seconds=config["reference_seconds"], seed_index=group, reference=True))
        for seed_index in range(config["seeds"]):
            group = []
            for arm in ("dimension", "fixed"):
                ranks_list = config["ranks"] if arm == "dimension" else [config["sensitivity_ranks"]]
                for ranks in ranks_list:
                    particles = list(config["particles"]) if arm == "dimension" else [config["sensitivity_particles"]]
                    budgets = list(config["mcmc_seconds"]) if arm == "dimension" else [config["sensitivity_mcmc_seconds"]]
                    if arm == "dimension" and ranks == 64:
                        particles.append(config["extra_particles_at_64"])
                        budgets.append(config["extra_mcmc_seconds_at_64"])
                    for method, values in (("ordersmc", particles), ("order-mcmc", budgets)):
                        for value in values:
                            task = dict(task_id=f"{d['key']}--{arm}--{method}--r{ranks}--b{value}--s{seed_index}",
                                dataset_key=d["key"], arm=arm, method=method, ranks=ranks,
                                seed_index=seed_index, reference=False,
                                **({"particles": value} if method == "ordersmc" else {"wall_seconds": value}))
                            task["purpose"] = "fixed-work-scaling" if method == "ordersmc" and value == config["scaling_particles"] else "accuracy-time"
                            group.append(len(tasks)); tasks.append(task)
            schedule_seed = seed_record(config, d["key"], seed_index, "schedule")
            order = np.random.Generator(np.random.PCG64DXSM(schedule_seed["seed"])).permutation(group).tolist()
            bundles.append(dict(dataset_key=d["key"], seed_index=seed_index, tasks=order, schedule_seed=schedule_seed))
    return tasks, bundles, references


def freeze(root, config_path=DEFAULT_CONFIG):
    if root.exists(): raise FileExistsError("use a new timing-pilot root")
    config = read(config_path)
    if "extends" in config:
        config = {**read(REPO / config["extends"]), **config}
    source = REPO / config["source"]
    source_plan = catchup.checked_source(source / "plan.json")
    datasets = [next(d for d in source_plan["datasets"] if d["key"] == key) for key in config["dataset_keys"]]
    for d in datasets:
        checked_input(d["candidate"])
        for a in d["inputs"]: checked_input(a)
    tasks, bundles, references = make_tasks(config, datasets)
    reporting_repair.snapshot(root, dict(schema="ordersmc-timing-pilot-plan-v1", config=config,
        config_file=artifact(config_path), source_plan=artifact(source / "plan.json"),
        datasets=datasets, tasks=tasks, bundles=bundles, references=references,
        confirmatory=False, package_defaults_changed=False, primary_prior="lambda(d)=log(d-1)"))


def preflight(root):
    plan = catchup.checked_plan(root); checked_input(plan["source_plan"])
    for d in plan["datasets"]:
        table = load_candidate_table(checked_input(d["candidate"]), expected_sha256=d["candidate_sha256"])
        validate_candidate_provenance(table, config=OrderSMCConfig.from_mapping(
            table.provenance["candidate_build_config"]), missing_policy="require")
        for a in d["inputs"]: checked_input(a)
    _atomic_json(root / "validation/preflight.json", dict(all_passed=True, plan_sha256=plan["plan_sha256"]))


def accepted_plan(root):
    plan = catchup.checked_plan(root)
    if read(root / "validation/preflight.json") != dict(all_passed=True, plan_sha256=plan["plan_sha256"]):
        raise RuntimeError("timing preflight failed")
    return plan


def prepare(root, index):
    plan = accepted_plan(root); d = plan["datasets"][index]
    destination = root / "prepared" / d["key"]
    if (destination / "prepared.json").exists():
        value = read(destination / "prepared.json")
        if value["plan_sha256"] != plan["plan_sha256"]: raise RuntimeError("stale preparation")
        for c in value["cases"].values(): checked_input(c["candidate"])
        return
    source = load_candidate_table(checked_input(d["candidate"]), expected_sha256=d["candidate_sha256"])
    cfg = OrderSMCConfig.from_mapping(source.provenance["candidate_build_config"])
    manifest = next(a for a in d["inputs"] if Path(a["path"]).name == "generator_manifest.json")
    train = next(a for a in d["inputs"] if Path(a["path"]).name == "train.csv")
    checked_input(manifest); checked_input(train)
    started = time.perf_counter()
    m = read(manifest["path"]); data = pd.read_csv(train["path"])
    # Rebuild only to measure common preparation cost; both samplers consume
    # exactly the same immutable, retargeted source likelihood arrays.
    rebuilt = build_candidates(data, m["variable_types"], m["cardinalities"], config=cfg, verbose=False)
    preparation_seconds = time.perf_counter() - started
    if candidate_support_sha256(rebuilt) != candidate_support_sha256(source):
        raise RuntimeError("timed preparation changed candidate support")
    deviations = []
    for i in range(len(source.nodes)):
        mapping = lambda t: {tuple(sorted(int(p) for p in row if p >= 0)): float(v)
            for row, v in zip(t.parent_sets[i], t.loglik[i])}
        a, b = mapping(source), mapping(rebuilt)
        deviations.extend(abs(a[key] - b[key]) for key in a)
    if max(deviations, default=0.) > 1e-6: raise RuntimeError("timed rescore differs materially from frozen scores")
    cases = {}
    for arm, penalty in (("fixed", 1.), ("dimension", math.log(len(source.nodes) - 1))):
        if arm not in plan["config"].get("arms", ["fixed", "dimension"]):
            continue
        model = cfg.with_overrides(lam=0., edge_penalty=penalty, **plan["config"]["smc"])
        table, audit = retarget_edge_count_prior(source, target_label="order", lambda_target=penalty, target_config=model)
        path = save_candidate_table(table, destination / f"{arm}.json.gz")
        case = dict(candidate=artifact(path), candidate_sha256=candidate_table_content_sha256(table),
            config=model.to_dict(), retarget_audit=audit.to_dict())
        if len(source.nodes) <= 12:
            exact = exact_order_dp(table, compute_graph_size=False)
            path = destination / f"{arm}-exact.json"
            _atomic_json(path, dict(edge_probabilities=catchup._all_edges(source.nodes, exact.edge_probs),
                log_normalizer=exact.log_normalizer, target="candidate_restricted_order_modular"))
            case["exact"] = artifact(path)
        cases[arm] = case
    _atomic_json(destination / "prepared.json", dict(status="complete", plan_sha256=plan["plan_sha256"],
        dataset_key=d["key"], cases=cases, shared_preparation_seconds=preparation_seconds,
        preparation_ranks=1, hardware=hardware(), max_rescore_difference=max(deviations, default=0.)))
    print("prepared", d["key"], flush=True)


def pooled_chains(chains, nodes):
    """Equal independent-chain weights; no treating correlated draws as ESS."""
    edges = [(u, v) for u in nodes for v in nodes if u != v]
    return {e: float(np.mean([c["pips"].get(e, 0.) for c in chains])) for e in edges}


def split_rank_rhat(chains):
    """Rank-normalized and folded split R-hat for equal-length scalar chains.

    Constants are explicitly flagged by the caller; they are not evidence that
    every mode has been explored. Implements the definition documented by Stan.
    """
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.isfinite(x).all(): return math.inf
    half = x.shape[1] // 2
    split = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    def calculate(values):
        n = values.shape[1]; within = np.var(values, axis=1, ddof=1).mean()
        between = n * np.var(values.mean(axis=1), ddof=1)
        if within == 0: return 1. if between == 0 else math.inf
        return math.sqrt(((n - 1) * within / n + between / n) / within)
    def normal(values):
        ranks = rankdata(values.ravel(), method="average")
        return ndtri((ranks - .375) / (ranks.size + .25)).reshape(values.shape)
    return max(calculate(normal(split)), calculate(normal(np.abs(split - np.median(split)))))


def mcmc_chain(table, task, config, rank):
    component = f"{'reference' if task['reference'] else 'evaluation'}-mcmc-rank{rank}"
    record = seed_record(config, task["dataset_key"], task["seed_index"], component)
    result = order_mcmc(table, OrderMCMCConfig(iterations=10**8, burnin=0, thin=config["mcmc_thin"],
        wall_seconds=task["wall_seconds"], wall_burnin_fraction=.25, seed=record["seed"]))
    chain = dict(pips=result.edge_probs, seed_record=record, stored_samples=result.stored_samples,
        retained_samples=result.retained_samples, completed_iterations=result.completed_iterations,
        acceptance_rate=result.acceptance_rate, log_target_ess=result.log_target_effective_sample_size,
        min_edge_ess=result.effective_sample_size)
    return chain, result


def reference_diagnostic(result, table, monitored):
    # Bounded trace storage, preserving chronological order. Diagnostics are
    # outside timed inference, and explicitly separate from posterior estimates.
    count = min(result.stored_samples, 1000)
    indices = np.linspace(0, result.stored_samples - 1, count, dtype=int)
    series = []; halves = [{e: 0. for e in result.edge_probs}, {e: 0. for e in result.edge_probs}]; counts = [0, 0]
    for index in indices:
        values = edge_probs_for_order(result.orders[index], table, beta=1.)
        series.append([order_log_target(result.orders[index], table, 1.), *(values.get(e, 0.) for e in monitored)])
        h = int(index >= result.stored_samples // 2); counts[h] += 1
        for edge in halves[h]: halves[h][edge] += values.get(edge, 0.)
    for h in (0, 1):
        for edge in halves[h]: halves[h][edge] /= max(1, counts[h])
    return dict(monitored_edges=[list(e) for e in monitored], trace=series,
        half_pips=[catchup._all_edges(table.nodes, h) for h in halves],
        trace_policy="up to 1000 evenly spaced retained orders; half means diagnostic, not the reference estimator")


def run(root, index, comm, reference=False):
    from mpi4py import MPI
    plan = root_call(comm, lambda: accepted_plan(root))
    task = plan["references" if reference else "tasks"][index]
    if comm.size != task["ranks"]: raise RuntimeError("incorrect active rank allocation")
    d = next(d for d in plan["datasets"] if d["key"] == task["dataset_key"])
    destination = root / ("references" if reference else "runs") / task["task_id"] / "result.json"
    def previous():
        if not destination.exists(): return False
        r = read(destination)
        if r["status"] != "complete" or r["task"] != task or r["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("stale timing artifact")
        return True
    if root_call(comm, previous): return
    prep = root_call(comm, lambda: read(root / "prepared" / d["key"] / "prepared.json"))
    if prep["plan_sha256"] != plan["plan_sha256"]: raise RuntimeError("stale candidate preparation")
    case = prep["cases"][task["arm"]]
    comm.Barrier(); setup_start = time.perf_counter()
    table = root_call(comm, lambda: load_candidate_table(checked_input(case["candidate"]), expected_sha256=case["candidate_sha256"]))
    cfg = OrderSMCConfig.from_mapping(case["config"])
    # Identical tiny target-evaluation warm-up for both implementations.
    order_log_target(np.arange(len(table.nodes), dtype=np.int32), table, 1.)
    comm.Barrier(); setup_seconds = comm.allreduce(time.perf_counter() - setup_start, op=MPI.MAX)
    machines = comm.gather(hardware(), root=0)
    def check_binding():
        if len({m["host"] for m in machines}) != 1 or len({m["cpu_model"] for m in machines}) != 1:
            raise RuntimeError("single-node homogeneous timing allocation required")
        affinities = [set(m["affinity"]) for m in machines]
        if any(a & b for i, a in enumerate(affinities) for b in affinities[i + 1:]):
            raise RuntimeError("benchmark ranks have overlapping CPU affinity")
    root_call(comm, check_binding)
    comm.Barrier(); started = time.perf_counter()
    if task["method"] == "ordersmc":
        record = seed_record(plan["config"], d["key"], task["seed_index"], "evaluation-smc")
        cfg = cfg.with_overrides(particles=task["particles"], seed=record["seed"])
        if task["purpose"] == "fixed-work-scaling":
            cfg = cfg.with_overrides(adaptive_tempering=False, num_betas=plan["config"]["scaling_num_betas"],
                mh_steps_resample=cfg.mh_steps)
        result = sample(table, config=cfg, backend="mpi", comm=comm, parent_seed_record=record,
            provenance_policy="require", verbose=False)
        probabilities = result.edge_probs if comm.rank == 0 else None
        chain_results = None
        diagnostics = dict(seed_record=record, history=result.history, run_info=result.run_info) if comm.rank == 0 else None
    else:
        error = None; chain = result = None
        try: chain, result = mcmc_chain(table, task, plan["config"], comm.rank)
        except Exception as exc: error = repr(exc)
        errors = comm.allgather(error)
        if any(errors): raise RuntimeError(f"MCMC rank failure: {errors}")
        chain_results = comm.gather(chain, root=0)
        probabilities = pooled_chains(chain_results, table.nodes) if comm.rank == 0 else None
        diagnostics = None
    comm.Barrier(); seconds = comm.allreduce(time.perf_counter() - started, op=MPI.MAX)
    # Reference-only diagnostics, never charged to benchmark MCMC alone.
    if reference:
        panel = (sorted(probabilities, key=lambda e: (-probabilities[e] * (1 - probabilities[e]), e))
            [:plan["config"]["reference_monitor_edges"]]) if comm.rank == 0 else None
        panel = comm.bcast(panel, root=0)
        diagnostic = reference_diagnostic(result, table, panel)
        all_diagnostics = comm.gather(diagnostic, root=0)
    else: all_diagnostics = None
    peak_rss = comm.reduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.MAX, root=0)
    def save():
        from .orientation_audit import graph
        truth = graph(d["nodes"], d["truth_edges"])
        posterior = summarize_posterior(probabilities, truth, d["nodes"], d["max_parents"])
        chains = []
        for c in chain_results or []:
            chains.append({**{k: v for k, v in c.items() if k != "pips"},
                "edge_probabilities": catchup._all_edges(table.nodes, c["pips"])})
        if task.get("purpose") == "fixed-work-scaling":
            observed = sum(int(h.get("proposed_moves", 0)) for h in result.history)
            expected = task["particles"] * (plan["config"]["scaling_num_betas"] - 1) * cfg.mh_steps
            if observed != expected: raise RuntimeError("fixed-work scaling proposal count mismatch")
        _atomic_json(destination, dict(status="complete", plan_sha256=plan["plan_sha256"], task=task,
            candidate_sha256=case["candidate_sha256"], setup_seconds=setup_seconds,
            inference_seconds=seconds, active_core_hours=seconds * comm.size / 3600,
            hardware=machines, posterior=posterior, chains=chains, smc_diagnostics=diagnostics,
            reference_diagnostics=all_diagnostics, shared_preparation_seconds=prep["shared_preparation_seconds"],
            max_rank_peak_rss_kib=peak_rss))
        print("complete", task["task_id"], "seconds", seconds, flush=True)
    root_call(comm, save)


def bundle(root, index):
    """Run all predeclared task steps on the same exclusively allocated node."""
    plan = accepted_plan(root); group = plan["bundles"][index]
    attempts = root / "attempts"; attempts.mkdir(exist_ok=True)
    for task_index in group["tasks"]:
        task = plan["tasks"][task_index]; attempt = attempts / (task["task_id"] + ".json")
        if attempt.exists():
            old = read(attempt)
            if old["plan_sha256"] != plan["plan_sha256"] or old["task"] != task: raise RuntimeError("stale attempt")
            continue  # Failures are preserved, not selectively retried for speed.
        cmd = ["srun", "--mpi=" + os.environ.get("ORDERSMC_SLURM_MPI_TYPE", "pmi2"), "--cpu-bind=cores", "--exact",
            "--kill-on-bad-exit=1", f"--ntasks={task['ranks']}", "--cpus-per-task=1",
            f"--time={plan['config']['step_time_limit_minutes']}", sys.executable,
            "-m", "ordersmc_experiments.timing_pilot", "run", "--mpi", "--output", str(root), "--task", str(task_index)]
        started = time.perf_counter()
        completed = subprocess.run(cmd, check=False)
        elapsed = time.perf_counter() - started
        _atomic_json(attempt, dict(task=task, plan_sha256=plan["plan_sha256"], returncode=completed.returncode,
            launcher_wall_seconds=elapsed, status="completed" if completed.returncode == 0 else "failed_or_time_limited",
            hardware=hardware(), command=cmd,
            reserved_core_hours=elapsed * int(os.environ.get("SLURM_CPUS_ON_NODE", task["ranks"])) / 3600))
    _atomic_json(root / "bundles" / f"{index}.json", dict(status="attempted_all", plan_sha256=plan["plan_sha256"], bundle=group))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "preflight", "prepare", "run", "bundle", "aggregate"])
    p.add_argument("--output", type=Path, required=True); p.add_argument("--task", type=int)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--mpi", action="store_true"); p.add_argument("--reference", action="store_true")
    a = p.parse_args()
    if a.action == "run":
        from mpi4py import MPI
        run(a.output, a.task, MPI.COMM_WORLD, a.reference)
    elif a.action in {"prepare", "bundle"}: globals()[a.action](a.output, a.task)
    elif a.action == "aggregate":
        if accepted_plan(a.output)["config"].get("task_design") == "scaling-quality-v1":
            from .scaling_quality import aggregate
        else:
            from .timing_report import aggregate
        aggregate(a.output)
    elif a.action == "freeze": freeze(a.output, a.config)
    else: globals()[a.action](a.output)


if __name__ == "__main__": main()
