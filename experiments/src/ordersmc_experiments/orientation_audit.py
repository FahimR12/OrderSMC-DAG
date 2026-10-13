"""Same-data direction audit: distinguish sampling, target, decision and equivalence.

This is a development diagnostic, not a score/prior selection procedure. Original
fits are read-only. All six signal cases (three datasets x two priors) are used.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig, build_candidates
from ordersmc.graphs import dag_to_cpdag, optimal_bayes_action_dag
from ordersmc.persistence import load_candidate_table, save_candidate_table
from . import catchup, reporting_repair
from .bnlearn_adapter import run_bnlearn
from .exact_order import exact_order_dp
from .exact_partition_log import exact_partition_log_dp
from .execution import _atomic_json
from .order_mcmc import OrderMCMCConfig, order_mcmc
from .rank_budget import complete_probabilities
from .readiness import summarize_posterior, validated_probabilities
from .readiness_audits import pips, read, rmse, sha
from .sampler_corrections import artifact, checked_input

REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "results/reporting-null-audit/v1/null_pilot"
REPORTS = REPO / "results/reporting-null-audit/v1/reporting_repair"


def graph(nodes, edges):
    result = nx.DiGraph(); result.add_nodes_from(nodes); result.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(result):
        raise ValueError("orientation diagnostics require a DAG")
    return result


def vstructures(dag):
    return {(min(a, b), child, max(a, b)) for child in dag
            for a, b in itertools.combinations(dag.predecessors(child), 2)
            if not dag.has_edge(a, b) and not dag.has_edge(b, a)}


def direction_metrics(selected, truth):
    actual, predicted = set(truth.edges()), set(selected.edges())
    skel = lambda edges: {frozenset(e) for e in edges}
    true_cp, pred_cp = dag_to_cpdag(truth), dag_to_cpdag(selected)
    def states(cp):
        return {frozenset(e): tuple(e) for e in cp.directed_edges} | {
            pair: None for pair in cp.undirected_edges}
    a, b = states(true_cp), states(pred_cp)
    cp_shd = sum(pair not in a or pair not in b or a[pair] != b[pair]
                 for pair in a.keys() | b.keys())
    correct, reversed_count = len(actual & predicted), len({(v, u) for u, v in actual} & predicted)
    return dict(
        correct_arrows=correct, reversed_arrows=reversed_count,
        extra_adjacencies=len(skel(predicted) - skel(actual)),
        missing_adjacencies=len(skel(actual) - skel(predicted)),
        orientation_accuracy_on_recovered_adjacencies=(correct / (correct + reversed_count)
            if correct + reversed_count else None),
        truth_compelled_arrows=len(true_cp.directed_edges),
        predicted_compelled_arrows=len(pred_cp.directed_edges),
        cpdag_pair_state_shd=cp_shd,
        markov_equivalent=(skel(actual) == skel(predicted) and vstructures(truth) == vstructures(selected)),
        true_vstructures=len(vstructures(truth)),
        extra_vstructures=len(vstructures(selected) - vstructures(truth)),
        missing_vstructures=len(vstructures(truth) - vstructures(selected)))


def summarize(probs, truth, max_parents, *, loss="pair_state_hamming", posterior=True):
    probs = validated_probabilities(complete_probabilities(probs, list(truth)))
    result = summarize_posterior(probs, truth, list(truth), max_parents)
    chosen = optimal_bayes_action_dag(probs, list(truth), max_indegree=max_parents, loss=loss)
    result.update(representative_edges=sorted(map(list, chosen.edges())),
                  representative_graph_diagnostics=dict(chosen.graph))
    actual, selected = set(truth.edges()), set(chosen.edges())
    skeleton = lambda edges: {frozenset(e) for e in edges}
    result.update(directed_f1=2 * len(actual & selected) / max(1, len(actual) + len(selected)),
        skeleton_f1=2 * len(skeleton(actual) & skeleton(selected)) /
        max(1, len(skeleton(actual)) + len(skeleton(selected))), **direction_metrics(chosen, truth))
    # Do not call a point estimator's 0/1 representation posterior confidence.
    result["high_confidence_reversed_truth_edges"] = (
        sum(probs[v, u] >= .9 for u, v in truth.edges()) if posterior else None)
    return result


def chain_orientation_scores(table, truth):
    """All collider-free orientations of the known chain, NOT a fitted posterior.

    Local-prior mass is constant because every orientation has d-1 edges.
    Order-modular graph weight additionally includes its linear-extension count.
    Conditional weights here are restricted to this truth-MEC for diagnosis only.
    """
    skeleton = truth.to_undirected()
    if not nx.is_tree(skeleton) or max(dict(skeleton.degree()).values()) > 2 or vstructures(truth):
        raise ValueError("expected a collider-free chain")
    lookup = []
    for families in table.parent_sets:
        lookup.append({frozenset(int(p) for p in row if p >= 0): i for i, row in enumerate(families)})
    index = {node: i for i, node in enumerate(table.nodes)}
    endpoint = min(node for node, degree in skeleton.degree() if degree == 1)
    rows = []
    for root in table.nodes:
        dag = nx.bfs_tree(skeleton, root)
        ll, prior = 0., 0.
        for child in table.nodes:
            i = index[child]; parents = frozenset(index[p] for p in dag.predecessors(child))
            family = lookup[i][parents]
            ll += float(table.loglik[i][family]); prior += float(table.logprior[i][family])
        position = nx.shortest_path_length(skeleton, endpoint, root)
        extensions = math.comb(len(table.nodes) - 1, position)
        rows.append(dict(root=root, log_likelihood=ll, local_log_prior=prior,
            linear_extensions=extensions, structure_log_weight=ll + prior,
            order_log_weight=ll + prior + math.log(extensions),
            is_truth=set(dag.edges()) == set(truth.edges()), edges=sorted(map(list, dag.edges()))))
    from scipy.special import logsumexp
    for target in ("order", "structure"):
        normalizer = logsumexp([r[f"{target}_log_weight"] for r in rows])
        for row in rows:
            row[f"conditional_mec_{target}_weight"] = math.exp(row[f"{target}_log_weight"] - normalizer)
    return rows


def prepare(root):
    if root.exists():
        raise FileExistsError("use a new orientation diagnostic root")
    source = catchup.checked_source(PILOT / "plan.json")
    if read(PILOT / "analysis/completion.json")["status"] != "complete":
        raise RuntimeError("requires the complete null/signal pilot")
    cases = []
    for case in source["cases"]:
        if case["dataset"]["kind"] != "fresh-signal":
            continue
        records = []
        for task in source["tasks"]:
            if task["case_id"] == case["case_id"]:
                records.append(dict(task=task, result=artifact(PILOT / "runs" / task["task_id"] / "result.json")))
        if len(records) != 4:
            raise RuntimeError("requires all four seeds")
        cases.append(dict(**case, smc_records=records))
    if len(cases) != 6:
        raise RuntimeError("requires all six prior/dataset combinations")
    historical = {cohort: artifact(REPORTS / "analysis" / cohort / "dataset_mean_metrics.csv")
                  for cohort in ("v1", "v3_mixed", "v3_clg", "metadata")}
    reporting_repair.snapshot(root, dict(schema="ordersmc-orientation-audit-v1", cases=cases,
        source_plan=artifact(PILOT / "plan.json"), historical=historical,
        mcmc=dict(iterations=20000, burnin=5000, thin=5),
        analysis="all cases and seeds; dataset means before cohort means; no best-seed selection",
        core_model_changed=False, confirmatory=False,
        cpdag_scope="generic observational Markov equivalence, not parametric mixed-family identifiability",
        native_sensitivity="same data/support/prior, remove only marginal rank transform; not a promoted score",
        native_bnlearn="BIC-CG has restricted support: continuous-to-discrete arrows forbidden",
        equal_compute=False))


def preflight(root):
    plan = catchup.checked_plan(root)
    checked_input(plan["source_plan"])
    for case in plan["cases"]:
        checked_input(case["candidate"]); checked_input(case["exact"])
        for record in case["smc_records"]:
            checked_input(record["result"])
        for record in case["dataset"]["source_inputs"]:
            checked_input(record)
    for record in plan["historical"].values():
        checked_input(record)
    _atomic_json(root / "validation/preflight.json", dict(all_passed=True, plan_sha256=plan["plan_sha256"]))


def run(root, index):
    plan = catchup.checked_plan(root); case = plan["cases"][index]
    if read(root / "validation/preflight.json") != dict(all_passed=True, plan_sha256=plan["plan_sha256"]):
        raise RuntimeError("preflight failed")
    destination = root / "runs" / case["case_id"] / "result.json"
    if destination.exists():
        existing = read(destination)
        if existing["status"] != "complete" or existing["plan_sha256"] != plan["plan_sha256"]:
            raise RuntimeError("stale orientation diagnostic")
        return
    d = case["dataset"]; truth = graph(d["nodes"], d["truth_edges"])
    cfg = OrderSMCConfig.from_mapping(case["config"])
    table = load_candidate_table(checked_input(case["candidate"]), expected_sha256=case["candidate_sha256"])
    data_record, = [r for r in d["source_inputs"] if Path(r["path"]).name == "data.csv"]
    data = pd.read_csv(checked_input(data_record))
    types = {f"x{i}": t for i, t in enumerate(("con", "bin", "cat", "con", "bin", "con", "cat", "con"))}
    cards = {node: 2 if t == "bin" else 3 if t == "cat" else 0 for node, t in types.items()}
    native = build_candidates(data, types, cards, config=cfg.with_overrides(score_method="native"), verbose=False)
    support = lambda families: {tuple(sorted(int(p) for p in row if p >= 0)) for row in families}
    if not all(support(a) == support(b) for a, b in zip(table.parent_sets, native.parent_sets)):
        raise RuntimeError("native sensitivity changed candidate support")
    saved_native = save_candidate_table(native, destination.parent / "native_candidate.json.gz")
    exact = exact_order_dp(table, compute_graph_size=False)
    if rmse(exact.edge_probs, pips(read(checked_input(case["exact"])))) > 1e-10:
        raise RuntimeError("exact reference failed replay")
    structure = exact_partition_log_dp(table)
    native_exact = exact_order_dp(native, compute_graph_size=False)
    records = []
    def add(method, probs, seed=-1, steps=0, loss="pair_state_hamming", posterior=True, **details):
        records.append(dict(method=method, seed_index=seed, final_target_steps=steps, loss=loss,
            **summarize(probs, truth, cfg.max_parents, loss=loss, posterior=posterior), **details))
    for name, ref in (("exact-order", exact), ("exact-structure", structure), ("exact-order-native", native_exact)):
        for loss in ("pair_state_hamming", "directed_edge_hamming"):
            add(name, ref.edge_probs, loss=loss, log_normalizer=ref.log_normalizer)
    for record in case["smc_records"]:
        result = read(checked_input(record["result"]))
        if result["task"] != record["task"] or result["candidate_sha256"] != case["candidate_sha256"]:
            raise RuntimeError("SMC source identity mismatch")
        for snapshot in result["snapshots"]:
            add("ordersmc", pips(snapshot), record["task"]["seed_index"], snapshot["final_target_steps"],
                own_target_pip_rmse=rmse(pips(snapshot), exact.edge_probs))
        seed = record["task"]["seed_record"]["seed"]
        result = order_mcmc(table, OrderMCMCConfig(seed=seed, **plan["mcmc"]))
        add("order-mcmc", result.edge_probs, record["task"]["seed_index"],
            own_target_pip_rmse=rmse(result.edge_probs, exact.edge_probs),
            acceptance_rate=result.acceptance_rate, iterations=result.completed_iterations,
            retained_samples=result.retained_samples, stored_samples=result.stored_samples)
    for algorithm in ("hc", "tabu"):
        for mode in ("shared-mixed-score", "native-bic-cg"):
            result = run_bnlearn(data, types, table, algorithm=algorithm, scoring_mode=mode, seed=20260921)
            add(f"bnlearn-{algorithm}-{mode}", result.edge_probs, posterior=False,
                fitted_graph_score=result.score, bnlearn_version=result.bnlearn_version,
                unsupported_truth_arrows=(sum(types[u] == "con" and types[v] != "con" for u, v in truth.edges())
                    if mode == "native-bic-cg" else 0))
    edge_rows = []
    for label, ref in (("exact-order", exact), ("exact-structure", structure), ("exact-order-native", native_exact)):
        for u, v in truth.edges():
            forward, reverse = ref.edge_probs[u, v], ref.edge_probs[v, u]
            edge_rows.append(dict(method=label, parent=u, child=v, parent_type=types[u], child_type=types[v],
                true_direction_pip=forward, reverse_direction_pip=reverse,
                adjacency_pip=forward + reverse,
                true_direction_given_adjacency=forward / (forward + reverse) if forward + reverse else None))
    _atomic_json(destination, dict(status="complete", plan_sha256=plan["plan_sha256"],
        case_id=case["case_id"], dataset_key=d["key"], arm=case["arm"], records=records,
        truth_edges=d["truth_edges"], truth_compelled_arrows=len(dag_to_cpdag(truth).directed_edges),
        chain_orientations={"rank-native": chain_orientation_scores(table, truth),
                            "native": chain_orientation_scores(native, truth)},
        edge_directions=edge_rows, native_candidate=artifact(saved_native)))
    print("complete", case["case_id"], flush=True)


def historical_comparison(plan, output):
    rows = []
    for cohort, record in plan["historical"].items():
        frame = pd.read_csv(checked_input(record))
        frame = frame[(frame.budget == "baseline") & (frame.final_target_steps == 0)]
        if cohort == "metadata":
            frame = frame[frame.cell_id.str.startswith("m1_matched-")]
        baseline = frame[frame.method == "ordersmc-v07"].set_index("dataset_key")
        for method, group in frame.groupby("method"):
            if set(group.dataset_key) != set(baseline.index):
                continue  # exact small-subset references are not an all-dataset comparison
            paired = group.set_index("dataset_key").join(baseline, rsuffix="_smc")
            difference = paired.directed_f1 - paired.directed_f1_smc
            rows.append(dict(cohort=cohort, method=method, datasets=len(group),
                directed_f1=group.directed_f1.mean(), skeleton_f1=group.skeleton_f1.mean(),
                paired_directed_f1_delta_vs_smc=difference.mean(),
                wins=int((difference > 1e-10).sum()), losses=int((difference < -1e-10).sum()),
                ties=int((difference.abs() <= 1e-10).sum()), equal_compute=False))
    pd.DataFrame(rows).to_csv(output / "historical_comparison.csv", index=False)


def aggregate(root):
    plan = catchup.checked_plan(root); rows, edges, hashes = [], [], {}
    for case in plan["cases"]:
        path = root / "runs" / case["case_id"] / "result.json"
        value = read(path)
        if value["status"] != "complete" or value["plan_sha256"] != plan["plan_sha256"] or value["case_id"] != case["case_id"]:
            raise RuntimeError("incomplete or stale orientation audit")
        hashes[str(path)] = sha(path)
        for record in case["smc_records"]:
            checked_input(record["result"])
        common = {k: value[k] for k in ("dataset_key", "arm")}
        for r in value["records"]:
            rows.append(dict(**common, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}))
        edges.extend(dict(**common, **row) for row in value["edge_directions"])
    output = root / "analysis"; output.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(output / "per_run_metrics.csv", index=False)
    fields = ["directed_f1", "skeleton_f1", "correct_arrows", "reversed_arrows", "extra_adjacencies",
              "missing_adjacencies", "orientation_accuracy_on_recovered_adjacencies", "cpdag_pair_state_shd",
              "truth_compelled_arrows", "predicted_compelled_arrows", "markov_equivalent", "extra_vstructures",
              "high_confidence_reversed_truth_edges", "own_target_pip_rmse"]
    means = frame.groupby(["dataset_key", "arm", "method", "loss", "final_target_steps"], as_index=False)[fields].mean()
    means.to_csv(output / "dataset_metrics.csv", index=False)
    means.groupby(["arm", "method", "loss", "final_target_steps"], as_index=False)[fields].mean().to_csv(output / "summary.csv", index=False)
    pd.DataFrame(edges).to_csv(output / "edge_directions.csv", index=False)
    historical_comparison(plan, output)
    _atomic_json(output / "artifact_hashes.json", hashes)
    _atomic_json(output / "completion.json", dict(status="complete", cases=len(plan["cases"]),
        plan_sha256=plan["plan_sha256"], confirmatory=False, model_default_changed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "preflight", "run", "aggregate"])
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.action == "run": run(args.output, args.task)
    else: globals()[args.action](args.output)


if __name__ == "__main__": main()
