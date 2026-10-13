"""Post-hoc failure diagnosis and non-destructive uncertainty-aware reporting."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew

from ordersmc import uncertainty_aware_graph, build_candidates
from ordersmc.api import make_scorer
from ordersmc.persistence import load_candidate_table
from .exact_order import exact_order_dp
from .readiness import score_config, summarize_posterior, validated_probabilities
from .readiness_audits import read, write, sha, pips, rmse
from .score_validation_v2 import (
    independent_nig_log_marginal, manual_bdeu_log_marginal,
    _logit_log_joint_derivatives, _softmax_log_joint_derivatives, _contrast_basis,
)


def graph(edges, nodes):
    result = nx.DiGraph()
    result.add_nodes_from(nodes)
    result.add_edges_from(edges)
    return result


def linear_extensions(dag):
    """Exact count of topological orders, used only for a small diagnostic DAG."""
    nodes = tuple(sorted(dag))
    if len(nodes) > 12 or not nx.is_directed_acyclic_graph(dag):
        raise ValueError("linear-extension audit requires a DAG with at most 12 nodes")
    parent_masks = [sum(1 << nodes.index(p) for p in dag.predecessors(v)) for v in nodes]
    counts = [0] * (1 << len(nodes))
    counts[0] = 1
    for mask in range(len(counts)):
        for i, parents in enumerate(parent_masks):
            if not mask & (1 << i) and parents & mask == parents:
                counts[mask | (1 << i)] += counts[mask]
    return counts[-1]


def family_lookup(table, child):
    i = table.nodes.index(child)
    return {tuple(sorted(table.nodes[int(p)] for p in row if p >= 0)): k
            for k, row in enumerate(table.parent_sets[i])}


def mask_metrics(summary, truth):
    selected = set(summary.directed_edges)
    correct = len(selected & truth)
    return {"asserted_arrows": len(selected), "unresolved_edges": len(summary.undirected_edges),
            "correct_asserted_arrows": correct, "incorrect_asserted_arrows": len(selected) - correct,
            "arrow_coverage": len(selected) / max(1, len(selected) + len(summary.undirected_edges)),
            "asserted_arrow_precision": correct / len(selected) if selected else None,
            "asserted_true_edge_recall": correct / len(truth) if truth else None}


def signature(summary):
    return {("directed", *e) for e in summary.directed_edges} | {("undirected", *e) for e in summary.undirected_edges}


def uncertainty_report(source, output, thresholds=(.80, .90, .95)):
    rows, stability = [], []
    plan = read(source / "plan.json")
    paths = sorted((source / "cases").glob("rep*/*-n*.json"))
    if len(paths) != 60:
        raise RuntimeError("uncertainty report requires all 60 original conditions")
    seen = set()
    folder = output / "uncertainty_graphs"
    folder.mkdir()
    for path in paths:
        result = read(path)
        key = (result["replicate"], result["variant"], result["n_train"])
        if (key in seen or result["status"] != "complete" or len(result["smc_runs"]) != 4
            or result["plan_sha256"] != plan["plan_sha256"]):
            raise RuntimeError(f"incompatible or duplicate source result: {path}")
        seen.add(key)
        nodes = sorted(result["variable_types"])
        truth = set(map(tuple, result["truth_edges"]))
        common = {key: result[key] for key in ("replicate", "variant", "n_train")}
        saved = {**common, "source_file_sha256": sha(path), "summaries": []}
        for threshold in thresholds:
            signatures = []
            sources = [(m, [r]) for m, r in result["references"].items()] + [("ordersmc", result["smc_runs"])]
            for method, records in sources:
                for index, record in enumerate(records):
                    base = graph(record["representative_edges"], nodes)
                    summary = uncertainty_aware_graph(pips(record), base, orientation_threshold=threshold)
                    metrics = mask_metrics(summary, truth)
                    rows.append({**common, "threshold": threshold, "method": method, "seed_index": index,
                                 "original_directed_f1": record["directed_f1"],
                                 "original_skeleton_f1": record["skeleton_f1"], **metrics})
                    saved["summaries"].append({"method": method, "seed_index": index, "threshold": threshold,
                                               **summary.to_dict()})
                    if method == "ordersmc":
                        signatures.append(signature(summary))
            jaccards = [len(a & b) / len(a | b) if a | b else 1.
                        for a, b in itertools.combinations(signatures, 2)]
            stability.append({**common, "threshold": threshold,
                              "all_seed_summaries_identical": all(s == signatures[0] for s in signatures),
                              "mean_summary_jaccard": float(np.mean(jaccards))})
        write(folder / f"rep{result['replicate']:03d}-{path.name}", saved)
    runs = pd.DataFrame(rows)
    runs.to_csv(output / "uncertainty_runs.csv", index=False)
    fields = ["asserted_arrows", "unresolved_edges", "correct_asserted_arrows", "incorrect_asserted_arrows",
              "arrow_coverage", "asserted_arrow_precision", "asserted_true_edge_recall",
              "original_directed_f1", "original_skeleton_f1"]
    dataset = runs.groupby(["variant", "n_train", "method", "threshold", "replicate"], as_index=False)[fields].mean()
    dataset.to_csv(output / "uncertainty_dataset_means.csv", index=False)
    dataset.groupby(["variant", "n_train", "method", "threshold"], as_index=False)[fields].mean().to_csv(output / "uncertainty_summary.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "uncertainty_stability.csv", index=False)


def contexts(scorer, parents):
    discrete = [p for p in parents if scorer.vtypes[p] != "con"]
    n = len(next(iter(scorer.Xi.values()))) if scorer.Xi else len(next(iter(scorer.Xf.values())))
    if not discrete:
        return [np.arange(n)]
    _, inverse = np.unique(np.column_stack([scorer.Xi[p] for p in discrete]), axis=0, return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]


def independent_family_check(scorer, child, parents):
    """Independent closed forms or a separately optimized Laplace calculation.

    The latter checks MAP/Hessian arithmetic, NOT the error of the Laplace
    approximation against the integral; this limitation is explicit in output.
    """
    production = scorer.score(child, parents)
    continuous = sorted(p for p in parents if scorer.vtypes[p] == "con")
    discrete = sorted(p for p in parents if scorer.vtypes[p] != "con")
    if production.family == "disc_bdeu":
        n = len(scorer.Xi[child]); config = np.zeros(n, dtype=int); q = 1
        for parent in discrete:
            config = config * scorer.cards[parent] + scorer.Xi[parent]
            q *= scorer.cards[parent]
        reference = manual_bdeu_log_marginal(scorer.Xi[child], scorer.cards[child], config, q,
                                             alpha=scorer.disc_std.alpha)
        return {"reference_log_score": reference, "max_mode_gradient": None,
                "method": "independent Dirichlet-multinomial count formula"}
    value, gradients, successful = 0., [], []
    for indices in contexts(scorer, parents):
        design = np.column_stack([np.ones(len(indices)), *[scorer.Xf[p][indices] for p in continuous]])
        if production.family == "clg":
            value += independent_nig_log_marginal(scorer.Xf[child][indices], design,
                tau=scorer.clg.tau, shape=scorer.clg.ig_shape, scale=scorer.clg.ig_scale)
            continue
        y = scorer.Xi[child][indices]
        if production.family == "logit":
            dimension = design.shape[1]
            derivative = lambda b: _logit_log_joint_derivatives(b, y, design, scorer.logit.tau)
        elif production.family == "softmax":
            basis = _contrast_basis(scorer.cards[child]); dimension = design.shape[1] * basis.shape[1]
            derivative = lambda b: _softmax_log_joint_derivatives(b, y, design, scorer.softmax.tau, basis)
        else:
            raise ValueError(f"unsupported diagnostic branch: {production.family}")
        fit = minimize(lambda b: -derivative(b)[0], np.zeros(dimension),
                       jac=lambda b: -derivative(b)[1], hess=lambda b: -derivative(b)[2],
                       method="trust-exact", options={"gtol": 1e-8, "maxiter": 1000})
        logjoint, grad, hess = derivative(fit.x)
        sign, logdet = np.linalg.slogdet(-hess)
        if sign <= 0:
            raise ArithmeticError("independent posterior Hessian not positive definite")
        gradients.append(float(np.max(np.abs(grad))))
        successful.append(bool(fit.success))
        value += logjoint + .5 * dimension * np.log(2*np.pi) - .5 * logdet
    return {"reference_log_score": value, "max_mode_gradient": max(gradients, default=0.),
            "optimizer_reports_success_all_contexts": all(successful),
            "method": "independent NIG closed form" if production.family == "clg" else
            "independent trust-region MAP and analytic Hessian; Laplace-integral error NOT certified"}


def diagnose_failure(source, output, replicate=7):
    config = read(source / "plan.json")["config"]
    root = source / "cases" / f"rep{replicate:03d}"
    manifest = read(root / "generator_manifest.json")
    types, cards = manifest["variable_types"], manifest["cardinalities"]
    native = pd.read_csv(root / "unrestricted_native.csv")
    transformed = pd.read_csv(root / "paired_nongaussian.csv")
    inverse = transformed.copy()
    for node, kind in types.items():
        if kind == "con":
            inverse[node] = np.sinh((np.arcsinh(transformed[node]) - .5) / 1.25)
    inverse_error = float(np.max(np.abs(inverse.to_numpy() - native.to_numpy())))
    true_graph = graph(manifest["edges"], types)
    true_orders = linear_extensions(true_graph)
    comparisons, local_rows, checks, oracle, moments = [], [], [], [], []
    raw_hashes = {}
    for variant, data in [("unrestricted_native", native), ("paired_nongaussian", transformed)]:
        for n in config["sample_sizes"]:
            stem = f"{variant}-n{n:05d}"
            record = read(root / f"{stem}.json")
            if sha(root / f"{variant}.csv") != record["data_sha256"]:
                raise RuntimeError("dataset byte identity mismatch")
            table = load_candidate_table(root / f"{stem}-candidate.json.gz", expected_sha256=record["candidate_sha256"])
            raw_hashes[stem] = {"result_sha256": sha(root / f"{stem}.json"), "candidate_content_sha256": record["candidate_sha256"]}
            scorer = make_scorer(data.iloc[:n], types, cards, score_config(config))
            action = graph(record["references"]["exact-order"]["representative_edges"], types)
            action_orders = linear_extensions(action)
            max_recompute_error, delta_evidence, delta_prior = 0., 0., 0.
            for i, child in enumerate(table.nodes):
                lookup = family_lookup(table, child)
                if len(lookup) != sum(math.comb(len(types)-1, k) for k in range(config["maximum_indegree"]+1)):
                    raise RuntimeError("diagnostic candidate pool is not complete")
                for parents, k in lookup.items():
                    max_recompute_error = max(max_recompute_error, abs(scorer.score(child, parents).log_marginal-table.loglik[i][k]))
                families = [tuple(sorted(g.predecessors(child))) for g in (true_graph, action)]
                scores = [float(table.loglik[i][lookup[p]]) for p in families]
                priors = [float(table.logprior[i][lookup[p]]) for p in families]
                local_rows.append({"variant": variant, "n_train": n, "child": child, "type": types[child],
                    "true_parents": ",".join(families[0]), "representative_parents": ",".join(families[1]),
                    "true_family": scorer.score(child, families[0]).family,
                    "representative_family": scorer.score(child, families[1]).family,
                    "true_log_evidence": scores[0], "representative_log_evidence": scores[1],
                    "delta_log_evidence": scores[1]-scores[0], "delta_log_prior": priors[1]-priors[0]})
                delta_evidence += scores[1]-scores[0]; delta_prior += priors[1]-priors[0]
                for parents in sorted(set(families)):
                    independent = independent_family_check(scorer, child, parents)
                    production = scorer.score(child, parents).log_marginal
                    error = abs(independent["reference_log_score"]-production)
                    gradient = independent.get("max_mode_gradient")
                    checks.append({"variant": variant, "n_train": n, "child": child, "parents": ",".join(parents),
                                   "family": scorer.score(child, parents).family,
                                   "implementation_log_score": production, "absolute_score_error": error,
                                   "arithmetic_check_pass": bool(error < 1e-5 and (gradient is None or gradient < 1e-6*(1+n))), **independent})
            log_extensions = math.log(action_orders/true_orders)
            comparisons.append({"variant": variant, "n_train": n, "true_edges": true_graph.number_of_edges(),
                "representative_edges": action.number_of_edges(), "true_topological_orders": true_orders,
                "representative_topological_orders": action_orders,
                "delta_log_evidence": delta_evidence, "delta_edge_log_prior": delta_prior,
                "delta_log_order_multiplicity": log_extensions,
                "structure_log_posterior_ratio": delta_evidence + delta_prior,
                "order_log_posterior_ratio": delta_evidence + delta_prior + log_extensions,
                "candidate_rescore_max_absolute_error": max_recompute_error,
                "smc_directed_f1": float(np.mean([s["directed_f1"] for s in record["smc_runs"]])),
                "exact_order_directed_f1": record["references"]["exact-order"]["directed_f1"],
                "exact_structure_directed_f1": record["references"]["exact-structure"]["directed_f1"]})
            for child in (c for c in types if types[c] == "con"):
                parents = tuple(sorted(true_graph.predecessors(child)))
                for k, indices in enumerate(contexts(scorer, parents)):
                    continuous = [p for p in parents if types[p] == "con"]
                    design = np.column_stack([np.ones(len(indices)), *[scorer.Xf[p][indices] for p in continuous]])
                    y = scorer.Xf[child][indices]
                    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
                    moments.append({"variant": variant, "n_train": n, "child": child, "context_index": k,
                                    "context_size": len(indices), "residual_skewness": float(skew(residual)),
                                    "residual_excess_kurtosis": float(kurtosis(residual))})
            print(f"diagnosed replicate {replicate}: {stem}", flush=True)
    for n in config["sample_sizes"]:
        rebuilt = build_candidates(inverse.iloc[:n], types, cards, config=score_config(config), verbose=False)
        native_table = load_candidate_table(root / f"unrestricted_native-n{n:05d}-candidate.json.gz")
        score_error = max(abs(float(a-b)) for left, right in zip(rebuilt.loglik, native_table.loglik) for a,b in zip(left,right))
        exact = exact_order_dp(rebuilt, compute_graph_size=False)
        baseline = read(root / f"unrestricted_native-n{n:05d}.json")["references"]["exact-order"]
        oracle.append({"n_train": n, "inverse_data_max_absolute_error": inverse_error,
                       "candidate_score_max_absolute_error": score_error,
                       "pip_rmse_to_original_native": rmse(exact.edge_probs, pips(baseline)),
                       **{k:v for k,v in summarize_posterior(exact.edge_probs, true_graph, rebuilt.nodes, config["maximum_indegree"]).items() if isinstance(v,(int,float))}})
    for name, rows in [("failure_graph_score_decomposition", comparisons), ("failure_local_scores", local_rows),
                       ("failure_score_checks", checks), ("failure_residual_moments", moments), ("oracle_inverse_control", oracle)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    write(output / "failure_audit.json", {
        "replicate": replicate, "selection": "post-hoc follow-up of previously observed largest non-Gaussian deterioration",
        "source_artifacts": raw_hashes, "inverse_control": "known-generator inverse; diagnostic oracle only, not a proposed deployable preprocessing rule",
        "all_checked_score_arithmetic_passed": all(r["arithmetic_check_pass"] for r in checks),
        "maximum_recomputed_candidate_error": max(r["candidate_rescore_max_absolute_error"] for r in comparisons),
        "maximum_order_multiplicity_log_ratio_for_any_8_node_pair": math.lgamma(9),
        "limitations": "Exact graph posteriors are exact conditional on the implemented scores. Native GLM scores use Laplace approximations. MAP/Hessian checks do not certify integration error. The representative is a Bayes action, not necessarily the MAP DAG. No hyperparameters, priors, directions, or recovery data were used to select a new model.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(); plan = read(source / "plan.json")
    completion = read(source / "analysis/completion.json")
    if completion["status"] != "complete" or completion["missing"] or completion["completed_smc_runs"] != 240:
        raise RuntimeError("follow-up requires the complete 240-fit diagnostic cohort")
    repo = Path(__file__).resolve().parents[3]
    for name in ("scoring.py", "config.py", "api.py", "core.py", "data.py"):
        path = f"src/ordersmc/{name}"
        if sha(repo / path) != plan["source_files"][path]:
            raise RuntimeError(f"the inferential implementation changed since the source run: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", {
        "schema": "ordersmc-readiness-followup-v1", "status": "post-hoc-diagnostic",
        "source_root": str(source), "source_plan_sha256": plan["plan_sha256"],
        "source_completion_sha256": sha(source / "analysis/completion.json"),
        "primary_orientation_threshold": .9, "sensitivity_thresholds": [.8,.9,.95],
        "cutoff_policy": "declared before computing this display; not optimized for truth, F1 or stability; report all three",
        "skeleton_policy": "unchanged representative DAG skeleton; no new adjacency selection or inference",
        "preserve_original_results": True,
        "code_sha256": {path:sha(repo/path) for path in ["src/ordersmc/uncertainty.py",
            "experiments/src/ordersmc_experiments/readiness_followup.py",
            "experiments/src/ordersmc_experiments/score_validation_v2.py"]},
    })
    uncertainty_report(source, args.output)
    diagnose_failure(source, args.output)
    write(args.output / "completion.json", {"status": "complete", "source_datasets": 60,
          "source_smc_runs": 240, "fresh_inference_fits": 0,
          "oracle_inverse_exact_controls": 3, "source_plan_sha256": plan["plan_sha256"]})


if __name__ == "__main__":
    main()
