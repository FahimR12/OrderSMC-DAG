"""Figures and auditable publication addendum for the completed mixed follow-up."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tarfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ordersmc import uncertainty_aware_graph
from .paper_figures import _style, _save_figure
from .readiness_audits import read, write, sha, pips
from .readiness_followup import graph


def figures(source, output):
    _style()
    target = output / "figures"
    target.mkdir(exist_ok=True)
    full = pd.read_csv(source / "analysis/summary.csv")
    failure = pd.read_csv(output / "failure_graph_score_decomposition.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for variant, label, color in [("unrestricted_native", "Unrestricted mixed", "#0072B2"),
                                   ("paired_nongaussian", "Paired non-Gaussian", "#D55E00")]:
        rows = full[(full.variant == variant) & (full.method == "ordersmc")].sort_values("n_train")
        axes[0].plot(rows.n_train, rows.directed_f1, "o-", color=color, label=label)
        exact = failure[failure.variant == variant].sort_values("n_train")
        axes[1].plot(exact.n_train, exact.exact_order_directed_f1, "o-", color=color, label="Exact: " + label)
        axes[1].plot(exact.n_train, exact.smc_directed_f1, "x--", color=color, label="SMC: " + label)
    for axis in axes[:2]:
        axis.set_xscale("log"); axis.set_xticks([250,1000,5000], ["250","1,000","5,000"])
        axis.set_xlabel("Training observations"); axis.set_ylim(0, 1.02); axis.set_ylabel("Directed F1")
        axis.legend(fontsize=7, frameon=False)
    axes[0].set_title("a  Cohort mean (10 graphs)")
    axes[1].set_title("b  Failing graph: replicate 7")
    nongaussian = failure[failure.variant == "paired_nongaussian"].sort_values("n_train")
    x = np.arange(len(nongaussian)); width=.25
    for i, (field, label, color) in enumerate([
        ("delta_log_evidence", "Local evidence", "#D55E00"),
        ("delta_edge_log_prior", "Edge prior", "#009E73"),
        ("delta_log_order_multiplicity", "Order multiplicity", "#0072B2")]):
        axes[2].bar(x + (i-1)*width, nongaussian[field], width, label=label, color=color)
    axes[2].axhline(0, color="black", linewidth=.6)
    axes[2].set_xticks(x, ["250","1,000","5,000"])
    axes[2].set_xlabel("Training observations"); axes[2].set_ylabel("Log-weight difference: representative − truth")
    axes[2].set_title("c  Failure decomposition")
    axes[2].legend(fontsize=7, frameon=False)
    figure.suptitle("Exploratory mixed-data diagnostic — no retuning", fontsize=12)
    figure.tight_layout()
    generated = _save_figure(figure, target / "mixed_recovery_and_failure")

    record = read(source / "cases/rep007/paired_nongaussian-n05000.json")
    truth = graph(record["truth_edges"], record["variable_types"])
    exact = record["references"]["exact-order"]
    inferred = graph(exact["representative_edges"], truth)
    masked = uncertainty_aware_graph(pips(exact), inferred)
    positions = {}
    layers = list(nx.topological_generations(truth))
    for layer, nodes in enumerate(layers):
        for i, node in enumerate(sorted(nodes)):
            positions[node] = ((i - (len(nodes)-1)/2) * 1.8, -layer)
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 6.7))
    labels = {node: node + "\n" + record["variable_types"][node] for node in truth}
    colors = {"con":"#BFE5D7", "cat":"#F7DEAE", "bin":"#BCD9EB"}
    for index, axis in enumerate(axes):
        directed = list(truth.edges()) if index == 0 else list(inferred.edges()) if index == 1 else list(masked.directed_edges)
        nx.draw_networkx_nodes(truth, positions, node_color=[colors[record["variable_types"][n]] for n in truth], node_size=660, ax=axis)
        nx.draw_networkx_labels(truth, positions, labels, font_size=7, ax=axis)
        for edge in directed:
            u, v = edge
            bend = .35 if abs(positions[u][0]-positions[v][0]) < .1 and abs(positions[u][1]-positions[v][1]) > 1.5 else .07
            nx.draw_networkx_edges(graph([edge], truth), positions, edgelist=[edge],
                edge_color="#444444" if edge in truth.edges() else "#C23934", ax=axis,
                arrows=True, arrowsize=14, node_size=660, connectionstyle=f"arc3,rad={bend}")
        if index == 2:
            undirected = graph(masked.undirected_edges, truth)
            nx.draw_networkx_edges(undirected, positions, ax=axis, edge_color="#666666", style="dashed",
                                   node_size=660, arrows=True, arrowstyle="-", connectionstyle="arc3,rad=0.35")
        axis.set_axis_off(); axis.margins(.2)
    axes[0].set_title("a  Generating DAG")
    axes[1].set_title("b  Exact-posterior representative\nDirected F1 = 0.16")
    axes[2].set_title("c  90% direction display\n11 of 12 asserted arrows are wrong")
    figure.suptitle("Confidence is not correctness under model mismatch", fontsize=13)
    figure.text(.5, .02, "Replicate 7, non-Gaussian, n=5,000. Red: incorrect arrows. Dashed: unresolved direction. Not a CPDAG.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0,.04,1,.96))
    generated += _save_figure(figure, target / "failure_truth_dag_uncertainty")
    write(target / "manifest.json", {"status":"complete", "source_plan_sha256":read(source/"plan.json")["plan_sha256"],
        "figure_sha256":{p.name:sha(p) for p in generated},
        "interpretation":"descriptive post-hoc figures; curves show dataset means, not confidence intervals; preserve the failure"})


def report(source, output):
    # This reviewed narrative and its numeric captions describe this exact
    # post-hoc cohort, not arbitrary future data passed to the diagnostic code.
    if read(source / "plan.json")["plan_sha256"] != "1244d65c8223f2c22f65e9f7a996499186aef7c38b2796045fb6c17178284746":
        raise ValueError("this diagnostic narrative is bound to the reviewed seed-20260918-v2 plan")
    if read(output / "completion.json")["status"] != "complete":
        raise RuntimeError("follow-up analysis is incomplete")
    audit = read(output / "failure_audit.json")
    if not audit["all_checked_score_arithmetic_passed"] or audit["maximum_recomputed_candidate_error"] > 1e-6:
        raise RuntimeError("numerical audit is not clean: inspect its raw results before writing the interpretation")
    figures(source, output)
    stability = pd.read_csv(output/"uncertainty_stability.csv")
    summary = stability.groupby("threshold").agg(identical=("all_seed_summaries_identical","sum"),
                                                 jaccard=("mean_summary_jaccard","mean"))
    lines=["| Conditional direction cutoff | Identical across four seeds | Mean summary Jaccard |",
           "|---:|---:|---:|"]
    for threshold, row in summary.iterrows():
        lines.append(f"| {threshold:.2f} | {int(row.identical)}/60 | {row.jaccard:.4f} |")
    text = """# Mixed-data failure diagnosis and uncertainty-aware summaries

This is a post-hoc diagnostic addendum, not a corrected inference model or a new
confirmatory experiment. It uses all 60 completed dataset conditions and 240
SMC fits from `seed-20260918-v2`. No original result was changed, excluded,
relabelled as successful, or replaced by a new high-performing fit.

## Outcome

The severe non-Gaussian failure is primarily a score-model mismatch, supported
by the matched-data control, exact-target results, score decomposition, residual
diagnostics and independent arithmetic checks. It is not resolved by simply
increasing SMC particles, removing the ordering-induced prior, or changing the
representative graph to an uncertainty-aware display. The native model has
Gaussian linear continuous-child regressions within discrete contexts and
linear binary/multinomial logits. These assumptions are not invariant under the
nonlinear transformation used by this stress test.

In replicate 7, non-Gaussian directed F1 falls from 0.545 at n=250 to 0.100 at
n=5,000 for SMC. The exact order and exact structure representatives both have
F1=0.160 at n=5,000. The remaining SMC/exact point-graph difference concerns a
near-tied direction; it does not explain the broad wrong orientation pattern.

## Score, prior, and graph-construction separation

For the exact-order representative versus the generating DAG at n=5,000:

- Local log marginal evidence difference: **+157.414** in favor of the wrong graph.
- Edge log prior difference: **−1.000**, favoring the generating graph.
- Log topological-order multiplicity difference: **−0.693**, also favoring truth.
- Structure-target log posterior ratio: **+156.414**.
- Order-target log posterior ratio: **+155.721**.

For a graph G the order-target unnormalized weight is
`L(G) * exp(sum_i(log_evidence_i + log_prior_i))`, where `L(G)` counts its
topological orders. The structure target omits `L(G)`. These graph-to-graph
ratios cancel the corresponding normalizers; the representative is a Bayes
action, not necessarily the MAP DAG. An eight-node order-count ratio can change
the log weight by at most log(8!)=10.605. It cannot account for the observed
157-unit score preference. At n=250/1000 the local score gaps are +11.764/+34.226.

Every possible parent family up to indegree two is present. Recomputing the
1,392 candidate-family scores across the six paired case conditions agrees with
the saved tables to 2.9e-8 maximum absolute difference. All 83 independent
true/representative-family arithmetic checks pass; the largest difference is
9.3e-8. Discrete and Gaussian families use independent closed forms; GLMs use
a separate trust-region MAP optimizer and independently evaluated Hessian.
This checks implementation arithmetic, **not** the exactness of the Laplace
integral approximation. Exact graph posteriors are exact conditional on those
implemented local scores, not necessarily exact model evidences or causal truth.

The two continuous variables in this failure have approximately Gaussian
within-context residuals before transformation. At n=5,000 their mean residual
skewness is 0.004/−0.056 before versus 0.869/0.728 after transformation; mean
excess kurtosis changes from 0.043/−0.192 to 1.590/1.111. The nonlinear map also
changes the link between a transformed continuous parent and a discrete child.

## Oracle inverse control — diagnostic only

Undoing the **known synthetic** map reconstructs the original data to 2.7e-15
and recovers its exact order PIPs to less than 7e-13 RMSE at all three n values.
At n=5,000, the recovered native-data representative F1 is 0.833, versus 0.160
for the transformed-data exact representative. This identifies the stress
transformation as the source of the difference while holding graph, data draws,
score settings and prior fixed. It does not make every native-data direction
identifiable, and it is **not** a deployable inverse transform for unknown real
data or a new result to substitute into the original benchmark table.

## New display: keep connections, leave uncertain directions unresolved

The optional API is `ordersmc.uncertainty_aware_graph(probabilities, dag)`.
For each adjacency in the existing representative DAG, let

`s = P(u -> v) + P(v -> u)` and `q = P(u -> v) / s`.

Keep the existing u->v arrow only if q >= 0.90; otherwise display u--v.
When s=0, direction is unresolved. Export s separately: a large conditional
direction probability does not imply a large joint edge probability. No
adjacencies are added/deleted, and no arrow is reversed. The directed part
therefore remains acyclic and has the original DAG as a completion.

This is an **uncertainty-aware posterior display, not a CPDAG**, a new posterior,
or a causal-identification guarantee. It does not imply all displayed DAG
completions are Markov-equivalent or supported by the same likelihood.

The 0.90 display cutoff was declared before computing these summaries, after
the original failure had been inspected; it is exploratory. Fixed sensitivity
cutoffs 0.80 and 0.95 are also reported, not searched to choose the best result.
Original directed F1, skeleton F1 and PIPs remain unchanged. Precision/recall
among asserted arrows are accompanied by arrow coverage and unresolved counts;
unresolved edges are not counted as correctly directed edges.

Original full DAGs are identical across four seeds in 45/60 conditions (mean
Jaccard 0.9787); their skeletons are identical in all 60 conditions. Results for
the new display, comparing both arrow directions and unresolved states:

TABLE

At 0.90, reproducibility improves to 54/60 identical displays, but is **not
guaranteed**: probabilities near the new cutoff can make arrowheads appear or
disappear. Some individual conditions have lower display Jaccard than the
original DAG Jaccard, even though cohort-average agreement improves. These are
four-seed, same-data results, not resampling stability or a many-seed guarantee.

Most importantly, the failing n=5,000 graph still has **11 incorrect arrows
among 12 asserted arrows**, plus one unresolved adjacency. Higher display
stability cannot rescue a confidently wrong posterior. The weakness remains
visible in the tables and graph figure.

## Files

- `protocol.json`, `completion.json`: source identity, thresholds, scope and completion.
- `uncertainty_graphs/`: every seed/exact graph at all three display cutoffs;
  edge-wise adjacency, joint direction and conditional direction probabilities.
- `uncertainty_runs.csv`, `uncertainty_dataset_means.csv`, `uncertainty_summary.csv`:
  original recovery plus asserted-arrow performance and coverage; seeds are
  averaged within dataset before the ten-graph mean.
- `uncertainty_stability.csv`: all per-dataset, per-cutoff seed comparisons.
- `failure_graph_score_decomposition.csv`, `failure_local_scores.csv`: complete
  global and child-family evidence/prior decomposition for both variants and all n.
- `failure_score_checks.csv`: all independent numerical checks, including optimizer
  status and gradient norms; optimizer status alone does not override residual checks.
- `failure_residual_moments.csv`: all true-family continuous residual contexts.
- `oracle_inverse_control.csv`: the labelled, non-deployable inverse control.
- `figures/`: two diagnostic figures in PDF, SVG and PNG; source numbers are in
  the CSVs above. Curves are descriptive means without confidence intervals.
- `SHA256SUMS`: checksums for this addendum; original fits remain untouched.

The publication copy additionally includes `original_readiness_artifacts.tar.gz`
(original data, candidates, results, plan and source snapshot), `followup_code/`
(the diagnosis/report implementation), and `source_manifest.json` (member hashes).
The reviewed narrative and numeric captions are explicitly bound to the original
plan hash; they cannot silently be reused for a different cohort.

## What remains before the metadata study

Do not freeze full metadata inference on the claim that this case has been fixed.
The next modeling step should be a general, explicitly specified treatment of
non-Gaussian continuous mechanisms and transformed continuous parents, validated
across fresh mixed datasets without knowing their synthetic inverse. This is a
new modeling decision and has not been silently implemented or tuned here.
Any such change requires new score checks and affected benchmark reruns; this
display-only addition does not change existing inferential results. The separate
protein-pathway strong-correlation calibration mismatch also remains unresolved.
Keep failures, direction uncertainty and distinctions between sampling accuracy,
graph recovery and biological validation in the manuscript.
""".replace("TABLE", "\n".join(lines))
    (output/"README.md").write_text(text)
    files=sorted(p for p in output.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (output/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(output)}\n" for p in files))


def package(source, output, publication):
    if publication.exists():
        raise FileExistsError(f"publication addendum already exists: {publication}")
    shutil.copytree(output, publication)
    with tarfile.open(publication/"original_readiness_artifacts.tar.gz", "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
    repo=Path(__file__).resolve().parents[3]
    for relative in ["src/ordersmc/uncertainty.py", "experiments/src/ordersmc_experiments/readiness_followup.py",
                     "experiments/src/ordersmc_experiments/readiness_followup_report.py",
                     "experiments/src/ordersmc_experiments/score_validation_v2.py"]:
        target=publication/"followup_code"/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/relative,target)
    write(publication/"source_manifest.json", {"source_result_root":str(source),"followup_root":str(output),
          "original_members":{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob("*")) if p.is_file() and "__pycache__" not in p.parts},
          "status":"post-hoc diagnostic addendum; no corrected scientific fits"})
    files=sorted(p for p in publication.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (publication/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(publication)}\n" for p in files))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--publication",type=Path)
    args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve()
    report(source,output)
    if args.publication is not None:
        package(source,output,args.publication.resolve())
    print(f"Completed diagnostic report: {output / 'README.md'}")


if __name__ == "__main__":
    main()
