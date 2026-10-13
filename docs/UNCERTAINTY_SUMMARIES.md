# Uncertainty-aware graph display

This optional display preserves a representative DAG's adjacencies while
leaving directions unresolved when their conditional posterior support is low.
It is not a new posterior, CPDAG, or causal-identification claim. Existing DAG
and PIP output is unchanged.

```python
from ordersmc import optimal_bayes_action_dag, uncertainty_aware_graph

dag = optimal_bayes_action_dag(edge_probabilities, nodes, max_indegree=2)
display = uncertainty_aware_graph(
    edge_probabilities, dag, orientation_threshold=0.90
)
payload = display.to_dict()
# payload["directed_edges"]: retained arrows
# payload["undirected_edges"]: adjacencies with unresolved directions
# payload["edge_records"]: joint, reverse, adjacency and conditional probabilities
```

For each selected arrow u->v, retain its arrowhead only when
`P(u->v)/(P(u->v)+P(v->u)) >= 0.90`; otherwise show u--v. The denominator is
the adjacency probability and is exported separately. No direction is reversed
and no adjacency is selected or deleted by this function. Directed edges remain
a subset of the input DAG, ensuring an acyclic directed part and at least the
original DAG as a completion. This does not characterize an equivalence class.

The default 0.90 is a display policy, not a statistically estimated tuning
parameter. Report sensitivity and retained-arrow coverage alongside accuracy.
Do not count unresolved edges as correctly directed, use a lexicographic
tie-break to imply evidence for a direction, or interpret high confidence as
proof of correctness under a misspecified model.

Finite-sample Monte Carlo variation near any display cutoff can still change
arrowheads. A reproducible adjacency pattern does not guarantee reproducible
directions; neither guarantees that the graph is correct. Averaging multiple
seeds, if done, must be labelled and must not replace individual-run stability.

The completed exploratory comparison is documented in
`results/publication/2026-09-15-mixed-followup/README.md`. All three declared
cutoffs (0.80, 0.90, 0.95) are retained, including worse individual comparisons.
The original full-DAG recovery is preserved.
