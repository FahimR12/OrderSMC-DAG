"""Uncertainty-aware views of existing representative DAGs.

These are posterior display summaries, NOT CPDAGs or identified causal graphs.
They preserve the chosen DAG's skeleton and only remove arrowheads. They do
not change inference, select new adjacencies, or manufacture reproducibility
by imposing a preferred direction on a near tie.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import networkx as nx

Edge = tuple[str, str]


@dataclass(frozen=True)
class UncertaintyGraphSummary:
    nodes: tuple[str, ...]
    directed_edges: tuple[Edge, ...]
    undirected_edges: tuple[Edge, ...]
    orientation_threshold: float
    edge_records: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ordersmc-uncertainty-graph-v1",
            "semantics": "arrowhead masking of a representative DAG; not a CPDAG or a causal-identification guarantee",
            "nodes": list(self.nodes),
            "directed_edges": [list(e) for e in self.directed_edges],
            "undirected_edges": [list(e) for e in self.undirected_edges],
            "orientation_threshold": self.orientation_threshold,
            "skeleton_policy": "preserve every adjacency of the supplied representative DAG",
            "orientation_formula": "P(parent -> child) / (P(parent -> child) + P(child -> parent))",
            "edge_records": list(self.edge_records),
        }


def uncertainty_aware_graph(
    edge_probs: Mapping[Edge, float],
    representative_dag: nx.DiGraph,
    *,
    orientation_threshold: float = .90,
    roundoff_tolerance: float = 1e-8,
) -> UncertaintyGraphSummary:
    """Keep an existing arrow only if its conditional direction PIP meets a cutoff.

    For each selected adjacency let s=p_uv+p_vu. Its arrow u->v is retained
    iff s>0 and p_uv/s >= orientation_threshold; otherwise show u--v.
    Adjacency inclusion probability s is exported separately: conditional
    direction support is not a joint edge probability. No arrows are reversed.
    As the directed part is a subset of the input DAG, it stays acyclic and
    has at least one DAG completion (the original). This is not a CPDAG.
    """
    if isinstance(orientation_threshold, bool) or not math.isfinite(orientation_threshold) or not .5 < orientation_threshold <= 1.:
        raise ValueError("orientation_threshold must be finite and in (0.5, 1]")
    if not math.isfinite(roundoff_tolerance) or not 0 <= roundoff_tolerance <= 1e-8:
        raise ValueError("roundoff_tolerance must be in [0, 1e-8]")
    if not isinstance(representative_dag, nx.DiGraph) or representative_dag.is_multigraph() or not nx.is_directed_acyclic_graph(representative_dag):
        raise ValueError("representative_dag must be a simple directed acyclic graph")
    if any(not isinstance(node, str) for node in representative_dag):
        raise ValueError("node labels must be strings")
    nodes = tuple(sorted(representative_dag.nodes()))
    probabilities = {}
    for (u, v), raw in edge_probs.items():
        value = float(raw)
        if u not in nodes or v not in nodes or u == v:
            raise ValueError("edge probabilities contain an unknown node or self edge")
        if not math.isfinite(value) or value < -roundoff_tolerance or value > 1 + roundoff_tolerance:
            raise ValueError("edge probabilities must be finite and in [0, 1]")
        probabilities[u, v] = min(1., max(0., value))
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            total = probabilities.get((u, v), 0.) + probabilities.get((v, u), 0.)
            if total > 1 + roundoff_tolerance:
                raise ValueError("opposite-edge probabilities cannot sum to more than one")
            if total > 1:
                probabilities[u, v] = probabilities.get((u, v), 0.) / total
                probabilities[v, u] = probabilities.get((v, u), 0.) / total
    directed, undirected, records = [], [], []
    for u, v in sorted(representative_dag.edges()):
        forward, reverse = probabilities.get((u, v), 0.), probabilities.get((v, u), 0.)
        adjacency = forward + reverse
        confidence = forward / adjacency if adjacency > 0 else None
        retain = confidence is not None and confidence >= orientation_threshold
        if retain:
            directed.append((u, v))
        else:
            undirected.append(tuple(sorted((u, v))))
        records.append({
            "representative_parent": u, "representative_child": v,
            "adjacency_probability": adjacency, "forward_probability": forward,
            "reverse_probability": reverse, "conditional_direction_probability": confidence,
            "display_state": "directed" if retain else "undirected",
        })
    return UncertaintyGraphSummary(nodes, tuple(directed), tuple(sorted(undirected)),
                                   float(orientation_threshold), tuple(records))
