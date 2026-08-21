"""Utilities for turning posterior edge probabilities into a DAG."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

Edge = Tuple[str, str]


def dag_from_edge_probs(
    edge_probs: Dict[Edge, float],
    nodes: Sequence[str],
    *,
    threshold: float = 0.5,
) -> nx.DiGraph:
    """Greedily retain high-probability edges while preserving acyclicity."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for (parent, child), probability in sorted(edge_probs.items(), key=lambda item: -item[1]):
        if probability < threshold:
            break
        if parent == child or graph.has_edge(parent, child):
            continue
        if nx.has_path(graph, child, parent):
            continue
        graph.add_edge(parent, child, posterior_probability=float(probability))
    return graph


def edges_above_threshold(
    edge_probs: Dict[Edge, float],
    nodes: Sequence[str],
    *,
    threshold: float = 0.5,
) -> List[Edge]:
    """Return an acyclic posterior-threshold edge list."""
    return sorted(dag_from_edge_probs(edge_probs, nodes, threshold=threshold).edges())

