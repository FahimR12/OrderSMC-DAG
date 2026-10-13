from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from ordersmc_experiments.generator import MixedDAGConfig, _generate_graph


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "known_topologies_v1.json"


def test_frozen_topologies_have_published_sizes_and_are_acyclic() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema"] == "ordersmc-known-topologies-v1"
    expected = {
        "asia": (8, 8, 2),
        "insurance": (27, 52, 3),
        "alarm": (37, 46, 4),
        "hailfinder": (56, 66, 4),
    }
    for name, (node_count, edge_count, maximum_indegree) in expected.items():
        graph = nx.DiGraph(tuple(map(tuple, fixture["networks"][name]["edges"])))
        assert graph.number_of_nodes() == node_count
        assert graph.number_of_edges() == edge_count
        assert nx.is_directed_acyclic_graph(graph)
        assert max(dict(graph.in_degree()).values()) == maximum_indegree


def test_fixed_topology_assignment_is_deterministic_and_all_mixed_directions_exist() -> None:
    config = MixedDAGConfig(
        d=8,
        n_train=20,
        pilot_size=100,
        max_indegree=2,
        fixed_topology_id="asia",
        experiment_id="v3",
        cell_id="asia-test",
    )
    nodes = [f"X{index:03d}" for index in range(8)]
    first = _generate_graph(config, nodes)
    second = _generate_graph(config, nodes)
    assert sorted(first[0].edges()) == sorted(second[0].edges())
    assert first[2] == second[2]
    assert len(first[6]["source_to_generated_node"]) == 8
    directions = {
        (first[2][parent], first[2][child])
        for parent, child in first[0].edges()
        if first[2][parent] != first[2][child]
    }
    assert len(directions) == 6

