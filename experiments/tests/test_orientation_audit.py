import math
from types import SimpleNamespace

import numpy as np
import pytest

from ordersmc.graphs import dag_to_cpdag
from ordersmc_experiments.orientation_audit import (
    chain_orientation_scores, direction_metrics, graph, summarize, vstructures,
)


def test_chain_reversal_is_equivalent_but_not_correct_arrows():
    truth = graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    reverse = graph(list(truth), [("b", "a"), ("c", "b")])
    result = direction_metrics(reverse, truth)
    assert result["markov_equivalent"]
    assert result["truth_compelled_arrows"] == 0
    assert result["cpdag_pair_state_shd"] == 0
    assert result["reversed_arrows"] == 2
    assert result["orientation_accuracy_on_recovered_adjacencies"] == 0


def test_collider_is_not_equivalent_to_chain():
    truth = graph(["a", "b", "c"], [("a", "b"), ("c", "b")])
    chain = graph(list(truth), [("a", "b"), ("b", "c")])
    assert len(dag_to_cpdag(truth).directed_edges) == 2
    result = direction_metrics(chain, truth)
    assert not result["markov_equivalent"]
    assert result["missing_vstructures"] == 1
    assert result["cpdag_pair_state_shd"] == 2


def test_loss_sensitivity_is_labelled_and_point_graph_not_posterior_confidence():
    truth = graph(["a", "b"], [("a", "b")])
    probabilities = {("a", "b"): .46, ("b", "a"): .45}
    pair = summarize(probabilities, truth, 1)
    directed = summarize(probabilities, truth, 1, loss="directed_edge_hamming")
    assert pair["skeleton_f1"] == 1 and directed["skeleton_f1"] == 0
    point = summarize({("b", "a"): 1.}, truth, 1, posterior=False)
    assert point["high_confidence_reversed_truth_edges"] is None


def test_chain_prior_decomposition_counts_extensions_not_edges():
    nodes = ["a", "b", "c"]
    table = SimpleNamespace(nodes=nodes, parent_sets=[], logprior=[], loglik=[])
    for child in range(3):
        rows = [[-1]] + [[p] for p in range(3) if p != child]
        table.parent_sets.append(np.array(rows))
        table.logprior.append(np.array([0., -1., -1.]))
        table.loglik.append(np.zeros(3))
    truth = graph(nodes, [("a", "b"), ("b", "c")])
    rows = {r["root"]: r for r in chain_orientation_scores(table, truth)}
    assert [rows[n]["linear_extensions"] for n in nodes] == [1, 2, 1]
    assert all(r["local_log_prior"] == -2 for r in rows.values())
    assert rows["b"]["order_log_weight"] == pytest.approx(-2 + math.log(2))
    assert rows["b"]["conditional_mec_order_weight"] == pytest.approx(.5)
    assert rows["b"]["conditional_mec_structure_weight"] == pytest.approx(1 / 3)
    assert all(not vstructures(graph(nodes, r["edges"])) for r in rows.values())


def test_direction_metrics_reject_cycles():
    with pytest.raises(ValueError, match="DAG"):
        graph(["a", "b"], [("a", "b"), ("b", "a")])
