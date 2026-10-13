import itertools
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ordersmc import UnifiedFamilyScorer, uncertainty_aware_graph
from ordersmc_experiments.readiness_followup import linear_extensions, mask_metrics, independent_family_check


@pytest.mark.parametrize("edges,nodes", [([], ['a','b','c']), ([('a','b'),('b','c')], ['a','b','c']),
    ([('a','c'),('b','c')], ['a','b','c']), ([('a','c'),('b','d')], ['a','b','c','d'])])
def test_linear_extension_counts_match_direct_permutation_enumeration(edges, nodes):
    graph=nx.DiGraph(); graph.add_nodes_from(nodes); graph.add_edges_from(edges)
    brute=sum(all(order.index(u)<order.index(v) for u,v in edges) for order in itertools.permutations(nodes))
    assert linear_extensions(graph)==brute


def test_uncertain_edges_are_not_counted_as_correct_asserted_arrows():
    graph=nx.DiGraph([('a','b'),('b','c')])
    summary=uncertainty_aware_graph({('a','b'):.5,('b','a'):.5,('b','c'):.99},graph)
    metrics=mask_metrics(summary,set(graph.edges()))
    assert metrics['correct_asserted_arrows']==1
    assert metrics['unresolved_edges']==1
    assert metrics['asserted_true_edge_recall']==.5


def test_independent_reconstruction_covers_all_native_score_branches():
    rng=np.random.default_rng(31)
    data=pd.DataFrame({'x':rng.normal(size=60),'z':rng.integers(0,2,60),
                       'b':rng.integers(0,2,60),'c':rng.integers(0,3,60)})
    scorer=UnifiedFamilyScorer(data,{'x':'con','z':'bin','b':'bin','c':'cat'},
                              {'x':1,'z':2,'b':2,'c':3})
    for child,parents in [('x',('z',)),('b',('z',)),('b',('x','z')),('c',('x','z'))]:
        checked=independent_family_check(scorer,child,parents)
        assert abs(checked['reference_log_score']-scorer.score(child,parents).log_marginal)<1e-5
