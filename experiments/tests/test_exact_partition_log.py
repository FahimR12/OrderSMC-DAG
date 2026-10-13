import numpy as np
from ordersmc_experiments.adapter_validation import adapter_fixture_table
from ordersmc_experiments.exact_partition_log import exact_partition_log_dp
from ordersmc_experiments.exact_structure import exact_structure_dp
from ordersmc_experiments.structure_mcmc import exact_partition_pips


def test_positive_reference_matches_two_independent_exact_formulations():
    table = adapter_fixture_table()
    positive = exact_partition_log_dp(table)
    sink = exact_structure_dp(table, compute_graph_size=False)
    enumerated = exact_partition_pips(table)
    assert np.isclose(positive.log_normalizer, sink.log_normalizer, atol=1e-12)
    for edge, value in positive.edge_probs.items():
        assert np.isclose(value, sink.edge_probs[edge], atol=1e-12)
        assert np.isclose(value, enumerated.get(edge, 0.0), atol=1e-12)


def test_restricted_support_and_extreme_dynamic_range_remain_valid():
    table = adapter_fixture_table()
    # Force a -> b -> c with a completely forbidden c -> a edge.
    table.parent_sets[0] = np.asarray([[-1, -1]], dtype=np.int32)
    table.loglik[0] = np.asarray([0.0])
    table.logprior[0] = np.asarray([0.0])
    for child, parent in [(1, 0), (2, 1)]:
        table.loglik[child] = np.asarray([
            0.0 if set(row[row >= 0]) == {parent} else -100000.0
            for row in table.parent_sets[child]
        ])
    result = exact_partition_log_dp(table)
    assert result.edge_probs['a', 'b'] == 1.0
    assert result.edge_probs['b', 'c'] == 1.0
    assert result.edge_probs['c', 'a'] == 0.0
