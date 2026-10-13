from __future__ import annotations

from itertools import combinations

import pytest

import ordersmc_experiments.reliability as reliability_module
from ordersmc_experiments import (
    bootstrap_reliability as exported_bootstrap_reliability,
)
from ordersmc_experiments import fit_reliability as exported_fit_reliability
from ordersmc_experiments.reliability import bootstrap_reliability, fit_reliability


def _pairs(nodes):
    return tuple(combinations(nodes, 2))


def test_reliability_api_is_exported_from_package():
    assert exported_fit_reliability is fit_reliability
    assert exported_bootstrap_reliability is bootstrap_reliability


def test_dataset_weighting_gives_each_dataset_total_mass_one():
    node_sets = {"small": ("a", "b", "c"), "large": ("a", "b", "c", "d")}
    probabilities = {
        "small": {edge: 0.0 for edge in _pairs(node_sets["small"])},
        "large": {edge: 1.0 for edge in _pairs(node_sets["large"])},
    }
    outcomes = {
        "small": {edge: 0 for edge in probabilities["small"]},
        "large": {edge: 1 for edge in probabilities["large"]},
    }

    result = fit_reliability(
        probabilities,
        outcomes,
        nodes=node_sets,
        bins=1,
        minimum_effective_datasets=0.0,
    )

    reliability_bin = result.bins[0]
    assert reliability_bin.total_weight == pytest.approx(2.0)
    assert reliability_bin.mean_probability == pytest.approx(0.5)
    assert reliability_bin.observed_frequency == pytest.approx(0.5)
    assert reliability_bin.effective_dataset_count == pytest.approx(2.0)
    assert result.node_counts == (("large", 4), ("small", 3))
    assert result.expected_pair_counts == (("large", 6), ("small", 3))


def test_equal_pip_ties_use_dataset_and_unordered_edge_ids_deterministically():
    nodes = ("a", "b", "c", "d")
    edges = _pairs(nodes)
    probabilities = {
        "dataset": {tuple(reversed(edge)): 0.5 for edge in reversed(edges)}
    }
    outcomes = {
        "dataset": {edge: int(index >= 3) for index, edge in enumerate(edges)}
    }

    first = fit_reliability(
        probabilities,
        outcomes,
        nodes={"dataset": nodes},
        bins=2,
        minimum_effective_datasets=0.0,
    )
    second = fit_reliability(
        {"dataset": dict(reversed(list(probabilities["dataset"].items())))},
        {"dataset": dict(reversed(list(outcomes["dataset"].items())))},
        nodes={"dataset": tuple(reversed(nodes))},
        bins=2,
        minimum_effective_datasets=0.0,
    )

    assert first == second
    assert [reliability_bin.observed_frequency for reliability_bin in first.bins] == [
        0.0,
        1.0,
    ]


def test_low_effective_bin_merges_with_lower_pip_neighbour_on_distance_tie():
    probabilities = {
        "a": {("x", "y"): 0.25},
        "b": {("x", "y"): 0.25},
        "c": {("x", "y"): 0.50},
        "d": {("x", "y"): 0.75},
        "e": {("x", "y"): 0.75},
    }
    outcomes = {
        dataset_id: {("x", "y"): int(dataset_id >= "c")}
        for dataset_id in probabilities
    }

    result = fit_reliability(
        probabilities,
        outcomes,
        nodes={dataset_id: ("x", "y") for dataset_id in probabilities},
        bins=3,
        minimum_effective_datasets=1.5,
    )

    assert len(result.bins) == 2
    assert result.bins[0].initial_bin_indices == (0, 1)
    assert result.bins[1].initial_bin_indices == (2,)
    assert len(result.merges) == 1
    assert result.merges[0].deficient_initial_bins == (1,)
    assert result.merges[0].neighbour_initial_bins == (0,)


def test_bootstrap_is_deterministic_and_resamples_whole_datasets_not_edges():
    nodes = tuple(f"n{index}" for index in range(7))
    edges = {edge: 0.5 for edge in _pairs(nodes)}
    probabilities = {"negative": dict(edges), "positive": dict(edges)}
    outcomes = {
        "negative": {edge: 0 for edge in edges},
        "positive": {edge: 1 for edge in edges},
    }

    first = bootstrap_reliability(
        probabilities,
        outcomes,
        nodes={"negative": nodes, "positive": nodes},
        strata={"negative": "same", "positive": "same"},
        bins=1,
        minimum_effective_datasets=0.0,
        resamples=200,
        confidence=0.8,
        seed=19,
    )
    second = bootstrap_reliability(
        probabilities,
        outcomes,
        nodes={"negative": nodes, "positive": nodes},
        strata={"negative": "same", "positive": "same"},
        bins=1,
        minimum_effective_datasets=0.0,
        resamples=200,
        confidence=0.8,
        seed=19,
    )

    assert first == second
    assert first.bootstrap_unit == "complete graph-dataset pair"
    assert first.edge_resampling is False
    assert first.bins_recomputed_each_resample is True
    assert first.bands[0].observed_frequency_lower == pytest.approx(0.0)
    assert first.bands[0].observed_frequency_upper == pytest.approx(1.0)

    fixed_strata = bootstrap_reliability(
        probabilities,
        outcomes,
        nodes={"negative": nodes, "positive": nodes},
        strata={"negative": "negative", "positive": "positive"},
        bins=1,
        minimum_effective_datasets=0.0,
        resamples=100,
        confidence=0.8,
        seed=19,
    )
    assert fixed_strata.bands[0].observed_frequency_lower == pytest.approx(0.5)
    assert fixed_strata.bands[0].observed_frequency_upper == pytest.approx(0.5)


def test_bootstrap_recomputes_weighted_quantile_bins_for_every_draw(monkeypatch):
    calls = 0
    implementation = reliability_module._weighted_quantile_groups

    def counted(observations, bins):
        nonlocal calls
        calls += 1
        return implementation(observations, bins)

    monkeypatch.setattr(reliability_module, "_weighted_quantile_groups", counted)
    probabilities = {
        "one": {("a", "b"): 0.1, ("a", "c"): 0.2, ("b", "c"): 0.3},
        "two": {("a", "b"): 0.7, ("a", "c"): 0.8, ("b", "c"): 0.9},
    }
    outcomes = {
        "one": {("a", "b"): 0, ("a", "c"): 0, ("b", "c"): 0},
        "two": {("a", "b"): 1, ("a", "c"): 1, ("b", "c"): 1},
    }

    bootstrap_reliability(
        probabilities,
        outcomes,
        nodes={"one": ("a", "b", "c"), "two": ("a", "b", "c")},
        bins=2,
        minimum_effective_datasets=0.0,
        resamples=100,
        seed=7,
    )

    assert calls == 101  # one point fit plus one fresh binning per bootstrap draw


def test_invalid_or_misaligned_unordered_edge_tables_are_rejected():
    with pytest.raises(ValueError, match="identical dataset IDs"):
        fit_reliability(
            {"one": {("a", "b"): 0.5}},
            {"two": {("a", "b"): 1}},
            nodes={"one": ("a", "b")},
            bins=1,
        )
    with pytest.raises(ValueError, match="duplicate unordered edge"):
        fit_reliability(
            {"one": {("a", "b"): 0.5, ("b", "a"): 0.6}},
            {"one": {("a", "b"): 1, ("b", "a"): 1}},
            nodes={"one": ("a", "b")},
            bins=1,
        )


def test_complete_pair_universe_rejects_top_only_missing_and_extraneous_pairs():
    nodes = {"one": ("a", "b", "c")}
    top_only_probabilities = {"one": {("a", "b"): 0.9, ("a", "c"): 0.8}}
    top_only_outcomes = {"one": {("a", "b"): 1, ("a", "c"): 0}}
    with pytest.raises(ValueError, match="complete unordered-pair table.*missing"):
        fit_reliability(
            top_only_probabilities,
            top_only_outcomes,
            nodes=nodes,
            bins=1,
        )

    with pytest.raises(ValueError, match=r"extraneous=.*\('a', 'd'\)"):
        fit_reliability(
            {
                "one": {
                    ("a", "b"): 0.1,
                    ("a", "c"): 0.2,
                    ("b", "c"): 0.3,
                    ("a", "d"): 0.4,
                }
            },
            {
                "one": {
                    ("a", "b"): 0,
                    ("a", "c"): 0,
                    ("b", "c"): 1,
                    ("a", "d"): 1,
                }
            },
            nodes=nodes,
            bins=1,
        )


def test_node_schema_rejects_duplicate_nodes_and_wrong_dataset_ids():
    tables = {"one": {("a", "b"): 0.5}}
    truths = {"one": {("a", "b"): 1}}
    with pytest.raises(ValueError, match="duplicate node IDs"):
        fit_reliability(tables, truths, nodes={"one": ("a", "b", "b")}, bins=1)
    with pytest.raises(ValueError, match="exactly the reliability dataset IDs"):
        fit_reliability(tables, truths, nodes={"two": ("a", "b")}, bins=1)


def test_unattainable_final_effective_dataset_minimum_fails_closed():
    probabilities = {
        "one": {("a", "b"): 0.2},
        "two": {("a", "b"): 0.8},
    }
    outcomes = {
        "one": {("a", "b"): 0},
        "two": {("a", "b"): 1},
    }
    with pytest.raises(ValueError, match="cannot be met even after all possible"):
        fit_reliability(
            probabilities,
            outcomes,
            nodes={"one": ("a", "b"), "two": ("a", "b")},
            bins=2,
            minimum_effective_datasets=2.01,
        )


def test_bootstrap_rejects_incomplete_pair_tables_before_resampling():
    with pytest.raises(ValueError, match="complete unordered-pair table"):
        bootstrap_reliability(
            {"one": {("a", "b"): 0.5}},
            {"one": {("a", "b"): 1}},
            nodes={"one": ("a", "b", "c")},
            bins=1,
            minimum_effective_datasets=0.0,
            resamples=100,
        )
