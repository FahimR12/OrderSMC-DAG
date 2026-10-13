from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ordersmc_experiments.design import build_experiment_plan
from ordersmc_experiments.execution import _execution_config, _profile_config
from ordersmc_experiments.readiness import paired_variants, summarize_posterior, validated_probabilities
from ordersmc_experiments.readiness_audits import pips


def test_pair_transform_is_fixed_invertible_and_leaves_categories_unchanged():
    data = pd.DataFrame({"x": [-2., -1., 0., 1., 2.], "b": [0, 1, 0, 1, 1]})
    full = paired_variants(data, {"x": "con", "b": "bin"})
    prefix = paired_variants(data.iloc[:3], {"x": "con", "b": "bin"})
    for key in full:
        pd.testing.assert_frame_equal(full[key].iloc[:3], prefix[key])
        assert full[key].b.equals(data.b)
    y = full["paired_nongaussian"].x
    assert np.all(np.diff(y) > 0)
    assert np.allclose(np.sinh((np.arcsinh(y) - .5) / 1.25), data.x)


def test_m4_requires_gaussian_mechanisms_for_restricted_score():
    repo = Path(__file__).resolve().parents[2]
    execution = _execution_config(repo / "experiments/configs/execution_section2_pilot_v2.json")
    plan = build_experiment_plan("s2", master_seed=20260918, profile="pilot")
    m4 = asdict(next(d for d in plan.datasets if d.role == "m4_cg_compatible"))
    assert m4["data_variant"] == "gaussian"
    assert _profile_config(m4, execution, "pilot", 1).score_method == "clg-bic"
    for data_variant, causal_regime in [("metadata", "cg_compatible"), ("gaussian", "unrestricted"),
                                        ("metadata", "unrestricted")]:
        spec = {**m4, "data_variant": data_variant, "causal_regime": causal_regime}
        assert _profile_config(spec, execution, "pilot", 1).score_method == "native"
    execution["metadata_score_policy"]["default"] = "clg-bic"
    with pytest.raises(ValueError, match="Gaussian mechanisms"):
        _profile_config({**m4, "data_variant": "metadata"}, execution, "pilot", 1)


def test_saved_run_pips_use_parent_child_schema():
    assert pips({"edge_probabilities": [{"parent": "a", "child": "b", "probability": .25}]}) == {("a", "b"): .25}


def test_probability_boundaries_accept_only_small_roundoff():
    assert validated_probabilities({("a", "b"): 1 + 1e-12, ("b", "a"): -1e-12}) == {("a", "b"): 1., ("b", "a"): 0.}
    probabilities = validated_probabilities({("a", "b"): .6, ("b", "a"): .4 + 1e-12})
    assert np.isclose(sum(probabilities.values()), 1., rtol=0., atol=1e-15)
    for value in [-.1, 1.1, np.nan, np.inf]:
        with pytest.raises(ArithmeticError):
            validated_probabilities({("a", "b"): value})
    with pytest.raises(ArithmeticError, match="opposite-edge"):
        validated_probabilities({("a", "b"): .7, ("b", "a"): .4})


def test_posterior_summary_uses_certified_pair_state_action():
    import networkx as nx
    graph = nx.DiGraph([("a", "b"), ("b", "c")])
    probabilities = {(u, v): float(graph.has_edge(u, v)) for u in graph for v in graph if u != v}
    summary = summarize_posterior(probabilities, graph, tuple(graph), 2)
    assert summary["directed_f1"] == summary["skeleton_f1"] == 1.
    assert summary["representative_edges"] == [["a", "b"], ["b", "c"]]
