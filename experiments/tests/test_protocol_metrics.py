from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import networkx as nx
import numpy as np
import pytest

from ordersmc.core import OrderCandidateTable, order_smc

from ordersmc_experiments.exact_order import exact_order_dp
from ordersmc_experiments.exact_structure import exact_structure_dp
from ordersmc_experiments.metrics import (
    ReferenceCoverageDiagnostics,
    build_posterior_dag_draw_bundle,
    dagbagm_graph_recovery_metrics,
    evaluate_edge_probabilities,
    exact_posterior_dag_bundle,
    high_posterior_region_coverage,
    posterior_dag_bundle_from_ordersmc_samples,
    threshold_attainment_curve,
    graph_size_wasserstein,
)


def test_dagbagm_metrics_count_reversal_for_direction_but_not_skeleton() -> None:
    nodes = ["a", "b", "c", "d"]
    truth = nx.DiGraph()
    truth.add_nodes_from(nodes)
    truth.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    estimated = nx.DiGraph()
    estimated.add_nodes_from(nodes)
    estimated.add_edges_from([("b", "a"), ("b", "c"), ("a", "c")])

    metrics = dagbagm_graph_recovery_metrics(
        estimated,
        truth,
        nodes,
        node_types={"a": "con", "b": "bin", "c": "cat", "d": "con"},
    )
    assert metrics["skeleton_tpr"] == pytest.approx(2 / 3)
    assert metrics["skeleton_fdr"] == pytest.approx(1 / 3)
    assert metrics["skeleton_f1"] == pytest.approx(2 / 3)
    assert metrics["directed_cross_type_tpr"] == pytest.approx(1 / 3)
    assert metrics["directed_cross_type_fdr"] == pytest.approx(2 / 3)
    assert metrics["directed_cross_type_f1"] == pytest.approx(1 / 3)
    assert metrics["directed_con_to_bin_false_negative_count"] == 1
    assert metrics["directed_bin_to_con_false_positive_count"] == 1
    assert metrics["directed_bin_to_cat_f1"] == pytest.approx(1.0)


def test_dagbagm_empty_graph_uses_false_edge_count_not_power() -> None:
    nodes = ["a", "b", "c"]
    truth = nx.DiGraph()
    truth.add_nodes_from(nodes)
    estimated = nx.DiGraph()
    estimated.add_nodes_from(nodes)
    estimated.add_edges_from([("a", "b"), ("a", "c")])

    metrics = dagbagm_graph_recovery_metrics(estimated, truth, nodes)
    assert metrics["skeleton_false_positive_count"] == 2
    assert metrics["directed_false_positive_count"] == 2
    assert metrics["skeleton_fdr"] == pytest.approx(1.0)
    assert np.isnan(metrics["skeleton_tpr"])
    assert np.isnan(metrics["skeleton_f1"])


def test_posterior_evaluation_includes_mixed_recovery_for_representative_dag() -> None:
    nodes = ["a", "b", "c"]
    truth = nx.DiGraph()
    truth.add_nodes_from(nodes)
    truth.add_edges_from([("a", "b"), ("b", "c")])
    probabilities = {
        (parent, child): float(truth.has_edge(parent, child))
        for parent in nodes
        for child in nodes
        if parent != child
    }
    metrics = evaluate_edge_probabilities(
        probabilities,
        truth,
        nodes,
        representative_dag=truth.copy(),
        node_types={"a": "con", "b": "bin", "c": "cat"},
    )
    assert metrics["skeleton_tpr"] == pytest.approx(1.0)
    assert metrics["directed_cross_type_fdr"] == pytest.approx(0.0)
    assert metrics["directed_cross_type_f1"] == pytest.approx(1.0)


def test_approximate_graph_size_output_uses_the_protocol_acyclic_grid() -> None:
    table = OrderCandidateTable(
        nodes=["a", "b", "c"],
        parent_sets=[
            np.asarray([[-1], [1], [2]], dtype=np.int32),
            np.asarray([[-1], [0], [2]], dtype=np.int32),
            np.asarray([[-1], [0], [1]], dtype=np.int32),
        ],
        loglik=[np.asarray([0.0, 0.2, 0.1])] * 3,
        logprior=[np.zeros(3)] * 3,
    )
    result = order_smc(
        table,
        particles=24,
        betas=[0.0, 1.0],
        mh_steps=0,
        seed=7,
        verbose=False,
    )
    mass = result.graph_size_distribution(table)
    assert set(mass) <= {0, 1, 2}
    exact_order_mass = exact_order_dp(table).graph_size_probs
    assert exact_order_mass is not None
    distance = graph_size_wasserstein(mass, exact_order_mass)
    assert distance["k_cap"] == 2
    assert 0.0 <= distance["graph_size_W1_edges"] <= 2.0
    assert distance["support_id"] == mass.support_id
    assert graph_size_wasserstein(mass, mass)["graph_size_W1_edges"] == pytest.approx(0.0)

    exact_structure_mass = exact_structure_dp(table).graph_size_probs
    assert exact_structure_mass is not None
    with pytest.raises(ValueError, match="matching targets"):
        graph_size_wasserstein(mass, exact_structure_mass)


def test_exact_graph_size_normalization_is_stable_for_large_log_score_offsets() -> None:
    """Large-data likelihood offsets must not corrupt normalized graph mass."""

    offset = -1.0e12
    table = OrderCandidateTable(
        nodes=["a", "b", "c"],
        parent_sets=[
            np.asarray([[-1], [1], [2]], dtype=np.int32),
            np.asarray([[-1], [0], [2]], dtype=np.int32),
            np.asarray([[-1], [0], [1]], dtype=np.int32),
        ],
        loglik=[
            np.asarray([offset, offset + 0.2, offset + 0.1]),
            np.asarray([offset, offset + 0.3, offset - 0.1]),
            np.asarray([offset, offset - 0.2, offset + 0.4]),
        ],
        logprior=[np.zeros(3) for _ in range(3)],
    )
    mass = exact_order_dp(table).graph_size_probs
    assert mass is not None
    assert math.fsum(mass.values()) == pytest.approx(1.0, abs=8e-16)


def _hpr_table(*, likelihood_shift: float = 0.0) -> OrderCandidateTable:
    return OrderCandidateTable(
        nodes=["a", "b", "c"],
        parent_sets=[
            np.asarray([[-1], [1], [2]], dtype=np.int32),
            np.asarray([[-1], [0], [2]], dtype=np.int32),
            np.asarray([[-1], [0], [1]], dtype=np.int32),
        ],
        loglik=[
            np.asarray([likelihood_shift, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
        ],
        logprior=[np.zeros(3) for _ in range(3)],
    )


def _run_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bundle(
    draws,
    weights,
    table: OrderCandidateTable,
    *,
    target: str = "order",
    run: str,
):
    return build_posterior_dag_draw_bundle(
        draws,
        weights,
        table,
        target_label=target,
        run_ids=[_run_digest(run)] * len(draws),
    )


def test_high_posterior_region_coverage_is_deterministic_and_method_neutral() -> None:
    nodes = ["a", "b", "c"]
    table = _hpr_table()
    method = [[("a", "b")], [("a", "b"), ("b", "c")], []]
    method_weights = [0.5, 0.25, 0.25]
    method_bundle = _bundle(method, method_weights, table, run="method")
    reference_bundle = exact_posterior_dag_bundle(table, target_label="order")
    first = high_posterior_region_coverage(
        method_bundle,
        reference_bundle,
        reference_bundle,
        reference_is_exact=True,
    )
    second = high_posterior_region_coverage(
        method_bundle,
        reference_bundle,
        reference_bundle,
        reference_is_exact=True,
    )
    assert first == second
    assert first.definition_region_mass >= 0.90
    assert first.reference_coverage > 0.0
    assert first.coverage_ratio == pytest.approx(
        first.method_coverage / first.reference_coverage
    )
    assert len(first.anchors) == 6
    assert first.method_bundle_sha256 == method_bundle.bundle_sha256
    assert first.candidate_target_id == method_bundle.candidate_target_id
    assert reference_bundle.bundle_kind == "exact"
    assert reference_bundle.source_result_sha256 is not None

    with pytest.raises(ValueError, match="q=min"):
        high_posterior_region_coverage(
            method_bundle,
            reference_bundle,
            reference_bundle,
            reference_is_exact=True,
            maximum_anchors=11,
        )
    with pytest.raises(ValueError, match="mass 0.90"):
        high_posterior_region_coverage(
            method_bundle,
            reference_bundle,
            reference_bundle,
            reference_is_exact=True,
            region_mass=0.91,
        )

    sampled_reference = _bundle(
        [[], [("a", "b")]], [0.5, 0.5], table, run="changed-exact"
    )
    with pytest.raises(ValueError, match="must agree with provenance"):
        high_posterior_region_coverage(
            method_bundle,
            reference_bundle,
            sampled_reference,
            reference_is_exact=True,
        )


def test_hpr_bundle_validates_support_hashes_and_ordersmc_pushforward() -> None:
    table = _hpr_table()
    result = order_smc(
        table,
        particles=24,
        betas=[0.0, 1.0],
        mh_steps=1,
        seed=13,
        verbose=False,
    )
    samples = result.sample_posterior_dags(table, draws=40, seed=22)
    bundle = posterior_dag_bundle_from_ordersmc_samples(samples, table)
    assert bundle.target_label == "order"
    assert bundle.support_id.startswith("candidate-sha256:")
    assert bundle.candidate_target_id.startswith("candidate-target-sha256:")
    assert len(set(bundle.draw_ids)) == len(bundle.draw_ids)

    with pytest.raises(ValueError, match="outside the candidate support"):
        _bundle(
            [[("a", "c"), ("b", "c")]],
            [1.0],
            table,
            run="unsupported-family",
        )
    with pytest.raises(ValueError, match="run_id"):
        build_posterior_dag_draw_bundle(
            [[]], [1.0], table, target_label="order", run_ids=["not-a-digest"]
        )

    tampered = _bundle([[]], [1.0], table, run="tampered")
    object.__setattr__(tampered, "bundle_sha256", "0" * 64)
    with pytest.raises(ValueError, match="content-hash verification"):
        high_posterior_region_coverage(
            tampered, bundle, bundle, reference_is_exact=True
        )


def test_high_posterior_region_coverage_enforces_reference_resolution() -> None:
    table = _hpr_table()
    draws = [[], [("a", "b")]] * 4
    weights = [1.0] * len(draws)
    method = _bundle(draws, weights, table, run="method-simulation")
    definition = _bundle(draws, weights, table, run="definition-simulation")
    evaluation = _bundle(draws, weights, table, run="evaluation-simulation")
    diagnostics = ReferenceCoverageDiagnostics(
        definition_bundle_sha256=definition.bundle_sha256,
        evaluation_bundle_sha256=evaluation.bundle_sha256,
        definition_weight_ess=10_000,
        evaluation_weight_ess=10_000,
        anchor_min_ess=12_000,
        region_indicator_ess=12_000,
        heldout_coverage_mcse=0.0,
    )
    # Caller-filled diagnostics are evidence, not proof. Until the named
    # estimators and between-run checks are computed in trusted code, the
    # simulation-reference branch must remain fail-closed.
    with pytest.raises(ValueError, match="trusted implementation"):
        high_posterior_region_coverage(
            method,
            definition,
            evaluation,
            reference_is_exact=False,
            diagnostics=diagnostics,
        )
    with pytest.raises(ValueError, match="must agree with provenance"):
        high_posterior_region_coverage(
            method,
            definition,
            evaluation,
            reference_is_exact=True,
            diagnostics=None,
        )

    tiny = _bundle([[], [("a", "b")]], [0.5, 0.5], table, run="tiny")
    tiny_diagnostics = replace(
        diagnostics,
        definition_bundle_sha256=tiny.bundle_sha256,
        evaluation_bundle_sha256=tiny.bundle_sha256,
    )
    with pytest.raises(ValueError, match="run IDs must be disjoint"):
        high_posterior_region_coverage(
            method,
            tiny,
            tiny,
            reference_is_exact=False,
            diagnostics=tiny_diagnostics,
        )


def test_high_posterior_region_coverage_enforces_target_support_and_independent_halves() -> None:
    table = _hpr_table()
    draws = [[], [("a", "b")]]
    method = _bundle(draws, [0.5, 0.5], table, run="method")
    definition = _bundle(draws, [0.5, 0.5], table, run="definition")
    evaluation = _bundle(draws, [0.5, 0.5], table, run="evaluation")
    wrong_target = _bundle(
        draws, [0.5, 0.5], table, target="structure", run="wrong-target"
    )
    with pytest.raises(ValueError, match="not_applicable_target_mismatch"):
        high_posterior_region_coverage(
            method,
            wrong_target,
            wrong_target,
            reference_is_exact=True,
        )

    changed_table = _hpr_table(likelihood_shift=0.01)
    wrong_target_content = _bundle(
        draws, [0.5, 0.5], changed_table, run="wrong-content"
    )
    with pytest.raises(ValueError, match="not_applicable_target_mismatch"):
        high_posterior_region_coverage(
            method,
            wrong_target_content,
            wrong_target_content,
            reference_is_exact=True,
        )

    with pytest.raises(TypeError, match="PosteriorDAGDrawBundle"):
        high_posterior_region_coverage(
            draws, definition, evaluation, reference_is_exact=True
        )

    diagnostics = ReferenceCoverageDiagnostics(
        definition_bundle_sha256=definition.bundle_sha256,
        evaluation_bundle_sha256=evaluation.bundle_sha256,
        definition_weight_ess=2.0,
        evaluation_weight_ess=2.0,
        anchor_min_ess=2_000,
        region_indicator_ess=2_000,
        heldout_coverage_mcse=0.01,
    )
    shared_method = _bundle(draws, [0.5, 0.5], table, run="definition")
    with pytest.raises(ValueError, match="evaluated-method and reference"):
        high_posterior_region_coverage(
            shared_method,
            definition,
            evaluation,
            reference_is_exact=False,
            diagnostics=diagnostics,
        )


def test_threshold_attainment_curve_keeps_right_censored_runs() -> None:
    curve = threshold_attainment_curve(
        [1.0, 2.0, 2.0, 4.0],
        [True, True, False, False],
    )
    assert [(point.time, point.at_risk) for point in curve] == [
        (1.0, 4),
        (2.0, 3),
        (4.0, 1),
    ]
    assert curve[0].cumulative_attainment == pytest.approx(0.25)
    assert curve[1].cumulative_attainment == pytest.approx(0.5)
    assert curve[-1].censored == 1
    assert curve[-1].cumulative_attainment == pytest.approx(0.5)
