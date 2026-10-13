from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

from ordersmc_experiments.design import build_experiment_plan
from ordersmc_experiments.metadata_profiles import (
    MetadataDAGConfig,
    generate_metadata_dag_dataset,
    load_metadata_dag_artifact,
    load_metadata_profile,
    write_metadata_dag_artifact,
)
from ordersmc_experiments.section2_analysis import _paired_contrasts, analyse


def _small_config(**overrides):
    base = MetadataDAGConfig(
        profile_id="protein_pathway",
        dimension=12,
        n_train=40,
        n_test=30,
        pilot_size=300,
        calibration_scales=(.5, 1.),
        master_seed=917,
        pairing_id="protein-pathway-test",
    )
    return replace(base, **overrides)


def test_protein_pathway_profile_is_privacy_preserving_and_complete():
    profile = load_metadata_profile("protein_pathway")
    assert profile.participants == 222
    assert profile.total_variables == 196
    assert profile.biological_variables == profile.protein_variables == 193
    assert profile.rna_variables == 0
    assert len(profile.source_hashes) == 8


def test_metadata_generator_is_deterministic_and_respects_roles():
    first = generate_metadata_dag_dataset(_small_config())
    second = generate_metadata_dag_dataset(_small_config())
    assert first.train.equals(second.train)
    assert sorted(first.dag.edges()) == sorted(second.dag.edges())
    assert nx.is_directed_acyclic_graph(first.dag)
    assert max(dict(first.dag.in_degree()).values()) <= 2
    assert first.variable_types["age_z"] == "con"
    assert first.variable_types["female"] == first.variable_types["PD"] == "bin"
    assert first.dag.in_degree("age_z") == first.dag.in_degree("female") == 0
    assert first.dag.has_edge("age_z", "PD")
    assert first.manifest["calibration"]["selection_uses_graph_recovery"] is False


def test_m4_gaussian_mechanisms_and_graph_are_clg_compatible():
    data = generate_metadata_dag_dataset(_small_config(data_variant="gaussian", causal_regime="cg_compatible"))
    for parent, child in data.dag.edges():
        assert not (data.variable_types[parent] == "con" and data.variable_types[child] != "con")
    for node, mechanism in data.manifest["mechanisms"].items():
        if data.variable_types[node] == "con":
            assert mechanism["transform"]["kind"] != "monotone-quantile-map"


def test_paired_variants_share_truth_and_artifact_hashes_are_checked(tmp_path: Path):
    metadata = generate_metadata_dag_dataset(_small_config())
    gaussian = generate_metadata_dag_dataset(_small_config(data_variant="gaussian"))
    assert sorted(metadata.dag.edges()) == sorted(gaussian.dag.edges())
    destination = tmp_path / "artifact"
    write_metadata_dag_artifact(metadata, destination)
    loaded = load_metadata_dag_artifact(destination)
    pd.testing.assert_frame_equal(
        loaded.train, metadata.train, check_exact=False, rtol=0.0, atol=5e-16
    )
    assert sorted(loaded.dag.edges()) == sorted(metadata.dag.edges())


def test_section2_plans_are_staged_and_pairing_is_explicit():
    smoke = build_experiment_plan("s2", master_seed=20260914, profile="smoke")
    pilot = build_experiment_plan("s2", master_seed=20260914, profile="pilot")
    full = build_experiment_plan("s2", master_seed=20260915, profile="full")
    assert (len(smoke.datasets), len(smoke.runs)) == (1, 4)
    assert (len(pilot.datasets), len(pilot.runs)) == (23, 226)
    assert (len(full.datasets), len(full.runs)) == (340, 3320)
    assert {row.metadata_profile for row in pilot.datasets} == {"protein_pathway"}
    assert tuple(full.settings["profile_priority"]) == (
        "protein_pathway", "rna_pathway", "protein_molecule",
        "rna_molecule", "joint_molecule", "joint_pathway",
    )
    primary = next(row for row in pilot.datasets if row.role == "m1_matched")
    gaussian = next(row for row in pilot.datasets if row.role == "m2_gaussian_pair")
    assert primary.pairing_id == gaussian.pairing_id
    primary_smc = [
        row for row in pilot.runs
        if row.dataset_id == primary.dataset_id and row.method == "ordersmc"
    ]
    assert [row.seed_index for row in primary_smc if row.role == "evaluation"] == [0, 1, 2, 3]
    assert [
        row.seed_index for row in primary_smc
        if row.role == "reproducibility-extension"
    ] == [4, 5, 6, 7]


def test_section2_paired_contrasts_match_conditions_and_methods_correctly():
    common = {
        "metadata_profile": "protein_pathway",
        "pairing_id": "protein-primary",
        "replicate": 0,
    }
    metadata = [{**common, "method": "ordersmc", "directed_f1": .8}]
    gaussian = [{**common, "method": "ordersmc", "directed_f1": .6}]
    condition = _paired_contrasts(
        metadata,
        gaussian,
        contrast="metadata_minus_gaussian",
        formula="left-right",
        metrics=("directed_f1",),
        rng=np.random.Generator(np.random.PCG64DXSM(7)),
    )
    assert len(condition) == 1
    assert np.isclose(condition[0]["mean_difference"], .2)
    baseline = [{**common, "method": "order-mcmc", "directed_f1": .5}]
    methods = _paired_contrasts(
        metadata,
        baseline,
        contrast="ordersmc_minus_order_mcmc",
        formula="left-right",
        metrics=("directed_f1",),
        rng=np.random.Generator(np.random.PCG64DXSM(8)),
        match_method=False,
    )
    assert len(methods) == 1
    assert methods[0]["left_method"] == "ordersmc"
    assert methods[0]["right_method"] == "order-mcmc"
    assert np.isclose(methods[0]["mean_difference"], .3)


def test_section2_analysis_writes_prespecified_m2_contrast(tmp_path: Path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps({
        "experiment": "s2", "master_seed": 19, "plan_sha256": "test-plan"
    }))
    common = {
        "metadata_profile": "protein_pathway",
        "pairing_id": "protein-primary",
        "replicate": 0,
        "method": "ordersmc",
        "role": "evaluation",
        "causal_regime": "unrestricted",
        "negative_control": "none",
        "sample_multiplier": 1.0,
    }
    rows = [
        {
            **common,
            "dataset_id": "matched-0",
            "cell_id": "matched",
            "dataset_role": "m1_matched",
            "data_variant": "metadata",
            "directed_f1": .8,
        },
        {
            **common,
            "dataset_id": "gaussian-0",
            "cell_id": "gaussian",
            "dataset_role": "m2_gaussian_pair",
            "data_variant": "gaussian",
            "directed_f1": .6,
        },
    ]
    with (tmp_path / "analysis" / "dagbagm_metrics_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    analyse(tmp_path)
    with (
        tmp_path / "analysis" / "section2" / "m2_metadata_minus_gaussian.csv"
    ).open(newline="", encoding="utf-8") as handle:
        contrast = list(csv.DictReader(handle))
    assert len(contrast) == 1
    assert np.isclose(float(contrast[0]["mean_difference"]), .2)
