from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import ordersmc_experiments.generator as generator_module

from ordersmc_experiments.cli import main
from ordersmc_experiments.generator import (
    MixedDAGConfig,
    generate_mixed_dag_dataset,
    regenerate_e7_signal_variant,
    replay_mixed_dag_artifact,
    replay_mixed_dag_dataset,
    validate_mixed_dag_manifest,
    write_mixed_dag_artifact,
)


@lru_cache(maxsize=1)
def _pilot_dataset():
    return generate_mixed_dag_dataset(
        MixedDAGConfig(
            d=12,
            n_train=35,
            n_test=12,
            pilot_size=250,
            graph_family="random_order",
            master_seed=771,
            max_graph_attempts=1_000,
        )
    )


def _fingerprint_without_self(manifest: dict[str, object]) -> str:
    body = dict(manifest)
    body.pop("artifact_fingerprint_sha256")
    payload = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_e1_density_is_a_protocol_default_not_a_silent_generator_cap() -> None:
    e1 = MixedDAGConfig(
        d=6,
        n_train=60,
        expected_indegree=1.5,
        max_indegree=2,
        experiment_id="e1",
        cell_id="d006-r010-mixed",
        master_seed=20260807,
    )
    assert e1.protocol_deviations() == ()
    wrong_density = MixedDAGConfig(
        d=6,
        n_train=60,
        expected_indegree=2.0,
        max_indegree=2,
        experiment_id="e1",
        cell_id="d006-r010-mixed",
        master_seed=20260807,
    )
    assert "expected_indegree" in wrong_density.protocol_deviations()

    nodes = tuple(f"X{index:03d}" for index in range(e1.d))
    graph, _, _, _, _, _, details, _ = generator_module._generate_graph(e1, nodes)
    assert 1.0 <= graph.number_of_edges() / e1.d <= 1.8
    assert details["edge_probability"] < 1.0
    assert details["allowed_average_indegree_interval"] == [1.0, 1.8]


def test_extreme_probability_amendment_is_explicit_and_legacy_safe() -> None:
    legacy = MixedDAGConfig()
    amended = MixedDAGConfig(max_extreme_probability_fraction=0.05)
    diagnostic_only = MixedDAGConfig(enforce_extreme_probability_fraction=False)
    assert "max_extreme_probability_fraction" not in generator_module._manifest_config(
        legacy
    )
    assert generator_module._manifest_config(amended)[
        "max_extreme_probability_fraction"
    ] == 0.05
    assert "maximum_extreme_probability_fraction" not in legacy.protocol_deviations()
    assert "maximum_extreme_probability_fraction" in amended.protocol_deviations()
    assert "enforce_extreme_probability_fraction" not in generator_module._manifest_config(
        legacy
    )
    assert generator_module._manifest_config(diagnostic_only)[
        "enforce_extreme_probability_fraction"
    ] is False
    assert "extreme_probability_fraction_is_diagnostic_only" in (
        diagnostic_only.protocol_deviations()
    )
    with pytest.raises(ValueError, match="max_extreme_probability_fraction"):
        MixedDAGConfig(max_extreme_probability_fraction=0.26)


def test_mixed_type_regimes_have_frozen_counts_and_distinct_seeded_assignments() -> None:
    nodes = tuple(f"X{index:03d}" for index in range(12))
    expected = {
        "balanced_mixed": {"con": 4, "bin": 4, "cat": 4},
        "continuous_dominant_mixed": {"con": 7, "bin": 3, "cat": 2},
        "discrete_dominant_mixed": {"con": 2, "bin": 5, "cat": 5},
    }
    assignments = {}
    for regime, counts in expected.items():
        config = MixedDAGConfig(
            d=12,
            variable_type_regime=regime,
            master_seed=20260903,
            experiment_id="v1",
            cell_id=f"d012-{regime}",
        )
        variable_types, _ = generator_module._draw_types(config, nodes)
        assignments[regime] = variable_types
        assert {
            kind: sum(value == kind for value in variable_types.values())
            for kind in ("con", "bin", "cat")
        } == counts
    assert len({tuple(sorted(row.items())) for row in assignments.values()}) == 3
    assert "variable_type_regime" not in generator_module._manifest_config(
        MixedDAGConfig()
    )
    assert generator_module._manifest_config(
        MixedDAGConfig(variable_type_regime="continuous_dominant_mixed")
    )["variable_type_regime"] == "continuous_dominant_mixed"


@pytest.mark.parametrize(
    "regime", ["continuous_dominant_mixed", "discrete_dominant_mixed"]
)
def test_nonbalanced_mixed_regime_manifest_replays(regime: str) -> None:
    dataset = generate_mixed_dag_dataset(
        MixedDAGConfig(
            d=8,
            n_train=24,
            n_test=12,
            pilot_size=250,
            expected_indegree=1.5,
            max_indegree=2,
            variable_type_regime=regime,
            enforce_extreme_probability_fraction=False,
            master_seed=20260903,
            experiment_id="v1",
            cell_id=f"d008-{regime}",
        )
    )
    replayed = replay_mixed_dag_dataset(dataset.manifest())
    assert replayed.variable_types == dataset.variable_types
    assert replayed.manifest() == dataset.manifest()


@pytest.mark.parametrize(
    "noise",
    [
        "gaussian",
        "student_t_df3",
        "student_t_df5",
        "centered_gamma_shape1_scale2",
    ],
)
def test_continuous_noise_variants_are_centered_and_unit_variance(noise: str) -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(20260903))
    values = generator_module._draw_continuous_noise(rng, noise, 200_000, 1.0)
    assert abs(float(np.mean(values))) < 0.035
    assert abs(float(np.var(values)) - 1.0) < 0.08


def test_empty_mixed_graph_and_nongaussian_manifest_replay() -> None:
    dataset = generate_mixed_dag_dataset(
        MixedDAGConfig(
            d=8,
            n_train=24,
            n_test=12,
            pilot_size=250,
            graph_family="empty",
            continuous_noise="student_t_df5",
            enforce_extreme_probability_fraction=False,
            master_seed=20260903,
            experiment_id="v2",
            cell_id="d008-empty-student-t-df5-balanced-mixed",
        )
    )
    assert dataset.dag.number_of_edges() == 0
    assert dataset.generation_diagnostics["cross_type_fraction"] == 0.0
    assert replay_mixed_dag_dataset(dataset.manifest()).manifest() == dataset.manifest()


def test_modular_graph_is_acyclic_and_respects_indegree() -> None:
    config = MixedDAGConfig(
        d=20,
        n_train=40,
        pilot_size=250,
        graph_family="modular",
        enforce_extreme_probability_fraction=False,
        master_seed=20260903,
        experiment_id="v2",
        cell_id="d020-modular-balanced-mixed",
    )
    nodes = tuple(f"X{index:03d}" for index in range(config.d))
    graph, order, *_ = generator_module._generate_graph(config, nodes)
    assert generator_module.nx.is_directed_acyclic_graph(graph)
    assert max(dict(graph.in_degree()).values()) <= config.max_indegree
    positions = {node: index for index, node in enumerate(order)}
    assert all(positions[parent] < positions[child] for parent, child in graph.edges())


def test_complete_manifest_and_raw_bundle_replay_are_deterministic(tmp_path: Path) -> None:
    dataset = _pilot_dataset()
    manifest = dataset.manifest()
    assert validate_mixed_dag_manifest(manifest) == dataset.config
    assert set(manifest["serialized_artifacts"]) == {
        "train.csv",
        "test.csv",
        "adjacency.csv",
    }
    inventory = manifest["seed_record_inventory"]
    # types/order/graph/train/test/candidate plus parameters/calibration per node
    assert inventory["accepted_record_count"] == 6 + 2 * dataset.config.d
    assert len(inventory["sha256"]) == 64

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_audit = write_mixed_dag_artifact(dataset, first)
    second_audit = write_mixed_dag_artifact(dataset, second)
    assert first_audit == second_audit
    assert {path.name for path in first.iterdir()} == {
        "train.csv",
        "test.csv",
        "adjacency.csv",
        "generator_manifest.json",
        "artifact_audit.json",
    }
    for filename in {path.name for path in first.iterdir()}:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    replayed = replay_mixed_dag_artifact(first)
    assert replayed.manifest() == dataset.manifest()


def test_complete_replay_rejects_mechanism_diagnostic_and_seed_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _pilot_dataset()
    # The preceding test performs a real deterministic regeneration.  Reuse
    # that accepted object here so these cases isolate field-level rejection.
    monkeypatch.setattr(
        generator_module, "generate_mixed_dag_dataset", lambda config: dataset
    )
    manifest = copy.deepcopy(dataset.manifest())
    node = sorted(manifest["mechanisms"])[0]
    parameters = manifest["mechanisms"][node]["parameters"]
    parameter_name = sorted(parameters)[0]
    parameters[parameter_name] = "tampered"
    manifest["artifact_fingerprint_sha256"] = _fingerprint_without_self(manifest)
    with pytest.raises(RuntimeError, match="mechanisms"):
        replay_mixed_dag_dataset(manifest)

    seed_manifest = copy.deepcopy(dataset.manifest())
    seed_manifest["seed_namespaces"]["train"]["seed"] += 1
    seed_manifest["seed_record_inventory"] = copy.deepcopy(
        seed_manifest["seed_record_inventory"]
    )
    seed_manifest["artifact_fingerprint_sha256"] = _fingerprint_without_self(seed_manifest)
    with pytest.raises(ValueError, match="seed record for train"):
        validate_mixed_dag_manifest(seed_manifest)

    diagnostic_manifest = copy.deepcopy(dataset.manifest())
    diagnostic_manifest["diagnostics"]["edge_count"] += 1
    diagnostic_manifest["artifact_fingerprint_sha256"] = _fingerprint_without_self(
        diagnostic_manifest
    )
    with pytest.raises(RuntimeError, match="diagnostics"):
        replay_mixed_dag_dataset(diagnostic_manifest)


def test_raw_bundle_hash_check_precedes_semantic_replay(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    write_mixed_dag_artifact(_pilot_dataset(), destination)
    with (destination / "train.csv").open("ab") as handle:
        handle.write(b"tamper\n")
    with pytest.raises(RuntimeError, match="train.csv.*frozen bytes"):
        replay_mixed_dag_artifact(destination)


def test_generate_cli_persists_invalid_configuration_audit(tmp_path: Path) -> None:
    output = tmp_path / "failed"
    assert main(
        [
            "generate",
            "--output",
            str(output),
            "--d",
            "2",
            "--n-train",
            "10",
            "--pilot-size",
            "100",
            "--master-seed",
            "9",
        ]
    ) == 1
    record = json.loads((output / "generation_failure.json").read_text(encoding="utf-8"))
    assert record["schema"] == "ordersmc-mixed-dag-generation-failure-v1"
    assert record["status"] == "failed"
    assert record["failure_type"] == "ValueError"
    assert record["request"]["d"] == 2
    assert len(record["request_sha256"]) == 64

    rejected_output = tmp_path / "rejected-mechanism"
    assert main(
        [
            "generate",
            "--output",
            str(rejected_output),
            "--d",
            "6",
            "--n-train",
            "8",
            "--n-test",
            "5",
            "--pilot-size",
            "100",
            "--max-parameter-attempts",
            "1",
            "--master-seed",
            "0",
            "--experiment-id",
            "e7",
            "--cell-id",
            "d006-r002-random",
        ]
    ) == 1
    rejected = json.loads(
        (rejected_output / "generation_failure.json").read_text(encoding="utf-8")
    )
    assert rejected["failure_type"] == "GeneratorFailure"
    assert set(rejected["diagnostics"]["accepted_structure"]["seed_records"]) == {
        "types",
        "order",
        "graph",
    }
    assert "parameter_rejections" in rejected["diagnostics"]


def test_protocol_default_generator_produces_byte_identical_complete_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = generate_mixed_dag_dataset(
        MixedDAGConfig(
            d=12,
            n_train=35,
            master_seed=771,
            experiment_id="artifact-acceptance",
            cell_id="d012-r003-random",
        )
    )
    assert dataset.manifest()["protocol_compliant"] is True
    first = tmp_path / "protocol-default-first"
    second = tmp_path / "protocol-default-second"
    write_mixed_dag_artifact(dataset, first)
    write_mixed_dag_artifact(dataset, second)
    assert {
        path.name: path.read_bytes() for path in first.iterdir()
    } == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    # A real independent replay is covered by the faster pilot fixture.  This
    # case makes the full frozen defaults pass through the exact same strict
    # manifest, seed-inventory, and raw-byte acceptance path.
    monkeypatch.setattr(
        generator_module, "generate_mixed_dag_dataset", lambda config: dataset
    )
    assert replay_mixed_dag_artifact(first).manifest() == dataset.manifest()


def test_e7_variant_bundle_replays_through_its_embedded_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = generate_mixed_dag_dataset(
        MixedDAGConfig(
            d=6,
            n_train=8,
            n_test=5,
            pilot_size=100,
            master_seed=0,
            experiment_id="e7",
            cell_id="d006-r002-random",
        )
    )
    weak = regenerate_e7_signal_variant(base, "weak")
    destination = tmp_path / "e7-weak"
    write_mixed_dag_artifact(weak, destination)
    # Regeneration mathematics and namespace separation are exercised directly
    # in test_protocol_generators.  Here reuse the already materialized objects
    # so this fixture isolates strict embedded-base dispatch and byte replay.
    monkeypatch.setattr(generator_module, "generate_mixed_dag_dataset", lambda config: base)
    monkeypatch.setattr(
        generator_module,
        "regenerate_e7_signal_variant",
        lambda replayed_base, signal: weak,
    )
    replayed = replay_mixed_dag_artifact(destination)
    assert replayed.signal_variant == "weak"
    assert replayed.manifest() == weak.manifest()
