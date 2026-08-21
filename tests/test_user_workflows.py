from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ordersmc import (
    OrderSMCConfig,
    build_candidates,
    fit,
    infer_metadata_and_encode,
    load_candidate_table,
    metadata_from_mapping,
    metadata_to_mapping,
    save_candidate_table,
    validate_data,
)
from ordersmc.cli import main
from ordersmc.persistence import load_mapping, save_mapping


def make_continuous_data(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(55)
    x = rng.normal(size=rows)
    y = 1.3 * x + rng.normal(scale=0.45, size=rows)
    z = -0.9 * y + rng.normal(scale=0.55, size=rows)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def tiny_config(seed: int = 3) -> OrderSMCConfig:
    return OrderSMCConfig.preset(
        "quick",
        particles=12,
        num_betas=5,
        mh_steps=1,
        mh_steps_resample=2,
        screen_k=2,
        pair_rescue_anchor_k=1,
        pair_rescue_k=1,
        parent_pool_max=2,
        edge_penalty=1.0,
        seed=seed,
    )


def test_metadata_config_checkpoint_and_result_artifacts(tmp_path: Path) -> None:
    data = make_continuous_data()
    encoded, vtypes, cards = infer_metadata_and_encode(data)
    validate_data(encoded, vtypes, cards, max_parents=2)

    metadata_path = tmp_path / "metadata.json"
    save_mapping(metadata_to_mapping(vtypes, cards), metadata_path)
    loaded_types, loaded_cards = metadata_from_mapping(load_mapping(metadata_path))
    assert loaded_types == vtypes
    assert loaded_cards == cards

    config = tiny_config()
    table = build_candidates(encoded, vtypes, cards, config=config, verbose=False)
    table_path = save_candidate_table(table, tmp_path / "candidates.pkl.gz")
    restored = load_candidate_table(table_path)
    assert restored.nodes == table.nodes
    for expected, actual in zip(table.parent_sets, restored.parent_sets):
        assert np.array_equal(expected, actual)

    result = fit(
        encoded,
        vtypes,
        cards,
        config=config,
        candidate_table=restored,
        verbose=False,
    )
    output = tmp_path / "result"
    result.save(output, threshold=0.5)
    expected_files = {
        "best_edges.csv",
        "edge_confidence.csv",
        "posterior_dag.csv",
        "smc_diagnostics.csv",
        "summary.json",
        "run.json",
        "orders.npy",
        "weights.npy",
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    confidence = pd.read_csv(output / "edge_confidence.csv")
    assert {
        "PosteriorProbability",
        "ReverseProbability",
        "SkeletonProbability",
        "OrientationConfidence",
        "MonteCarloSE",
        "MC95Lower",
        "MC95Upper",
        "Confidence",
    } <= set(confidence.columns)
    assert confidence["PosteriorProbability"].between(0.0, 1.0).all()
    assert confidence["MC95Lower"].between(0.0, 1.0).all()
    assert confidence["MC95Upper"].between(0.0, 1.0).all()


def test_cli_fit_resume_summarize_and_multiseed(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    make_continuous_data(70).to_csv(data_path, index=False)
    metadata_path = tmp_path / "metadata.json"
    encoded_path = tmp_path / "encoded.csv"

    assert main([
        "inspect", str(data_path), "--metadata-out", str(metadata_path),
        "--encoded-output", str(encoded_path),
    ]) == 0
    assert metadata_path.exists() and encoded_path.exists()

    fit_output = tmp_path / "fit"
    common = [
        "--preset", "quick", "--particles", "10", "--betas", "4",
        "--mh-steps", "1", "--max-parents", "2", "--quiet",
    ]
    assert main([
        "fit", str(encoded_path), "--metadata", str(metadata_path),
        "--output", str(fit_output), "--save-table", *common,
    ]) == 0
    checkpoint = fit_output / "candidate_table.pkl.gz"
    assert checkpoint.exists()
    assert main(["summarize", str(fit_output), "--top", "3"]) == 0

    resumed_output = tmp_path / "resumed"
    assert main([
        "sample", str(checkpoint), "--output", str(resumed_output), *common,
    ]) == 0
    assert (resumed_output / "edge_confidence.csv").exists()

    multiseed_output = tmp_path / "multiseed"
    assert main([
        "multiseed", str(encoded_path), "--metadata", str(metadata_path),
        "--output", str(multiseed_output), "--seeds", "7,8", *common,
    ]) == 0
    consensus = pd.read_csv(multiseed_output / "consensus_edge_confidence.csv")
    assert {"MeanPosterior", "PosteriorSD", "SelectionFrequency", "Seeds"} <= set(consensus.columns)
    assert set(consensus["Seeds"]) == {2}
    assert (multiseed_output / "consensus_dag.csv").exists()


def test_validation_rejects_unsafe_inputs() -> None:
    data = pd.DataFrame({"A": [0, 1, 0], "B": [1.0, np.nan, 2.0]})
    try:
        validate_data(data, {"A": "bin", "B": "con"}, {"A": 2, "B": 0})
    except ValueError as exc:
        assert "Missing values" in str(exc)
    else:
        raise AssertionError("missing values should fail validation")

