from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ordersmc_experiments.paper_figures import (
    _bootstrap_mean,
    _require_complete,
    _save_figure,
    plot_algorithm_overview,
)


def test_bootstrap_is_keyed_and_deterministic() -> None:
    first = _bootstrap_mean([0.1, 0.2, 0.4], "fixed-key")
    second = _bootstrap_mean([0.1, 0.2, 0.4], "fixed-key")
    assert first == second
    assert first[0] <= 0.1 + 0.2 + 0.4
    assert second[1] >= 0.2


def test_save_figure_writes_all_publication_formats(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    outputs = _save_figure(figure, tmp_path / "test")
    assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 100 for path in outputs)


def test_algorithm_overview_is_standalone(tmp_path: Path) -> None:
    outputs = plot_algorithm_overview(tmp_path)
    assert len(outputs) == 3
    assert all(path.is_file() for path in outputs)


def test_figure_builder_rejects_an_incomplete_cohort(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "completion.json").write_text(
        '{"planned_runs": 10, "completed_runs": 9, "missing_or_failed_runs": 1}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="complete cohort"):
        _require_complete(tmp_path)


def test_corrected_clg_rows_replace_rather_than_pool_legacy_rows(tmp_path):
    import json
    import pandas as pd
    from ordersmc_experiments.paper_figures import _v3_stability_inputs
    old, new = tmp_path / "old", tmp_path / "new"
    for root, value in [(old, .2), (new, .99)]:
        (root / "analysis").mkdir(parents=True)
        (root / "validation").mkdir()
        rows = [{"dataset_id": "clg-0", "method": "ordersmc", "jaccard": value}]
        cells = [{"cell_id": "clg"}]
        if root == old:
            rows.append({"dataset_id": "mixed-0", "method": "ordersmc", "jaccard": .8})
            cells.append({"cell_id": "mixed"})
        pd.DataFrame(rows).to_csv(root / "analysis/seed_stability_dagbagm_metrics.csv", index=False)
        (root / "validation/prepared-audit.json").write_text(json.dumps({"cells": cells, "minimum_recall": .95}))
    (new / "plan.json").write_text(json.dumps({"datasets": [{"dataset_id": "clg-0", "cell_id": "clg"}]}))
    stability, cells, _ = _v3_stability_inputs(old, new)
    assert len(stability) == len(cells) == 2
    assert stability.set_index("dataset_id").loc["clg-0", "jaccard"] == .99
    assert stability.set_index("dataset_id").loc["mixed-0", "jaccard"] == .8
