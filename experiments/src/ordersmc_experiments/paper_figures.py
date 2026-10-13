"""Deterministic, publication-oriented figures for the brain-study paper.

The figure pipeline consumes immutable result roots and writes PDF, SVG, and
300-dpi PNG versions of every figure.  Pilot cohorts are labelled in the
figure itself so they cannot be mistaken for confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "ordersmc",
    "order-mcmc",
    "partition-mcmc",
    "rev-mcmc",
    "bnlearn-hc-shared-score",
    "bnlearn-tabu-shared-score",
    "bnlearn-hc-bic-cg",
    "bnlearn-tabu-bic-cg",
)
METHOD_LABELS = {
    "ordersmc": "OrderSMC",
    "order-mcmc": "Order MCMC",
    "partition-mcmc": "Partition MCMC",
    "rev-mcmc": "REV MCMC",
    "bnlearn-hc-shared-score": "HC + shared score",
    "bnlearn-tabu-shared-score": "Tabu + shared score",
    "bnlearn-hc-bic-cg": "bnlearn HC (BIC-CG)",
    "bnlearn-tabu-bic-cg": "bnlearn Tabu (BIC-CG)",
}
# Okabe-Ito-derived, colour-vision-deficiency-safe palette.
METHOD_COLOURS = {
    "ordersmc": "#0072B2",
    "order-mcmc": "#E69F00",
    "partition-mcmc": "#009E73",
    "rev-mcmc": "#CC79A7",
    "bnlearn-hc-shared-score": "#7A7A7A",
    "bnlearn-tabu-shared-score": "#111111",
    "bnlearn-hc-bic-cg": "#56B4E9",
    "bnlearn-tabu-bic-cg": "#D55E00",
}
METHOD_MARKERS = {
    "ordersmc": "o",
    "order-mcmc": "s",
    "partition-mcmc": "^",
    "rev-mcmc": "D",
    "bnlearn-hc-shared-score": "v",
    "bnlearn-tabu-shared-score": "P",
    "bnlearn-hc-bic-cg": "X",
    "bnlearn-tabu-bic-cg": "*",
}
TYPE_CODES = ("con", "bin", "cat")
TYPE_LABELS = ("Continuous", "Binary", "Categorical")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ordersmc-paper-v1",
        }
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / "analysis" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing analysis input: {path}")
    return pd.read_csv(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_complete(root: Path) -> dict[str, object]:
    path = root / "analysis" / "completion.json"
    completion = _read_json(path)
    if (
        int(completion.get("planned_runs", -1))
        != int(completion.get("completed_runs", -2))
        or int(completion.get("missing_or_failed_runs", -1)) != 0
    ):
        raise RuntimeError(
            f"paper figures require a complete cohort; inspect {path}"
        )
    return completion


def _stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _bootstrap_mean(
    values: Sequence[float], key: str, replicates: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.Generator(np.random.PCG64DXSM(_stable_seed(key)))
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    return tuple(map(float, np.quantile(np.mean(array[indices], axis=1), [0.025, 0.975])))


def _panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _save_figure(figure: Figure, destination: Path) -> list[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs = [destination.with_suffix(extension) for extension in (".pdf", ".svg", ".png")]
    figure.savefig(
        outputs[0],
        bbox_inches="tight",
        metadata={"Creator": "OrderSMC-DAG paper figure pipeline", "CreationDate": None},
    )
    figure.savefig(
        outputs[1],
        bbox_inches="tight",
        metadata={"Creator": "OrderSMC-DAG paper figure pipeline", "Date": None},
    )
    figure.savefig(
        outputs[2],
        bbox_inches="tight",
        dpi=300,
        metadata={"Software": "OrderSMC-DAG paper figure pipeline"},
    )
    plt.close(figure)
    return outputs


def _write_source_data(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _profile(root: Path) -> str:
    plan = _read_json(root / "plan.json")
    settings = plan.get("settings", {})
    if not isinstance(settings, Mapping):
        return "unknown"
    return str(settings.get("profile", "unknown"))


def _pilot_stamp(figure: Figure, profile: str) -> None:
    if profile != "full":
        figure.text(
            0.995,
            0.006,
            f"{profile.upper()} COHORT — NOT CONFIRMATORY",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#A51C30",
            fontweight="bold",
        )


def plot_algorithm_overview(output_dir: Path) -> list[Path]:
    """Draw the methods overview as an editable vector figure."""

    _style()
    figure, axis = plt.subplots(figsize=(12.2, 4.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = (
        (
            0.015,
            0.24,
            0.15,
            0.54,
            "Mixed observations",
            "Continuous\nBinary\nCategorical",
            "#EAF3F8",
        ),
        (
            0.205,
            0.16,
            0.19,
            0.70,
            "Local family evidence",
            "Gaussian–NIG\nBernoulli–logistic\nCategorical–softmax\nDiscrete–BDeu",
            "#F6F0E4",
        ),
        (
            0.435,
            0.20,
            0.16,
            0.62,
            "Candidate families",
            "Screened parent pools\nComplete local scores\nDeclared indegree bound",
            "#EEF5EA",
        ),
        (
            0.635,
            0.16,
            0.16,
            0.70,
            "Distributed OrderSMC",
            "Tempered particles\nMPI substreams\nResample + rejuvenate\nWeighted orders",
            "#E9EEF8",
        ),
        (
            0.83,
            0.20,
            0.155,
            0.62,
            "Posterior output",
            "Directed edge PIPs\nUncertainty diagnostics\nBayes-action DAG",
            "#F5EAF2",
        ),
    )
    for x, y, width, height, title, body, colour in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.2,
            edgecolor="#2E4053",
            facecolor=colour,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height - 0.09,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1F2D3D",
        )
        axis.text(
            x + width / 2,
            y + height / 2 - 0.045,
            body,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.45,
            color="#263238",
        )
    for left, right in zip(boxes[:-1], boxes[1:]):
        start = (left[0] + left[2] + 0.006, 0.51)
        end = (right[0] - 0.006, 0.51)
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color="#455A64",
            )
        )
    axis.text(
        0.5,
        0.96,
        "OrderSMC for posterior structure learning with heterogeneous variables",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.045,
        "Target: candidate-restricted order-modular posterior; logistic and softmax evidence use proper-prior Laplace approximations",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#455A64",
    )
    return _save_figure(figure, output_dir / "figure_01_ordersmc_overview")


def _v1_dataset_rmse(v1_root: Path) -> pd.DataFrame:
    runs = _read_csv(v1_root, "dagbagm_metrics_runs.csv")
    rows = runs[
        runs["method"].isin(["ordersmc", "order-mcmc"])
        & runs["pip_rmse_to_exact_order"].notna()
    ].copy()
    group_fields = [
        "dataset_id",
        "cell_id",
        "dimension",
        "n_train",
        "n_over_d",
        "variable_type_regime",
        "dataset_role",
        "method",
    ]
    return (
        rows.groupby(group_fields, as_index=False, dropna=False)["pip_rmse_to_exact_order"]
        .mean()
        .rename(columns={"pip_rmse_to_exact_order": "dataset_mean_pip_rmse"})
    )


def _v1_edge_pips(
    v1_root: Path, dataset_ids: set[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[dict[tuple[str, str], float]]] = {}
    for path in sorted((v1_root / "runs").glob("*.json")):
        result = _read_json(path)
        if result.get("status") != "complete":
            continue
        run = result["run"]
        if not isinstance(run, Mapping):
            continue
        dataset_id = str(run["dataset_id"])
        method = str(run["method"])
        if dataset_id not in dataset_ids or method not in {
            "exact-order-dp",
            "ordersmc",
            "order-mcmc",
        }:
            continue
        probabilities = {
            (str(row["parent"]), str(row["child"])): float(row["probability"])
            for row in result["edge_probabilities"]
        }
        grouped.setdefault((dataset_id, method), []).append(probabilities)
    output: dict[str, tuple[list[float], list[float]]] = {
        "ordersmc": ([], []),
        "order-mcmc": ([], []),
    }
    for dataset_id in sorted(dataset_ids):
        exact_runs = grouped.get((dataset_id, "exact-order-dp"), [])
        if len(exact_runs) != 1:
            raise RuntimeError(f"expected one exact result for {dataset_id}")
        exact = exact_runs[0]
        for method in output:
            method_runs = grouped.get((dataset_id, method), [])
            if not method_runs:
                raise RuntimeError(f"missing {method} results for {dataset_id}")
            for edge in sorted(exact):
                output[method][0].append(exact[edge])
                output[method][1].append(
                    float(np.mean([probabilities[edge] for probabilities in method_runs]))
                )
    return {
        method: (np.asarray(values[0]), np.asarray(values[1]))
        for method, values in output.items()
    }


def _mean_interval_plot(
    axis: Axes,
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    methods: Sequence[str],
    *,
    key_prefix: str,
) -> None:
    x_values = sorted(frame[x_field].dropna().unique())
    offsets = np.linspace(-0.13, 0.13, len(methods))
    for offset, method in zip(offsets, methods):
        subset = frame[frame["method"] == method]
        for index, x_value in enumerate(x_values):
            values = subset.loc[subset[x_field] == x_value, y_field].to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            lower, upper = _bootstrap_mean(values, f"{key_prefix}|{method}|{x_value}")
            mean = float(np.mean(values))
            xpos = index + offset
            axis.scatter(
                np.full(len(values), xpos),
                values,
                s=11,
                color=METHOD_COLOURS[method],
                alpha=0.25,
                linewidths=0,
                zorder=2,
            )
            axis.errorbar(
                xpos,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt=METHOD_MARKERS[method],
                color=METHOD_COLOURS[method],
                markersize=5,
                capsize=2,
                linewidth=1.1,
                zorder=4,
                label=METHOD_LABELS[method] if index == 0 else None,
            )
    axis.set_xticks(range(len(x_values)), [str(value) for value in x_values])


def plot_v1_exact_validation(
    v1_root: Path, output_dir: Path, source_dir: Path
) -> list[Path]:
    _style()
    dataset = _v1_dataset_rmse(v1_root)
    gate = _read_csv(v1_root, "exact_pip_gate.csv")
    primary = dataset[dataset["dataset_role"] == "primary"].copy()
    primary["cell_label"] = primary.apply(
        lambda row: f"d={int(row['dimension'])}\n{str(row['variable_type_regime']).replace('_mixed', '').replace('_', ' ')}",
        axis=1,
    )
    cell_order = [
        f"d={dimension}\n{regime}"
        for dimension in (8, 12)
        for regime in ("balanced", "continuous dominant", "discrete dominant")
    ]
    primary["cell_label"] = pd.Categorical(
        primary["cell_label"], categories=cell_order, ordered=True
    )
    _write_source_data(dataset, source_dir / "v1_dataset_pip_rmse.csv")
    _write_source_data(gate, source_dir / "v1_exact_pip_gate.csv")

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    axes = np.asarray(axes)
    methods = ("ordersmc", "order-mcmc")
    ordered = primary.sort_values("cell_label")
    ordered["cell_index"] = ordered["cell_label"].cat.codes
    offsets = {"ordersmc": -0.13, "order-mcmc": 0.13}
    for method in methods:
        subset = ordered[ordered["method"] == method]
        for cell_index in range(len(cell_order)):
            values = subset.loc[
                subset["cell_index"] == cell_index, "dataset_mean_pip_rmse"
            ].to_numpy(float)
            lower, upper = _bootstrap_mean(values, f"v1-primary|{method}|{cell_index}")
            mean = float(np.mean(values))
            xpos = cell_index + offsets[method]
            axes[0, 0].scatter(
                np.full(len(values), xpos),
                values,
                s=12,
                alpha=0.28,
                color=METHOD_COLOURS[method],
                linewidths=0,
            )
            axes[0, 0].errorbar(
                xpos,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt=METHOD_MARKERS[method],
                color=METHOD_COLOURS[method],
                capsize=2,
                markersize=5,
                label=METHOD_LABELS[method] if cell_index == 0 else None,
            )
    axes[0, 0].axhline(0.012, color="#A51C30", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(range(len(cell_order)), cell_order, rotation=24, ha="right")
    axes[0, 0].set_ylabel("Dataset-mean directed PIP RMSE")
    axes[0, 0].set_title("Exact-posterior accuracy across mixed compositions")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(axes[0, 0], "a")

    gate_primary = gate[gate["dataset_role"] == "primary"].copy()
    cell_lookup = {
        row.cell_id: (
            f"d={int(row.dimension)}\n"
            f"{str(row.cell_id).split('-', 2)[2].replace('_mixed', '').replace('_', ' ')}"
        )
        for row in gate_primary.itertuples()
    }
    gate_primary["cell_label"] = gate_primary["cell_id"].map(cell_lookup)
    gate_primary["cell_label"] = pd.Categorical(
        gate_primary["cell_label"], categories=cell_order, ordered=True
    )
    for method in methods:
        subset = gate_primary[gate_primary["method"] == method].sort_values("cell_label")
        axes[0, 1].plot(
            range(len(subset)),
            subset["empirical_p95_pip_rmse"],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLOURS[method],
            linewidth=1.4,
            markersize=5,
            label=METHOD_LABELS[method],
        )
    axes[0, 1].axhline(0.020, color="#A51C30", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(range(len(cell_order)), cell_order, rotation=24, ha="right")
    axes[0, 1].set_ylabel("Empirical 95th-percentile PIP RMSE")
    axes[0, 1].set_title("Tail-error gate")
    axes[0, 1].legend(frameon=False)
    _panel_label(axes[0, 1], "b")

    sample = dataset[
        (dataset["dimension"] == 12)
        & (dataset["variable_type_regime"] == "balanced_mixed")
    ].copy()
    ratios = [5, 10, 20, 50]
    for method in methods:
        subset = sample[sample["method"] == method]
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for ratio in ratios:
            values = subset.loc[
                subset["n_over_d"] == ratio, "dataset_mean_pip_rmse"
            ].to_numpy(float)
            lower, upper = _bootstrap_mean(values, f"v1-sample|{method}|{ratio}")
            means.append(float(np.mean(values)))
            lowers.append(lower)
            uppers.append(upper)
        means_array = np.asarray(means)
        axes[0, 2].errorbar(
            ratios,
            means,
            yerr=[means_array - lowers, np.asarray(uppers) - means_array],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLOURS[method],
            capsize=2,
            linewidth=1.4,
            label=METHOD_LABELS[method],
        )
    axes[0, 2].axhline(0.012, color="#A51C30", linestyle="--", linewidth=1)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xticks(ratios, [str(value) for value in ratios])
    axes[0, 2].set_xlabel("Training sample size / dimension")
    axes[0, 2].set_ylabel("Mean directed PIP RMSE")
    axes[0, 2].set_title("Sample-size sensitivity (d=12, balanced mixed)")
    axes[0, 2].legend(frameon=False)
    _panel_label(axes[0, 2], "c")

    primary_ids = set(primary["dataset_id"].astype(str))
    edge_pips = _v1_edge_pips(v1_root, primary_ids)
    for axis, method, label in zip(axes[1, :2], methods, ("d", "e")):
        exact, estimate = edge_pips[method]
        axis.hexbin(
            exact,
            estimate,
            gridsize=38,
            extent=(0, 1, 0, 1),
            mincnt=1,
            bins="log",
            cmap="Blues" if method == "ordersmc" else "Oranges",
            linewidths=0,
        )
        axis.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Exact directed-edge PIP")
        axis.set_ylabel(f"{METHOD_LABELS[method]} seed-mean PIP")
        axis.set_title(f"Edge-level posterior agreement: {METHOD_LABELS[method]}")
        axis.text(
            0.04,
            0.94,
            f"{len(primary_ids)} datasets; {len(exact):,} edge probabilities",
            transform=axis.transAxes,
            va="top",
            fontsize=7.5,
        )
        _panel_label(axis, label)

    for method in methods:
        values = np.sort(
            primary.loc[
                primary["method"] == method, "dataset_mean_pip_rmse"
            ].to_numpy(float)
        )
        axes[1, 2].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=METHOD_COLOURS[method],
            linewidth=1.6,
            label=METHOD_LABELS[method],
        )
    axes[1, 2].axvline(0.012, color="#A51C30", linestyle="--", linewidth=1)
    axes[1, 2].set_xlabel("Dataset-mean directed PIP RMSE")
    axes[1, 2].set_ylabel("Empirical cumulative fraction")
    axes[1, 2].set_ylim(0, 1.02)
    axes[1, 2].set_title("Distribution over independent datasets")
    axes[1, 2].legend(frameon=False)
    _panel_label(axes[1, 2], "f")

    figure.suptitle(
        "Exact mixed-data posterior validation",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    return _save_figure(figure, output_dir / "figure_02_v1_exact_validation")


def _distribution_panel(
    axis: Axes,
    frame: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    ylabel: str,
    *,
    log_scale: bool = False,
) -> None:
    positions = np.arange(len(methods))
    values = [
        frame.loc[frame["method"] == method, metric]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(float)
        for method in methods
    ]
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLOURS[method])
        patch.set_alpha(0.22)
        patch.set_edgecolor(METHOD_COLOURS[method])
    rng = np.random.Generator(np.random.PCG64DXSM(_stable_seed(metric)))
    for position, method, method_values in zip(positions, methods, values):
        jitter = rng.uniform(-0.16, 0.16, len(method_values))
        axis.scatter(
            position + jitter,
            method_values,
            s=9,
            alpha=0.35,
            color=METHOD_COLOURS[method],
            linewidths=0,
        )
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in methods], rotation=22, ha="right")
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")


def plot_v1_stability(
    v1_root: Path, output_dir: Path, source_dir: Path
) -> list[Path]:
    _style()
    stability = _read_csv(v1_root, "seed_stability_dagbagm_metrics.csv")
    stability = stability[stability["method"].isin(["ordersmc", "order-mcmc"])].copy()
    _write_source_data(stability, source_dir / "v1_seed_stability.csv")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.65))
    specifications = (
        ("mean_pairwise_pip_rmse", "Pairwise PIP RMSE", True),
        ("mean_directed_graph_jaccard", "Directed-DAG Jaccard", False),
        ("mean_skeleton_graph_jaccard", "Skeleton Jaccard", False),
    )
    for label, axis, (metric, ylabel, log_scale) in zip("abc", axes, specifications):
        _distribution_panel(
            axis,
            stability,
            metric,
            ("ordersmc", "order-mcmc"),
            ylabel,
            log_scale=log_scale,
        )
        axis.set_title(ylabel)
        if "jaccard" in metric:
            axis.set_ylim(max(0.70, float(stability[metric].min()) - 0.03), 1.01)
        _panel_label(axis, label)
    figure.suptitle(
        "Across-seed reproducibility in exact mixed-data validation",
        fontsize=12.5,
        fontweight="bold",
    )
    figure.tight_layout()
    return _save_figure(figure, output_dir / "extended_data_01_v1_seed_stability")


def _type_metric_matrix(
    frame: pd.DataFrame, method: str, measure: str
) -> np.ndarray:
    subset = frame[frame["method"] == method]
    matrix = np.full((3, 3), np.nan)
    for parent_index, parent in enumerate(TYPE_CODES):
        for child_index, child in enumerate(TYPE_CODES):
            column = f"directed_{parent}_to_{child}_{measure}"
            if column not in subset:
                continue
            values = subset[column].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                matrix[parent_index, child_index] = float(values.mean())
    return matrix


def _plot_type_heatmaps(
    frame: pd.DataFrame,
    methods: Sequence[str],
    title: str,
    destination: Path,
    profile: str = "full",
) -> list[Path]:
    _style()
    measures = ("f1", "tpr", "fdr")
    figure, axes = plt.subplots(
        len(methods),
        len(measures),
        figsize=(10.2, 3.2 * len(methods)),
        squeeze=False,
    )
    for row_index, method in enumerate(methods):
        for column_index, measure in enumerate(measures):
            axis = axes[row_index, column_index]
            matrix = _type_metric_matrix(frame, method, measure)
            cmap = "viridis" if measure != "fdr" else "viridis_r"
            axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")
            for parent_index in range(3):
                for child_index in range(3):
                    value = matrix[parent_index, child_index]
                    text = "NA" if not np.isfinite(value) else f"{value:.2f}"
                    colour = "white" if np.isfinite(value) and (value < 0.25 or value > 0.78) else "black"
                    axis.text(child_index, parent_index, text, ha="center", va="center", color=colour, fontsize=8)
            axis.set_xticks(range(3), TYPE_LABELS, rotation=24, ha="right")
            axis.set_yticks(range(3), TYPE_LABELS)
            axis.set_xlabel("Child type")
            if column_index == 0:
                axis.set_ylabel(f"{METHOD_LABELS[method]}\nParent type")
            if row_index == 0:
                axis.set_title({"f1": "Directed F1", "tpr": "Directed TPR", "fdr": "Directed FDR"}[measure])
    figure.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    _pilot_stamp(figure, profile)
    figure.subplots_adjust(
        left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.38, hspace=0.42
    )
    return _save_figure(figure, destination)


def plot_v1_type_recovery(
    v1_root: Path, output_dir: Path, source_dir: Path
) -> list[Path]:
    runs = _read_csv(v1_root, "dagbagm_metrics_runs.csv")
    runs = runs[
        (runs["dataset_role"] == "primary")
        & runs["method"].isin(["ordersmc", "order-mcmc"])
    ].copy()
    numeric = [column for column in runs if column.startswith("directed_")]
    dataset = runs.groupby(["dataset_id", "method"], as_index=False)[numeric].mean()
    _write_source_data(dataset, source_dir / "v1_type_recovery.csv")
    return _plot_type_heatmaps(
        dataset,
        ("ordersmc", "order-mcmc"),
        "Recovery across all parent–child variable-type combinations",
        output_dir / "extended_data_02_v1_type_recovery",
    )


def _v3_plot_frame(v3_root: Path) -> pd.DataFrame:
    averaged = _read_csv(v3_root, "seed_averaged_dagbagm_metrics.csv")
    runs = _read_csv(v3_root, "dagbagm_metrics_runs.csv")
    identity = (
        runs[["dataset_id", "n_train", "dataset_role"]]
        .drop_duplicates("dataset_id")
        .set_index("dataset_id")
    )
    averaged = averaged.copy()
    averaged["n_train"] = averaged["dataset_id"].map(identity["n_train"])
    averaged["dataset_role"] = averaged["dataset_id"].map(identity["dataset_role"])
    baseline = runs[runs["method"].str.startswith("bnlearn-")].copy()
    common = sorted(set(averaged.columns) & set(baseline.columns))
    frame = pd.concat([averaged[common], baseline[common]], ignore_index=True)
    frame["network"] = frame["dataset_id"].str.extract(r"^v3-([^-]+)")[0]
    return frame


def _line_summary(
    axis: Axes,
    frame: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    key_prefix: str,
) -> None:
    for method in methods:
        subset = frame[frame["method"] == method]
        xs: list[float] = []
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for sample_size in sorted(subset["n_train"].dropna().unique()):
            values = subset.loc[subset["n_train"] == sample_size, metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            lower, upper = _bootstrap_mean(
                values, f"{key_prefix}|{metric}|{method}|{sample_size}"
            )
            xs.append(float(sample_size))
            means.append(float(np.mean(values)))
            lows.append(lower)
            highs.append(upper)
        if not xs:
            continue
        means_array = np.asarray(means)
        axis.errorbar(
            xs,
            means,
            yerr=[means_array - lows, np.asarray(highs) - means_array],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLOURS[method],
            linewidth=1.15 if method in {"ordersmc", "order-mcmc"} else 0.9,
            alpha=1.0 if method in {"ordersmc", "order-mcmc"} else 0.75,
            markersize=4.5,
            capsize=1.8,
            label=METHOD_LABELS[method],
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)


def plot_v3_benchmark_recovery(
    v3_root: Path, output_dir: Path, source_dir: Path
) -> list[Path]:
    _style()
    frame = _v3_plot_frame(v3_root)
    frame = frame[frame["dataset_role"] == "topology-preserving-semi-synthetic-mixed"].copy()
    _write_source_data(frame, source_dir / "v3_benchmark_recovery.csv")
    networks = ("asia", "insurance", "alarm", "hailfinder")
    metrics = (
        ("skeleton_f1", "Skeleton F1"),
        ("directed_cross_type_f1", "Directed cross-type F1"),
        ("directed_auprc", "Directed AUPRC"),
    )
    methods = METHOD_ORDER[:6]
    figure, axes = plt.subplots(3, 4, figsize=(14.2, 8.2), sharey="row")
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column_index, network in enumerate(networks):
            axis = axes[row_index, column_index]
            subset = frame[frame["network"] == network]
            _line_summary(axis, subset, metric, methods, f"v3|{network}")
            if row_index == 0:
                axis.set_title(network.capitalize())
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Training samples")
            _panel_label(axis, chr(ord("a") + row_index * 4 + column_index))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(
        "Topology-preserving heterogeneous benchmark recovery",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    _pilot_stamp(figure, _profile(v3_root))
    figure.tight_layout(rect=(0, 0.02, 1, 0.93))
    return _save_figure(figure, output_dir / "figure_03_v3_benchmark_recovery")


def plot_v3_clg_overlap(
    v3_root: Path, output_dir: Path, source_dir: Path
) -> list[Path]:
    _style()
    frame = _v3_plot_frame(v3_root)
    frame = frame[frame["dataset_role"] == "restricted-native-mixed-benchmark"].copy()
    _write_source_data(frame, source_dir / "v3_clg_overlap.csv")
    methods = METHOD_ORDER
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for label, axis, metric, ylabel in zip(
        "abc",
        axes,
        ("skeleton_f1", "directed_auprc", "directed_brier"),
        ("Skeleton F1", "Directed AUPRC", "Directed Brier score"),
    ):
        _line_summary(axis, frame, metric, methods, "v3-clg")
        if metric == "directed_brier":
            maximum = float(frame[metric].replace([np.inf, -np.inf], np.nan).max())
            axis.set_ylim(-0.005, min(1.0, maximum * 1.18 + 0.01))
        axis.set_xlabel("Training samples")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        _panel_label(axis, label)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle(
        "Restricted conditional-Gaussian overlap with native bnlearn",
        fontsize=12.5,
        fontweight="bold",
        y=1.15,
    )
    _pilot_stamp(figure, _profile(v3_root))
    figure.tight_layout(rect=(0, 0.02, 1, 0.90))
    return _save_figure(figure, output_dir / "figure_04_v3_clg_overlap")


def _v3_stability_inputs(v3_root: Path, clg_root: Path | None = None):
    stability = _read_csv(v3_root, "seed_stability_dagbagm_metrics.csv")
    stability["source_result_root"] = str(v3_root)
    audit = _read_json(v3_root / "validation" / "prepared-audit.json")
    cells = pd.DataFrame(audit["cells"])
    cells["source_result_root"] = str(v3_root)
    if clg_root is not None:
        plan = _read_json(clg_root / "plan.json")
        ids = {row["dataset_id"] for row in plan["datasets"]}
        cell_ids = {row["cell_id"] for row in plan["datasets"]}
        corrected = _read_csv(clg_root, "seed_stability_dagbagm_metrics.csv")
        corrected["source_result_root"] = str(clg_root)
        corrected_audit = _read_json(clg_root / "validation" / "prepared-audit.json")
        corrected_cells = pd.DataFrame(corrected_audit["cells"])
        corrected_cells["source_result_root"] = str(clg_root)
        if audit["minimum_recall"] != corrected_audit["minimum_recall"]:
            raise ValueError("candidate recall thresholds differ between figure cohorts")
        stability = pd.concat([stability[~stability.dataset_id.isin(ids)], corrected], ignore_index=True)
        cells = pd.concat([cells[~cells.cell_id.isin(cell_ids)], corrected_cells], ignore_index=True)
    return stability[stability["method"].isin(METHOD_ORDER[:4])].copy(), cells, audit


def plot_v3_stability_and_candidates(
    v3_root: Path, output_dir: Path, source_dir: Path, clg_root: Path | None = None
) -> list[Path]:
    _style()
    stability, cells, audit = _v3_stability_inputs(v3_root, clg_root)
    _write_source_data(stability, source_dir / "v3_seed_stability.csv")
    _write_source_data(cells, source_dir / "v3_candidate_recall.csv")
    methods = METHOD_ORDER[:4]
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.2))
    specifications = (
        ("mean_pairwise_pip_rmse", "Pairwise PIP RMSE", True),
        ("mean_directed_graph_jaccard", "Directed-DAG Jaccard", False),
        ("mean_skeleton_graph_jaccard", "Skeleton Jaccard", False),
    )
    for label, axis, (metric, ylabel, log_scale) in zip("abc", axes.flat[:3], specifications):
        _distribution_panel(
            axis,
            stability,
            metric,
            methods,
            ylabel,
            log_scale=log_scale,
        )
        axis.set_title(ylabel)
        if "jaccard" in metric:
            axis.set_ylim(max(0.0, float(stability[metric].min()) - 0.04), 1.01)
        _panel_label(axis, label)
    cells = cells.sort_values(["dimension", "cell_id"]).reset_index(drop=True)
    axis = axes[1, 1]
    network_colours = {
        "asia": "#0072B2",
        "clgaussian.test": "#56B4E9",
        "insurance": "#009E73",
        "alarm": "#E69F00",
        "hailfinder": "#D55E00",
    }
    labels: list[str] = []
    used: set[str] = set()
    for index, row in cells.iterrows():
        network = str(row["cell_id"]).split("-topology")[0].split("-restricted")[0]
        labels.append(str(row["cell_id"]).replace("-topology-mixed-", "\n").replace("-restricted-native-mixed-", "\n"))
        axis.scatter(
            index,
            row["mean_complete_family_recall"],
            s=30,
            color=network_colours.get(network, "#555555"),
            marker="o",
            label=network if network not in used else None,
            zorder=3,
        )
        used.add(network)
    axis.axhline(float(audit["minimum_recall"]), color="#A51C30", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(cells)), labels, rotation=55, ha="right", fontsize=6.5)
    axis.set_ylim(min(0.90, float(cells["mean_complete_family_recall"].min()) - 0.01), 1.005)
    axis.set_ylabel("Mean complete-family recall")
    axis.set_title("Candidate-screening audit")
    axis.legend(frameon=False, ncol=2, fontsize=6.5)
    _panel_label(axis, "d")
    figure.suptitle(
        "Sampling stability and candidate-family coverage",
        fontsize=13,
        fontweight="bold",
    )
    _pilot_stamp(figure, _profile(v3_root))
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save_figure(figure, output_dir / "extended_data_03_v3_stability_candidates")


def plot_v3_type_recovery(
    v3_root: Path, output_dir: Path, source_dir: Path
) -> list[Path]:
    frame = _v3_plot_frame(v3_root)
    frame = frame[
        (frame["dataset_role"] == "topology-preserving-semi-synthetic-mixed")
        & frame["method"].isin(["ordersmc", "order-mcmc"])
    ].copy()
    numeric = [column for column in frame if column.startswith("directed_")]
    dataset = frame.groupby(["dataset_id", "method"], as_index=False)[numeric].mean()
    _write_source_data(dataset, source_dir / "v3_type_recovery.csv")
    return _plot_type_heatmaps(
        dataset,
        ("ordersmc", "order-mcmc"),
        "Type-stratified recovery on heterogeneous benchmark topologies",
        output_dir / "extended_data_04_v3_type_recovery",
        profile=_profile(v3_root),
    )


def build_figures(v1_root: Path, v3_root: Path, output_dir: Path,
                  clg_root: Path | None = None) -> dict[str, object]:
    v1_root = v1_root.resolve()
    v3_root = v3_root.resolve()
    output_dir = output_dir.resolve()
    completions = {
        "v1": _require_complete(v1_root),
        "v3": _require_complete(v3_root),
    }
    if clg_root is not None:
        clg_root = clg_root.resolve()
        completions["clg"] = _require_complete(clg_root)
        clg_plan = _read_json(clg_root / "plan.json")
        if not clg_plan["datasets"] or any(
            row.get("data_kind") != "native_mixed_conditional_gaussian"
            for row in clg_plan["datasets"]
        ):
            raise ValueError("CLG replacement must contain only the restricted CLG benchmark")
        if _profile(clg_root) != _profile(v3_root):
            raise ValueError("cannot combine pilot and full figure cohorts")
    source_dir = output_dir / "source_data"
    generated: list[Path] = []
    generated.extend(plot_algorithm_overview(output_dir))
    generated.extend(plot_v1_exact_validation(v1_root, output_dir, source_dir))
    generated.extend(plot_v1_stability(v1_root, output_dir, source_dir))
    generated.extend(plot_v1_type_recovery(v1_root, output_dir, source_dir))
    generated.extend(plot_v3_benchmark_recovery(v3_root, output_dir, source_dir))
    generated.extend(plot_v3_clg_overlap(clg_root or v3_root, output_dir, source_dir))
    generated.extend(plot_v3_stability_and_candidates(v3_root, output_dir, source_dir, clg_root))
    generated.extend(plot_v3_type_recovery(v3_root, output_dir, source_dir))
    plans = {
        "v1": _read_json(v1_root / "plan.json"),
        "v3": _read_json(v3_root / "plan.json"),
    }
    manifest: dict[str, object] = {
        "schema": "ordersmc-paper-figures-v2",
        "v1_result_root": str(v1_root),
        "v1_plan_sha256": str(plans["v1"]["plan_sha256"]),
        "v1_completion_raw_sha256": _file_sha256(
            v1_root / "analysis" / "completion.json"
        ),
        "v1_completed_runs": int(completions["v1"]["completed_runs"]),
        "v1_profile": _profile(v1_root),
        "v3_result_root": str(v3_root),
        "v3_plan_sha256": str(plans["v3"]["plan_sha256"]),
        "v3_completion_raw_sha256": _file_sha256(
            v3_root / "analysis" / "completion.json"
        ),
        "v3_completed_runs": int(completions["v3"]["completed_runs"]),
        "v3_profile": _profile(v3_root),
        "bootstrap_replicates": 10_000,
        "bootstrap_unit": "whole dataset",
        "formats": ["pdf", "svg", "png"],
        "png_dpi": 300,
        "generated_files": [path.name for path in generated],
        "generated_sha256": {
            path.name: _file_sha256(path) for path in generated
        },
        "source_data_files": sorted(path.name for path in source_dir.glob("*.csv")),
        "source_data_sha256": {
            path.name: _file_sha256(path)
            for path in sorted(source_dir.glob("*.csv"))
        },
    }
    if clg_root is not None:
        manifest["corrected_clg"] = {
            "result_root": str(clg_root), "plan_sha256": clg_plan["plan_sha256"],
            "completed_runs": int(completions["clg"]["completed_runs"]),
            "completion_raw_sha256": _file_sha256(clg_root / "analysis/completion.json"),
            "replacement_policy": "Replace CLG overlap and CLG stability/candidate rows; keep unchanged native-score topology-mixed V3 and V1 cohorts. Do not pool old and corrected CLG runs.",
        }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", required=True, type=Path)
    parser.add_argument("--v3-root", required=True, type=Path)
    parser.add_argument("--v3-clg-root", type=Path,
                        help="Completed corrected CLG cohort replacing only the old CLG arm")
    parser.add_argument("--output-dir", default=Path("figures/generated"), type=Path)
    arguments = parser.parse_args()
    manifest = build_figures(arguments.v1_root, arguments.v3_root, arguments.output_dir,
                             arguments.v3_clg_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
