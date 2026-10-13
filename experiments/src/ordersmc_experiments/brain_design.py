"""Deterministic design manifests for the validation-to-brain study.

This module enumerates scientific cells and expected row counts. Execution is
deliberately enabled only after the matching generators/data adapters exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DESIGN = Path(__file__).resolve().parents[2] / "configs" / "brain_study_v1.json"
PHASES = ("g1", "v1", "v2", "v3")
PROFILES = ("smoke", "pilot", "full")


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_design(path: Path = DEFAULT_DESIGN) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "ordersmc-brain-study-design-v1":
        raise ValueError("unsupported brain-study design")
    if value.get("status") != "design-review":
        raise ValueError("brain-study design must remain in review until adapters pass")
    if value["v4_living_brain"]["status"] != "blocked-until-brain-data-profile-complete":
        raise ValueError("V4 cannot be enabled before its metadata profile is complete")
    return value


def _profile_settings(design: Mapping[str, Any], phase: str, profile: str) -> tuple[int, int]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    key = profile
    if profile == "full":
        key = "full_v1" if phase == "v1" else "full_v2_v3"
    row = design["profiles"][key]
    return int(row["datasets_per_cell"]), int(row["method_seeds"])


def _v1_cells(design: Mapping[str, Any]) -> list[dict[str, object]]:
    spec = design["v1_exact"]
    cells: list[dict[str, object]] = []
    baseline_ratio = int(spec["n_over_d"][0])
    for dimension in spec["dimensions"]:
        for regime in spec["variable_type_regimes"]:
            cells.append(
                {
                    "cell_id": f"d{int(dimension):03d}-n{baseline_ratio:02d}d-{regime}",
                    "dimension": int(dimension),
                    "n_over_d": baseline_ratio,
                    "n_train": baseline_ratio * int(dimension),
                    "variable_type_regime": regime,
                    "role": "primary",
                    "exact_reference": True,
                }
            )
    supplement = spec["sample_size_supplement"]
    dimension = int(supplement["dimension"])
    regime = str(supplement["variable_type_regime"])
    for ratio in supplement["n_over_d"]:
        ratio = int(ratio)
        if ratio == baseline_ratio:
            continue
        cells.append(
            {
                "cell_id": f"d{dimension:03d}-n{ratio:02d}d-{regime}",
                "dimension": dimension,
                "n_over_d": ratio,
                "n_train": ratio * dimension,
                "variable_type_regime": regime,
                "role": "sample-size-supplement",
                "exact_reference": True,
            }
        )
    return sorted(cells, key=lambda row: str(row["cell_id"]))


def _g1_cells(design: Mapping[str, Any]) -> list[dict[str, object]]:
    spec = design["g1_comparator_capability"]
    cells = [
        {
            "cell_id": f"mixed-{motif}",
            "motif": motif,
            "role": "required-mixed-capability",
        }
        for motif in spec["required_mixed_motifs"]
    ]
    cells.extend(
        {
            "cell_id": f"homogeneous-{kind}-installation-control",
            "motif": kind,
            "role": "installation-control-not-scientific",
        }
        for kind in spec["homogeneous_installation_controls"]
    )
    return sorted(cells, key=lambda row: str(row["cell_id"]))


def _v2_cells(design: Mapping[str, Any]) -> list[dict[str, object]]:
    spec = design["v2_brain_matched_synthetic"]
    grid = spec["reference_grid"]
    baseline_dimension = 50
    baseline_ratio = int(grid["n_over_d_baseline"])
    baseline_topology = "random_order"
    baseline_signal = "moderate"
    baseline_noise = "gaussian"
    structures: set[tuple[int, int, str, str, str]] = set()
    structures.update(
        (int(d), baseline_ratio, baseline_topology, baseline_signal, baseline_noise)
        for d in grid["dimension_baseline"]
    )
    structures.update(
        (baseline_dimension, int(ratio), baseline_topology, baseline_signal, baseline_noise)
        for ratio in grid["sample_size_at_d50"]
    )
    structures.update(
        (baseline_dimension, baseline_ratio, str(topology), baseline_signal, baseline_noise)
        for topology in grid["topology_at_d50"]
    )
    structures.update(
        (baseline_dimension, baseline_ratio, baseline_topology, str(signal), baseline_noise)
        for signal in grid["signal_at_d50"]
    )
    structures.update(
        (baseline_dimension, baseline_ratio, baseline_topology, baseline_signal, str(noise))
        for noise in grid["continuous_noise_at_d50"]
    )
    surrogate = str(spec["preprofile_pilot_surrogate"])
    cells = [
        {
            "cell_id": f"d{dimension:03d}-n{ratio:02d}d-{topology}-{signal}-{noise}-{surrogate}",
            "dimension": dimension,
            "n_over_d": ratio,
            "n_train": dimension * ratio,
            "topology": topology,
            "signal": signal,
            "continuous_noise": noise,
            "variable_type_regime": surrogate,
            "role": "structural-grid",
            "exact_reference": False,
        }
        for dimension, ratio, topology, signal, noise in sorted(structures)
    ]
    if bool(grid["mixed_null_graph_at_d50"]):
        cells.append(
            {
                "cell_id": f"d{baseline_dimension:03d}-n{baseline_ratio:02d}d-empty-none-{baseline_noise}-{surrogate}",
                "dimension": baseline_dimension,
                "n_over_d": baseline_ratio,
                "n_train": baseline_dimension * baseline_ratio,
                "topology": "empty",
                "signal": "none",
                "continuous_noise": baseline_noise,
                "variable_type_regime": surrogate,
                "role": "mixed-null-negative-control",
                "exact_reference": False,
            }
        )
    baseline = (
        baseline_dimension,
        baseline_ratio,
        baseline_topology,
        baseline_signal,
        baseline_noise,
    )
    for regime in spec["type_regime_sensitivity_at_baseline"]:
        if regime == surrogate:
            continue
        dimension, ratio, topology, signal, noise = baseline
        cells.append(
            {
                "cell_id": f"d{dimension:03d}-n{ratio:02d}d-{topology}-{signal}-{noise}-{regime}",
                "dimension": dimension,
                "n_over_d": ratio,
                "n_train": dimension * ratio,
                "topology": topology,
                "signal": signal,
                "continuous_noise": noise,
                "variable_type_regime": regime,
                "role": "type-regime-sensitivity",
                "exact_reference": False,
            }
        )
    return sorted(cells, key=lambda row: str(row["cell_id"]))


def _v3_cells(design: Mapping[str, Any]) -> list[dict[str, object]]:
    spec = design["v3_known_networks"]
    cells: list[dict[str, object]] = []
    for network in spec["restricted_native_mixed_networks"]:
        network_id = str(network["id"])
        for sample_size in network["sample_sizes"]:
            cells.append(
                {
                    "cell_id": f"{network_id}-restricted-native-mixed-n{int(sample_size):05d}",
                    "network": network_id,
                    "dimension": int(network["nodes"]),
                    "n_train": int(sample_size),
                    "data_kind": network["kind"],
                    "role": "restricted-native-mixed-benchmark",
                    "exact_reference": True,
                }
            )
    for network in spec["topology_preserving_mixed_networks"]:
        network_id = str(network["id"])
        for sample_size in network["sample_sizes"]:
            cells.append(
                {
                    "cell_id": f"{network_id}-topology-mixed-n{int(sample_size):05d}",
                    "network": network_id,
                    "dimension": int(network["nodes"]),
                    "n_train": int(sample_size),
                    "data_kind": "topology_preserving_balanced_mixed",
                    "role": "topology-preserving-semi-synthetic-mixed",
                    "exact_reference": network_id == "asia",
                }
            )
    return sorted(cells, key=lambda row: str(row["cell_id"]))


def build_design_manifest(
    phase: str,
    profile: str,
    *,
    design_path: Path = DEFAULT_DESIGN,
) -> dict[str, object]:
    phase = phase.lower()
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    design = load_design(design_path)
    builders = {"g1": _g1_cells, "v1": _v1_cells, "v2": _v2_cells, "v3": _v3_cells}
    cells = builders[phase](design)
    if phase == "g1":
        methods = len(design["g1_comparator_capability"]["methods"])
        body: dict[str, object] = {
            "schema": "ordersmc-brain-study-plan-v1",
            "status": "design-only-not-executable",
            "phase": phase,
            "profile": profile,
            "master_seed": int(design["pilot_master_seed"]),
            "design_sha256": _canonical_sha(design),
            "cells": cells,
            "cell_count": len(cells),
            "method_count": methods,
            "run_count": len(cells) * methods,
            "execution_blocker": "external comparator adapters are not yet accepted",
        }
        body["plan_sha256"] = _canonical_sha(body)
        return body
    datasets_per_cell, method_seeds = _profile_settings(design, phase, profile)
    stochastic_methods = 2 if phase == "v1" else 4
    dataset_count = len(cells) * datasets_per_cell
    stochastic_rows = dataset_count * stochastic_methods * method_seeds
    exact_rows = datasets_per_cell * sum(bool(cell["exact_reference"]) for cell in cells)
    comparator_rows = 0
    if phase == "v3":
        shared_score_methods = 2
        native_methods = 2
        native_cells = sum(
            cell["role"] == "restricted-native-mixed-benchmark" for cell in cells
        )
        comparator_rows = (
            dataset_count * shared_score_methods
            + native_cells * datasets_per_cell * native_methods
        )
    master_seed = int(
        design["confirmatory_master_seed"] if profile == "full" else design["pilot_master_seed"]
    )
    body: dict[str, object] = {
        "schema": "ordersmc-brain-study-plan-v1",
        "status": "design-only-not-executable",
        "phase": phase,
        "profile": profile,
        "master_seed": master_seed,
        "design_sha256": _canonical_sha(design),
        "datasets_per_cell": datasets_per_cell,
        "method_seeds": method_seeds,
        "cells": cells,
        "cell_count": len(cells),
        "dataset_count": dataset_count,
        "stochastic_run_count": stochastic_rows,
        "comparator_run_count": comparator_rows,
        "exact_run_count": exact_rows,
        "run_count": stochastic_rows + comparator_rows + exact_rows,
        "execution_blocker": "matching data generators and benchmark adapters are not yet accepted",
    }
    body["plan_sha256"] = _canonical_sha(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ordersmc_experiments.brain_design")
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = build_design_manifest(args.phase, args.profile, design_path=args.design)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"existing design manifest differs: {args.output}")
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
