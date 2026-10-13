"""Deterministic cohort plans for the lean Version 3 experimental protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .seeds import protocol_seed_record
from .brain_design import _v1_cells, _v2_cells, _v3_cells, load_design
from .metadata_profiles import PROFILE_PRIORITY, load_metadata_profile


PROTOCOL_VERSION = "3.0"
EXPERIMENT_IDS = (
    "e1",
    "e2",
    "e3",
    "e4",
    "v1",
    "v2",
    "v3",
    "v3-clg",
    "s2",
)
SECTION2_DESIGN = Path(__file__).resolve().parents[2] / "configs" / "section2_metadata_v3.json"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    experiment: str
    dimension: int
    n_train: int
    replicate: int
    graph_family: str
    expected_indegree: float
    max_indegree: int
    signal: str
    dataset_seed: Mapping[str, Any]


@dataclass(frozen=True)
class BrainV1DatasetSpec:
    """Executable dataset row for the mixed-composition V1 protocol."""

    dataset_id: str
    experiment: str
    dimension: int
    n_train: int
    replicate: int
    graph_family: str
    expected_indegree: float
    max_indegree: int
    signal: str
    dataset_seed: Mapping[str, Any]
    cell_id: str
    variable_type_regime: str
    n_over_d: int
    role: str


@dataclass(frozen=True)
class BrainV2DatasetSpec:
    """Executable dataset row for the mixed synthetic V2 pilot grid."""

    dataset_id: str
    experiment: str
    dimension: int
    n_train: int
    replicate: int
    graph_family: str
    expected_indegree: float
    max_indegree: int
    signal: str
    dataset_seed: Mapping[str, Any]
    cell_id: str
    variable_type_regime: str
    continuous_noise: str
    n_over_d: int
    role: str


@dataclass(frozen=True)
class BrainV3DatasetSpec:
    """Executable row for a native or topology-preserving known benchmark."""

    dataset_id: str
    experiment: str
    dimension: int
    n_train: int
    replicate: int
    graph_family: str
    expected_indegree: float
    max_indegree: int
    signal: str
    dataset_seed: Mapping[str, Any]
    cell_id: str
    variable_type_regime: str
    n_over_d: int
    role: str
    network: str
    data_kind: str
    fixed_topology_id: str | None
    benchmark_n_train: int


@dataclass(frozen=True)
class MetadataDatasetSpec:
    """One known-truth dataset calibrated to a privacy-preserving profile."""

    dataset_id: str
    experiment: str
    dimension: int
    n_train: int
    replicate: int
    graph_family: str
    expected_indegree: float
    max_indegree: int
    signal: str
    dataset_seed: Mapping[str, Any]
    cell_id: str
    role: str
    data_kind: str
    metadata_profile: str
    data_variant: str
    causal_regime: str
    negative_control: str
    sample_multiplier: float
    pairing_id: str
    variable_type_regime: str
    n_over_d: float


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset_id: str
    method: str
    target: str
    role: str
    seed_index: int | None
    method_seed: Mapping[str, Any] | None
    wall_seconds: int | None = None
    mpi_ranks: int = 1
    timing_repeat: int | None = None


@dataclass(frozen=True)
class ExperimentPlan:
    protocol_version: str
    experiment: str
    master_seed: int
    datasets: tuple[DatasetSpec, ...]
    runs: tuple[RunSpec, ...]
    settings: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["dataset_count"] = len(self.datasets)
        body["run_count"] = len(self.runs)
        body["plan_sha256"] = _canonical_sha256(body)
        return body

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _seed(master_seed: int, experiment: str, cell: str, replicate: int, component: str) -> dict[str, Any]:
    return asdict(
        protocol_seed_record(
            master_seed,
            experiment=experiment,
            cell=cell,
            replicate=replicate,
            component=component,
        )
    )


def _datasets(
    *,
    master_seed: int,
    experiment: str,
    dimensions: tuple[int, ...],
    replicates: int,
    expected_indegree: float,
    max_indegree: int,
    difficult: bool = False,
) -> tuple[DatasetSpec, ...]:
    rows: list[DatasetSpec] = []
    for d in dimensions:
        for replicate in range(replicates):
            family = (
                "order_blocks" if replicate < replicates // 2 else "weak_bridges"
            ) if difficult else "random_order"
            cell = f"d{d:03d}-{family}"
            dataset_id = f"{experiment}-{cell}-rep{replicate:03d}"
            rows.append(
                DatasetSpec(
                    dataset_id=dataset_id,
                    experiment=experiment,
                    dimension=d,
                    n_train=20 * d,
                    replicate=replicate,
                    graph_family=family,
                    expected_indegree=expected_indegree,
                    max_indegree=max_indegree,
                    signal="mixed-difficulty" if difficult else "moderate",
                    dataset_seed=_seed(master_seed, experiment, cell, replicate, "dataset"),
                )
            )
    return tuple(rows)


def _method_run(
    *,
    master_seed: int,
    dataset: DatasetSpec,
    method: str,
    target: str,
    seed_index: int,
    role: str = "evaluation",
    wall_seconds: int | None = None,
) -> RunSpec:
    component = f"{role}-{method}-seed{seed_index:02d}"
    run_id = f"{dataset.dataset_id}-{method}-{role}-s{seed_index:02d}"
    cell = str(
        getattr(dataset, "cell_id", f"d{dataset.dimension:03d}-{dataset.graph_family}")
    )
    return RunSpec(
        run_id=run_id,
        dataset_id=dataset.dataset_id,
        method=method,
        target=target,
        role=role,
        seed_index=seed_index,
        method_seed=_seed(
            master_seed,
            dataset.experiment,
            cell,
            dataset.replicate,
            component,
        ),
        wall_seconds=wall_seconds,
    )


def _common_settings() -> dict[str, Any]:
    return {
        "candidate_policy": {
            "e1": "complete-parent-sets-through-indegree-2",
            "main": {
                "max_parents": 3,
                "screen_k": 20,
                "parent_pool_max": 30,
                "keep_per_child": None,
                "pair_rescue": "frozen-implementation-rule",
            },
            "wider_sensitivity": {
                "datasets_per_dimension": 5,
                "screen_k": 30,
                "parent_pool_max": 40,
            },
        },
        "method_seeds_per_dataset": 4,
        "statistical_unit": "graph-dataset-pair",
        "seed_aggregation": "average-within-dataset-before-paired-bootstrap",
        "failure_policy": "retain-all-planned-rows",
    }


def _build_e1(master_seed: int) -> ExperimentPlan:
    datasets = _datasets(
        master_seed=master_seed,
        experiment="e1",
        dimensions=(8, 12),
        replicates=10,
        expected_indegree=1.5,
        max_indegree=2,
    )
    runs: list[RunSpec] = []
    for dataset in datasets:
        runs.append(
            RunSpec(
                run_id=f"{dataset.dataset_id}-exact-order",
                dataset_id=dataset.dataset_id,
                method="exact-order-dp",
                target="order-modular",
                role="reference",
                seed_index=None,
                method_seed=None,
            )
        )
        for method in ("ordersmc", "order-mcmc"):
            for seed_index in range(4):
                runs.append(
                    _method_run(
                        master_seed=master_seed,
                        dataset=dataset,
                        method=method,
                        target="order-modular",
                        seed_index=seed_index,
                        wall_seconds=300,
                    )
                )
    settings = _common_settings()
    settings.update({"primary_metric": "directed-pip-rmse-to-exact-order"})
    return ExperimentPlan(PROTOCOL_VERSION, "e1", master_seed, datasets, tuple(runs), settings)


def _build_v1(master_seed: int, profile: str) -> ExperimentPlan:
    """Build executable exact-validation cells for all mixed compositions."""

    design = load_design()
    if profile not in {"smoke", "pilot", "full"}:
        raise ValueError("V1 profile must be smoke, pilot, or full")
    profile_key = "full_v1" if profile == "full" else profile
    profile_settings = design["profiles"][profile_key]
    replicates = int(profile_settings["datasets_per_cell"])
    method_seeds = int(profile_settings["method_seeds"])
    specification = design["v1_exact"]
    datasets: list[BrainV1DatasetSpec] = []
    for cell in _v1_cells(design):
        cell_id = str(cell["cell_id"])
        for replicate in range(replicates):
            datasets.append(
                BrainV1DatasetSpec(
                    dataset_id=f"v1-{cell_id}-rep{replicate:03d}",
                    experiment="v1",
                    dimension=int(cell["dimension"]),
                    n_train=int(cell["n_train"]),
                    replicate=replicate,
                    graph_family="random_order",
                    expected_indegree=float(specification["expected_indegree"]),
                    max_indegree=int(specification["max_indegree"]),
                    signal=str(specification["signal"]),
                    dataset_seed=_seed(master_seed, "v1", cell_id, replicate, "dataset"),
                    cell_id=cell_id,
                    variable_type_regime=str(cell["variable_type_regime"]),
                    n_over_d=int(cell["n_over_d"]),
                    role=str(cell["role"]),
                )
            )
    runs: list[RunSpec] = []
    for dataset in datasets:
        runs.append(
            RunSpec(
                run_id=f"{dataset.dataset_id}-exact-order",
                dataset_id=dataset.dataset_id,
                method="exact-order-dp",
                target="order-modular",
                role="reference",
                seed_index=None,
                method_seed=None,
            )
        )
        for method in ("ordersmc", "order-mcmc"):
            for seed_index in range(method_seeds):
                runs.append(
                    _method_run(
                        master_seed=master_seed,
                        dataset=dataset,
                        method=method,
                        target="order-modular",
                        seed_index=seed_index,
                        wall_seconds=300,
                    )
                )
    settings = _common_settings()
    settings.update(
        {
            "study_version": str(design["study_version"]),
            "profile": profile,
            "method_seeds_per_dataset": method_seeds,
            "mixed_composition_cells": True,
            "primary_metric": specification["primary_metric"],
            "pass_gate": dict(specification["pass_gate"]),
        }
    )
    return ExperimentPlan(
        f"brain-{design['study_version']}",
        "v1",
        master_seed,
        tuple(datasets),
        tuple(runs),
        settings,
    )


def _build_v2(master_seed: int, profile: str) -> ExperimentPlan:
    """Build the executable pre-profile, all-mixed V2 smoke or pilot grid."""

    design = load_design()
    if profile not in {"smoke", "pilot"}:
        raise ValueError(
            "V2 full execution is blocked until brain_data_profile.json is frozen"
        )
    profile_settings = design["profiles"][profile]
    replicates = int(profile_settings["datasets_per_cell"])
    method_seeds = int(profile_settings["method_seeds"])
    specification = design["v2_brain_matched_synthetic"]
    datasets: list[BrainV2DatasetSpec] = []
    for cell in _v2_cells(design):
        cell_id = str(cell["cell_id"])
        for replicate in range(replicates):
            datasets.append(
                BrainV2DatasetSpec(
                    dataset_id=f"v2-{cell_id}-rep{replicate:03d}",
                    experiment="v2",
                    dimension=int(cell["dimension"]),
                    n_train=int(cell["n_train"]),
                    replicate=replicate,
                    graph_family=str(cell["topology"]),
                    expected_indegree=2.0,
                    max_indegree=3,
                    signal=str(cell["signal"]),
                    dataset_seed=_seed(master_seed, "v2", cell_id, replicate, "dataset"),
                    cell_id=cell_id,
                    variable_type_regime=str(cell["variable_type_regime"]),
                    continuous_noise=str(cell["continuous_noise"]),
                    n_over_d=int(cell["n_over_d"]),
                    role=str(cell["role"]),
                )
            )
    methods = (
        ("ordersmc", "order-modular"),
        ("order-mcmc", "order-modular"),
        ("partition-mcmc", "structure-modular"),
        ("rev-mcmc", "structure-modular"),
    )
    runs = tuple(
        _method_run(
            master_seed=master_seed,
            dataset=dataset,
            method=method,
            target=target,
            seed_index=seed_index,
            wall_seconds=300,
        )
        for dataset in datasets
        for method, target in methods
        for seed_index in range(method_seeds)
    )
    settings = _common_settings()
    settings.update(
        {
            "study_version": str(design["study_version"]),
            "profile": profile,
            "method_seeds_per_dataset": method_seeds,
            "brain_profile_status": specification["status"],
            "variable_type_regime": specification["preprofile_pilot_surrogate"],
            "cell_count": len(_v2_cells(design)),
            "comparison_panels": {
                "same_target": ["ordersmc", "order-mcmc"],
                "end_to_end": ["ordersmc", "partition-mcmc", "rev-mcmc"],
            },
            "budget": "equal-300-second-envelope-with-completed-beta-smc",
        }
    )
    return ExperimentPlan(
        f"brain-{design['study_version']}",
        "v2",
        master_seed,
        tuple(datasets),
        runs,
        settings,
    )


def _build_v3(master_seed: int, profile: str) -> ExperimentPlan:
    """Build recognized mixed-network benchmarks without using living-brain metadata."""

    design = load_design()
    if profile not in {"smoke", "pilot", "full"}:
        raise ValueError("V3 profile must be smoke, pilot, or full")
    profile_key = "full_v2_v3" if profile == "full" else profile
    profile_settings = design["profiles"][profile_key]
    replicates = int(profile_settings["datasets_per_cell"])
    method_seeds = int(profile_settings["method_seeds"])
    maximum_indegree = {
        "clgaussian.test": 4,
        "asia": 2,
        "insurance": 3,
        "alarm": 4,
        "hailfinder": 4,
    }
    edge_counts = {
        "clgaussian.test": 10,
        "asia": 8,
        "insurance": 52,
        "alarm": 46,
        "hailfinder": 66,
    }
    datasets: list[BrainV3DatasetSpec] = []
    for cell in _v3_cells(design):
        cell_id = str(cell["cell_id"])
        network = str(cell["network"])
        native = str(cell["role"]) == "restricted-native-mixed-benchmark"
        benchmark_n_train = int(cell["n_train"])
        execution_n_train = (
            min(benchmark_n_train, max(100, 5 * int(cell["dimension"])))
            if profile == "smoke"
            else benchmark_n_train
        )
        for replicate in range(replicates):
            datasets.append(
                BrainV3DatasetSpec(
                    dataset_id=f"v3-{cell_id}-rep{replicate:03d}",
                    experiment="v3",
                    dimension=int(cell["dimension"]),
                    n_train=execution_n_train,
                    replicate=replicate,
                    graph_family="random_order",
                    expected_indegree=edge_counts[network] / int(cell["dimension"]),
                    max_indegree=maximum_indegree[network],
                    signal="moderate",
                    dataset_seed=_seed(master_seed, "v3", cell_id, replicate, "dataset"),
                    cell_id=cell_id,
                    variable_type_regime="native_clg" if native else "balanced_mixed",
                    n_over_d=max(1, int(cell["n_train"]) // int(cell["dimension"])),
                    role=str(cell["role"]),
                    network=network,
                    data_kind=str(cell["data_kind"]),
                    fixed_topology_id=None if native else network,
                    benchmark_n_train=benchmark_n_train,
                )
            )
    runs: list[RunSpec] = []
    smc_ranks = 8 if profile == "smoke" else 64
    for dataset in datasets:
        if dataset.dimension <= 8:
            runs.append(
                RunSpec(
                    run_id=f"{dataset.dataset_id}-exact-order",
                    dataset_id=dataset.dataset_id,
                    method="exact-order-dp",
                    target="order-modular",
                    role="reference",
                    seed_index=None,
                    method_seed=None,
                )
            )
        for method, target in (
            ("ordersmc", "order-modular"),
            ("order-mcmc", "order-modular"),
            ("partition-mcmc", "structure-modular"),
            ("rev-mcmc", "structure-modular"),
        ):
            for seed_index in range(method_seeds):
                run = _method_run(
                    master_seed=master_seed,
                    dataset=dataset,
                    method=method,
                    target=target,
                    seed_index=seed_index,
                    wall_seconds=300,
                )
                if method == "ordersmc":
                    run = RunSpec(**{**asdict(run), "mpi_ranks": smc_ranks})
                runs.append(run)
        for algorithm in ("hc", "tabu"):
            method = f"bnlearn-{algorithm}-shared-score"
            runs.append(
                _method_run(
                    master_seed=master_seed,
                    dataset=dataset,
                    method=method,
                    target="shared-mixed-score-search",
                    seed_index=0,
                    role="external-baseline",
                    wall_seconds=300,
                )
            )
            if dataset.data_kind == "native_mixed_conditional_gaussian":
                method = f"bnlearn-{algorithm}-bic-cg"
                runs.append(
                    _method_run(
                        master_seed=master_seed,
                        dataset=dataset,
                        method=method,
                        target="native-bic-cg-search",
                        seed_index=0,
                        role="external-baseline",
                        wall_seconds=300,
                    )
                )
    settings = _common_settings()
    settings.update(
        {
            "study_version": str(design["study_version"]),
            "profile": profile,
            "method_seeds_per_dataset": method_seeds,
            "living_brain_metadata_used": False,
            "ordersmc_mpi_ranks": smc_ranks,
            "native_baseline": ["bnlearn-hc-bic-cg", "bnlearn-tabu-bic-cg"],
            "shared_score_search_control": [
                "bnlearn-hc-shared-score",
                "bnlearn-tabu-shared-score",
            ],
            "unsupported_policy": "native-bic-cg-only-on-clg-compatible-network",
        }
    )
    return ExperimentPlan(
        f"brain-{design['study_version']}",
        "v3",
        master_seed,
        tuple(datasets),
        tuple(runs),
        settings,
    )


def _build_v3_clg(master_seed: int, profile: str) -> ExperimentPlan:
    """Held-out corrective validation on the native CLG V3 benchmark only."""

    source = _build_v3(master_seed, profile)
    datasets = tuple(
        replace(dataset, n_train=dataset.benchmark_n_train)
        if profile == "smoke"
        else dataset
        for dataset in source.datasets
        if dataset.data_kind == "native_mixed_conditional_gaussian"
    )
    dataset_ids = {dataset.dataset_id for dataset in datasets}
    runs = tuple(run for run in source.runs if run.dataset_id in dataset_ids)
    settings = dict(source.settings)
    settings.update(
        {
            "corrective_validation": True,
            "corrective_scope": "native-restricted-conditional-gaussian-benchmark",
            "score_selection_basis": "statistical-model-class-not-dataset-name",
            "legacy_comparator": "published V3 native-score results",
            "living_brain_metadata_used": False,
        }
    )
    return ExperimentPlan(
        source.protocol_version,
        "v3-clg",
        master_seed,
        datasets,
        runs,
        settings,
    )


def _build_e2(master_seed: int) -> ExperimentPlan:
    datasets = _datasets(
        master_seed=master_seed,
        experiment="e2",
        dimensions=(20, 50, 100),
        replicates=20,
        expected_indegree=2.0,
        max_indegree=3,
    )
    methods = (
        ("ordersmc", "order-modular"),
        ("order-mcmc", "order-modular"),
        ("partition-mcmc", "structure-modular"),
        ("rev-mcmc", "structure-modular"),
    )
    runs = tuple(
        _method_run(
            master_seed=master_seed,
            dataset=dataset,
            method=method,
            target=target,
            seed_index=seed_index,
            wall_seconds=300,
        )
        for dataset in datasets
        for method, target in methods
        for seed_index in range(4)
    )
    settings = _common_settings()
    settings.update(
        {
            "comparison_panels": {
                "same_target": ["ordersmc", "order-mcmc"],
                "end_to_end": ["ordersmc", "partition-mcmc", "rev-mcmc"],
            },
            "co_primary_metrics": ["directed-auprc", "directed-brier"],
            "budget": "equal-wall-clock-frozen-by-pilot",
        }
    )
    return ExperimentPlan(PROTOCOL_VERSION, "e2", master_seed, datasets, runs, settings)


def _build_e3(master_seed: int) -> ExperimentPlan:
    datasets = _datasets(
        master_seed=master_seed,
        experiment="e3",
        dimensions=(50, 100),
        replicates=10,
        expected_indegree=2.0,
        max_indegree=3,
        difficult=True,
    )
    run_index: dict[str, RunSpec] = {}
    for dataset in datasets:
        for method, target in (
            ("ordersmc", "order-modular"),
            ("order-mcmc", "order-modular"),
            ("partition-mcmc", "structure-modular"),
            ("rev-mcmc", "structure-modular"),
        ):
            for wall_seconds in (30, 60, 180, 300):
                for seed_index in range(4):
                    base = _method_run(
                        master_seed=master_seed,
                        dataset=dataset,
                        method=method,
                        target=target,
                        seed_index=seed_index,
                        wall_seconds=wall_seconds,
                    )
                    run = RunSpec(
                        **{
                            **asdict(base),
                            "run_id": f"{base.run_id}-b{wall_seconds:03d}",
                        }
                    )
                    run_index[run.run_id] = run

        for target in ("order-modular", "structure-modular"):
            for seed_index in range(4):
                run = _method_run(
                    master_seed=master_seed,
                    dataset=dataset,
                    method=f"{target}-reference",
                    target=target,
                    seed_index=seed_index,
                    role="reference",
                )
                run_index[run.run_id] = run

        if dataset.replicate < 5:
            for method in ("ordersmc", "order-mcmc"):
                for seed_index in range(20):
                    run = _method_run(
                        master_seed=master_seed,
                        dataset=dataset,
                        method=method,
                        target="order-modular",
                        seed_index=seed_index,
                        role="reproducibility",
                        wall_seconds=300,
                    )
                    run_index[run.run_id] = run

    settings = _common_settings()
    settings.update(
        {
            "checkpoints_seconds": [30, 60, 180, 300],
            "primary_comparison": ["ordersmc", "order-mcmc"],
            "primary_metric": "time-to-independent-order-reference",
            "reference_seeds_per_target": 4,
            "reproducibility_subset": {
                "datasets_per_dimension": 5,
                "methods": ["ordersmc", "order-mcmc"],
                "seeds": 20,
            },
        }
    )
    return ExperimentPlan(
        PROTOCOL_VERSION,
        "e3",
        master_seed,
        datasets,
        tuple(run_index[key] for key in sorted(run_index)),
        settings,
    )


def _build_e4(master_seed: int) -> ExperimentPlan:
    datasets = _datasets(
        master_seed=master_seed,
        experiment="e4",
        dimensions=(100, 200),
        replicates=5,
        expected_indegree=2.0,
        max_indegree=3,
    )
    ranks = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    runs: list[RunSpec] = []
    for dataset in datasets:
        cell = f"d{dataset.dimension:03d}-{dataset.graph_family}"
        for repeat in range(3):
            repeat_seed = _seed(
                master_seed,
                "e4",
                cell,
                dataset.replicate,
                f"timing-repeat{repeat:02d}",
            )
            for rank in ranks:
                runs.append(
                    RunSpec(
                        run_id=f"{dataset.dataset_id}-ordersmc-r{rank:03d}-t{repeat:02d}",
                        dataset_id=dataset.dataset_id,
                        method="ordersmc",
                        target="order-modular",
                        role="strong-scaling",
                        seed_index=repeat,
                        method_seed=repeat_seed,
                        mpi_ranks=rank,
                        timing_repeat=repeat,
                    )
                )
    settings = _common_settings()
    settings.update(
        {
            "particles": 512,
            "mpi_ranks": list(ranks),
            "timing_repeats": 3,
            "primary_useful_scaling_max_ranks": 32,
            "saturation_ranks": [64, 128, 256],
        }
    )
    return ExperimentPlan(PROTOCOL_VERSION, "e4", master_seed, datasets, tuple(runs), settings)


def _load_section2_design() -> dict[str, Any]:
    value = json.loads(SECTION2_DESIGN.read_text(encoding="utf-8"))
    if value.get("schema") != "ordersmc-section2-metadata-design-v1":
        raise ValueError("unsupported Section 2 metadata design")
    if tuple(value.get("profile_priority", ())) != PROFILE_PRIORITY:
        raise ValueError("Section 2 profile priority has changed")
    generator = value.get("generator", {})
    if (
        generator.get("uses_participant_values") is not False
        or generator.get("uses_empirical_edges_or_directions") is not False
        or generator.get("true_graph_source") != "independent_seeded_draw"
    ):
        raise ValueError("Section 2 independent-DAG guardrail is not frozen")
    return value


def _build_s2(master_seed: int, profile: str) -> ExperimentPlan:
    """Build M0--M5 without touching participant-level observations."""

    design = _load_section2_design()
    if profile not in {"smoke", "pilot", "full"}:
        raise ValueError("Section 2 profile must be smoke, pilot, or full")
    cohort = dict(design["cohorts"][profile])
    profile_ids = tuple(str(value) for value in cohort["profiles"])
    if tuple(item for item in PROFILE_PRIORITY if item in profile_ids) != profile_ids:
        raise ValueError("cohort profiles do not respect the frozen priority")
    enabled_panels = set(map(str, cohort["panels"]))
    primary_replicates = int(cohort["primary_datasets"])
    secondary_replicates = int(cohort["secondary_datasets"])
    generator = dict(design["generator"])
    panel_scope = dict(design["panel_scope"])
    datasets: list[MetadataDatasetSpec] = []

    def in_scope(panel: str, profile_id: str) -> bool:
        scope = panel_scope[panel]
        return scope == "all" or profile_id in scope

    def add_cell(
        *,
        profile_id: str,
        panel: str,
        data_variant: str,
        causal_regime: str,
        negative_control: str,
        sample_multiplier: float,
        pairing_suffix: str,
        replicates: int,
    ) -> None:
        metadata = load_metadata_profile(profile_id)
        n_train = int(round(metadata.participants * sample_multiplier))
        multiplier_token = f"{int(round(100 * sample_multiplier)):03d}"
        control_token = "" if negative_control == "none" else f"-{negative_control}"
        cell_id = (
            f"{panel}-{profile_id}-n{multiplier_token}-{data_variant}-"
            f"{causal_regime}{control_token}"
        )
        pairing_id = f"{profile_id}-{pairing_suffix}"
        for replicate in range(replicates):
            datasets.append(
                MetadataDatasetSpec(
                    dataset_id=f"s2-{cell_id}-rep{replicate:03d}",
                    experiment="s2",
                    dimension=metadata.total_variables,
                    n_train=n_train,
                    replicate=replicate,
                    graph_family=str(generator["graph_family"]),
                    expected_indegree=float(generator["expected_indegree"]),
                    max_indegree=int(generator["maximum_indegree"]),
                    signal="metadata-calibrated",
                    dataset_seed=_seed(master_seed, "s2", pairing_id, replicate, "dataset"),
                    cell_id=cell_id,
                    role=panel,
                    data_kind="metadata_calibrated",
                    metadata_profile=profile_id,
                    data_variant=data_variant,
                    causal_regime=causal_regime,
                    negative_control=negative_control,
                    sample_multiplier=float(sample_multiplier),
                    pairing_id=pairing_id,
                    variable_type_regime="metadata_profile",
                    n_over_d=n_train / metadata.total_variables,
                )
            )

    for profile_id in profile_ids:
        if "m1_matched" in enabled_panels:
            add_cell(
                profile_id=profile_id,
                panel="m1_matched",
                data_variant="metadata",
                causal_regime="unrestricted",
                negative_control="none",
                sample_multiplier=1.0,
                pairing_suffix="primary",
                replicates=primary_replicates,
            )
        if (
            secondary_replicates
            and "m2_gaussian_pair" in enabled_panels
            and in_scope("m2_gaussian_pair", profile_id)
        ):
            add_cell(
                profile_id=profile_id,
                panel="m2_gaussian_pair",
                data_variant="gaussian",
                causal_regime="unrestricted",
                negative_control="none",
                sample_multiplier=1.0,
                pairing_suffix="primary",
                replicates=secondary_replicates,
            )
        if (
            secondary_replicates
            and "m3_sample_size" in enabled_panels
            and in_scope("m3_sample_size", profile_id)
        ):
            for multiplier in (0.5, 2.0):
                add_cell(
                    profile_id=profile_id,
                    panel="m3_sample_size",
                    data_variant="metadata",
                    causal_regime="unrestricted",
                    negative_control="none",
                    sample_multiplier=multiplier,
                    pairing_suffix="primary",
                    replicates=secondary_replicates,
                )
        if (
            secondary_replicates
            and "m4_cg_compatible" in enabled_panels
            and in_scope("m4_cg_compatible", profile_id)
        ):
            add_cell(
                profile_id=profile_id,
                panel="m4_cg_compatible",
                data_variant=str(design["panels"]["m4_cg_compatible"]["data_variant"]),
                causal_regime="cg_compatible",
                negative_control="none",
                sample_multiplier=1.0,
                pairing_suffix="cg-compatible",
                replicates=secondary_replicates,
            )
        if (
            secondary_replicates
            and "m5_negative_controls" in enabled_panels
            and in_scope("m5_negative_controls", profile_id)
        ):
            for control in ("empty", "pd_shuffle"):
                add_cell(
                    profile_id=profile_id,
                    panel="m5_negative_controls",
                    data_variant="metadata",
                    causal_regime="unrestricted",
                    negative_control=control,
                    sample_multiplier=1.0,
                    pairing_suffix=(
                        f"null-{control}" if control == "empty" else "primary"
                    ),
                    replicates=secondary_replicates,
                )

    smc_ranks = 8 if profile == "smoke" else 64
    runs: list[RunSpec] = []

    def add_method_runs(
        dataset: MetadataDatasetSpec,
        method: str,
        target: str,
        count: int,
        role: str = "evaluation",
        seed_start: int = 0,
    ) -> None:
        for seed_index in range(seed_start, seed_start + count):
            run = _method_run(
                master_seed=master_seed,
                dataset=dataset,
                method=method,
                target=target,
                seed_index=seed_index,
                role=role,
                wall_seconds=1200,
            )
            if method == "ordersmc":
                run = RunSpec(**{**asdict(run), "mpi_ranks": smc_ranks})
            runs.append(run)

    order_seeds = int(cohort["ordersmc_seeds"])
    reproducibility_seeds = int(cohort["reproducibility_seeds"])
    comparator_seeds = int(cohort["comparator_seeds"])
    for dataset in datasets:
        add_method_runs(
            dataset,
            "ordersmc",
            "order-modular",
            order_seeds,
        )
        if dataset.role == "m1_matched" and reproducibility_seeds > order_seeds:
            add_method_runs(
                dataset,
                "ordersmc",
                "order-modular",
                reproducibility_seeds - order_seeds,
                role="reproducibility-extension",
                seed_start=order_seeds,
            )
        if dataset.role != "m5_negative_controls":
            add_method_runs(
                dataset,
                "order-mcmc",
                "order-modular",
                comparator_seeds,
            )
        add_method_runs(
            dataset, "bnlearn-hc-shared-score", "order-modular", 1
        )
        if dataset.role in {"m1_matched", "m2_gaussian_pair", "m3_sample_size", "m4_cg_compatible"}:
            add_method_runs(
                dataset, "bnlearn-tabu-shared-score", "order-modular", 1
            )
        if dataset.role == "m4_cg_compatible":
            add_method_runs(
                dataset, "bnlearn-hc-bic-cg", "native-bic-cg", 1
            )
            add_method_runs(
                dataset, "bnlearn-tabu-bic-cg", "native-bic-cg", 1
            )

    settings = _common_settings()
    settings.update(
        {
            "study_section": 2,
            "study_version": str(design["study_version"]),
            "profile": profile,
            "profile_priority": list(PROFILE_PRIORITY),
            "metadata_profiles": list(profile_ids),
            "panels": sorted(enabled_panels),
            "ordersmc_mpi_ranks": smc_ranks,
            "m0_source_panel": "m1_matched",
            "statistical_unit": "independent-DAG-dataset-pair",
            "method_seeds_nested_within_dataset": True,
            "full_status": design["status"],
            "gated_future_adapters": design["gated_future_adapters"],
        }
    )
    return ExperimentPlan(
        f"section2-metadata-{design['study_version']}",
        "s2",
        master_seed,
        tuple(datasets),
        tuple(runs),
        settings,
    )


def build_experiment_plan(
    experiment: str, *, master_seed: int, profile: str = "full"
) -> ExperimentPlan:
    """Build a complete deterministic plan without generating data or running inference."""

    key = experiment.strip().lower()
    if key == "v1":
        return _build_v1(master_seed, profile)
    if key == "v2":
        return _build_v2(master_seed, profile)
    if key == "v3":
        return _build_v3(master_seed, profile)
    if key == "v3-clg":
        return _build_v3_clg(master_seed, profile)
    if key == "s2":
        return _build_s2(master_seed, profile)
    builders = {"e1": _build_e1, "e2": _build_e2, "e3": _build_e3, "e4": _build_e4}
    if key not in builders:
        raise ValueError(f"unknown experiment {experiment!r}; choose one of {EXPERIMENT_IDS}")
    return builders[key](master_seed)
