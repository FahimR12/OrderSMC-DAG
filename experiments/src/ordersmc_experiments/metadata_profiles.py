"""Metadata-calibrated known-truth DAGs for paper Section 2.

Only distributional summaries are imported. Each truth DAG is independently
drawn: no empirical variable label, edge, direction, or Markov blanket is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kurtosis, skew

from .seeds import protocol_seed_record


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_ROOT = (
    REPOSITORY_ROOT / "summary_statistics_multiopmic_data_pathway_molecule_level"
)
PROFILE_DIRECTORIES: Mapping[str, str] = {
    "rna_pathway": "01_rna_pathway_196var",
    "protein_pathway": "02_protein_pathway_196var",
    "joint_pathway": "03_joint_pathway_196var",
    "rna_molecule": "04_rna_molecule_22var",
    "protein_molecule": "05_protein_molecule_70var",
    "joint_molecule": "06_joint_molecule_89var",
}
PROFILE_PRIORITY = (
    "protein_pathway",
    "rna_pathway",
    "protein_molecule",
    "rna_molecule",
    "joint_molecule",
    "joint_pathway",
)
DataVariant = Literal["gaussian", "metadata"]
CausalRegime = Literal["unrestricted", "cg_compatible"]
NegativeControl = Literal["none", "empty", "pd_shuffle"]
_QPROBS = np.asarray([.001, .01, .05, .25, .5, .75, .95, .99, .999])
_QCOLS = ("minimum", "q01", "q05", "q25", "median", "q75", "q95", "q99", "maximum")


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class MetadataProfile:
    profile_id: str
    directory: Path
    participants: int
    total_variables: int
    biological_variables: int
    rna_variables: int
    protein_variables: int
    pd_prevalence: float
    effective_rank: float
    components_50pct: int
    components_80pct: int
    components_90pct: int
    components_95pct: int
    correlation_targets: tuple[Mapping[str, Any], ...]
    variable_statistics: tuple[Mapping[str, Any], ...]
    covariate_correlations: Mapping[str, Mapping[str, float]]
    source_hashes: Mapping[str, str]

    @property
    def all_biological_correlation_target(self) -> Mapping[str, Any]:
        return next(
            row
            for row in self.correlation_targets
            if str(row["block"]) == "all biological variables"
        )


def load_metadata_profile(
    profile_id: str, metadata_root: Path = DEFAULT_METADATA_ROOT
) -> MetadataProfile:
    """Load and fail closed on a malformed privacy-preserving profile."""

    key = profile_id.strip().lower()
    if key not in PROFILE_DIRECTORIES:
        raise ValueError(f"unknown metadata profile {profile_id!r}")
    directory = metadata_root / PROFILE_DIRECTORIES[key]
    summary = json.loads((directory / "profile_summary.json").read_text())
    targets = json.loads((directory / "simulation_targets.json").read_text())
    if summary.get("profile_id") != key:
        raise ValueError("profile id mismatch")
    if summary.get("contains_participant_level_values") is not False:
        raise ValueError("Section 2 accepts summary metadata, not participant values")
    if summary.get("contains_fitted_graph_edges") is not False:
        raise ValueError("Section 2 must not import a fitted graph")
    matched = targets.get("matched_scenario", {})
    if int(matched.get("total_p", -1)) != int(summary["total_variables"]):
        raise ValueError("simulation target and profile dimensions disagree")
    if "Generate true DAGs independently" not in str(targets.get("guardrail", "")):
        raise ValueError("independent-DAG guardrail is missing")
    variable_frame = pd.read_csv(directory / "variable_statistics.csv")
    if len(variable_frame) != int(summary["biological_variables"]):
        raise ValueError("variable-statistics row count disagrees with biological p")
    correlation_frame = pd.read_csv(directory / "correlation_quantiles.csv")
    covariate_frame = pd.read_csv(directory / "covariate_correlation_matrix.csv", index_col=0)
    required = (
        "profile_summary.json", "simulation_targets.json", "variable_statistics.csv",
        "correlation_quantiles.csv", "distribution_quantiles.csv", "latent_spectrum.csv",
        "covariate_correlation_matrix.csv", "biological_to_covariate_correlations.csv",
    )
    return MetadataProfile(
        profile_id=key,
        directory=directory,
        participants=int(summary["participants"]),
        total_variables=int(summary["total_variables"]),
        biological_variables=int(summary["biological_variables"]),
        rna_variables=int(summary["RNA_variables"]),
        protein_variables=int(summary["protein_variables"]),
        pd_prevalence=float(summary["PD_prevalence"]),
        effective_rank=float(summary["effective_rank"]),
        components_50pct=int(summary["components_50pct"]),
        components_80pct=int(summary["components_80pct"]),
        components_90pct=int(summary["components_90pct"]),
        components_95pct=int(summary["components_95pct"]),
        correlation_targets=tuple(correlation_frame.to_dict(orient="records")),
        variable_statistics=tuple(variable_frame.to_dict(orient="records")),
        covariate_correlations={
            str(row): {str(column): float(value) for column, value in values.items()}
            for row, values in covariate_frame.to_dict(orient="index").items()
        },
        source_hashes={name: _file_sha(directory / name) for name in required},
    )


@dataclass(frozen=True)
class MetadataDAGConfig:
    profile_id: str
    n_train: int | None = None
    n_test: int | None = None
    dimension: int | None = None
    data_variant: DataVariant = "metadata"
    causal_regime: CausalRegime = "unrestricted"
    negative_control: NegativeControl = "none"
    graph_family: Literal["modular", "scale_free"] = "modular"
    expected_indegree: float = 1.5
    max_indegree: int = 2
    # Female prevalence is absent from the supplied summaries. The explicit
    # neutral assumption must be replaced if that non-sensitive count is supplied.
    female_prevalence: float = 0.5
    pilot_size: int = 2_000
    calibration_scales: tuple[float, ...] = (
        .35, .55, .75, 1.00, 1.30, 1.60, 1.90, 2.20, 2.50, 2.80, 3.20
    )
    calibration_method: Literal["population-v1", "finite-sample-v2"] = "population-v1"
    calibration_repeats: int = 8
    covariate_effect_policy: Literal["legacy", "metadata-ratio-v1"] = "legacy"
    master_seed: int = 0
    experiment_id: str = "s2"
    cell_id: str = "protein-pathway-matched"
    pairing_id: str = "protein-pathway-primary"
    replicate_id: int = 0

    def __post_init__(self) -> None:
        if self.profile_id not in PROFILE_DIRECTORIES:
            raise ValueError("unknown profile_id")
        if self.n_train is not None and self.n_train < 10:
            raise ValueError("n_train must be at least 10")
        if self.n_test is not None and self.n_test < 10:
            raise ValueError("n_test must be at least 10")
        if self.dimension is not None and self.dimension < 6:
            raise ValueError("dimension must be at least 6")
        if self.data_variant not in {"gaussian", "metadata"}:
            raise ValueError("unsupported data_variant")
        if self.causal_regime not in {"unrestricted", "cg_compatible"}:
            raise ValueError("unsupported causal_regime")
        if self.negative_control not in {"none", "empty", "pd_shuffle"}:
            raise ValueError("unsupported negative_control")
        if self.graph_family not in {"modular", "scale_free"}:
            raise ValueError("unsupported graph_family")
        if not 0 < self.expected_indegree <= self.max_indegree or self.max_indegree != 2:
            raise ValueError("Section 2 v1 requires expected indegree in (0, 2] and max indegree 2")
        if not 0 < self.female_prevalence < 1:
            raise ValueError("female_prevalence must lie in (0, 1)")
        if self.pilot_size < 200 or not self.calibration_scales:
            raise ValueError("pilot_size >= 200 and a nonempty scale grid are required")
        if any(scale <= 0 for scale in self.calibration_scales):
            raise ValueError("calibration scales must be positive")
        if self.calibration_method not in {"population-v1", "finite-sample-v2"}:
            raise ValueError("unsupported calibration_method")
        if self.covariate_effect_policy not in {"legacy", "metadata-ratio-v1"}:
            raise ValueError("unsupported covariate_effect_policy")
        if isinstance(self.calibration_repeats, bool) or not isinstance(self.calibration_repeats, int) or self.calibration_repeats < 2:
            raise ValueError("calibration_repeats must be an integer >= 2")
        if self.master_seed < 0 or not 0 <= self.replicate_id < 10**6:
            raise ValueError("invalid seed or replicate")
        for value in (self.experiment_id, self.cell_id, self.pairing_id):
            if not value or value != value.strip().lower() or "|" in value:
                raise ValueError("seed tokens must be canonical lowercase strings")


@dataclass
class MetadataDAGDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    dag: nx.DiGraph
    variable_types: dict[str, str]
    cardinalities: dict[str, int]
    manifest: dict[str, Any]


def metadata_config_for_spec(spec, master_seed, settings, pilot_size):
    """One shared configuration path for fidelity audits and scientific fits."""
    return MetadataDAGConfig(
        profile_id=str(spec["metadata_profile"]), n_train=int(spec["n_train"]),
        n_test=int(spec["n_train"]), dimension=int(spec["dimension"]),
        data_variant=str(spec["data_variant"]), causal_regime=str(spec["causal_regime"]),
        negative_control=str(spec["negative_control"]), graph_family=str(spec["graph_family"]),
        expected_indegree=float(spec["expected_indegree"]), max_indegree=int(spec["max_indegree"]),
        female_prevalence=float(settings.get("female_prevalence", .5)), pilot_size=pilot_size,
        calibration_method=str(settings.get("calibration_method", "population-v1")),
        calibration_repeats=int(settings.get("calibration_repeats", 8)),
        covariate_effect_policy=str(settings.get("covariate_effect_policy", "legacy")),
        master_seed=int(master_seed), experiment_id="s2", cell_id=str(spec["cell_id"]),
        pairing_id=str(spec["pairing_id"]), replicate_id=int(spec["replicate"]))


def _record(config: MetadataDAGConfig, component: str):
    return protocol_seed_record(
        config.master_seed, experiment=config.experiment_id, cell=config.pairing_id,
        replicate=config.replicate_id, component=component,
    )


def _rng(config: MetadataDAGConfig, component: str) -> np.random.Generator:
    return _record(config, component).generator()


def _biological_counts(profile: MetadataProfile, dimension: int) -> tuple[int, int]:
    biological = dimension - 3
    if biological < 3 or biological > profile.biological_variables:
        raise ValueError("dimension cannot exceed the profile and must retain three covariates")
    if profile.rna_variables == 0:
        return 0, biological
    if profile.protein_variables == 0:
        return biological, 0
    rna = round(biological * profile.rna_variables / profile.biological_variables)
    rna = min(biological - 1, max(1, int(rna)))
    return rna, biological - rna


def _nodes_and_targets(profile, config, dimension):
    rna_count, protein_count = _biological_counts(profile, dimension)
    nodes = [f"RNA_{i:03d}" for i in range(rna_count)]
    nodes += [f"PROTEIN_{i:03d}" for i in range(protein_count)]
    rows_by_assay = {
        assay: [row for row in profile.variable_statistics if str(row["assay"]).lower() == assay]
        for assay in ("rna", "protein")
    }
    targets = {}
    rng = _rng(config, "marginal-target-assignment")
    for assay, assay_nodes in (
        ("rna", [node for node in nodes if node.startswith("RNA_")]),
        ("protein", [node for node in nodes if node.startswith("PROTEIN_")]),
    ):
        rows = rows_by_assay[assay]
        selected = rng.permutation(len(rows))[:len(assay_nodes)]
        for node, index in zip(assay_nodes, selected):
            row = rows[int(index)]
            targets[node] = {
                key: float(row[key])
                for key in (*_QCOLS, "skewness", "excess_kurtosis")
            }
    modalities = {node: ("rna" if node.startswith("RNA_") else "protein") for node in nodes}
    return nodes, targets, modalities


def _choose(rng, values, weights, count):
    count = min(count, len(values))
    if count <= 0:
        return []
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    indices = rng.choice(len(values), size=count, replace=False, p=probabilities)
    return [str(values[int(index)]) for index in np.atleast_1d(indices)]


def _generate_independent_dag(biological_nodes, profile, config):
    rng = _rng(config, "independent-dag")
    perm = rng.permutation(len(biological_nodes))
    biological_order = [str(biological_nodes[int(index)]) for index in perm]
    # Primary: age -> observed PD state -> molecular measurements. Sex is
    # exogenous; its weak observed PD association remains a calibration target.
    # This is a cross-sectional state model, not a claim about disease onset. The
    # continuous age -> binary PD edge is deliberately outside native CLG.
    order = (
        ["age_z", "female", "PD", *biological_order]
        if config.causal_regime == "unrestricted"
        else ["female", "PD", "age_z", *biological_order]
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    if config.negative_control == "empty":
        return graph, tuple(order), {node: 0 for node in biological_nodes}
    module_count = max(2, min(profile.components_50pct, int(math.sqrt(len(biological_nodes)))))
    modules = {node: index % module_count for index, node in enumerate(biological_order)}
    outdegree = {node: 0 for node in order}
    for child in biological_order:
        available = order[:order.index(child)]
        count = 1 + int(rng.random() < max(0.0, config.expected_indegree - 1))
        weights = []
        for parent in available:
            if parent in {"age_z", "female", "PD"}:
                weight = {"PD": 3.0, "age_z": .05, "female": .03}[parent]
            elif modules[parent] == modules[child]:
                weight = 5.
            else:
                weight = 1.
            if config.graph_family == "scale_free":
                weight *= (1 + outdegree[parent]) ** 1.25
            weights.append(weight)
        for parent in _choose(rng, available, weights, min(config.max_indegree, count)):
            graph.add_edge(parent, child)
            outdegree[parent] += 1
    if config.causal_regime == "unrestricted":
        graph.add_edge("age_z", "PD")
    else:
        graph.add_edge("PD", "age_z")
        graph.add_edge("female", "age_z")
    if config.negative_control == "pd_shuffle":
        graph.remove_edges_from(list(graph.in_edges("PD")) + list(graph.out_edges("PD")))
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("metadata generator produced a cycle")
    if max(dict(graph.in_degree()).values(), default=0) > config.max_indegree:
        raise RuntimeError("metadata generator violated maximum indegree")
    return graph, tuple(order), modules


def _binary_intercept(linear, target):
    low, high = -40., 40.
    for _ in range(100):
        middle = .5 * (low + high)
        if float(np.mean(expit(middle + linear))) < target:
            low = middle
        else:
            high = middle
    return .5 * (low + high)


def _standardize_parent(values, kind, prevalence=None):
    raw = np.asarray(values, dtype=float)
    if kind == "bin":
        prevalence = float(np.mean(raw)) if prevalence is None else prevalence
        return (raw - prevalence) / math.sqrt(max(1e-8, prevalence * (1 - prevalence)))
    return raw


def _interp(values, x, y):
    result = np.interp(values, x, y)
    left, right = values < x[0], values > x[-1]
    if np.any(left):
        result[left] = y[0] + (values[left] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if np.any(right):
        result[right] = y[-1] + (values[right] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return result


def _edge_coefficients(graph, modules, config):
    coefficients = {}
    for child in graph.nodes:
        rng = _rng(config, f"coefficient-{child.lower()}")
        coefficients[child] = {}
        for parent in sorted(graph.predecessors(child)):
            if child == "PD" and parent == "age_z":
                coefficients[child][parent] = .25
                continue
            if parent == "PD" and child in modules:
                coefficients[child][parent] = .17 * (-1 if rng.random() < .45 else 1)
                continue
            same = parent in modules and child in modules and modules[parent] == modules[child]
            magnitude = float(rng.uniform(*((.45, .75) if same else (.30, .65))))
            coefficients[child][parent] = magnitude * (-1 if rng.random() < .45 else 1)
    return coefficients


def _build_mechanisms(
    graph, order, variable_types, marginal_targets, base_coefficients,
    profile, config, scale, variant,
):
    mechanisms, values = {}, {}
    prevalences = {"female": config.female_prevalence, "PD": profile.pd_prevalence}
    n = config.pilot_size
    for node in order:
        kind = variable_types[node]
        parents = tuple(sorted(graph.predecessors(node)))
        rng = _rng(config, f"calibration-noise-{node.lower()}")
        if kind == "bin":
            linear = np.zeros(n)
            coefficients = {}
            for parent in parents:
                coefficient = scale * float(base_coefficients[node][parent])
                coefficients[parent] = coefficient
                linear += coefficient * _standardize_parent(
                    values[parent], variable_types[parent], prevalences.get(parent)
                )
            intercept = _binary_intercept(linear, prevalences[node])
            values[node] = rng.binomial(1, expit(intercept + linear), size=n).astype(np.int64)
            mechanisms[node] = {
                "kind": "binomial-logit", "parents": list(parents),
                "intercept": intercept, "coefficients": coefficients,
                "target_prevalence": prevalences[node],
            }
            continue
        raw, coefficients = np.zeros(n), {}
        for parent in parents:
            coefficient = (
                .37 if node == "age_z" and parent == "PD"
                else scale * float(base_coefficients[node][parent])
            )
            coefficients[parent] = coefficient
            raw += coefficient * _standardize_parent(
                values[parent], variable_types[parent], prevalences.get(parent)
            )
        raw += rng.normal(size=n)
        if variant == "metadata" and node in marginal_targets:
            x = np.quantile(raw, _QPROBS)
            y = np.asarray([marginal_targets[node][name] for name in _QCOLS])
            mapped = _interp(raw, x, y)
            transform = {
                "kind": "monotone-quantile-map", "x_knots": x.tolist(),
                "y_knots": y.tolist(), "post_mean": float(mapped.mean()),
                "post_sd": float(mapped.std(ddof=0)),
            }
        else:
            mapped = raw
            transform = {
                "kind": "standardize", "post_mean": float(raw.mean()),
                "post_sd": float(raw.std(ddof=0)),
            }
        values[node] = (mapped - transform["post_mean"]) / transform["post_sd"]
        mechanisms[node] = {
            "kind": "continuous-sem", "parents": list(parents),
            "coefficients": coefficients, "transform": transform,
        }
    return mechanisms, pd.DataFrame({node: values[node] for node in sorted(values)})


def _sample(n, split, order, mechanisms, variable_types, profile, config):
    values = {}
    prevalences = {"female": config.female_prevalence, "PD": profile.pd_prevalence}
    for node in order:
        mechanism = mechanisms[node]
        parents = tuple(mechanism["parents"])
        rng = _rng(config, f"{split}-noise-{node.lower()}")
        if variable_types[node] == "bin":
            linear = np.full(n, float(mechanism["intercept"]))
            for parent in parents:
                linear += float(mechanism["coefficients"][parent]) * _standardize_parent(
                    values[parent], variable_types[parent], prevalences.get(parent)
                )
            values[node] = rng.binomial(1, expit(linear), size=n).astype(np.int64)
            continue
        raw = np.zeros(n)
        for parent in parents:
            raw += float(mechanism["coefficients"][parent]) * _standardize_parent(
                values[parent], variable_types[parent], prevalences.get(parent)
            )
        raw += rng.normal(size=n)
        transform = mechanism["transform"]
        mapped = (
            _interp(raw, np.asarray(transform["x_knots"]), np.asarray(transform["y_knots"]))
            if transform["kind"] == "monotone-quantile-map" else raw
        )
        values[node] = (mapped - transform["post_mean"]) / transform["post_sd"]
    return pd.DataFrame({node: values[node] for node in sorted(values)})


def metadata_fidelity_statistics(data, biological_nodes, profile):
    biological = data.loc[:, list(biological_nodes)].astype(float)
    correlation = biological.corr().to_numpy()
    upper = correlation[np.triu_indices(len(biological_nodes), 1)]
    eigenvalues = np.sort(np.clip(np.linalg.eigvalsh(correlation), 0, None))[::-1]
    weights = eigenvalues / max(eigenvalues.sum(), np.finfo(float).tiny)
    positive = weights > 0
    cumulative = np.cumsum(weights)
    count = lambda fraction: float(np.searchsorted(cumulative, fraction) + 1)
    matrix = biological.to_numpy()
    result = {
        "pd_prevalence": float(data["PD"].mean()),
        "age_pd_correlation": float(data["age_z"].corr(data["PD"])),
        "female_pd_correlation": float(data["female"].corr(data["PD"])),
        "absolute_correlation_median": float(np.median(np.abs(upper))),
        "absolute_correlation_q95": float(np.quantile(np.abs(upper), .95)),
        "effective_rank": float(np.exp(-np.sum(weights[positive] * np.log(weights[positive])))),
        "components_50pct": count(.5), "components_80pct": count(.8),
        "components_90pct": count(.9), "components_95pct": count(.95),
        "skewness_median": float(np.median(skew(matrix, axis=0, bias=False))),
        "excess_kurtosis_median": float(
            np.median(kurtosis(matrix, axis=0, fisher=True, bias=False))
        ),
    }
    for covariate in ("age_z", "female", "PD"):
        result[f"mean_abs_correlation_with_{covariate}"] = float(np.mean([
            abs(float(data[node].corr(data[covariate]))) for node in biological_nodes
        ]))
    return result


def _target_statistics(profile, dimension):
    correlation = profile.all_biological_correlation_target
    fraction = (dimension - 3) / profile.biological_variables
    frame = pd.DataFrame(profile.variable_statistics)
    result = {
        "pd_prevalence": profile.pd_prevalence,
        "age_pd_correlation": float(profile.covariate_correlations["age_z"]["PD"]),
        "female_pd_correlation": float(profile.covariate_correlations["female"]["PD"]),
        "absolute_correlation_median": float(correlation["abs_cor_median"]),
        "absolute_correlation_q95": float(correlation["abs_cor_q95"]),
        "effective_rank": max(1., profile.effective_rank * fraction),
        "components_50pct": max(1., round(profile.components_50pct * fraction)),
        "components_80pct": max(1., round(profile.components_80pct * fraction)),
        "components_90pct": max(1., round(profile.components_90pct * fraction)),
        "components_95pct": max(1., round(profile.components_95pct * fraction)),
        "skewness_median": float(frame["skewness"].median()),
        "excess_kurtosis_median": float(frame["excess_kurtosis"].median()),
    }
    for covariate in ("age_z", "female", "PD"):
        result[f"mean_abs_correlation_with_{covariate}"] = float(
            frame[f"correlation_with_{covariate}"].abs().mean()
        )
    return result


def _loss(achieved, target):
    denominators = {
        "absolute_correlation_median": .05, "absolute_correlation_q95": .08,
        "effective_rank": max(2., .20 * target["effective_rank"]),
        "components_80pct": max(2., .20 * target["components_80pct"]),
        "mean_abs_correlation_with_age_z": .05,
        "mean_abs_correlation_with_female": .05,
        "mean_abs_correlation_with_PD": .05,
        "age_pd_correlation": .05,
        "female_pd_correlation": .05,
    }
    return float(sum(
        ((achieved[name] - target[name]) / denominator) ** 2
        for name, denominator in denominators.items()
    ))


def generate_metadata_dag_dataset(
    config: MetadataDAGConfig, metadata_root: Path = DEFAULT_METADATA_ROOT
) -> MetadataDAGDataset:
    profile = load_metadata_profile(config.profile_id, metadata_root)
    dimension = profile.total_variables if config.dimension is None else config.dimension
    n_train = profile.participants if config.n_train is None else config.n_train
    n_test = n_train if config.n_test is None else config.n_test
    biological, marginal_targets, modalities = _nodes_and_targets(profile, config, dimension)
    graph, order, modules = _generate_independent_dag(biological, profile, config)
    types = {node: "con" for node in biological}
    types.update({"age_z": "con", "female": "bin", "PD": "bin"})
    cards = {"female": 2, "PD": 2}
    coefficients = _edge_coefficients(graph, modules, config)
    target = _target_statistics(profile, dimension)
    if config.covariate_effect_policy == "metadata-ratio-v1":
        # The legacy template treats direct age/sex effects like arbitrary
        # molecular edges (.30--.65), while direct PD effects have magnitude
        # .17. Put exogenous covariates on the same baseline, scaled by the
        # uploaded mean absolute covariate associations. Retain sampled signs
        # and relative magnitudes; never change edges or consult recovery.
        pd_association = max(target["mean_abs_correlation_with_PD"], 1e-12)
        for child in biological:
            for parent in ("age_z", "female"):
                if parent in coefficients[child]:
                    ratio = target[f"mean_abs_correlation_with_{parent}"] / pd_association
                    coefficients[child][parent] *= (.17 / .475) * ratio
    scale_rows, best = [], None
    for scale in config.calibration_scales:
        mechanisms, pilot = _build_mechanisms(
            graph, order, types, marginal_targets, coefficients, profile, config,
            float(scale), "metadata",
        )
        if config.calibration_method == "finite-sample-v2":
            # Metadata are sample statistics at the original study size, not
            # population targets. Always use profile.participants here, even
            # for M3 half/double-n arms, to preserve identical mechanisms.
            draws = [metadata_fidelity_statistics(
                _sample(profile.participants, f"calibration-evaluate-{repeat:03d}",
                        order, mechanisms, types, profile, config), biological, profile)
                for repeat in range(config.calibration_repeats)]
            achieved = {key: float(np.mean([row[key] for row in draws])) for key in draws[0]}
        else:
            achieved = metadata_fidelity_statistics(pilot, biological, profile)
        row = {"scale": float(scale), "loss": _loss(achieved, target), **achieved}
        scale_rows.append(row)
        if best is None or row["loss"] < best[0]:
            best = (row["loss"], float(scale), mechanisms)
    assert best is not None
    selected_loss, selected_scale, metadata_mechanisms = best
    if config.data_variant == "metadata":
        mechanisms = metadata_mechanisms
    else:
        mechanisms, _ = _build_mechanisms(
            graph, order, types, marginal_targets, coefficients, profile, config,
            selected_scale, "gaussian",
        )
    train = _sample(n_train, "train", order, mechanisms, types, profile, config)
    test = _sample(n_test, "test", order, mechanisms, types, profile, config)
    if config.negative_control == "pd_shuffle":
        for split, frame in (("train", train), ("test", test)):
            permutation = _rng(config, f"{split}-pd-shuffle").permutation(len(frame))
            frame.loc[:, "PD"] = frame["PD"].to_numpy()[permutation]
    components = (
        "independent-dag", "marginal-target-assignment",
        *(f"coefficient-{node.lower()}" for node in graph.nodes),
        *(f"train-noise-{node.lower()}" for node in graph.nodes),
        *(f"test-noise-{node.lower()}" for node in graph.nodes),
    )
    manifest = {
        "schema": "ordersmc-metadata-calibrated-dag-v1",
        "generator": "independent-dag-aggregate-metadata-calibration-v1",
        "guardrail": (
            "Distributional summaries only; no participant values, empirical labels, "
            "fitted edges, directions, or Markov blankets are used as truth."
        ),
        "config": asdict(config),
        "profile": {
            "profile_id": profile.profile_id, "participants": profile.participants,
            "total_variables": profile.total_variables,
            "biological_variables": profile.biological_variables,
            "rna_variables": profile.rna_variables,
            "protein_variables": profile.protein_variables,
            "source_hashes": dict(profile.source_hashes),
        },
        "dimension": dimension, "n_train": n_train, "n_test": n_test,
        "variable_types": dict(sorted(types.items())), "cardinalities": cards,
        "modalities": dict(sorted(modalities.items())), "topological_order": list(order),
        "edges": [list(edge) for edge in sorted(graph.edges())],
        "mechanisms": mechanisms,
        "calibration": {
            "selection_uses_graph_recovery": False, "selected_scale": selected_scale,
            "selected_loss": selected_loss, "scale_grid": scale_rows, "target": target,
            "train": metadata_fidelity_statistics(train, biological, profile),
            "independent_test": metadata_fidelity_statistics(test, biological, profile),
        },
        "seed_records": {component: asdict(_record(config, component)) for component in components},
    }
    if config.calibration_method == "population-v1":
        # Preserve default legacy artifacts and manifests byte-for-byte.
        manifest["config"].pop("calibration_method")
        manifest["config"].pop("calibration_repeats")
    else:
        manifest["generator"] = "independent-dag-finite-sample-metadata-calibration-v2"
        manifest["calibration"].update(
            method=config.calibration_method, evaluation_sample_size=profile.participants,
            evaluation_repeats=config.calibration_repeats,
            evaluation_independent_of_training_and_test=True,
            selection_metrics="mean finite-sample summaries at the metadata cohort n",
        )
    if config.covariate_effect_policy == "legacy":
        manifest["config"].pop("covariate_effect_policy")
    else:
        manifest["calibration"]["covariate_effect_policy"] = config.covariate_effect_policy
    return MetadataDAGDataset(
        train, test, graph, dict(sorted(types.items())), cards, manifest
    )


def _frame_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()


def _adjacency_bytes(graph):
    nodes = sorted(graph.nodes)
    frame = pd.DataFrame(
        nx.to_numpy_array(graph, nodelist=nodes, dtype=np.int8), index=nodes, columns=nodes
    )
    return frame.to_csv(index=True, lineterminator="\n").encode()


def write_metadata_dag_artifact(dataset: MetadataDAGDataset, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    payloads = {
        "train.csv": _frame_bytes(dataset.train),
        "test.csv": _frame_bytes(dataset.test),
        "adjacency.csv": _adjacency_bytes(dataset.dag),
    }
    artifacts = {
        name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        for name, payload in sorted(payloads.items())
    }
    manifest = dict(dataset.manifest)
    manifest["artifacts"] = artifacts
    manifest["artifact_fingerprint_sha256"] = _canonical_sha(manifest)
    payloads["generator_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    audit = {
        "schema": "ordersmc-metadata-calibrated-bundle-v1", "status": "complete",
        "artifacts": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    audit["bundle_sha256"] = _canonical_sha(audit["artifacts"])
    (destination / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def load_metadata_dag_artifact(source: Path) -> MetadataDAGDataset:
    manifest = json.loads((source / "generator_manifest.json").read_text())
    if manifest.get("schema") != "ordersmc-metadata-calibrated-dag-v1":
        raise ValueError("not a metadata-calibrated DAG artifact")
    claimed = manifest.get("artifact_fingerprint_sha256")
    body = dict(manifest)
    body.pop("artifact_fingerprint_sha256", None)
    if claimed != _canonical_sha(body):
        raise RuntimeError("metadata-calibrated manifest fingerprint mismatch")
    for name, record in manifest["artifacts"].items():
        payload = (source / name).read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise RuntimeError(f"metadata-calibrated artifact hash mismatch: {name}")
    train, test = pd.read_csv(source / "train.csv"), pd.read_csv(source / "test.csv")
    types = {str(key): str(value) for key, value in manifest["variable_types"].items()}
    for node, kind in types.items():
        if kind == "bin":
            train[node], test[node] = train[node].astype(np.int64), test[node].astype(np.int64)
    graph = nx.DiGraph()
    graph.add_nodes_from(types)
    graph.add_edges_from(tuple(map(tuple, manifest["edges"])))
    return MetadataDAGDataset(
        train, test, graph, types,
        {str(key): int(value) for key, value in manifest["cardinalities"].items()},
        manifest,
    )
