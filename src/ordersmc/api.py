"""Validated high-level local and MPI APIs."""

from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Mapping

import numpy as np
import pandas as pd

from ._version import SCORE_SCHEMA_VERSION, __version__
from .config import DAGConfig, OrderSMCConfig, candidate_build_config, sampling_config
from .core import OrderCandidateTable, OrderSMCResult, build_order_candidate_table, order_smc
from .data import metadata_to_mapping, validate_data
from .scoring import UnifiedFamilyScorer

Backend = Literal["auto", "local", "mpi"]
ProvenancePolicy = Literal["require", "warn", "ignore"]
PROVENANCE_VERSION = 2


def make_scorer(
    data: pd.DataFrame,
    vtypes: Dict[str, str],
    cards: Dict[str, int],
    config: OrderSMCConfig,
) -> UnifiedFamilyScorer:
    return UnifiedFamilyScorer(
        data,
        vtypes=vtypes,
        cards=cards,
        alpha=config.alpha,
        tau=config.tau,
        glm_tau=config.glm_tau,
        use_sparse_bdeu=config.discrete_score == "bds",
        clg_ig_shape=config.clg_ig_shape,
        clg_ig_scale=config.clg_ig_scale,
        standardize_continuous=config.standardize_continuous,
        score_method=config.score_method,
        dg_penalty_discount=config.dg_penalty_discount,
        dg_ridge=config.dg_ridge,
    )


def make_prior(config: OrderSMCConfig) -> DAGConfig:
    return DAGConfig(
        lam=config.lam,
        edge_penalty=config.edge_penalty,
        max_parents=config.max_parents,
    )


def build_candidates(
    data: pd.DataFrame,
    vtypes: Dict[str, str],
    cards: Dict[str, int],
    *,
    config: OrderSMCConfig = OrderSMCConfig(),
    backend: Backend = "local",
    comm=None,
    verbose: bool = True,
) -> OrderCandidateTable:
    """Build a validated candidate table locally or across MPI ranks."""
    validate_data(data, vtypes, cards, max_parents=config.max_parents)
    backend, comm = _resolve_backend(backend, comm)
    scorer = make_scorer(data, vtypes, cards, config)
    kwargs = dict(
        screen_k=config.screen_k,
        max_parents=config.max_parents,
        keep_per_child=config.keep_per_child,
        pair_rescue_anchor_k=config.pair_rescue_anchor_k,
        pair_rescue_k=config.pair_rescue_k,
        parent_pool_max=config.parent_pool_max,
        triplet_rescue_pair_k=config.triplet_rescue_pair_k,
        triplet_rescue_k=config.triplet_rescue_k,
        tail_edge_penalty_after=config.tail_edge_penalty_after,
        tail_edge_penalty=config.tail_edge_penalty,
    )
    if backend == "mpi":
        from .mpi import build_order_candidate_table_mpi

        table = build_order_candidate_table_mpi(
            scorer, list(data.columns), make_prior(config), comm=comm, verbose=verbose, **kwargs
        )
    else:
        table = build_order_candidate_table(scorer, list(data.columns), make_prior(config), **kwargs)
    build_diagnostics = dict(getattr(table, "build_diagnostics", {}))
    table.provenance = candidate_table_provenance(data, vtypes, cards, config)
    if build_diagnostics:
        table.provenance["candidate_build_diagnostics"] = build_diagnostics
    return table


def sample(
    table: OrderCandidateTable,
    *,
    config: OrderSMCConfig = OrderSMCConfig(),
    backend: Backend = "local",
    comm=None,
    parent_seed_record: Mapping[str, object] | None = None,
    verbose: bool = True,
    provenance_policy: ProvenancePolicy = "warn",
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_every_stages: int = 1,
):
    """Run SMC from a new or restored candidate table."""
    from .persistence import candidate_table_content_sha256

    started = time.perf_counter()
    validate_candidate_provenance(
        table, config=config, missing_policy=provenance_policy
    )
    backend, comm = _resolve_backend(backend, comm)
    resample_fraction = 0.0 if config.resampling == "none" else config.resample_ess_frac
    kwargs = dict(
        particles=config.particles,
        betas=None if config.adaptive_tempering else np.linspace(0.0, 1.0, config.num_betas),
        mh_steps=config.mh_steps,
        mh_steps_resample=config.mh_steps_resample,
        resample_ess_frac=resample_fraction,
        seed=config.seed,
        verbose=verbose,
        tempering_path=config.tempering_path,
        proposal_weights=config.proposal_weights(),
        adaptive_cess_target=config.adaptive_cess_target if config.adaptive_tempering else None,
        adaptive_min_beta_increment=config.adaptive_min_beta_increment,
        adaptive_max_stages=config.adaptive_max_stages,
        final_target_snapshot_steps=config.final_target_snapshot_steps,
    )
    if backend == "mpi":
        if checkpoint_path is not None or resume_checkpoint is not None:
            raise NotImplementedError("resumable SMC checkpoints are currently available only locally")
        from .mpi import order_smc_mpi

        result = order_smc_mpi(
            table,
            comm=comm,
            gather_final=True,
            parent_seed_record=parent_seed_record,
            **kwargs,
        )
    else:
        from .persistence import load_smc_checkpoint, save_smc_checkpoint

        if isinstance(checkpoint_every_stages, (bool, np.bool_)) or not isinstance(
            checkpoint_every_stages, (int, np.integer)
        ):
            raise ValueError("checkpoint_every_stages must be a positive integer")
        checkpoint_interval = int(checkpoint_every_stages)
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_every_stages must be a positive integer")
        resume_state = (
            load_smc_checkpoint(resume_checkpoint, table=table)
            if resume_checkpoint is not None
            else None
        )
        checkpoint_destination = Path(checkpoint_path) if checkpoint_path is not None else None

        def checkpoint_callback(state) -> None:
            if checkpoint_destination is None:
                return
            event_index = state.completed_tempering_stages + len(state.final_target_snapshots)
            requested = set(config.final_target_snapshot_steps)
            snapshots_complete = requested <= {
                snapshot.final_target_mh_sweeps for snapshot in state.final_target_snapshots
            }
            is_complete = state.current_beta >= 1.0 and snapshots_complete
            if event_index % checkpoint_interval == 0 or is_complete:
                save_smc_checkpoint(state, checkpoint_destination)

        local_kwargs = {
            **kwargs,
            "betas": None if config.adaptive_tempering else kwargs["betas"],
            "proposal_weights": config.proposal_weights(),
            "resampling": config.resampling,
            "adaptive_cess_target": (
                config.adaptive_cess_target if config.adaptive_tempering else None
            ),
            "adaptive_min_beta_increment": config.adaptive_min_beta_increment,
            "adaptive_max_stages": config.adaptive_max_stages,
            "final_target_snapshot_steps": config.final_target_snapshot_steps,
            "resume_state": resume_state,
            "checkpoint_callback": checkpoint_callback if checkpoint_destination is not None else None,
        }
        result = order_smc(table, **local_kwargs)
    provenance = getattr(table, "provenance", {})
    result.run_info.update(
        {
            "nodes": list(table.nodes),
            "ordersmc_version": __version__,
            "score_schema_version": SCORE_SCHEMA_VERSION,
            "candidate_score_schema_version": provenance.get("score_schema_version"),
            "config": config.to_dict(),
            "candidate_build_config": (
                dict(provenance["candidate_build_config"])
                if provenance.get("candidate_build_config")
                else None
            ),
            "sampling_config": sampling_config(config),
            "candidate_provenance": dict(provenance),
            "candidate_table_sha256": candidate_table_content_sha256(table),
            "backend": backend,
            "sampling_seconds": time.perf_counter() - started,
            "resumed_from_checkpoint": (
                str(Path(resume_checkpoint)) if resume_checkpoint is not None else None
            ),
            "smc_checkpoint_path": (
                str(checkpoint_destination)
                if backend == "local" and checkpoint_destination is not None
                else None
            ),
        }
    )
    return result


def fit(
    data: pd.DataFrame,
    vtypes: Dict[str, str],
    cards: Dict[str, int],
    *,
    config: OrderSMCConfig = OrderSMCConfig(),
    backend: Backend = "local",
    comm=None,
    candidate_table: OrderCandidateTable | None = None,
    verbose: bool = True,
    provenance_policy: ProvenancePolicy = "warn",
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    checkpoint_every_stages: int = 1,
):
    """Fit pair-rescue Order-SMC with one interface for local and MPI runs."""
    backend, comm = _resolve_backend(backend, comm)
    started = time.perf_counter()
    if candidate_table is None:
        build_started = time.perf_counter()
        candidate_table = build_candidates(
            data, vtypes, cards, config=config, backend=backend, comm=comm, verbose=verbose
        )
        build_seconds = time.perf_counter() - build_started
    else:
        validate_data(data, vtypes, cards, max_parents=config.max_parents)
        if list(data.columns) != candidate_table.nodes:
            raise ValueError("Candidate-table nodes do not match the data columns")
        validate_candidate_provenance(
            candidate_table,
            data=data,
            vtypes=vtypes,
            cards=cards,
            config=config,
            missing_policy=provenance_policy,
        )
        build_seconds = 0.0
    sample_started = time.perf_counter()
    result = sample(
        candidate_table,
        config=config,
        backend=backend,
        comm=comm,
        verbose=verbose,
        provenance_policy="ignore" if candidate_table is not None else provenance_policy,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=resume_checkpoint,
        checkpoint_every_stages=checkpoint_every_stages,
    )
    sample_seconds = time.perf_counter() - sample_started
    run_info = {
        "nodes": list(data.columns),
        "config": config.to_dict(),
        "candidate_build_config": (
            dict(candidate_table.provenance["candidate_build_config"])
            if getattr(candidate_table, "provenance", {}).get("candidate_build_config")
            else None
        ),
        "sampling_config": sampling_config(config),
        "candidate_provenance": dict(getattr(candidate_table, "provenance", {})),
        "backend": backend,
        "data_rows": int(data.shape[0]),
        "data_columns": int(data.shape[1]),
        "data_fingerprint": data_fingerprint(data),
        "candidate_build_seconds": build_seconds,
        "sampling_seconds": sample_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    if backend == "mpi":
        result.run_info.update(run_info)
    else:
        result.run_info.update(run_info)
    return result


def data_fingerprint(data: pd.DataFrame) -> str:
    """Stable digest of encoded values, row index, column order, and dtypes."""
    values = pd.util.hash_pandas_object(data, index=True).values
    digest = hashlib.sha256(values.tobytes())
    digest.update("\0".join(map(str, data.columns)).encode("utf-8"))
    digest.update("\0".join(map(str, data.dtypes)).encode("utf-8"))
    digest.update(f"{data.shape[0]}x{data.shape[1]}".encode("ascii"))
    return digest.hexdigest()


def candidate_table_provenance(
    data: pd.DataFrame,
    vtypes: Mapping[str, str],
    cards: Mapping[str, int],
    config: OrderSMCConfig,
) -> Dict[str, Any]:
    """Describe everything frozen into a candidate-score checkpoint."""
    levels = data.attrs.get("ordersmc_levels")
    return {
        "schema_version": PROVENANCE_VERSION,
        "ordersmc_version": __version__,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "data_fingerprint": data_fingerprint(data),
        "nodes": list(map(str, data.columns)),
        "metadata": metadata_to_mapping(vtypes, cards, levels=levels),
        "candidate_build_config": candidate_build_config(config),
        "score_implementation": _score_implementation(config),
    }


def _score_implementation(config: OrderSMCConfig) -> Dict[str, Any]:
    """Machine-readable statistical conventions behind stored local scores."""
    if config.score_method == "rank-native":
        return {
            "method": "rank_native_plugin_v1",
            "continuous_preprocessing": "Phi^-1((average_rank - 0.5) / n); training rows only",
            "standardize_transformed_continuous": config.standardize_continuous,
            "discrete_preprocessing": "unchanged encoded categories, not ranked",
            "discrete": config.discrete_score,
            "continuous": "normal_inverse_gamma on transformed observations",
            "continuous_parent_discrete_child": "normal-prior Laplace GLM on transformed predictors",
            "clg_coefficient_scale": config.tau,
            "glm_coefficient_scale": config.glm_tau,
            "two_category_prior": "log-odds N(0, glm_tau^2), identical for bin and cat(K=2)",
            "transformation_scope": "one per continuous variable, independent of candidate parents",
            "weight_interpretation": "plug-in generalized posterior; not original-scale marginal evidence",
            "marginal_transform_uncertainty_integrated": False,
            "experimental": True,
        }
    if config.score_method == "dg":
        return {
            "method": "degenerate_gaussian_bic",
            "categorical_embedding": "K-1 one-hot indicators; drop encoded level K-1",
            "continuous_preprocessing": (
                "population-standardized" if config.standardize_continuous else "original scale"
            ),
            "covariance_denominator": "n (maximum likelihood)",
            "penalty_discount": config.dg_penalty_discount,
            "ridge": config.dg_ridge,
            "singularity_check": "min_eigenvalue > eps * dimension * spectral_radius",
            "weight_interpretation": "BIC pseudo-posterior",
        }
    if config.score_method == "clg-bic":
        return {
            "method": "restricted_conditional_gaussian_bic",
            "structural_restriction": (
                "discrete children may have only discrete parents; continuous "
                "children may have discrete and continuous parents"
            ),
            "discrete": "multinomial maximum likelihood",
            "continuous": "context-specific Gaussian regression maximum likelihood",
            "continuous_preprocessing": (
                "population-standardized" if config.standardize_continuous else "original scale"
            ),
            "continuous_variance_denominator": "observations in discrete-parent context",
            "parameter_penalty": "0.5 * k * log(total observations)",
            "hyperparameter_free": True,
            "weight_interpretation": "BIC generalized-posterior weight",
        }
    return {
        "method": "native_mixed_marginal",
        "discrete": config.discrete_score,
        "continuous": "normal_inverse_gamma",
        "continuous_parent_discrete_child": "normal-prior Laplace GLM",
        "clg_coefficient_scale": config.tau,
        "glm_coefficient_scale": config.glm_tau,
        "two_category_prior": "log-odds N(0, glm_tau^2), identical for bin and cat(K=2)",
        "weight_interpretation": "mixed exact/approximate marginal evidence",
    }


def validate_candidate_provenance(
    table: OrderCandidateTable,
    *,
    data: pd.DataFrame | None = None,
    vtypes: Mapping[str, str] | None = None,
    cards: Mapping[str, int] | None = None,
    config: OrderSMCConfig | None = None,
    missing_policy: ProvenancePolicy = "warn",
) -> None:
    """Reject reuse when checkpoint inputs or score settings have changed.

    ``missing_policy`` applies only to legacy tables with no provenance. A
    populated but incompatible provenance record is always rejected.
    """
    if missing_policy not in {"require", "warn", "ignore"}:
        raise ValueError("missing_policy must be 'require', 'warn', or 'ignore'")
    provenance = getattr(table, "provenance", None)
    if not provenance:
        message = (
            "Candidate table has no provenance, so its data, metadata, and scoring "
            "configuration cannot be verified. Rebuild it, or explicitly choose "
            "provenance_policy='ignore' only if the checkpoint is known to match."
        )
        if missing_policy == "require":
            raise ValueError(message)
        if missing_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    if not isinstance(provenance, Mapping):
        raise ValueError("Candidate-table provenance must be a mapping")
    if int(provenance.get("schema_version", -1)) != PROVENANCE_VERSION:
        raise ValueError(
            f"Unsupported candidate provenance version {provenance.get('schema_version')!r}"
        )
    stored_score_schema = int(provenance.get("score_schema_version", -1))
    # v3 only changes native/rank-native scores for two-category children
    # with continuous parents. Reuse unaffected v2 tables WITHOUT relabelling
    # their provenance or changing any score. Ambiguous metadata fails closed.
    compatible_v2 = False
    if stored_score_schema == 2 and SCORE_SCHEMA_VERSION == 3:
        build = provenance.get("candidate_build_config", {})
        metadata = provenance.get("metadata", {})
        variables = metadata.get("variables", {}) if isinstance(metadata, Mapping) else {}
        if isinstance(build, Mapping) and isinstance(variables, Mapping):
            compatible_v2 = build.get("score_method") in {"dg", "clg-bic"}
            if set(variables) == set(table.nodes) and all(isinstance(v, Mapping) for v in variables.values()):
                affected = any(
                    v.get("type") == "cat" and v.get("cardinality") == 2
                    for v in variables.values()
                )
                complete_types = all(
                    v.get("type") in {"bin", "cat", "con"}
                    and (v.get("type") != "cat" or
                         isinstance(v.get("cardinality"), int) and v["cardinality"] >= 2)
                    for v in variables.values()
                )
                compatible_v2 |= build.get("score_method") in {"native", "rank-native"} and complete_types and not affected
    if stored_score_schema != SCORE_SCHEMA_VERSION and not compatible_v2:
        raise ValueError(
            "Candidate table uses an incompatible local-score implementation; rebuild it"
        )

    stored_nodes = list(provenance.get("nodes", []))
    if stored_nodes != list(table.nodes):
        raise ValueError("Candidate-table provenance nodes do not match the table")

    if config is not None:
        stored_config = provenance.get("candidate_build_config")
        expected_config = candidate_build_config(config)
        if not isinstance(stored_config, Mapping):
            raise ValueError("Candidate-table provenance has no candidate-building configuration")
        differences = [
            name
            for name in sorted(set(stored_config) | set(expected_config))
            if stored_config.get(name) != expected_config.get(name)
        ]
        if differences:
            details = ", ".join(
                f"{name}: table={stored_config.get(name)!r}, requested={expected_config.get(name)!r}"
                for name in differences
            )
            raise ValueError(
                "Candidate-table scoring/build settings do not match the requested config; "
                f"rebuild the table ({details})"
            )

    if data is not None:
        fingerprint = provenance.get("data_fingerprint")
        current = data_fingerprint(data)
        if fingerprint != current:
            raise ValueError(
                "Candidate table was built from different encoded data (fingerprint mismatch); rebuild it"
            )
    if (vtypes is None) != (cards is None):
        raise ValueError("vtypes and cards must be supplied together for provenance validation")
    if vtypes is not None and cards is not None:
        stored_metadata = provenance.get("metadata")
        if not isinstance(stored_metadata, Mapping):
            raise ValueError("Candidate-table provenance has no variable metadata")
        current_metadata = metadata_to_mapping(
            vtypes, cards, levels=data.attrs.get("ordersmc_levels") if data is not None else None
        )
        if _metadata_core(stored_metadata) != _metadata_core(current_metadata):
            raise ValueError("Candidate-table variable types/cardinalities do not match the supplied metadata")
        stored_levels = _metadata_levels(stored_metadata)
        current_levels = _metadata_levels(current_metadata)
        if stored_levels and current_levels and stored_levels != current_levels:
            raise ValueError("Candidate-table categorical level mappings do not match the supplied metadata")


def _metadata_core(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    variables = metadata.get("variables", {})
    if not isinstance(variables, Mapping):
        return {}
    return {
        str(name): {
            "type": spec.get("type"),
            "cardinality": spec.get("cardinality"),
        }
        for name, spec in variables.items()
        if isinstance(spec, Mapping)
    }


def _metadata_levels(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    variables = metadata.get("variables", {})
    if not isinstance(variables, Mapping):
        return {}
    return {
        str(name): list(spec["levels"])
        for name, spec in variables.items()
        if isinstance(spec, Mapping) and "levels" in spec
    }


def _resolve_backend(backend: Backend, comm):
    if backend not in {"auto", "local", "mpi"}:
        raise ValueError("backend must be 'auto', 'local', or 'mpi'")
    if backend == "auto":
        try:
            from mpi4py import MPI

            comm = comm or MPI.COMM_WORLD
            backend = "mpi" if comm.Get_size() > 1 else "local"
        except ImportError:
            backend = "local"
    if backend == "mpi" and comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    return backend, comm
