"""Validated high-level local and MPI APIs."""

from __future__ import annotations

import hashlib
import time
from typing import Dict, Literal

import numpy as np
import pandas as pd

from .config import DAGConfig, OrderSMCConfig
from .core import OrderCandidateTable, OrderSMCResult, build_order_candidate_table, order_smc
from .data import validate_data
from .scoring import UnifiedFamilyScorer

Backend = Literal["auto", "local", "mpi"]


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
        use_sparse_bdeu=True,
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

        return build_order_candidate_table_mpi(
            scorer, list(data.columns), make_prior(config), comm=comm, verbose=verbose, **kwargs
        )
    return build_order_candidate_table(scorer, list(data.columns), make_prior(config), **kwargs)


def sample(
    table: OrderCandidateTable,
    *,
    config: OrderSMCConfig = OrderSMCConfig(),
    backend: Backend = "local",
    comm=None,
    verbose: bool = True,
):
    """Run SMC from a new or restored candidate table."""
    backend, comm = _resolve_backend(backend, comm)
    kwargs = dict(
        particles=config.particles,
        betas=np.linspace(0.0, 1.0, config.num_betas),
        mh_steps=config.mh_steps,
        mh_steps_resample=config.mh_steps_resample,
        resample_ess_frac=config.resample_ess_frac,
        seed=config.seed,
        verbose=verbose,
    )
    if backend == "mpi":
        from .mpi import order_smc_mpi

        return order_smc_mpi(table, comm=comm, gather_final=True, **kwargs)
    return order_smc(table, **kwargs)


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
        build_seconds = 0.0
    sample_started = time.perf_counter()
    result = sample(candidate_table, config=config, backend=backend, comm=comm, verbose=verbose)
    sample_seconds = time.perf_counter() - sample_started
    run_info = {
        "nodes": list(data.columns),
        "config": config.to_dict(),
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
    values = pd.util.hash_pandas_object(data, index=True).values
    digest = hashlib.sha256(values.tobytes())
    digest.update("\0".join(map(str, data.columns)).encode("utf-8"))
    return digest.hexdigest()


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
