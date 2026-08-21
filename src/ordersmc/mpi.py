"""MPI-parallel candidate construction and Order-SMC sampling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI

from .config import DAGConfig
from .core import (
    OrderCandidateTable,
    OrderSMCResult,
    _normalize_logweights,
    _propose_order,
    _systematic_resample,
    build_order_candidate_table_parts,
    best_edges_for_order,
    order_log_target,
    weighted_edge_summary_for_orders,
)


@dataclass
class OrderSMCResultMPI:
    orders: Optional[List[np.ndarray]]
    weights: Optional[np.ndarray]
    logw: Optional[np.ndarray]
    best_order: Optional[np.ndarray]
    best_edges: Optional[List[Tuple[str, str]]]
    edge_probs: Optional[Dict[Tuple[str, str], float]]
    edge_mcse: Optional[Dict[Tuple[str, str], float]]
    effective_sample_size: Optional[float]
    history: List[Dict[str, float]]
    local_orders: List[np.ndarray]
    local_logw: np.ndarray
    rank: int
    size: int
    run_info: Dict[str, object] = field(default_factory=dict)

    def to_local_result(self) -> OrderSMCResult:
        """Convert the root result to the common serializable result type."""
        if any(value is None for value in (self.orders, self.weights, self.logw, self.best_order, self.best_edges, self.edge_probs)):
            raise RuntimeError("Only the MPI root rank has a complete result")
        return OrderSMCResult(
            orders=self.orders,
            weights=self.weights,
            logw=self.logw,
            best_order=self.best_order,
            best_edges=self.best_edges,
            edge_probs=self.edge_probs,
            history=self.history,
            edge_mcse=self.edge_mcse or {},
            effective_sample_size=float(self.effective_sample_size or 0.0),
            run_info=self.run_info,
        )

    def save(self, path, *, threshold: float = 0.7) -> None:
        self.to_local_result().save(path, threshold=threshold)


def _split_counts(n: int, size: int) -> Tuple[List[int], List[int]]:
    counts = [n // size + (1 if r < (n % size) else 0) for r in range(size)]
    displs = [0]
    for c in counts[:-1]:
        displs.append(displs[-1] + c)
    return counts, displs


def _ess_from_logw_mpi(comm: MPI.Comm, local_logw: np.ndarray) -> float:
    local_logw = np.asarray(local_logw, dtype=float)
    local_max = float(np.max(local_logw)) if local_logw.size else -math.inf
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    if not np.isfinite(global_max):
        return float("nan")

    local_sum = float(np.sum(np.exp(local_logw - global_max))) if local_logw.size else 0.0
    global_sum = float(comm.allreduce(local_sum, op=MPI.SUM))
    logz = global_max + math.log(max(global_sum, 1e-300))

    local_max2 = float(np.max(2.0 * local_logw)) if local_logw.size else -math.inf
    global_max2 = float(comm.allreduce(local_max2, op=MPI.MAX))
    local_sum2 = float(np.sum(np.exp(2.0 * local_logw - global_max2))) if local_logw.size else 0.0
    global_sum2 = float(comm.allreduce(local_sum2, op=MPI.SUM))
    logz2 = global_max2 + math.log(max(global_sum2, 1e-300))
    return float(math.exp(2.0 * logz - logz2))


def _rejuvenate_orders(
    orders: List[np.ndarray],
    targets: np.ndarray,
    table: OrderCandidateTable,
    beta: float,
    rng: np.random.Generator,
    mh_steps: int,
) -> Tuple[np.ndarray, int, int]:
    if mh_steps <= 0:
        return targets, 0, 0
    accepted = 0
    proposed = 0
    for i, order in enumerate(orders):
        target = float(targets[i])
        for _ in range(int(mh_steps)):
            proposed += 1
            prop = _propose_order(order, rng)
            prop_target = order_log_target(prop, table, beta)
            if math.log(rng.random()) < min(0.0, prop_target - target):
                accepted += 1
                order = prop
                target = prop_target
        orders[i] = order
        targets[i] = target
    return targets, accepted, proposed


def build_order_candidate_table_mpi(
    scorer,
    nodes: Sequence[str],
    cfg: DAGConfig,
    *,
    screen_k: int = 20,
    max_parents: int = 2,
    keep_per_child: Optional[int] = None,
    pair_rescue_anchor_k: int = 0,
    pair_rescue_k: int = 0,
    parent_pool_max: Optional[int] = None,
    triplet_rescue_pair_k: int = 0,
    triplet_rescue_k: int = 0,
    tail_edge_penalty_after: Optional[int] = None,
    tail_edge_penalty: Optional[float] = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    verbose: bool = True,
) -> OrderCandidateTable:
    """
    Build the order-SMC candidate table in parallel across children.

    Each child family table is independent, so ranks build disjoint child-index
    subsets and root assembles the final table before broadcasting it to all
    ranks for the SMC stage.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_root = rank == 0
    nodes = list(nodes)
    local_child_indices = list(range(rank, len(nodes), size))

    t0 = time.time()
    local_parts = build_order_candidate_table_parts(
        scorer,
        nodes,
        cfg,
        local_child_indices,
        screen_k=screen_k,
        max_parents=max_parents,
        keep_per_child=keep_per_child,
        pair_rescue_anchor_k=pair_rescue_anchor_k,
        pair_rescue_k=pair_rescue_k,
        parent_pool_max=parent_pool_max,
        triplet_rescue_pair_k=triplet_rescue_pair_k,
        triplet_rescue_k=triplet_rescue_k,
        tail_edge_penalty_after=tail_edge_penalty_after,
        tail_edge_penalty=tail_edge_penalty,
    )
    local_elapsed = time.time() - t0
    max_elapsed = float(comm.allreduce(float(local_elapsed), op=MPI.MAX))
    gathered = comm.gather(local_parts, root=0)

    if is_root:
        parent_sets: List[Optional[np.ndarray]] = [None] * len(nodes)
        logliks: List[Optional[np.ndarray]] = [None] * len(nodes)
        logpriors: List[Optional[np.ndarray]] = [None] * len(nodes)
        for child_idxs, ps_list, ll_list, prior_list in gathered:
            for child_idx, ps, ll, prior in zip(child_idxs, ps_list, ll_list, prior_list):
                parent_sets[int(child_idx)] = ps
                logliks[int(child_idx)] = ll
                logpriors[int(child_idx)] = prior

        missing = [i for i, ps in enumerate(parent_sets) if ps is None]
        if missing:
            raise RuntimeError(f"Missing order candidate entries for child indices: {missing[:10]}")

        table = OrderCandidateTable(
            nodes=nodes,
            parent_sets=[ps for ps in parent_sets if ps is not None],
            loglik=[ll for ll in logliks if ll is not None],
            logprior=[prior for prior in logpriors if prior is not None],
        )
        if verbose:
            counts = [len(x) for x in table.loglik]
            print(
                "[order-smc-mpi] parallel candidate table ready "
                f"ranks={size} children={len(counts)} min={min(counts)} "
                f"median={float(np.median(counts)):.1f} max={max(counts)} "
                f"build_sec_max={max_elapsed:.2f}",
                flush=True,
            )
    else:
        table = None

    return comm.bcast(table, root=0)


def order_smc_mpi(
    table: OrderCandidateTable,
    *,
    particles: int = 200,
    betas: Optional[Sequence[float]] = None,
    mh_steps: int = 20,
    mh_steps_resample: Optional[int] = None,
    resample_ess_frac: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
    gather_final: bool = True,
) -> OrderSMCResultMPI:
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_root = rank == 0

    counts, displs = _split_counts(int(particles), size)
    local_n = counts[rank]
    rng = np.random.default_rng(int(seed) + 1000003 * rank)
    n_nodes = len(table.nodes)

    if betas is None:
        betas = np.linspace(0.0, 1.0, 100)
    betas_arr = np.asarray(list(betas), dtype=float)
    if betas_arr.size == 0:
        betas_arr = np.array([0.0, 1.0], dtype=float)
    if betas_arr[0] != 0.0:
        betas_arr = np.concatenate([[0.0], betas_arr])
    if betas_arr[-1] != 1.0:
        betas_arr = np.concatenate([betas_arr, [1.0]])

    local_orders = [rng.permutation(n_nodes).astype(np.int32) for _ in range(local_n)]
    local_logw = np.zeros(local_n, dtype=np.float64)
    local_targets = np.array([order_log_target(o, table, 0.0) for o in local_orders], dtype=np.float64)
    history: List[Dict[str, float]] = []

    if mh_steps_resample is None:
        mh_steps_resample = int(mh_steps)

    for stage_idx, beta in enumerate(betas_arr[1:], start=1):
        beta = float(beta)
        new_targets = np.array([order_log_target(o, table, beta) for o in local_orders], dtype=np.float64)
        local_logw += new_targets - local_targets
        local_targets = new_targets

        ess = _ess_from_logw_mpi(comm, local_logw)
        local_max_target = float(np.max(local_targets)) if local_targets.size else -math.inf
        max_target = float(comm.allreduce(local_max_target, op=MPI.MAX))
        local_sum_target = float(np.sum(local_targets)) if local_targets.size else 0.0
        sum_target = float(comm.allreduce(local_sum_target, op=MPI.SUM))
        mean_target = sum_target / max(1, int(particles))

        if is_root:
            history.append(
                {
                    "beta": beta,
                    "ess": float(ess),
                    "max_log_target": max_target,
                    "mean_log_target": float(mean_target),
                    "resampled": 0.0,
                }
            )
        if verbose and is_root:
            print(
                f"[order-smc-mpi beta={beta:.4f}] ESS={ess:.1f}/{particles} "
                f"max_log_target={max_target:.3f}",
                flush=True,
            )

        did_resample = bool(ess < float(resample_ess_frac) * int(particles))
        did_resample = bool(comm.bcast(did_resample, root=0))
        if did_resample:
            gathered_logw = comm.gather(local_logw, root=0)
            gathered_orders = comm.gather(local_orders, root=0)
            if is_root:
                logw_global = np.concatenate(gathered_logw) if gathered_logw else np.zeros(0, dtype=float)
                weights = _normalize_logweights(logw_global)
                rng_res = np.random.default_rng(int(seed) + 99991 * stage_idx)
                anc = _systematic_resample(weights, rng_res).astype(int)
                orders_global: List[np.ndarray] = []
                for chunk in gathered_orders:
                    orders_global.extend(chunk)
                new_global = [orders_global[int(i)].copy() for i in anc]
                chunks = []
                for r in range(size):
                    a = displs[r]
                    b = a + counts[r]
                    chunks.append(new_global[a:b])
            else:
                chunks = None
            local_orders = comm.scatter(chunks, root=0)
            local_n = len(local_orders)
            local_logw = np.zeros(local_n, dtype=np.float64)
            local_targets = np.array([order_log_target(o, table, beta) for o in local_orders], dtype=np.float64)
            if verbose and is_root:
                print("  -> resampled orders", flush=True)
            if is_root:
                history[-1]["resampled"] = 1.0

        steps = int(mh_steps_resample if did_resample else mh_steps)
        local_targets, local_accepted, local_proposed = _rejuvenate_orders(
            local_orders, local_targets, table, beta, rng, steps
        )
        accepted = int(comm.allreduce(local_accepted, op=MPI.SUM))
        proposed = int(comm.allreduce(local_proposed, op=MPI.SUM))
        if is_root:
            history[-1]["acceptance_rate"] = float(accepted / proposed) if proposed else float("nan")
            history[-1]["accepted_moves"] = float(accepted)
            history[-1]["proposed_moves"] = float(proposed)

    if gather_final:
        gathered_orders = comm.gather(local_orders, root=0)
        gathered_logw = comm.gather(local_logw, root=0)
        if is_root:
            orders_global: List[np.ndarray] = []
            for chunk in gathered_orders:
                orders_global.extend(chunk)
            logw_global = np.concatenate(gathered_logw) if gathered_logw else np.zeros(0, dtype=float)
            weights = _normalize_logweights(logw_global)
            final_targets = np.array([order_log_target(o, table, 1.0) for o in orders_global], dtype=float)
            best_idx = int(np.argmax(final_targets)) if len(final_targets) else 0
            best_order = orders_global[best_idx].copy()
            best_edges = best_edges_for_order(best_order, table, beta=1.0)

            edge_mass, edge_mcse, final_ess = weighted_edge_summary_for_orders(
                orders_global, weights, table, beta=1.0
            )

            return OrderSMCResultMPI(
                orders=orders_global,
                weights=weights,
                logw=logw_global,
                best_order=best_order,
                best_edges=best_edges,
                edge_probs=edge_mass,
                edge_mcse=edge_mcse,
                effective_sample_size=final_ess,
                history=history,
                local_orders=local_orders,
                local_logw=local_logw,
                rank=rank,
                size=size,
                run_info={"nodes": list(table.nodes), "mpi_ranks": size},
            )

    return OrderSMCResultMPI(
        orders=None,
        weights=None,
        logw=None,
        best_order=None,
        best_edges=None,
        edge_probs=None,
        edge_mcse=None,
        effective_sample_size=None,
        history=history,
        local_orders=local_orders,
        local_logw=local_logw,
        rank=rank,
        size=size,
        run_info={"nodes": list(table.nodes), "mpi_ranks": size},
    )
