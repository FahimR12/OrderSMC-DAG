"""MPI-parallel candidate construction and Order-SMC sampling."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI

from .config import DAGConfig
from .core import (
    OrderCandidateTable,
    OrderSMCResult,
    OrderSMCSnapshot,
    _candidate_table_signature,
    _conditional_ess_fraction,
    _draw_proposal_name,
    _geometric_beta,
    _genealogy_summary,
    _integer_argument,
    _logsumexp,
    _normalize_logweights,
    _normalize_proposal_weights,
    _path_target_and_lookups,
    _prepare_snapshot_steps,
    _propose_order,
    _validate_tempering_path,
    _systematic_resample,
    _validate_smc_inputs,
    best_edges_and_score_for_order,
    candidate_max_indegree,
    build_order_candidate_table_parts,
    order_log_target,
    weighted_edge_summary_for_orders,
)


_PROTOCOL_SEED_INPUT = re.compile(
    r"ordersmc-protocol-v2\|(?P<master>[0-9]{20})\|"
    r"(?P<experiment>[a-z0-9][a-z0-9._/-]*)\|"
    r"(?P<cell>[a-z0-9][a-z0-9._/-]*)\|(?P<replicate>[0-9]{6})\|"
    r"(?P<component>[a-z0-9][a-z0-9._/-]*)\|(?P<attempt>[0-9]{6})\Z",
    flags=re.ASCII,
)


def _validated_parent_seed_record(
    seed: int,
    record: Optional[Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    """Validate an optional protocol seed record at the model boundary."""

    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise TypeError("parent_seed_record must be a mapping or None")
    required = {
        "input_string",
        "seed",
        "sha256",
        "bit_generator",
        "numpy_version",
    }
    if set(record) != required:
        raise ValueError(
            "parent_seed_record must contain exactly the protocol seed fields"
        )
    input_string = record["input_string"]
    if (
        not isinstance(input_string, str)
        or _PROTOCOL_SEED_INPUT.fullmatch(input_string) is None
    ):
        raise ValueError("parent_seed_record input_string is not canonical protocol v2")
    digest = hashlib.sha256(input_string.encode("utf-8")).digest()
    expected_seed = int.from_bytes(digest[:8], "big", signed=False)
    record_seed = record["seed"]
    if (
        isinstance(record_seed, bool)
        or not isinstance(record_seed, (int, np.integer))
        or int(record_seed) != expected_seed
        or int(record_seed) != int(seed)
    ):
        raise ValueError("parent_seed_record seed does not match its digest and sampler seed")
    if record["sha256"] != digest.hex():
        raise ValueError("parent_seed_record SHA-256 does not match input_string")
    if record["bit_generator"] != "PCG64DXSM":
        raise ValueError("parent_seed_record must declare PCG64DXSM")
    if record["numpy_version"] != np.__version__:
        raise ValueError("parent_seed_record NumPy version does not match this environment")
    return {
        "input_string": input_string,
        "seed": expected_seed,
        "sha256": digest.hex(),
        "bit_generator": "PCG64DXSM",
        "numpy_version": np.__version__,
    }


def _mpi_substream_record(
    parent_seed: int,
    component: str,
    index: int,
    *,
    parent_seed_record: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Derive and serialize one MPI RNG substream without arithmetic overlap.

    With a validated parent protocol record, the child remains in the frozen
    protocol-v2 namespace by extending the parent's component token.  Raw-seed
    calls retain a deterministic development fallback that is explicitly
    marked non-protocol in ``run_info``.
    """

    if component not in {"rank", "resampling-stage"}:
        raise ValueError("unsupported MPI RNG component")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("MPI RNG substream index must be an integer")
    if not 0 <= int(index) < 10**6:
        raise ValueError("MPI RNG substream index must lie in [0, 10^6)")

    if parent_seed_record is not None:
        parent_input = str(parent_seed_record["input_string"])
        match = _PROTOCOL_SEED_INPUT.fullmatch(parent_input)
        if match is None:  # guarded by _validated_parent_seed_record
            raise ValueError("parent_seed_record input_string is not canonical protocol v2")
        fields = match.groupdict()
        child_component = (
            f"{fields['component']}/mpi-{component}-{int(index):06d}"
        )
        input_string = (
            f"ordersmc-protocol-v2|{fields['master']}|{fields['experiment']}|"
            f"{fields['cell']}|{fields['replicate']}|{child_component}|"
            f"{fields['attempt']}"
        )
        schema = "ordersmc-protocol-v2"
        derivation_rule = "extend-parent-component-with-mpi-role-and-six-digit-index"
        parent_sha256: object = parent_seed_record["sha256"]
    else:
        input_string = (
            f"ordersmc-mpi-substream-v1|{int(parent_seed):020d}|{component}|"
            f"{int(index):06d}"
        )
        schema = "ordersmc-mpi-substream-v1"
        derivation_rule = "raw-seed-development-fallback"
        parent_sha256 = None

    digest = hashlib.sha256(input_string.encode("utf-8")).digest()
    return {
        "schema": schema,
        "derivation_rule": derivation_rule,
        "parent_seed": int(parent_seed),
        "parent_seed_sha256": parent_sha256,
        "component": component,
        "index": int(index),
        "input_string": input_string,
        "sha256": digest.hex(),
        "seed": int.from_bytes(digest[:8], "big", signed=False),
        "bit_generator": "PCG64DXSM",
        "numpy_version": np.__version__,
    }


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
    history: List[Dict[str, object]]
    local_orders: List[np.ndarray]
    local_logw: np.ndarray
    rank: int
    size: int
    run_info: Dict[str, object] = field(default_factory=dict)
    final_target_snapshots: List[OrderSMCSnapshot] = field(default_factory=list)
    log_normalizing_constant_estimate: float = float("nan")
    log_prior_normalizing_constant_estimate: float = float("nan")
    log_evidence_ratio_estimate: float = float("nan")

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
            final_target_snapshots=self.final_target_snapshots,
            log_normalizing_constant_estimate=self.log_normalizing_constant_estimate,
            log_prior_normalizing_constant_estimate=self.log_prior_normalizing_constant_estimate,
            log_evidence_ratio_estimate=self.log_evidence_ratio_estimate,
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
    *,
    score_current_orders: bool = True,
    best_cache: Optional[Dict[Tuple[int, ...], Tuple[List[Tuple[str, str]], float]]] = None,
    proposal_weights: Optional[Mapping[str, float]] = None,
    tempering_path: str = "collapsed_likelihood",
    proposal_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[np.ndarray, int, int, Optional[np.ndarray], float]:
    mixture = _normalize_proposal_weights(proposal_weights)
    tempering_path = _validate_tempering_path(tempering_path)
    if proposal_counts is None:
        proposal_counts = {name: {"proposed": 0, "accepted": 0} for name in mixture}
    if best_cache is None:
        best_cache = {}
    best_order: Optional[np.ndarray] = None
    best_joint_score = -math.inf
    best_tie_key = None
    if score_current_orders:
        orders_to_score = orders
    else:
        orders_to_score = []
    for order in orders_to_score:
        order_key = tuple(map(int, order))
        candidate = best_cache.get(order_key)
        if candidate is None:
            candidate = best_edges_and_score_for_order(order, table, beta=1.0)
            if len(best_cache) < 4096:
                best_cache[order_key] = candidate
        candidate_edges, candidate_score = candidate
        candidate_key = (tuple(sorted(candidate_edges)), order_key)
        if candidate_score > best_joint_score or (candidate_score == best_joint_score and (best_tie_key is None or candidate_key < best_tie_key)):
            best_order = order.copy()
            best_joint_score = candidate_score
            best_tie_key = candidate_key
    if mh_steps <= 0:
        return targets, 0, 0, best_order, best_joint_score
    accepted = 0
    proposed = 0
    target_cache: Dict[Tuple[int, ...], float] = {
        tuple(map(int, order)): float(target)
        for order, target in zip(orders, targets)
    }
    for i, order in enumerate(orders):
        target = float(targets[i])
        for _ in range(int(mh_steps)):
            proposed += 1
            proposal_name = _draw_proposal_name(rng, mixture)
            proposal_counts[proposal_name]["proposed"] += 1
            prop = _propose_order(order, rng, proposal_name)
            prop_key = tuple(map(int, prop))
            prop_target = target_cache.get(prop_key)
            if prop_target is None:
                prop_target, _ = _path_target_and_lookups(prop, table, beta, tempering_path)
                if len(target_cache) < 4096:
                    target_cache[prop_key] = prop_target
            if math.log(rng.random()) < min(0.0, prop_target - target):
                accepted += 1
                proposal_counts[proposal_name]["accepted"] += 1
                order = prop
                target = prop_target
                candidate = best_cache.get(prop_key)
                if candidate is None:
                    candidate = best_edges_and_score_for_order(order, table, beta=1.0)
                    if len(best_cache) < 4096:
                        best_cache[prop_key] = candidate
                candidate_edges, candidate_score = candidate
                candidate_key = (tuple(sorted(candidate_edges)), prop_key)
                if candidate_score > best_joint_score or (candidate_score == best_joint_score and (best_tie_key is None or candidate_key < best_tie_key)):
                    best_order = order.copy()
                    best_joint_score = candidate_score
                    best_tie_key = candidate_key
        orders[i] = order
        targets[i] = target
    return targets, accepted, proposed, best_order, best_joint_score


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
    local_build_diagnostics: Dict[str, object] = {}
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
        build_diagnostics=local_build_diagnostics,
    )
    local_elapsed = time.time() - t0
    max_elapsed = float(comm.allreduce(float(local_elapsed), op=MPI.MAX))
    gathered = comm.gather((local_parts, local_build_diagnostics), root=0)

    if is_root:
        parent_sets: List[Optional[np.ndarray]] = [None] * len(nodes)
        logliks: List[Optional[np.ndarray]] = [None] * len(nodes)
        logpriors: List[Optional[np.ndarray]] = [None] * len(nodes)
        gathered_diagnostics: List[Dict[str, object]] = []
        for parts, diagnostics in gathered:
            child_idxs, ps_list, ll_list, prior_list = parts
            gathered_diagnostics.append(diagnostics)
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
            build_diagnostics={
                    "schema": "ordersmc-candidate-build-diagnostics-v1",
                    "attempted_score_requests": int(
                        sum(int(row["attempted_score_requests"]) for row in gathered_diagnostics)
                    ),
                    "unique_family_requests": int(
                        sum(int(row["unique_family_requests"]) for row in gathered_diagnostics)
                    ),
                    "retained_family_count": int(
                        sum(int(row["retained_family_count"]) for row in gathered_diagnostics)
                    ),
                    "retained_families_by_child": {
                        child: count
                        for row in gathered_diagnostics
                        for child, count in dict(row["retained_families_by_child"]).items()
                    },
                    "fit_failures": 0,
                    "excluded_nonestimable_family_requests": int(
                        sum(
                            int(row.get("excluded_nonestimable_family_requests", 0))
                            for row in gathered_diagnostics
                        )
                    ),
                    "attempted_count_semantics": (
                        "calls to UnifiedFamilyScorer.score, including repeated requests "
                        "made by screening/rescue/final enumeration"
                    ),
            },
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
    parent_seed_record: Optional[Mapping[str, object]] = None,
    verbose: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
    gather_final: bool = True,
    proposal_weights: Optional[Mapping[str, float]] = None,
    tempering_path: str = "collapsed_likelihood",
    adaptive_cess_target: Optional[float] = None,
    adaptive_min_beta_increment: float = 1e-8,
    adaptive_max_stages: int = 1000,
    final_target_snapshot_steps: Optional[Sequence[int]] = None,
) -> OrderSMCResultMPI:
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_root = rank == 0
    tempering_path = _validate_tempering_path(tempering_path)
    mixture = _normalize_proposal_weights(proposal_weights)
    snapshot_steps = _prepare_snapshot_steps(final_target_snapshot_steps)
    adaptive_max_stages = _integer_argument("adaptive_max_stages", adaptive_max_stages, minimum=1)
    if isinstance(adaptive_min_beta_increment, (bool, np.bool_)) or not np.isfinite(adaptive_min_beta_increment) or not 0 < adaptive_min_beta_increment <= 1:
        raise ValueError("adaptive_min_beta_increment must be in (0, 1]")
    if adaptive_cess_target is not None:
        if isinstance(adaptive_cess_target, (bool, np.bool_)) or not np.isfinite(adaptive_cess_target) or not 0 < adaptive_cess_target < 1:
            raise ValueError("adaptive_cess_target must be in (0, 1)")
        if betas is not None:
            raise ValueError("betas and adaptive_cess_target are mutually exclusive")
        if tempering_path != "geometric":
            raise ValueError("MPI adaptive tempering requires tempering_path='geometric'")

    def target(order, beta):
        return _path_target_and_lookups(order, table, beta, tempering_path)[0]

    particles, betas_arr, mh_steps, mh_steps_resample, resample_ess_frac, seed = _validate_smc_inputs(
        table,
        particles,
        betas,
        mh_steps,
        mh_steps_resample,
        resample_ess_frac,
        seed,
    )
    validated_parent_seed_record = _validated_parent_seed_record(
        seed, parent_seed_record
    )

    counts, displs = _split_counts(particles, size)
    local_n = counts[rank]
    rank_seed_records = tuple(
        _mpi_substream_record(
            seed,
            "rank",
            rank_index,
            parent_seed_record=validated_parent_seed_record,
        )
        for rank_index in range(size)
    )
    resampling_seed_records = []
    rng = np.random.Generator(
        np.random.PCG64DXSM(int(rank_seed_records[rank]["seed"]))
    )
    n_nodes = len(table.nodes)

    local_orders = [rng.permutation(n_nodes).astype(np.int32) for _ in range(local_n)]
    local_targets = np.array([target(o, 0.0) for o in local_orders], dtype=np.float64)
    # Uniformly proposed orders need gamma_0 / q_0 importance weights.  The
    # uniform q_0 factor is common to all particles and can be omitted.
    local_logw = local_targets.copy()
    initial_ess = _ess_from_logw_mpi(comm, local_logw)
    initial_chunks = comm.gather(local_logw, root=0)
    log_initial_normalizer = (
        math.lgamma(n_nodes + 1) + _logsumexp(np.concatenate(initial_chunks)) - math.log(particles)
        if is_root else float("nan")
    )
    log_normalizer = log_initial_normalizer
    log_path_ratio = 0.0
    local_lineage = np.arange(displs[rank], displs[rank] + local_n, dtype=np.int64)
    history: List[Dict[str, object]] = []
    unique_roots = particles
    genealogical_ess = float(particles)
    visited_order_states = particles
    best_cache: Dict[Tuple[int, ...], Tuple[List[Tuple[str, str]], float]] = {}
    _, _, _, local_best_order, local_best_joint_score = _rejuvenate_orders(
        local_orders,
        local_targets,
        table,
        0.0,
        rng,
        0,
        best_cache=best_cache,
        proposal_weights=mixture,
        tempering_path=tempering_path,
    )

    def best_key(order):
        return (tuple(sorted(best_edges_and_score_for_order(order, table, beta=1.0)[0])), tuple(map(int, order)))

    def update_best(stage_order, stage_score):
        nonlocal local_best_order, local_best_joint_score
        if stage_order is not None and (
            stage_score > local_best_joint_score or
            (stage_score == local_best_joint_score and
             (local_best_order is None or best_key(stage_order) < best_key(local_best_order)))
        ):
            local_best_order = stage_order.copy()
            local_best_joint_score = stage_score

    def rejuvenate(steps, beta):
        nonlocal local_targets, visited_order_states
        counts_by_type = {name: {"proposed": 0, "accepted": 0} for name in mixture}
        local_targets, local_accepted, local_proposed, stage_order, stage_score = _rejuvenate_orders(
            local_orders, local_targets, table, beta, rng, steps,
            score_current_orders=False, best_cache=best_cache,
            proposal_weights=mixture, tempering_path=tempering_path,
            proposal_counts=counts_by_type,
        )
        update_best(stage_order, stage_score)
        counters = np.array([[counts_by_type[name]["proposed"], counts_by_type[name]["accepted"]] for name in mixture], dtype=np.int64)
        counters = comm.allreduce(counters, op=MPI.SUM)
        proposed, accepted = map(int, counters.sum(axis=0))
        if is_root:
            visited_order_states += accepted
            row = history[-1]
            row.update(acceptance_rate=accepted / proposed if proposed else float("nan"),
                       accepted_moves=accepted, proposed_moves=proposed,
                       unique_root_ancestors=unique_roots, genealogical_ess=genealogical_ess)
            for name, (p, a) in zip(mixture, counters):
                row[f"{name}_proposed"] = int(p)
                row[f"{name}_accepted"] = int(a)
                row[f"{name}_acceptance_rate"] = float(a / p) if p else float("nan")
        return accepted, proposed

    current_beta = 0.0
    stage_idx = 0
    while current_beta < 1.0:
        stage_idx += 1
        if adaptive_cess_target is not None:
            if stage_idx > adaptive_max_stages:
                raise RuntimeError("adaptive tempering exceeded adaptive_max_stages before beta=1")
            potentials = np.array([target(o, 1.0) for o in local_orders])
            chunks = comm.gather((local_logw, potentials), root=0)
            decision = None
            if is_root:
                try:
                    decision = (_geometric_beta(
                        np.concatenate([x[0] for x in chunks]),
                        np.concatenate([x[1] for x in chunks]), current_beta,
                        adaptive_cess_target, adaptive_min_beta_increment,
                    ), None)
                except RuntimeError as error:
                    decision = (None, str(error))
            beta, failure = comm.bcast(decision, root=0)
            if failure is not None:
                raise RuntimeError(failure)
        else:
            beta = float(betas_arr[stage_idx])
        resampling_seed_records.append(_mpi_substream_record(
            seed, "resampling-stage", stage_idx, parent_seed_record=validated_parent_seed_record
        ))
        new_targets = np.array([target(o, beta) for o in local_orders], dtype=np.float64)
        increment = new_targets - local_targets
        chunks = comm.gather((local_logw.copy(), increment), root=0)
        conditional_ess = None
        if is_root:
            old = np.concatenate([x[0] for x in chunks])
            inc = np.concatenate([x[1] for x in chunks])
            conditional_ess = _conditional_ess_fraction(old, inc)
            log_increment = _logsumexp(old + inc) - _logsumexp(old)
            log_normalizer += log_increment
            log_path_ratio += log_increment
        local_logw += increment
        local_targets = new_targets
        current_beta = float(beta)

        ess = _ess_from_logw_mpi(comm, local_logw)
        local_max_target = float(np.max(local_targets)) if local_targets.size else -math.inf
        max_target = float(comm.allreduce(local_max_target, op=MPI.MAX))
        local_sum_target = float(np.sum(local_targets)) if local_targets.size else 0.0
        sum_target = float(comm.allreduce(local_sum_target, op=MPI.SUM))
        mean_target = sum_target / max(1, int(particles))

        if is_root:
            history.append(
                {
                    "phase": "tempering",
                    "beta": beta,
                    "initial_effective_sample_size": initial_ess,
                    "conditional_ess_fraction": conditional_ess,
                    "log_normalizing_constant_increment": log_increment,
                    "log_normalizing_constant_estimate": log_normalizer,
                    "log_path_normalizer_ratio_estimate": log_path_ratio,
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
            gathered_lineage = comm.gather(local_lineage, root=0)
            if is_root:
                logw_global = np.concatenate(gathered_logw) if gathered_logw else np.zeros(0, dtype=float)
                weights = _normalize_logweights(logw_global)
                rng_res = np.random.Generator(
                    np.random.PCG64DXSM(
                        int(resampling_seed_records[stage_idx - 1]["seed"])
                    )
                )
                anc = _systematic_resample(weights, rng_res).astype(int)
                orders_global: List[np.ndarray] = []
                for chunk in gathered_orders:
                    orders_global.extend(chunk)
                lineage_global = np.concatenate(gathered_lineage)
                new_global = [orders_global[int(i)].copy() for i in anc]
                new_lineage_global = lineage_global[anc].copy()
                chunks = []
                lineage_chunks = []
                for r in range(size):
                    a = displs[r]
                    b = a + counts[r]
                    chunks.append(new_global[a:b])
                    lineage_chunks.append(new_lineage_global[a:b])
                unique_roots, genealogical_ess = _genealogy_summary(new_lineage_global)
            else:
                chunks = None
                lineage_chunks = None
            local_orders = comm.scatter(chunks, root=0)
            local_lineage = comm.scatter(lineage_chunks, root=0)
            local_n = len(local_orders)
            local_logw = np.zeros(local_n, dtype=np.float64)
            local_targets = np.array([target(o, beta) for o in local_orders], dtype=np.float64)
            if verbose and is_root:
                print("  -> resampled orders", flush=True)
            if is_root:
                history[-1]["resampled"] = 1.0

        steps = int(mh_steps_resample if did_resample else mh_steps)
        rejuvenate(steps, beta)

    snapshots = []
    completed_final_steps = 0
    for requested in snapshot_steps:
        ess = _ess_from_logw_mpi(comm, local_logw)
        if is_root:
            history.append({"phase": "final_target", "beta": 1.0, "ess": ess,
                            "resampled": 0.0, "final_target_mh_sweeps": requested})
        accepted, proposed = rejuvenate(requested - completed_final_steps, 1.0)
        completed_final_steps = requested
        chunks = comm.gather((local_orders, local_logw), root=0)
        if is_root:
            snapshot_orders = [o for chunk in chunks for o in chunk[0]]
            snapshot_weights = _normalize_logweights(np.concatenate([chunk[1] for chunk in chunks]))
            probs, mcse, snapshot_ess = weighted_edge_summary_for_orders(snapshot_orders, snapshot_weights, table)
            snapshots.append(OrderSMCSnapshot(
                beta=1.0, final_target_mh_sweeps=requested, edge_probs=probs,
                edge_mcse=mcse, effective_sample_size=snapshot_ess,
                unique_orders=len({tuple(o) for o in snapshot_orders}),
                accepted_moves=accepted, proposed_moves=proposed,
            ))

    if gather_final:
        gathered_orders = comm.gather(local_orders, root=0)
        gathered_logw = comm.gather(local_logw, root=0)
        gathered_lineage = comm.gather(local_lineage, root=0)
        gathered_best = comm.gather((local_best_joint_score, local_best_order), root=0)
        if is_root:
            orders_global: List[np.ndarray] = []
            for chunk in gathered_orders:
                orders_global.extend(chunk)
            logw_global = np.concatenate(gathered_logw) if gathered_logw else np.zeros(0, dtype=float)
            weights = _normalize_logweights(logw_global)
            valid_best = [(score, order) for score, order in gathered_best if order is not None]
            if not valid_best:
                raise RuntimeError("SMC did not visit any order")
            best_joint_score, best_order_value = min(valid_best, key=lambda item: (-item[0], best_key(item[1])))
            best_order = best_order_value.copy()
            best_edges, verified_best_score = best_edges_and_score_for_order(best_order, table, beta=1.0)
            best_joint_score = float(verified_best_score)

            edge_mass, edge_mcse, final_ess = weighted_edge_summary_for_orders(
                orders_global, weights, table, beta=1.0
            )
            lineage_global = np.concatenate(gathered_lineage)
            unique_roots, genealogical_ess = _genealogy_summary(lineage_global)
            final_unique_orders = len({tuple(map(int, order)) for order in orders_global})

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
                final_target_snapshots=snapshots,
                log_normalizing_constant_estimate=float(log_normalizer),
                log_prior_normalizing_constant_estimate=(
                    float(log_initial_normalizer) if tempering_path == "collapsed_likelihood" else float("nan")
                ),
                log_evidence_ratio_estimate=(
                    float(log_path_ratio) if tempering_path == "collapsed_likelihood" else float("nan")
                ),
                run_info={
                    "nodes": list(table.nodes),
                    "candidate_table_signature": _candidate_table_signature(table),
                    "candidate_max_indegree": candidate_max_indegree(table),
                    "mpi_ranks": size,
                    "posterior_semantics": "candidate_restricted_order_modular",
                    "target_label": "order",
                    "order_target": "order-modular marginal over retained compatible parent sets",
                    "initial_order_proposal": "uniform over permutations",
                    "tempering_path": tempering_path,
                    "initial_importance_weight": (
                        "zero: geometric bridge starts uniformly" if tempering_path == "geometric"
                        else "log_gamma_0 (uniform proposal constant omitted)"
                    ),
                    "initial_effective_sample_size": float(initial_ess),
                    "tempering_schedule": [0.0] + [row["beta"] for row in history if row["phase"] == "tempering"],
                    "adaptive_cess_target": adaptive_cess_target,
                    "resampling_method": "none" if resample_ess_frac == 0 else "systematic",
                    "resampling_ess_fraction": resample_ess_frac,
                    "proposal_mixture": dict(mixture),
                    "proposal_diagnostics": {
                        name: {
                            "proposed": sum(row[f"{name}_proposed"] for row in history),
                            "accepted": sum(row[f"{name}_accepted"] for row in history),
                        } for name in mixture
                    },
                    "log_normalizing_constant_estimate": float(log_normalizer),
                    "log_path_normalizer_ratio_estimate": float(log_path_ratio),
                    "log_path_initial_normalizer_estimate": float(log_initial_normalizer),
                    "normalizing_constant_semantics": "log Z_1 includes the d! proposal factor",
                    "evidence_ratio_semantics": (
                        "unavailable on geometric path; path ratio is log(Z_1/d!), not prior evidence"
                        if tempering_path == "geometric" else "log(Z_1/Z_0) for the structural baseline"
                    ),
                    "final_target_snapshot_steps": list(snapshot_steps),
                    "mh_step_semantics": "one order proposal per particle, not a sweep over variables",
                    "best_edges_semantics": "visited_joint_graph_order_mode_not_marginal_DAG_MAP",
                    "best_dag_selection": "highest joint log-score DAG among visited particle states",
                    "best_dag_log_score": float(best_joint_score),
                    "best_dag_score_beta": 1.0,
                    "visited_order_states": int(visited_order_states),
                    "final_unique_orders": int(final_unique_orders),
                    "final_unique_order_fraction": float(final_unique_orders / len(orders_global)),
                    "unique_root_ancestors": int(unique_roots),
                    "genealogical_ess": float(genealogical_ess),
                    "edge_mcse_calibrated": False,
                    "convergence_established": False,
                    "rng_lineage": {
                        "schema": "ordersmc-mpi-rng-lineage-v1",
                        "parent_seed": int(seed),
                        "parent_seed_record": validated_parent_seed_record,
                        "protocol_lineage_complete": (
                            validated_parent_seed_record is not None
                        ),
                        "rank_substreams": list(rank_seed_records),
                        "resampling_stage_substreams": list(resampling_seed_records),
                    },
                    "edge_mcse_method": (
                        "unbiased weighted between-order variance divided by importance-weight ESS; "
                        "heuristic because SMC genealogy and MH dependence are ignored"
                    ),
                },
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
        run_info={
            "nodes": list(table.nodes),
            "mpi_ranks": size,
            "posterior_semantics": "candidate_restricted_order_modular",
            "target_label": "order",
            "order_target": "order-modular marginal over retained compatible parent sets",
            "initial_order_proposal": "uniform over permutations",
            "tempering_path": tempering_path,
            "proposal_mixture": dict(mixture),
            "initial_effective_sample_size": float(initial_ess),
            "edge_mcse_calibrated": False,
            "convergence_established": False,
            "rng_lineage": {
                "schema": "ordersmc-mpi-rng-lineage-v1",
                "parent_seed": int(seed),
                "parent_seed_record": validated_parent_seed_record,
                "protocol_lineage_complete": validated_parent_seed_record is not None,
                "rank_substreams": list(rank_seed_records),
                "resampling_stage_substreams": list(resampling_seed_records),
            },
        },
    )
