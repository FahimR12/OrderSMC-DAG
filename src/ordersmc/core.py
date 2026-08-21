"""Order-based sequential Monte Carlo for Bayesian DAG structure learning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scoring import UnifiedFamilyScorer
from .config import DAGConfig


@dataclass
class OrderCandidateTable:
    nodes: List[str]
    parent_sets: List[np.ndarray]
    loglik: List[np.ndarray]
    logprior: List[np.ndarray]


@dataclass
class OrderSMCResult:
    orders: List[np.ndarray]
    weights: np.ndarray
    logw: np.ndarray
    best_order: np.ndarray
    best_edges: List[Tuple[str, str]]
    edge_probs: Dict[Tuple[str, str], float]
    history: List[Dict[str, float]]
    edge_mcse: Dict[Tuple[str, str], float] = field(default_factory=dict)
    effective_sample_size: float = 0.0
    run_info: Dict[str, object] = field(default_factory=dict)

    def best_dag(self):
        """Return the best-order graph as a NetworkX DAG."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(self.best_order_names())
        graph.add_edges_from(self.best_edges)
        return graph

    def best_order_names(self) -> List[str]:
        nodes = self.run_info.get("nodes", [])
        return [str(nodes[int(i)]) for i in self.best_order] if nodes else []

    def posterior_dag(self, threshold: float = 0.7):
        from .graphs import dag_from_edge_probs

        nodes = self.run_info.get("nodes")
        if not nodes:
            nodes = sorted({node for edge in self.edge_probs for node in edge})
        return dag_from_edge_probs(self.edge_probs, nodes, threshold=threshold)

    def edge_table(self):
        from .confidence import edge_confidence_table

        nodes = self.run_info.get("nodes")
        if not nodes:
            nodes = sorted({node for edge in self.edge_probs for node in edge})
        return edge_confidence_table(
            self.edge_probs,
            nodes,
            edge_mcse=self.edge_mcse,
            effective_sample_size=self.effective_sample_size,
        )

    def summary(self) -> Dict[str, object]:
        min_ess = min((float(row["ess"]) for row in self.history), default=self.effective_sample_size)
        acceptance = [
            float(row["acceptance_rate"])
            for row in self.history
            if "acceptance_rate" in row and np.isfinite(float(row["acceptance_rate"]))
        ]
        mean_acceptance = float(np.mean(acceptance)) if acceptance else float("nan")
        warnings = []
        if self.orders and min_ess < 0.1 * len(self.orders):
            warnings.append("Stage ESS fell below 10% of particles; consider more beta stages or particles.")
        if acceptance and mean_acceptance < 0.05:
            warnings.append("Mean MH acceptance was below 5%; order proposals may be mixing poorly.")
        if self.orders and self.effective_sample_size < 0.25 * len(self.orders):
            warnings.append("Final particle ESS was below 25%; posterior MC uncertainty may be understated.")
        return {
            "particles": len(self.orders),
            "best_edges": len(self.best_edges),
            "posterior_edges_p50": int(sum(p >= 0.5 for p in self.edge_probs.values())),
            "effective_sample_size": float(self.effective_sample_size),
            "minimum_stage_ess": float(min_ess),
            "mean_acceptance_rate": mean_acceptance if np.isfinite(mean_acceptance) else None,
            "warnings": warnings,
        }

    def save(self, path, *, threshold: float = 0.7) -> None:
        from .persistence import save_result

        save_result(self, path, threshold=threshold)


def _logsumexp(x: np.ndarray) -> float:
    if x.size == 0:
        return -math.inf
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def _normalize_logweights(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, dtype=float)
    m = float(np.max(logw))
    w = np.exp(logw - m)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.full(logw.shape, 1.0 / max(1, logw.size), dtype=float)
    return w / s


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), positions)


def build_order_candidate_table_parts(
    scorer: UnifiedFamilyScorer,
    nodes: Sequence[str],
    cfg: DAGConfig,
    child_indices: Sequence[int],
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
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Build order-SMC candidate-table entries for a subset of children."""
    nodes = list(nodes)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    child_indices = [int(i) for i in child_indices]
    screen_k = max(1, int(screen_k))
    max_parents = max(0, int(max_parents))
    pair_rescue_anchor_k = max(0, int(pair_rescue_anchor_k))
    pair_rescue_k = max(0, int(pair_rescue_k))
    triplet_rescue_pair_k = max(0, int(triplet_rescue_pair_k))
    triplet_rescue_k = max(0, int(triplet_rescue_k))
    if tail_edge_penalty_after is not None:
        tail_edge_penalty_after = max(0, int(tail_edge_penalty_after))
    if tail_edge_penalty is not None:
        tail_edge_penalty = float(tail_edge_penalty)
    if keep_per_child is not None:
        keep_per_child = max(1, int(keep_per_child))
    if parent_pool_max is not None:
        parent_pool_max = max(screen_k, int(parent_pool_max))

    parent_sets: List[np.ndarray] = []
    logliks: List[np.ndarray] = []
    logpriors: List[np.ndarray] = []

    def family_terms(child: str, parents: Tuple[str, ...]) -> Tuple[float, float]:
        res = scorer.score(child, parents)
        n_parents = len(parents)
        if tail_edge_penalty_after is None or tail_edge_penalty is None:
            edge_cost = cfg.edge_penalty * n_parents
        else:
            base_edges = min(n_parents, tail_edge_penalty_after)
            tail_edges = max(0, n_parents - tail_edge_penalty_after)
            edge_cost = cfg.edge_penalty * base_edges + tail_edge_penalty * tail_edges
        prior = -cfg.lam * int(res.k_params) - edge_cost
        return float(res.log_marginal), float(prior)

    for child_idx in child_indices:
        child = nodes[child_idx]
        empty_ll, empty_prior = family_terms(child, tuple())
        empty_obj = empty_ll + empty_prior

        single_scores: List[Tuple[float, str]] = []
        single_obj: Dict[str, float] = {}
        for p in nodes:
            if p == child:
                continue
            ll, prior = family_terms(child, (p,))
            obj = ll + prior
            single_scores.append((obj - empty_obj, p))
            single_obj[p] = obj

        single_ranked = sorted(single_scores, reverse=True)
        screened = [p for _, p in single_ranked[:screen_k]]

        if max_parents >= 2 and pair_rescue_anchor_k > 0 and pair_rescue_k > 0:
            anchors = [p for _, p in single_ranked[:pair_rescue_anchor_k]]
            seen_pairs: set[Tuple[str, str]] = set()
            pair_scores: List[Tuple[float, float, Tuple[str, str]]] = []
            for anchor in anchors:
                for p in nodes:
                    if p == child or p == anchor:
                        continue
                    pair = tuple(sorted((anchor, p)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    ll, prior = family_terms(child, pair)
                    pair_obj = ll + prior
                    gain_over_components = pair_obj - max(empty_obj, single_obj[anchor], single_obj[p])
                    if gain_over_components > 0.0:
                        pair_scores.append((gain_over_components, pair_obj, pair))

            pool_limit = parent_pool_max if parent_pool_max is not None else len(nodes) - 1
            screened_set = set(screened)
            for _, _, pair in sorted(pair_scores, reverse=True)[:pair_rescue_k]:
                for p in pair:
                    if p not in screened_set and len(screened) < pool_limit:
                        screened.append(p)
                        screened_set.add(p)
                if len(screened) >= pool_limit:
                    break

        if max_parents >= 3 and triplet_rescue_pair_k > 0 and triplet_rescue_k > 0:
            pool_limit = parent_pool_max if parent_pool_max is not None else len(nodes) - 1
            screened_set = set(screened)
            anchor_pairs: List[Tuple[float, Tuple[str, str]]] = []
            for pair in combinations(screened, 2):
                pair = tuple(sorted(pair))
                ll, prior = family_terms(child, pair)
                anchor_pairs.append((ll + prior, pair))

            triplet_scores: List[Tuple[float, float, Tuple[str, str, str]]] = []
            seen_triplets: set[Tuple[str, str, str]] = set()
            for pair_obj, pair in sorted(anchor_pairs, reverse=True)[:triplet_rescue_pair_k]:
                for p in nodes:
                    if p == child or p in pair:
                        continue
                    triplet = tuple(sorted((pair[0], pair[1], p)))
                    if triplet in seen_triplets:
                        continue
                    seen_triplets.add(triplet)
                    ll, prior = family_terms(child, triplet)
                    triplet_obj = ll + prior
                    gain_over_pair = triplet_obj - pair_obj
                    if gain_over_pair > 0.0:
                        triplet_scores.append((gain_over_pair, triplet_obj, triplet))

            for _, _, triplet in sorted(triplet_scores, reverse=True)[:triplet_rescue_k]:
                for p in triplet:
                    if p not in screened_set and len(screened) < pool_limit:
                        screened.append(p)
                        screened_set.add(p)
                if len(screened) >= pool_limit:
                    break

        scored: Dict[Tuple[str, ...], Tuple[float, float]] = {tuple(): (empty_ll, empty_prior)}
        for r in range(1, max_parents + 1):
            for ps in combinations(screened, r):
                ps = tuple(sorted(ps))
                scored[ps] = family_terms(child, ps)

        ranked = sorted(scored.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)
        if keep_per_child is not None:
            ranked = ranked[:keep_per_child]
            if tuple() not in [ps for ps, _ in ranked]:
                ranked.append((tuple(), (empty_ll, empty_prior)))

        width = max(1, max((len(ps) for ps, _ in ranked), default=0))
        ps_arr = np.full((len(ranked), width), -1, dtype=np.int32)
        ll_arr = np.empty(len(ranked), dtype=np.float64)
        prior_arr = np.empty(len(ranked), dtype=np.float64)
        for i, (ps, (ll, prior)) in enumerate(ranked):
            for j, p in enumerate(ps):
                ps_arr[i, j] = node_to_idx[p]
            ll_arr[i] = ll
            prior_arr[i] = prior

        parent_sets.append(ps_arr)
        logliks.append(ll_arr)
        logpriors.append(prior_arr)

    return child_indices, parent_sets, logliks, logpriors


def build_order_candidate_table(
    scorer: UnifiedFamilyScorer,
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
) -> OrderCandidateTable:
    """
    Precompute screened parent sets for order-SMC.

    Parent candidates are screened by single-parent local objective gain. Parent
    sets up to max_parents are then enumerated from the screened pool. Scores are
    stored as data log-likelihood and structure-prior terms separately so SMC can
    temper only the data term.

    If pair_rescue_* is enabled, high-scoring two-parent families can add parents
    that look weak alone. This is meant for v-structures/interactions where a
    single-parent screen drops a true parent before order-SMC ever sees it.

    If triplet_rescue_* is enabled, high-scoring two-parent anchors can rescue a
    third candidate parent. tail_edge_penalty_* optionally makes the edge penalty
    non-linear, for example keeping the first two parents conservative while
    using a softer penalty for parent 3 and beyond.
    """
    nodes = list(nodes)
    child_indices, parent_sets, logliks, logpriors = build_order_candidate_table_parts(
        scorer,
        nodes,
        cfg,
        range(len(nodes)),
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
    assert child_indices == list(range(len(nodes)))
    return OrderCandidateTable(nodes=nodes, parent_sets=parent_sets, loglik=logliks, logprior=logpriors)


def order_log_target(order: np.ndarray, table: OrderCandidateTable, beta: float) -> float:
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    total = 0.0
    for child_idx in range(n):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        vals = table.logprior[child_idx][ok] + float(beta) * table.loglik[child_idx][ok]
        total += _logsumexp(vals)
    return float(total)


def best_edges_for_order(order: np.ndarray, table: OrderCandidateTable, beta: float = 1.0) -> List[Tuple[str, str]]:
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    edges: List[Tuple[str, str]] = []
    for child_idx, child in enumerate(table.nodes):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        idxs = np.flatnonzero(ok)
        vals = table.logprior[child_idx][idxs] + float(beta) * table.loglik[child_idx][idxs]
        best = int(idxs[int(np.argmax(vals))])
        for pidx in ps[best]:
            if pidx >= 0:
                edges.append((table.nodes[int(pidx)], child))
    return sorted(edges)


def sample_edges_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    rng: np.random.Generator,
    beta: float = 1.0,
) -> List[Tuple[str, str]]:
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    edges: List[Tuple[str, str]] = []
    for child_idx, child in enumerate(table.nodes):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        idxs = np.flatnonzero(ok)
        vals = table.logprior[child_idx][idxs] + float(beta) * table.loglik[child_idx][idxs]
        m = float(np.max(vals))
        probs = np.exp(vals - m)
        probs = probs / float(np.sum(probs))
        chosen = int(idxs[int(rng.choice(len(idxs), p=probs))])
        for pidx in ps[chosen]:
            if pidx >= 0:
                edges.append((table.nodes[int(pidx)], child))
    return sorted(edges)


def edge_probs_for_order(
    order: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Exact conditional edge probabilities given an ordering.

    Parent-set choices are conditionally independent by child once the order is
    fixed, so we can analytically average over every compatible parent set
    instead of sampling one graph from the order.
    """
    n = len(table.nodes)
    pos = np.empty(n, dtype=np.int32)
    pos[order] = np.arange(n, dtype=np.int32)
    edge_mass: Dict[Tuple[str, str], float] = {}
    for child_idx, child in enumerate(table.nodes):
        ps = table.parent_sets[child_idx]
        child_pos = pos[child_idx]
        ok = np.all((ps < 0) | (pos[np.maximum(ps, 0)] < child_pos), axis=1)
        idxs = np.flatnonzero(ok)
        vals = table.logprior[child_idx][idxs] + float(beta) * table.loglik[child_idx][idxs]
        m = float(np.max(vals))
        probs = np.exp(vals - m)
        probs = probs / float(np.sum(probs))
        for idx, prob in zip(idxs, probs):
            if prob <= 0.0:
                continue
            for pidx in ps[int(idx)]:
                if pidx >= 0:
                    edge = (table.nodes[int(pidx)], child)
                    edge_mass[edge] = edge_mass.get(edge, 0.0) + float(prob)
    return edge_mass


def weighted_edge_probs_for_orders(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    edge_mass, _, _ = weighted_edge_summary_for_orders(orders, weights, table, beta=beta)
    return edge_mass


def weighted_edge_summary_for_orders(
    orders: Sequence[np.ndarray],
    weights: np.ndarray,
    table: OrderCandidateTable,
    beta: float = 1.0,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], float]:
    """Return Rao-Blackwellized edge means and approximate particle MCSEs."""
    edge_mass: Dict[Tuple[str, str], float] = {}
    edge_second_moment: Dict[Tuple[str, str], float] = {}
    for order, w in zip(orders, weights):
        if float(w) <= 0.0:
            continue
        for edge, prob in edge_probs_for_order(order, table, beta=beta).items():
            edge_mass[edge] = edge_mass.get(edge, 0.0) + float(w) * prob
            edge_second_moment[edge] = edge_second_moment.get(edge, 0.0) + float(w) * prob * prob
    weight_array = np.asarray(weights, dtype=float)
    effective_sample_size = float(1.0 / np.sum(weight_array**2)) if weight_array.size else 0.0
    edge_mcse = {
        edge: math.sqrt(max(0.0, edge_second_moment[edge] - mean * mean) / max(1.0, effective_sample_size))
        for edge, mean in edge_mass.items()
    }
    return edge_mass, edge_mcse, effective_sample_size


def _propose_order(order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = order.copy()
    n = len(new)
    if n < 2:
        return new
    if rng.random() < 0.75:
        i, j = rng.choice(n, size=2, replace=False)
        new[i], new[j] = new[j], new[i]
    else:
        i = int(rng.integers(0, n - 1))
        new[i], new[i + 1] = new[i + 1], new[i]
    return new


def order_smc(
    table: OrderCandidateTable,
    *,
    particles: int = 200,
    betas: Optional[Sequence[float]] = None,
    mh_steps: int = 20,
    mh_steps_resample: Optional[int] = None,
    resample_ess_frac: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> OrderSMCResult:
    if int(particles) < 2:
        raise ValueError("particles must be at least 2")
    rng = np.random.default_rng(seed)
    n = len(table.nodes)
    if betas is None:
        betas = np.linspace(0.0, 1.0, 100)
    betas_arr = np.asarray(list(betas), dtype=float)
    if betas_arr[0] != 0.0:
        betas_arr = np.concatenate([[0.0], betas_arr])
    if betas_arr[-1] != 1.0:
        betas_arr = np.concatenate([betas_arr, [1.0]])

    orders = [rng.permutation(n).astype(np.int32) for _ in range(int(particles))]
    logw = np.zeros(len(orders), dtype=np.float64)
    cur_targets = np.array([order_log_target(o, table, 0.0) for o in orders], dtype=np.float64)
    history: List[Dict[str, float]] = []
    if mh_steps_resample is None:
        mh_steps_resample = int(mh_steps)

    for beta in betas_arr[1:]:
        beta = float(beta)
        new_targets = np.array([order_log_target(o, table, beta) for o in orders], dtype=np.float64)
        logw += new_targets - cur_targets
        cur_targets = new_targets

        weights = _normalize_logweights(logw)
        ess = float(1.0 / np.sum(weights**2))
        best_idx = int(np.argmax(cur_targets))
        history.append(
            {
                "beta": beta,
                "ess": ess,
                "max_log_target": float(cur_targets[best_idx]),
                "mean_log_target": float(np.mean(cur_targets)),
                "resampled": 0.0,
            }
        )
        if verbose:
            print(
                f"[order-smc beta={beta:.4f}] ESS={ess:.1f}/{len(orders)} "
                f"max_log_target={cur_targets[best_idx]:.3f}",
                flush=True,
            )

        did_resample = ess < float(resample_ess_frac) * len(orders)
        if did_resample:
            anc = _systematic_resample(weights, rng)
            orders = [orders[int(i)].copy() for i in anc]
            cur_targets = cur_targets[anc].copy()
            logw = np.zeros(len(orders), dtype=np.float64)
            history[-1]["resampled"] = 1.0
            if verbose:
                print("  -> resampled orders", flush=True)

        steps = int(mh_steps_resample if did_resample else mh_steps)
        accepted = 0
        proposed = 0
        for i, order in enumerate(orders):
            target = float(cur_targets[i])
            for _ in range(steps):
                proposed += 1
                prop = _propose_order(order, rng)
                prop_target = order_log_target(prop, table, beta)
                if math.log(rng.random()) < min(0.0, prop_target - target):
                    accepted += 1
                    order = prop
                    target = prop_target
            orders[i] = order
            cur_targets[i] = target
        history[-1]["acceptance_rate"] = float(accepted / proposed) if proposed else float("nan")
        history[-1]["accepted_moves"] = float(accepted)
        history[-1]["proposed_moves"] = float(proposed)

    weights = _normalize_logweights(logw)
    best_idx = int(np.argmax(cur_targets))
    best_order = orders[best_idx].copy()
    best_edges = best_edges_for_order(best_order, table, beta=1.0)

    edge_mass, edge_mcse, final_ess = weighted_edge_summary_for_orders(orders, weights, table, beta=1.0)

    return OrderSMCResult(
        orders=orders,
        weights=weights,
        logw=logw,
        best_order=best_order,
        best_edges=best_edges,
        edge_probs=edge_mass,
        history=history,
        edge_mcse=edge_mcse,
        effective_sample_size=final_ess,
        run_info={"nodes": list(table.nodes)},
    )
