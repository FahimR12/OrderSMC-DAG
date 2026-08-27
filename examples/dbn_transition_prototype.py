"""
Path A proof-of-concept: inter-slice DBN via OrderSMC-DAG's EXACT fixed-order posterior.

Goal of this script (not production code -- a readable demonstration):
  1. Show the *transition-table construction*: reshape a kinase-activity time-course
     into lagged (t -> t+1) rows with a two-slice node set {X__t} (sources) and
     {X__t1} (targets).
  2. Show the *parent-mask*: each target __t1 node may only take parents from the
     source slice __t. Sources take no parents (exogenous inputs).
  3. Fix the temporal order (all __t before all __t1) and call the package's
     analytic Rao-Blackwellised posterior `edge_probs_for_order` -- ONE call,
     deterministic, no ordering search, no seed variance.
  4. Contrast: run the stock UNCONSTRAINED order-SMC on the same frame across
     seeds -> the answer wanders (this is the egfr-style instability).

Uses synthetic VAR(1) data with a *known* planted transition chain so we can see
whether the exact path recovers it: A->B, B->C, C-|D (negative), E = noise.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

# networkx is a package dependency but only used inside graphs.dag_from_edge_probs,
# which this script never calls. Stub it so ordersmc/__init__ imports without it.
import sys as _sys, types as _types
try:
    import networkx  # noqa: F401
except ModuleNotFoundError:
    _sys.modules["networkx"] = _types.ModuleType("networkx")

from ordersmc.scoring import UnifiedFamilyScorer
from ordersmc.core import (
    OrderCandidateTable,
    edge_probs_for_order,
    build_order_candidate_table,
    order_smc,
)
from ordersmc.config import DAGConfig


# ---------------------------------------------------------------------------
# 1. Synthetic time-course with a planted t->t+1 structure
# ---------------------------------------------------------------------------
def simulate(n_series: int = 3, T: int = 7, noise: float = 0.15, seed: int = 1):
    """Return (node_names, list_of_(T x p)_trajectories).

    Each 'series' stands in for one dose (or one replicate): an independent
    trajectory of the same dynamical system. Planted transition map M:
        B_{t+1} =  0.9 * A_t
        C_{t+1} =  0.9 * B_t
        D_{t+1} = -0.85 * C_t
        A_{t+1} =  0.1 * A_t   (weak -> A is effectively an exogenous driver)
        E_{t+1} =  noise       (no parent)
    """
    rng = np.random.default_rng(seed)
    nodes = ["A", "B", "C", "D", "E"]
    ix = {n: i for i, n in enumerate(nodes)}
    M = np.zeros((5, 5))
    M[ix["B"], ix["A"]] = 0.9
    M[ix["C"], ix["B"]] = 0.9
    M[ix["D"], ix["C"]] = -0.85
    M[ix["A"], ix["A"]] = 0.1
    series = []
    for _ in range(n_series):
        X = np.zeros((T, 5))
        X[0] = rng.normal(0, 1, 5)
        for t in range(1, T):
            X[t] = M @ X[t - 1] + rng.normal(0, noise, 5)
        series.append(X)
    return nodes, series


# ---------------------------------------------------------------------------
# 2. Transition-table construction: lagged (t, t+1) rows
# ---------------------------------------------------------------------------
def transition_frame(nodes, series):
    """Each row = one (t -> t+1) step. Columns = {X__t} then {X__t1}."""
    src = [f"{n}__t" for n in nodes]
    tgt = [f"{n}__t1" for n in nodes]
    rows = []
    for X in series:
        for t in range(X.shape[0] - 1):
            row = {}
            for i, n in enumerate(nodes):
                row[f"{n}__t"] = X[t, i]
                row[f"{n}__t1"] = X[t + 1, i]
            rows.append(row)
    return pd.DataFrame(rows, columns=src + tgt), src, tgt


# ---------------------------------------------------------------------------
# 3. Parent-mask hook: build an OrderCandidateTable with per-child allowed parents
#    (at kinase-activity scale we can enumerate parent sets exactly -- no screening)
# ---------------------------------------------------------------------------
def build_masked_table(scorer, nodes_order, allowed_parents, cfg) -> OrderCandidateTable:
    node_to_idx = {n: i for i, n in enumerate(nodes_order)}
    parent_sets, logliks, logpriors = [], [], []
    for child in nodes_order:
        allowed = allowed_parents.get(child, [])
        families = [tuple()]
        for r in range(1, cfg.max_parents + 1):
            families += [tuple(sorted(ps)) for ps in combinations(allowed, r)]
        recs = []
        for ps in families:
            res = scorer.score(child, ps)
            prior = -cfg.lam * int(res.k_params) - cfg.edge_penalty * len(ps)
            recs.append((ps, float(res.log_marginal), float(prior)))
        width = max(1, max(len(ps) for ps, _, _ in recs))
        ps_arr = np.full((len(recs), width), -1, dtype=np.int32)
        ll = np.empty(len(recs)); pr = np.empty(len(recs))
        for i, (ps, l, p) in enumerate(recs):
            for j, name in enumerate(ps):
                ps_arr[i, j] = node_to_idx[name]
            ll[i] = l; pr[i] = p
        parent_sets.append(ps_arr); logliks.append(ll); logpriors.append(pr)
    return OrderCandidateTable(nodes=list(nodes_order), parent_sets=parent_sets,
                               loglik=logliks, logprior=logpriors)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    nodes, series = simulate(n_series=3, T=7, seed=1)  # 3 series x 6 transitions = 18 rows
    df, src, tgt = transition_frame(nodes, series)
    print(f"Transition frame: {df.shape[0]} rows (transitions) x {df.shape[1]} nodes "
          f"({len(src)} source + {len(tgt)} target)\n")

    vtypes = {c: "con" for c in df.columns}
    cards = {c: 0 for c in df.columns}
    scorer = UnifiedFamilyScorer(df, vtypes, cards, alpha=1.0, tau=3.0)
    cfg = DAGConfig(lam=0.5, edge_penalty=2.0, max_parents=2)

    nodes_order = src + tgt                          # sources first == temporal order
    allowed = {**{s: [] for s in src},               # sources: exogenous, no parents
               **{t: list(src) for t in tgt}}        # targets: parents only from slice t

    # --- Path A: exact fixed-order posterior (deterministic, no sampler) ---
    table = build_masked_table(scorer, nodes_order, allowed, cfg)
    fixed_order = np.arange(len(nodes_order), dtype=np.int32)
    edge_probs = edge_probs_for_order(fixed_order, table, beta=1.0)
    inter = {e: p for e, p in edge_probs.items() if e[0].endswith("__t") and e[1].endswith("__t1")}

    print("=== Path A: EXACT inter-slice posterior (edge_probs_for_order, fixed temporal order) ===")
    print("planted chain: A__t->B__t1, B__t->C__t1, C__t->D__t1\n")
    for e, p in sorted(inter.items(), key=lambda kv: -kv[1]):
        if p >= 0.30:
            print(f"  {e[0]:>6} -> {e[1]:<7}  posterior = {p:.3f}")

    # determinism check: recompute a few times, results are byte-identical (no RNG)
    reps = [edge_probs_for_order(fixed_order, table, beta=1.0) for _ in range(3)]
    same = all(r == edge_probs for r in reps)
    print(f"\n  deterministic across recomputes: {same}  "
          f"(the fixed-order path has no RNG -> zero seed variance)\n")

    # --- Contrast: stock UNCONSTRAINED order-SMC on the same frame, several seeds ---
    print("=== Contrast: stock unconstrained order-SMC (samples the ORDER) across seeds ===")
    stock = build_order_candidate_table(scorer, nodes_order, cfg, screen_k=9, max_parents=2)
    betas = np.linspace(0.0, 1.0, 40)
    for seed in (1, 2, 3, 4, 5):
        res = order_smc(stock, particles=80, betas=betas, mh_steps=10,
                        seed=seed, verbose=False)
        strong = sorted([f"{a}->{b}" for (a, b), p in res.edge_probs.items() if p >= 0.7])
        print(f"  seed {seed}: {len(strong):>2} edges @>=0.7 :: {strong}")


if __name__ == "__main__":
    main()
