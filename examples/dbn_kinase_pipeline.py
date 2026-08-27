"""
Inter-slice DBN on kinase-activity time-course, driven from a CSV.
Path A of the OrderSMC-DAG DBN reframe: KSEA -> lagged transitions -> EXACT
fixed-temporal-order posterior (deterministic, seed-free).

Pipeline
--------
    [phospho log2FC + kinase-substrate table]        (optional KSEA pre-step)
                    |  ksea_activity()
                    v
    activity_long: [kinase, dose, timepoint, replicate, activity]   <-- "the matrix"
                    |  build_transition_frame()
                    v
    transition frame: rows = (t -> t+1) steps, cols = {K__t} then {K__t1}
                    |  inter_slice_posterior()   (masked candidate table + edge_probs_for_order)
                    v
    edge table: parent_kinase -> child_kinase, lag 1, posterior  (exact, no seeds)

Two ways in:
    python dbn_kinase_pipeline.py --activity activity_long.csv \
        --timepoint-order "5min,20min,1h,4h,8h" --out edges.csv
    python dbn_kinase_pipeline.py --phospho phospho_long.csv --ksub kinase_substrate.csv \
        --timepoint-order "5min,20min,1h,4h,8h" --out edges.csv
    python dbn_kinase_pipeline.py --demo         # self-test on synthetic Claire-shaped data

NOTE on Nefeli's "no biological prior" rule: KSEA *does* inject kinase->substrate
prior knowledge -- but at the FEATURE-CONSTRUCTION level, not as edge priors. The
learned edges are still prior-free. The prior is explicit and swappable (the --ksub
table). If that is unacceptable, skip KSEA and pass a data-driven activity matrix.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

# networkx is only used inside ordersmc.graphs, which we never call -> stub if absent.
import sys as _sys, types as _types
try:
    import networkx  # noqa: F401
except ModuleNotFoundError:
    _sys.modules["networkx"] = _types.ModuleType("networkx")

from ordersmc.scoring import UnifiedFamilyScorer
from ordersmc.core import OrderCandidateTable, edge_probs_for_order
from ordersmc.config import DAGConfig


# ---------------------------------------------------------------------------
# Optional KSEA pre-step: phosphosite log2FC -> per-condition kinase activity
# ---------------------------------------------------------------------------
def ksea_activity(
    phospho_long: pd.DataFrame,
    ksub: pd.DataFrame,
    *,
    site_col: str = "site",
    value_col: str = "log2fc",
    cond_cols: Sequence[str] = ("dose", "timepoint", "replicate"),
    kinase_col: str = "kinase",
    ksub_site_col: str = "site",
    min_substrates: int = 3,
) -> pd.DataFrame:
    """Classic KSEA mean method (Casado et al. 2013), computed per condition.

    z = (mean_FC_substrates - mean_FC_all) * sqrt(m) / sd_FC_all

    Returns activity_long: [kinase, *cond_cols, activity]. Kinases with fewer
    than ``min_substrates`` quantified sites in a condition are dropped there.
    This is a transparent, dependency-light KSEA; swap in decoupleR/KSEAapp if
    preferred -- the downstream code only needs the activity_long schema.
    """
    cond_cols = list(cond_cols)
    k2s = ksub[[kinase_col, ksub_site_col]].dropna().drop_duplicates()
    rows: List[dict] = []
    for cond, grp in phospho_long.groupby(cond_cols, dropna=False):
        cond = cond if isinstance(cond, tuple) else (cond,)
        fc = grp[[site_col, value_col]].dropna()
        mP = float(fc[value_col].mean())
        sd = float(fc[value_col].std(ddof=1))
        if not np.isfinite(sd) or sd == 0.0:
            continue
        fc_by_site = fc.set_index(site_col)[value_col]
        merged = k2s.merge(fc_by_site.rename("fc"), left_on=ksub_site_col,
                           right_index=True, how="inner")
        for kinase, ksub_grp in merged.groupby(kinase_col):
            m = int(ksub_grp.shape[0])
            if m < min_substrates:
                continue
            mS = float(ksub_grp["fc"].mean())
            z = (mS - mP) * np.sqrt(m) / sd
            rec = dict(zip(cond_cols, cond))
            rec[kinase_col] = kinase
            rec["activity"] = z
            rows.append(rec)
    out = pd.DataFrame(rows, columns=[kinase_col, *cond_cols, "activity"])
    return out


# ---------------------------------------------------------------------------
# Transition-table construction (generalised: real conditions, replicates)
# ---------------------------------------------------------------------------
def build_transition_frame(
    activity_long: pd.DataFrame,
    timepoint_order: Sequence[str],
    *,
    kinase_col: str = "kinase",
    timepoint_col: str = "timepoint",
    value_col: str = "activity",
    group_cols: Sequence[str] = ("dose", "replicate"),
    include_self: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, List[str]]]:
    """Reshape long kinase-activity into lagged (t -> t+1) rows.

    Transitions are formed only within the same ``group_cols`` block (e.g. same
    dose and replicate) between *consecutive* entries of ``timepoint_order``.
    Timepoints not in ``timepoint_order`` (e.g. 24h) are dropped.

    Returns (frame, src_nodes, tgt_nodes, allowed_parents), where
      src node "K__t"  = kinase K at time t   (exogenous; no parents)
      tgt node "K__t1" = kinase K at time t+1 (parents drawn from src slice)
    ``include_self=False`` forbids the autoregressive edge K__t -> K__t1.
    """
    group_cols = [c for c in group_cols if c in activity_long.columns]
    order_ix = {tp: i for i, tp in enumerate(timepoint_order)}
    df = activity_long[activity_long[timepoint_col].isin(order_ix)].copy()
    kinases = sorted(df[kinase_col].unique())
    src = [f"{k}__t" for k in kinases]
    tgt = [f"{k}__t1" for k in kinases]

    rows: List[dict] = []
    grouper = df.groupby(group_cols) if group_cols else [((), df)]
    for _, block in grouper:
        wide = block.pivot_table(index=kinase_col, columns=timepoint_col,
                                 values=value_col, aggfunc="mean")
        present = [tp for tp in timepoint_order if tp in wide.columns]
        for a, b in zip(present[:-1], present[1:]):
            row = {}
            for k in kinases:
                va = wide.at[k, a] if k in wide.index else np.nan
                vb = wide.at[k, b] if k in wide.index else np.nan
                row[f"{k}__t"] = va
                row[f"{k}__t1"] = vb
            rows.append(row)

    frame = pd.DataFrame(rows, columns=src + tgt)
    before = len(frame)
    frame = frame.dropna(axis=0, how="any").reset_index(drop=True)
    if len(frame) < before:
        print(f"[transition] dropped {before - len(frame)} rows with missing values")

    allowed: Dict[str, List[str]] = {s: [] for s in src}
    for k in kinases:
        parents = list(src)
        if not include_self:
            parents = [p for p in parents if p != f"{k}__t"]
        allowed[f"{k}__t1"] = parents
    return frame, src, tgt, allowed


# ---------------------------------------------------------------------------
# Masked candidate table + EXACT fixed-order posterior
# ---------------------------------------------------------------------------
def _build_masked_table(scorer, nodes_order, allowed_parents, cfg) -> OrderCandidateTable:
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


def inter_slice_posterior(
    frame: pd.DataFrame,
    src: Sequence[str],
    tgt: Sequence[str],
    allowed: Dict[str, List[str]],
    *,
    lam: float = 2.0,
    edge_penalty: float = 3.5,
    tau: float = 1.0,
    alpha: float = 1.0,
    max_parents: int = 2,
) -> pd.DataFrame:
    """Return exact inter-slice edge posteriors as a tidy DataFrame."""
    vtypes = {c: "con" for c in frame.columns}
    cards = {c: 0 for c in frame.columns}
    scorer = UnifiedFamilyScorer(frame, vtypes, cards, alpha=alpha, tau=tau)
    cfg = DAGConfig(lam=lam, edge_penalty=edge_penalty, max_parents=max_parents)

    nodes_order = list(src) + list(tgt)            # sources first == temporal order
    table = _build_masked_table(scorer, nodes_order, allowed, cfg)
    fixed_order = np.arange(len(nodes_order), dtype=np.int32)
    edge_probs = edge_probs_for_order(fixed_order, table, beta=1.0)

    rows = []
    for (parent, child), p in edge_probs.items():
        if parent.endswith("__t") and child.endswith("__t1"):
            pk, ck = parent[:-3], child[:-4]
            rows.append({"parent_kinase": pk, "child_kinase": ck, "lag": 1,
                         "self_edge": pk == ck, "posterior": float(p)})
    out = pd.DataFrame(rows).sort_values("posterior", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Synthetic Claire-shaped data for the self-test
# ---------------------------------------------------------------------------
def _make_demo(seed: int = 1):
    """12 kinases, 2 doses, 6 timepoints (24h to be dropped), 3 replicates.
    Planted t->t+1 chain: AKT1->GSK3B, EGFR->MAPK1, MAPK1->RPS6KB1, PRKCA-|MARCKS."""
    rng = np.random.default_rng(seed)
    kinases = ["EGFR", "MAPK1", "AKT1", "GSK3B", "PRKCA", "MARCKS",
               "RPS6KB1", "SHC1", "GAB1", "PLCG1", "RPS6KA1", "EIF4EBP1"]
    ki = {k: i for i, k in enumerate(kinases)}
    M = np.zeros((12, 12))
    M[ki["GSK3B"], ki["AKT1"]] = 0.9
    M[ki["MAPK1"], ki["EGFR"]] = 0.9
    M[ki["RPS6KB1"], ki["MAPK1"]] = 0.85
    M[ki["MARCKS"], ki["PRKCA"]] = -0.85
    M[ki["EGFR"], ki["EGFR"]] = 0.1
    tps = ["5min", "20min", "1h", "4h", "8h", "24h"]
    recs = []
    for dose in ["10nM", "1uM"]:
        for rep in [1, 2, 3]:
            X = np.zeros((len(tps), 12))
            X[0] = rng.normal(0, 1, 12)
            for t in range(1, len(tps)):
                X[t] = M @ X[t - 1] + rng.normal(0, 0.15, 12)
            for t, tp in enumerate(tps):
                for k in kinases:
                    recs.append({"kinase": k, "dose": dose, "timepoint": tp,
                                 "replicate": rep, "activity": float(X[t, ki[k]])})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Inter-slice DBN on kinase-activity time-course.")
    ap.add_argument("--activity", help="long-format activity CSV [kinase,dose,timepoint,replicate,activity]")
    ap.add_argument("--phospho", help="long-format phospho CSV [site,dose,timepoint,replicate,log2fc] (KSEA mode)")
    ap.add_argument("--ksub", help="kinase-substrate table CSV [kinase,site] (KSEA mode)")
    ap.add_argument("--timepoint-order", default="5min,20min,1h,4h,8h",
                    help="comma list, in order; timepoints outside it are dropped (e.g. 24h)")
    ap.add_argument("--group-cols", default="dose,replicate")
    ap.add_argument("--include-self", action="store_true", help="allow autoregressive K__t -> K__t1 edges")
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--edge-penalty", type=float, default=3.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--max-parents", type=int, default=2)
    ap.add_argument("--out", default="dbn_inter_slice_edges.csv")
    ap.add_argument("--demo", action="store_true", help="run self-test on synthetic Claire-shaped data")
    args = ap.parse_args()

    tp_order = [s.strip() for s in args.timepoint_order.split(",") if s.strip()]
    group_cols = [s.strip() for s in args.group_cols.split(",") if s.strip()]

    if args.demo:
        activity = _make_demo()
        print(f"[demo] synthetic activity_long: {activity.shape[0]} rows, "
              f"{activity['kinase'].nunique()} kinases, "
              f"{activity['dose'].nunique()} doses x {activity['timepoint'].nunique()} timepoints "
              f"x {activity['replicate'].nunique()} replicates")
    elif args.phospho and args.ksub:
        phospho = pd.read_csv(args.phospho)
        ksub = pd.read_csv(args.ksub)
        activity = ksea_activity(phospho, ksub, cond_cols=[c for c in ["dose", "timepoint", "replicate"]
                                                           if c in phospho.columns])
        print(f"[ksea] {activity['kinase'].nunique()} kinases scored across "
              f"{activity.drop(columns=['kinase','activity']).drop_duplicates().shape[0]} conditions")
    elif args.activity:
        activity = pd.read_csv(args.activity)
    else:
        ap.error("provide --activity, or --phospho with --ksub, or --demo")

    frame, src, tgt, allowed = build_transition_frame(
        activity, tp_order, group_cols=group_cols, include_self=args.include_self)
    print(f"[transition] {frame.shape[0]} transition rows x {len(src)} source + {len(tgt)} target nodes")
    if frame.shape[0] < 2 * args.max_parents + 2:
        print(f"[warn] only {frame.shape[0]} rows -- posteriors will be weakly identified "
              f"(replicate-level data or fewer kinases would help)")

    edges = inter_slice_posterior(
        frame, src, tgt, allowed,
        lam=args.lam, edge_penalty=args.edge_penalty, tau=args.tau, max_parents=args.max_parents)

    edges.to_csv(args.out, index=False)
    print(f"\nTop inter-slice edges (written to {args.out}):")
    show = edges[edges["posterior"] >= 0.30]
    with pd.option_context("display.max_rows", 40):
        print(show.to_string(index=False) if not show.empty else "  (none >= 0.30)")


if __name__ == "__main__":
    main()
