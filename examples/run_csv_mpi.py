#!/usr/bin/env python3
"""Fit MPI-parallel OrderSMC-DAG to a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

from ordersmc import DAGConfig, UnifiedFamilyScorer, dag_from_edge_probs, infer_metadata_and_encode
from ordersmc.mpi import build_order_candidate_table_mpi, order_smc_mpi


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data", type=Path)
parser.add_argument("--output", type=Path, default=Path("results/ordersmc-mpi"))
parser.add_argument("--threshold", type=float, default=0.7)
parser.add_argument("--particles", type=int, default=400)
parser.add_argument("--betas", type=int, default=160)
parser.add_argument("--mh-steps", type=int, default=20)
parser.add_argument("--mh-steps-resample", type=int, default=50)
parser.add_argument("--screen-k", type=int, default=16)
parser.add_argument("--max-parents", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    raw = pd.read_csv(args.data)
    data, vtypes, cards = infer_metadata_and_encode(raw)
    nodes = list(data.columns)
else:
    data = vtypes = cards = nodes = None

data = comm.bcast(data, root=0)
vtypes = comm.bcast(vtypes, root=0)
cards = comm.bcast(cards, root=0)
nodes = comm.bcast(nodes, root=0)

scorer = UnifiedFamilyScorer(data, vtypes, cards, alpha=1.0, tau=3.0, use_sparse_bdeu=True)
prior = DAGConfig(lam=0.75, edge_penalty=8.0, max_parents=args.max_parents)
table = build_order_candidate_table_mpi(
    scorer,
    nodes,
    prior,
    screen_k=args.screen_k,
    max_parents=args.max_parents,
    pair_rescue_anchor_k=min(8, args.screen_k),
    pair_rescue_k=20,
    parent_pool_max=max(24, args.screen_k),
    comm=comm,
    verbose=rank == 0,
)
result = order_smc_mpi(
    table,
    particles=args.particles,
    betas=np.linspace(0.0, 1.0, args.betas),
    mh_steps=args.mh_steps,
    mh_steps_resample=args.mh_steps_resample,
    seed=args.seed,
    comm=comm,
    verbose=rank == 0,
)

if rank == 0:
    dag = dag_from_edge_probs(result.edge_probs, nodes, threshold=args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.best_edges, columns=["Parent", "Child"]).to_csv(
        args.output / "best_edges.csv", index=False
    )
    pd.DataFrame(sorted(dag.edges()), columns=["Parent", "Child"]).to_csv(
        args.output / "posterior_dag.csv", index=False
    )
    pd.DataFrame(
        [(u, v, p) for (u, v), p in sorted(result.edge_probs.items())],
        columns=["Parent", "Child", "PosteriorProbability"],
    ).to_csv(args.output / "edge_posterior.csv", index=False)
    pd.DataFrame(result.history).to_csv(args.output / "smc_history.csv", index=False)
    print(f"Saved results to {args.output.resolve()}")

