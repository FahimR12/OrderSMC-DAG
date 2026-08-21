#!/usr/bin/env python3
"""Fit OrderSMC-DAG to a CSV file in one process."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ordersmc import OrderSMCConfig, dag_from_edge_probs, fit, infer_metadata_and_encode


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data", type=Path)
parser.add_argument("--output", type=Path, default=Path("results/ordersmc"))
parser.add_argument("--threshold", type=float, default=0.7)
parser.add_argument("--particles", type=int, default=400)
parser.add_argument("--betas", type=int, default=160)
parser.add_argument("--mh-steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

raw = pd.read_csv(args.data)
data, vtypes, cards = infer_metadata_and_encode(raw)
result = fit(
    data,
    vtypes,
    cards,
    config=OrderSMCConfig(
        particles=args.particles,
        num_betas=args.betas,
        mh_steps=args.mh_steps,
        seed=args.seed,
    ),
)
dag = dag_from_edge_probs(result.edge_probs, data.columns, threshold=args.threshold)

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

