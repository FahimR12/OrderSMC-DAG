#!/usr/bin/env python3
"""Small mixed-data example that can be run after editable installation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig, dag_from_edge_probs, fit


def make_data(samples: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.5, size=samples)
    b = 1.8 * a + rng.normal(0.0, 0.7, size=samples)
    c_prob = 1.0 / (1.0 + np.exp(-(-0.5 + 1.7 * a)))
    c = rng.binomial(1, c_prob)
    d = 1.2 * b + 1.5 * c + rng.normal(0.0, 0.8, size=samples)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d})


data = make_data()
vtypes = {"A": "bin", "B": "con", "C": "bin", "D": "con"}
cards = {"A": 2, "B": 0, "C": 2, "D": 0}

config = OrderSMCConfig(
    particles=80,
    num_betas=40,
    mh_steps=5,
    screen_k=3,
    pair_rescue_anchor_k=2,
    pair_rescue_k=3,
    parent_pool_max=3,
    edge_penalty=3.0,
    seed=11,
)
result = fit(data, vtypes, cards, config=config, verbose=False)
dag = dag_from_edge_probs(result.edge_probs, data.columns, threshold=0.5)

print("Best-order DAG:", result.best_edges)
print("Posterior DAG:", sorted(dag.edges()))
print("Edge posterior probabilities:")
for edge, probability in sorted(result.edge_probs.items(), key=lambda item: -item[1]):
    print(f"  {edge[0]} -> {edge[1]}: {probability:.3f}")

