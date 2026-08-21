from __future__ import annotations

import numpy as np
import pandas as pd

from ordersmc import OrderSMCConfig, dag_from_edge_probs, fit


def test_end_to_end_continuous_smoke() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=120)
    y = 1.4 * x + rng.normal(scale=0.4, size=120)
    z = -0.8 * y + rng.normal(scale=0.5, size=120)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})
    vtypes = {name: "con" for name in data.columns}
    cards = {name: 0 for name in data.columns}

    result = fit(
        data,
        vtypes,
        cards,
        config=OrderSMCConfig(
            particles=16,
            num_betas=6,
            mh_steps=2,
            screen_k=2,
            max_parents=2,
            pair_rescue_anchor_k=1,
            pair_rescue_k=1,
            parent_pool_max=2,
            edge_penalty=1.0,
            seed=4,
        ),
        verbose=False,
    )
    graph = dag_from_edge_probs(result.edge_probs, data.columns, threshold=0.5)
    assert len(result.orders) == 16
    assert np.isclose(result.weights.sum(), 1.0)
    assert set(graph.nodes()) == set(data.columns)
    assert nx_is_dag(graph)


def nx_is_dag(graph) -> bool:
    import networkx as nx

    return nx.is_directed_acyclic_graph(graph)

