#!/usr/bin/env python3
"""Multi-rank smoke test for parallel table construction and Order-SMC."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from mpi4py import MPI

from ordersmc import DAGConfig, UnifiedFamilyScorer, dag_from_edge_probs
from ordersmc.mpi import build_order_candidate_table_mpi, order_smc_mpi


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    rng = np.random.default_rng(91)
    x = rng.normal(size=160)
    y = 1.6 * x + rng.normal(scale=0.45, size=160)
    z = -1.1 * y + rng.normal(scale=0.5, size=160)
    w = 0.9 * x + 0.8 * z + rng.normal(scale=0.55, size=160)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z, "W": w})
else:
    data = None

data = comm.bcast(data, root=0)
nodes = list(data.columns)
vtypes = {node: "con" for node in nodes}
cards = {node: 0 for node in nodes}
scorer = UnifiedFamilyScorer(data, vtypes, cards)
prior = DAGConfig(lam=0.25, edge_penalty=1.0, max_parents=2)

table = build_order_candidate_table_mpi(
    scorer,
    nodes,
    prior,
    screen_k=3,
    max_parents=2,
    pair_rescue_anchor_k=2,
    pair_rescue_k=3,
    parent_pool_max=3,
    comm=comm,
    verbose=rank == 0,
)
result = order_smc_mpi(
    table,
    particles=32,
    betas=np.linspace(0.0, 1.0, 10),
    mh_steps=2,
    mh_steps_resample=4,
    seed=17,
    comm=comm,
    verbose=False,
)

assert len(table.nodes) == 4
assert all(len(parent_sets) > 0 for parent_sets in table.parent_sets)
if rank == 0:
    assert result.orders is not None
    assert result.weights is not None
    assert result.edge_probs is not None
    assert len(result.orders) == 32
    assert np.isclose(result.weights.sum(), 1.0)
    graph = dag_from_edge_probs(result.edge_probs, nodes, threshold=0.5)
    assert nx.is_directed_acyclic_graph(graph)
    print(
        "MPI_SMOKE_OK "
        f"ranks={comm.Get_size()} particles={len(result.orders)} "
        f"best_edges={result.best_edges} posterior_edges={sorted(graph.edges())}",
        flush=True,
    )

