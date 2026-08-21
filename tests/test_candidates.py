from __future__ import annotations

from typing import Sequence

import numpy as np

from ordersmc import (
    DAGConfig,
    FamilyScoreResult,
    OrderCandidateTable,
    build_order_candidate_table,
    edge_probs_for_order,
)


class ToyFamilyScorer:
    def __init__(self) -> None:
        self.log_marginals = {
            ("C", tuple()): 0.0,
            ("C", ("A",)): 10.0,
            ("C", ("B",)): -10.0,
            ("C", ("D",)): -9.0,
            ("C", ("A", "B")): 30.0,
            ("C", ("A", "D")): 11.0,
        }

    def score(self, child: str, parents: Sequence[str]) -> FamilyScoreResult:
        return FamilyScoreResult(
            family="clg",
            log_marginal=self.log_marginals.get((child, tuple(sorted(parents))), 0.0),
            k_params=0,
        )


def parent_sets_for(table: OrderCandidateTable, child: str) -> set[tuple[str, ...]]:
    child_idx = table.nodes.index(child)
    return {
        tuple(table.nodes[int(i)] for i in row if int(i) >= 0)
        for row in table.parent_sets[child_idx]
    }


def test_pair_rescue_recovers_parent_that_is_weak_alone() -> None:
    nodes = ["A", "B", "C", "D"]
    cfg = DAGConfig(max_parents=2)
    baseline = build_order_candidate_table(
        ToyFamilyScorer(), nodes, cfg, screen_k=1, max_parents=2
    )
    rescued = build_order_candidate_table(
        ToyFamilyScorer(),
        nodes,
        cfg,
        screen_k=1,
        max_parents=2,
        pair_rescue_anchor_k=1,
        pair_rescue_k=1,
        parent_pool_max=2,
    )
    assert ("A", "B") not in parent_sets_for(baseline, "C")
    assert ("A", "B") in parent_sets_for(rescued, "C")


def test_edge_probabilities_average_parent_sets_exactly() -> None:
    table = OrderCandidateTable(
        nodes=["A", "C"],
        parent_sets=[
            np.array([[-1]], dtype=np.int32),
            np.array([[-1], [0]], dtype=np.int32),
        ],
        loglik=[np.array([0.0]), np.array([0.0, 0.0])],
        logprior=[np.array([0.0]), np.array([0.0, 0.0])],
    )
    forward = edge_probs_for_order(np.array([0, 1], dtype=np.int32), table)
    backward = edge_probs_for_order(np.array([1, 0], dtype=np.int32), table)
    assert forward[("A", "C")] == 0.5
    assert backward.get(("A", "C"), 0.0) == 0.0

