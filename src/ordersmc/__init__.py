"""OrderSMC-DAG: scalable Bayesian DAG structure learning."""

from .api import build_candidates, fit, sample
from .confidence import edge_confidence_table, seed_consensus_table
from .config import DAGConfig, OrderSMCConfig
from .core import (
    OrderCandidateTable,
    OrderSMCResult,
    best_edges_for_order,
    build_order_candidate_table,
    build_order_candidate_table_parts,
    edge_probs_for_order,
    order_log_target,
    order_smc,
    weighted_edge_probs_for_orders,
    weighted_edge_summary_for_orders,
)
from .data import (
    infer_metadata_and_encode,
    metadata_from_mapping,
    metadata_summary,
    metadata_to_mapping,
    validate_data,
)
from .graphs import dag_from_edge_probs, edges_above_threshold
from .scoring import FamilyScoreResult, UnifiedFamilyScorer
from .persistence import load_candidate_table, load_config, save_candidate_table

__all__ = [
    "DAGConfig",
    "FamilyScoreResult",
    "OrderCandidateTable",
    "OrderSMCConfig",
    "OrderSMCResult",
    "UnifiedFamilyScorer",
    "best_edges_for_order",
    "build_candidates",
    "build_order_candidate_table",
    "build_order_candidate_table_parts",
    "dag_from_edge_probs",
    "edge_probs_for_order",
    "edge_confidence_table",
    "edges_above_threshold",
    "fit",
    "infer_metadata_and_encode",
    "load_candidate_table",
    "load_config",
    "metadata_from_mapping",
    "metadata_summary",
    "metadata_to_mapping",
    "order_log_target",
    "order_smc",
    "sample",
    "save_candidate_table",
    "seed_consensus_table",
    "validate_data",
    "weighted_edge_probs_for_orders",
    "weighted_edge_summary_for_orders",
]

__version__ = "0.2.0"
