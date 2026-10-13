"""Reproducible utilities for the OrderSMC-DAG Version 3 experiments."""

from .design import (
    EXPERIMENT_IDS,
    PROTOCOL_VERSION,
    DatasetSpec,
    ExperimentPlan,
    RunSpec,
    build_experiment_plan,
)
from .exact_order import ExactOrderDPResult, exact_order_dp
from .exact_structure import ExactStructureDPResult, exact_structure_dp
from .generator import (
    MixedDAGConfig,
    MixedDAGDataset,
    generate_mixed_dag_dataset,
    replay_mixed_dag_artifact,
    write_mixed_dag_artifact,
)
from .order_mcmc import OrderMCMCConfig, OrderMCMCResult, order_mcmc
from .reliability import bootstrap_reliability, fit_reliability
from .registry import method_registry
from .structure_mcmc import (
    PartitionMCMCConfig,
    REVMCMCConfig,
    StructureMCMCResult,
    partition_mcmc,
    rev_mcmc,
    validate_structure_candidate_table,
)
from .reproducibility import (
    MCSECalibrationCell,
    MCSECalibrationResult,
    RankLayoutConsistencyResult,
    assess_internal_mcse_calibration,
    assess_rank_layout_consistency,
    summarize_seed_stability,
)

__all__ = [
    "DatasetSpec",
    "EXPERIMENT_IDS",
    "ExactOrderDPResult",
    "ExactStructureDPResult",
    "ExperimentPlan",
    "MixedDAGConfig",
    "MixedDAGDataset",
    "OrderMCMCConfig",
    "OrderMCMCResult",
    "PROTOCOL_VERSION",
    "PartitionMCMCConfig",
    "REVMCMCConfig",
    "MCSECalibrationCell",
    "MCSECalibrationResult",
    "RankLayoutConsistencyResult",
    "RunSpec",
    "StructureMCMCResult",
    "bootstrap_reliability",
    "build_experiment_plan",
    "exact_order_dp",
    "exact_structure_dp",
    "fit_reliability",
    "generate_mixed_dag_dataset",
    "method_registry",
    "order_mcmc",
    "partition_mcmc",
    "replay_mixed_dag_artifact",
    "rev_mcmc",
    "assess_internal_mcse_calibration",
    "assess_rank_layout_consistency",
    "summarize_seed_stability",
    "write_mixed_dag_artifact",
    "validate_structure_candidate_table",
]
