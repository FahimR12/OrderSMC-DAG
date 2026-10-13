from __future__ import annotations

from ordersmc_experiments.generator import MixedDAGConfig, generate_mixed_dag_dataset


def test_e3_module_topology_is_integrated_with_the_mixed_sem_generator() -> None:
    dataset = generate_mixed_dag_dataset(
        MixedDAGConfig(
            d=50,
            n_train=25,
            graph_family="order_blocks",
            expected_indegree=2.0,
            max_indegree=3,
            signal_strength="moderate",
            pilot_size=300,
            master_seed=73,
            experiment_id="e3",
            cell_id="d050-order_blocks",
            replicate_id=0,
        )
    )
    details = dataset.generation_diagnostics["graph_details"]
    assert details["topology_manifest"]["schema"] == "ordersmc-e3-module-topology-v1"
    assert len(details["weak_bridge_edges"]) == 3
    assert dataset.dag.number_of_edges() / 50 == 2.06
    assert max(dict(dataset.dag.in_degree()).values()) <= 3
