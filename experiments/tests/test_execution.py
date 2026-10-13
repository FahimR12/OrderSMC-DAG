from __future__ import annotations

import json
from pathlib import Path

import pytest

from ordersmc_experiments.execution import (
    DEFAULT_CONFIG,
    _all_edges,
    _compressed_indices,
    _sharded_indices,
    _validate_plan_execution_binding,
    _sha,
    _execution_config,
    _full_mcmc_controls,
    _profile_config,
    aggregate,
    write_plan,
)


def test_probability_rows_are_clipped_and_indices_are_compacted() -> None:
    rows = _all_edges(
        ["A", "B"], {("A", "B"): 1.0 + 1e-15, ("B", "A"): -1e-15}
    )
    assert rows == [
        {"parent": "A", "child": "B", "probability": 1.0},
        {"parent": "B", "child": "A", "probability": 0.0},
    ]
    assert _compressed_indices([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert _compressed_indices([]) == ""


def test_global_run_indices_are_mapped_to_bounded_slurm_shards() -> None:
    assert _sharded_indices([0, 1, 999, 1000, 1002, 5259], 1000) == [
        (0, "0-1,999"),
        (1000, "0,2"),
        (5000, "259"),
    ]
    with pytest.raises(ValueError, match="positive integer"):
        _sharded_indices([0], 0)


def test_write_plan_is_idempotent_after_json_tuple_normalization(tmp_path) -> None:
    destination = tmp_path / "plan.json"
    write_plan("e1", 20260814, destination)
    first = destination.read_bytes()
    write_plan("e1", 20260814, destination)
    assert destination.read_bytes() == first


def test_pilot_frozen_sampler_controls_are_bound_to_total_budget() -> None:
    execution = _execution_config(DEFAULT_CONFIG)
    assert execution["ordersmc"] == {
        "particles": 512,
        "num_betas": 100,
        "mh_steps": 10,
        "mh_steps_resample": 25,
        "resample_ess_frac": 0.5,
        "posterior_threshold": 0.5,
    }
    evaluation = {"role": "evaluation", "wall_seconds": 300}
    assert _full_mcmc_controls("order-mcmc", evaluation, execution) == (50, 300.0, 270.0)
    assert _full_mcmc_controls("partition-mcmc", evaluation, execution) == (10, 300.0, 300.0)
    reference = {"role": "reference", "wall_seconds": None}
    assert _full_mcmc_controls("order-modular-reference", reference, execution) == (
        50,
        1200.0,
        1200.0,
    )


def test_confirmatory_config_fails_closed_after_tampering(tmp_path) -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    freeze_path = DEFAULT_CONFIG.parent / config["freeze_record"]
    destination = tmp_path / DEFAULT_CONFIG.name
    destination.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / freeze_path.name).write_text(freeze_path.read_text(encoding="utf-8"), encoding="utf-8")
    assert _execution_config(destination)["status"] == "confirmatory-frozen"
    config["ordersmc"]["particles"] = 513
    destination.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot freeze"):
        _execution_config(destination)


def test_mixed_paper_execution_amendment_is_freeze_bound() -> None:
    path = DEFAULT_CONFIG.with_name("execution_mixed_paper_v1.json")
    value = _execution_config(path)
    assert value["schema"] == "ordersmc-execution-config-v4"
    assert value["generator"]["max_extreme_probability_fraction"] == 0.05

    final_path = DEFAULT_CONFIG.with_name("execution_mixed_paper_v2.json")
    final = _execution_config(final_path)
    assert final["schema"] == "ordersmc-execution-config-v5"
    assert final["generator"]["enforce_extreme_probability_fraction"] is False


def test_runtime_calibrated_config_has_dimension_specific_completed_smc_schedules() -> None:
    path = DEFAULT_CONFIG.with_name("execution_budget_pilot_v1.json")
    value = _execution_config(path)
    assert value["schema"] == "ordersmc-execution-config-v6"
    assert value["status"] == "confirmatory-frozen"
    assert set(value["e2_ordersmc_by_dimension"]) == {"20", "50", "100"}
    assert set(value["e3_ordersmc_ladder_by_dimension"]) == {"50", "100"}
    assert all(
        set(ladder) == {"30", "60", "180", "300"}
        for ladder in value["e3_ordersmc_ladder_by_dimension"].values()
    )


def test_v3_confirmatory_config_is_bound_to_resource_only_pilot_freeze() -> None:
    path = DEFAULT_CONFIG.with_name("execution_v3_confirmatory_v1.json")
    value = _execution_config(path)
    assert value["schema"] == "ordersmc-execution-config-v7"
    assert value["status"] == "confirmatory-frozen"
    assert value["ordersmc"]["particles"] == 4096
    assert value["v3_high_indegree_candidate"] == {
        "screen_k": 10,
        "parent_pool_max": 15,
    }
    plan = {
        "experiment": "v3",
        "master_seed": 20260904,
    }
    _validate_plan_execution_binding(plan, path, value, "full")
    plan["master_seed"] = 20260908
    with pytest.raises(ValueError, match="frozen value 20260904"):
        _validate_plan_execution_binding(plan, path, value, "full")


def test_v3_clg_corrective_config_selects_score_by_model_class() -> None:
    path = DEFAULT_CONFIG.with_name("execution_v3_clg_corrective_v1.json")
    execution = _execution_config(path)
    common = {
        "experiment": "v3",
        "dimension": 8,
        "max_indegree": 4,
    }
    clg = _profile_config(
        {**common, "data_kind": "native_mixed_conditional_gaussian"},
        execution,
        "full",
        7,
    )
    unrelated = _profile_config(
        {**common, "data_kind": "topology_preserving_mixed"},
        execution,
        "full",
        7,
    )
    assert clg.score_method == "clg-bic"
    assert unrelated.score_method == "native"


def test_v1_aggregation_preserves_failed_exact_rows_without_reading_pips(
    tmp_path: Path,
) -> None:
    run = {
        "run_id": "dataset-exact-order",
        "dataset_id": "dataset",
        "method": "exact-order-dp",
        "target": "order-modular",
        "role": "reference",
        "seed_index": None,
        "wall_seconds": None,
        "mpi_ranks": 1,
    }
    plan = {
        "experiment": "v1",
        "master_seed": 17,
        "runs": [run],
        "datasets": [],
    }
    plan["plan_sha256"] = _sha(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "dataset-exact-order.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_type": "RuntimeError",
                "failure_message": "diagnostic normalization failed",
            }
        ),
        encoding="utf-8",
    )

    aggregate(plan_path, tmp_path)

    metrics = (tmp_path / "analysis" / "run_metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "diagnostic normalization failed" in metrics
    completion = json.loads(
        (tmp_path / "analysis" / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["planned_runs"] == 1
    assert completion["completed_runs"] == 0
    assert completion["status"] == "incomplete"


def test_aggregation_preserves_undefined_metrics_and_separates_metadata_arms(tmp_path):
    import csv
    runs=[]
    (tmp_path/'runs').mkdir()
    for i,(arm,value) in enumerate([('m1_matched',.8),('m1_matched',None),('m5_negative_controls',None)]):
        run=dict(run_id=f'run{i}',dataset_id=f'dataset{i}',method='ordersmc',target='order-modular',
                 role='evaluation',seed_index=0,wall_seconds=10,mpi_ranks=1)
        runs.append(run)
        result=dict(status='complete',dataset=dict(dimension=3,role=arm,cell_id=arm,n_train=100),
                    runtime_seconds=1,metrics=dict(directed_recall=value,directed_precision=.5))
        (tmp_path/'runs'/f'run{i}.json').write_text(json.dumps(result))
    plan=dict(experiment='s2',master_seed=17,runs=runs,datasets=[])
    plan['plan_sha256']=_sha(plan)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    aggregate(path,tmp_path)
    with (tmp_path/'analysis/summary.csv').open() as handle: rows=list(csv.DictReader(handle))
    rows={r['dataset_role']:r for r in rows if r['metric']=='directed_recall'}
    assert set(rows)=={'m1_matched','m5_negative_controls'}
    assert float(rows['m1_matched']['mean'])==.8
    assert rows['m1_matched']['n_datasets']=='1'
    assert rows['m1_matched']['n_total_datasets']=='2'
    assert rows['m1_matched']['n_undefined_runs']=='1'
    assert rows['m5_negative_controls']['mean']==''
    assert rows['m5_negative_controls']['n_defined_runs']=='0'
    assert rows['m5_negative_controls']['n_undefined_runs']=='1'
