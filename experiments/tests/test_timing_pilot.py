from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from ordersmc import OrderSMCConfig, build_candidates
from ordersmc.persistence import save_candidate_table, candidate_table_content_sha256
from ordersmc_experiments import timing_pilot as pilot, timing_report as report, timing_submit
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.exact_order import exact_order_dp
from ordersmc_experiments.readiness import canonical_sha
from ordersmc_experiments.readiness_audits import read
from ordersmc_experiments.sampler_corrections import artifact


def config():
    return read(Path(__file__).resolve().parents[1] / "configs/timing_pilot_v1.json")


def tiny_timing_plan(root, ranks=1):
    cfg = OrderSMCConfig.protocol_v2(lambda_target=1., max_parents=1, particles=16,
        score_method="native", adaptive_tempering=False, num_betas=3, mh_steps=1, mh_steps_resample=1,
        final_target_snapshot_steps=(0,))
    data = pd.DataFrame(dict(x=np.linspace(-2, 2, 30), y=np.arange(30) % 2))
    table = build_candidates(data, dict(x="con", y="bin"), dict(x=0, y=2), config=cfg, verbose=False)
    cpath = save_candidate_table(table, root / "candidate.json.gz")
    exact = exact_order_dp(table, compute_graph_size=False)
    epath = root / "exact.json"; _atomic_json(epath, dict(edge_probabilities=pilot.catchup._all_edges(table.nodes, exact.edge_probs)))
    settings = config(); settings.update(seeds=1, particles=[16], ranks=[ranks], scaling_particles=16,
        scaling_num_betas=3, mcmc_seconds=[.1], reference_ranks=ranks,
        reference_seconds=.15, reference_min_stored_per_chain=1)
    d = dict(key="tiny", nodes=table.nodes, truth_edges=[["x", "y"]], max_parents=1)
    common = dict(dataset_key="tiny", arm="dimension", ranks=ranks, seed_index=0, reference=False)
    tasks = [dict(**common, task_id="tiny-smc", method="ordersmc", particles=16, purpose="fixed-work-scaling"),
             dict(**common, task_id="tiny-mcmc", method="order-mcmc", wall_seconds=.1, purpose="accuracy-time")]
    references = [dict(common, task_id="tiny-reference", method="order-mcmc", wall_seconds=.15, reference=True)]
    plan = dict(config=settings, datasets=[d], tasks=tasks, references=references,
        bundles=[dict(dataset_key="tiny", seed_index=0, tasks=[0, 1])], source_files={})
    plan["plan_sha256"] = canonical_sha(plan); _atomic_json(root / "plan.json", plan)
    _atomic_json(root / "validation/preflight.json", dict(all_passed=True, plan_sha256=plan["plan_sha256"]))
    _atomic_json(root / "prepared/tiny/prepared.json", dict(status="complete", plan_sha256=plan["plan_sha256"],
        cases={"dimension": dict(config=cfg.to_dict(), candidate=artifact(cpath),
            candidate_sha256=candidate_table_content_sha256(table), exact=artifact(epath))},
        shared_preparation_seconds=.01, hardware=pilot.hardware()))
    return read(root / "plan.json")


def test_design_counts_targets_and_randomized_schedule():
    settings = config(); datasets = [dict(key=f"d{n}", nodes=[str(i) for i in range(n)]) for n in (12, 56, 196)]
    tasks, bundles, refs = pilot.make_tasks(settings, datasets)
    assert len(tasks) == 180 and len(bundles) == 9 and len(refs) == 8
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    assert sorted(i for b in bundles for i in b["tasks"]) == list(range(180))
    assert pilot.make_tasks(settings, datasets) == (tasks, bundles, refs)
    assert all(t["ranks"] == 64 for t in tasks if t["arm"] == "fixed")
    assert sum(t["purpose"] == "fixed-work-scaling" for t in tasks) == 36
    for seed in range(3):
        evaluation = pilot.seed_record(settings, "d56", seed, "evaluation-mcmc-rank0")
        reference = pilot.seed_record(settings, "d56", seed, "reference-mcmc-rank0")
        assert evaluation["seed"] != reference["seed"]


def test_rhat_detects_location_scale_and_drift_failures():
    rng = np.random.default_rng(381)
    x = rng.normal(size=(8, 2000))
    assert pilot.split_rank_rhat(x) < 1.01
    assert pilot.split_rank_rhat(x + np.arange(8)[:, None]) > 1.1
    assert pilot.split_rank_rhat(x * np.exp(np.arange(8))[:, None]) > 1.1
    assert pilot.split_rank_rhat(x + np.linspace(0, 8, 2000)) > 1.1
    assert pilot.split_rank_rhat(np.ones((2, 12))) == 1.
    assert np.isinf(pilot.split_rank_rhat(np.array([[0] * 12, [1] * 12])))
    assert np.isinf(pilot.split_rank_rhat(np.ones((1, 12))))


def test_chain_pooling_does_not_weight_by_stored_draw_count():
    records = [dict(pips={("x", "y"): .2}, stored_samples=100), dict(pips={("x", "y"): .8}, stored_samples=1000)]
    pooled = pilot.pooled_chains(records, ["x", "y"])
    assert pooled["x", "y"] == pytest.approx(.5)
    assert pooled["y", "x"] == 0.


def test_sparse_zero_edges_in_timing_error_report():
    reference = {("X000", "X004"): 0., ("a", "b"): .7}
    assert report.maximum_pip_error({("a", "b"): .6}, reference) == pytest.approx(.1)
    assert report.maximum_pip_error({}, reference) == pytest.approx(.7)


def crossing_fixture():
    rows = []
    for method in ("ordersmc", "order-mcmc"):
        for seed in range(3):
            rows.append(dict(dataset_key="tiny", arm="dimension", ranks=8, purpose="accuracy-time",
                method=method, budget=512 if method == "ordersmc" else 120, seed_index=seed,
                reference_usable=True, pip_rmse=.009, inference_seconds=10 if method == "ordersmc" else 20))
    return pd.DataFrame(rows)


def test_speedup_requires_all_seeds_and_a_usable_reference():
    frame = crossing_fixture(); row = report.quality_crossings(frame, [.01], 3).iloc[0]
    assert row.speedup == 2 and row.status == "both_reached"
    frame.loc[0, "pip_rmse"] = .02
    assert report.quality_crossings(frame, [.01], 3).iloc[0].speedup is None
    frame = crossing_fixture(); frame.loc[0, "reference_usable"] = False
    assert report.quality_crossings(frame, [.01], 3).iloc[0].speedup is None
    frame = crossing_fixture().iloc[1:]
    assert report.quality_crossings(frame, [.01], 3).iloc[0].speedup is None


def test_scaling_requires_same_hardware_and_work():
    frame = pd.DataFrame([dict(dataset_key="tiny", arm="dimension", purpose="fixed-work-scaling", seed_index=0,
        ranks=r, particles=128, inference_seconds=64/r, host="node", cpu_model="same", proposed_moves=39680,
        stages=31, pip_rmse=.01, reference_usable=True) for r in (1, 8, 64)])
    out = report.scaling_rows(frame, config()); assert list(out.speedup) == [1, 8, 64]
    frame.loc[1, "proposed_moves"] += 1
    with pytest.raises(RuntimeError, match="proposal count"): report.scaling_rows(frame, config())


def test_reference_gate_rejects_disagreeing_groups():
    rng = np.random.default_rng(126)
    def records(offset):
        p = .4 + offset
        edge_rows = [dict(parent="a", child="b", probability=p), dict(parent="b", child="a", probability=.2)]
        chains = [dict(stored_samples=1000) for _ in range(8)]
        diagnostics = [dict(monitored_edges=[["a", "b"]],
            trace=np.column_stack([rng.normal(size=1000), p + .02 * rng.normal(size=1000)]).tolist(),
            half_pips=[edge_rows, edge_rows]) for _ in chains]
        return dict(posterior=dict(edge_probabilities=edge_rows), chains=chains, reference_diagnostics=diagnostics)
    left, right = records(0), records(0)
    _, gate = report.reference_check([left, right], config(), ["a", "b"])
    assert gate["usable"] and not gate["exact"] and not gate["convergence_proved"]
    _, gate = report.reference_check([left, records(.2)], config(), ["a", "b"])
    assert not gate["usable"] and gate["group_max_abs_difference"] > .05


def test_real_single_rank_worker_and_report(tmp_path):
    from mpi4py import MPI
    plan = tiny_timing_plan(tmp_path)
    for index in range(2):
        pilot.run(tmp_path, index, MPI.COMM_SELF)
        task = plan["tasks"][index]
        path = tmp_path / "runs" / task["task_id"] / "result.json"
        before = path.read_bytes(); pilot.run(tmp_path, index, MPI.COMM_SELF); assert path.read_bytes() == before
        _atomic_json(tmp_path / "attempts" / (task["task_id"] + ".json"), dict(status="completed", task=task,
            plan_sha256=plan["plan_sha256"], launcher_wall_seconds=.5, reserved_core_hours=.001))
    _atomic_json(tmp_path / "bundles/0.json", dict(status="attempted_all", plan_sha256=plan["plan_sha256"]))
    report.aggregate(tmp_path)
    done = read(tmp_path / "analysis/completion.json")
    assert done["succeeded"] == 2 and not done["publication_ready"]
    assert len(list((tmp_path / "analysis/figures").glob("*.pdf"))) == 5
    assert read(tmp_path / "runs/tiny-smc/result.json")["inference_seconds"] > 0


def test_scheduler_resumes_and_does_not_launch_before_preflight(tmp_path):
    plan = tiny_timing_plan(tmp_path); calls = []
    def sbatch(args, **kwargs):
        calls.append(args)
        if len(calls) == 2: raise RuntimeError("submission limit")
        return "123\n"
    with pytest.raises(RuntimeError, match="submission limit"):
        timing_submit.submit(tmp_path, "start", sbatch=sbatch)
    assert read(tmp_path / "submission.json")["jobs"] == {"preflight": "123"}
    resumed = []
    timing_submit.submit(tmp_path, "start", sbatch=lambda args, **kw: resumed.append(args) or "124\n")
    assert len(resumed) == 1 and "--dependency=afterany:123" in resumed[0]
