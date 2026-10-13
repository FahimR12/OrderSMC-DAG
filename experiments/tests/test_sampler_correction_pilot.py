from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from ordersmc import OrderSMCConfig, build_candidates
from ordersmc.persistence import save_candidate_table, candidate_table_content_sha256
from ordersmc_experiments import sampler_corrections as pilot
from ordersmc_experiments.execution import _atomic_json
from ordersmc_experiments.readiness import canonical_sha
from ordersmc_experiments.seeds import protocol_seed_record


def miniature_source(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    (tmp_path / "pyproject.toml").write_text("# test fixture\n")
    (tmp_path / "README.md").write_text("test fixture\n")
    source = tmp_path / "source"
    dataset = dict(role="m1_matched", replicate=0, dataset_id="tiny")
    plan = dict(datasets=[dataset])
    plan["plan_sha256"] = canonical_sha(plan)
    _atomic_json(source / "plan.json", plan)
    _atomic_json(source / "analysis/completion.json", dict(status="complete", missing_or_failed_runs=0))
    data = pd.DataFrame(dict(x=np.linspace(-1, 1, 24), y=np.arange(24) % 2))
    cfg = OrderSMCConfig(screen_k=1, parent_pool_max=1, max_parents=1)
    table = build_candidates(data, dict(x="con", y="bin"), dict(x=0, y=2), config=cfg, verbose=False)
    directory = source / "datasets/tiny"
    save_candidate_table(table, directory / "candidate.json.gz")
    digest = candidate_table_content_sha256(table)
    _atomic_json(directory / "prepared.json", dict(candidate_table_sha256=digest))
    _atomic_json(directory / "generator_manifest.json", dict(edges=[["x", "y"]], variable_types=dict(x="con", y="bin")))
    for index in [0, 1]:
        seed = protocol_seed_record(173, experiment="correction-test", cell="tiny", replicate=0, component=f"seed{index}")
        _atomic_json(source / f"runs/tiny-ordersmc-evaluation-s{index:02d}.json", dict(
            status="complete", plan_sha256=plan["plan_sha256"], candidate_table_sha256=digest,
            run=dict(method_seed=asdict(seed)), runtime_seconds=0.,
            edge_probabilities=[dict(parent="x", child="y", probability=.5),dict(parent="y", child="x", probability=.1)],
        ))
    config = dict(source="source", source_plan_sha256=plan["plan_sha256"],
                  replicates=[0], seed_indices=[0, 1], mpi_ranks=1,
                  common=dict(particles=16, num_betas=3, mh_steps=1, mh_steps_resample=2),
                  arms=dict(proposal_only=dict(tempering_path="collapsed_likelihood", adaptive_tempering=False,
                                               final_target_snapshot_steps=[0]),
                            geometric_adaptive=dict(tempering_path="geometric", adaptive_tempering=True,
                                                    final_target_snapshot_steps=[0, 2])))
    path = tmp_path / "config.json"
    _atomic_json(path, config)
    return path


def test_paired_pipeline_preserves_source_and_fails_closed_on_missing_results(tmp_path, monkeypatch):
    from mpi4py import MPI
    config = miniature_source(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in (tmp_path / "source").rglob("*") if p.is_file()}
    output = tmp_path / "paired"
    plan = pilot.freeze(config, output)
    assert len(plan["tasks"]) == 4
    assert {t["config"]["seed"] for t in plan["tasks"] if t["arm"] == "proposal_only"} == {
        t["config"]["seed"] for t in plan["tasks"] if t["arm"] == "geometric_adaptive"}
    with pytest.raises(FileExistsError):
        pilot.freeze(config, output)
    with pytest.raises(RuntimeError, match="missing runs"):
        pilot.aggregate(output)
    assert not (output / "analysis/seed_stability.csv").exists()
    for index in range(len(plan["tasks"])):
        pilot.run(output, index, MPI.COMM_SELF)
    pilot.aggregate(output)
    stability = pd.read_csv(output / "analysis/seed_stability.csv")
    assert set(stability.arm) == {"proposal_only", "geometric_adaptive", "original_v06_global_swaps"}
    assert len(stability) == 4
    assert (stability.seeds == 2).all()
    assert all(Path(p).read_bytes() == value for p, value in before.items())
    (tmp_path / "README.md").write_text("changed\n")
    with pytest.raises(RuntimeError, match="source changed"):
        pilot.checked_plan(output)
