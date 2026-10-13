"""Official DAG-GFlowNet, with shared scores and the structure-modular target.

The published network and transition rules are unchanged. JAX and the upstream
checkout are optional dependencies, loaded only for integration/training.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import subprocess
import time

import networkx as nx
import numpy as np

UPSTREAM_REVISION = "53350bbfba3ab24f9ffa8196a3de42f139d9c5cc"
LocalScore = namedtuple("LocalScore", ["key", "score", "prior"])


class CandidateTableScore:
    """Upstream scorer protocol; no screening or arbitrary missing families.

    The official environment only masks cycles and indegree. Reject restricted
    pools rather than silently change support. Scores include the prior once.
    """
    def __init__(self, table, max_parents):
        self.num_variables = len(table.nodes)
        if not 1 <= max_parents < self.num_variables:
            raise ValueError("DAG-GFlowNet needs 1 <= max_parents < d")
        self.families = []
        for child in range(self.num_variables):
            eligible = [p for p in range(self.num_variables) if p != child]
            expected = {p for k in range(max_parents + 1) for p in combinations(eligible, k)}
            scores = {}
            for row, likelihood, prior in zip(table.parent_sets[child], table.loglik[child], table.logprior[child], strict=True):
                parents = tuple(sorted(int(p) for p in row if p >= 0))
                if parents in scores or not np.isfinite([likelihood, prior]).all():
                    raise ValueError("duplicate or non-finite family score")
                scores[parents] = LocalScore((child, parents), float(likelihood), float(prior))
            if set(scores) != expected:
                raise ValueError("DAG-GFlowNet needs an exhaustive unrestricted candidate pool")
            self.families.append(scores)

    def get_local_scores(self, target, indices, indices_after=None):
        before = self.families[target][tuple(indices)]
        if indices_after is None:
            return None, before
        return before, self.families[target][tuple(indices_after)]


def check_upstream(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise RuntimeError("unexpected DAG-GFlowNet upstream revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("DAG-GFlowNet tracked source has local modifications")
    import dag_gflownet
    if Path(dag_gflownet.__file__).resolve().parent.parent != path.resolve():
        raise RuntimeError("imported DAG-GFlowNet is not the declared checkout")


@dataclass(frozen=True)
class DAGGFNConfig:
    # Official train.py defaults, except a larger recorded posterior draw count.
    iterations: int = 100_000
    prefill: int = 1000
    replay_capacity: int = 100_000
    batch_size: int = 32
    num_envs: int = 8
    learning_rate: float = 1e-5
    delta: float = 1.0
    update_target_every: int = 1000
    min_exploration: float = 0.1
    posterior_samples: int = 16384
    seed: int = 0
    max_parents: int = 2


def train(table, config):
    """Mirror official train.py; return PIPs and training/resource diagnostics."""
    import jax
    import jax.numpy as jnp
    import optax
    from dag_gflownet.env import GFlowNetDAGEnv
    from dag_gflownet.gflownet import DAGGFlowNet
    from dag_gflownet.utils.replay_buffer import ReplayBuffer
    from dag_gflownet.utils.gflownet import posterior_estimate
    if (config.iterations < 2 or config.prefill < config.batch_size
            or config.replay_capacity < config.batch_size or config.posterior_samples < 1
            or not 0 <= config.seed < 2**32):
        raise ValueError("invalid training/sample/seed configuration")
    env = GFlowNetDAGEnv(config.num_envs, CandidateTableScore(table, config.max_parents),
                        max_parents=config.max_parents, num_workers=0)
    replay = ReplayBuffer(config.replay_capacity, num_variables=len(table.nodes))
    model = DAGGFlowNet(delta=config.delta, update_target_every=config.update_target_every)
    rng = np.random.default_rng(config.seed)
    key, subkey = jax.random.split(jax.random.PRNGKey(config.seed))
    started = time.perf_counter()
    params, state = model.init(subkey, optax.adam(config.learning_rate),
                              replay.dummy['adjacency'], replay.dummy['mask'])
    schedule = jax.jit(optax.linear_schedule(init_value=jnp.array(0.),
        end_value=jnp.array(1. - config.min_exploration), transition_steps=config.iterations // 2,
        transition_begin=config.prefill))
    indices = None
    observations = env.reset()
    losses, trace = [], []
    for iteration in range(config.prefill + config.iterations):
        actions, key, logs = model.act(params.online, key, observations, schedule(iteration))
        next_observations, deltas, dones, _ = env.step(np.asarray(actions))
        indices = replay.add(observations, actions, logs['is_exploration'], next_observations,
                             deltas, dones, prev_indices=indices)
        observations = next_observations
        if iteration >= config.prefill:
            params, state, logs = model.step(params, state, replay.sample(config.batch_size, rng=rng))
            loss = float(logs['loss'])
            if not np.isfinite(loss):
                raise RuntimeError("non-finite DAG-GFlowNet training loss")
            losses.append(loss)
            if len(losses) == 1 or len(losses) % 1000 == 0 or len(losses) == config.iterations:
                row = dict(updates=len(losses), loss=loss, elapsed_seconds=time.perf_counter()-started)
                trace.append(row)
                print(row, flush=True)
    training_seconds = time.perf_counter()-started
    started = time.perf_counter()
    posterior, _ = posterior_estimate(model, params.online, env, key,
                                      num_samples=config.posterior_samples, verbose=False)
    sampling_seconds = time.perf_counter()-started
    env.close()
    if posterior.shape != (config.posterior_samples, len(table.nodes), len(table.nodes)):
        raise RuntimeError("incorrect DAG-GFlowNet posterior shape")
    unique = np.unique(posterior, axis=0)
    for adjacency in unique:
        if (not nx.is_directed_acyclic_graph(nx.from_numpy_array(adjacency, create_using=nx.DiGraph))
                or np.max(adjacency.sum(axis=0)) > config.max_parents):
            raise RuntimeError("sample violates DAG/indegree support")
    probabilities = posterior.mean(axis=0)
    return probabilities, dict(config=asdict(config), upstream_revision=UPSTREAM_REVISION,
        target="exact-structure", training_seconds=training_seconds, sampling_seconds=sampling_seconds,
        completed_updates=len(losses), posterior_samples=len(posterior), loss_trace=trace,
        final_loss_mean=float(np.mean(losses[-1000:])), unique_sampled_graphs=int(len(unique)),
        # Conditional sampling noise, NOT uncertainty about training bias.
        maximum_conditional_sampling_mcse=float(np.sqrt(probabilities*(1-probabilities)/len(posterior)).max()),
        jax_backend=jax.default_backend(), devices=[str(device) for device in jax.devices()])


def validate_environment():
    """Exhaustively check all 25 three-node DAGs and all valid edge actions."""
    from dag_gflownet.env import GFlowNetDAGEnv
    from .adapter_validation import adapter_fixture_table
    from .structure_mcmc import enumerate_candidate_dags, dag_log_target, dag_state_graph, empty_dag_state
    table = adapter_fixture_table()
    env = GFlowNetDAGEnv(1, CandidateTableScore(table, 2), max_parents=2, num_workers=0)
    states = enumerate_candidate_dags(table)
    maximum, transitions = 0., 0
    empty = empty_dag_state(table)
    families = [[tuple(int(p) for p in row if p >= 0) for row in rows] for rows in table.parent_sets]
    for state in states:
        graph = dag_state_graph(state, table)
        observations = env.reset()
        for child in nx.topological_sort(graph):
            for parent in families[child][state[child]]:
                observations, _, _, _ = env.step(np.array([parent*3+child]))
        target = dag_log_target(state, table)-dag_log_target(empty, table)
        maximum = max(maximum, abs(float(observations['score'][0])-target))
        for parent in range(3):
            for child in range(3):
                legal = parent != child and not graph.has_edge(parent, child)
                legal = legal and graph.in_degree(child) < 2 and not nx.has_path(graph, child, parent)
                if bool(observations['mask'][0,parent,child]) != legal:
                    raise RuntimeError("DAG-GFlowNet mask mismatch")
                if legal:
                    before_family = families[child][state[child]]
                    after_family = tuple(sorted((*before_family,parent)))
                    after = list(state); after[child]=families[child].index(after_family)
                    expected = dag_log_target(tuple(after), table)-dag_log_target(state, table)
                    before_score, after_score = env.scorer.get_local_scores(child, before_family, after_family)
                    maximum = max(maximum, abs(expected-(after_score.score+after_score.prior-before_score.score-before_score.prior)))
                    transitions += 1
        reset, delta, done, _ = env.step(np.array([9]))
        if not done[0] or delta[0] != 0 or reset['adjacency'].any():
            raise RuntimeError("DAG-GFlowNet stop/reset mismatch")
    env.close()
    if maximum > 1e-12:
        raise RuntimeError("DAG-GFlowNet local reward mismatch")
    return dict(all_passed=True, dag_states=len(states), valid_edge_transitions=transitions,
                maximum_reward_error=maximum, target="exact-structure")


def main():
    """Bounded adapter/resource check, explicitly excluded from scientific results."""
    import argparse
    from ordersmc.persistence import load_candidate_table, candidate_table_content_sha256
    from .adapter_validation import adapter_fixture_table
    from .execution import _atomic_json
    from .exact_partition_log import exact_partition_log_dp
    from .readiness_audits import sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260925)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite an existing DAG-GFlowNet probe")
    check_upstream(args.upstream)
    validation = validate_environment()
    _atomic_json(args.output.with_suffix('.acceptance.json'), validation)
    if args.candidate is None:
        table = adapter_fixture_table()
    else:
        if not args.candidate_sha256:
            raise ValueError("candidate content hash is required")
        table = load_candidate_table(args.candidate, expected_sha256=args.candidate_sha256)
    config = DAGGFNConfig(iterations=args.iterations, prefill=32, posterior_samples=128, seed=args.seed)
    matrix, diagnostics = train(table, config)
    exact = exact_partition_log_dp(table)
    error = np.sqrt(np.mean([(matrix[i,j]-exact.edge_probs.get((u,v),0.))**2
        for i,u in enumerate(table.nodes) for j,v in enumerate(table.nodes) if i!=j]))
    _atomic_json(args.output, dict(status="complete", purpose="adapter-resource-probe-not-scientific-results",
        validation=validation, nodes=table.nodes, candidate_sha256=candidate_table_content_sha256(table),
        edge_probabilities=matrix.tolist(), own_target_pip_rmse=float(error), diagnostics=diagnostics,
        adapter_sha256=sha(Path(__file__))))


if __name__ == '__main__':
    main()
