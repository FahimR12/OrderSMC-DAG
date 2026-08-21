# OrderSMC-DAG

**Scalable Bayesian structure learning for mixed-data directed acyclic graphs.**

OrderSMC-DAG samples topological orderings with sequential Monte Carlo (SMC),
then analytically averages over parent sets compatible with each ordering. It
supports mixed binary, categorical, and continuous data, pair-rescue candidate
screening, reusable candidate tables, MPI execution, multi-seed consensus, and
edge-confidence diagnostics.

The implementation is self-contained and has no runtime dependency on ATLAS.

## Installation

```bash
cd /users/psedrou/fastscratch/OrderSMC-DAG
python -m pip install -e '.[mpi,test]'
```

Check the installation with `ordersmc --version` and `pytest`.

## Recommended workflow

Inspect the data and write editable metadata:

```bash
ordersmc inspect data.csv \
    --metadata-out analysis/metadata.json \
    --encoded-output analysis/data_encoded.csv
```

Review the inferred types, then run a quick analysis:

```bash
ordersmc fit analysis/data_encoded.csv \
    --metadata analysis/metadata.json \
    --preset quick \
    --output analysis/quick
```

For a final run, checkpoint the expensive parent-set table and sample from it:

```bash
ordersmc build-table analysis/data_encoded.csv \
    --metadata analysis/metadata.json \
    --config examples/config.standard.yaml \
    --backend mpi \
    --output analysis/candidate_table.pkl.gz

ordersmc sample analysis/candidate_table.pkl.gz \
    --config examples/config.standard.yaml \
    --backend mpi \
    --output analysis/final
```

Summarize results or estimate stability across seeds:

```bash
ordersmc summarize analysis/final --top 20

ordersmc multiseed analysis/data_encoded.csv \
    --metadata analysis/metadata.json \
    --config examples/config.standard.yaml \
    --seeds 200,201,202,203,204 \
    --backend mpi \
    --output analysis/consensus
```

Under `mpiexec`, `srun`, or `mpirun`, `--backend auto` selects MPI
automatically. See `scripts/run_csv_mpi.slurm` for a cluster template.

## Python API

```python
import pandas as pd
from ordersmc import OrderSMCConfig, fit, infer_metadata_and_encode

raw = pd.read_csv("data.csv")
data, variable_types, cardinalities = infer_metadata_and_encode(raw)
config = OrderSMCConfig.preset("standard", seed=202)

result = fit(data, variable_types, cardinalities, config=config)
print(result.summary())
print(result.edge_table().head(20))

best_graph = result.best_dag()
posterior_graph = result.posterior_dag(threshold=0.7)
result.save("results/seed_202", threshold=0.7)
```

Use `backend="mpi"` and an MPI communicator with the same `fit` function.
On MPI root, `result.to_local_result()` provides the common result interface.

## Configuration

Three presets are included: `quick` for pipeline checks, `standard` for the
BPGbio pair-rescue configuration, and `thorough` for larger final runs. JSON
and YAML files use the same names as `OrderSMCConfig`; CLI flags override file
values. Every result records the effective configuration, data fingerprint,
backend, seed, and runtime.

Variable types are `bin`, `cat`, or `con`. Discrete columns must use
contiguous integer codes beginning at zero. Automatic inference is useful for an
initial audit, but final analyses should use reviewed explicit metadata.

## Edge confidence

Every run writes `edge_confidence.csv` with:

- `PosteriorProbability`: posterior inclusion probability for the directed edge.
- `ReverseProbability`: posterior probability of the opposite direction.
- `SkeletonProbability`: probability of connection in either direction.
- `OrientationConfidence`: direction support conditional on a connection.
- `MonteCarloSE`: approximate uncertainty from the finite weighted particle set.
- `MC95Lower` and `MC95Upper`: approximate Monte Carlo error interval.
- `BinaryEntropy`: edge-presence uncertainty, highest near probability 0.5.
- `Confidence`: a readable confidence category.

The Monte Carlo interval measures sampler precision; it is not a Bayesian
credible interval for an effect size. Multi-seed output adds posterior standard
deviation, minimum/maximum support, and selection frequency.

## Saved outputs

A result directory includes `best_edges.csv`, `posterior_dag.csv`,
`edge_confidence.csv`, `smc_diagnostics.csv`, `summary.json`, `run.json`,
configuration and metadata, plus NumPy particle arrays. Diagnostics include ESS,
resampling, MH acceptance, runtime, and warnings for particle collapse or poor
rejuvenation.

Candidate tables can be saved and reused across seeds. Only load candidate files
from trusted sources because their compressed representation uses Python pickle.

## Development

```bash
pytest
PYTHONPATH=src mpirun -n 2 python tests/mpi_smoke.py
```

This is research software. Screening widths, structural penalties, parent
limits, and posterior thresholds should be validated with realistic simulations
and multiple independent seeds.
