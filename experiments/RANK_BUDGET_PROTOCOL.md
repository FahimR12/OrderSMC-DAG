# Fixed-data compute-budget comparison — 2026-09-17

## Question and scope

Does increasing OrderSMC computation improve same-data reproducibility under
the experimental rank-native score? This is a computational follow-up, not
independent validation of recovery or a demonstration of MPI speedup.

Use **all eight** existing n=5000 rank-native cases: four Gaussian and four
Student-t(df=3) noise cases, sharing four replicate seed blocks. No case is
selected or excluded by its recovery or repeatability. Do not duplicate the
monotone-transformed variants: their local scores and exact posteriors were
identical in the completed source comparison, and equality of the full
candidate targets is rechecked before freezing this experiment.

Original source: `results/rank-validation/20260917-v1/`. Both budgets reuse
byte-checked copies of the **same** saved candidate table, observations and
exact reference per case. Exact-order probabilities are independently
recomputed at preparation and checked against the saved reference. The core
sampler/model implementation must still match its previous frozen source.

## Fixed comparison

| Setting | Baseline | Extended |
|---|---:|---:|
| Particles | 4,096 | 16,384 |
| Temperature points | 100 | 100 |
| Ordinary MH sweeps | 10 | 20 |
| Post-resampling MH sweeps | 25 | 50 |
| Inference seeds per dataset | 20 | 20 |
| MPI ranks per fit | 64 | 64 |

Eight datasets × two budgets × twenty seeds = **320 fits**. These are not 320
independent datasets. Seed IDs are paired between budgets and use the fresh
master seed 20260920; changing particle count changes random-number consumption,
so identical seed IDs do not imply identical particle trajectories. Both arms
are rerun for twenty seeds: comparing the old eight-seed identity rate directly
with a twenty-seed rate would be misleading.

Score, candidate support, indegree cap, graph prior, temperature schedule,
proposal mixture, resampling rule and representative-graph construction remain
unchanged. The comparison changes particles and MH sweeps **together**; it cannot
attribute an effect to either one separately. No recovery-dependent tuning or
automatic promotion is performed. This budget experiment holds n fixed and
does not test the effect of collecting more observations.

## Prespecified output

For each dataset and budget report all-twenty-seed identity, pairwise Jaccard,
number of distinct directed DAGs and modal-DAG frequency. Report skeleton and
the existing fixed-0.90 uncertainty display separately. Compare PIP error to
the unchanged exact-order target and PIP disagreement across seeds. The inherited
sampling-error flags are diagnostics, not success criteria for reproducibility.

For every unordered node pair, export exact forward/reverse/absent probabilities,
selection counts across seeds and the across-seed PIP standard deviation.
This makes it possible to distinguish a reproducible well-supported direction
from consistent selection of one side of a near tie. Agreement with the exact
representative is descriptive: another Bayes action may have almost the same
loss, and neither one is guaranteed to equal biological truth.

Retain directed/skeleton F1, AUPRC and Brier; seed-average within each dataset
before between-dataset comparisons. The primary contrasts are paired extended
minus baseline per dataset; no independent-replicate claim is made for seeds or
paired noise variants. Better reproducibility must not be substituted for better
recovery or scientific confidence.

## Execution and safety

From the repository root:

```bash
PROFILE=smoke bash scripts/submit_rank_budget.sh
# Once the smoke finishes successfully:
bash scripts/submit_rank_budget.sh
```

The environment is activated in both the submitter and workers. Pilot uses
64 array tasks, five fits per task, capped at four concurrent tasks. Each task
has a 45-minute limit. Results are persisted after every fit; complete matching
records can be reused on explicit task resubmission. Existing incompatible
records are rejected. No source results are overwritten. `afterany` aggregation
reports missing results instead of hanging in `DependencyNeverSatisfied`.

Outputs: `results/rank-budget/20260917-pilot-v1/`. The plan includes hashed code
and immutable input copies; `submission.json` records each submitted job ID.
`analysis/completion.json` is authoritative for completion. The final tables are
`stability.csv`, `edge_stability.csv`, `paired_extended_minus_baseline.csv`,
`dataset_means.csv` and `recovery_summary.csv`. No metadata runs start automatically.
