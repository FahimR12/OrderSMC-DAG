# Matched competitor follow-up — 2026-09-17

This closes the missing *head-to-head* comparison for the accepted rank-native
development model. It is not independent confirmation of a model that was chosen
after earlier failures. No participant-level living-brain data are used. The
existing protein-pathway pilot is separate and is not expanded by this launcher.

## Fixed cases, seeds and targets

Reuse all 24 unique fresh rank-native targets: four graph replicate blocks,
Gaussian and Student-t(df=3) mechanisms, n=250/1000/5000, d=8 and maximum indegree
two. Each is paired with a monotone-transformed measurement variant. Before
omitting duplicate computation, verify the two variants have identical parent
families, likelihoods and priors. They represent two measurement labels, not two
independent graph replicates. Observations, scores, true graphs, exact references
and completed SMC outputs are copied and hashed; no data are regenerated.

At n=250/1000 use eight seeds per stochastic method. At n=5000 use twenty seeds
per method/budget on all eight cases, reusing the completed SMC budget study.
Consequently n=5000 SMC recovery uses the **twenty-seed budget-study cohort**, not
the older eight-seed recovery average. Both SMC budgets remain reported. SMC
settings are not changed and no further SMC computation is required.

| Method | Score/support | Target or output | New fits |
|---|---|---|---:|
| Order MCMC | Same rank-native candidate table | Same order-modular posterior as SMC | 448 |
| REV MCMC | Same rank-native candidate table | Structure-modular posterior | 448 |
| Partition MCMC | Same rank-native candidate table | Structure-modular posterior; existing split/merge adapter | 448 |
| bnlearn HC | Shared rank-native local scores/prior | One point-estimate graph per case | 24 |
| bnlearn Tabu | Shared rank-native local scores/prior | One point-estimate graph per case | 24 |

Total **1392 new fits**, plus **448 reused SMC fits** (128 lower-n and 320 n=5000).
The three MCMC methods each have 288 baseline and 160 extended fits. This is not
1392 independent datasets. HC/Tabu are genuine R bnlearn searches using custom
family scores; do not label them native bnlearn mixed-data statistical models.
Their 0/1 edge outputs are not posterior-calibrated probability estimates.

Exact-order error is appropriate for Order MCMC/SMC; the separately checked
positive log-space exact structure reference is used for REV/Partition. Do not
interpret a target-prior difference as sampling failure. Prior edge lambda is
one throughout; the order-multiplicity factor remains in the order target.

## Fixed computation and failure reporting

MCMC baseline sampling is 300 seconds, extended sampling 1200 seconds at n=5000,
25% time burn-in, thinning 50 for Order MCMC and 10 for structure chains. The
iteration ceiling is 10^9, not an intended stopping target. Posterior-summary
time is measured separately from the sampling allocation where applicable.
Report completed iterations, retained/stored samples, acceptance and ESS.
Constant-chain ESS can be misleading: always inspect own-target PIP error and
between-seed disagreement. These kernels previously mixed poorly on some cases;
the longer allocation is a prespecified diagnostic, not tuning against recovery.
Do not generalize a failed kernel/budget to every implementation of its method.

Budget names align sensitivity panels, not equal wall-clock or core-second
allocations across SMC/MCMC. Existing SMC used 64 ranks; these chains/searches
use one core per independent fit. Concurrency accelerates the cohort; there is
no MPI speedup claim and no effort to parallelize a serial Markov transition.

The Slurm launcher uses at most 24 concurrent jobs, with up to five seeds per
task. It submits 240 primary tasks; only after their termination does a
controller submit the 96 extended tasks, reducing concurrent pending-job count.
`afterany` and explicit checks avoid unsatisfiable dependencies. Each successful
submission is immediately recorded. Each failed fit receives an explicit JSON
record and is neither silently retried nor omitted from completion counts.

## Reporting

- Per-fit directed/skeleton F1, AUPRC, Brier and expected pair-state SHD.
- Per-dataset means across seeds, then descriptive means across graph datasets.
- Separate mean-PIP-then-Bayes-action graph recovery, explicitly labelled so it
  is not confused with mean per-seed F1.
- Same-data directed, skeleton and fixed-0.90 uncertainty-display Jaccard and
  identity, distinct DAG count, modal frequency and PIP disagreement.
- Own-target PIP RMSE and fixed diagnostic flags (mean .012, p95 .020). These
  flags are not filters: flagged runs remain in recovery/stability summaries.
- Paired SMC-minus-comparator differences with dataset-level bootstrap intervals.
  Four graph blocks are a small development sample; inference seeds do not
  inflate the independent sample size. Noise variants remain paired.
- Partial outputs explicitly report missing/failed runs and group completeness;
  a section cannot be called complete from partial summary rows.

Run `bash scripts/submit_rank_competitors.sh`; environment activation and source
snapshotting are automatic. Outputs are under
`results/rank-competitors/20260917-pilot-v1/`; stage IDs are in `submission.json`,
logs in `code_snapshot/logs`, results in `runs` and tables in `analysis`.
`PROFILE=smoke` selects a short implementation check, not scientific results.

### Reporting amendment — 18 September 2026

The 912 primary fits completed, but sparse legacy SMC probability exports
caused the controller's report to fail before extended submission. The repair
expands omitted zero-probability edges over the complete directed-pair inventory
for mean-PIP graphs and between-seed RMSE. It does not alter inference, scoring,
priors, seeds, budgets, per-run recovery, or saved graphs. The existing frozen
worker snapshot remains unchanged; resumed extended fits use it. A separately
hashed reporting overlay, exact source diff, original-artifact hashes and
pre-repair analysis backup document the amendment. Current recovery tooling is
`ordersmc_experiments.reporting_recovery`; see the result-root README for
submitted job IDs and the overlay location. The final report must remain
incomplete until all 1392 new fits are accounted for without missing/failed fits.

## External-model coverage still requires separate validation

The official [DAG-GFlowNet implementation](https://github.com/tristandeleu/jax-dag-gflownet)
is pinned at `53350bbfba3ab24f9ffa8196a3de42f139d9c5cc`. The optional
`daggfn_adapter.py` score-table adapter compares inference under the **structure**
target, not an independent mixed likelihood. It rejects screened/restricted
candidate pools because the official environment masks only cycles/indegree.
Reward and mask checks passed on all 25 three-node DAGs and 48 valid edge
transitions (maximum reward error 5.56e-17). Training/sampling/resource checks
were completed by Slurm job 10560435, launched by
`bash scripts/submit_daggfn_probe.sh`. Its outputs and source manifest are in
`results/adapter-validation/daggfn-20260917-v1/`: 32 toy updates and 64 d=8
updates, followed by 128 valid DAG samples each. The d=8 check took 162.3 seconds
for initialization/training and 5.8 seconds for sampling on one CPU core.
These short checks are not converged scientific comparisons and must not
be included in recovery tables. A GPU execution/resource check and a frozen
scientific cohort remain to be done; available GPU partitions were identified,
but no GPU or full scientific DAG-GFlowNet jobs have been submitted. Upstream model,
optimizer and transition implementations are not edited. Dependencies are
pinned in `experiments/adapters/daggfn-requirements-py312.txt`, Python 3.12.10,
in the separate `ordersmc-daggfn` environment; the main SMC environment is unchanged.

The mixed-data [BCD Nets paper](https://proceedings.mlr.press/v323/mokhoro26a.html)
links [the official implementation](https://gitlab.com/moabi_mok/bcd-lim),
inspected at `24632202045adebeda3b07867e7a0bbea1ee39d4`.
Its mixed likelihood combines binary logistic terms with Laplace continuous
residuals, with optional intercepts. The paper discusses one-hot categorical
encoding, but the present benchmarks contain nominal variables with several
levels. A dummy-level graph is not automatically a valid original-variable DAG;
block aggregation can introduce cycles. Do not treat nominal codes as binary
values, discretize continuous data, silently regenerate easier datasets, or
claim a direct matched comparison without a declared and validated mapping.
Until that integration is resolved, record it as pending adapter/model-space
validation, not a failure, zero score or completed competitor result.

Native bnlearn BIC-CG results remain in the separately corrected, genuinely
Gaussian/CG-compatible panel. They cannot be substituted for unrestricted
rank-native mixed-data comparisons. All model editions stay distinct.
