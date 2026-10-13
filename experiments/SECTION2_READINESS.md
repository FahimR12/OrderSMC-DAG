# Pre-metadata readiness work — 2026-09-15

## Current next step — 2026-09-17

The optional `rank-native` scoring candidate and a frozen native-versus-rank
development comparison are now implemented; see `../docs/RANK_NATIVE.md` and
`configs/rank_validation_v1.json`. This is an actual model-robustness experiment,
not another re-analysis of the same SMC budget. The native default and all prior
outputs remain unchanged. The planned fresh pilot uses 8 paired SMC seeds and
64 MPI ranks per fit, with Gaussian and heavy-tailed mixed mechanisms.

Progress and job IDs belong in `results/rank-validation/20260917-v1/`; completion
must be established from `analysis/completion.json`, not this implementation note.
The metadata study remains blocked on evaluated model suitability and generator
calibration. A rank transform is not claimed to fix every non-Gaussian mechanism.

## Scope and status

The rank-native development comparison subsequently completed: all 216
conditions and 768 SMC fits, with its source checks and sampling flags passing.
Recovery improved in several important settings, but exact directed-DAG
repeatability did not uniformly improve. The next bounded computational test
is documented in `RANK_BUDGET_PROTOCOL.md`: all eight n=5000 rank-native targets,
two fixed budgets, twenty new paired seeds. It leaves the data, score and prior
unchanged and does not launch the metadata study.

These checks follow the completed restricted-CLG corrective validation. They
do not use participant-level living-brain data and do not launch the six-profile
metadata recovery study. No prior, score hyperparameter, sampler setting or
dataset is selected by the recovery results of these checks.

1. **Score transition: complete.** Ten independent eight-node mixed graphs,
   two paired variants, n=250/1000/5000, four SMC seeds: 60 datasets and 240 fits.
   Array `10529700` and aggregation `10529701` completed successfully: all
   240 fits and 60 conditions, no missing results. All six sampling flags pass.
   Output: `results/section2-readiness/seed-20260918-v2`.
2. **Partition MCMC own-target audit: complete.** All 60 corrected CLG datasets,
   four seeds per method, exact structure and order references. Original output:
   `results/section2-readiness/seed-20260918-v1/audits` (job `10529370`).
3. **Protein-pathway calibration audit: complete.** Fifty size-222 samples
   conditional on each of five frozen M1 mechanisms; same audit output folder.
4. **Publication figures and evidence: updated.** See
   `results/publication/2026-09-15-corrected/README.md` and
   `results/publication/CURRENT.md`. Earlier evidence is preserved.
5. **M4 protocol correction: implemented, not yet run.** See the Version 2
   metadata design/execution configurations and `SECTION2_METADATA_PROTOCOL.md`.
6. **Non-Gaussian failure and uncertainty follow-up: complete.** See
   `results/publication/2026-09-15-mixed-followup/README.md`. Original inference
   is unchanged; this is a separately labelled post-hoc diagnostic addendum.

## Score-transition design

The first variant is the generic unrestricted mixed generator, with continuous,
binary and categorical variables. The second applies the fixed invertible map
`sinh(1.25 * asinh(x) + 0.5)` to each continuous column; discrete columns are
unchanged. This preserves the graph's conditional-independence structure but
deliberately tests misspecification of the parametric local families. It is not
a matched-likelihood guarantee. Generate the largest sample once; smaller
sample sizes use its row prefixes, preserving the graph and mechanisms.

Both variants use the **native unrestricted score**, an edge-penalty lambda of
1, maximum indegree 2, and every candidate parent family within that bound.
There is no screening-recall ambiguity in this tiny-network check. SMC uses
4096 particles, 100 temperatures, 10 ordinary moves and 25 post-resampling
moves. Every fit uses 64 MPI ranks, up to four fits concurrently; this is not
an MPI scaling comparison. The scientific seed is 20260918.

For each dataset, compute exact order and exact structure edge probabilities,
as well as four SMC posteriors. Report directed and skeleton recovery, AUPRC,
Brier, pair-state expected SHD, own-target PIP RMSE and seed reproducibility.
Representative graphs are certified constrained Bayes actions under pair-state
Hamming loss. Seed-level recovery is averaged within each dataset, then across
the ten independent graphs. Seeds are not independent experimental replicates.
The exact references distinguish sampling error from score/prior/model effects.

SMC mean/P95 PIP-RMSE flags of 0.012/0.020 are inherited diagnostic thresholds,
not recovery-based inclusion criteria. All poor outcomes and deterioration
with n remain reportable. Do not turn exploratory findings into untouched
confirmatory claims; freeze a fresh design/seed before confirmation.

## Completed audit findings

Partition MCMC own-structure-target PIP RMSE averages 0.406, 0.461 and 0.466
at the three sample sizes. The discrepancy persists when its target is matched.
Low acceptance and across-seed variation support poor exploration by the
current split/merge implementation and budget, not a general rejection of the
published method. OrderSMC's own-order-target errors are approximately 0.00151,
0.000751 and 0.0000530 on the completed CLG cohort. Structure and order edge
penalties are both 1 in that source execution configuration.

The new exact reference sums unique source-layer partitions using positive
log-space arithmetic, counting each DAG once. It agrees with the existing
sink inclusion–exclusion reference on 40 datasets (maximum PIP RMSE
5.4e-13). On the remaining 20 n=5000 datasets, the old reference explicitly
rejects an unrepresentable family-weight range; the positive reference remains
usable. Tiny fixture enumeration and extreme-weight tests also pass.

The calibration audit shows prevalence averaging 0.593 against 0.595, and
age–PD correlation 0.306 against 0.347, with all five conditional means within
the existing tolerances. Strong biological correlations are persistently high:
absolute-correlation Q95 averages 0.523 against 0.433 (two of five systematic
tolerance misses). Mean absolute biological–PD correlation averages 0.238
against 0.193 (one of five systematic misses). Full per-graph, per-metric and
per-repeat outputs are retained. Do not describe the generator as an exact
metadata match or freeze it for confirmation yet.

These predictive intervals are conditional on the five simulated mechanisms;
they do not quantify uncertainty in the uploaded real-data targets. The female
prevalence assumption remains 0.5 because that aggregate count was not supplied.

## Operational provenance and reruns

The initial score-transition array `10529369` was stopped, and its dependent
summary `10529371` cancelled, after a summary-wrapper boundary failure. One
exact PIP was 1.00000000000038; similar round-off can arise in MPI reductions.
The wrapper now rejects substantive violations, but clips boundary errors of
at most 1e-8 and normalizes equally small opposite-edge-sum overflow. Regression
tests reject NaN, infinity, genuine out-of-range values and incompatible edge
probabilities. This changes neither the score nor the SMC target or draws.
Initial partial results and the original frozen code remain under `v1`.

The rerun uses the same data seeds and scientific configuration in a new `v2`
folder. Workers execute hashed source snapshots, so later working-tree edits
cannot silently alter the running study. The successful audits were not rerun.

Start a new complete diagnostic with `bash scripts/submit_section2_readiness.sh`.
An active-job guard prevents duplicate submission; a changed source/config
requires a fresh `OUTPUT_ROOT`. Set `RUN_AUDITS=0` only when deliberately reusing
the already completed audits. Aggregation uses `afterany` and explicitly reports
missing datasets instead of presenting a partially failed cohort as complete.

The full metadata study remains locked. Future smoke/pilot preparation uses
`scripts/submit_section2_prepare.sh`, Version 2 configuration, and the new
`results/section2-v2/` output namespace. M4 now disables nonlinear quantile
transforms and applies `clg-bic` only when both mechanisms and directions are
CLG-compatible. The old non-Gaussian M4 preparation is a historical artifact.

## Completed follow-up findings

The failing non-Gaussian replicate 7 is not explained by SMC approximation or
by its order prior: at n=5000, local evidence favors its wrong representative
over truth by 157.414 log units, while the edge prior and order multiplicity
favor truth by 1.000 and 0.693. All 83 targeted score-arithmetic checks pass;
known-generator inversion recovers native-data exact PIPs to below 7e-13 RMSE.
The evidence supports score-model mismatch under the nonlinear transform;
it does not establish that Laplace integral error is exactly zero.

Optional direction masking at conditional probability 0.90 improves identical
four-seed graph displays from 45/60 to 54/60 conditions while preserving every
selected adjacency and all original metrics. Both alternative cutoffs (0.80,
0.95), direction coverage and retained-arrow accuracy are reported. This is
not a guarantee of reproducibility or correctness: the failing n=5000 graph
still has 11 wrong arrows among 12 retained arrows. A general non-Gaussian
modeling change and the metadata calibration discrepancy remain future work.
