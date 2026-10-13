# Reporting repair and null/signal audit (development follow-up)

## Reporting-only correction

The V3 mixed report failed on the **Partition-MCMC mean-PIP graph** for
`v3-hailfinder-topology-mixed-n05000-rep011`, not on an SMC posterior fit.
HiGHS' excluded-incumbent audit exposed objective regret 1.224926216e-9,
above the existing 1e-9 guard. All 880 fits were complete.

Every MILP objective is now uniformly multiplied by exactly 2^20 before
solving, and returned objective diagnostics are converted back. Positive
uniform scaling changes neither the optimizer nor the lexicographic rule in
exact arithmetic. The original binary64-rational objective audit and 1e-9
guard remain unchanged; no near-tie band is widened and no edge-specific
perturbation is introduced. The small-domain exact oracle remains in place.
Large-domain MILP results remain floating-point solver results, not exact
rational optimality certificates. The real failure is retained as a test.

Relevant solver documentation: [HiGHS numerical considerations](https://ergo-code.github.io/HiGHS/dev/guide/numerics/)
and [SciPy MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

All six original catch-up cohorts are re-summarized into a **new edition**:
`results/reporting-null-audit/v1/reporting_repair/analysis/`.
Original source snapshots, fits and reports are preserved. Raw fit hashes are
checked before and after reporting. Every selected graph carries numerical
solver diagnostics; changes from previously available metrics are recorded.
The failed old report is historical and is not relabelled successful.

## Null diagnosis

Three independent empty-DAG datasets have 196 variables and 222 observations.
The recorded prior is `exp(-lambda * number_of_edges)` with lambda=1 and
maximum indegree two. It is an order-modular, candidate-restricted posterior,
not a uniform prior over DAGs and not a multiplicity-calibrated discovery rule.
The mean selected false-edge count was 22.17; extending final-target mixing did
not remove it. Shared-score hill climbing selected 46--56 edges. This does not
by itself establish a coding error or a predeclared failure threshold.

For complete parent support, the exact order-prior expected number of edges is

    sum_{m=0}^{d-1} [sum_{k=0}^{min(2,m)} k*C(m,k)*exp(-lambda*k)]
                     / [sum_{k=0}^{min(2,m)} C(m,k)*exp(-lambda*k)].

At d=196, lambda=1 gives approximately 364.32 prior edges. This is an
illustration, **not** the actual screened-table prior expectation. Screening
changes support, is data-dependent, and is held fixed in the new sensitivity.
The Gaussian regression marginal is independently checked against a
multivariate Student-t density. The audit also recomputes stored local scores
for empty families and families implicated in false positives, checks prior
potentials, and compares pair-state versus directed-edge loss summaries.
These checks cannot alone establish calibration of the whole mixed model.

## Fixed two-prior diagnostic

Compare the existing lambda=1 with one globally specified dimension rule:

    lambda(d) = log(d-1).

The latter corresponds to edge odds 1/(d-1), or probability 1/d before the
indegree, ordering and candidate-support restrictions. No claim is made that
this is the actual marginal edge probability under the restricted prior.
This is a **model sensitivity**, not a correction to the SMC kernel or an
automatically promoted default. No parameters are selected from graph recovery.

| Input | Datasets | Priors | Seeds | New fits | Sampling |
|---|---:|---:|---:|---:|---|
| Fresh 8-variable null/signal pairs, n=222 | 6 | 2 | 4 | 48 | 512 particles, serial |
| Existing 196-variable metadata null/signal, n=222 | 6 | 2 | 4 | 48 | 1,024 particles, 64 MPI ranks |

The fresh data include continuous, binary and three-category variables. Each
small posterior has an exact order-DP reference and exact prior expectation.
The larger study uses **all three empty controls** and **M1 replicates 0--2**,
selected by index, not results. Both priors use the same retained parent
families, bit-identical local likelihoods, seeds and sampling settings.
Audited prior-retargeting provenance records that support was not rebuilt.

All fits retain snapshots before and after 250 final-target MH proposals per
particle. Budgets are deliberately smaller than the publication cohorts;
these are diagnostic runs, not replacement publication results. Large runs
have four concurrent 64-rank fits; small ones have twelve concurrent serial
fits. No MPI speedup experiment is being run.

Report false/true edges, recovery, seed stability and exact-posterior RMSE
where available. On empty graphs, F1/AUPRC are not informative; use false-edge
counts and posterior edge mass instead. Report any loss of true-edge recovery.
If a new prior is adopted, its corresponding model comparisons must be rerun
for all relevant algorithms and verified on untouched data. Do not reuse
old-prior competitor fits as if they targeted the new posterior.

## Execution and outputs

`bash scripts/submit_report_null_audit.sh` freezes both workflows, activates the
project environment and submits preflight-controlled, resumable jobs.
`PREPARE_ONLY=1` freezes without submission. Full tests and a four-rank MPI smoke
check precede each workflow. `afterany` controllers reject failed prerequisites.

The new prior pilot is under `results/reporting-null-audit/v1/null_pilot/`.
Its `analysis/` contains individual and dataset-averaged metrics, seed stability
and an explicit completion record. Numerical reporting repairs live separately
under `reporting_repair/`. No old posterior or default prior is overwritten.
