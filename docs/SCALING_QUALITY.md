# Parallel SMC and recovery-versus-time follow-up

This is the primary computational story: adding cores accelerates SMC; separately,
test whether it reaches roughly comparable graph recovery to MCMC sooner. Neither
claim is assumed true. Model scores, candidate support, priors and core sampler
code are unchanged. This follow-up was designed after inspecting the first timing
pilot, so it is exploratory, not a preregistered confirmatory study.

## Frozen design

Submit with `bash scripts/submit_scaling_quality.sh`. The script activates the
project environment, freezes code/configuration and submits regression/MPI
preflight, candidate preparation, nine timing bundles, and the report. Default
root: `results/scaling-quality/v1`. Repeating the command does not duplicate
accepted jobs. Sources run from the snapshot. Maximum two bundles run at once.

Same predefined datasets as timing-pilot/v2: 12 variables/240 observations,
56/5,000, and protein-pathway metadata-based synthetic data with 196/222. No
original living-brain observations are used. There are three NEW inference seeds,
but only one dataset per size; seed ranges are not confidence intervals across
datasets. These are pilot figures, not evidence of broad generalization.

There are **243 attempts**, 27 per dataset/seed:

- Fixed-work SMC: 512 particles, 32 equally spaced temperatures, ten MH moves per
  particle at every stage, at 1/8/32/64 cores. Four fits per bundle. Total proposals
  must be 512 * 31 * 10 = 158,720 at every rank count. These settings are not
  asserted sufficient for accurate posteriors; quality is measured alongside speed.
- Adaptive SMC: 512 and 1,024 particles at 64 cores only. Same adaptive settings as
  the earlier pilot. These two fits are a separately labelled quality arm, not a
  substitute for the fixed-work scaling arm.
- MCMC: one independent chain per active core, 0.5/2/10/30/120-second sampling-loop
  budgets at all four core counts, plus 480 seconds at 64 cores. Twenty-one fits.
  Actual measured inference time includes posterior summaries and aggregation;
  nominal loop budgets are not runtime measurements. Very short budgets may give
  poor estimates; they are neither discarded for poor quality nor called converged.

All fits within a dataset/seed bundle run on the same exclusively reserved node in
a frozen randomized order. Each rank uses one thread and disjoint core affinity.
Report active-core hours and reserved-core hours: 64 active ranks still reserve a
168-core node. No queue delay is included in inference timing. Each step has a
four-hour cap; a bundle has eight hours. Failed/time-limited attempts remain in the
inventory and cannot qualify for speed ratios. Earlier one-hour adaptive SMC
timeouts remain unchanged; these new fixed-work fits do not replace those runs.

Primary prior remains lambda(d)=log(d-1), without changing package defaults. Native
local scores are retained at d=12/56 and rank-native at d=196. Previous lambda=1
sensitivity and long-reference results remain in timing-pilot/v2; they are not
pooled across priors or silently declared successful.

## Separate claims and measurements

1. Strong scaling: S(p)=T(1)/T(p), efficiency E(p)=S(p)/p, paired by dataset/seed.
   Identical particles, stage/proposal counts, hostname and CPU model are required.
   Figures show median and minimum/maximum over three seeds. Compare recovery and
   edge probabilities across core counts rather than assuming identical results.
2. Graph recovery: directed/skeleton F1, representative-graph SHD (one edit for a
   reversal), SHD divided by max(1, true edge count), and directed/skeleton AUPRC,
   all against known synthetic truth. This does NOT require converged MCMC.
   Low recovery of both methods is still low recovery, not a success by agreement.
3. Posterior quantities: exact edge-probability RMSE at d=12; elsewhere empirical
   disagreement with 64-core, 480-second MCMC. For both methods, agreement uses the
   mean of the OTHER two seeds' long-MCMC estimates to avoid comparing a chain
   with itself or its longer prefix. Report all-edge RMSE, informative-edge RMSE
   (benchmark probability between .05 and .95) and maximum difference. These are
   NOT posterior error or a convergence certificate. Existing larger-reference
   convergence failures remain explicitly relevant to any posterior-accuracy claim.
4. Reproducibility: pairwise representative-graph Jaccard and edge-probability RMSE
   across all three seeds at each complete setting. These assess repeatability,
   not correctness. Empty-graph agreement must be interpreted with recovery metrics.

The descriptive time-to-recovery table uses the mean truth recovery of ALL three
long-MCMC runs at 64 cores as its benchmark. Each of the three seeds of a candidate
budget must achieve directed AND skeleton F1 within 0.02 below this benchmark and
SHD/true-edge-count within 0.05 above it. Select the minimum median time among
qualifying tested budgets, separately for fixed/adaptive SMC and MCMC at the SAME
core allocation. Missing/failed repetitions never qualify. Ratios may favor MCMC;
missing crossings stay unestablished. This finite-grid, tolerance-based comparison
is not a formal noninferiority test, an optimal runtime, or matched posterior accuracy.
Margins were chosen for this exploratory follow-up, not retrospectively validated
on the earlier data. Report the benchmark's absolute recovery as well as the ratio.

## Outputs and earlier-pilot preview

`analysis/` contains per-run and grouped metrics, fixed-work scaling, reproducibility,
descriptive time-to-recovery, failed/missing attempts, input hashes and completion
metadata. Seven figure families, each PDF/PNG:

- `smc_strong_scaling`: speedup and efficiency.
- `smc_quality_vs_cores`: directed/skeleton F1 and PIP differences from serial SMC.
- `recovery_vs_time`, `recovery_vs_core_hours`, `recovery_vs_end_to_end`.
- `edge_probability_agreement`: explicitly labelled disagreement, NOT accuracy.
- `seed_reproducibility`.

Composed end-to-end cost adds shared serial candidate preparation to each measured
launch-to-saved-result time; it is not an independently timed cold-pipeline run.

The same reporter can read the unchanged earlier edition into a NEW directory:

```bash
source scripts/_common.sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m ordersmc_experiments.scaling_quality \
  --source results/timing-pilot/v2 --output results/scaling-quality/pilot-v2-preview-20260923 \
  --allow-partial
```

An incomplete preview is visibly labelled PARTIAL and cannot replace a final
report. Old 128-particle scaling probes retain their own labels and particle count;
they are never merged with the new 512-particle scaling results. The reporter
refuses to overwrite a prior report directory. No new long-reference fits are
launched by this follow-up, and their absence does not block truth-recovery figures.
