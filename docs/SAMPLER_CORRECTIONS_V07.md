# v0.7 corrections and paired validation

This release addresses the mathematical/code audit of the metadata-matched
protein-pathway pilot. Existing scientific artifacts and their frozen source
trees remain unchanged. This document describes corrections, not evidence
that the large-data reproducibility problem has already been solved.

## Corrections

1. **MPI proposals:** the configured normalized mixture is now actually used,
   including adjacent swaps and optional segment reversals. Proposed and
   accepted counts are recorded by move type. Exact joint-mode ties use the
   same lexical rule as serial execution. Previously MPI always used global
   swaps while recording the default 75/25 mixture.
2. **Uniform-start bridge:** the new high-level default is adaptive geometric
   tempering, `log gamma_t(O) = t * log gamma_1(O)`. Its starting distribution
   is genuinely uniform and its endpoint is exactly the previous posterior.
   It does not divide away the ordering-induced prior, change screening,
   or retune local scores. Legacy collapsed-likelihood tempering remains
   explicitly available for controlled comparisons.
3. **Adaptive MPI:** one global conditional-ESS decision controls all ranks.
   Invalid progress is reported to every rank; it never silently jumps to the
   final target. MPI also supports final-target snapshots and extra MH moves.
   An MH step means one proposal per particle, not a full variable sweep.
4. **Evidence semantics:** the geometric bridge's initial normalizer is d!,
   not the old structural-prior normalizer. Its path ratio is labelled
   separately; prior-relative evidence is unavailable, not silently repurposed.
5. **Uncertainty:** within-run particle MCSE remains a labelled diagnostic,
   not a calibrated confidence interval. Unknown errors stay unknown.
   Independent-seed reports distinguish run-to-run SD from SE of the mean.
   They do not certify convergence or measure bias common to every run.
6. **Binary/categorical consistency:** a two-category softmax child now has
   the same Normal prior on log-odds coefficients as a binary-logit child.
   The Helmert coefficient variance is tau²/2 for K=2; it remains tau² for
   K>=3. This preserves category-label symmetry and leaves binary families
   unchanged. Local-score schema is incremented from 2 to 3.
7. **Graph interpretation:** `best_edges` and `best_dag()` are labelled visited
   joint graph/order modes, not marginal DAG MAP estimates. Paper graphs
   continue to use the globally constrained Bayes action for pair-state SHD.

The low-level sampler functions retain their legacy path defaults for API
compatibility. `OrderSMCConfig` / `sample` / `fit` default to the corrected
geometric/adaptive path. Historical execution configs without explicit path
fields retain their old path and schedule; use the new pilot below, not an
old submission script, to validate the correction.

## Reuse and migration

Candidate tables contain scores, so sampling corrections do not require
rebuilding unchanged scores. Provenance validation permits score-schema-2
tables only when the v3 correction cannot affect their scores: the unchanged
DG/CLG-BIC methods or complete metadata with no two-category categorical child.
An ambiguous or potentially affected native/rank-native table is rejected.
Old provenance is preserved verbatim; reuse is not relabelled as new scoring.

The current protein-pathway pilot has binary PD/female variables and continuous
remaining variables: its score tables can be reused exactly. If another cohort
has K=2 categorical children with continuous parents, rebuild its tables and
rerun **all** methods that share that score, not only OrderSMC. Old results stay
versioned as historical evidence. Old serial sampler checkpoints are not
silently upgraded to the new configuration.

## Paired pilot (prepared separately; not automatically submitted)

Configuration: `experiments/configs/sampler_correction_pilot_v1.json`.
It selects metadata M1 graph replicates 0 and 1, eight existing inference
seeds each, 4,096 particles, and 64 MPI ranks per fit. There are 32 new fits:

| Arm | Bridge and moves | Purpose |
|---|---|---|
| Original v0.6 | Existing fixed-path, global-only MPI results | Read-only baseline |
| Proposal-only | Original 100-temperature path, corrected 75/25 mixture | Isolate the proposal implementation correction |
| Geometric/adaptive | Same mixture and 10/25 proposals per stage; CESS 0.9 | Diagnose bridge correction |

The geometric arm reports beta-one snapshots after 0, 250 and 1,000 extra
proposals per particle. The zero-step snapshot separates bridge changes from
the extra final-target computation. All arms share the same observations,
candidate families, local-score arrays, structural prior and seed identities.
No graph truth is used by sampling or to choose schedules. Original runs,
input artifacts and executable source are content-addressed before submission.
Fresh results never overwrite the old pilot or publication folder.

This is not an equal-compute comparison or speedup study. Report every arm and
snapshot, and compare recovery, directed/skeleton Jaccard, PIP RMSE between
independent seeds, and maximum edge-wise seed SD. Averaging seeds can improve
the final graph but does not demonstrate single-run reproducibility. There is
no automatic winner selection or promotion to publication-ready evidence.

Prepare without submitting:

```bash
cd /users/psedrou/fastscratch/OrderSMC-DAG-paper
PREPARE_ONLY=1 bash scripts/submit_sampler_correction_pilot.sh
```

Submit the already-frozen pilot when ready:

```bash
bash scripts/submit_sampler_correction_pilot.sh
```

Default output: `results/sampler-corrections/v07-pilot-v1/`. Two MPI fits may run
concurrently. `MAX_MPI_CONCURRENT` adjusts concurrency only. The submission
script activates the project environment. The report runs after the array and
fails closed if any fit is missing. Inspect `submission.json` before restarting
after a partial submission; duplicate submissions are refused.

An implementation smoke plan uses `PROFILE=smoke` and a **different**
`OUTPUT_ROOT`; its 32-particle results are not scientific evidence.

## Validation

The full regression suite passed (434 tests). The adaptive four-rank and
legacy fixed-path two-rank integration checks passed. A bounded check on the
actual saved d=196, replicate-0 candidate table used 256 uniform orders:
initial ESS was 256, the first adaptive temperature was approximately
0.00108084, and its ESS was 230.4 (the requested 90%). This checked only the
initial bridge, not full posterior recovery or high-dimensional convergence.

Regression coverage includes endpoint/normalizer identities, adaptive CESS
guards, actual MPI proposal calls, proposal accounting, deterministic ties,
binary/two-category and label invariance, safe old-table reuse, geometric
checkpoint replay, uncertainty semantics, and joint-mode versus marginal-MAP
counterexamples. `tests/mpi_corrective_smoke.py` exercises real four-rank
execution, ranks with zero particles, snapshots, and collective failure.

```bash
source scripts/_common.sh
python -m pytest tests experiments/tests
mpiexec -n 4 python tests/mpi_corrective_smoke.py
```

Passing these tests validates implementation invariants, not high-dimensional
posterior convergence. The paired pilot remains necessary before claiming
better recovery or reproducibility.
