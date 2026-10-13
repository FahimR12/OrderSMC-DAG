# Posterior accuracy and parallel timing pilot

Run `bash scripts/submit_timing_pilot.sh`. The source/configuration freeze,
jobs and results are under `results/timing-pilot/v2`. The wrapper activates the
project environment. Repeating submission resumes recorded stages; it does
not silently duplicate accepted jobs or retry failed timings.

Edition v1 passed all 470 regression tests but its MPI launch was blocked by
conflicting CPU-per-task environment variables: Slurm allocated two CPUs per
task despite the one-CPU request and retained a cpu=1 task TRES.
No scientific fit started in v1. Edition v2 explicitly normalizes these variables
to one active CPU per rank after activation, while reporting the actual larger
reservation separately. All scientific settings are
unchanged; v1 sources, plan and failed-job logs remain historical.

## Scope and frozen targets

This is an implementation-level timing pilot, not a universal SMC-versus-MCMC
claim or a population-level confirmatory study. Use the first predeclared
replicate of three existing mixed datasets, selected by identity, not outcome:

| Variables | Observations | Data | Local score |
|---|---:|---|---|
| 12 | 240 | V1 balanced mixed, replicate 0 | native |
| 56 | 5,000 | Hailfinder-topology mixed, replicate 0 | native |
| 196 | 222 | Protein-pathway M1 synthetic, replicate 0 | rank-native |

Primary prior: lambda(d)=log(d-1). Sensitivity: lambda=1. Candidate support,
likelihoods and the order-modular posterior are identical between samplers
within every case. Native versus rank-native is a preserved *cohort difference*,
not a score selected after timing. Core sampler/model/package defaults remain
unchanged. Preserve all older fits with their original labels.

One dataset per size and three inference repetitions are sufficient to test
instrumentation and choose subsequent budgets, not to quantify generalization
across datasets. Full paper runs require more independently selected datasets.

## 180 timed attempts plus eight independent reference jobs

Three repetitions at 1, 8, 32 and 64 active physical cores. Every process has
one thread. MCMC uses one independently seeded Order-MCMC chain per rank and
equal-weights the chain posterior estimates; correlated stored draws are not
called independent samples. Both use the same order-target evaluation and
swap/adjacent-swap mixture. This benchmarks these validated implementations,
not every available optimized MCMC implementation.

- Main adaptive SMC: 512 particles at every core count, plus 1,024 at 64 cores;
  geometric bridge, CESS 0.9, 10/25 moves, final target beta=1, no extra final
  moves. These budgets are candidates to measure, not asserted adequate fits.
- MCMC: 30/120-second sampling-loop budgets at every core count, plus 480 seconds
  at 64 cores; first quarter burn-in, thinning five. Record actual elapsed time
  including Rao-Blackwell summaries, aggregation and communication. Nominal
  loop budgets are not treated as measured total runtime.
- Fixed-work SMC scaling probe, **separate from the main accuracy comparison**:
  128 particles, 32 equally spaced temperatures, 10 MH moves after every
  stage whether or not resampling occurs. Same proposal counts at all ranks;
  report measured accuracy too. This tests parallel execution, not the adequacy
  of a 128-particle posterior or scaling of every adaptive configuration.
- Old-prior sensitivity: 512-particle adaptive SMC and 120-second MCMC at 64
  cores, all three repetitions, on all three datasets.

For each dataset/repetition, a single exclusively allocated node executes the
20 steps in a frozen randomized order, preventing other jobs from sharing its
CPUs. Only two such bundles run concurrently. Lower-rank steps leave some
reserved cores idle: report active core-hours **and** reserved core-hours.
Do not describe a 64-rank allocation as a 64-physical-core node: this cluster
has 168 physical cores per node. CPU model, hostname, binding, software and
actual allocations are recorded. Scaling ratios require the same host and
identical realized proposal counts. A one-hour per-step cap is recorded as a
failed/time-limited observation; it never supplies an invented speedup.

## Posterior references and quality gates

At d=12, recompute exact order posteriors for both priors. At d=56/196, run two
independent groups of eight MCMC chains, each with a 1,800-second loop budget,
under each prior. Reference streams are disjoint from benchmark streams.

Large references are usable only if all these predeclared checks pass:

- At least 100 stored draws per chain and nonconstant target traces.
- Independent-group PIP RMSE <=0.005 and maximum edge difference <=0.05.
- Group mean first/second-half PIP drift <=0.005.
- Rank-normalized, folded split R-hat <=1.01 for log target and 64 edges chosen
  by pooled uncertainty within each reference group. All ranks in a group use
  the same monitored panel. Up to 1,000 chronologically ordered draws per
  chain are retained for this diagnostic; constants are counted explicitly.

These are finite diagnostics, **not proof of convergence**. Groups can share
an undiscovered mode. Failed references remain visible and block accuracy-based
speedup claims; their plots say "unvalidated reference". Exact-target accuracy
and sampling error must not be confused with graph-truth accuracy or causal
identifiability. See the rank-normalized R-hat definition in
https://mc-stan.org/posterior/reference/rhat.html.

Only beta=1 SMC estimates are evaluated; intermediate tempered targets are not
posterior convergence checkpoints. Report all-edge PIP RMSE, informative-edge
RMSE (reference PIP in [0.05,0.95]), maximum error and expected-edge-count error,
alongside directed/skeleton F1 and cross-seed graph Jaccard.

For each method/core count and RMSE tolerance 0.02/0.01/0.005, select the smallest
median runtime among tested budgets for which **all three repetitions** meet
the tolerance. Do not choose the best seed or interpolate between budgets.
Report the MCMC/SMC ratio only when both methods qualify under a usable reference.
This is a finite-grid estimate, not the optimal possible time-to-accuracy.

## Timing boundaries and figures

Inference time uses synchronized MPI barriers and maximum rank elapsed time;
includes sampling, posterior PIP summaries and cross-rank aggregation. Input
loading/broadcast/warm-up, graph decision and serialization are recorded outside
that interval. The controller additionally measures each launch through saved
result and process exit. Queue delays and long reference construction are not
per-fit inference time.

Rebuild candidates once per dataset to measure shared serial preparation,
checking the same support and local-score agreement within 1e-6 (CSV round-trip
roundoff allowed). Actual samplers use identical frozen source likelihoods.
"Composed end-to-end time" means measured shared preparation plus measured
launch-to-saved-result time, **not** an independently timed cold full-pipeline run.
Preparation includes data reading; it does not include synthetic-data generation.

Outputs under `analysis/`: per-run/summary CSVs, seed stability, reference gates,
failed attempts, hashes, finite-grid time-to-accuracy and scaling tables. Figures
are generated automatically as PDF/PNG: posterior error versus inference time,
core-hours and composed end-to-end time, quality-matched speedup, and fixed-work
MPI speedup/efficiency. Three repetitions give descriptive ranges, not population
confidence intervals. Full reports require all planned attempts; failures are
reported, not silently dropped. Existing successful records are never replaced.
