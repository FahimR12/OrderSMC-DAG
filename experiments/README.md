# OrderSMC-DAG experiments

This directory contains the lean Version 3 experiment specification and reusable experiment utilities.

## Current pre-metadata model validation

The non-Gaussian scoring follow-up is specified in
[`../docs/RANK_NATIVE.md`](../docs/RANK_NATIVE.md). Run
`bash scripts/submit_rank_validation.sh` from the repository root. The launcher
activates the project environment, freezes every arm's source, submits a
one-process exact-reference replay and a 64-rank fresh-data SMC pilot, and then
aggregates with `afterany` so failed fits are explicitly reported. It refuses
duplicate submissions and never launches the full metadata study automatically.

Results live in `results/rank-validation/20260917-v1/`. `submission.json` records
job IDs immediately after each successful submission; if submission is partial,
inspect those IDs before any resubmission. Completed `analysis/summary.csv`
contains recovery and sampling error, `paired_differences_rank_minus_native.csv`
contains dataset-level contrasts, and `stability.csv` separates directed,
skeleton and uncertainty-display repeatability. The complete suite has 120
historical exact-only conditions plus 96 fresh conditions / 768 SMC fits.
This is development evidence, not a publication-sized confirmatory study.

### Fixed-data budget follow-up

The rank-native comparison has completed. Its reproducibility follow-up holds
the eight n=5000 candidate tables fixed and compares 4096 particles / 10–25 MH
moves against 16384 particles / 20–50 moves, with 20 paired seeds per budget.
See [`RANK_BUDGET_PROTOCOL.md`](RANK_BUDGET_PROTOCOL.md). Submit with
`bash scripts/submit_rank_budget.sh`; outputs go to
`results/rank-budget/20260917-pilot-v1/`. This is 320 fits using 64 MPI ranks per
fit and a four-task concurrency cap. It changes no local scores, priors or data.
Do not compare all-twenty-seed identity directly with the previous eight-seed
identity rate; both budgets are evaluated on the same new set of twenty seeds.

## Metadata study (after model validation and calibration)

The metadata-calibrated synthetic study for paper Section 2 is specified in
`SECTION2_METADATA_PROTOCOL.md` and
`configs/section2_metadata_v3.json`. The accepted rank-native and fixed-data
budget results are archived in
[`../results/publication/2026-09-17-rank-accepted/README.md`](../results/publication/2026-09-17-rank-accepted/README.md).
Launch the revised, protein-only calibration and pilot pipeline with:

```bash
bash scripts/submit_section2_v3_pilot.sh
```

This activates the environment and freezes the full execution tree. Metadata-only
calibration validation and score/adapter checks must pass before preparation;
candidate integrity/recall must pass before fits. It submits 23 datasets / 226
fits in gated stages, using 64 MPI ranks per SMC fit. Results are under
`results/section2-v3/seed-20260923-pilot`, stage IDs in `submission_pipeline.json`,
inference IDs in `submission_inference.json`, and logs in `code_snapshot/logs/`.
Do not independently submit the older two-phase scripts while this pipeline runs.
Calibration uses repeated size-222 samples and retains the original tolerances;
it does not consult recovery. Female prevalence is explicitly assumed to be 0.5,
with 0.4/0.6 metadata-fidelity sensitivity checks.

Full six-profile submission is deliberately locked until the pilot is reviewed
and a fresh confirmatory seed is frozen. Aggregation writes M1--M5 dataset-level
summaries and paired contrasts to `analysis/section2/`. Evaluation-seed PIPs are
averaged within dataset before constructing the primary representative graph;
per-seed recovery and stability are also reported. Versions 1 and 2 remain
available for historical reproduction, not for pooling with this new edition.

The latest completed mixed-data evidence and its limitations are summarized in
`PILOT_RESULTS_2026-09-03.md`.

For the validation-to-application paper centred on the living-brain data, start
with `BRAIN_STUDY_PROTOCOL.md` (or its review PDF), the machine-readable design in
`configs/brain_study_v1.json`, and the metadata-only template in
`configs/brain_data_profile.template.json`. The living-brain phase deliberately
remains non-runnable until that template is completed and the sampling
assumptions are reviewed.

V1 exact validation and V3 recognized-network validation are independent of the
living-brain metadata. V3 uses a separate pinned R environment for the native
bnlearn 5.2.1 baseline:

```bash
mamba env create -p /users/psedrou/fastscratch/.conda/envs/ordersmc-bnlearn \
  -f experiments/configs/bnlearn_environment.yml
PROFILE=smoke bash scripts/submit_v3.sh
```

The V3 launcher allocates within-run MPI ranks only to OrderSMC. Exact DP,
MCMC, and bnlearn runs remain single-process jobs. The completed pilot resource
audit is frozen in `configs/v3_pilot_freeze_v1.json`; the full V3 cohort can now
be launched in two restart-safe phases:

```bash
bash scripts/submit_v3_full_prepare.sh
# After the candidate-recall audit passes:
bash scripts/submit_v3_full_runs.sh
```

The full plan has 280 datasets and 5,260 fits. The first phase prepares all
datasets and candidate tables, and the second refuses to submit inference until
their integrity and recall audit passes. Point `V3_ROOT` at the completed full
root to regenerate the final, unstamped V3 paper figures:

```bash
V3_ROOT=results/brain-study/v3/seed-20260904-full \
  bash scripts/build_paper_figures.sh
```

### Corrective restricted-CLG validation

The original native mixed score is retained for reproducibility, but it is
misspecified for the native restricted-CLG benchmark. Version 0.6 adds the
explicit `clg-bic` model and a separate corrective-validation plan containing
only the three native CLG cells. Score selection is keyed by the declared
`native_mixed_conditional_gaussian` model class, never by `clgaussian.test` or
another dataset name. Candidate policy, order/structure priors, SMC settings,
MCMC controls, and compute allocations remain unchanged.

First run the small operational check:

```bash
PROFILE=smoke bash scripts/submit_v3_clg_prepare.sh
# After the prepared-data audit passes:
PROFILE=smoke bash scripts/submit_v3_clg_runs.sh
```

Then run the paper cohort with the held-out master seed `20260917`:

```bash
PROFILE=full bash scripts/submit_v3_clg_prepare.sh
# After the prepared-data audit passes:
PROFILE=full bash scripts/submit_v3_clg_runs.sh
```

The full corrective plan has 60 independently generated datasets and 1,260
runs. Its output is written below
`results/brain-study/v3-clg-score-v1/seed-20260917-full`. Operational smoke
uses the disjoint master seed `20260916`; smoke and full datasets therefore do
not overlap. The d=8 target is exactly integrated by order DP for every
dataset; the stochastic algorithms
are evaluated against that corrected target as well as against generating-DAG
recovery. Because the earlier poor-recovery results were inspected before the
score was changed, this phase must be labelled corrective validation rather
than untouched confirmation.

Inference indices are split into 1,000-index Slurm arrays because the target
cluster has `MaxArraySize=1001`. Shards of the same resource class are chained
with `afterany`, keeping `MAX_LOCAL_CONCURRENT` and `MAX_MPI_CONCURRENT` as
global caps while allowing restart of only missing immutable runs.

Do not modify `src/`, `experiments/src/`, `experiments/configs/`, or `scripts/`
between the preparation preflight and completion of inference. Their content is
covered by the preflight source hash; queued workers fail closed if it changes.

Smoke retains every network and method but caps observations at
`max(100, 5 * dimension)`; the original benchmark sample size remains recorded
as `benchmark_n_train`. Pilot and full profiles use the declared sample sizes.

The V3 pilot is deliberately split into two restart-safe phases. Candidate
tables must pass the integrity and recall audit before inference can begin:

```bash
bash scripts/submit_v3_pilot_prepare.sh
# After the audit job passes:
bash scripts/submit_v3_pilot_runs.sh
```

Rerunning either phase submits only missing or stale immutable work. The pilot
contains 42 datasets and 453 fits. OrderSMC uses 4,096 total particles and 64
MPI ranks per fit; this is a diagnostic resource allocation, not a speedup or
equal-compute claim.

- `PROTOCOL.md` is the authoritative human-readable design.
- `PROTOCOL.pdf` is a deterministic rendering for review and archiving.
- `configs/protocol_v3.json` is the machine-readable design summary.
- `ADAPTERS.md` defines the executable comparator adapters and their acceptance evidence.
- `validation/adapter_validation.json` is the checked tiny-state acceptance report.
- `src/ordersmc_experiments/` contains deterministic plans, mixed-data generation, exact references, all four core samplers, execution, metrics, and reproducibility helpers.
- `tests/` validates the included experiment primitives.

The launch scripts always run score and adapter preflight first. The supplied execution configuration is `confirmatory-frozen`; `configs/pilot_freeze_v3.json` binds it to the completed, predeclared E1 tuning pilot and its selection rule.

```bash
ordersmc-experiments validate-scores --full --strict
ordersmc-experiments validate-adapters --strict
ordersmc-experiments plan e1 --master-seed 20260814
ordersmc-experiments plan e2 --master-seed 20260814
ordersmc-experiments plan e3 --master-seed 20260814
ordersmc-experiments plan e4 --master-seed 20260814
```

Submit the complete Slurm workflows from the repository root:

```bash
bash scripts/submit_e1.sh
bash scripts/submit_e2.sh
bash scripts/submit_e3.sh
bash scripts/submit_e4.sh
```

The launchers explicitly activate `/users/psedrou/fastscratch/.conda/envs/myenv` in both the submission shell and every Slurm worker. Override it with `ORDERSMC_CONDA_ENV=/path/to/environment`; `ORDERSMC_PYTHON` remains an explicit no-activation escape hatch. You may also set `MASTER_SEED`, `PROFILE=smoke|full`, `MAX_CONCURRENT`, and `ORDERSMC_SBATCH_ARGS`. Each workflow is resumable and writes its immutable plan, dataset bundles, candidate tables, raw per-run JSON, completion ledger, and CSV summaries beneath `results/v3/`.

The default confirmatory master seed is `20260815`. Seed `20260814` belongs to the exploratory E1 pilot and its tuning subset and must not be reused in confirmatory summaries. The tuning workflow itself is reproducible with `bash scripts/submit_e1_tuning.sh`; its immutable outcome hashes and selected settings are recorded in `configs/pilot_freeze_v3.json`.

The plans are side-effect free: building one does not generate data, fit a score, initialize a sampler, or consume global RNG state.
