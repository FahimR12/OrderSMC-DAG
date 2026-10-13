# Independent prior replication, frozen before seeing outcomes

Submit with `bash scripts/submit_prior_holdout.sh`. Output:
`results/prior-holdout/v1`; fitted results will be under `fits/analysis`.

Use all three new protein-pathway null datasets and all three new positive
datasets, with 196 variables and 222 training observations each. Keep the
development generator, graph density, effect calibration and score settings
unchanged. Master seed 2026092102, experiment namespace
`prior-independent-validation`, replicates 100--102 are new. There is no
screening of replicates by accuracy or metadata agreement. Save metadata
fidelity statistics and report them rather than removing difficult datasets.

For each dataset, compare lambda=1 and lambda(d)=log(d-1), using four paired
inference seeds, 1,024 particles, 64 MPI ranks and snapshots at 0/250 extra
target moves: 48 fits in total. Build candidate support once per dataset and
retain bit-identical likelihoods across priors. These settings match the
development pilot; only graph/data/inference random draws change.

Report null false edges, posterior expected null edges, signal directed and
skeleton F1, and directed cross-seed Jaccard. Average seeds within datasets
first. Treat the three datasets as the replicate units, not the twelve seeds.
This is a small independent replication, not a definitive confirmatory study;
no automatic prior promotion or post-hoc success threshold is permitted.
Competitor fits under the old prior are not reused as new-prior comparisons.

Pipeline: source freeze -> regression/MPI preflight -> six parallel dataset
preparations -> immutable fit plan -> inference preflight -> 64-rank fits ->
complete-only aggregation. Submission records are resumable; old snapshots
and fitted artifacts are never edited. Preparation completion is not scientific
completion: check `fits/analysis/completion.json` for the latter.
