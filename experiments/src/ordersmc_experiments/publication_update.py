"""Package completed CLG correction and regenerated figures without deleting history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile

import pandas as pd

from .paper_figures import _require_complete
from .readiness_audits import read, write, sha


def package(source: Path, output: Path, audits: Path | None = None):
    prior_audit_manifest = output / "readiness_audits/source_manifest.json"
    if audits is None and prior_audit_manifest.is_file():
        audits = Path(read(prior_audit_manifest)["source_root"])
    completion = _require_complete(source)
    figure_manifest = read(output / "figures/figure_manifest.json")
    if Path(figure_manifest["corrected_clg"]["result_root"]).resolve() != source.resolve():
        raise ValueError("figures and correction artifacts come from different cohorts")
    for path, expected in figure_manifest["generated_sha256"].items():
        if sha(output / "figures" / path) != expected:
            raise ValueError(f"figure hash mismatch: {path}")
    for path, expected in figure_manifest["source_data_sha256"].items():
        if sha(output / "figures/source_data" / path) != expected:
            raise ValueError(f"figure source-data hash mismatch: {path}")
    for folder in ("analysis", "validation"):
        shutil.copytree(source / folder, output / "corrected_clg" / folder, dirs_exist_ok=True)
    shutil.copy2(source / "plan.json", output / "corrected_clg/plan.json")
    repo = Path(__file__).resolve().parents[3]
    protocol = output / "protocol"
    protocol.mkdir(exist_ok=True)
    for name in ["execution_v3_clg_corrective_v1.json", "section2_readiness_v1.json",
                 "section2_metadata_v2.json", "execution_section2_pilot_v2.json"]:
        shutil.copy2(repo / "experiments/configs" / name, protocol / name)
    shutil.copy2(repo / "experiments/SECTION2_METADATA_PROTOCOL.md", protocol)
    raw_files = sorted(p for folder in ("datasets", "runs") for p in (source / folder).rglob("*") if p.is_file())
    with tarfile.open(output / "corrected_clg/raw_artifacts.tar.gz", "w:gz") as archive:
        for path in raw_files:
            archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
    write(output / "corrected_clg/raw_artifacts_manifest.json", {
        "source_root": str(source.resolve()),
        "files": {str(p.relative_to(source)): sha(p) for p in raw_files}})
    frame = pd.read_csv(source / "analysis/dagbagm_metrics_runs.csv")
    frame = frame.rename(columns={"pip_rmse_to_exact_order": "pip_rmse"})
    metrics = ["directed_f1", "skeleton_f1", "directed_auprc", "directed_brier", "pip_rmse"]
    dataset = frame.groupby(["method", "n_train", "dataset_id"], as_index=False)[metrics].mean()
    summary = dataset.groupby(["method", "n_train"], as_index=False)[metrics].mean()
    summary.to_csv(output / "corrected_clg/method_summary.csv", index=False)
    smc = summary[summary.method == "ordersmc"]
    table = ["| Training n | Directed F1 | Skeleton F1 | Directed AUPRC | Directed Brier | Exact-order PIP RMSE |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in smc.to_dict(orient="records"):
        table.append(f"| {row['n_train']} | {row['directed_f1']:.4f} | {row['skeleton_f1']:.4f} | {row['directed_auprc']:.4f} | {row['directed_brier']:.6g} | {row['pip_rmse']:.6g} |")
    stability = pd.read_csv(source / "analysis/seed_stability_dagbagm_metrics.csv")
    stability = stability[stability.method == "ordersmc"]
    identical = int((stability.mean_directed_graph_jaccard == 1.).sum())
    readme = f"""# Publication update — corrected CLG validation, 2026-09-15

This edition supersedes the CLG figure and CLG stability/candidate rows in the
older publication bundle. It is an **addendum**, not a replacement for the
unchanged V1 and topology-mixed V3 raw artifacts in the parent publication
folder. Historical results are preserved and must not be pooled with the
correction. None of these experiments uses living-brain participant values.

## Completed correction

There are 60 independent datasets (20 at each n=250, 1000, 5000), eight
variables, and {completion['completed_runs']} completed runs with zero missing or failed runs.
The benchmark is the restricted conditional-Gaussian `clgaussian.test` model.
The held-out correction seed is 20260917. Each stochastic method has four
seeds per dataset. OrderSMC uses 4096 particles, 100 temperatures, 10 ordinary
and 25 post-resampling moves, with 64 MPI ranks per fit. This is **not** a
speedup study.

The version 0.6 `clg-bic` score uses the declared CLG model restriction and the
ordinary local BIC penalty. Continuous parents of discrete children are
prohibited. The graph prior and sampler controls were not selected by recovery.
The correction followed inspection of the earlier misspecified results and
must be called corrective validation, not untouched confirmation. General
unrestricted mixed-data panels still use the native mixed-family score.

The numbers below average the four method seeds within each dataset, then
average the 20 independent datasets. They are not seed-averaged consensus
graph scores and do not treat the 80 seeds as 80 independent datasets.

{chr(10).join(table)}

All exact-order sampling gates pass and all true parent families were retained.
The representative DAG is the constrained Bayes action under pair-state
Hamming loss, not independent PIP thresholding. All corrected representative
graph optimization audits are certified. All four SMC seeds select an identical
representative DAG in {identical}/60 datasets; continuous PIP reproducibility
and skeleton stability are provided separately in the CSVs.

Native bnlearn HC/Tabu are competitive and have higher directed F1 at the two
smaller sample sizes. Do not claim uniform OrderSMC dominance. Partition MCMC
has poor recovery/stability in these runs; its structure-modular target differs
from the order-modular target. The new exact structure-target audit is needed
to distinguish prior differences from non-mixing, and no poor results are removed.

## What each figure contains

- Figure 1: algorithm overview; a schematic, not an empirical result.
- Figure 2: unchanged full V1 exact-posterior validation on mixed data.
- Figure 3: unchanged full V3 topology-preserving mixed benchmarks; excludes CLG.
- Figure 4: **corrected** full restricted-CLG overlap with native bnlearn.
- Extended Data 1–2: unchanged full V1 seed stability and edge-type recovery.
- Extended Data 3: full V3 stability/candidate audit with old CLG rows replaced
  by corrected rows. The source CSV records the originating cohort for every row.
- Extended Data 4: unchanged full V3 mixed-topology edge-type recovery.

PDF/SVG are vector formats; PNG is 300 dpi. All plotted observations are under
`figures/source_data/`. Figure intervals are dataset-level bootstrap intervals
with 10,000 replicates. `figure_manifest.json` records source roots, plan hashes,
completion counts, figure hashes and the explicit CLG replacement policy.
Posterior methods and point-estimate baselines are labelled separately; AUPRC
from a hard graph is not a posterior-calibration claim.

## Contents and provenance

- `corrected_clg/analysis/`: complete original summary, run-level metrics,
  exact-order gates, seed-stability and posterior-summary tables.
- `corrected_clg/validation/`: original preflight and candidate-support audits.
- `corrected_clg/plan.json`: frozen dataset/run identities and seed records.
- `corrected_clg/raw_artifacts.tar.gz`: all original datasets, generator manifests,
  candidate tables, preparation identities and 1260 raw run JSON records.
- `corrected_clg/raw_artifacts_manifest.json`: SHA-256 per original archive member.
- `corrected_clg/method_summary.csv`: seed-within-dataset averaged metrics for all methods.
- `protocol/`: correction execution settings and proposed Section 2 v2/readiness designs.
- `SHA256SUMS`: integrity of every file in this edition (excluding the checksum file itself).

This edition stores the original run provenance, not a claim that today's
post-processing source hash equals the source hash used during the fits.
The intact earlier bundle contains V1 and unchanged V3 raw evidence. Its
legacy CLG results are retained solely for the methodological correction audit.

## Before proceeding to metadata recovery

`results/section2-readiness/seed-20260918-v2` is a separate exploratory diagnostic:
10 graphs, two paired variants (unrestricted native and fixed non-Gaussian
transform), three nested sample sizes, and four SMC seeds. It compares SMC with
exact order posteriors and order with exact structure posteriors, without tuning.
Separate audits resample the five existing protein-pathway mechanisms 50 times
each and assess Partition MCMC against its own exact target. Pending outputs
are not silently incorporated as completed publication results.

Section 2 v2 corrects M4 to use genuinely Gaussian continuous mechanisms as well
as CLG-compatible directions. Old Section 2 v1 prepared data remain historical.
The metadata generator tolerances are not yet confirmatory-frozen. Recovery,
stability, calibration limitations and any deterioration with n must all be
reported. This update is evidence for methodological validation, not a claim
that a living-brain application or a publication submission is ready.

## Rebuild and verify

From the repository root, run `bash scripts/build_paper_figures.sh`, then activate
the project environment (`source scripts/_common.sh`) and run:

```bash
"${{PYTHON}}" -m ordersmc_experiments.publication_update \\
  --clg-root results/brain-study/v3-clg-score-v1/seed-20260917-full \\
  --output results/publication/2026-09-15-corrected
```

Verify from this edition's directory with `sha256sum -c SHA256SUMS`.
"""
    if audits is not None:
        partition = read(audits / "partition_reference.json")
        calibration = read(audits / "calibration_audit.json")
        if partition["status"] != "diagnostic-complete" or calibration["status"] != "diagnostic-complete":
            raise RuntimeError("only completed audits may enter the publication addendum")
        if partition["plan_sha256"] != read(source / "plan.json")["plan_sha256"]:
            raise RuntimeError("audit and publication CLG cohorts differ")
        shutil.copytree(audits, output / "readiness_audits", dirs_exist_ok=True)
        write(output / "readiness_audits/source_manifest.json", {
            "source_root": str(audits.resolve()),
            "files": {p.name: sha(p) for p in sorted(audits.iterdir()) if p.is_file()},
            "status": "completed exploratory audits, distinct from pending score-transition inference"})
        readme += """
## Completed pre-metadata diagnostic audits

The seven original audit outputs are included under `readiness_audits/`, with
an additional manifest identifying their source folder. They are diagnostics,
not extra confirmatory replicates. They completed in job 10529370.

Partition MCMC's mean edge-PIP RMSE against **its own exact structure target**
is 0.406, 0.461, and 0.466 at n=250, 1000, and 5000 respectively. Comparing
against the correct target therefore does not resolve the problem. Alongside
the very low acceptance rates and between-seed differences, this supports
poor exploration under the implemented split/merge kernel and budget. It
does not establish that the published Partition-MCMC method generally fails.
The positive log-space source-layer reference also works where the existing
long-double inclusion–exclusion reference cannot represent the family-weight
range; cross-check statuses and every exact PIP are retained in the JSON.
Both order and structure edge-penalty lambdas in this source cohort are 1.0.

The M0 audit draws 50 independent size-222 samples from each of five fixed
protein-pathway mechanisms. No coefficients or scales were changed. PD
prevalence averages 0.593 versus the 0.595 target; age–PD correlation averages
0.306 versus 0.347, with every fixed mechanism's mean within the existing
tolerance. Some original single-sample failures are thus sampling fluctuation.
However, the 95th percentile of absolute biological correlations averages
0.523 versus the 0.433 target, with systematic tolerance misses in two of five
mechanisms. Mean absolute biological–PD correlation averages 0.238 versus
0.193, with a systematic tolerance miss in one of five mechanisms. This is
not an exact metadata match and the generator is not ready for a confirmatory
freeze. Rank, tail and moment discrepancies within tolerance are still reported,
not described as distributional equality. Original target uncertainty is unknown.

The initial score-transition array was stopped after tiny floating-point
probability-boundary errors in its new summary wrapper. Its partial outputs
and source snapshot remain in `seed-20260918-v1`; they are not final results.
The `v2` rerun uses the same data seeds and scientific settings, with a bounded
round-off correction only. It remains separate from the completed audits above.
"""
    (output / "README.md").write_text(readme)
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(output)}\n" for p in files))
    print(json.dumps({"status": "complete", "files": len(files), "output": str(output.resolve())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clg-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-audits", type=Path)
    args = parser.parse_args()
    package(args.clg_root, args.output, args.readiness_audits)


if __name__ == "__main__":
    main()
