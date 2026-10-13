"""Archive the accepted, completed rank-score and compute-budget evidence."""
from pathlib import Path
import argparse
import shutil
import tarfile

from .readiness_audits import read, write, sha

REPO = Path(__file__).resolve().parents[3]


def freeze(output):
    if output.exists():
        raise FileExistsError("publication freeze is non-overwriting")
    config = REPO / "experiments/configs/rank_acceptance_freeze_v1.json"
    accepted = read(config)
    roots = {"rank_validation": REPO / "results/rank-validation/20260917-v1",
             "rank_budget": REPO / "results/rank-budget/20260917-pilot-v1"}
    inventories = {}
    for name, root in roots.items():
        complete = read(root / "analysis/completion.json")
        if complete["status"] != "complete" or complete["missing"]:
            raise RuntimeError("cannot freeze incomplete results")
        key, expected = (("suite_sha256", accepted["rank_suite_sha256"]) if name == "rank_validation"
                         else ("plan_sha256", accepted["budget_plan_sha256"]))
        if complete[key] != expected:
            raise RuntimeError("source identity differs from the accepted cohort")
        checked = read(root / "analysis" / ("artifact_hashes.json" if name == "rank_validation" else "run_hashes.json"))
        for relative, digest in checked.items():
            if sha(root / relative) != digest:
                raise RuntimeError(f"source checksum mismatch: {relative}")
        inventories[name] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                             if p.is_file() and "__pycache__" not in p.parts}
    output.mkdir(parents=True)
    shutil.copy2(config, output / config.name)
    for name, root in roots.items():
        shutil.copytree(root / "analysis", output / name)
    with tarfile.open(output / "original_artifacts.tar.gz", "w:gz") as archive:
        for name, files in inventories.items():
            for relative in files:
                archive.add(roots[name] / relative, arcname=f"{name}/{relative}", recursive=False)
    write(output / "original_artifacts_manifest.json", inventories)
    text = """# Accepted rank-native development evidence — 2026-09-17

The user accepted these completed results as sufficient to progress to metadata
calibration and a protein-pathway pilot. This does not turn exploratory work
into untouched confirmation or make all directions identifiable.

## Contents and interpretation

`rank_validation/` contains 216 conditions, including 120 historical exact-only
conditions and 96 fresh conditions / 768 SMC fits. All source checks and sampling
flags passed. There are four fresh replicate blocks, with paired noise and
measurement variants—not 768 independent datasets. At n=5000, mean directed F1
for native versus rank-native is 0.910/0.931 (Gaussian mechanisms), 0.624/0.931
(skew-transformed Gaussian), 0.634/0.827 (Student-t), and 0.617/0.827 (transformed
Student-t). Small-sample improvements are not universal. Rank transformation
invariance is algebraic, not independent evidence of recovery.

`rank_budget/` contains all 320 fixed-data fits: eight n=5000 cases, two budgets,
twenty paired seeds. Mean exact-target PIP RMSE fell from 0.001410 to 0.000582;
identical twenty-seed DAGs increased from 4/8 to 5/8. Skeletons and fixed-0.90
uncertainty displays were identical in 8/8 cases at both budgets. Remaining
disagreements involve nearly tied directions. Recovery was broadly unchanged.

The explicit Section 2 primary score is rank-native, with 4096 particles and
10/25 MH moves. The 16384-particle, 20/50-move budget is a high-precision
reference. The genuinely Gaussian, direction-restricted M4 arm retains clg-bic.
The package-wide default is not changed. Rank-native is a plug-in generalized
posterior, not an integrated rank likelihood or original-scale marginal evidence.

Original artifacts, raw fit records, candidate tables, data and original source
snapshots are in `original_artifacts.tar.gz`; member hashes are in
`original_artifacts_manifest.json`. Previous publication editions remain intact.
The archive does not contain participant-level living-brain data.
"""
    (output / "README.md").write_text(text)
    write(output / "CHECKSUMS.json", {str(p.relative_to(output)): sha(p)
          for p in sorted(output.rglob("*")) if p.is_file()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    freeze(p.parse_args().output)


if __name__ == "__main__":
    main()
