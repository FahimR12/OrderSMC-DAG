"""Snapshot the complete pilot execution tree before gated cluster submission."""
from pathlib import Path
import argparse
import shutil

from .execution import REPOSITORY_ROOT, _source_sha256, _atomic_json
from .readiness_audits import sha


def freeze(root):
    if root.exists():
        raise FileExistsError("use a fresh Section 2 output root; pipeline freeze is non-overwriting")
    folders = ("src", "experiments/src", "experiments/adapters", "experiments/fixtures", "scripts",
               "experiments/configs", "summary_statistics_multiopmic_data_pathway_molecule_level")
    digest = _source_sha256()
    files = {str(p.relative_to(REPOSITORY_ROOT)):sha(p) for folder in folders
             for p in sorted((REPOSITORY_ROOT/folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    for relative in files:
        dest = root / "code_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, dest)
    if _source_sha256() != digest or any(sha(root/"code_snapshot"/p)!=h for p,h in files.items()):
        raise RuntimeError("source changed during pipeline freeze")
    _atomic_json(root / "source_freeze.json", dict(source_sha256=digest, files=files))
    print(digest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    freeze(p.parse_args().output)


if __name__ == "__main__": main()
