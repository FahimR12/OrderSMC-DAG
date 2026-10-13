"""Versioned subprocess adapter for the native R ``bnlearn`` package."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

from ordersmc.core import OrderCandidateTable


DEFAULT_RSCRIPT = Path(
    "/users/psedrou/fastscratch/.conda/envs/ordersmc-bnlearn/bin/Rscript"
)
ADAPTER_SCRIPT = Path(__file__).resolve().parents[2] / "adapters" / "bnlearn_structure.R"


@dataclass(frozen=True)
class BnlearnResult:
    edges: tuple[tuple[str, str], ...]
    score: float
    algorithm: str
    scoring_mode: str
    rscript: str
    bnlearn_version: str

    @property
    def edge_probs(self) -> dict[tuple[str, str], float]:
        return {edge: 1.0 for edge in self.edges}


def _r_seed(seed: int) -> int:
    return int(seed) % 2_147_483_646 + 1


def rscript_path() -> Path:
    return Path(os.environ.get("ORDERSMC_BNLEARN_RSCRIPT", str(DEFAULT_RSCRIPT)))


def bnlearn_version(executable: Path | None = None) -> str:
    binary = rscript_path() if executable is None else executable
    if not binary.is_file():
        raise FileNotFoundError(f"bnlearn Rscript is unavailable: {binary}")
    completed = subprocess.run(
        [
            str(binary),
            "--vanilla",
            "-e",
            'suppressPackageStartupMessages(library(bnlearn)); cat(as.character(packageVersion("bnlearn")))',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _write_types(path: Path, variable_types: Mapping[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["node", "type"])
        for node in sorted(variable_types):
            writer.writerow([node, variable_types[node]])


def _write_score_table(path: Path, table: OrderCandidateTable) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["node", "parents", "score"])
        for child, node in enumerate(table.nodes):
            parent_sets = np.asarray(table.parent_sets[child], dtype=int)
            scores = np.asarray(table.loglik[child], dtype=float) + np.asarray(
                table.logprior[child], dtype=float
            )
            for parents, score in zip(parent_sets, scores):
                names = sorted(table.nodes[int(parent)] for parent in parents if int(parent) >= 0)
                writer.writerow([node, "|".join(names), format(float(score), ".17g")])


def run_bnlearn(
    data: pd.DataFrame,
    variable_types: Mapping[str, str],
    table: OrderCandidateTable,
    *,
    algorithm: str,
    scoring_mode: str,
    seed: int,
    timeout_seconds: int = 900,
) -> BnlearnResult:
    """Run deterministic HC or Tabu with native BIC-CG or the shared score table."""

    if algorithm not in {"hc", "tabu"}:
        raise ValueError("bnlearn algorithm must be hc or tabu")
    if scoring_mode not in {"native-bic-cg", "shared-mixed-score"}:
        raise ValueError("unsupported bnlearn scoring mode")
    if list(data.columns) != list(table.nodes):
        data = data.loc[:, table.nodes]
    if set(variable_types) != set(table.nodes):
        raise ValueError("bnlearn variable types do not match the candidate table")
    executable = rscript_path()
    version = bnlearn_version(executable)
    maximum_parents = max(
        (int(np.count_nonzero(np.asarray(row) >= 0)) for rows in table.parent_sets for row in rows),
        default=0,
    )
    with tempfile.TemporaryDirectory(prefix="ordersmc-bnlearn-") as raw_directory:
        directory = Path(raw_directory)
        data_path = directory / "data.csv"
        types_path = directory / "types.tsv"
        scores_path = directory / "scores.tsv"
        output_path = directory / "output.tsv"
        data.to_csv(data_path, index=False, lineterminator="\n", float_format="%.17g")
        _write_types(types_path, variable_types)
        _write_score_table(scores_path, table)
        completed = subprocess.run(
            [
                str(executable),
                "--vanilla",
                str(ADAPTER_SCRIPT),
                str(data_path),
                str(types_path),
                str(scores_path),
                scoring_mode,
                algorithm,
                str(maximum_parents),
                str(_r_seed(seed)),
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"bnlearn {algorithm}/{scoring_mode} failed: {message}")
        rows = list(csv.DictReader(output_path.open(encoding="utf-8"), delimiter="\t"))
    if not rows or rows[0]["record"] != "metadata":
        raise RuntimeError("bnlearn adapter returned an invalid result")
    edges = tuple(
        sorted((str(row["parent"]), str(row["child"])) for row in rows if row["record"] == "arc")
    )
    if any(parent not in table.nodes or child not in table.nodes for parent, child in edges):
        raise RuntimeError("bnlearn returned an unknown node")
    return BnlearnResult(
        edges=edges,
        score=float(rows[0]["value"]),
        algorithm=algorithm,
        scoring_mode=scoring_mode,
        rscript=str(executable),
        bnlearn_version=version,
    )


def validation_report() -> dict[str, object]:
    executable = rscript_path()
    version = bnlearn_version(executable)
    syntax = subprocess.run(
        [
            str(executable),
            "--vanilla",
            "-e",
            f"parse(file={json.dumps(str(ADAPTER_SCRIPT))}); cat('ok')",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "schema": "ordersmc-bnlearn-adapter-validation-v1",
        "status": "complete" if version == "5.2.1" and syntax.returncode == 0 else "failed",
        "all_passed": bool(version == "5.2.1" and syntax.returncode == 0),
        "rscript": str(executable),
        "bnlearn_version": version,
        "adapter_script": str(ADAPTER_SCRIPT),
        "adapter_script_sha256": hashlib.sha256(ADAPTER_SCRIPT.read_bytes()).hexdigest(),
        "r_syntax_check": syntax.stdout.strip(),
        "r_syntax_stderr": syntax.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validation_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.strict and not report["all_passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
