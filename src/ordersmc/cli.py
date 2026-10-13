"""Command-line interface for reproducible OrderSMC-DAG analyses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from . import __version__
from .api import build_candidates, fit, sample
from .confidence import seed_consensus_table
from .config import OrderSMCConfig
from .data import (
    encode_data_with_metadata,
    infer_metadata_and_encode,
    metadata_from_mapping,
    metadata_summary,
    metadata_to_mapping,
    validate_data,
)
from .graphs import dag_from_edge_probs
from .persistence import (
    candidate_table_artifact_record,
    load_candidate_table,
    load_config,
    load_edge_probability_map,
    load_mapping,
    save_candidate_table,
    save_mapping,
)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args) or 0)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ordersmc",
        description="Scalable Bayesian DAG structure learning with Order-SMC",
    )
    parser.add_argument("--version", action="version", version=f"OrderSMC-DAG {__version__}")
    commands = parser.add_subparsers(dest="command")

    inspect_parser = commands.add_parser("inspect", help="infer and audit variable metadata")
    inspect_parser.add_argument("data", type=Path)
    inspect_parser.add_argument("--metadata", type=Path, help="existing JSON/YAML metadata")
    inspect_parser.add_argument("--metadata-out", type=Path, default=Path("metadata.json"))
    inspect_parser.add_argument("--encoded-output", type=Path)
    inspect_parser.add_argument("--categorical-max-unique", type=int, default=10)
    _add_input_encoding_argument(inspect_parser)
    inspect_parser.set_defaults(handler=_command_inspect)

    fit_parser = commands.add_parser("fit", help="build candidates, run SMC, and save results")
    fit_parser.add_argument("data", type=Path)
    _add_analysis_arguments(fit_parser)
    fit_parser.add_argument("--output", type=Path, default=Path("results/ordersmc"))
    fit_parser.add_argument("--candidate-table", type=Path, help="resume from a saved table")
    fit_parser.add_argument("--save-table", action="store_true")
    fit_parser.set_defaults(handler=_command_fit)

    build_parser = commands.add_parser("build-table", help="build and checkpoint parent-set candidates")
    build_parser.add_argument("data", type=Path)
    _add_analysis_arguments(build_parser)
    build_parser.add_argument("--output", type=Path, default=Path("candidate_table.pkl.gz"))
    build_parser.set_defaults(handler=_command_build_table)

    sample_parser = commands.add_parser("sample", help="run SMC from a saved candidate table")
    sample_parser.add_argument("candidate_table", type=Path)
    _add_config_arguments(sample_parser)
    sample_parser.add_argument("--backend", choices=["auto", "local", "mpi"], default="auto")
    sample_parser.add_argument("--output", type=Path, default=Path("results/ordersmc"))
    sample_parser.set_defaults(handler=_command_sample)

    summary_parser = commands.add_parser("summarize", help="print a saved run and its strongest edges")
    summary_parser.add_argument("result", type=Path)
    summary_parser.add_argument("--top", type=int, default=20)
    summary_parser.set_defaults(handler=_command_summarize)

    seeds_parser = commands.add_parser("multiseed", help="run several seeds and build a consensus graph")
    seeds_parser.add_argument("data", type=Path)
    _add_analysis_arguments(seeds_parser)
    seeds_parser.add_argument("--seeds", default="0,1,2,3,4")
    seeds_parser.add_argument("--output", type=Path, default=Path("results/ordersmc-multiseed"))
    seeds_parser.set_defaults(handler=_command_multiseed)
    return parser


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata", type=Path, help="JSON/YAML variable metadata")
    _add_input_encoding_argument(parser)
    parser.add_argument("--backend", choices=["auto", "local", "mpi"], default="auto")
    _add_config_arguments(parser)


def _add_input_encoding_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-encoding",
        choices=["auto", "raw", "encoded"],
        default="auto",
        help=(
            "interpret discrete columns as raw metadata levels or integer codes; "
            "auto rejects ambiguous numeric-looking labels"
        ),
    )


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="JSON/YAML OrderSMC configuration")
    parser.add_argument("--preset", choices=["quick", "standard", "thorough"], default="standard")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--betas", type=int, help="Temperature count for --no-adaptive-tempering; ignored by adaptive schedules")
    parser.add_argument("--tempering-path", choices=["geometric", "collapsed_likelihood"])
    parser.add_argument("--adaptive-tempering", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-cess-target", type=float)
    parser.add_argument("--final-target-steps", type=int, nargs="+", help="Additional proposals per particle at beta=1; increasing snapshot counts")
    parser.add_argument("--mh-steps", type=int)
    parser.add_argument("--max-parents", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--discrete-score", choices=["bdeu", "bds"])
    parser.add_argument(
        "--glm-tau",
        type=float,
        help="Normal prior SD for logistic and softmax coefficients",
    )
    parser.add_argument(
        "--score-method",
        choices=["native", "rank-native", "dg", "clg-bic"],
        help=(
            "local-family score: native mixed evidence, experimental rank-native "
            "plug-in weights, score-equivalent DG-BIC, "
            "or restricted conditional-Gaussian BIC"
        ),
    )
    parser.add_argument(
        "--dg-penalty-discount",
        type=float,
        help="DG-BIC penalty multiplier (ordinary BIC: 1.0)",
    )
    parser.add_argument(
        "--dg-ridge",
        type=float,
        help="optional non-negative DG covariance ridge (published score: 0.0)",
    )
    parser.add_argument(
        "--provenance-policy",
        choices=["require", "warn", "ignore"],
        default="warn",
        help="how to handle legacy candidate tables without embedded provenance",
    )
    parser.add_argument("--quiet", action="store_true")


def _config_from_args(args) -> OrderSMCConfig:
    config = load_config(args.config, preset=args.preset)
    return config.with_overrides(
        particles=args.particles,
        num_betas=args.betas,
        tempering_path=args.tempering_path,
        adaptive_tempering=args.adaptive_tempering,
        adaptive_cess_target=args.adaptive_cess_target,
        final_target_snapshot_steps=args.final_target_steps,
        mh_steps=args.mh_steps,
        max_parents=getattr(args, "max_parents", None),
        seed=args.seed,
        posterior_threshold=args.threshold,
        discrete_score=args.discrete_score,
        glm_tau=args.glm_tau,
        score_method=args.score_method,
        dg_penalty_discount=args.dg_penalty_discount,
        dg_ridge=args.dg_ridge,
    )


def _actual_backend(requested: str):
    if requested == "local":
        return "local", None, 0
    try:
        from mpi4py import MPI
    except ImportError:
        if requested == "mpi":
            raise RuntimeError("The MPI backend requires mpi4py")
        return "local", None, 0
    comm = MPI.COMM_WORLD
    backend = "mpi" if requested == "mpi" or comm.Get_size() > 1 else "local"
    return backend, comm if backend == "mpi" else None, comm.Get_rank() if backend == "mpi" else 0


def _read_data(
    path: Path, metadata_path: Path | None, *, input_encoding: str = "auto"
):
    if metadata_path is None:
        return infer_metadata_and_encode(pd.read_csv(path))
    vtypes, cards, levels = metadata_from_mapping(
        load_mapping(metadata_path), return_levels=True
    )
    # Preserve string categories such as zero-padded identifiers. Encoded CSVs
    # remain supported because numeric code strings are detected downstream.
    string_columns = {
        name: "string"
        for name, values in levels.items()
        if values and all(isinstance(value, str) for value in values)
    }
    raw = pd.read_csv(path, dtype=string_columns or None)
    data = encode_data_with_metadata(
        raw, vtypes, cards, levels, input_encoding=input_encoding
    )
    return data, vtypes, cards


def _distributed_data(args, backend, comm, rank):
    if rank == 0:
        values = _read_data(
            args.data,
            getattr(args, "metadata", None),
            input_encoding=getattr(args, "input_encoding", "auto"),
        )
    else:
        values = None
    return comm.bcast(values, root=0) if backend == "mpi" else values


def _command_inspect(args) -> int:
    if args.metadata is None:
        data, vtypes, cards, levels = infer_metadata_and_encode(
            pd.read_csv(args.data),
            categorical_max_unique=args.categorical_max_unique,
            return_levels=True,
        )
    else:
        data, vtypes, cards = _read_data(
            args.data, args.metadata, input_encoding=args.input_encoding
        )
        levels = data.attrs.get("ordersmc_levels", {})
    validate_data(data, vtypes, cards)
    summary = metadata_summary(data, vtypes, cards)
    print(summary.to_string(index=False))
    save_mapping(metadata_to_mapping(vtypes, cards, levels=levels), args.metadata_out)
    if args.encoded_output:
        args.encoded_output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.encoded_output, index=False)
    print(f"\nMetadata saved to {args.metadata_out}")
    return 0


def _command_fit(args) -> int:
    config = _config_from_args(args)
    backend, comm, rank = _actual_backend(args.backend)
    data, vtypes, cards = _distributed_data(args, backend, comm, rank)
    if args.candidate_table:
        if rank == 0:
            table = load_candidate_table(args.candidate_table)
        else:
            table = None
        table = comm.bcast(table, root=0) if backend == "mpi" else table
    else:
        table = None
    checkpoint_build_seconds = 0.0
    if table is None and args.save_table:
        checkpoint_started = time.perf_counter()
        table = build_candidates(
            data,
            vtypes,
            cards,
            config=config,
            backend=backend,
            comm=comm,
            verbose=not args.quiet,
        )
        checkpoint_build_seconds = time.perf_counter() - checkpoint_started
        if rank == 0:
            save_candidate_table(table, args.output / "candidate_table.pkl.gz")
    result = fit(
        data,
        vtypes,
        cards,
        config=config,
        backend=backend,
        comm=comm,
        candidate_table=table,
        verbose=not args.quiet,
        provenance_policy=args.provenance_policy,
    )
    if rank == 0:
        local_result = result.to_local_result() if backend == "mpi" else result
        if checkpoint_build_seconds:
            local_result.run_info["candidate_build_seconds"] = checkpoint_build_seconds
            local_result.run_info["total_seconds"] = float(local_result.run_info.get("total_seconds", 0.0)) + checkpoint_build_seconds
        local_result.save(args.output, threshold=config.posterior_threshold)
        _save_reproducibility_files(
            args.output, config, vtypes, cards, levels=data.attrs.get("ordersmc_levels")
        )
        _print_completion(local_result, args.output)
    return 0


def _command_build_table(args) -> int:
    config = _config_from_args(args)
    backend, comm, rank = _actual_backend(args.backend)
    data, vtypes, cards = _distributed_data(args, backend, comm, rank)
    table = build_candidates(
        data, vtypes, cards, config=config, backend=backend, comm=comm, verbose=not args.quiet
    )
    if rank == 0:
        saved = save_candidate_table(table, args.output)
        sidecar = saved.with_suffix(saved.suffix + ".json")
        save_mapping(
            {
                "nodes": table.nodes,
                "config": config.to_dict(),
                "artifact": candidate_table_artifact_record(table, saved),
                "candidate_provenance": table.provenance,
                "metadata": metadata_to_mapping(
                    vtypes, cards, levels=data.attrs.get("ordersmc_levels")
                ),
            },
            sidecar,
        )
        print(f"Candidate table saved to {saved}")
    return 0


def _command_sample(args) -> int:
    config = _config_from_args(args)
    backend, comm, rank = _actual_backend(args.backend)
    if rank == 0:
        table = load_candidate_table(args.candidate_table)
    else:
        table = None
    table = comm.bcast(table, root=0) if backend == "mpi" else table
    started = time.perf_counter()
    result = sample(
        table,
        config=config,
        backend=backend,
        comm=comm,
        verbose=not args.quiet,
        provenance_policy=args.provenance_policy,
    )
    if rank == 0:
        local_result = result.to_local_result() if backend == "mpi" else result
        local_result.run_info.update(
            {
                "candidate_table": str(args.candidate_table),
                "sampling_seconds": time.perf_counter() - started,
            }
        )
        local_result.save(args.output, threshold=config.posterior_threshold)
        save_mapping(config.to_dict(), args.output / "config.json")
        _print_completion(local_result, args.output)
    return 0


def _command_summarize(args) -> int:
    summary_path = args.result / "summary.json" if args.result.is_dir() else args.result
    if summary_path.exists() and summary_path.suffix == ".json":
        print(json.dumps(load_mapping(summary_path), indent=2, sort_keys=True))
    confidence_path = args.result / "edge_confidence.csv" if args.result.is_dir() else args.result
    if confidence_path.exists() and confidence_path.suffix == ".csv":
        frame = pd.read_csv(confidence_path).head(args.top)
        columns = [c for c in ["Parent", "Child", "PosteriorProbability", "MeanPosterior", "OrientationConfidence", "Confidence"] if c in frame]
        print("\nStrongest edges")
        print(frame[columns].to_string(index=False))
    return 0


def _command_multiseed(args) -> int:
    base_config = _config_from_args(args)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    backend, comm, rank = _actual_backend(args.backend)
    data, vtypes, cards = _distributed_data(args, backend, comm, rank)
    table = build_candidates(
        data, vtypes, cards, config=base_config, backend=backend, comm=comm, verbose=not args.quiet
    )
    probability_maps = []
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        save_candidate_table(table, args.output / "candidate_table.pkl.gz")
    for seed in seeds:
        config = base_config.with_overrides(seed=seed)
        result = sample(
            table,
            config=config,
            backend=backend,
            comm=comm,
            verbose=not args.quiet,
            provenance_policy=args.provenance_policy,
        )
        if rank == 0:
            local_result = result.to_local_result() if backend == "mpi" else result
            local_result.run_info.update(
                {"nodes": table.nodes, "config": config.to_dict(), "backend": backend, "seed": seed}
            )
            seed_dir = args.output / f"seed_{seed}"
            local_result.save(seed_dir, threshold=config.posterior_threshold)
            probability_maps.append(local_result.edge_probs)
    if rank == 0:
        consensus = seed_consensus_table(probability_maps, threshold=base_config.posterior_threshold)
        consensus.to_csv(args.output / "consensus_edge_confidence.csv", index=False)
        mean_probs = {
            (str(row.Parent), str(row.Child)): float(row.MeanPosterior)
            for row in consensus.itertuples(index=False)
        }
        graph = dag_from_edge_probs(mean_probs, table.nodes, threshold=base_config.posterior_threshold)
        pd.DataFrame(sorted(graph.edges()), columns=["Parent", "Child"]).to_csv(
            args.output / "consensus_dag.csv", index=False
        )
        _save_reproducibility_files(
            args.output,
            base_config,
            vtypes,
            cards,
            levels=data.attrs.get("ordersmc_levels"),
        )
        save_mapping(
            {"seeds": seeds, "threshold": base_config.posterior_threshold, "consensus_edges": graph.number_of_edges()},
            args.output / "consensus_summary.json",
        )
        print(f"Multi-seed consensus saved to {args.output}")
    return 0


def _save_reproducibility_files(
    output: Path,
    config: OrderSMCConfig,
    vtypes: Dict[str, str],
    cards: Dict[str, int],
    *,
    levels=None,
) -> None:
    save_mapping(config.to_dict(), output / "config.json")
    save_mapping(metadata_to_mapping(vtypes, cards, levels=levels), output / "metadata.json")


def _print_completion(result, output: Path) -> None:
    summary = result.summary()
    print(f"Results saved to {output}")
    print(
        f"Best edges: {summary['best_edges']} | posterior edges p>=0.5: "
        f"{summary['posterior_edges_p50']} | final ESS: {summary['effective_sample_size']:.1f}"
    )
    for warning in summary.get("warnings", []):
        print(f"Warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
