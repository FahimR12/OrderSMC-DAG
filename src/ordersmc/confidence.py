"""Edge-confidence and multi-seed stability summaries."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

Edge = Tuple[str, str]


def confidence_label(probability: float) -> str:
    if probability >= 0.90:
        return "very_high"
    if probability >= 0.75:
        return "high"
    if probability >= 0.50:
        return "moderate"
    if probability >= 0.25:
        return "low"
    return "very_low"


def edge_confidence_table(
    edge_probs: Mapping[Edge, float],
    nodes: Sequence[str],
    *,
    edge_mcse: Mapping[Edge, float] | None = None,
    effective_sample_size: float | None = None,
) -> pd.DataFrame:
    """Create directed, skeleton, orientation, and Monte Carlo confidence columns."""
    edge_mcse = edge_mcse or {}
    rows = []
    for parent, child in sorted(edge_probs):
        probability = float(edge_probs[(parent, child)])
        reverse = float(edge_probs.get((child, parent), 0.0))
        skeleton = min(1.0, probability + reverse)
        orientation = probability / skeleton if skeleton > 0.0 else 0.0
        mcse = float(edge_mcse.get((parent, child), 0.0))
        entropy = 0.0
        if 0.0 < probability < 1.0:
            entropy = -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))
        rows.append(
            {
                "Parent": parent,
                "Child": child,
                "PosteriorProbability": probability,
                "ReverseProbability": reverse,
                "SkeletonProbability": skeleton,
                "OrientationConfidence": orientation,
                "MonteCarloSE": mcse,
                "MC95Lower": max(0.0, probability - 1.96 * mcse),
                "MC95Upper": min(1.0, probability + 1.96 * mcse),
                "BinaryEntropy": entropy,
                "EffectiveSampleSize": float(effective_sample_size or 0.0),
                "Confidence": confidence_label(probability),
            }
        )
    columns = [
        "Parent", "Child", "PosteriorProbability", "ReverseProbability",
        "SkeletonProbability", "OrientationConfidence", "MonteCarloSE",
        "MC95Lower", "MC95Upper", "BinaryEntropy", "EffectiveSampleSize", "Confidence",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["PosteriorProbability", "Parent", "Child"], ascending=[False, True, True]
    ).reset_index(drop=True)


def seed_consensus_table(
    edge_probability_maps: Sequence[Mapping[Edge, float]],
    *,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Summarize edge stability across independently seeded SMC runs."""
    if not edge_probability_maps:
        raise ValueError("At least one seed result is required")
    edges = sorted(set().union(*(set(values) for values in edge_probability_maps)))
    rows = []
    for parent, child in edges:
        values = np.array([float(run.get((parent, child), 0.0)) for run in edge_probability_maps])
        reverse_values = np.array([float(run.get((child, parent), 0.0)) for run in edge_probability_maps])
        mean = float(np.mean(values))
        reverse_mean = float(np.mean(reverse_values))
        skeleton = min(1.0, mean + reverse_mean)
        rows.append(
            {
                "Parent": parent,
                "Child": child,
                "MeanPosterior": mean,
                "PosteriorSD": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "MinimumPosterior": float(np.min(values)),
                "MaximumPosterior": float(np.max(values)),
                "SelectionFrequency": float(np.mean(values >= threshold)),
                "ReverseMeanPosterior": reverse_mean,
                "SkeletonProbability": skeleton,
                "OrientationConfidence": mean / skeleton if skeleton > 0.0 else 0.0,
                "Seeds": len(edge_probability_maps),
                "Confidence": confidence_label(mean),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["MeanPosterior", "Parent", "Child"], ascending=[False, True, True]
    ).reset_index(drop=True)
