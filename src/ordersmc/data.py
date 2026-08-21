"""Input preparation helpers for mixed tabular data."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def infer_metadata_and_encode(
    data: pd.DataFrame,
    *,
    categorical_max_unique: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, int]]:
    """Infer variable types and encode discrete columns as contiguous integers.

    Integer-like columns with at most ``categorical_max_unique`` distinct values
    are treated as binary/categorical. All other columns are continuous. For
    scientific analyses, pass explicit metadata to ``UnifiedFamilyScorer`` when
    domain knowledge gives a better type assignment.
    """
    if data.columns.has_duplicates:
        raise ValueError("Column names must be unique.")
    if data.isna().any().any():
        raise ValueError("Missing values must be imputed before fitting OrderSMC-DAG.")

    encoded = data.copy()
    vtypes: Dict[str, str] = {}
    cards: Dict[str, int] = {}
    for col in encoded.columns:
        values = np.sort(encoded[col].unique())
        integer_like = bool(np.all(np.isclose(values, np.round(values))))
        if len(values) <= categorical_max_unique and integer_like:
            mapping = {old: i for i, old in enumerate(values)}
            encoded[col] = encoded[col].map(mapping).astype("int64")
            vtypes[col] = "bin" if len(values) == 2 else "cat"
            cards[col] = int(len(values))
        else:
            encoded[col] = pd.to_numeric(encoded[col], errors="raise").astype("float64")
            vtypes[col] = "con"
            cards[col] = 0
    return encoded, vtypes, cards


def metadata_to_mapping(vtypes: Mapping[str, str], cards: Mapping[str, int]) -> Dict[str, Any]:
    """Return the portable metadata representation used by JSON/YAML files."""
    return {
        "variables": {
            name: {
                "type": variable_type,
                "cardinality": int(cards.get(name, 0)) if variable_type in {"bin", "cat"} else None,
            }
            for name, variable_type in vtypes.items()
        }
    }


def metadata_from_mapping(values: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Parse either the portable variable mapping or legacy vtypes/cards maps."""
    if "variables" in values:
        variables = values["variables"]
        if not isinstance(variables, Mapping):
            raise ValueError("metadata.variables must be a mapping")
        vtypes = {str(name): str(spec["type"]) for name, spec in variables.items()}
        cards = {
            str(name): int(spec.get("cardinality") or 0)
            for name, spec in variables.items()
        }
        return vtypes, cards
    if "vtypes" in values and "cards" in values:
        return (
            {str(k): str(v) for k, v in values["vtypes"].items()},
            {str(k): int(v) for k, v in values["cards"].items()},
        )
    raise ValueError("Metadata must contain 'variables' or both 'vtypes' and 'cards'")


def validate_data(
    data: pd.DataFrame,
    vtypes: Mapping[str, str],
    cards: Mapping[str, int],
    *,
    max_parents: int | None = None,
) -> None:
    """Fail early with actionable messages before expensive score construction."""
    if data.empty or data.shape[1] < 2:
        raise ValueError("Data must contain at least one row and two variables")
    if data.columns.has_duplicates:
        raise ValueError("Column names must be unique")
    nodes = list(data.columns)
    if set(vtypes) != set(nodes):
        missing = sorted(set(nodes) - set(vtypes))
        extra = sorted(set(vtypes) - set(nodes))
        raise ValueError(f"Variable metadata does not match data columns; missing={missing}, extra={extra}")
    if data.isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns {bad}; impute them before fitting")
    if max_parents is not None and max_parents >= len(nodes):
        raise ValueError(f"max_parents must be less than the number of variables ({len(nodes)})")

    for name in nodes:
        variable_type = vtypes[name]
        if variable_type not in {"bin", "cat", "con"}:
            raise ValueError(f"{name}: type must be 'bin', 'cat', or 'con', not {variable_type!r}")
        values = np.asarray(data[name])
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{name}: values must be numeric after encoding")
        if not np.all(np.isfinite(values.astype(float))):
            raise ValueError(f"{name}: contains non-finite values")
        unique = np.unique(values)
        if unique.size < 2:
            raise ValueError(f"{name}: constant columns are not supported")
        if variable_type in {"bin", "cat"}:
            card = int(cards.get(name, 0))
            expected = np.arange(card)
            if card < 2 or not np.array_equal(unique.astype(int), expected) or not np.allclose(unique, unique.astype(int)):
                raise ValueError(
                    f"{name}: discrete values must be contiguous integers 0..{card - 1}; observed {unique.tolist()}"
                )
            if variable_type == "bin" and card != 2:
                raise ValueError(f"{name}: binary variables must have cardinality 2")


def metadata_summary(
    data: pd.DataFrame, vtypes: Mapping[str, str], cards: Mapping[str, int]
) -> pd.DataFrame:
    """Build a human-readable variable audit table."""
    rows = []
    for name in data.columns:
        values = data[name]
        rows.append(
            {
                "Variable": name,
                "Type": vtypes[name],
                "Cardinality": int(cards.get(name, 0)) if vtypes[name] != "con" else None,
                "Unique": int(values.nunique()),
                "Missing": int(values.isna().sum()),
                "Minimum": float(values.min()),
                "Maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)
