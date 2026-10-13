"""Input preparation helpers for mixed tabular data."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


class VariableTypeMap(dict):
    """Dict-compatible variable types carrying optional category levels.

    This keeps the long-standing three-value inference return API while letting
    ``metadata_to_mapping(vtypes, cards)`` preserve inferred levels by default.
    """

    def __init__(self, *args, levels: Mapping[str, Sequence[Any]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = {str(name): list(values) for name, values in (levels or {}).items()}


def infer_metadata_and_encode(
    data: pd.DataFrame,
    *,
    categorical_max_unique: int = 10,
    return_levels: bool = False,
):
    """Infer variable types and encode discrete columns as contiguous integers.

    Boolean, string, object, and pandas-categorical columns are treated as
    categorical. Integer-like numeric columns are categorical when they have at
    most ``categorical_max_unique`` distinct values; other numeric columns are
    continuous. Discrete level-to-code mappings are stored in
    ``encoded.attrs['ordersmc_levels']`` and can optionally be returned as a
    fourth value with ``return_levels=True``.
    """
    if data.columns.has_duplicates:
        raise ValueError("Column names must be unique.")
    if any(not isinstance(name, str) for name in data.columns):
        raise ValueError("Column names must be strings; rename columns before metadata inference.")
    if data.isna().any().any():
        raise ValueError("Missing values must be imputed before fitting OrderSMC-DAG.")

    if categorical_max_unique < 2:
        raise ValueError("categorical_max_unique must be at least 2")

    encoded = data.copy()
    vtypes: Dict[str, str] = {}
    cards: Dict[str, int] = {}
    levels: Dict[str, List[Any]] = {}
    for col in encoded.columns:
        series = encoded[col]
        categorical_dtype = isinstance(series.dtype, pd.CategoricalDtype)
        nonnumeric_categorical = (
            pd.api.types.is_bool_dtype(series.dtype)
            or categorical_dtype
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        )
        numeric = None if nonnumeric_categorical else pd.to_numeric(series, errors="raise")
        integer_like = bool(
            numeric is not None
            and np.all(np.isfinite(numeric.to_numpy(dtype=float)))
            and np.allclose(numeric, np.round(numeric))
        )
        is_discrete = nonnumeric_categorical or (
            integer_like and int(series.nunique(dropna=False)) <= categorical_max_unique
        )
        if is_discrete:
            column_levels = _ordered_levels(series)
            codes = pd.Categorical(series, categories=column_levels, ordered=True).codes
            if np.any(codes < 0):  # defensive; missing values were rejected above
                raise ValueError(f"{col}: could not encode every categorical value")
            encoded[col] = codes.astype("int64", copy=False)
            vtypes[col] = "bin" if len(column_levels) == 2 else "cat"
            cards[col] = int(len(column_levels))
            levels[col] = [_portable_level(value) for value in column_levels]
        else:
            encoded[col] = numeric.astype("float64") if numeric is not None else pd.to_numeric(series, errors="raise").astype("float64")
            vtypes[col] = "con"
            cards[col] = 0
    encoded.attrs["ordersmc_levels"] = levels
    vtypes = VariableTypeMap(vtypes, levels=levels)
    if return_levels:
        return encoded, vtypes, cards, levels
    return encoded, vtypes, cards


def _ordered_levels(series: pd.Series) -> List[Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.dropna().tolist())
        return [value for value in series.cat.categories.tolist() if value in observed]
    values = series.drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, repr(value)))


def _portable_level(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # CSV round trips represent uncommon object categories as text. Keeping the
    # same representation in metadata makes those files reusable.
    return str(value)


def metadata_to_mapping(
    vtypes: Mapping[str, str],
    cards: Mapping[str, int],
    levels: Mapping[str, Sequence[Any]] | None = None,
) -> Dict[str, Any]:
    """Return the portable metadata representation used by JSON/YAML files."""
    if levels is None:
        levels = getattr(vtypes, "levels", None)
    return {
        "variables": {
            name: {
                "type": variable_type,
                "cardinality": int(cards.get(name, 0)) if variable_type in {"bin", "cat"} else None,
                **(
                    {"levels": [_portable_level(value) for value in levels[name]]}
                    if levels is not None and name in levels and variable_type in {"bin", "cat"}
                    else {}
                ),
            }
            for name, variable_type in vtypes.items()
        }
    }


def metadata_from_mapping(
    values: Mapping[str, Any], *, return_levels: bool = False
):
    """Parse either the portable variable mapping or legacy vtypes/cards maps."""
    if "variables" in values:
        variables = values["variables"]
        if not isinstance(variables, Mapping):
            raise ValueError("metadata.variables must be a mapping")
        if any(not isinstance(spec, Mapping) for spec in variables.values()):
            raise ValueError("Each metadata variable specification must be a mapping")
        vtypes_values = {str(name): str(spec["type"]) for name, spec in variables.items()}
        cards = {
            str(name): int(spec.get("cardinality") or 0)
            for name, spec in variables.items()
        }
        levels = {
            str(name): list(spec["levels"])
            for name, spec in variables.items()
            if spec.get("levels") is not None
        }
        _validate_metadata_levels(vtypes_values, cards, levels)
        vtypes = VariableTypeMap(vtypes_values, levels=levels)
        return (vtypes, cards, levels) if return_levels else (vtypes, cards)
    if "vtypes" in values and "cards" in values:
        vtypes_values = {str(k): str(v) for k, v in values["vtypes"].items()}
        cards = {str(k): int(v) for k, v in values["cards"].items()}
        levels = {str(k): list(v) for k, v in values.get("levels", {}).items()}
        _validate_metadata_levels(vtypes_values, cards, levels)
        vtypes = VariableTypeMap(vtypes_values, levels=levels)
        return (vtypes, cards, levels) if return_levels else (vtypes, cards)
    raise ValueError("Metadata must contain 'variables' or both 'vtypes' and 'cards'")


def _validate_metadata_levels(
    vtypes: Mapping[str, str], cards: Mapping[str, int], levels: Mapping[str, Sequence[Any]]
) -> None:
    for name, column_levels in levels.items():
        if name not in vtypes:
            raise ValueError(f"Level mapping supplied for unknown variable {name!r}")
        if vtypes[name] not in {"bin", "cat"}:
            raise ValueError(f"{name}: level mappings are only valid for discrete variables")
        if len(column_levels) != int(cards.get(name, 0)):
            raise ValueError(
                f"{name}: metadata has {len(column_levels)} levels but cardinality {cards.get(name, 0)}"
            )
        if len({_level_key(value) for value in column_levels}) != len(column_levels):
            raise ValueError(f"{name}: metadata levels must be unique")


def _level_key(value: Any) -> Tuple[str, str]:
    return type(value).__name__, repr(value)


def encode_data_with_metadata(
    data: pd.DataFrame,
    vtypes: Mapping[str, str],
    cards: Mapping[str, int],
    levels: Mapping[str, Sequence[Any]] | None = None,
    *,
    input_encoding: Literal["auto", "raw", "encoded"] = "auto",
) -> pd.DataFrame:
    """Encode raw data according to explicit metadata without lossy casts.

    A discrete column without an explicit level mapping must already contain
    exact integer codes. In particular, fractional values are rejected before
    conversion to ``int64``. With a level mapping, either raw category values or
    already-encoded contiguous integer codes are accepted. ``auto`` rejects the
    rare ambiguous case where values are valid both as raw numeric-looking
    labels and as codes but the two interpretations give different encodings.
    """
    if input_encoding not in {"auto", "raw", "encoded"}:
        raise ValueError("input_encoding must be 'auto', 'raw', or 'encoded'")
    encoded = data.copy()
    levels = {str(name): list(values) for name, values in (levels or {}).items()}
    _validate_metadata_levels(vtypes, cards, levels)
    for name, variable_type in vtypes.items():
        if name not in encoded:
            continue
        series = encoded[name]
        if variable_type in {"bin", "cat"}:
            card = int(cards.get(name, 0))
            if name in levels:
                mapping = {_level_key(value): idx for idx, value in enumerate(levels[name])}
                keys = series.map(_level_key)
                raw_valid = bool(all(key in mapping for key in keys))
                codes_valid = _is_encoded_discrete(series, card)

                if input_encoding == "raw":
                    use_raw = True
                elif input_encoding == "encoded":
                    use_raw = False
                elif raw_valid and codes_valid:
                    raw_codes = keys.map(mapping).to_numpy(dtype=np.int64)
                    numeric_codes = np.rint(
                        pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
                    ).astype(np.int64)
                    if not np.array_equal(raw_codes, numeric_codes):
                        raise ValueError(
                            f"{name}: values are ambiguous between raw category labels and "
                            "integer codes; set input_encoding='raw' or 'encoded' explicitly"
                        )
                    use_raw = True
                else:
                    use_raw = raw_valid

                if use_raw:
                    if not raw_valid:
                        unknown = sorted(
                            {repr(value) for value, key in zip(series, keys) if key not in mapping}
                        )
                        raise ValueError(
                            f"{name}: values not present in metadata levels: {unknown[:5]}"
                        )
                    encoded[name] = keys.map(mapping).astype("int64")
                    continue
                if not codes_valid:
                    raise ValueError(
                        f"{name}: input_encoding='encoded' requires contiguous integer codes "
                        f"in 0..{card - 1}"
                    )

            # No level mapping, or an explicit/automatic encoded interpretation.
            if name not in levels or not use_raw:
                numeric = pd.to_numeric(series, errors="raise")
                values = numeric.to_numpy(dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{name}: contains non-finite values")
                integral = values == np.rint(values)
                if not np.all(integral):
                    bad = np.unique(values[~integral]).tolist()
                    raise ValueError(
                        f"{name}: discrete values must be exact integers before encoding; observed fractional values {bad[:5]}"
                    )
                encoded[name] = np.round(values).astype("int64")
        else:
            encoded[name] = pd.to_numeric(series, errors="raise").astype("float64")
    encoded.attrs["ordersmc_levels"] = levels
    return encoded


def _is_encoded_discrete(series: pd.Series, card: int) -> bool:
    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        numeric.size
        and np.all(np.isfinite(numeric))
        and np.all(numeric == np.rint(numeric))
        and np.all(numeric >= 0)
        and np.all(numeric < card)
    )


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
    if any(not isinstance(name, str) for name in nodes):
        raise ValueError("Column names must be strings; rename columns before fitting")
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
            integer_values = unique.astype(int)
            if (
                card < 2
                or not np.array_equal(integer_values, expected)
                or not np.all(unique == integer_values)
            ):
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
