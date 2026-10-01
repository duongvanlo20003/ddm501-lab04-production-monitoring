"""
Data validation stage — the quality gate in front of training.

"""

import logging
from typing import Any, Dict, List

import pandas as pd

from pipeline.config import (
    MAX_MISSING_FRACTION,
    MAX_POSITIVE_RATE,
    MIN_POSITIVE_RATE,
    MIN_ROWS,
    RAW_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails a check that must not be ignored."""


# Value domains, from the dataset documentation.
DOMAINS: Dict[str, Any] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
RANGES: Dict[str, tuple] = {
    "LIMIT_BAL": (10_000, 2_000_000),
    "AGE": (18, 100),
    **{c: (-2, 8) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
}


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Level 1 — are the expected columns present, with usable types?"""
    errors = []
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    for col in df.columns:
        if col in RAW_FEATURES + [TARGET] and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"column '{col}' is not numeric (got {df[col].dtype})")
    return errors


def validate_statistics(df: pd.DataFrame) -> List[str]:
    """Level 2 — is the shape of the data what training assumes?"""
    errors = []
    if len(df) < MIN_ROWS:
        errors.append(f"only {len(df)} rows, need at least {MIN_ROWS}")
    for col in df.columns:
        frac = float(df[col].isna().mean())
        if frac > MAX_MISSING_FRACTION:
            errors.append(
                f"column '{col}' is {frac:.1%} missing (limit {MAX_MISSING_FRACTION:.0%})"
            )
    if TARGET in df.columns:
        rate = float(df[TARGET].mean())
        if not MIN_POSITIVE_RATE <= rate <= MAX_POSITIVE_RATE:
            errors.append(
                f"target positive rate {rate:.1%} outside [{MIN_POSITIVE_RATE:.0%}, "
                f"{MAX_POSITIVE_RATE:.0%}] — check the upstream extract"
            )
    return errors


def validate_semantics(df: pd.DataFrame) -> List[str]:
    """Level 3 — do the values mean what the business says they mean?"""
    errors = []
    for col, allowed in DOMAINS.items():
        if col in df.columns:
            unexpected = set(df[col].dropna().unique()) - allowed
            if unexpected:
                errors.append(
                    f"column '{col}' has values outside {sorted(allowed)}: {sorted(unexpected)[:5]}"
                )
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if out:
                errors.append(f"column '{col}' has {out} values outside [{lo}, {hi}]")
    for col in [c for c in df.columns if c.startswith("PAY_AMT")]:
        neg = int((df[col] < 0).sum())
        if neg:
            errors.append(f"column '{col}' has {neg} negative payments")
    return errors


def validate_dataset(df: pd.DataFrame, raise_on_error: bool = True) -> Dict[str, Any]:
    """Run all three levels and return a report.

    The report is logged to MLflow as an artifact, so a model that was trained
    on data with known problems carries the evidence with it.
    """
    schema = validate_schema(df)
    statistics = validate_statistics(df)
    semantics = validate_semantics(df)
    errors = schema + statistics + semantics

    report = {
        "passed": not errors,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "schema_errors": schema,
        "statistical_errors": statistics,
        "semantic_errors": semantics,
        "n_errors": len(errors),
    }

    if errors:
        for e in errors:
            logger.error("validation: %s", e)
        if raise_on_error:
            raise DataValidationError(f"{len(errors)} validation error(s): {errors[:3]}")
    else:
        logger.info("Data validation passed (%s rows, %s columns)", len(df), df.shape[1])

    return report
