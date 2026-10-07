"""Column contract of the Telco customer table and the validation that enforces it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from churn_colony import ID_COLUMN, TARGET

YES_NO = ("Yes", "No")
ADDON_LEVELS = ("Yes", "No", "No internet service")

CATEGORICAL: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ADDON_LEVELS,
    "OnlineBackup": ADDON_LEVELS,
    "DeviceProtection": ADDON_LEVELS,
    "TechSupport": ADDON_LEVELS,
    "StreamingTV": ADDON_LEVELS,
    "StreamingMovies": ADDON_LEVELS,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

# name -> (minimum, maximum)
NUMERIC: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 120_000),
}

ADDON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", *ADDON_COLUMNS, "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges",
    ]
)
REQUIRED_COLUMNS: tuple[str, ...] = (ID_COLUMN, *FEATURE_COLUMNS, TARGET)


class SchemaError(ValueError):
    """The table breaks the column contract."""


@dataclass
class ValidationReport:
    rows: int
    churn_rate: float | None
    fixes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rate = "n/a" if self.churn_rate is None else f"{self.churn_rate:.3f}"
        lines = [f"rows={self.rows} churn_rate={rate}"]
        lines += [f"fix: {f}" for f in self.fixes]
        return "\n".join(lines)


def validate(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, ValidationReport]:
    """Check and clean a raw table. Returns a new frame with typed columns and a report.

    Hard problems raise SchemaError. The only silent repair is a blank TotalCharges for a
    customer with tenure 0, which becomes 0.0 (no month has been billed). The report records it.
    """
    required = REQUIRED_COLUMNS if require_target else REQUIRED_COLUMNS[:-1]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = df.loc[:, [c for c in required]].copy()
    fixes: list[str] = []

    ids = out[ID_COLUMN].astype(str).str.strip()
    if (ids == "").any():
        raise SchemaError(f"{ID_COLUMN} has empty values")
    dup = ids[ids.duplicated()].unique()
    if len(dup):
        raise SchemaError(f"{ID_COLUMN} has {len(dup)} duplicate values, for example {dup[0]!r}")
    out[ID_COLUMN] = ids

    for col, levels in CATEGORICAL.items():
        values = out[col].astype(str).str.strip()
        bad = sorted(set(values) - set(levels))
        if bad:
            raise SchemaError(f"{col} has unknown values {bad[:5]} (allowed: {list(levels)})")
        out[col] = values

    for col, (lo, hi) in NUMERIC.items():
        raw = out[col]
        if not pd.api.types.is_numeric_dtype(raw):
            raw = raw.astype(str).str.strip().replace("", np.nan)
        values = pd.to_numeric(raw, errors="coerce")
        if col == "TotalCharges":
            blank_new = values.isna() & (pd.to_numeric(out["tenure"], errors="coerce") == 0)
            if blank_new.any():
                values = values.where(~blank_new, 0.0)
                fixes.append(f"TotalCharges blank for {int(blank_new.sum())} rows with tenure 0 -> 0.0")
        if values.isna().any():
            raise SchemaError(f"{col} has {int(values.isna().sum())} missing or non-numeric values")
        if ((values < lo) | (values > hi)).any():
            raise SchemaError(f"{col} has values outside [{lo}, {hi}]")
        out[col] = values.astype(float)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    out["tenure"] = out["tenure"].astype(int)

    _check_consistency(out)

    churn_rate = None
    if require_target:
        target = out[TARGET].astype(str).str.strip()
        bad = sorted(set(target) - set(YES_NO))
        if bad:
            raise SchemaError(f"{TARGET} must be Yes or No, found {bad[:5]}")
        out[TARGET] = (target == "Yes").astype(int)
        churn_rate = float(out[TARGET].mean())
        if out[TARGET].nunique() < 2:
            raise SchemaError(f"{TARGET} has only one class")
    return out.reset_index(drop=True), ValidationReport(len(out), churn_rate, fixes)


def _check_consistency(df: pd.DataFrame) -> None:
    no_internet = df["InternetService"] == "No"
    for col in ADDON_COLUMNS:
        mismatch = no_internet != (df[col] == "No internet service")
        if mismatch.any():
            raise SchemaError(f"{col} disagrees with InternetService in {int(mismatch.sum())} rows")
    no_phone = df["PhoneService"] == "No"
    mismatch = no_phone != (df["MultipleLines"] == "No phone service")
    if mismatch.any():
        raise SchemaError(f"MultipleLines disagrees with PhoneService in {int(mismatch.sum())} rows")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features (raw columns, never the target or the ID) and the 0/1 target."""
    return df.loc[:, list(FEATURE_COLUMNS)].copy(), df[TARGET].astype(int)


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV with every column as text, so that validation sees the raw values."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)
