"""Load a validated table (real CSV or synthetic) and split it into train, validation and test."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_colony import ID_COLUMN, TARGET
from churn_colony.schema import ValidationReport, read_csv, validate
from churn_colony.synthetic import generate


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def load(path: str | None, seed: int = 42, synthetic_rows: int = 5000) -> tuple[pd.DataFrame, ValidationReport, str]:
    """Return (clean frame, report, source label). With no path, generate synthetic rows."""
    if path:
        raw = read_csv(path)
        source = path
    else:
        raw = generate(synthetic_rows, seed=seed)
        source = f"synthetic(n={synthetic_rows}, seed={seed})"
    df, report = validate(raw)
    return df, report, source


def split(df: pd.DataFrame, seed: int = 42, val_size: float = 0.2, test_size: float = 0.2) -> Splits:
    """Stratified, seeded split. Each customer ID is in exactly one part."""
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in (0, 1) with a sum below 1")
    rest, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(rest, test_size=rel_val, stratify=rest[TARGET], random_state=seed)
    parts = Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))
    ids = [set(p[ID_COLUMN]) for p in (parts.train, parts.val, parts.test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise AssertionError("a customer ID is in two splits")
    return parts
