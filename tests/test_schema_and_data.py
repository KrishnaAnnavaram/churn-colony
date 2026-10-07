import numpy as np
import pandas as pd
import pytest

from churn_colony import ID_COLUMN, TARGET
from churn_colony.data import load, split
from churn_colony.schema import FEATURE_COLUMNS, SchemaError, read_csv, split_xy, validate
from churn_colony.synthetic import generate


def test_synthetic_is_seeded_and_valid():
    a, b = generate(300, seed=1), generate(300, seed=1)
    pd.testing.assert_frame_equal(a, b)
    df, report = validate(a)
    assert report.rows == 300
    assert 0.1 < report.churn_rate < 0.5
    assert set(df[TARGET].unique()) == {0, 1}


def test_blank_total_charges_for_new_customers_become_zero():
    raw = generate(500, seed=2)
    new = raw["tenure"] == "0"
    assert new.any(), "the generator must give some tenure-0 rows"
    df, report = validate(raw)
    assert (df.loc[new.to_numpy(), "TotalCharges"] == 0.0).all()
    assert any("TotalCharges blank" in f for f in report.fixes)


def test_missing_column_is_refused(raw_small):
    with pytest.raises(SchemaError, match="missing columns"):
        validate(raw_small.drop(columns=["Contract"]))


def test_unknown_category_is_refused(raw_small):
    bad = raw_small.copy()
    bad.loc[0, "Contract"] = "Three year"
    with pytest.raises(SchemaError, match="Contract"):
        validate(bad)


def test_duplicate_ids_are_refused(raw_small):
    bad = raw_small.copy()
    bad.loc[1, ID_COLUMN] = bad.loc[0, ID_COLUMN]
    with pytest.raises(SchemaError, match="duplicate"):
        validate(bad)


def test_inconsistent_addon_is_refused(raw_small):
    bad = raw_small.copy()
    i = bad.index[bad["InternetService"] == "No"][0]
    bad.loc[i, "TechSupport"] = "Yes"
    with pytest.raises(SchemaError, match="TechSupport"):
        validate(bad)


def test_non_numeric_charge_is_refused(raw_small):
    bad = raw_small.copy()
    i = bad.index[bad["tenure"] != "0"][0]
    bad.loc[i, "TotalCharges"] = "abc"
    with pytest.raises(SchemaError, match="TotalCharges"):
        validate(bad)


def test_split_xy_drops_target_and_id(clean_small):
    X, y = split_xy(clean_small)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert TARGET not in X.columns and ID_COLUMN not in X.columns
    assert y.isin([0, 1]).all()


def test_split_is_stratified_disjoint_and_seeded(clean_small):
    a, b = split(clean_small, seed=5), split(clean_small, seed=5)
    assert a.sizes() == {"train": 480, "val": 160, "test": 160}
    pd.testing.assert_frame_equal(a.test, b.test)
    ids = [set(p[ID_COLUMN]) for p in (a.train, a.val, a.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    rates = [p[TARGET].mean() for p in (a.train, a.val, a.test)]
    assert np.ptp(rates) < 0.02


def test_load_reads_csv_and_falls_back_to_synthetic(tmp_path):
    path = tmp_path / "t.csv"
    generate(120, seed=4).to_csv(path, index=False)
    df, report, source = load(str(path))
    assert source == str(path) and len(df) == 120
    df2, _, source2 = load(None, seed=1, synthetic_rows=150)
    assert source2.startswith("synthetic") and len(df2) == 150
    assert not pd.api.types.is_numeric_dtype(read_csv(str(path))["TotalCharges"])
