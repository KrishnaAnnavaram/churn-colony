"""Reference problems 1, 2 and 7: target leak, fit on full data, ordinal codes."""

import numpy as np
import pandas as pd
import pytest

from churn_colony.features import TargetInFeatures, TelcoFeatures, leakage_screen, single_feature_auc
from churn_colony.models import build_pipeline, make_preprocessor
from churn_colony.schema import FEATURE_COLUMNS


def test_features_refuse_the_target_column(clean_small):
    with pytest.raises(TargetInFeatures):
        TelcoFeatures().fit(clean_small.drop(columns=["customerID"]))


def test_features_do_not_change_when_labels_change(xy_small):
    X, y = xy_small
    a = TelcoFeatures().fit(X, y).transform(X)
    b = TelcoFeatures().fit(X, 1 - y).transform(X)
    pd.testing.assert_frame_equal(a, b)


def test_engineered_values(xy_small):
    X, _ = xy_small
    out = TelcoFeatures().fit(X).transform(X)
    row = out.iloc[0]
    if row["tenure"] > 0:
        assert row["avg_monthly_spend"] == pytest.approx(row["TotalCharges"] / row["tenure"])
    assert set(out["autopay"]) <= {0, 1}
    fiber_ns = (X["InternetService"] == "Fiber optic") & (X["TechSupport"] == "No")
    assert (out["fiber_no_support"] == fiber_ns.astype(int)).all()
    assert set(out["tenure_band"]) <= {"0-12", "13-24", "25-48", "49+"}


def test_high_value_limit_comes_from_fit_rows_only(xy_small):
    X, _ = xy_small
    train, other = X.iloc[:400], X.iloc[400:].copy()
    other["MonthlyCharges"] = other["MonthlyCharges"] + 500
    f = TelcoFeatures().fit(train)
    assert f.high_value_threshold_ == pytest.approx(np.quantile(train["MonthlyCharges"], 0.75))
    f.transform(other)
    assert f.high_value_threshold_ == pytest.approx(np.quantile(train["MonthlyCharges"], 0.75))


def test_leakage_screen_flags_a_label_derived_column(xy_small):
    X, y = xy_small
    leaky = X.copy()
    # the prototype's feature: fiber AND churned
    leaky["FiberOpticDissatisfaction"] = ((X["InternetService"] == "Fiber optic") & (y == 1)).astype(int)
    leaky["label_copy"] = y.astype(str)
    flagged = leakage_screen(leaky, y)
    assert "label_copy" in flagged
    scores = single_feature_auc(leaky, y)
    assert scores["FiberOpticDissatisfaction"] > scores.drop(["FiberOpticDissatisfaction", "label_copy"]).max()


def test_engineered_features_pass_the_leakage_screen(xy_small):
    X, y = xy_small
    assert leakage_screen(TelcoFeatures().fit(X).transform(X), y) == []


def test_scaler_is_fitted_on_training_rows_only(xy_small):
    X, y = xy_small
    train = X.iloc[:500]
    pipe = build_pipeline("logreg").fit(train, y.iloc[:500])
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    engineered = TelcoFeatures().fit(train).transform(train)
    assert scaler.mean_[1] == pytest.approx(engineered["tenure"].mean())
    assert scaler.mean_[1] != pytest.approx(TelcoFeatures().fit(X).transform(X)["tenure"].mean())


def test_nominal_columns_are_one_hot_not_ordinal(xy_small):
    X, _ = xy_small
    feats = TelcoFeatures().fit(X).transform(X)
    prep = make_preprocessor().fit(feats)
    names = list(prep.get_feature_names_out())
    contract = [n for n in names if n.startswith("cat__Contract_")]
    assert sorted(contract) == ["cat__Contract_Month-to-month", "cat__Contract_One year", "cat__Contract_Two year"]
    assert not any(n == "num__Contract" for n in names)


def test_unknown_level_at_predict_time_does_not_crash(xy_small):
    X, y = xy_small
    pipe = build_pipeline("logreg").fit(X, y)
    new = X.iloc[:3].copy()
    new["PaymentMethod"] = "Crypto wallet"
    p = pipe.predict_proba(new)[:, 1]
    assert p.shape == (3,) and np.all((p > 0) & (p < 1))
    assert list(X.columns) == list(FEATURE_COLUMNS)
