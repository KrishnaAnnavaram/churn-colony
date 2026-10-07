"""Reference problems 4, 6 and 8: tuning on SMOTE folds, no baseline or stack, stale loop variable."""

import numpy as np
import pytest

from churn_colony.explain import explain_customer, permutation_importance_raw, reference_values
from churn_colony.models import CORE_MODELS, MissingExtra, build_pipeline, make_estimator, selected_feature_names
from churn_colony.schema import FEATURE_COLUMNS
from churn_colony.tuning import make_search, nested_cv


@pytest.mark.parametrize("name", ["logreg", "rf", "hgb"])
def test_core_models_fit_and_give_probabilities(xy_small, name):
    X, y = xy_small
    pipe = build_pipeline(name, seed=0).fit(X.iloc[:600], y.iloc[:600])
    p = pipe.predict_proba(X.iloc[600:])[:, 1]
    assert p.shape == (200,) and np.all((p >= 0) & (p <= 1))


def test_stacking_ensemble_is_built(xy_small):
    X, y = xy_small
    pipe = build_pipeline("stack", seed=0)
    est = pipe.named_steps["model"].estimator
    assert [n for n, _ in est.estimators] == ["logreg", "rf", "hgb"]
    pipe.set_params(model__estimator__rf__n_estimators=20, model__estimator__cv=3)
    pipe.fit(X.iloc[:600], y.iloc[:600])
    assert pipe.predict_proba(X.iloc[600:]).shape == (200, 2)


def test_class_weight_and_smote_options(xy_small):
    X, y = xy_small
    assert make_estimator("logreg", balanced=True).class_weight == "balanced"
    pipe = build_pipeline("logreg", resampling="smote").fit(X, y)
    assert pipe.named_steps["model"].n_fit_rows_ > len(X)
    with pytest.raises(ValueError):
        build_pipeline("logreg", resampling="magic")
    with pytest.raises(ValueError):
        make_estimator("svm")


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "catboost"])
def test_optional_boosters(xy_small, name):
    pytest.importorskip(name)
    X, y = xy_small
    pipe = build_pipeline(name, seed=0).fit(X.iloc[:600], y.iloc[:600])
    assert pipe.predict_proba(X.iloc[600:]).shape == (200, 2)


def test_missing_extra_message():
    assert issubclass(MissingExtra, ImportError)
    assert "logreg" in CORE_MODELS


def test_aco_inside_the_pipeline_reduces_columns(xy_small):
    X, y = xy_small
    pipe = build_pipeline("logreg", selector="aco", aco_params={"n_ants": 4, "n_iterations": 3, "size_penalty": 0.1}).fit(X, y)
    all_cols = len(pipe.named_steps["prep"].get_feature_names_out())
    assert 0 < len(selected_feature_names(pipe)) < all_cols


def test_nested_cv_tunes_inside_outer_training_folds(xy_small):
    X, y = xy_small
    res = nested_cv(X, y, "logreg", resampling="smote", seed=0, outer_folds=3, inner_folds=2, n_iter=2)
    assert len(res["folds"]) == 3
    for f in res["folds"]:
        assert f["n_train"] + f["n_test"] == len(X)
        assert 0.5 < f["roc_auc"] <= 1.0
        assert "model__estimator__C" in f["best_params"]
    search = make_search("logreg", resampling="smote", n_iter=2, inner_folds=2)
    assert search.estimator.named_steps["model"].method == "smote"
    with pytest.raises(ValueError):
        make_search("stack")


def test_permutation_importance_finds_the_strong_drivers(clean_small):
    from churn_colony.schema import split_xy
    from churn_colony.synthetic import generate
    from churn_colony.schema import validate

    df, _ = validate(generate(2000, seed=8))
    X, y = split_xy(df)
    pipe = build_pipeline("logreg").fit(X.iloc[:1400], y.iloc[:1400])
    imp = permutation_importance_raw(pipe, X.iloc[1400:], y.iloc[1400:], n_repeats=3)
    top = set(imp["feature"].head(5))
    assert {"Contract", "tenure"} & top
    assert set(imp["feature"]) == set(FEATURE_COLUMNS)


def test_explain_customer_is_chosen_by_id(clean_small, xy_small):
    X, y = xy_small
    pipe = build_pipeline("logreg").fit(X, y)
    ref = reference_values(X)
    cid = clean_small["customerID"].iloc[7]
    row = clean_small[clean_small["customerID"] == cid].loc[:, list(FEATURE_COLUMNS)]
    table = explain_customer(pipe, row, ref)
    assert len(table) == len(FEATURE_COLUMNS)
    assert table.attrs["score"] == pytest.approx(pipe.predict_proba(row)[0, 1])
    assert table["effect"].abs().is_monotonic_decreasing
    with pytest.raises(ValueError):
        explain_customer(pipe, X.iloc[:2], ref)
