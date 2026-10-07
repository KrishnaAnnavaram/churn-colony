"""Reference problems 4 and 5: SMOTE before CV, inverted and unseeded ACO."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from churn_colony.resample import ResampledClassifier, random_oversample, smote
from churn_colony.selection import AntColonySelector, make_selector, subset_fitness


def _toy(seed=0, n=300, informative=3, noise=9):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, informative + noise))
    logit = X[:, :informative] @ np.array([2.0, -1.5, 1.0][:informative])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_fitness_prefers_smaller_subsets_at_equal_score():
    assert subset_fitness(0.8, 3, 10, 0.05) > subset_fitness(0.8, 7, 10, 0.05)
    assert subset_fitness(0.8, 10, 10, 0.0) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        subset_fitness(0.8, 0, 10, 0.05)


def test_aco_is_reproducible_with_one_seed():
    X, y = _toy()
    a = AntColonySelector(n_ants=6, n_iterations=4, random_state=11).fit(X, y)
    b = AntColonySelector(n_ants=6, n_iterations=4, random_state=11).fit(X, y)
    assert np.array_equal(a.get_support(), b.get_support())
    assert a.best_fitness_ == b.best_fitness_


def test_aco_keeps_informative_columns_and_drops_noise():
    X, y = _toy()
    sel = AntColonySelector(n_ants=10, n_iterations=8, size_penalty=0.05, random_state=0).fit(X, y)
    support = sel.get_support()
    assert support[:3].all()
    assert support.sum() < X.shape[1]
    assert sel.history_ == sorted(sel.history_)  # global best never gets worse
    assert sel.transform(X).shape[1] == support.sum()


def test_aco_respects_min_features():
    X, y = _toy(noise=2)
    sel = AntColonySelector(n_ants=4, n_iterations=2, min_features=4, p_min=0.0, p_max=0.01,
                            initial_pheromone=0.0, random_state=0).fit(X, y)
    assert sel.get_support().sum() >= 4


def test_make_selector_names():
    assert make_selector("none") == "passthrough"
    X, y = _toy()
    for name in ("l1", "rfe", "aco"):
        params = {"n_ants": 3, "n_iterations": 2} if name == "aco" else None
        sel = make_selector(name, 0, params).fit(X, y)
        assert 0 < sel.get_support().sum() <= X.shape[1]
    with pytest.raises(ValueError):
        make_selector("magic")


def test_smote_balances_and_interpolates_between_minority_rows():
    X, y = _toy(n=200)
    Xr, yr = smote(X, y, k_neighbors=3, random_state=0)
    counts = np.bincount(yr)
    assert counts[0] == counts[1]
    minority = np.argmin(np.bincount(y))
    new = Xr[len(X):]
    lo, hi = X[y == minority].min(axis=0), X[y == minority].max(axis=0)
    assert np.all(new >= lo - 1e-9) and np.all(new <= hi + 1e-9)


def test_random_oversample_duplicates_rows():
    X, y = _toy(n=150)
    Xr, yr = random_oversample(X, y, random_state=0)
    assert np.bincount(yr)[0] == np.bincount(yr)[1]
    assert {tuple(r) for r in Xr[len(X):]} <= {tuple(r) for r in X}


def test_resampling_happens_in_fit_only():
    X, y = _toy(n=240)
    clf = ResampledClassifier(LogisticRegression(), method="smote", random_state=0).fit(X[:180], y[:180])
    assert clf.n_fit_rows_ > 180
    assert clf.predict_proba(X[180:]).shape == (60, 2)
    # in CV each validation fold is scored on its real rows only
    scores = cross_val_score(ResampledClassifier(LogisticRegression(), method="smote"), X, y, cv=3, scoring="roc_auc")
    assert scores.shape == (3,) and np.all(scores > 0.6)
    with pytest.raises(ValueError):
        ResampledClassifier(LogisticRegression(), method="adasyn").fit(X, y)
