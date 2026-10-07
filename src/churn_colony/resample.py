"""Class balance inside `fit` only.

`ResampledClassifier` wraps a classifier. It oversamples the minority class when it fits,
and it never touches the rows that it predicts. In cross-validation and in a search, the
validation fold therefore has only real rows (the prototype tuned on SMOTE output).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

RESAMPLING = ("none", "smote", "random", "class_weight")


def smote(X, y, k_neighbors: int = 5, random_state: int | None = None):
    """Plain SMOTE for a binary target: new minority rows on lines between minority neighbours."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        raise ValueError("smote needs exactly two classes")
    minority = classes[np.argmin(counts)]
    n_new = int(counts.max() - counts.min())
    if n_new == 0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(random_state)
    Xm = X[y == minority]
    k = min(k_neighbors, len(Xm) - 1)
    if k < 1:
        new = Xm[rng.integers(0, len(Xm), n_new)]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
        neighbours = nn.kneighbors(Xm, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xm), n_new)
        partner = neighbours[base, rng.integers(0, k, n_new)]
        gap = rng.random((n_new, 1))
        new = Xm[base] + gap * (Xm[partner] - Xm[base])
    return np.vstack([X, new]), np.concatenate([y, np.full(n_new, minority, dtype=y.dtype)])


def random_oversample(X, y, random_state: int | None = None):
    """Duplicate random minority rows until both classes have the same count."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    n_new = int(counts.max() - counts.min())
    rng = np.random.default_rng(random_state)
    idx = rng.choice(np.flatnonzero(y == minority), size=n_new, replace=True)
    return np.vstack([X, X[idx]]), np.concatenate([y, y[idx]])


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on resampled rows. Predict on the given rows unchanged."""

    def __init__(self, estimator=None, method: str = "none", k_neighbors: int = 5, random_state: int = 0):
        self.estimator = estimator
        self.method = method
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        if self.method not in ("none", "smote", "random"):
            raise ValueError(f"unknown resampling method {self.method!r}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.method == "smote":
            X, y = smote(X, y, self.k_neighbors, self.random_state)
        elif self.method == "random":
            X, y = random_oversample(X, y, self.random_state)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.n_fit_rows_ = len(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))
