"""Feature selection: a seeded ant-colony selector and simple baselines.

All selectors are scikit-learn transformers. Inside a Pipeline they fit on the training
rows of each fold only, so the selection never sees validation or test rows.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import RFE, SelectFromModel, SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

SELECTOR_NAMES = ("none", "l1", "rfe", "aco")


def subset_fitness(score: float, n_selected: int, n_total: int, size_penalty: float) -> float:
    """Fitness of one feature subset: CV score minus a cost for each selected column.

    For the same score, a smaller subset always has a higher fitness.
    """
    if n_total <= 0 or not 0 < n_selected <= n_total:
        raise ValueError("need 0 < n_selected <= n_total")
    return float(score) - size_penalty * (n_selected / n_total)


def l1_logistic(C: float = 0.05, seed: int = 0) -> LogisticRegression:
    """L1 logistic regression for scikit-learn before and after the 1.8 `penalty` change."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, random_state=seed)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=seed)


def _set_seed(estimator, seed: int):
    est = clone(estimator)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est


class AntColonySelector(SelectorMixin, BaseEstimator):
    """Ant colony optimisation over column masks.

    Each ant draws a mask: column j is in the mask with a probability equal to its pheromone
    (clipped to [p_min, p_max]). The fitness of a mask is `subset_fitness` of its inner
    cross-validated score. After each iteration the pheromone evaporates by `evaporation`
    and the iteration-best and global-best masks deposit new pheromone.
    One `random_state` seeds the ants, the inner folds and the inner estimator.
    """

    def __init__(
        self,
        estimator=None,
        n_ants: int = 12,
        n_iterations: int = 10,
        evaporation: float = 0.25,
        size_penalty: float = 0.02,
        initial_pheromone: float = 0.5,
        p_min: float = 0.05,
        p_max: float = 0.95,
        min_features: int = 1,
        cv: int = 3,
        scoring: str = "roc_auc",
        random_state: int = 0,
    ):
        self.estimator = estimator
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation = evaporation
        self.size_penalty = size_penalty
        self.initial_pheromone = initial_pheromone
        self.p_min = p_min
        self.p_max = p_max
        self.min_features = min_features
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = X.shape[1]
        if n < 1:
            raise ValueError("no columns to select from")
        if not 0 < self.evaporation <= 1:
            raise ValueError("evaporation must be in (0, 1]")
        rng = np.random.default_rng(self.random_state)
        base = self.estimator if self.estimator is not None else LogisticRegression(max_iter=2000)
        base = _set_seed(base, self.random_state)
        folds = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        cache: dict[bytes, tuple[float, float]] = {}

        def evaluate(mask: np.ndarray) -> tuple[float, float]:
            key = np.packbits(mask).tobytes()
            if key not in cache:
                score = cross_val_score(clone(base), X[:, mask], y, cv=folds, scoring=self.scoring).mean()
                cache[key] = (float(score), subset_fitness(score, int(mask.sum()), n, self.size_penalty))
            return cache[key]

        tau = np.full(n, float(self.initial_pheromone))
        best_mask, best_fit, best_score = None, -np.inf, np.nan
        history = []
        k_min = min(max(1, self.min_features), n)
        for _ in range(self.n_iterations):
            prob = np.clip(tau, self.p_min, self.p_max)
            it_mask, it_fit = None, -np.inf
            for _ant in range(self.n_ants):
                mask = rng.random(n) < prob
                if mask.sum() < k_min:
                    order = np.argsort(-(prob + 1e-6 * rng.random(n)))
                    mask[order[:k_min]] = True
                score, fit = evaluate(mask)
                if fit > it_fit:
                    it_mask, it_fit = mask.copy(), fit
                if fit > best_fit:
                    best_mask, best_fit, best_score = mask.copy(), fit, score
            deposit = 0.5 * it_mask.astype(float) + 0.5 * best_mask.astype(float)
            tau = (1 - self.evaporation) * tau + self.evaporation * deposit
            history.append(best_fit)

        self.support_ = best_mask
        self.best_fitness_ = float(best_fit)
        self.best_score_ = float(best_score)
        self.pheromone_ = tau
        self.history_ = history
        self.n_evaluations_ = len(cache)
        self.n_features_in_ = n
        return self

    def _get_support_mask(self):
        return self.support_


def make_selector(name: str, seed: int = 0, aco_params: dict | None = None):
    """Return a selector step for a Pipeline, or 'passthrough' for no selection."""
    if name == "none":
        return "passthrough"
    if name == "l1":
        return SelectFromModel(
            l1_logistic(0.05, seed),
            threshold=1e-6,
        )
    if name == "rfe":
        return RFE(LogisticRegression(max_iter=2000, random_state=seed), n_features_to_select=0.5, step=2)
    if name == "aco":
        return AntColonySelector(random_state=seed, **(aco_params or {}))
    raise ValueError(f"unknown selector {name!r}; choose from {SELECTOR_NAMES}")
