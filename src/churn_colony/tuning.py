"""Hyperparameter search and nested cross-validation on the full Pipeline.

The search object holds the whole Pipeline. Feature fitting, selection, scaling and SMOTE
happen in each inner training fold, and every score comes from predicted probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from churn_colony.models import build_pipeline

PARAM_SPACES = {
    "logreg": {"model__estimator__C": loguniform(1e-3, 1e2)},
    "rf": {
        "model__estimator__min_samples_leaf": randint(1, 30),
        "model__estimator__max_features": uniform(0.1, 0.8),
    },
    "hgb": {
        "model__estimator__learning_rate": loguniform(0.01, 0.3),
        "model__estimator__max_leaf_nodes": randint(4, 40),
        "model__estimator__l2_regularization": loguniform(1e-3, 10),
    },
    "lightgbm": {
        "model__estimator__learning_rate": loguniform(0.01, 0.2),
        "model__estimator__num_leaves": randint(4, 64),
    },
    "xgboost": {
        "model__estimator__learning_rate": loguniform(0.01, 0.2),
        "model__estimator__max_depth": randint(2, 8),
    },
    "catboost": {
        "model__estimator__learning_rate": loguniform(0.01, 0.2),
        "model__estimator__depth": randint(3, 8),
    },
}


def make_search(model: str, selector: str = "none", resampling: str = "none", seed: int = 0,
                n_iter: int = 10, inner_folds: int = 3, scoring: str = "roc_auc") -> RandomizedSearchCV:
    if model not in PARAM_SPACES:
        raise ValueError(f"no search space for {model!r}; choose from {sorted(PARAM_SPACES)}")
    return RandomizedSearchCV(
        build_pipeline(model, selector, resampling, seed),
        PARAM_SPACES[model],
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,
        refit=True,
    )


def nested_cv(X, y, model: str, selector: str = "none", resampling: str = "none", seed: int = 0,
              outer_folds: int = 5, inner_folds: int = 3, n_iter: int = 10) -> dict:
    """Outer folds estimate the score of the whole tuning procedure. Returns per-fold results."""
    y = np.asarray(y).astype(int)
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        search = make_search(model, selector, resampling, seed + k, n_iter, inner_folds)
        search.fit(X.iloc[tr], y[tr])
        p = search.predict_proba(X.iloc[te])[:, 1]
        folds.append(
            {
                "fold": k,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "roc_auc": float(roc_auc_score(y[te], p)),
                "pr_auc": float(average_precision_score(y[te], p)),
                "inner_best_roc_auc": float(search.best_score_),
                "best_params": {k2: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                                for k2, v in search.best_params_.items()},
            }
        )
    roc = np.array([f["roc_auc"] for f in folds])
    pr = np.array([f["pr_auc"] for f in folds])
    return {
        "model": model,
        "selector": selector,
        "resampling": resampling,
        "roc_auc_mean": float(roc.mean()),
        "roc_auc_std": float(roc.std(ddof=1)) if roc.size > 1 else 0.0,
        "pr_auc_mean": float(pr.mean()),
        "folds": folds,
    }
