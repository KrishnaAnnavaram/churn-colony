"""Model explanations.

* `permutation_importance_raw`: global importance of each raw column. It permutes the raw
  column, so all one-hot columns of a feature move together.
* `explain_customer`: per-customer effect of each raw column. The column takes its training
  reference value (median or most frequent level) and the change in churn score is recorded.
  The customer is chosen by ID, never by a loop variable.
* `shap_values`: optional, needs the `shap` extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from churn_colony.schema import CATEGORICAL, FEATURE_COLUMNS


def reference_values(X_train: pd.DataFrame) -> dict:
    ref = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL:
            ref[col] = X_train[col].mode().iloc[0]
        else:
            ref[col] = float(X_train[col].median())
    return ref


def permutation_importance_raw(pipeline, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    return (
        pd.DataFrame(
            {"feature": list(X.columns), "importance": result.importances_mean, "std": result.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def explain_customer(pipeline, row: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Return one line per raw column: its value, its reference value and its effect on the score."""
    if len(row) != 1:
        raise ValueError("pass exactly one customer row")
    row = row.loc[:, list(FEATURE_COLUMNS)]
    base = float(pipeline.predict_proba(row)[0, 1])
    variants = []
    for col in FEATURE_COLUMNS:
        alt = row.copy()
        alt[col] = reference[col]
        variants.append(alt)
    scores = pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    out = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "value": [row.iloc[0][c] for c in FEATURE_COLUMNS],
            "reference": [reference[c] for c in FEATURE_COLUMNS],
            "effect": base - scores,
        }
    )
    out.attrs["score"] = base
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def shap_values(pipeline, X: pd.DataFrame, background: pd.DataFrame, max_rows: int = 200):  # pragma: no cover
    """SHAP values on the encoded columns for the model step. Needs `pip install churn-colony[shap]`."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is an optional extra: pip install 'churn-colony[shap]'") from exc
    transform = pipeline[:-1]
    model = pipeline.named_steps["model"]
    bg = transform.transform(background.iloc[:max_rows])
    explainer = shap.Explainer(lambda a: model.predict_proba(np.asarray(a))[:, 1], bg)
    return explainer(transform.transform(X.iloc[:max_rows]))
