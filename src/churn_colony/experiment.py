"""Train, compare and evaluate. Every function returns plain dicts that the CLI prints or saves.

Protocol of `train`:
1. Split the clean table: 60 % train, 20 % validation, 20 % test (stratified, seeded).
2. Run the leakage screen on the engineered training features. A hit stops the run.
3. Fit each candidate Pipeline on the training rows.
4. Score each candidate on the validation rows. Keep the best by PR-AUC.
5. Select the decision threshold on the validation rows with the cost model.
6. Score the kept Pipeline once on the test rows: probability metrics, bootstrap intervals,
   calibration, threshold metrics, lift and profit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn_colony import ID_COLUMN, TARGET, __version__
from churn_colony.cost import CostModel, best_threshold, profit_at, reference_profits
from churn_colony.data import Splits, split
from churn_colony.explain import permutation_importance_raw, reference_values
from churn_colony.features import TelcoFeatures, leakage_screen
from churn_colony.metrics import (
    bootstrap_ci,
    calibration_table,
    lift_at,
    probability_metrics,
    threshold_metrics,
)
from churn_colony.models import build_pipeline, selected_feature_names
from churn_colony.schema import split_xy
from churn_colony.selection import SELECTOR_NAMES

ARTIFACT = "model.joblib"


class LeakageError(RuntimeError):
    """A feature separates the classes almost perfectly on the training rows."""


def _p(pipeline, X) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]


def train(
    df: pd.DataFrame,
    models=("logreg", "hgb", "stack"),
    selector: str = "none",
    resampling: str = "none",
    seed: int = 42,
    cost: CostModel = CostModel(),
    leakage_threshold: float = 0.95,
    n_boot: int = 500,
    aco_params: dict | None = None,
) -> tuple[dict, object, Splits]:
    parts = split(df, seed=seed)
    X_tr, y_tr = split_xy(parts.train)
    X_va, y_va = split_xy(parts.val)
    X_te, y_te = split_xy(parts.test)

    engineered = TelcoFeatures().fit(X_tr).transform(X_tr)
    leaks = leakage_screen(engineered, y_tr, leakage_threshold)
    if leaks:
        raise LeakageError(f"possible target leak in {leaks}. Remove the column or raise the threshold")

    candidates = {}
    for name in models:
        t0 = time.perf_counter()
        pipe = build_pipeline(name, selector, resampling, seed, aco_params).fit(X_tr, y_tr)
        val = probability_metrics(y_va, _p(pipe, X_va))
        val["fit_seconds"] = round(time.perf_counter() - t0, 2)
        val["n_model_columns"] = len(selected_feature_names(pipe))
        candidates[name] = (pipe, val)

    best_name = max(candidates, key=lambda m: (candidates[m][1]["pr_auc"], m))
    best, _ = candidates[best_name]

    p_va = _p(best, X_va)
    v_va = cost.customer_value(X_va["MonthlyCharges"])
    threshold, val_profit = best_threshold(y_va, p_va, v_va, cost)

    p_te = _p(best, X_te)
    v_te = cost.customer_value(X_te["MonthlyCharges"])
    test = probability_metrics(y_te, p_te)
    test["roc_auc_ci95"] = bootstrap_ci(y_te, p_te, "roc_auc", n_boot, seed=seed)
    test["pr_auc_ci95"] = bootstrap_ci(y_te, p_te, "pr_auc", n_boot, seed=seed)
    calib, ece = calibration_table(y_te, p_te)
    test["ece"] = ece
    test["lift_top10"] = lift_at(y_te, p_te, 0.10)
    test["at_threshold"] = threshold_metrics(y_te, p_te, threshold)
    test["at_0.5"] = threshold_metrics(y_te, p_te, 0.5)
    test["profit_at_threshold"] = profit_at(y_te, p_te, v_te, threshold, cost)
    test["reference_profits"] = reference_profits(y_te, v_te, cost)

    importance = permutation_importance_raw(best, X_te, y_te, n_repeats=5, seed=seed)

    summary = {
        "version": __version__,
        "seed": seed,
        "splits": parts.sizes(),
        "train_churn_rate": float(y_tr.mean()),
        "selector": selector,
        "resampling": resampling,
        "candidates": {m: v for m, (_, v) in candidates.items()},
        "chosen_model": best_name,
        "chosen_by": "validation pr_auc",
        "model_columns": selected_feature_names(best),
        "threshold": threshold,
        "threshold_rule": "maximum validation profit",
        "validation_profit": val_profit,
        "cost_model": cost.__dict__,
        "test": test,
        "calibration": calib.to_dict(orient="records"),
        "importance": importance.head(10).to_dict(orient="records"),
    }
    bundle = {
        "pipeline": best,
        "threshold": threshold,
        "reference": reference_values(X_tr),
        "cost": cost.__dict__,
        "summary": summary,
    }
    return summary, bundle, parts


def compare_selectors(df: pd.DataFrame, model: str = "logreg", selectors=SELECTOR_NAMES, seed: int = 42,
                      folds: int = 5, aco_params: dict | None = None) -> pd.DataFrame:
    """Cross-validate each selector on the train+validation rows. Selection runs inside each fold."""
    parts = split(df, seed=seed)
    dev = pd.concat([parts.train, parts.val], ignore_index=True)
    X, y = split_xy(dev)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows = []
    for name in selectors:
        t0 = time.perf_counter()
        res = cross_validate(
            build_pipeline(model, name, "none", seed, aco_params), X, y, cv=cv,
            scoring=("roc_auc", "average_precision"), return_estimator=True, n_jobs=1,
        )
        n_cols = [len(selected_feature_names(est)) for est in res["estimator"]]
        rows.append(
            {
                "selector": name,
                "roc_auc_mean": float(np.mean(res["test_roc_auc"])),
                "roc_auc_std": float(np.std(res["test_roc_auc"], ddof=1)),
                "pr_auc_mean": float(np.mean(res["test_average_precision"])),
                "columns_mean": float(np.mean(n_cols)),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
    return pd.DataFrame(rows)


def evaluate(bundle: dict, df: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Score a saved model on a labelled table (for example a later month of customers)."""
    X, y = split_xy(df)
    pipe = bundle["pipeline"]
    cost = CostModel(**bundle["cost"])
    p = _p(pipe, X)
    out = probability_metrics(y, p)
    out["roc_auc_ci95"] = bootstrap_ci(y, p, "roc_auc", n_boot, seed=seed)
    out["at_threshold"] = threshold_metrics(y, p, bundle["threshold"])
    out["lift_top10"] = lift_at(y, p, 0.10)
    out["profit_at_threshold"] = profit_at(y, p, cost.customer_value(X["MonthlyCharges"]), bundle["threshold"], cost)
    return out


def score(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Churn score and contact flag for each customer. The target column is not needed."""
    X = df.loc[:, [c for c in df.columns if c not in (ID_COLUMN, TARGET)]]
    p = _p(bundle["pipeline"], X)
    return pd.DataFrame({ID_COLUMN: df[ID_COLUMN].to_numpy(), "churn_score": p, "contact": p >= bundle["threshold"]})


def save_run(out_dir: str | Path, summary: dict, bundle: dict, model_card: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / ARTIFACT)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "model_card.md").write_text(model_card, encoding="utf-8")
    return out


def load_run(run_dir: str | Path) -> dict:
    path = Path(run_dir) / ARTIFACT
    if not path.is_file():
        raise FileNotFoundError(f"no {ARTIFACT} in {run_dir}. Run 'churn-colony train' first")
    return joblib.load(path)
