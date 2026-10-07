"""Probability metrics, bootstrap intervals, threshold metrics and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


class HardLabelsError(ValueError):
    """Scores look like 0/1 labels, but a ranking metric needs probabilities."""


def check_scores(p) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if np.any((p < 0) | (p > 1)) or np.isnan(p).any():
        raise ValueError("scores must be probabilities in [0, 1]")
    if p.size > 10 and np.isin(p, (0.0, 1.0)).all():
        raise HardLabelsError("scores are all 0 or 1. Pass predict_proba(...)[:, 1], not predict(...)")
    return p


def probability_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = check_scores(p)
    return {
        "n": int(y.size),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }


def bootstrap_ci(y, p, metric: str = "roc_auc", n_boot: int = 500, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap interval for roc_auc or pr_auc. Resamples rows with replacement."""
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    y = np.asarray(y).astype(int)
    p = check_scores(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(fn(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def threshold_metrics(y, p, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (check_scores(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / y.size),
        "contacted_share": float(pred.mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def calibration_table(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Equal-width bins. Returns the table and the expected calibration error (ECE)."""
    y = np.asarray(y).astype(int)
    p = check_scores(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        mean_p, rate = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_score": mean_p, "churn_rate": rate})
        ece += m.mean() * abs(mean_p - rate)
    return pd.DataFrame(rows), float(ece)


def lift_at(y, p, share: float = 0.1) -> float:
    """Churn rate in the top `share` of scores divided by the base rate."""
    y = np.asarray(y).astype(int)
    p = check_scores(p)
    k = max(1, int(round(share * y.size)))
    top = np.argsort(-p, kind="mergesort")[:k]
    return float(y[top].mean() / y.mean())
