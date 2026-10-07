"""Reference problems 3 and 9: AUC from hard labels, no threshold or cost analysis."""

import numpy as np
import pytest

from churn_colony.cost import CostModel, best_threshold, profit_at, profit_curve, reference_profits
from churn_colony.metrics import (
    HardLabelsError,
    bootstrap_ci,
    calibration_table,
    check_scores,
    lift_at,
    probability_metrics,
    threshold_metrics,
)


def _scores(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.3 + 0.35 * (y - 0.3) + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, p


def test_hard_labels_are_refused():
    y, p = _scores()
    with pytest.raises(HardLabelsError):
        probability_metrics(y, (p >= 0.5).astype(float))
    with pytest.raises(ValueError):
        check_scores([0.2, 1.4])


def test_probability_metrics_rank_better_than_hard_labels():
    y, p = _scores()
    m = probability_metrics(y, p)
    from sklearn.metrics import roc_auc_score

    assert m["roc_auc"] > roc_auc_score(y, (p >= 0.5).astype(int))
    assert 0 < m["brier"] < 0.25 and m["pr_auc"] > m["base_rate"]


def test_bootstrap_interval_contains_the_point_estimate():
    y, p = _scores()
    lo, hi = bootstrap_ci(y, p, "roc_auc", n_boot=200, seed=1)
    assert lo < probability_metrics(y, p)["roc_auc"] < hi
    assert bootstrap_ci(y, p, "roc_auc", n_boot=200, seed=1) == (lo, hi)


def test_threshold_metrics_count_the_confusion_matrix():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.6, 0.7, 0.1, 0.2, 0.3, 0.05, 0.4, 0.2, 0.1, 0.3, 0.8])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 7)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)


def test_calibration_table_and_ece():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    table, ece = calibration_table(y, p)
    assert len(table) == 10 and table["n"].sum() == 5000
    assert ece < 0.03
    _, ece_bad = calibration_table(y, np.clip(p * 0.3, 0.001, 1))
    assert ece_bad > 0.2


def test_lift_of_a_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(0.99, 0.01, 100)
    assert lift_at(y, p, 0.1) == pytest.approx(10.0)


def test_profit_matches_a_hand_calculation():
    cost = CostModel(contact_cost=10, success_rate=0.5, value_months=12)
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    value = cost.customer_value([50, 60, 70, 80])  # 600, 720, 840, 960
    # contact the first two: 1 * 0.5 * 600 - 2 * 10 = 280
    assert profit_at(y, p, value, 0.5, cost) == pytest.approx(280.0)
    # contact three: 0.5 * (600 + 840) - 30 = 690
    assert profit_at(y, p, value, 0.25, cost) == pytest.approx(690.0)
    assert reference_profits(y, value, cost)["contact_all"] == pytest.approx(690.0 - 10)


def test_best_threshold_maximises_profit():
    cost = CostModel(contact_cost=10, success_rate=0.5, value_months=12)
    y, p = _scores()
    value = np.full(y.size, 600.0)
    t, best = best_threshold(y, p, value, cost)
    curve = profit_curve(y, p, value, cost)
    assert best == pytest.approx(curve["profit"].max())
    assert best >= max(reference_profits(y, value, cost).values())
    assert 0.02 <= t <= 0.98
