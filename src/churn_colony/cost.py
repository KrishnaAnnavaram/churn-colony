"""Retention cost model: expected profit of a contact list and the best threshold.

For each contacted customer the campaign pays `contact_cost`. If the customer is a churner,
the offer keeps the customer with probability `success_rate`, which saves `value` (the
monthly charge times `value_months`). Profit of a threshold t on labelled rows:

    profit(t) = sum over p_i >= t of (y_i * success_rate * value_i - contact_cost)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from churn_colony.metrics import check_scores


@dataclass(frozen=True)
class CostModel:
    contact_cost: float = 25.0
    success_rate: float = 0.25
    value_months: float = 6.0

    def customer_value(self, monthly_charges) -> np.ndarray:
        return np.asarray(monthly_charges, dtype=float) * self.value_months


def profit_at(y, p, value, threshold: float, cost: CostModel) -> float:
    y = np.asarray(y).astype(int)
    p = check_scores(p)
    value = np.asarray(value, dtype=float)
    contact = p >= threshold
    return float(np.sum(y[contact] * cost.success_rate * value[contact]) - cost.contact_cost * contact.sum())


def profit_curve(y, p, value, cost: CostModel, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.99, 0.01), 2)
    p = check_scores(p)
    rows = []
    for t in thresholds:
        contact = p >= t
        rows.append(
            {
                "threshold": float(t),
                "contacted": int(contact.sum()),
                "profit": profit_at(y, p, value, t, cost),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(y, p, value, cost: CostModel, thresholds=None) -> tuple[float, float]:
    """Threshold with the highest profit. Ties go to the higher threshold (fewer contacts)."""
    curve = profit_curve(y, p, value, cost, thresholds)
    best = curve.sort_values(["profit", "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), float(best["profit"])


def reference_profits(y, value, cost: CostModel) -> dict[str, float]:
    """Profit of two plain policies: contact nobody and contact everybody."""
    y = np.asarray(y).astype(int)
    value = np.asarray(value, dtype=float)
    everyone = float(np.sum(y * cost.success_rate * value) - cost.contact_cost * y.size)
    return {"contact_none": 0.0, "contact_all": everyone}
