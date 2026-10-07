"""Synthetic Telco-like customers, so that the demo and the tests run with no download.

The generator follows the column contract in schema.py. Churn comes from a known logistic
model of contract, tenure, internet type, payment method, support and price. The rows are
fake. They are not a copy of any real customer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_colony.schema import ADDON_COLUMNS


def _choice(rng: np.random.Generator, levels, p, n):
    return rng.choice(np.asarray(levels, dtype=object), size=n, p=p)


def generate(n: int = 3000, seed: int = 0, blank_new_customers: bool = True) -> pd.DataFrame:
    """Return n synthetic customers with the raw text layout of the public Telco file."""
    if n < 20:
        raise ValueError("n must be at least 20")
    rng = np.random.default_rng(seed)

    contract = _choice(rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24], n)
    max_tenure = np.where(contract == "Month-to-month", 40, np.where(contract == "One year", 60, 72))
    tenure = np.floor(rng.beta(0.9, 1.1, n) * (max_tenure + 1)).astype(int)
    tenure[rng.random(n) < 0.01] = 0

    internet = _choice(rng, ["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], n)
    phone = np.where(rng.random(n) < 0.9, "Yes", "No")
    multiple = np.where(phone == "No", "No phone service", np.where(rng.random(n) < 0.45, "Yes", "No"))

    addons = {}
    addon_rate = {"OnlineSecurity": 0.35, "OnlineBackup": 0.42, "DeviceProtection": 0.42,
                  "TechSupport": 0.35, "StreamingTV": 0.5, "StreamingMovies": 0.5}
    for col in ADDON_COLUMNS:
        has = rng.random(n) < addon_rate[col]
        addons[col] = np.where(internet == "No", "No internet service", np.where(has, "Yes", "No"))

    payment = _choice(
        rng,
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21],
        n,
    )
    senior = (rng.random(n) < 0.16).astype(int)
    partner = np.where(rng.random(n) < 0.48, "Yes", "No")
    dependents = np.where((partner == "Yes") & (rng.random(n) < 0.6), "Yes", np.where(rng.random(n) < 0.1, "Yes", "No"))
    paperless = np.where(rng.random(n) < 0.59, "Yes", "No")
    gender = np.where(rng.random(n) < 0.5, "Female", "Male")

    base = np.where(internet == "Fiber optic", 70.0, np.where(internet == "DSL", 45.0, 20.0))
    base += np.where(phone == "Yes", 0.0, -5.0) + np.where(multiple == "Yes", 5.0, 0.0)
    n_addons = sum((addons[c] == "Yes").astype(float) for c in ADDON_COLUMNS)
    monthly = np.round(base + 5.0 * n_addons + rng.normal(0, 3, n), 2).clip(18.25, 120)
    drift = rng.normal(1.0, 0.05, n).clip(0.8, 1.2)
    total = np.round(monthly * tenure * drift, 2)

    logit = (
        -1.6
        + 1.3 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        - 0.045 * tenure
        + 0.8 * (internet == "Fiber optic")
        - 0.6 * (internet == "No")
        + 0.6 * (payment == "Electronic check")
        - 0.5 * (addons["TechSupport"] == "Yes")
        - 0.4 * (addons["OnlineSecurity"] == "Yes")
        + 0.3 * senior
        + 0.25 * (paperless == "Yes")
        + 0.012 * (monthly - 65)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    total_text = np.array([f"{v:.2f}" for v in total], dtype=object)
    if blank_new_customers:
        total_text[tenure == 0] = " "

    df = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(n)],
            "gender": gender,
            "SeniorCitizen": senior.astype(str),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure.astype(str),
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            **addons,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": [f"{v:.2f}" for v in monthly],
            "TotalCharges": total_text,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    return df
