# data/

Git ignores every file in this folder except this README. Do not commit datasets.

## Real data: IBM Telco Customer Churn

| Item | Value |
|---|---|
| Name | Telco Customer Churn (IBM sample data set) |
| Source | IBM sample data, also on Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |
| Terms | Read the terms on the source page before you use or share the file. This repository does not include it |
| Expected file | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (any path works with `--data` or `CHURN_COLONY_DATA`) |
| Size | 7,043 rows, 21 columns, one row for each customer |

Expected columns (the validator in `src/churn_colony/schema.py` checks each one):

| Column | Type | Allowed values |
|---|---|---|
| `customerID` | text | unique, not empty |
| `gender` | text | `Female`, `Male` |
| `SeniorCitizen` | 0/1 | `0`, `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | text | `Yes`, `No` |
| `tenure` | integer | 0 to 120 months |
| `MultipleLines` | text | `Yes`, `No`, `No phone service` |
| `InternetService` | text | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | text | `Yes`, `No`, `No internet service` |
| `Contract` | text | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | text | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | number | 0 to 1,000 |
| `TotalCharges` | number | 0 to 120,000. A blank value is allowed only when `tenure` is 0 |
| `Churn` | text | `Yes`, `No` (the target) |

## How to get the file

1. Download the CSV from the Kaggle page above (a free account is necessary).
2. Put it in this folder.
3. Run `churn-colony validate data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Synthetic data (no download)

`churn-colony synth --rows 5000 --out data/synthetic_telco.csv` writes fake customers with the same
columns. With no `--data` and no `CHURN_COLONY_DATA`, each command generates synthetic rows in memory.
The synthetic rows come from a known logistic model in `src/churn_colony/synthetic.py`.
