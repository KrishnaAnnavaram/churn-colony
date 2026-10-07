"""Model factory and the one Pipeline that holds every fitted step.

Order of the steps: features -> preprocess (impute, scale, one-hot) -> select -> model.
The `model` step is a ResampledClassifier, so SMOTE (when chosen) happens inside fit too.
LightGBM, XGBoost and CatBoost are optional extras. They are imported only when chosen.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_colony.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TelcoFeatures
from churn_colony.resample import RESAMPLING, ResampledClassifier
from churn_colony.selection import make_selector

CORE_MODELS = ("logreg", "rf", "hgb", "stack")
EXTRA_MODELS = ("lightgbm", "xgboost", "catboost")
MODEL_NAMES = CORE_MODELS + EXTRA_MODELS


class MissingExtra(ImportError):
    """An optional dependency is not installed."""


def _require(module: str, extra: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingExtra(f"model needs {module!r}: pip install 'churn-colony[{extra}]'") from exc


def make_estimator(name: str, seed: int = 0, balanced: bool = False):
    """Return an unfitted classifier. `balanced` turns on class weights where the model has them."""
    cw = "balanced" if balanced else None
    if name == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight=cw, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight=cw, n_jobs=1, random_state=seed
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight=cw, random_state=seed,
        )
    if name == "stack":
        return StackingClassifier(
            estimators=[
                ("logreg", make_estimator("logreg", seed, balanced)),
                ("rf", make_estimator("rf", seed, balanced)),
                ("hgb", make_estimator("hgb", seed, balanced)),
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            cv=5,
            stack_method="predict_proba",
        )
    if name == "lightgbm":
        lgb = _require("lightgbm", "boost")
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, class_weight=cw,
            random_state=seed, verbose=-1,
        )
    if name == "xgboost":
        xgb = _require("xgboost", "boost")
        return xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9,
            eval_metric="logloss", random_state=seed,
        )
    if name == "catboost":
        cb = _require("catboost", "boost")
        return cb.CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=4, random_seed=seed, verbose=0,
            auto_class_weights="Balanced" if balanced else None,
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, list(NUMERIC_FEATURES)), ("cat", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
    )


def build_pipeline(
    model: str = "logreg",
    selector: str = "none",
    resampling: str = "none",
    seed: int = 0,
    aco_params: dict | None = None,
) -> Pipeline:
    if resampling not in RESAMPLING:
        raise ValueError(f"unknown resampling {resampling!r}; choose from {RESAMPLING}")
    estimator = make_estimator(model, seed, balanced=resampling == "class_weight")
    method = resampling if resampling in ("smote", "random") else "none"
    return Pipeline(
        [
            ("features", TelcoFeatures()),
            ("prep", make_preprocessor()),
            ("select", make_selector(selector, seed, aco_params)),
            ("model", ResampledClassifier(estimator, method=method, random_state=seed)),
        ]
    )


def selected_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the encoded columns that reach the model."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    select = pipeline.named_steps["select"]
    if select != "passthrough":
        names = names[select.get_support()]
    return [str(n) for n in names]
