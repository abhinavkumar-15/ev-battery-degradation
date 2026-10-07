"""Estimator factories.  Every estimator is a full sklearn Pipeline so the imputer/scaler
fitted on TRAINING rows only is saved together with the model (no train/test statistic leak)."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover - xgboost optional
    HAS_XGB = False

BOOSTED_NAME = "xgboost" if HAS_XGB else "hist_gradient_boosting"


def make_models(seed: int) -> dict[str, tuple[Pipeline, dict | None]]:
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    models: dict[str, tuple[Pipeline, dict | None]] = {
        "linear_regression": (Pipeline([("imp", imp()), ("scale", StandardScaler()), ("model", LinearRegression())]), None),
        "random_forest": (
            Pipeline([("imp", imp()), ("model", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=4))]),
            {"model__max_depth": [6, 12], "model__min_samples_leaf": [3, 10]},
        ),
    }
    if HAS_XGB:
        models["xgboost"] = (
            Pipeline([("imp", imp()), ("model", XGBRegressor(random_state=seed, n_jobs=2, tree_method="hist", verbosity=0))]),
            {"model__n_estimators": [200], "model__max_depth": [2, 4], "model__learning_rate": [0.05, 0.1],
             "model__subsample": [0.8]},
        )
    else:
        models["hist_gradient_boosting"] = (
            Pipeline([("imp", imp()), ("model", HistGradientBoostingRegressor(random_state=seed))]),
            {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1], "model__max_iter": [150, 300]},
        )
    return models
