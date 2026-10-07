"""Model explainability: tree feature importance + SHAP.

WORDING: SHAP values describe what the fitted MODEL associates with its output. They are
not evidence that a feature physically causes degradation (features are correlated and the
data are observational), so the UI/README only say "associated with the prediction".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _final_estimator(pipe):
    return pipe.named_steps["model"]


def transform_features(pipe, X: pd.DataFrame) -> np.ndarray:
    Z = X.to_numpy(float)
    for name, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    return Z


def global_importance(pipe, X: pd.DataFrame, feats: list[str], max_rows: int = 400, seed: int = 42) -> dict:
    """Mean |SHAP| per feature (tree models) or |standardised coef| (linear), plus impurity importance."""
    import shap

    est = _final_estimator(pipe)
    Xs = X.sample(min(len(X), max_rows), random_state=seed) if len(X) > max_rows else X
    Z = transform_features(pipe, Xs[feats])
    out: dict = {"features": feats, "n_rows": int(len(Xs))}
    try:
        if hasattr(est, "feature_importances_"):
            sv = shap.TreeExplainer(est).shap_values(Z)
            out["method"] = "shap_tree"
            out["builtin_importance"] = dict(zip(feats, map(float, est.feature_importances_)))
        else:
            sv = shap.LinearExplainer(est, Z).shap_values(Z)
            out["method"] = "shap_linear"
        sv = np.asarray(sv)
        out["mean_abs_shap"] = dict(zip(feats, map(float, np.abs(sv).mean(axis=0))))
        out["sample"] = {"shap": np.round(sv[:150], 5).tolist(), "values": np.round(Z[:150], 5).tolist()}
    except Exception as e:  # SHAP is "if practical": degrade gracefully
        out["method"] = "builtin_only"
        out["error"] = str(e)
        if hasattr(est, "feature_importances_"):
            out["builtin_importance"] = dict(zip(feats, map(float, est.feature_importances_)))
            out["mean_abs_shap"] = out["builtin_importance"]
    return out


def local_explanation(pipe, row: pd.Series | dict, feats: list[str], top_k: int = 10, background: np.ndarray | None = None) -> dict:
    import shap

    est = _final_estimator(pipe)
    x = pd.DataFrame([{f: dict(row).get(f, np.nan) for f in feats}])
    Z = transform_features(pipe, x)
    if hasattr(est, "feature_importances_"):
        ex = shap.TreeExplainer(est)
    else:
        # a linear explainer needs a TRAINING background (not the single row being explained)
        ex = shap.LinearExplainer(est, background if background is not None else Z)
    sv = np.asarray(ex.shap_values(Z))[0]
    base = float(np.ravel(ex.expected_value)[0])
    order = np.argsort(-np.abs(sv))[:top_k]
    return {"base_value": base, "prediction_delta": float(base + sv.sum()),
            "contributions": [{"feature": feats[i], "value": float(Z[0, i]), "shap": float(sv[i])} for i in order]}
