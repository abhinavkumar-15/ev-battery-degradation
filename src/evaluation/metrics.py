from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    yt, yp = y_true[ok], y_pred[ok]
    r2 = float(r2_score(yt, yp)) if ok.sum() > 1 and np.var(yt) > 0 else None
    return {"mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))), "r2": r2, "n": int(ok.sum())}


def rul_metrics(y_true, y_pred) -> dict:
    """MAE/RMSE in cycles; MAPE only over rows with true RUL > 0 (undefined at RUL=0)."""
    m = regression_metrics(y_true, y_pred)
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    pos = (yt > 0) & np.isfinite(yp)
    m["mape_pct"] = float(np.mean(np.abs(yt[pos] - yp[pos]) / yt[pos]) * 100) if pos.any() else None
    return m
