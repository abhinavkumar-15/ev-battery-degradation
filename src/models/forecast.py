"""SOH forecasting: predict SOH at cycle t+h from information available at cycle t.

The model predicts the *change* delta = SOH[t+h] - SOH[t] (trees cannot extrapolate an
absolute level beyond what they saw; a delta target is stationary-ish) and the absolute
forecast is SOH[t] + delta_hat.  Metrics are reported on the ABSOLUTE SOH[t+h], and also
on delta (the part the model actually has to learn) so R^2 is not flattered by the
trivially predictable level.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.evaluation.metrics import regression_metrics
from src.models.tuning import fit_with_tuning
from src.models.validation import leave_one_battery_out
from src.models.zoo import make_models

log = logging.getLogger(__name__)


def naive_baselines(rows: pd.DataFrame, h: int) -> dict[str, np.ndarray]:
    soh = rows["soh"].to_numpy()
    slope = rows["soh_slope_10"].fillna(0).to_numpy() if "soh_slope_10" in rows else np.zeros(len(rows))
    return {"persistence (no model)": soh, "linear extrapolation (no model)": soh + h * slope}


def evaluate_horizon(rows: pd.DataFrame, feats: list[str], h: int, cfg: PipelineConfig, cache=None):
    """Leave-one-battery-out evaluation of every model at horizon h."""
    target = f"soh_future_h{h}"
    X = rows[feats]
    y_delta = (rows[target] - rows["soh"]).to_numpy()
    groups = rows["battery_id"].to_numpy()
    models = make_models(cfg.seed)
    oof: dict[str, np.ndarray] = {n: np.full(len(rows), np.nan) for n in models}
    params_log: dict[str, list] = {n: [] for n in models}

    for tr, te, bid in leave_one_battery_out(groups):
        key = f"forecast_h{h}_{bid}"
        hit = cache.get(key) if cache else None
        if hit is None:
            if cache:
                cache.check_budget()
            t0, hit = time.time(), {}
            for name, (est, grid) in models.items():
                fitted, best = fit_with_tuning(est, grid, X.iloc[tr], y_delta[tr], groups[tr], cfg.inner_cv_splits)
                hit[name] = (fitted.predict(X.iloc[te]), best)
            if cache:
                cache.put(key, hit, time.time() - t0)
        for name, (pred, best) in hit.items():
            oof[name][te] = pred
            params_log[name].append({"held_out": bid, **best})
        log.info("h=%d fold %s done", h, bid)

    preds = pd.DataFrame({"dataset": rows["dataset"].to_numpy(), "battery_id": groups, "cycle": rows["cycle"].to_numpy(),
                          "horizon": h, "soh": rows["soh"].to_numpy(), "y_true": rows[target].to_numpy()})
    for name, d in oof.items():
        preds[name] = rows["soh"].to_numpy() + d
    for name, v in naive_baselines(rows, h).items():
        preds[name] = v

    metrics = {}
    for name in [c for c in preds.columns if c not in ("dataset", "battery_id", "cycle", "horizon", "soh", "y_true")]:
        m = regression_metrics(preds["y_true"], preds[name])
        m["r2_delta"] = regression_metrics(preds["y_true"] - preds["soh"], preds[name] - preds["soh"])["r2"]
        m["per_battery"] = {b: regression_metrics(g["y_true"], g[name]) for b, g in preds.groupby("battery_id")}
        m["trained"] = name in models
        metrics[name] = m
    return preds, metrics, params_log
