"""Remaining-Useful-Life estimation.

Two independent estimators are produced and compared against the ground truth derived from
each cell's observed trajectory (cycles until SOH first <= threshold):

 1. DIRECT ML : regress RUL (cycles) on the cycle-t feature vector (Linear / RF / XGB).
 2. EXTRAPOLATION (no ML): RUL ~= (SOH[t] - threshold) / trailing degradation rate.

Cells that never reach the threshold are right-censored and excluded from RUL training and
scoring (we cannot know their true RUL).  RUL is an ESTIMATE with substantial uncertainty,
not a physical measurement; the direct models are tied to the threshold used at training time.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.evaluation.metrics import rul_metrics
from src.models.tuning import fit_with_tuning
from src.models.validation import leave_one_battery_out
from src.models.zoo import make_models


def extrapolate_rul(soh: np.ndarray, rate: np.ndarray, threshold: float, cap: float) -> np.ndarray:
    soh, rate = np.asarray(soh, float), np.asarray(rate, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = (soh - threshold) / rate
    est = np.where(np.isfinite(est) & (rate > 1e-6), est, cap)   # no measurable fade -> "beyond horizon"
    return np.clip(est, 0, cap)


def evaluate_rul(rows: pd.DataFrame, feats: list[str], cfg: PipelineConfig, cache=None):
    y = rows["rul"].to_numpy()
    X, groups = rows[feats], rows["battery_id"].to_numpy()
    cap = float(rows["eol_cycle"].max() * 2)
    models = make_models(cfg.seed)
    oof = {n: np.full(len(rows), np.nan) for n in models}
    for tr, te, bid in leave_one_battery_out(groups):
        key = f"rul_{bid}"
        hit = cache.get(key) if cache else None
        if hit is None:
            if cache:
                cache.check_budget()
            t0, hit = time.time(), {}
            for name, (est, grid) in models.items():
                fitted, _ = fit_with_tuning(est, grid, X.iloc[tr], y[tr], groups[tr], cfg.inner_cv_splits)
                hit[name] = np.clip(fitted.predict(X.iloc[te]), 0, None)
            if cache:
                cache.put(key, hit, time.time() - t0)
        for name, pred in hit.items():
            oof[name][te] = pred
    rate = rows["rolling_degradation_rate_25"].fillna(rows.get("rolling_degradation_rate_10"))
    oof["trajectory extrapolation (no ML)"] = extrapolate_rul(rows["soh"], rate, cfg.eol_threshold_soh, cap)

    preds = pd.DataFrame({"dataset": rows["dataset"].to_numpy(), "battery_id": groups, "cycle": rows["cycle"].to_numpy(),
                          "soh": rows["soh"].to_numpy(), "y_true": y, **oof})
    metrics = {}
    near = y <= 50
    for name in oof:
        m = rul_metrics(y, preds[name])
        m["mae_when_true_rul_le_50"] = rul_metrics(y[near], preds.loc[near, name])["mae"] if near.any() else None
        m["per_battery"] = {b: rul_metrics(g["y_true"], g[name]) for b, g in preds.groupby("battery_id")}
        m["trained"] = name in models
        metrics[name] = m
    return preds, metrics
