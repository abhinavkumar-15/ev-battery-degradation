"""Inference helpers shared by the pipeline, the API and the tests."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ArtifactMissing(RuntimeError):
    pass


def load_artifact(path: Path | str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ArtifactMissing(f"Model artifact not found: {p}. Run `python -m src.pipeline --dataset <name>` first.")
    return joblib.load(p)


def check_feature_ranges(art: dict, overrides: dict[str, float]) -> list[str]:
    """Return human-readable violations for what-if values outside the training range."""
    bad = []
    for k, v in overrides.items():
        if k not in art["feature_ranges"]:
            bad.append(f"unknown feature '{k}'")
            continue
        lo, hi = art["feature_ranges"][k]
        if not (lo <= v <= hi):
            bad.append(f"{k}={v} is outside the training range [{lo:.4g}, {hi:.4g}]")
    return bad


# Feature families moved TOGETHER by a what-if scenario, so the scenario feature vector stays internally consistent
# (a hotter cell has a hotter ambient, mean, peak and rolling-mean temperature - not just one of them).
TEMP_FAMILY = ["ambient_temp_c", "dis_t_mean", "dis_t_max", "chg_t_mean", "mean_temp_last_10", "mean_temp_last_25", "max_temp_last_25"]
CURRENT_FAMILY = ["dis_i_mean", "dis_i_max", "mean_current_last_10"]   # discharge load only; charging is not rescaled


def scenario_limits(row, features: list[str], ranges: dict | None = None) -> dict:
    """Physics-plausible slider ranges for what-if scenarios."""
    r = dict(row)
    t_feats = [f for f in TEMP_FAMILY if f in features and np.isfinite(r.get(f, np.nan))]
    c_feats = [f for f in CURRENT_FAMILY if f in features and np.isfinite(r.get(f, np.nan)) and r[f] > 0]

    t_lim = None
    if t_feats:
        if ranges and any(f in ranges for f in t_feats):
            lo_shifts = [ranges[f][0] - r[f] for f in t_feats if f in ranges]
            hi_shifts = [ranges[f][1] - r[f] for f in t_feats if f in ranges]
            t_lim = [round(float(max(lo_shifts)), 1), round(float(min(hi_shifts)), 1)]
        else:
            t_lim = [-15.0, 15.0]

    c_lim = None
    if c_feats:
        if ranges and any(f in ranges for f in c_feats):
            lo_scales = [ranges[f][0] / r[f] for f in c_feats if f in ranges]
            hi_scales = [ranges[f][1] / r[f] for f in c_feats if f in ranges]
            c_lim = [round(float(max(lo_scales)), 2), round(float(min(hi_scales)), 2)]
        else:
            c_lim = [0.5, 2.0]

    return {"temp_shift_c": t_lim, "current_scale": c_lim}


def scenario_overrides(row, features: list[str], temp_shift_c: float | None, current_scale: float | None, ranges: dict | None = None) -> dict:
    """Shift temperature / current features.  Always clamp to training range."""
    r, ov = dict(row), {}
    if temp_shift_c:
        for f in TEMP_FAMILY:
            if f in features and np.isfinite(r.get(f, np.nan)):
                val = r[f] + temp_shift_c
                if ranges and f in ranges:
                    val = float(np.clip(val, ranges[f][0], ranges[f][1]))
                ov[f] = val
    if current_scale and current_scale != 1.0:
        for f in CURRENT_FAMILY:
            if f in features and np.isfinite(r.get(f, np.nan)):
                val = r[f] * current_scale
                if ranges and f in ranges:
                    val = float(np.clip(val, ranges[f][0], ranges[f][1]))
                ov[f] = val
    return ov


def physical_stress_multiplier(temp_shift_c: float | None = 0.0, current_scale: float | None = 1.0) -> float:
    """Calculates Arrhenius thermal acceleration and C-rate stress multiplier.
    - Hotter & Heavier (temp_shift > 0 or current_scale > 1) -> gamma > 1.0 (faster fade, lower SOH, shorter RUL)
    - Cooler & Gentler (temp_shift < 0 or current_scale < 1) -> gamma < 1.0 (slower fade, higher SOH, longer RUL)
    """
    dt = float(temp_shift_c or 0.0)
    cs = float(current_scale if current_scale is not None else 1.0)
    gamma_t = np.exp(0.04 * dt)
    gamma_i = max(0.1, cs) ** 1.2
    return float(gamma_t * gamma_i)


def predict_future_soh(art: dict, row: pd.Series | dict, overrides: dict | None = None, temp_shift_c: float | None = None, current_scale: float | None = None) -> dict:
    r = dict(row)
    base_x = pd.DataFrame([{f: r.get(f, np.nan) for f in art["features"]}])
    base_raw_delta = float(art["pipeline"].predict(base_x)[0])
    base_delta = min(base_raw_delta, 0.0)

    gamma = physical_stress_multiplier(temp_shift_c, current_scale)
    scenario_delta = base_delta * gamma if (temp_shift_c or (current_scale and current_scale != 1.0)) else base_delta

    delta = min(scenario_delta, 0.0)
    soh_now = float(r["soh"])
    lo_q, hi_q = art["residual_quantiles"]
    pred = float(np.clip(soh_now + delta, 0, 120))
    return {"horizon": art["horizon"], "current_soh": soh_now, "predicted_soh": pred, "predicted_delta": delta, "raw_model_delta": scenario_delta, "monotonic_constraint_applied": scenario_delta > 0,
            "stress_multiplier": gamma,
            "interval_low": float(np.clip(pred + lo_q, 0, 120)), "interval_high": float(np.clip(pred + hi_q, 0, 120)),
            "model": art["model_name"]}
