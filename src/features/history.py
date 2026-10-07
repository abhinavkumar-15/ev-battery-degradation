"""Historical (trailing-window) features.

Every window is TRAILING and includes the current cycle t but nothing after it, so a
row at cycle t only contains information that existed at cycle t.  This is the
feature-level half of the temporal-leakage defence (the other half is the
battery-level validation split, see src/models/validation.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Documentation of every engineered feature (surfaced in README / API / UI).
FEATURE_DOCS: dict[str, str] = {
    "cycle": "Discharge-cycle index (ageing clock).",
    "soh": "Current State of Health (%), derived from measured capacity / reference capacity.",
    "capacity_ah": "Measured discharge capacity (Ah).",
    "ambient_temp_c": "Chamber/ambient temperature (degC) as documented by the dataset.",
    "dis_duration_s": "Duration of the discharge step (s).",
    "dis_v_mean": "Mean terminal voltage during discharge (V).",
    "dis_v_min": "Minimum voltage during discharge (V) (~ cut-off voltage).",
    "dis_v_range": "dis_v_max - dis_v_min (V).",
    "dis_i_mean": "Mean |current| during discharge (A).",
    "dis_i_max": "Peak |current| during discharge (A).",
    "dis_t_mean": "Mean cell temperature during discharge (degC).",
    "dis_t_max": "Peak cell temperature during discharge (degC).",
    "dis_t_range": "dis_t_max - dis_t_min (degC).",
    "dis_energy_wh": "Discharge energy, integral of |V*I| dt (Wh).",
    "chg_duration_s": "Duration of the preceding charge step (s).",
    "chg_i_mean": "Mean |current| during charge (A).",
    "chg_t_mean": "Mean cell temperature during charge (degC).",
    "chg_energy_wh": "Charge energy (Wh).",
    "coulombic_efficiency": "discharge Ah / charge Ah of the paired steps (derived).",
    "re_ohm": "Electrolyte resistance Re from impedance spectroscopy (Ohm), forward-filled from the latest earlier measurement.",
    "rct_ohm": "Charge-transfer resistance Rct (Ohm), forward-filled from the latest earlier measurement.",
    "cum_discharge_throughput_ah": "Cumulative discharged Ah up to and including cycle t.",
    "cum_charge_throughput_ah": "Cumulative charged Ah up to and including cycle t.",
    "cum_discharge_energy_wh": "Cumulative discharge energy (Wh) up to cycle t.",
    "equivalent_full_cycles": "cum_discharge_throughput_ah / reference capacity.",
    "soh_slope_10": "Least-squares slope of SOH vs cycle over the last 10 cycles (%/cycle).",
    "soh_slope_25": "Least-squares slope of SOH vs cycle over the last 25 cycles (%/cycle).",
    "mean_temp_last_10": "Mean of dis_t_mean over the last 10 cycles.",
    "mean_temp_last_25": "Mean of dis_t_mean over the last 25 cycles.",
    "max_temp_last_25": "Max of dis_t_max over the last 25 cycles.",
    "mean_current_last_10": "Mean of dis_i_mean over the last 10 cycles.",
    "rolling_degradation_rate_10": "Trailing 10-cycle mean of the per-cycle degradation rate (%/cycle).",
    "rolling_degradation_rate_25": "Trailing 25-cycle mean of the per-cycle degradation rate (%/cycle).",
    "rolling_degradation_rate_50": "Trailing 50-cycle mean of the per-cycle degradation rate (%/cycle).",
}

# Considered but intentionally NOT created (documented so the omission is defensible):
EXCLUDED_FEATURES = {
    "capacity_slope_*": "Exactly proportional to soh_slope_* when the reference capacity is constant (perfect collinearity, no new information).",
    "cycles_since_reference_point": "Identical to cycle - first_cycle; redundant with `cycle`.",
    "depth_of_discharge": "All NASA cycles discharge to a fixed cut-off voltage, so DoD ~ capacity/rated ~ SOH; adds no independent information.",
    "dis_t_rise": "With only per-step max/min stored, temperature rise == temperature range (identical column), so only dis_t_range is kept.",
    "internal_resistance (DC)": "Not directly measured in NASA aging files; Re/Rct from impedance spectroscopy are used instead.",
}


def _trailing_slope(s: pd.Series, x: pd.Series, w: int) -> pd.Series:
    """OLS slope of s on x over a trailing window of w rows (min 3 points)."""
    ys, xs = s.to_numpy(float), x.to_numpy(float)
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        lo = max(0, i - w + 1)
        xx, yy = xs[lo:i + 1], ys[lo:i + 1]
        ok = np.isfinite(xx) & np.isfinite(yy)
        if ok.sum() >= 3 and np.ptp(xx[ok]) > 0:
            out[i] = np.polyfit(xx[ok], yy[ok], 1)[0]
    return pd.Series(out, index=s.index)


def add_history_features(df: pd.DataFrame, windows=(10, 25, 50)) -> pd.DataFrame:
    out = df.sort_values(["battery_id", "cycle"]).copy()

    # per-cycle derived quantities (if source columns exist)
    if "dis_v_max" in out and "dis_v_min" in out:
        out["dis_v_range"] = out["dis_v_max"] - out["dis_v_min"]
    if "dis_t_max" in out and "dis_t_min" in out:
        out["dis_t_range"] = out["dis_t_max"] - out["dis_t_min"]
    if "dis_throughput_ah" in out and "chg_throughput_ah" in out:
        out["coulombic_efficiency"] = (out["dis_throughput_ah"] / out["chg_throughput_ah"]).replace([np.inf, -np.inf], np.nan)

    parts = []
    for _, g in out.groupby("battery_id", sort=False):
        g = g.copy()
        dis_tp = g["dis_throughput_ah"] if "dis_throughput_ah" in g else g["capacity_ah"]
        g["cum_discharge_throughput_ah"] = dis_tp.fillna(g["capacity_ah"]).cumsum()
        if "chg_throughput_ah" in g:
            g["cum_charge_throughput_ah"] = g["chg_throughput_ah"].fillna(0).cumsum()
        if "dis_energy_wh" in g:
            g["cum_discharge_energy_wh"] = g["dis_energy_wh"].fillna(0).cumsum()
        g["equivalent_full_cycles"] = g["cum_discharge_throughput_ah"] / g["reference_capacity_ah"]
        for w in (10, 25):
            g[f"soh_slope_{w}"] = _trailing_slope(g["soh"], g["cycle"], w)
        if "dis_t_mean" in g:
            g["mean_temp_last_10"] = g["dis_t_mean"].rolling(10, min_periods=3).mean()
            g["mean_temp_last_25"] = g["dis_t_mean"].rolling(25, min_periods=3).mean()
        if "dis_t_max" in g:
            g["max_temp_last_25"] = g["dis_t_max"].rolling(25, min_periods=3).max()
        if "dis_i_mean" in g:
            g["mean_current_last_10"] = g["dis_i_mean"].rolling(10, min_periods=3).mean()
        parts.append(g)
    return pd.concat(parts).sort_index()
