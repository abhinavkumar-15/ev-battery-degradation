"""Target construction: SOH, capacity loss, degradation rate, RUL.

Definitions
-----------
SOH(%)              = capacity_ah / reference_capacity_ah * 100
capacity_loss(%)    = 100 - SOH
degradation_rate    = (SOH[t-1] - SOH[t]) / (cycle[t] - cycle[t-1])    [% SOH per cycle]
rolling_rate_w      = trailing mean of degradation_rate over w cycles (past only)
RUL[t]              = first_cycle_with_SOH<=EOL - cycle[t]  (>=0; NaN if the cell never reaches EOL)

Reference capacity ("rated" mode) uses the dataset's documented rated capacity.
"first_valid" mode uses the first valid capacity of each cell - a documented
fallback, chosen explicitly via config, never silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_soh(df: pd.DataFrame, reference_mode: str = "rated", rated_capacity_ah: float | None = None) -> pd.DataFrame:
    out = df.copy()
    if reference_mode == "rated":
        if not rated_capacity_ah or rated_capacity_ah <= 0:
            raise ValueError("reference_mode='rated' requires a positive rated_capacity_ah")
        out["reference_capacity_ah"] = float(rated_capacity_ah)
    elif reference_mode == "first_valid":
        out = out.sort_values(["battery_id", "cycle"])
        out["reference_capacity_ah"] = out.groupby("battery_id")["capacity_ah"].transform("first")
    else:
        raise ValueError(f"unknown reference_mode {reference_mode!r}")
    out["soh"] = out["capacity_ah"] / out["reference_capacity_ah"] * 100.0
    out["capacity_loss"] = 100.0 - out["soh"]
    return out


def add_degradation_rates(df: pd.DataFrame, windows=(10, 25, 50)) -> pd.DataFrame:
    out = df.sort_values(["battery_id", "cycle"]).copy()
    g = out.groupby("battery_id", sort=False)
    dc = g["cycle"].diff()
    out["degradation_rate"] = (-g["soh"].diff()) / dc        # positive == losing health
    g = out.groupby("battery_id", sort=False)["degradation_rate"]
    for w in windows:
        out[f"rolling_degradation_rate_{w}"] = g.transform(lambda s: s.rolling(w, min_periods=3).mean())
    return out


def add_rul(df: pd.DataFrame, eol_threshold: float = 80.0) -> pd.DataFrame:
    """Ground-truth RUL from the *observed* trajectory (used only as a training/eval label)."""
    out = df.sort_values(["battery_id", "cycle"]).copy()
    eol_cycle = {}
    for bid, g in out.groupby("battery_id"):
        hit = g.loc[g["soh"] <= eol_threshold, "cycle"]
        eol_cycle[bid] = float(hit.iloc[0]) if len(hit) else np.nan
    out["eol_cycle"] = out["battery_id"].map(eol_cycle)
    rul = out["eol_cycle"] - out["cycle"]
    out["rul"] = rul.where(rul >= 0)            # after EOL -> NaN (not used)
    return out
