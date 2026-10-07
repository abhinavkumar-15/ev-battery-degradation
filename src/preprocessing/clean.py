"""Cycle-table cleaning + validation.  Returns the cleaned table AND a quality report
(what was dropped/flagged and why) so nothing is silently discarded."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import REQUIRED_COLUMNS


def clean_cycle_table(df: pd.DataFrame, min_cycles: int = 20) -> tuple[pd.DataFrame, dict]:
    if df is None or len(df) == 0:
        raise ValueError("Empty dataset: nothing to clean")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    report: dict = {"rows_in": int(len(df)), "dropped": {}, "flags": {}}
    out = df.copy()
    out["capacity_ah"] = pd.to_numeric(out["capacity_ah"], errors="coerce")
    out["cycle"] = pd.to_numeric(out["cycle"], errors="coerce")

    def drop(mask: pd.Series, why: str):
        n = int(mask.sum())
        if n:
            report["dropped"][why] = report["dropped"].get(why, 0) + n
        return out.loc[~mask]

    out = drop(out["capacity_ah"].isna() | out["cycle"].isna(), "missing capacity or cycle")
    out = drop(out["capacity_ah"] <= 0, "non-positive capacity")
    out = drop(out.duplicated(["dataset", "battery_id", "cycle"], keep="last"), "duplicate (battery, cycle)")
    out = out.sort_values(["dataset", "battery_id", "cycle"]).reset_index(drop=True)

    # Flag (do NOT delete) capacity jumps: real cells show "capacity regeneration" after rest.
    jump = out.groupby(["dataset", "battery_id"])["capacity_ah"].diff().abs()
    out["capacity_jump_flag"] = (jump > 0.10 * out["capacity_ah"]).fillna(False)
    report["flags"]["capacity_jumps(>10%)"] = int(out["capacity_jump_flag"].sum())

    # Physically implausible electrical/thermal readings -> NaN (kept row, value blanked)
    bounds = {"dis_v_mean": (0.5, 5.0), "dis_t_mean": (-40, 120), "chg_t_mean": (-40, 120),
              "dis_i_mean": (0, 100), "chg_i_mean": (0, 100)}
    for col, (lo, hi) in bounds.items():
        if col in out:
            bad = (out[col] < lo) | (out[col] > hi)
            if bad.any():
                report["flags"][f"{col}_out_of_range->NaN"] = int(bad.sum())
                out.loc[bad, col] = np.nan

    # Cells with too few cycles cannot support trailing-window features
    counts = out.groupby(["dataset", "battery_id"]).size()
    short = counts[counts < min_cycles]
    if len(short):
        report["dropped"]["cells with < %d cycles" % min_cycles] = [f"{d}/{b}" for d, b in short.index]
        keep = out.set_index(["dataset", "battery_id"]).index.isin(counts[counts >= min_cycles].index)
        out = out.loc[keep]

    if out.empty:
        raise ValueError("No usable data left after cleaning")
    report["rows_out"] = int(len(out))
    report["batteries"] = sorted(out["battery_id"].unique().tolist())
    report["missing_fraction"] = {c: round(float(v), 4) for c, v in out.isna().mean().items() if v > 0}
    return out.reset_index(drop=True), report
