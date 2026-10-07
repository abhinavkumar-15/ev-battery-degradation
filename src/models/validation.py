"""Leakage-safe validation.

TEMPORAL LEAKAGE: a battery's rows are strongly autocorrelated (SOH at cycle 100 is
almost identical to SOH at cycle 101) and the targets look *forward in time*.  A random
row split would put cycle 101 in training and cycle 100 in test, so the model could
"interpolate" the test point from its neighbours and report a wildly optimistic score
that does not reflect forecasting a *new* battery or a *future* cycle.

DEFENCES USED HERE
 1. Features are trailing-window only (src/features/history.py).
 2. Evaluation is LEAVE-ONE-BATTERY-OUT: every battery is held out entirely, in turn;
    the model never sees any cycle of the battery it is scored on (Strategy A).
    With few cells (NASA core set = 4) a fixed train/val/test battery split would
    leave 1 cell per partition and be dominated by luck; LOBO uses every cell as test once.
 3. Hyper-parameters are tuned with GroupKFold *by battery* inside each training fold.
 4. `chronological_split` (Strategy B) is provided for single-battery forecasting.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def leave_one_battery_out(groups: pd.Series | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    g = np.asarray(groups)
    for b in pd.unique(g):
        test = np.where(g == b)[0]
        train = np.where(g != b)[0]
        if len(train) and len(test):
            yield train, test, str(b)


def chronological_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    """Per battery: first 70% of cycles -> train, next 15% -> val, last 15% -> test."""
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("invalid split fractions")
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.sort_values(["battery_id", "cycle"]).groupby("battery_id"):
        n = len(g)
        a, b = int(n * train), int(n * (train + val))
        parts["train"].append(g.iloc[:a]); parts["val"].append(g.iloc[a:b]); parts["test"].append(g.iloc[b:])
    return tuple(pd.concat(parts[k]) for k in ("train", "val", "test"))
