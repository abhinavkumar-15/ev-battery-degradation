"""
DEMO / SAMPLE DATA generator  --  SYNTHETIC, NOT EXPERIMENTAL.

Purpose: let the whole stack (pipeline, API, UI, tests) run without the real NASA
download.  The cells below are produced by a simple, documented capacity-fade
simulator (16 cells, 2,440 cycle records) with noise, small capacity-regeneration bumps and per-cell operating
conditions.  Anything learned from this data reflects the simulator's own
assumptions, NOT real battery physics.  Never report demo metrics as results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_SAMPLE, SEED
from src.data.adapters.base import DatasetAdapter

# (id, ambient degC, discharge current A, fade-speed multiplier, n_cycles)
DEMO_CELLS = [
    ("D01", 24, 2.0, 1.00, 170), ("D02", 24, 2.0, 1.15, 170), ("D03", 24, 3.0, 1.45, 150),
    ("D04", 35, 2.0, 1.60, 140), ("D05", 35, 3.0, 2.05, 120), ("D06", 24, 1.0, 0.80, 200),
    ("D07", 43, 2.0, 2.30, 110), ("D08", 24, 2.5, 1.30, 160),
    # second batch (added to double the sample): wider mix of temperature / current,
    # plus D14, a deliberately gentle cell that never reaches the 80 % end-of-life threshold
    # (right-censored), so the RUL code path for censored cells is exercised.
    ("D09", 24, 1.5, 0.90, 165), ("D10", 30, 2.0, 1.35, 155), ("D11", 30, 3.0, 1.80, 150),
    ("D12", 35, 1.0, 1.25, 145), ("D13", 43, 3.0, 2.60, 130), ("D14", 15, 1.0, 0.20, 190),
    ("D15", 24, 2.0, 1.10, 115), ("D16", 35, 2.5, 1.85, 170),
]
RATED_AH = 2.0


def _simulate_cell(cid, amb, cur, speed, n, rng) -> pd.DataFrame:
    cyc = np.arange(1, n + 1)
    c0 = RATED_AH * rng.uniform(0.93, 1.01)
    # sub-linear fade that accelerates slowly (knee-like) + temperature/current stress through `speed`
    fade_frac = 0.0016 * speed * cyc ** 1.05 + 1.2e-6 * speed * cyc ** 2
    cap = c0 * (1 - fade_frac) + rng.normal(0, 0.006, n)
    bumps = (rng.random(n) < 0.04) * rng.uniform(0.01, 0.03, n)       # regeneration-like spikes
    cap = cap + bumps
    soh_proxy = cap / RATED_AH
    t_mean = amb + 2.0 + 1.5 * (cur - 1) + 3.0 * (1 - soh_proxy) + rng.normal(0, 0.25, n)
    dis_dur = cap / cur * 3600 * rng.normal(1, 0.003, n)
    re = 0.05 + 0.02 * (1 - soh_proxy) + rng.normal(0, 0.001, n)
    rct = 0.15 + 0.08 * (1 - soh_proxy) + rng.normal(0, 0.003, n)
    return pd.DataFrame({
        "battery_id": cid, "cycle": cyc, "capacity_ah": cap, "capacity_source": "synthetic",
        "ambient_temp_c": amb,
        "dis_duration_s": dis_dur,
        "dis_v_mean": 3.45 - 0.25 * (1 - soh_proxy) + rng.normal(0, 0.004, n),
        "dis_i_mean": cur + rng.normal(0, 0.01, n),
        "dis_t_mean": t_mean,
        "dis_throughput_ah": cap * rng.normal(1, 0.002, n),
        "re_ohm": re,
        "rct_ohm": rct,
    })


def generate_demo(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat([_simulate_cell(*c, rng) for c in DEMO_CELLS], ignore_index=True)
    df["dataset"] = "demo"
    return df


class DemoAdapter(DatasetAdapter):
    name = "demo"
    is_demo = True

    def load(self) -> pd.DataFrame:
        f = DATA_SAMPLE / "demo_cycles.csv"
        df = pd.read_csv(f) if f.exists() else generate_demo()
        df["dataset"] = "demo"
        return self.conform(df)


if __name__ == "__main__":
    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    out = DATA_SAMPLE / "demo_cycles.csv"
    generate_demo().round(6).to_csv(out, index=False)
    print("wrote", out)
