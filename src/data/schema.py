"""Common Battery Data Format (cycle-level).

Every dataset adapter must emit a DataFrame with (at least) the REQUIRED columns.
OPTIONAL columns may be NaN or absent; features that are unavailable are dropped
automatically downstream (never fabricated).

One row == one discharge cycle of one cell, paired with the charge step that
preceded it.  Units are SI-ish: Ah, V, A, degC, s, Wh, Ohm.
"""
from __future__ import annotations

REQUIRED_COLUMNS = ["dataset", "battery_id", "cycle", "capacity_ah"]

OPTIONAL_COLUMNS = [
    "ambient_temp_c",
    # discharge-step statistics
    "dis_duration_s", "dis_v_mean", "dis_v_min", "dis_v_max",
    "dis_i_mean", "dis_i_max", "dis_t_mean", "dis_t_max", "dis_t_min",
    "dis_energy_wh", "dis_throughput_ah",
    # charge-step statistics
    "chg_duration_s", "chg_v_mean", "chg_v_max", "chg_i_mean", "chg_i_max",
    "chg_t_mean", "chg_t_max", "chg_throughput_ah", "chg_energy_wh",
    # health indicators measured by impedance spectroscopy (forward-filled, past only)
    "re_ohm", "rct_ohm",
    # provenance
    "capacity_source",
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
