import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data.adapters import DemoAdapter
from src.features.pipeline import available_features, build_feature_table, modelling_rows
from src.models.validation import chronological_split, leave_one_battery_out

FEAT_CHECK = ["mean_temp_last_10", "soh_slope_10", "rolling_degradation_rate_10", "cum_discharge_throughput_ah"]


def _table():
    return build_feature_table(DemoAdapter().load(), PipelineConfig(), dataset="demo")[0]


def test_features_are_trailing_only_no_future_leakage():
    raw = DemoAdapter().load()
    full, _ = build_feature_table(raw, PipelineConfig(), dataset="demo")
    # Truncate cell D01 after cycle 100, rebuild: features at cycles <= 100 must be identical.
    cut = raw[~((raw["battery_id"] == "D01") & (raw["cycle"] > 100))]
    part, _ = build_feature_table(cut, PipelineConfig(), dataset="demo")
    a = full[(full.battery_id == "D01") & (full.cycle <= 100)].set_index("cycle")[FEAT_CHECK]
    b = part[(part.battery_id == "D01") & (part.cycle <= 100)].set_index("cycle")[FEAT_CHECK]
    pd.testing.assert_frame_equal(a, b)


def test_future_target_alignment_and_warmup():
    t = _table()
    g = t[t.battery_id == "D01"].sort_values("cycle")
    assert g["soh_future_h10"].iloc[0] == g["soh"].iloc[10]
    assert g["soh_future_h10"].iloc[-10:].isna().all()
    assert t.loc[t.is_warmup].groupby("battery_id").size().eq(PipelineConfig().warmup_cycles).all()
    assert modelling_rows(t, 10)["soh_future_h10"].notna().all()


def test_unavailable_features_are_dropped_not_invented():
    raw = DemoAdapter().load()
    raw[["re_ohm", "rct_ohm"]] = np.nan
    t, _ = build_feature_table(raw, PipelineConfig(), dataset="demo")
    feats = available_features(t)
    assert "re_ohm" not in feats and "rct_ohm" not in feats and "cycle" in feats


def test_leave_one_battery_out_never_shares_a_battery():
    g = np.array(list("AAABBBCCC"))
    folds = list(leave_one_battery_out(g))
    assert len(folds) == 3
    for tr, te, b in folds:
        assert set(g[tr]).isdisjoint(set(g[te])) and set(g[te]) == {b}


def test_chronological_split_is_ordered_per_battery():
    t = _table()
    tr, va, te = chronological_split(t)
    for b in t.battery_id.unique():
        assert tr[tr.battery_id == b].cycle.max() < va[va.battery_id == b].cycle.min() < te[te.battery_id == b].cycle.min()
