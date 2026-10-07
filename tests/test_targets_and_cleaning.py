import numpy as np
import pandas as pd
import pytest

from src.features.targets import add_degradation_rates, add_rul, compute_soh
from src.preprocessing.clean import clean_cycle_table


def mk(caps, bid="A"):
    return pd.DataFrame({"dataset": "t", "battery_id": bid, "cycle": np.arange(1, len(caps) + 1), "capacity_ah": caps})


def test_soh_rated_and_first_valid():
    df = mk([2.0, 1.8, 1.6])
    assert compute_soh(df, "rated", 2.0)["soh"].round(6).tolist() == [100.0, 90.0, 80.0]
    assert compute_soh(df, "first_valid")["soh"].iloc[-1] == pytest.approx(80.0)
    with pytest.raises(ValueError):
        compute_soh(df, "rated", None)          # never silently invent a reference capacity
    assert compute_soh(df, "rated", 2.0)["capacity_loss"].iloc[1] == pytest.approx(10.0)


def test_degradation_rate_matches_definition():
    t = add_degradation_rates(compute_soh(mk([2.0, 1.9, 1.7, 1.7]), "rated", 2.0), windows=(3,))
    assert np.isnan(t["degradation_rate"].iloc[0])
    assert t["degradation_rate"].iloc[1:].round(6).tolist() == [5.0, 10.0, 0.0]
    assert t["rolling_degradation_rate_3"].iloc[3] == pytest.approx(5.0)


def test_rul_definition_and_threshold_configurable():
    t = add_rul(compute_soh(mk([2.0, 1.9, 1.7, 1.5, 1.4]), "rated", 2.0), 80.0)   # SOH 100,95,85,75,70
    assert t["eol_cycle"].iloc[0] == 4
    assert t["rul"].iloc[:4].tolist() == [3, 2, 1, 0]
    assert np.isnan(t["rul"].iloc[4])                                    # after EOL: not used
    assert add_rul(compute_soh(mk([2.0, 1.9, 1.7, 1.5, 1.4]), "rated", 2.0), 70.0)["eol_cycle"].iloc[0] == 5
    assert add_rul(compute_soh(mk([2.0, 1.9]), "rated", 2.0), 80.0)["rul"].isna().all()   # censored


def test_cleaning_drops_and_reports():
    df = mk(list(np.linspace(2, 1.5, 30)))
    df.loc[3, "capacity_ah"] = np.nan
    df.loc[5, "capacity_ah"] = -1
    df = pd.concat([df, df.iloc[[10]]])                                  # duplicate
    out, rep = clean_cycle_table(df)
    assert len(out) == 28 and rep["dropped"]["duplicate (battery, cycle)"] == 1
    assert rep["dropped"]["non-positive capacity"] == 1


def test_cleaning_rejects_empty_and_malformed():
    with pytest.raises(ValueError):
        clean_cycle_table(pd.DataFrame())
    with pytest.raises(ValueError):
        clean_cycle_table(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        clean_cycle_table(mk([2.0, 1.9]))                                # < min cycles => nothing usable
    with pytest.raises(ValueError):
        clean_cycle_table(mk(["bad"] * 30))                              # non-numeric capacity


def test_capacity_jumps_flagged_not_deleted():
    caps = list(np.linspace(2, 1.6, 30)); caps[15] = caps[14] * 1.2
    out, rep = clean_cycle_table(mk(caps))
    assert len(out) == 30 and out["capacity_jump_flag"].sum() >= 1
