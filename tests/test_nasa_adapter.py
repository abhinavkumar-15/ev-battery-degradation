"""Exercises the NASA adapter on a tiny STRUCTURALLY-identical synthetic cell (not NASA data)."""
import numpy as np
import pytest
from scipy.io import savemat

from src.data.adapters.nasa import NasaAdapter, parse_nasa_cell


def fake_cycle(n_dis=3):
    t = np.arange(0, 3600, 10.0)
    recs = []
    for k in range(n_dis):
        recs.append({"type": "charge", "ambient_temperature": 24, "data": {
            "Time": t, "Voltage_measured": np.linspace(3.5, 4.2, t.size), "Current_measured": np.full(t.size, 1.5),
            "Temperature_measured": np.full(t.size, 25.0)}})
        recs.append({"type": "discharge", "ambient_temperature": 24, "data": {
            "Time": t, "Voltage_measured": np.linspace(4.2, 2.7, t.size), "Current_measured": np.full(t.size, -2.0),
            "Temperature_measured": np.linspace(25, 30, t.size), "Capacity": 2.0 - 0.1 * k}})
        if k == 0:
            recs.append({"type": "impedance", "ambient_temperature": 24, "data": {"Re": 0.05, "Rct": 0.07}})
    return {"cycle": recs}


def test_parse_cycle_definition_capacity_and_pairing():
    df = parse_nasa_cell(fake_cycle(), "B9999")
    assert df["cycle"].tolist() == [1, 2, 3]                       # n-th discharge
    assert df["capacity_ah"].round(3).tolist() == [2.0, 1.9, 1.8]
    assert (df["capacity_source"] == "dataset").all()
    assert df["chg_throughput_ah"].iloc[0] == pytest.approx(1.5, rel=0.02)   # paired charge step (1.5 A for 1 h)
    assert df["dis_t_max"].iloc[0] > df["dis_t_min"].iloc[0]


def test_missing_capacity_falls_back_to_coulomb_count_and_says_so():
    c = fake_cycle(1)
    del c["cycle"][1]["data"]["Capacity"]
    df = parse_nasa_cell(c, "B1")
    assert df["capacity_source"].iloc[0] == "coulomb_count" and df["capacity_ah"].iloc[0] > 1.9


def test_missing_files_give_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        NasaAdapter(root=tmp_path).load()


def test_mat_roundtrip_if_scipy_supports_layout(tmp_path):
    cell = fake_cycle(25)
    arr = np.empty(len(cell["cycle"]), dtype=object)
    for i, r in enumerate(cell["cycle"]):
        arr[i] = r
    savemat(tmp_path / "B0099.mat", {"B0099": {"cycle": arr}})
    out = NasaAdapter(root=tmp_path, batteries=None).load()
    assert out["battery_id"].unique().tolist() == ["B0099"] and out["cycle"].max() == 25
    # impedance was measured AFTER cycle 1 -> cycle 1 stays NaN (no look-ahead); later cycles are forward-filled
    assert out["re_ohm"].iloc[0] != out["re_ohm"].iloc[0] and out["re_ohm"].iloc[1:].notna().all()
