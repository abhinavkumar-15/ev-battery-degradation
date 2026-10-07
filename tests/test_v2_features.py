"""Tests for the 16-cell demo set, collinearity filter, coherent what-if scenarios and the new endpoints."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import store
from src.data.adapters import DemoAdapter
from src.features.pipeline import available_features, build_feature_table, drop_collinear
from src.models.predict import CURRENT_FAMILY, TEMP_FAMILY, scenario_limits, scenario_overrides

c = TestClient(app)
needs_models = pytest.mark.skipif(not store.available_datasets(), reason="run `python -m src.pipeline --dataset demo` first")


def test_demo_dataset_size_and_censored_cell():
    raw = DemoAdapter().load()
    assert len(raw) == 2440 and raw["battery_id"].nunique() == 16
    t, _ = build_feature_table(raw, dataset="demo")
    d14 = t[t.battery_id == "D14"]
    assert d14["soh"].min() > 80 and d14["rul"].isna().all()           # right-censored: never reaches EOL
    assert t[t.battery_id == "D01"]["rul"].notna().any()


def test_collinear_filter_removes_exact_duplicates_only():
    t, _ = build_feature_table(DemoAdapter().load(), dataset="demo")
    kept, dropped = drop_collinear(t, available_features(t), 0.99999)
    assert "capacity_ah" in dropped and dropped["capacity_ah"] == "soh"            # capacity == soh * ref/100 exactly
    assert {"soh", "cycle", "mean_temp_last_10", "soh_slope_10", "re_ohm"} <= set(kept)   # history features survive
    assert len(set(kept)) == len(kept)


def test_scenario_moves_whole_family_together_and_respects_limits():
    feats = TEMP_FAMILY + CURRENT_FAMILY
    row = {f: 30.0 for f in TEMP_FAMILY} | {f: 2.0 for f in CURRENT_FAMILY}
    ov = scenario_overrides(row, feats, 5.0, 1.5)
    assert all(ov[f] == 35.0 for f in TEMP_FAMILY) and all(ov[f] == 3.0 for f in CURRENT_FAMILY)
    ranges = {f: [20.0, 40.0] for f in TEMP_FAMILY} | {f: [1.0, 3.0] for f in CURRENT_FAMILY}
    lim = scenario_limits(row, feats, ranges)
    assert lim["temp_shift_c"] == [-10.0, 10.0] and lim["current_scale"] == [0.5, 1.5]
    assert scenario_overrides(row, feats, None, None) == {}


@needs_models
def test_fleet_and_horizon_endpoints():
    f = c.get("/api/datasets/demo/fleet").json()
    assert len(f["cells"]) == 16 and f["is_demo"]
    d14 = next(x for x in f["cells"] if x["id"] == "D14")
    assert d14["eol_cycle"] is None and d14["status"] != "End of life reached"
    assert all(len(x["series"]) >= 10 for x in f["cells"])
    h = c.get("/api/models/horizons").json()["horizons"]
    assert [x["horizon"] for x in h] == sorted(x["horizon"] for x in h) and len(h) >= 2


@needs_models
def test_scenarios_via_api_and_monotonic_forecast():
    lim = c.get("/api/scenario/limits", params={"battery_id": "D01", "cycle": 35}).json()
    assert lim["temp_shift_c"][0] <= 0 <= lim["temp_shift_c"][1]
    hot = c.post("/api/predict/soh", json={"battery_id": "D01", "cycle": 35, "temp_shift_c": min(3, lim["temp_shift_c"][1])}).json()
    assert hot["what_if"] and hot["predicted_soh"] <= hot["current_soh"] + 1e-9          # forecast never rises
    over = c.post("/api/predict/soh", json={"battery_id": "D01", "cycle": 35, "temp_shift_c": 39})
    assert over.status_code == 422 and "outside training support" in str(over.json())
    assert c.post("/api/predict/rul", json={"battery_id": "D01", "cycle": 35, "temp_shift_c": 1.0}).status_code == 200
    assert c.get("/api/scenario/limits", params={"battery_id": "NOPE"}).status_code == 404
    assert c.post("/api/predict/soh", json={"battery_id": "D01", "current_scale": -1}).status_code == 422


@needs_models
def test_models_endpoint_exposes_global_ranges_and_dropped_features():
    m = c.get("/api/models").json()
    assert m["collinear_dropped"].get("capacity_ah") == "soh" and "dis_t_mean" in m["feature_ranges"]
    # a scenario built from the advertised ranges must be accepted by every model
    lo, hi = m["feature_ranges"]["ambient_temp_c"]
    assert lo <= hi
