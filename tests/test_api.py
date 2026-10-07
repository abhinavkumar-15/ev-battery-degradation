import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import store
from src.models.predict import ArtifactMissing, load_artifact

c = TestClient(app)
pytestmark = pytest.mark.skipif(not store.available_datasets(), reason="run `python -m src.pipeline --dataset demo` first")


def test_health_and_datasets_label_demo():
    assert c.get("/api/health").json()["status"] == "ok"
    d = c.get("/api/datasets").json()["datasets"][0]
    assert d["is_demo"] and "Demo" in d["label"]


def test_battery_endpoints():
    ds = store.default_dataset()
    b = c.get(f"/api/datasets/{ds}/batteries").json()["batteries"]
    assert len(b) >= 3
    bid = b[0]["id"]
    assert 0 < c.get(f"/api/batteries/{bid}/summary").json()["current_soh"] <= 110
    h = c.get(f"/api/batteries/{bid}/history", params={"cycle_min": 10, "cycle_max": 20}).json()["records"]
    assert [r["cycle"] for r in h] == list(range(10, 21))


def test_predictions_have_sane_shape():
    bid = c.get("/api/datasets/demo/batteries").json()["batteries"][1]["id"]
    p = c.post("/api/predict/soh", json={"battery_id": bid, "cycle": 40}).json()
    assert p["interval_low"] <= p["predicted_soh"] <= p["interval_high"] and p["is_demo"]
    assert c.post("/api/predict/degradation", json={"battery_id": bid, "cycle": 40}).json()["kind"] == "predicted"
    r = c.post("/api/predict/rul", json={"battery_id": bid, "cycle": 40}).json()
    assert "80% SOH end-of-life threshold" in r["statement"] and r["ml_rul_cycles"] >= 0
    r2 = c.post("/api/predict/rul", json={"battery_id": bid, "cycle": 40, "eol_threshold": 70}).json()
    assert r2["ml_rul_cycles"] is None and r2["extrapolation_rul_cycles"] >= r["extrapolation_rul_cycles"]


def test_error_handling():
    assert c.post("/api/predict/soh", json={"battery_id": "NOPE"}).status_code == 404
    assert c.post("/api/predict/soh", json={"horizon": 5}).status_code == 422            # malformed: no battery_id
    assert c.post("/api/predict/soh", json={"battery_id": "D01", "horizon": 7}).status_code == 422
    assert c.get("/api/batteries/NOPE/summary").status_code == 404
    assert c.get("/api/datasets/nope/batteries").status_code == 404
    bad = c.post("/api/predict/soh", json={"battery_id": "D01", "overrides": {"dis_t_mean": 500}})
    assert bad.status_code == 422                                                      # what-if outside training support


def test_explainability_and_performance():
    e = c.get("/api/explainability").json()
    assert e["importance"][0]["mean_abs_shap"] >= e["importance"][-1]["mean_abs_shap"] and "causation" in e["interpretation"]
    perf = c.get("/api/models/performance").json()
    assert {"linear_regression", "random_forest"} <= set(perf["forecast"])


def test_model_loading_and_missing_model(tmp_path):
    with pytest.raises(ArtifactMissing):
        load_artifact(tmp_path / "nope.joblib")
    assert load_artifact(store.MODELS_DIR / "demo" / "forecast_h10.joblib")["is_demo"] is True


def test_prediction_before_model_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "MODELS_DIR", tmp_path)
    assert c.get("/api/health").json()["status"] == "no_models"
    assert c.post("/api/predict/soh", json={"battery_id": "D01"}).status_code == 503
