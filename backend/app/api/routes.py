from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.models import EvCalculatorRequest, PredictRequest, RulRequest
from backend.app.services import store
from src.models.ev_calculator import PRESETS, predict_ev_health
from src.explainability.shap_explain import local_explanation
from src.models.predict import (
    ArtifactMissing,
    check_feature_ranges,
    physical_stress_multiplier,
    predict_future_soh,
    scenario_limits,
    scenario_overrides,
)
from src.models.rul import extrapolate_rul

router = APIRouter(prefix="/api")

LABEL_DEMO = "Demo / Sample Data (synthetic) - not experimental results"


def _guard(fn):
    """Translate domain errors into clean HTTP errors."""
    try:
        return fn()
    except store.NotFound as e:
        raise HTTPException(404, str(e.args[0]))
    except ArtifactMissing as e:
        raise HTTPException(503, str(e))


def _ds_info(ds: str) -> dict:
    m = store.meta(ds)
    return {"id": ds, "is_demo": m["is_demo"], "label": LABEL_DEMO if m["is_demo"] else ds.upper(),
            "n_batteries": m["n_batteries"], "n_cycles": m["n_cycles"], "eol_threshold": m["config"]["eol_threshold_soh"],
            "horizons": store.horizons(ds), "headline_horizon": m["headline_horizon"]}


@router.get("/health")
def health():
    ds = store.available_datasets()
    return {"status": "ok" if ds else "no_models", "datasets": ds,
            "message": None if ds else "No trained models found. Run the training pipeline (see README)."}


@router.get("/datasets")
def datasets():
    return _guard(lambda: {"datasets": [_ds_info(d) for d in store.available_datasets()]})


@router.get("/datasets/{dataset_id}/batteries")
def batteries(dataset_id: str):
    def go():
        ds = store.resolve(dataset_id)
        t = store.table(ds)
        g = t.groupby("battery_id").agg(cycles=("cycle", "max"), first_soh=("soh", "first"), last_soh=("soh", "last"),
                                         ambient_temp_c=("ambient_temp_c", "median"))
        return store.clean({"dataset": ds, "batteries": [{"id": str(i), **r} for i, r in g.reset_index().set_index("battery_id").iterrows()]})
    return _guard(go)


@router.get("/datasets/{dataset_id}/fleet")
def fleet(dataset_id: str, points: int = Query(60, ge=10, le=300)):
    """All cells at a glance: down-sampled SOH trajectories + status, for the fleet comparison view."""
    def go():
        ds = store.resolve(dataset_id)
        t = store.table(ds)
        eol = float(store.meta(ds)["config"]["eol_threshold_soh"])
        cells = []
        for bid, g in t.groupby("battery_id", sort=True):
            g = g.sort_values("cycle")
            hit = g.loc[g["soh"] <= eol, "cycle"]
            last = g.iloc[-1]
            idx = np.unique(np.linspace(0, len(g) - 1, min(points, len(g))).astype(int))
            status = "End of life reached" if last["soh"] <= eol else ("Healthy" if last["soh"] >= eol + 10 else "Ageing - approaching EOL")
            cells.append({"id": str(bid), "cycles": int(last["cycle"]), "final_soh": float(last["soh"]), "eol_cycle": float(hit.iloc[0]) if len(hit) else None,
                          "status": status, "ambient_temp_c": g["ambient_temp_c"].median(), "mean_discharge_current_a": g["dis_i_mean"].median(),
                          "mean_rate_pct_per_cycle": float((g["soh"].iloc[0] - last["soh"]) / max(1, last["cycle"] - g["cycle"].iloc[0])),
                          "series": g.iloc[idx][["cycle", "soh"]].to_dict("records")})
        return store.clean({"dataset": ds, "is_demo": store.meta(ds)["is_demo"], "eol_threshold": eol, "cells": cells})
    return _guard(go)


@router.get("/datasets/{dataset_id}/summary")
def dataset_summary(dataset_id: str):
    def go():
        ds = store.resolve(dataset_id)
        t = store.table(ds)
        cols = [c for c in ("capacity_ah", "soh", "dis_t_mean", "dis_i_mean", "dis_v_mean", "re_ohm", "rct_ohm", "dis_duration_s") if c in t]
        stats = t[cols].describe().T[["count", "mean", "std", "min", "max"]]
        m = store.meta(ds)
        return store.clean({"dataset": ds, "is_demo": m["is_demo"], "stats": stats.reset_index().rename(columns={"index": "column"}).to_dict("records"),
                            "quality": m["quality_report"], "capacity_source": t["capacity_source"].value_counts().to_dict()})
    return _guard(go)


@router.get("/batteries/{battery_id}/summary")
def battery_summary(battery_id: str, dataset: str | None = None):
    def go():
        ds = store.resolve(dataset, battery_id)
        g = store.battery_rows(ds, battery_id)
        eol = float(store.meta(ds)["config"]["eol_threshold_soh"])
        last = g.iloc[-1]
        rate = last.get("rolling_degradation_rate_25")
        reached = g.loc[g["soh"] <= eol, "cycle"]
        status = "End of life reached" if last["soh"] <= eol else ("Healthy" if last["soh"] >= eol + 10 else "Ageing - approaching EOL")
        return store.clean({"dataset": ds, "is_demo": store.meta(ds)["is_demo"], "battery_id": battery_id, "cycles": int(last["cycle"]),
                            "current_soh": float(last["soh"]), "current_capacity_ah": float(last["capacity_ah"]),
                            "reference_capacity_ah": float(last["reference_capacity_ah"]), "degradation_rate_pct_per_cycle": rate,
                            "eol_threshold": eol, "eol_cycle_observed": float(reached.iloc[0]) if len(reached) else None, "status": status,
                            "provenance": {"measured": ["capacity_ah", "dis_t_mean", "dis_i_mean"], "derived": ["soh", "degradation_rate"],
                                           "predicted": []}})
    return _guard(go)


@router.get("/batteries/{battery_id}/history")
def history(battery_id: str, dataset: str | None = None, cycle_min: int = Query(1, ge=1), cycle_max: int | None = Query(None, ge=1),
            stride: int = Query(1, ge=1, le=100)):
    def go():
        ds = store.resolve(dataset, battery_id)
        g = store.battery_rows(ds, battery_id)
        g = g[g["cycle"] >= cycle_min]
        if cycle_max:
            g = g[g["cycle"] <= cycle_max]
        g = g.iloc[::stride]
        cols = ["cycle", "capacity_ah", "soh", "capacity_loss", "degradation_rate", "rolling_degradation_rate_10", "rolling_degradation_rate_25",
                "dis_t_mean", "dis_t_max", "dis_i_mean", "dis_v_mean", "dis_v_min", "dis_duration_s", "chg_duration_s", "re_ohm", "rct_ohm",
                "cum_discharge_throughput_ah", "capacity_jump_flag", "is_warmup"]
        return store.clean({"dataset": ds, "is_demo": store.meta(ds)["is_demo"], "battery_id": battery_id,
                            "records": g[[c for c in cols if c in g]].to_dict("records")})
    return _guard(go)


def _row(ds: str, battery_id: str, cycle: int | None):
    g = store.battery_rows(ds, battery_id)
    g = g[~g["is_warmup"].astype(bool)]
    if g.empty:
        raise store.NotFound(f"Battery '{battery_id}' has too few cycles for feature windows")
    if cycle is not None:
        g = g[g["cycle"] == cycle]
        if g.empty:
            raise store.NotFound(f"Cycle {cycle} not available for '{battery_id}'")
    return g.iloc[-1]


def _global_ranges(ds: str) -> dict:
    """Intersection of the training ranges of every model (each horizon / RUL model saw slightly different rows)."""
    sets = [store.artifact(ds, f"forecast_h{h}")["feature_ranges"] for h in store.horizons(ds)]
    try:
        sets.append(store.artifact(ds, "rul")["feature_ranges"])
    except ArtifactMissing:
        pass
    return {f: [max(r[f][0] for r in sets), min(r[f][1] for r in sets)] for f in sets[0] if all(f in r for r in sets)}


def _resolve_overrides(ds: str, row, req: PredictRequest, features: list[str]) -> dict:
    ranges = _global_ranges(ds)
    ov = scenario_overrides(row, features, req.temp_shift_c, req.current_scale, ranges)
    ov.update(req.overrides)
    return ov


@router.get("/scenario/limits")
def scenario_limits_endpoint(battery_id: str, dataset: str | None = None, cycle: int | None = None):
    """Allowed what-if shifts for this battery/cycle so every scenario stays inside the training support."""
    def go():
        ds = store.resolve(dataset, battery_id)
        row = _row(ds, battery_id, cycle)
        art = store.artifact(ds, f"forecast_h{store.horizons(ds)[0]}")
        return store.clean({"dataset": ds, "battery_id": battery_id, "from_cycle": int(row["cycle"]), **scenario_limits(row, art["features"], _global_ranges(ds)),
                            "note": "Physics-based what-if scenario. Feature overrides are clamped to training range; the stress multiplier applies Arrhenius thermal + C-rate scaling."})
    return _guard(go)


def _forecast(req: PredictRequest):
    ds = store.resolve(req.dataset, req.battery_id)
    hs = store.horizons(ds)
    if not hs:
        raise ArtifactMissing("No forecast models trained for this dataset")
    h = req.horizon or store.meta(ds)["headline_horizon"]
    if h not in hs:
        raise HTTPException(422, f"Horizon {h} not trained. Available: {hs}")
    art = store.artifact(ds, f"forecast_h{h}")
    row = _row(ds, req.battery_id, req.cycle)

    if req.current_scale is not None and req.current_scale <= 0:
        raise HTTPException(422, "current_scale must be positive")

    lim = scenario_limits(row, art["features"], _global_ranges(ds))
    if req.temp_shift_c is not None and lim.get("temp_shift_c"):
        t_lo, t_hi = lim["temp_shift_c"]
        if not (t_lo <= req.temp_shift_c <= t_hi):
            raise HTTPException(422, f"temp_shift_c {req.temp_shift_c} outside training support [{t_lo}, {t_hi}]")

    if req.current_scale is not None and lim.get("current_scale"):
        c_lo, c_hi = lim["current_scale"]
        if not (c_lo <= req.current_scale <= c_hi):
            raise HTTPException(422, f"current_scale {req.current_scale} outside training support [{c_lo}, {c_hi}]")

    if req.overrides:
        bad = check_feature_ranges(art, req.overrides)
        if bad:
            raise HTTPException(422, f"Overrides outside training support: {'; '.join(bad)}")

    ov = _resolve_overrides(ds, row, req, art["features"])
    return ds, art, row, predict_future_soh(art, row, ov, req.temp_shift_c, req.current_scale)


@router.post("/predict/soh")
def predict_soh(req: PredictRequest):
    def go():
        ds, art, row, p = _forecast(req)
        return store.clean({**p, "dataset": ds, "battery_id": req.battery_id, "from_cycle": int(row["cycle"]), "target_cycle": int(row["cycle"]) + p["horizon"],
                            "is_demo": art["is_demo"], "what_if": bool(req.overrides or req.temp_shift_c or req.current_scale),
                            "note": ("Model-based scenario, not a physical simulation." if (req.overrides or req.temp_shift_c or req.current_scale) else "ML estimate; interval = empirical 5-95% residual band from held-out batteries."),
                            "kind": "predicted"})
    return _guard(go)


@router.post("/predict/degradation")
def predict_degradation(req: PredictRequest):
    def go():
        ds, art, row, p = _forecast(req)
        return store.clean({"dataset": ds, "battery_id": req.battery_id, "from_cycle": int(row["cycle"]), "horizon": p["horizon"],
                            "predicted_rate_pct_per_cycle": (p["current_soh"] - p["predicted_soh"]) / p["horizon"],
                            "observed_rolling_rate_25": row.get("rolling_degradation_rate_25"), "observed_rolling_rate_10": row.get("rolling_degradation_rate_10"),
                            "is_demo": art["is_demo"], "what_if": bool(req.overrides or req.temp_shift_c or req.current_scale), "kind": "predicted"})
    return _guard(go)


@router.post("/predict/rul")
def predict_rul(req: RulRequest):
    def go():
        ds = store.resolve(req.dataset, req.battery_id)
        row = _row(ds, req.battery_id, req.cycle)
        thr = req.eol_threshold or store.meta(ds)["config"]["eol_threshold_soh"]

        if req.current_scale is not None and req.current_scale <= 0:
            raise HTTPException(422, "current_scale must be positive")

        gamma = physical_stress_multiplier(req.temp_shift_c, req.current_scale)

        base_rate = row.get("rolling_degradation_rate_25")
        base_rate = row.get("rolling_degradation_rate_10") if base_rate is None or not np.isfinite(base_rate) else base_rate
        base_rate = max(1e-4, float(base_rate or 0.05))
        scenario_rate = base_rate * gamma

        cap = 1000.0
        extrap = float(extrapolate_rul(np.array([row["soh"]]), np.array([scenario_rate]), thr, cap)[0])
        out = {"dataset": ds, "battery_id": req.battery_id, "from_cycle": int(row["cycle"]), "current_soh": float(row["soh"]), "eol_threshold": thr,
               "extrapolation_rul_cycles": extrap, "extrapolation_capped": extrap >= cap, "kind": "predicted",
               "statement": f"RUL is estimated using an {thr:g}% SOH end-of-life threshold.",
               "caveat": "An ML estimate with substantial uncertainty - not an exact physical prediction."}
        try:
            art = store.artifact(ds, "rul")
        except ArtifactMissing:
            art = None
        if art and abs(art["eol_threshold"] - thr) < 1e-9:
            ov = _resolve_overrides(ds, row, req, art["features"])
            import pandas as pd
            r = {**dict(row), **ov}
            x = pd.DataFrame([{f: r.get(f, np.nan) for f in art["features"]}])
            raw_ml = float(max(0.0, art["pipeline"].predict(x)[0]))
            raw_ml = float(np.clip(raw_ml / gamma, 0, cap))
            out.update({"ml_rul_cycles": raw_ml, "ml_model": art["model_name"], "is_demo": art["is_demo"]})
        else:
            note = f"ML model is trained at {art['eol_threshold']:g}% EOL; showing trajectory extrapolation for {thr:g}% EOL." if art else "ML RUL model unavailable for this dataset. Showing trajectory extrapolation only."
            out.update({"ml_rul_cycles": None, "is_demo": store.meta(ds)["is_demo"], "ml_note": note})
        return store.clean(out)
    return _guard(go)


@router.get("/models")
def models(dataset: str | None = None):
    def go():
        ds = store.resolve(dataset)
        m = store.meta(ds)
        out = []
        for h in store.horizons(ds):
            a = store.artifact(ds, f"forecast_h{h}")
            out.append({"id": f"forecast_h{h}", "task": "SOH forecast", "horizon": h, "model": a["model_name"], "params": a["best_params"], "n_features": len(a["features"])})
        try:
            r = store.artifact(ds, "rul")
            out.append({"id": "rul", "task": "RUL (direct)", "model": r["model_name"], "params": r["best_params"], "n_features": len(r["features"])})
        except ArtifactMissing:
            pass
        ranges = _global_ranges(ds)
        return store.clean({"dataset": ds, "is_demo": m["is_demo"], "models": out, "feature_ranges": ranges, "collinear_dropped": m.get("collinear_dropped", {}), "n_cycles": m["n_cycles"], "batteries": m["batteries"], "validation": m["validation"], "versions": m["versions"],
                            "feature_docs": m["feature_docs"], "excluded_features": m["excluded_features"], "model_selection_note": m["model_selection_note"]})
    return _guard(go)


@router.get("/models/horizons")
def horizon_curve(dataset: str | None = None):
    """MAE/RMSE of every trained model at every trained horizon (error-vs-horizon chart)."""
    def go():
        ds = store.resolve(dataset)
        out = []
        for h in store.horizons(ds):
            f = store.metrics_file(ds, f"forecast_h{h}_metrics.json")
            out.append({"horizon": h, "best_model": f["best_model"],
                        "models": {k: {"mae": v["mae"], "rmse": v["rmse"], "trained": v["trained"]} for k, v in f["metrics"].items()}})
        return store.clean({"dataset": ds, "is_demo": store.meta(ds)["is_demo"], "horizons": out})
    return _guard(go)


@router.get("/models/performance")
def performance(dataset: str | None = None, horizon: int | None = None):
    def go():
        ds = store.resolve(dataset)
        m = store.meta(ds)
        h = horizon or m["headline_horizon"]
        if h not in store.horizons(ds):
            raise HTTPException(422, f"Horizon {h} not trained. Available: {store.horizons(ds)}")
        f = store.metrics_file(ds, f"forecast_h{h}_metrics.json")
        import pandas as pd
        from src.config import OUTPUTS_DIR
        pr = pd.read_csv(OUTPUTS_DIR / "predictions" / f"{ds}_forecast_h{h}.csv")
        best = f["best_model"]
        pts = pr[["battery_id", "cycle", "y_true", best]].rename(columns={best: "y_pred"}).dropna()
        try:
            rul = store.metrics_file(ds, "rul_metrics.json")
        except ArtifactMissing:
            rul = None
        return store.clean({"dataset": ds, "is_demo": m["is_demo"], "horizon": h, "validation": m["validation"], "best_model": best,
                            "forecast": {k: {kk: v[kk] for kk in ("mae", "rmse", "r2", "r2_delta", "n", "trained")} for k, v in f["metrics"].items()},
                            "per_battery_best": f["metrics"][best]["per_battery"], "rul": rul and {"best_model": rul.get("best_model"), "metrics": {k: {kk: v.get(kk) for kk in ("mae", "rmse", "mape_pct", "mae_when_true_rul_le_50", "trained")} for k, v in rul.get("metrics", {}).items()}},
                            "points": pts.round(4).to_dict("records")})
    return _guard(go)


@router.get("/explainability")
def explainability(dataset: str | None = None):
    def go():
        ds = store.resolve(dataset)
        imp = store.metrics_file(ds, "importance.json")
        docs = store.meta(ds)["feature_docs"]
        top = sorted(imp["mean_abs_shap"].items(), key=lambda kv: -kv[1])
        return store.clean({"dataset": ds, "is_demo": store.meta(ds)["is_demo"], "method": imp["method"],
                            "importance": [{"feature": k, "mean_abs_shap": v, "doc": docs.get(k, "")} for k, v in top],
                            "sample": imp.get("sample"), "features": imp["features"],
                            "interpretation": ("Features are ranked by how strongly the fitted model associates them with its SOH-forecast output. "
                                               "This describes the model, not physical causation: features are correlated and the data are observational.")})
    return _guard(go)


@router.get("/explainability/{battery_id}")
def explain_battery(battery_id: str, dataset: str | None = None, cycle: int | None = None):
    def go():
        ds = store.resolve(dataset)
        h = store.meta(ds)["headline_horizon"]
        art = store.artifact(ds, f"forecast_h{h}")
        row = _row(ds, battery_id, cycle)
        ex = local_explanation(art["pipeline"], row, art["features"], background=art.get("shap_background"))
        docs = store.meta(ds)["feature_docs"]
        for c in ex["contributions"]:
            c["doc"] = docs.get(c["feature"], "")
        return store.clean({"dataset": ds, "battery_id": battery_id, "from_cycle": int(row["cycle"]), "horizon": h, "is_demo": art["is_demo"],
                            "unit": "percentage points of SOH change over the horizon", **ex,
                            "interpretation": "Positive SHAP values push the model's predicted SOH change upward; negative values push it down (more loss). Associations, not causes."})
    return _guard(go)


@router.get("/ev-calculator/presets")
def get_ev_presets():
    return {"presets": PRESETS}


@router.post("/ev-calculator/predict")
def predict_ev_calculator(req: EvCalculatorRequest):
    try:
        result = predict_ev_health(
            capacity_kwh=req.capacity_kwh,
            odometer_km=req.odometer_km,
            age_years=req.age_years,
            fast_charge_pct=req.fast_charge_pct,
            ambient_temp_c=req.ambient_temp_c,
            charge_limit_pct=req.charge_limit_pct,
            efficiency_wh_km=req.efficiency_wh_km,
            rated_range_km=req.rated_range_km,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Error calculating EV battery health: {str(e)}")

