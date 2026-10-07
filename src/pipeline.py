"""End-to-end training pipeline.

    python -m src.pipeline --dataset demo           # synthetic demo (pipeline smoke test only)
    python -m src.pipeline --dataset nasa           # real NASA data in data/raw/nasa/*.mat
    python -m src.pipeline --dataset nasa --eol 70  # different end-of-life threshold

Outputs (per dataset tag):
    data/processed/<tag>_features.csv        modelling table (ignored by git)
    models/<tag>/forecast_h{H}.joblib        final forecast pipeline per horizon (+ metadata)
    models/<tag>/rul.joblib                  final direct-RUL pipeline
    outputs/metrics/<tag>/*.json             metrics, importance, run metadata
    outputs/predictions/<tag>_*.csv          held-out-battery predictions
    outputs/figures/<tag>/*.png              required plots
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, PipelineConfig
from src.data.adapters import ADAPTERS
from src.evaluation.metrics import regression_metrics
from src.explainability.shap_explain import global_importance, transform_features
from src.features.history import EXCLUDED_FEATURES, FEATURE_DOCS
from src.features.pipeline import available_features, build_feature_table, drop_collinear, modelling_rows
from src.models.cache import FoldCache, PartialRun, fingerprint
from src.models.forecast import evaluate_horizon
from src.models.rul import evaluate_rul
from src.models.tuning import fit_with_tuning
from src.models.zoo import make_models
from src.visualization.plots import make_all

log = logging.getLogger("pipeline")

SPLIT_DESCRIPTION = (
    "Leave-one-battery-out (LOBO): each battery is held out entirely and scored by models trained on the others; "
    "hyper-parameters are tuned by GroupKFold-by-battery inside each training fold; features are trailing-window only. "
    "Random row splitting is never used (temporal leakage)."
)


def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    return o


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2))


def run(dataset: str, cfg: PipelineConfig, quick: bool = False, skip_rul: bool = False, max_seconds: float | None = None) -> dict:
    t0 = time.time()
    adapter = ADAPTERS[dataset]()
    is_demo = bool(getattr(adapter, "is_demo", False))
    raw = adapter.load()
    # the saved table always carries every default horizon's target column, whichever horizons this run trains
    from dataclasses import replace
    table_cfg = replace(cfg, horizons=tuple(sorted(set(cfg.horizons) | set(PipelineConfig().horizons))))
    table, qreport = build_feature_table(raw, table_cfg, dataset=dataset)
    feats, collinear_dropped = drop_collinear(table, available_features(table, cfg), cfg.max_feature_corr)
    if collinear_dropped:
        log.info("dropped near-duplicate features: %s", collinear_dropped)
    n_bat = table["battery_id"].nunique()
    log.info("%s: %d cells, %d cycles, %d features", dataset, n_bat, len(table), len(feats))
    if n_bat < 3:
        raise ValueError("Need >= 3 batteries for leave-one-battery-out validation")

    tag = dataset
    fp = fingerprint(feats=feats, cfg={k: v for k, v in asdict(cfg).items() if k not in ("horizons", "headline_horizon")}, n=len(table), soh=round(float(table["soh"].sum()), 4), dataset=dataset)
    cache = FoldCache(OUTPUTS_DIR / ".cache" / tag, fp, max_seconds)
    mdir, odir = MODELS_DIR / tag, OUTPUTS_DIR / "metrics" / tag
    table.to_csv(DATA_PROCESSED / f"{tag}_features.csv", index=False)

    horizons = (cfg.headline_horizon,) if quick else cfg.horizons
    summary: dict = {"forecast": {}, "rul": {}}
    headline = {}
    for h in horizons:
        rows = modelling_rows(table, horizon=h)
        preds, metrics, params = evaluate_horizon(rows, feats, h, cfg, cache)
        trained = {k: v for k, v in metrics.items() if v["trained"]}
        best = min(trained, key=lambda k: trained[k]["rmse"])
        summary["forecast"][h] = {"metrics": metrics, "best_model": best, "best_params_per_fold": params}
        _dump(summary["forecast"][h], odir / f"forecast_h{h}_metrics.json")
        preds.to_csv(OUTPUTS_DIR / "predictions" / f"{tag}_forecast_h{h}.csv", index=False)

        # ---- final model: refit best config on ALL batteries --------------------------------
        est, grid = make_models(cfg.seed)[best]
        y_delta = (rows[f"soh_future_h{h}"] - rows["soh"]).to_numpy()
        fitted, best_params = fit_with_tuning(est, grid, rows[feats], y_delta, rows["battery_id"].to_numpy(), cfg.inner_cv_splits)
        resid = (preds["y_true"] - preds[best]).dropna().to_numpy()
        q = np.quantile(resid, cfg.interval_quantiles)
        ranges = {f: [float(rows[f].min()), float(rows[f].max())] for f in feats}
        art = {"pipeline": fitted, "features": feats, "horizon": h, "model_name": best, "best_params": best_params,
               "residual_quantiles": [float(q[0]), float(q[1])], "feature_ranges": ranges, "dataset": dataset,
               "is_demo": is_demo, "eol_threshold": cfg.eol_threshold_soh, "seed": cfg.seed,
               "shap_background": transform_features(fitted, rows[feats].sample(min(100, len(rows)), random_state=cfg.seed)),
               "versions": {"sklearn": sklearn.__version__, "python": platform.python_version()}}
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(art, mdir / f"forecast_h{h}.joblib")
        if h == cfg.headline_horizon:
            headline = {"preds": preds, "metrics": metrics, "best": best, "art": art, "rows": rows}

    # ---- RUL -----------------------------------------------------------------------------
    rrows = modelling_rows(table, for_rul=True)
    rpreds = rmet = None
    if skip_rul:
        log.info("RUL stage skipped by flag (existing RUL artifacts are kept)")
    elif rrows["battery_id"].nunique() >= 3:
        rpreds, rmet = evaluate_rul(rrows, feats, cfg, cache)
        trained = {k: v for k, v in rmet.items() if v["trained"]}
        rbest = min(trained, key=lambda k: trained[k]["mae"])
        est, grid = make_models(cfg.seed)[rbest]
        fitted, rparams = fit_with_tuning(est, grid, rrows[feats], rrows["rul"].to_numpy(), rrows["battery_id"].to_numpy(), cfg.inner_cv_splits)
        joblib.dump({"pipeline": fitted, "features": feats, "model_name": rbest, "best_params": rparams,
                     "eol_threshold": cfg.eol_threshold_soh, "dataset": dataset, "is_demo": is_demo,
                     "feature_ranges": {f: [float(rrows[f].min()), float(rrows[f].max())] for f in feats},
                     "cap_cycles": float(rrows["eol_cycle"].max() * 2)}, mdir / "rul.joblib")
        summary["rul"] = {"metrics": rmet, "best_model": rbest}
        _dump(summary["rul"], odir / "rul_metrics.json")
        rpreds.to_csv(OUTPUTS_DIR / "predictions" / f"{tag}_rul.csv", index=False)
    else:
        log.warning("RUL skipped: <3 batteries reach the %.0f%% EOL threshold", cfg.eol_threshold_soh)
        summary["rul"] = {"skipped": "fewer than 3 batteries reach the EOL threshold"}

    if not headline:
        log.info("headline horizon %d not in this run: explainability, figures and run_meta left untouched", cfg.headline_horizon)
        return {"meta": None, **summary}

    # ---- explainability (final headline-horizon model) --------------------------------------
    h_art = headline["art"]
    imp = global_importance(h_art["pipeline"], headline["rows"], feats, seed=cfg.seed)
    _dump(imp, odir / "importance.json")

    # ---- figures -----------------------------------------------------------------------------
    make_all(table, headline["preds"], headline["metrics"], rpreds, rmet, imp,
             OUTPUTS_DIR / "figures" / tag, ("DEMO - synthetic" if is_demo else dataset.upper()), cfg.eol_threshold_soh, headline["best"])

    meta = {"dataset": dataset, "is_demo": is_demo, "config": asdict(cfg), "n_batteries": n_bat, "batteries": sorted(table["battery_id"].unique()),
            "n_cycles": int(len(table)), "features": feats, "collinear_dropped": collinear_dropped, "feature_docs": {f: FEATURE_DOCS.get(f, "") for f in feats},
            "excluded_features": EXCLUDED_FEATURES, "validation": SPLIT_DESCRIPTION, "quality_report": qreport,
            "headline_horizon": cfg.headline_horizon, "horizons": sorted(int(p.stem.split("_h")[1]) for p in mdir.glob("forecast_h*.joblib")), "runtime_s": round(time.time() - t0, 1),
            "versions": {"sklearn": sklearn.__version__, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "model_selection_note": "Best model is selected on the same LOBO scores that are reported, so those scores are mildly optimistic; "
                                    "see docs/methodology.md."}
    _dump(meta, odir / "run_meta.json")
    return {"meta": meta, **summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--eol", type=float, default=80.0, help="end-of-life SOH threshold (%%)")
    ap.add_argument("--quick", action="store_true", help="headline horizon only")
    ap.add_argument("--horizons", type=int, nargs="+", help="override forecast horizons (cycles), e.g. --horizons 10 25")
    ap.add_argument("--max-seconds", type=float, help="stop cleanly between folds after this many seconds; re-run to resume")
    ap.add_argument("--skip-rul", action="store_true", help="do not retrain RUL models (keeps existing artifacts)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = PipelineConfig(eol_threshold_soh=a.eol)
    if a.horizons:
        cfg.horizons = tuple(a.horizons)
    try:
        out = run(a.dataset, cfg, quick=a.quick, skip_rul=a.skip_rul, max_seconds=a.max_seconds)
    except PartialRun as e:
        print(f"PARTIAL: {e}")
        raise SystemExit(3)
    if out["meta"] is None:
        print("DONE (non-headline horizons only)")
        return
    h = cfg.headline_horizon
    print(f"\n=== {a.dataset} | LOBO | horizon {h} ===")
    for n, m in out["forecast"][h]["metrics"].items():
        print(f"{n:34s} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  R2 {m['r2']:.3f}  R2(delta) {m['r2_delta'] if m['r2_delta'] is None else round(m['r2_delta'],3)}")
    if "metrics" in out["rul"]:
        print("--- RUL (cycles) ---")
        for n, m in out["rul"]["metrics"].items():
            print(f"{n:34s} MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}")


if __name__ == "__main__":
    main()
