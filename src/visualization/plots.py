"""Static matplotlib figures written to outputs/figures (the 10 required plots)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def make_all(t: pd.DataFrame, fpreds: pd.DataFrame, fmetrics: dict, rpreds: pd.DataFrame | None, rmetrics: dict | None,
             importance: dict, out: Path, tag: str, eol: float, best: str):
    sfx = f" [{tag}]"
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for b, g in t.groupby("battery_id"): ax.plot(g["cycle"], g["capacity_ah"], label=b, lw=1.2)
    ax.set(xlabel="Cycle", ylabel="Capacity (Ah)", title="Capacity vs cycle" + sfx); ax.legend(ncol=2, fontsize=7)
    _save(fig, out / "01_capacity_vs_cycle.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for b, g in t.groupby("battery_id"): ax.plot(g["cycle"], g["soh"], label=b, lw=1.2)
    ax.axhline(eol, color="crimson", ls="--", lw=1, label=f"EOL {eol:g}%")
    ax.set(xlabel="Cycle", ylabel="SOH (%)", title="SOH vs cycle" + sfx); ax.legend(ncol=2, fontsize=7)
    _save(fig, out / "02_soh_vs_cycle.png")

    p = fpreds[fpreds["horizon"] == fpreds["horizon"].min()] if False else fpreds
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for b, g in p.groupby("battery_id"): ax.scatter(g["y_true"], g[best], s=6, alpha=.6, label=b)
    lo, hi = p["y_true"].min(), p["y_true"].max(); ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="Actual SOH (%)", ylabel="Predicted SOH (%)", title=f"Actual vs predicted ({best}, held-out cells)" + sfx); ax.legend(fontsize=6)
    _save(fig, out / "03_actual_vs_predicted_soh.png")

    res = p["y_true"] - p[best]
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axs[0].scatter(p[best], res, s=6, alpha=.5); axs[0].axhline(0, color="k", lw=1)
    axs[0].set(xlabel="Predicted SOH (%)", ylabel="Residual (pp)", title="Residuals vs predicted" + sfx)
    axs[1].hist(res.dropna(), bins=30); axs[1].set(xlabel="Residual (pp)", title="Residual distribution")
    _save(fig, out / "04_residuals.png")

    last = t.groupby("battery_id").tail(1)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for col, ax, lab in (("dis_t_mean", axs[0], "Mean discharge temperature (degC)"), ("dis_i_mean", axs[1], "Mean discharge current (A)")):
        if col in t and t[col].notna().any():
            ax.scatter(t[col], t["rolling_degradation_rate_25"], s=5, alpha=.4)
            ax.set(xlabel=lab, ylabel="Rolling degradation rate (%SOH/cycle)", title=lab.split(" (")[0] + " vs degradation" + sfx)
    _save(fig, out / "05_06_temperature_current_vs_degradation.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for b, g in t.groupby("battery_id"): ax.plot(g["cycle"], g["capacity_loss"], label=b, lw=1.2)
    ax.set(xlabel="Cycle", ylabel="Capacity loss (%)", title="Degradation trajectories" + sfx); ax.legend(ncol=2, fontsize=7)
    _save(fig, out / "07_degradation_trajectories.png")

    names = [k for k, v in fmetrics.items()]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(names, [fmetrics[n]["rmse"] for n in names], color=["#2a9d8f" if fmetrics[n]["trained"] else "#999" for n in names])
    ax.set(xlabel="RMSE (pp SOH), leave-one-battery-out", title="Model comparison" + sfx)
    _save(fig, out / "08_model_comparison.png")

    b0 = fpreds["battery_id"].iloc[0]
    g = fpreds[fpreds["battery_id"] == b0].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(g["cycle"] + g["horizon"], g["y_true"], label="actual", lw=1.4); ax.plot(g["cycle"] + g["horizon"], g[best], label="predicted", lw=1.1)
    ax.set(xlabel="Target cycle", ylabel="SOH (%)", title=f"Forecast of SOH at t+{int(g['horizon'].iloc[0])} for held-out cell {b0}" + sfx); ax.legend()
    _save(fig, out / "09_future_soh_trajectory.png")

    if rpreds is not None and len(rpreds):
        rb = rmetrics and min((k for k, v in rmetrics.items() if v["trained"]), key=lambda k: rmetrics[k]["mae"])
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for b, gg in rpreds.groupby("battery_id"): ax.scatter(gg["y_true"], gg[rb], s=6, alpha=.6, label=b)
        m = max(rpreds["y_true"].max(), 1); ax.plot([0, m], [0, m], "k--", lw=1)
        ax.set(xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)", title=f"RUL estimation ({rb}, held-out cells)" + sfx); ax.legend(fontsize=6)
        _save(fig, out / "10_rul_estimation.png")

    imp = sorted(importance.get("mean_abs_shap", {}).items(), key=lambda kv: kv[1])[-15:]
    if imp:
        fig, ax = plt.subplots(figsize=(6.5, 4.6)); ax.barh([k for k, _ in imp], [v for _, v in imp])
        ax.set(xlabel="mean |SHAP| (association with model output)", title="Global feature importance" + sfx)
        _save(fig, out / "11_shap_importance.png")
