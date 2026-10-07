"""Cycle table -> modelling table (features + forecast targets)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATASET_RATED_CAPACITY_AH, PipelineConfig
from src.features.history import FEATURE_DOCS, add_history_features
from src.features.targets import add_degradation_rates, add_rul, compute_soh
from src.preprocessing.clean import clean_cycle_table

CANDIDATE_FEATURES = list(FEATURE_DOCS.keys())


def build_feature_table(raw: pd.DataFrame, cfg: PipelineConfig | None = None, dataset: str | None = None):
    """Returns (table, report).  `table` has one row per (battery, cycle) with:
    candidate features, soh, degradation rates, rul, and `soh_future_h{H}` targets.
    Warm-up rows are kept here (flagged) so the UI can show full histories; use
    `modelling_rows` to get the rows used for training.
    """
    cfg = cfg or PipelineConfig()
    clean, report = clean_cycle_table(raw)
    ds = dataset or str(clean["dataset"].iloc[0])
    rated = cfg.rated_capacity_ah or DATASET_RATED_CAPACITY_AH.get(ds)
    mode = cfg.reference_mode if (cfg.reference_mode != "rated" or rated) else "first_valid"
    report["reference"] = {"mode": mode, "rated_capacity_ah": rated if mode == "rated" else None}

    t = compute_soh(clean, mode, rated)
    t = add_degradation_rates(t, cfg.windows)
    t = add_history_features(t, cfg.windows)
    t = add_rul(t, cfg.eol_threshold_soh)

    g = t.groupby("battery_id", sort=False)
    for h in cfg.horizons:
        t[f"soh_future_h{h}"] = g["soh"].shift(-h)       # target: SOH h cycles later (NaN at the end)
    t["row_in_battery"] = g.cumcount()
    t["is_warmup"] = t["row_in_battery"] < cfg.warmup_cycles
    return t.reset_index(drop=True), report


def available_features(t: pd.DataFrame, cfg: PipelineConfig | None = None) -> list[str]:
    """Features with enough non-null coverage in the usable rows. Unavailable variables are
    dropped, never invented."""
    cfg = cfg or PipelineConfig()
    use = t.loc[~t["is_warmup"]]
    cols = []
    for c in CANDIDATE_FEATURES + [f"rolling_degradation_rate_{w}" for w in cfg.windows if f"rolling_degradation_rate_{w}" not in CANDIDATE_FEATURES]:
        if c in use and use[c].notna().mean() >= cfg.min_feature_coverage and use[c].nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def modelling_rows(t: pd.DataFrame, horizon: int | None = None, for_rul: bool = False) -> pd.DataFrame:
    m = t.loc[~t["is_warmup"]]
    if horizon is not None:
        m = m.loc[m[f"soh_future_h{horizon}"].notna()]
    if for_rul:
        m = m.loc[m["rul"].notna()]
    return m


def drop_collinear(t: pd.DataFrame, feats: list[str], threshold: float = 0.999) -> tuple[list[str], dict[str, str]]:
    """Remove near-duplicate features (|r| >= threshold) so linear models stay well-posed and SHAP credit is
    not split arbitrarily between identical columns.  Unsupervised (no target used); earlier features in
    `feats` take priority.  Returns (kept, {dropped: kept_twin})."""
    use = t.loc[~t["is_warmup"], feats]
    corr = use.corr().abs()
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for f in feats:
        twin = next((k for k in kept if corr.loc[f, k] >= threshold), None)
        if twin is None:
            kept.append(f)
        else:
            dropped[f] = twin
    return kept, dropped
