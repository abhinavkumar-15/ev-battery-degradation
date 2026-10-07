"""Central, reproducible configuration for the VoltGuard AI ML pipeline.

Every tunable that affects results lives here so a run is reproducible from
(code version, config, raw data).  Paths are resolved relative to the repo root
so no machine-specific paths are ever stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_SAMPLE = REPO_ROOT / "data" / "sample"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"

SEED = 42


@dataclass
class PipelineConfig:
    # --- targets -----------------------------------------------------------
    eol_threshold_soh: float = 80.0          # End-of-life threshold (% SOH). Configurable.
    reference_mode: str = "rated"            # "rated" | "first_valid" (see targets.compute_soh)
    rated_capacity_ah: float | None = None   # used when reference_mode == "rated"

    # --- feature engineering ----------------------------------------------
    windows: tuple[int, ...] = (10, 25, 50)  # trailing windows (cycles)
    warmup_cycles: int = 5                   # drop first N cycles/battery (windows not yet meaningful)
    min_feature_coverage: float = 0.90       # drop features that are >10% missing in the modelling table
    max_feature_corr: float = 0.99999         # drop near-duplicate features (|Pearson r| >= this, i.e. exact duplicates); keeps the earlier one

    # --- forecasting -------------------------------------------------------
    horizons: tuple[int, ...] = (5, 10, 25, 50)   # predict SOH at t+h cycles
    headline_horizon: int = 10

    # --- validation / tuning ----------------------------------------------
    seed: int = SEED
    inner_cv_splits: int = 3                 # GroupKFold (by battery) inside each training fold
    interval_quantiles: tuple[float, float] = (0.05, 0.95)


# Rated capacity documented per dataset (Ah). None => fall back to first valid capacity.
DATASET_RATED_CAPACITY_AH = {
    "nasa": 2.0,      # NASA PCoE 18650 cells are rated 2 Ah (see data/README.md)
    "demo": 2.0,
}
