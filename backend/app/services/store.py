"""Loads trained artifacts + processed tables from disk. Nothing is trained here."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.features.history import FEATURE_DOCS
from src.models.predict import ArtifactMissing, load_artifact


class NotFound(KeyError):
    pass


def available_datasets() -> list[str]:
    tags = [p.name for p in MODELS_DIR.glob("*") if p.is_dir() and any(p.glob("forecast_h*.joblib"))
            and (OUTPUTS_DIR / "metrics" / p.name / "run_meta.json").exists()]
    return sorted(tags, key=lambda t: (t != "demo", t))      # demo first, then others


def default_dataset() -> str:
    ds = available_datasets()
    if not ds:
        raise ArtifactMissing("No trained dataset found. Run `python -m src.pipeline --dataset demo` (or nasa) first.")
    return ds[0]


def resolve(dataset: str | None = None, battery_id: str | None = None) -> str:
    if dataset:
        if dataset not in available_datasets():
            raise NotFound(f"Unknown or untrained dataset '{dataset}'. Available: {available_datasets()}")
        return dataset
    if battery_id:
        for ds in available_datasets():
            try:
                t = table(ds)
                if (t["battery_id"].astype(str) == str(battery_id)).any():
                    return ds
            except Exception:
                pass
    return default_dataset()


@lru_cache(maxsize=8)
def table(ds: str) -> pd.DataFrame:
    f = DATA_PROCESSED / f"{ds}_features.csv"
    if not f.exists():
        if ds != "demo":
            raise ArtifactMissing(f"Processed table {f.name} missing; re-run `python -m src.pipeline --dataset {ds}`")
        # the git-ignored processed demo table is cheap to rebuild from the tiny committed sample
        from src.data.adapters import DemoAdapter
        from src.features.pipeline import build_feature_table
        t, _ = build_feature_table(DemoAdapter().load(), dataset="demo")
        f.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(f, index=False)
    return pd.read_csv(f)


@lru_cache(maxsize=32)
def artifact(ds: str, name: str) -> dict:
    return load_artifact(MODELS_DIR / ds / f"{name}.joblib")


def horizons(ds: str) -> list[int]:
    return sorted(int(p.stem.split("_h")[1]) for p in (MODELS_DIR / ds).glob("forecast_h*.joblib"))


def meta(ds: str) -> dict:
    return _json(OUTPUTS_DIR / "metrics" / ds / "run_meta.json")


def _json(p: Path):
    if not p.exists():
        raise ArtifactMissing(f"Missing {p.name}; re-run the training pipeline")
    return json.loads(p.read_text())


def metrics_file(ds: str, name: str):
    return _json(OUTPUTS_DIR / "metrics" / ds / name)


def battery_rows(ds: str, battery_id: str) -> pd.DataFrame:
    t = table(ds)
    g = t[t["battery_id"].astype(str) == battery_id].sort_values("cycle")
    if g.empty:
        raise NotFound(f"Battery '{battery_id}' not found in dataset '{ds}'")
    return g


def clean(o):
    """NaN/inf -> None so responses are valid JSON."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


def feature_docs() -> dict:
    return FEATURE_DOCS
