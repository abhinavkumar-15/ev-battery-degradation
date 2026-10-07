"""Dataset adapter interface: raw files -> Common Battery Data Format."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.data.schema import ALL_COLUMNS, REQUIRED_COLUMNS


class DatasetAdapter(ABC):
    name: str = "base"
    is_demo: bool = False

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a cycle-level DataFrame (see src/data/schema.py)."""

    @staticmethod
    def conform(df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and add any missing optional ones as NaN."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Adapter output is missing required columns: {missing}")
        for c in ALL_COLUMNS:
            if c not in df.columns:
                df[c] = float("nan") if c != "capacity_source" else "unknown"
        return df[ALL_COLUMNS].sort_values(["battery_id", "cycle"]).reset_index(drop=True)
