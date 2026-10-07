"""Adapter for any dataset already aggregated to one row per cycle (e.g. processed
TBSI Sunwoda / Oxford / CALCE exports).  Provide a column mapping and, optionally,
a rated capacity; the rest of the pipeline is unchanged.

    GenericCycleCSVAdapter("sunwoda", "data/raw/sunwoda/cycles.csv",
        column_map={"cell": "battery_id", "cycle_index": "cycle", "Qd_Ah": "capacity_ah"})
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.adapters.base import DatasetAdapter


class GenericCycleCSVAdapter(DatasetAdapter):
    def __init__(self, name: str, csv_path: str | Path, column_map: dict[str, str]):
        self.name = name
        self.csv_path = Path(csv_path)
        self.column_map = column_map

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        df = pd.read_csv(self.csv_path).rename(columns=self.column_map)
        df["dataset"] = self.name
        df["battery_id"] = df["battery_id"].astype(str)
        if "capacity_source" not in df:
            df["capacity_source"] = "dataset"
        return self.conform(df)
