"""NASA Prognostics Center of Excellence - Li-ion Battery Aging Dataset adapter.

Source : https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
Cells  : commercial 18650 Li-ion cells (rated 2 Ah), cycled at room temperature
         (and 4 / 43 degC for other cell groups) until capacity fade.
Files  : one MATLAB .mat file per cell (e.g. B0005.mat).  Each file holds a struct
         named after the cell with a `cycle` array.  Every element of `cycle` is a
         step record with fields:
             type                 'charge' | 'discharge' | 'impedance'
             ambient_temperature  degC
             time                 start timestamp (MATLAB datevec)
             data                 struct of per-sample arrays
         charge     : Voltage_measured (V), Current_measured (A), Temperature_measured (degC),
                      Current_charge (A), Voltage_charge (V), Time (s)
         discharge  : Voltage_measured, Current_measured, Temperature_measured,
                      Current_load, Voltage_load, Time (s), Capacity (Ah)   <- scalar
         impedance  : Re (Ohm), Rct (Ohm) (+ raw sensed/battery impedance arrays)

CYCLE DEFINITION (this project): cycle n == the n-th *discharge* step of a cell
(1-based).  The charge step immediately preceding it is paired to it.  The most
recent impedance measurement made *before* the discharge is forward-filled
(past information only => no look-ahead).

CAPACITY: the dataset's own `Capacity` scalar of the discharge step (Ah, coulomb
counted by NASA down to the discharge cut-off voltage).  If it is missing for a
step we fall back to our own coulomb count of |I| dt and mark capacity_source.

NOTE: this adapter was written against the documented file layout; it could not be
executed against the real files in the authoring environment (NASA hosts were not
reachable).  tests/test_nasa_adapter.py exercises it on a tiny structurally
identical *synthetic* file.  If your copy differs, fix `parse_nasa_cell` only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_RAW
from src.data.adapters.base import DatasetAdapter

log = logging.getLogger(__name__)

DEFAULT_BATTERIES = ("B0005", "B0006", "B0007", "B0018")


def _arr(d: dict, key: str) -> np.ndarray:
    v = d.get(key)
    if v is None:
        return np.array([], dtype=float)
    return np.atleast_1d(np.asarray(v, dtype=float)).ravel()


def _scalar(d: dict, key: str) -> float:
    v = _arr(d, key)
    return float(np.real(v[0])) if v.size else float("nan")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x)) if y.size > 1 else float("nan")


def _step_stats(d: dict, prefix: str) -> dict[str, float]:
    t, v, i, temp = _arr(d, "Time"), _arr(d, "Voltage_measured"), _arr(d, "Current_measured"), _arr(d, "Temperature_measured")
    n = min(t.size, v.size, i.size, temp.size)
    if n < 2:
        return {}
    t, v, i, temp = t[:n], v[:n], i[:n], temp[:n]
    out = {
        f"{prefix}_duration_s": float(t[-1] - t[0]),
        f"{prefix}_v_mean": float(np.nanmean(v)),
        f"{prefix}_i_mean": float(np.nanmean(np.abs(i))),
        f"{prefix}_i_max": float(np.nanmax(np.abs(i))),
        f"{prefix}_t_mean": float(np.nanmean(temp)),
        f"{prefix}_t_max": float(np.nanmax(temp)),
        f"{prefix}_throughput_ah": _trapz(np.abs(i), t) / 3600.0,
        f"{prefix}_energy_wh": _trapz(np.abs(v * i), t) / 3600.0,
    }
    if prefix == "dis":
        out.update({"dis_v_min": float(np.nanmin(v)), "dis_v_max": float(np.nanmax(v)), "dis_t_min": float(np.nanmin(temp))})
    else:
        out.update({"chg_v_max": float(np.nanmax(v))})
    return out


def parse_nasa_cell(cell: dict[str, Any], battery_id: str) -> pd.DataFrame:
    """Convert one cell's `cycle` records (as dicts) into the cycle-level table."""
    records = cell["cycle"]
    rows: list[dict] = []
    last_charge: dict[str, float] = {}
    last_imp = {"re_ohm": np.nan, "rct_ohm": np.nan}
    n = 0
    for rec in records:
        typ = str(rec["type"]).strip().lower()
        data = rec["data"]
        if typ == "charge":
            last_charge = _step_stats(data, "chg")
        elif typ == "impedance":
            re_, rct = _scalar(data, "Re"), _scalar(data, "Rct")
            last_imp = {"re_ohm": re_, "rct_ohm": rct}
        elif typ == "discharge":
            n += 1
            stats = _step_stats(data, "dis")
            cap = _scalar(data, "Capacity")
            src = "dataset"
            if not np.isfinite(cap):
                cap, src = stats.get("dis_throughput_ah", np.nan), "coulomb_count"
            amb = rec.get("ambient_temperature", np.nan)
            rows.append({"battery_id": battery_id, "cycle": n, "capacity_ah": cap, "capacity_source": src,
                         "ambient_temp_c": float(np.real(np.asarray(amb).ravel()[0])) if np.size(amb) else np.nan,
                         **stats, **last_charge, **last_imp})
        # unknown step types are ignored on purpose
    return pd.DataFrame(rows)


def load_mat_cell(path: Path) -> pd.DataFrame:
    from scipy.io import loadmat  # local import: scipy only needed for real files

    mat = loadmat(str(path), simplify_cells=True)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{path} contains no variables")
    cell = mat[keys[0]]
    return parse_nasa_cell(cell, battery_id=keys[0] if keys[0].startswith("B") else path.stem)


from src.config import DATA_RAW, REPO_ROOT

class NasaAdapter(DatasetAdapter):
    name = "nasa"

    def __init__(self, root: Path | None = None, batteries: tuple[str, ...] | None = DEFAULT_BATTERIES):
        if root:
            self.root = Path(root)
        elif (DATA_RAW / "nasa").exists() and list((DATA_RAW / "nasa").glob("B*.mat")):
            self.root = DATA_RAW / "nasa"
        else:
            self.root = REPO_ROOT / "data" / "nasa"
        self.batteries = batteries      # None => every B*.mat found

    def _find_files(self) -> list[Path]:
        files = sorted(self.root.rglob("B*.mat"))
        if self.batteries is not None:
            files = [f for f in files if f.stem in self.batteries]
        return files

    def load(self) -> pd.DataFrame:
        files = self._find_files()
        if not files:
            raise FileNotFoundError(
                f"No NASA .mat files found under {self.root}. See data/README.md for download instructions.")
        frames = []
        for f in files:
            log.info("Parsing %s", f)
            df = load_mat_cell(f)
            if df.empty:
                log.warning("%s: no discharge cycles parsed, skipping", f)
                continue
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out["dataset"] = self.name
        # forward-fill impedance within a cell (past -> future only)
        out = out.sort_values(["battery_id", "cycle"])
        out[["re_ohm", "rct_ohm"]] = out.groupby("battery_id")[["re_ohm", "rct_ohm"]].ffill()
        return self.conform(out)
