"""Resumable fold cache.

Training with nested, battery-grouped tuning is slow on small machines.  Each held-out-battery
fold is stored on disk keyed by a fingerprint of (data, features, config); re-running the pipeline
resumes from finished folds, and `--max-seconds` lets a run stop cleanly between folds.
The fingerprint changes whenever data, features, seed, grids or horizon config change, so stale
results can never be reused.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib


class PartialRun(RuntimeError):
    """Raised when the time budget is exhausted; progress so far is saved."""


class FoldCache:
    def __init__(self, root: Path, fingerprint: str, max_seconds: float | None = None):
        self.dir = Path(root) / fingerprint
        self.dir.mkdir(parents=True, exist_ok=True)
        self.deadline = time.time() + max_seconds if max_seconds else None
        self.fold_times: list[float] = []

    def get(self, key: str):
        f = self.dir / f"{key}.joblib"
        return joblib.load(f) if f.exists() else None

    def put(self, key: str, obj, seconds: float):
        joblib.dump(obj, self.dir / f"{key}.joblib")
        self.fold_times.append(seconds)

    def check_budget(self):
        if self.deadline is None:
            return
        est = max(self.fold_times) if self.fold_times else 30.0
        if time.time() + est * 1.15 > self.deadline:
            raise PartialRun("time budget reached; re-run the same command to resume")


def fingerprint(**parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:12]
