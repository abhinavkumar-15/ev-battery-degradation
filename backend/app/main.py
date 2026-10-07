"""VoltGuard AI API.  Run from the repo root:  uvicorn backend.app.main:app --reload --port 8000"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router

app = FastAPI(title="VoltGuard AI API", version="1.0.0",
              description="Serves pre-trained battery SOH / RUL models. Models are loaded from disk, never retrained per request.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
