from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    battery_id: str = Field(..., min_length=1)
    dataset: str | None = Field(None, description="Dataset id; defaults to the preferred available dataset")
    cycle: int | None = Field(None, ge=1, description="Cycle to predict from; defaults to the latest cycle")
    horizon: int | None = Field(None, ge=1, le=500, description="Forecast horizon in cycles (must match a trained horizon)")
    overrides: dict[str, float] = Field(default_factory=dict, description="Raw what-if feature overrides (advanced; each value must be inside its training range)")
    temp_shift_c: float | None = Field(None, ge=-40, le=40, description="What-if: shift every temperature feature together (degC)")
    current_scale: float | None = Field(None, gt=0, le=5, description="What-if: rescale every current feature together (1.0 = unchanged)")


class RulRequest(PredictRequest):
    eol_threshold: float | None = Field(None, gt=0, lt=100, description="End-of-life SOH threshold (%)")
