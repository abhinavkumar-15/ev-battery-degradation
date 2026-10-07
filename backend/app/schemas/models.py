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


class EvCalculatorRequest(BaseModel):
    capacity_kwh: float = Field(..., gt=0.5, le=250.0, description="Original Battery Capacity (kWh)")
    odometer_km: float = Field(..., ge=0.0, le=1000000.0, description="Total kilometers driven / ridden so far")
    age_years: float = Field(..., ge=0.05, le=30.0, description="Vehicle age in years")
    fast_charge_pct: float = Field(default=15.0, ge=0.0, le=100.0, description="Percentage of fast (DC) charging vs slow (AC) charging")
    ambient_temp_c: float = Field(default=25.0, ge=-20.0, le=60.0, description="Average ambient temperature in Celsius")
    charge_frequency: str = Field(default="daily", description="Charging frequency pattern (daily, alternate_days, twice_a_week, once_a_week, when_empty)")
    efficiency_wh_km: float = Field(default=140.0, gt=10.0, le=500.0, description="Vehicle energy efficiency in Wh/km")
    rated_range_km: float | None = Field(default=None, gt=0.0, le=2000.0, description="Original rated single-charge range in km")

