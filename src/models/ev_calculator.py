"""Real-World EV Battery Health & SOH Calculator Engine.
Uses a physics-informed empirical degradation model trained on realistic EV fleet parameters:
- Calendar aging: Arrhenius temperature dependence + sqrt(t) SEI layer growth + high SOC dwell time
- Cycle aging: Equivalent Full Cycles (EFC) = (km * efficiency_wh_per_km) / (1000 * capacity_kwh)
- Fast-charging stress: C-rate thermal and mechanical stress factor
- Depth of Discharge (DOD) and daily charge limit (80% vs 100%)
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.config import MODELS_DIR

EV_CALCULATOR_DIR = MODELS_DIR / "ev_calculator"

# Popular EV Presets (2-wheelers & 4-wheelers)
PRESETS = [
    {
        "id": "tesla_model_3",
        "name": "Tesla Model 3 RWD",
        "category": "4-Wheeler",
        "capacity_kwh": 60.0,
        "efficiency_wh_km": 145,
        "rated_range_km": 491,
        "battery_chemistry": "LFP / NMC",
    },
    {
        "id": "tesla_model_y",
        "name": "Tesla Model Y Long Range",
        "category": "4-Wheeler",
        "capacity_kwh": 78.1,
        "efficiency_wh_km": 168,
        "rated_range_km": 533,
        "battery_chemistry": "NMC",
    },
    {
        "id": "tata_nexon_ev",
        "name": "Tata Nexon EV Long Range",
        "category": "4-Wheeler",
        "capacity_kwh": 40.5,
        "efficiency_wh_km": 135,
        "rated_range_km": 465,
        "battery_chemistry": "LFP",
    },
    {
        "id": "mg_zs_ev",
        "name": "MG ZS EV",
        "category": "4-Wheeler",
        "capacity_kwh": 50.3,
        "efficiency_wh_km": 150,
        "rated_range_km": 461,
        "battery_chemistry": "LFP",
    },
    {
        "id": "hyundai_ioniq_5",
        "name": "Hyundai Ioniq 5",
        "category": "4-Wheeler",
        "capacity_kwh": 72.6,
        "efficiency_wh_km": 170,
        "rated_range_km": 481,
        "battery_chemistry": "NMC",
    },
    {
        "id": "ather_450x",
        "name": "Ather 450X Gen 3",
        "category": "2-Wheeler",
        "capacity_kwh": 3.7,
        "efficiency_wh_km": 35,
        "rated_range_km": 146,
        "battery_chemistry": "NMC",
    },
    {
        "id": "ola_s1_pro",
        "name": "Ola S1 Pro Gen 2",
        "category": "2-Wheeler",
        "capacity_kwh": 4.0,
        "efficiency_wh_km": 38,
        "rated_range_km": 181,
        "battery_chemistry": "NMC",
    },
    {
        "id": "tvs_iqube",
        "name": "TVS iQube ST",
        "category": "2-Wheeler",
        "capacity_kwh": 4.56,
        "efficiency_wh_km": 36,
        "rated_range_km": 145,
        "battery_chemistry": "NMC",
    },
]


def physics_degradation_sim(
    capacity_kwh: float,
    odometer_km: float,
    age_years: float,
    fast_charge_pct: float = 15.0,
    ambient_temp_c: float = 25.0,
    charge_limit_pct: float = 90.0,
    efficiency_wh_km: float = 140.0,
    noise: float = 0.0,
) -> dict:
    """Calculates realistic EV degradation breakdown based on empirical battery aging physics."""
    # 1. Equivalent Full Cycles (EFC)
    total_energy_kwh = (odometer_km * efficiency_wh_km) / 1000.0
    efc = total_energy_kwh / max(capacity_kwh, 1.0)

    # 2. Calendar aging (% loss): SEI layer growth ~ t^0.75 modulated by temperature & high SOC
    # Arrhenius temperature multiplier (reference 25°C)
    temp_k = ambient_temp_c + 273.15
    ref_k = 298.15
    arrhenius = np.exp(0.045 * (temp_k - ref_k))

    # SOC dwell stress: holding battery above 85% accelerates calendar aging
    soc_stress = 1.0 + max(0.0, (charge_limit_pct - 80.0) / 100.0) * 0.45
    base_calendar_loss = 1.65 * (max(age_years, 0.05) ** 0.72) * arrhenius * soc_stress

    # 3. Cycle aging (% loss): mechanical degradation & lithium plating ~ EFC^0.82
    dod_factor = (charge_limit_pct / 100.0) ** 0.5
    base_cycle_loss = 0.019 * (efc ** 0.84) * dod_factor

    # 4. Fast charging penalty (% loss): high C-rate heating and localized plating
    fast_ratio = max(0.0, min(100.0, fast_charge_pct)) / 100.0
    fast_charge_loss = base_cycle_loss * (fast_ratio * 0.42 + (fast_ratio ** 2) * 0.35)

    # 5. Temperature extremes on cycling (extra thermal stress in hot climates > 30°C)
    thermal_cycling_penalty = 0.0
    if ambient_temp_c > 30.0:
        thermal_cycling_penalty = (ambient_temp_c - 30.0) * 0.06 * (efc / 200.0)
    elif ambient_temp_c < 5.0:
        thermal_cycling_penalty = (5.0 - ambient_temp_c) * 0.04 * (efc / 200.0)

    # Total degradation %
    total_loss = base_calendar_loss + base_cycle_loss + fast_charge_loss + thermal_cycling_penalty + noise
    total_loss = max(0.2, min(45.0, total_loss))

    soh = max(55.0, 100.0 - total_loss)

    return {
        "soh": float(round(soh, 2)),
        "total_loss_pct": float(round(total_loss, 2)),
        "calendar_loss_pct": float(round(base_calendar_loss, 2)),
        "cycle_loss_pct": float(round(base_cycle_loss, 2)),
        "fast_charge_loss_pct": float(round(fast_charge_loss, 2)),
        "thermal_loss_pct": float(round(thermal_cycling_penalty, 2)),
        "efc": float(round(efc, 1)),
    }


def generate_training_data(n_samples: int = 15000) -> pd.DataFrame:
    """Generates synthetic EV fleet data across broad operating envelopes."""
    np.random.seed(42)

    capacity_kwh = np.random.uniform(3.0, 100.0, n_samples)
    age_years = np.random.uniform(0.1, 10.0, n_samples)
    # Annual km between 4,000 and 35,000 km/year
    annual_km = np.random.uniform(4000.0, 35000.0, n_samples)
    odometer_km = annual_km * age_years + np.random.normal(0, 1500, n_samples)
    odometer_km = np.clip(odometer_km, 500.0, 300000.0)

    fast_charge_pct = np.random.beta(2, 5, n_samples) * 100.0
    ambient_temp_c = np.random.normal(26.0, 8.0, n_samples)
    ambient_temp_c = np.clip(ambient_temp_c, 5.0, 45.0)
    charge_limit_pct = np.random.choice([80.0, 85.0, 90.0, 95.0, 100.0], p=[0.25, 0.20, 0.25, 0.10, 0.20], size=n_samples)

    # Efficiency (Wh/km): correlated with capacity (2-wheelers ~35Wh/km, large SUVs ~200Wh/km)
    efficiency_wh_km = np.where(
        capacity_kwh < 10.0,
        np.random.uniform(30.0, 45.0, n_samples),
        np.random.uniform(120.0, 210.0, n_samples),
    )

    noise = np.random.normal(0, 0.35, n_samples)

    soh_list = []
    for cap, odo, age, fc, temp, lim, eff, n in zip(
        capacity_kwh, odometer_km, age_years, fast_charge_pct, ambient_temp_c, charge_limit_pct, efficiency_wh_km, noise
    ):
        res = physics_degradation_sim(cap, odo, age, fc, temp, lim, eff, n)
        soh_list.append(res["soh"])

    df = pd.DataFrame({
        "capacity_kwh": capacity_kwh,
        "odometer_km": odometer_km,
        "age_years": age_years,
        "fast_charge_pct": fast_charge_pct,
        "ambient_temp_c": ambient_temp_c,
        "charge_limit_pct": charge_limit_pct,
        "efficiency_wh_km": efficiency_wh_km,
        "soh": soh_list,
    })
    return df


def train_and_save_model() -> None:
    """Trains the GBDT regressor and lower/upper quantile uncertainty bounds."""
    EV_CALCULATOR_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_training_data(18000)

    features = ["capacity_kwh", "odometer_km", "age_years", "fast_charge_pct", "ambient_temp_c", "charge_limit_pct", "efficiency_wh_km"]
    X = df[features]
    y = df["soh"]

    # Mean predictor
    model_mean = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)
    model_mean.fit(X, y)

    # Lower bound (10th percentile)
    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=90, max_depth=3, random_state=42)
    model_lower.fit(X, y)

    # Upper bound (90th percentile)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=90, max_depth=3, random_state=42)
    model_upper.fit(X, y)

    artifact = {
        "model_mean": model_mean,
        "model_lower": model_lower,
        "model_upper": model_upper,
        "features": features,
        "feature_importances": dict(zip(features, [round(float(v), 4) for v in model_mean.feature_importances_])),
        "version": "1.0.0",
    }

    out_file = EV_CALCULATOR_DIR / "model.joblib"
    joblib.dump(artifact, out_file)
    print(f"EV Calculator model saved to {out_file}")


def predict_ev_health(
    capacity_kwh: float,
    odometer_km: float,
    age_years: float,
    fast_charge_pct: float = 15.0,
    ambient_temp_c: float = 25.0,
    charge_limit_pct: float = 90.0,
    efficiency_wh_km: float = 140.0,
    rated_range_km: float | None = None,
) -> dict:
    """Predicts SOH, remaining capacity, range, lifespan to 80% EOL, and recommendations."""
    model_path = EV_CALCULATOR_DIR / "model.joblib"
    if not model_path.exists():
        train_and_save_model()

    artifact = joblib.load(model_path)
    X = np.array([[capacity_kwh, odometer_km, age_years, fast_charge_pct, ambient_temp_c, charge_limit_pct, efficiency_wh_km]])

    pred_mean = float(np.clip(artifact["model_mean"].predict(X)[0], 50.0, 100.0))
    pred_lower = float(np.clip(artifact["model_lower"].predict(X)[0], 48.0, pred_mean))
    pred_upper = float(np.clip(artifact["model_upper"].predict(X)[0], pred_mean, 100.0))

    # Detailed physics decomposition
    phys = physics_degradation_sim(capacity_kwh, odometer_km, age_years, fast_charge_pct, ambient_temp_c, charge_limit_pct, efficiency_wh_km)

    current_usable_kwh = float(round((pred_mean / 100.0) * capacity_kwh, 2))
    capacity_loss_kwh = float(round(capacity_kwh - current_usable_kwh, 2))

    # Rated range vs estimated current range
    if rated_range_km is None or rated_range_km <= 0:
        rated_range_km = (capacity_kwh * 1000.0) / max(efficiency_wh_km, 1.0)
    current_estimated_range_km = float(round((pred_mean / 100.0) * rated_range_km, 1))

    # Lifespan extrapolation to 80% SOH (End of Life for EVs)
    degradation_pct = 100.0 - pred_mean
    km_per_year = odometer_km / max(age_years, 0.1)

    if pred_mean <= 80.0:
        km_to_eol = 0.0
        years_to_eol = 0.0
        eol_status = "End of Life (EOL) Reached (<80% SOH)"
    else:
        # Rate of degradation per 10k km
        deg_per_km = degradation_pct / max(odometer_km, 100.0)
        remaining_soh_to_lose = pred_mean - 80.0
        km_to_eol = float(round(remaining_soh_to_lose / max(deg_per_km, 0.00001), 0))
        km_to_eol = min(km_to_eol, 500000.0)
        years_to_eol = float(round(km_to_eol / max(km_per_year, 1000.0), 1))
        years_to_eol = min(years_to_eol, 25.0)
        eol_status = "Healthy" if pred_mean >= 90.0 else "Good Condition" if pred_mean >= 85.0 else "Moderate Wear"

    # Future SOH trajectory over kilometers (0 to max(200k, odo + 100k))
    max_sim_km = max(200000.0, odometer_km + 80000.0)
    sim_km_steps = np.linspace(0, max_sim_km, 25)
    trajectory = []
    for km in sim_km_steps:
        sim_age = (km / max(km_per_year, 1000.0))
        sim_res = physics_degradation_sim(capacity_kwh, km, sim_age, fast_charge_pct, ambient_temp_c, charge_limit_pct, efficiency_wh_km)
        trajectory.append({
            "odometer_km": int(round(km)),
            "soh": sim_res["soh"],
            "usable_kwh": round((sim_res["soh"] / 100.0) * capacity_kwh, 2),
            "range_km": round((sim_res["soh"] / 100.0) * rated_range_km, 1),
        })

    # AI Battery Care Recommendations
    recommendations = []
    if fast_charge_pct > 30.0:
        potential_gain = round((fast_charge_pct - 15.0) * 0.05 * (age_years ** 0.5), 1)
        recommendations.append({
            "type": "fast_charging",
            "title": "Reduce DC Fast Charging Frequency",
            "desc": f"Your fast-charge ratio ({fast_charge_pct:.0f}%) is high. Lowering DCFC to under 20% by using regular AC home/work charging could save ~{potential_gain}% SOH over vehicle life.",
            "impact": "High",
        })

    if charge_limit_pct > 85.0:
        recommendations.append({
            "type": "charge_limit",
            "title": "Cap Daily Charging to 80% - 85%",
            "desc": "High state-of-charge dwell increases electrolyte oxidation. Setting a daily charge cap of 80% for local commutes and reserving 100% for road trips reduces calendar aging significantly.",
            "impact": "High",
        })

    if ambient_temp_c > 33.0:
        recommendations.append({
            "type": "climate",
            "title": "Park in Shade / Garage during Peak Heat",
            "desc": f"Ambient operating temperatures ({ambient_temp_c:.0f}°C) accelerate Arrhenius thermal degradation. Keeping the battery cooled and avoiding charging right after high-speed highway driving protects cell health.",
            "impact": "Medium",
        })

    if not recommendations:
        recommendations.append({
            "type": "general",
            "title": "Excellent Battery Maintenance Habits!",
            "desc": "Your charging patterns and ambient parameters are optimal for maximum lithium-ion cell longevity.",
            "impact": "Optimal",
        })

    return {
        "soh_predicted": round(pred_mean, 2),
        "soh_lower_bound": round(pred_lower, 2),
        "soh_upper_bound": round(pred_upper, 2),
        "current_usable_kwh": current_usable_kwh,
        "capacity_loss_kwh": capacity_loss_kwh,
        "original_capacity_kwh": capacity_kwh,
        "rated_range_km": round(rated_range_km, 1),
        "current_estimated_range_km": current_estimated_range_km,
        "range_loss_km": round(rated_range_km - current_estimated_range_km, 1),
        "status": eol_status,
        "equivalent_full_cycles": phys["efc"],
        "lifespan_estimate": {
            "km_to_eol_80": km_to_eol,
            "years_to_eol_80": years_to_eol,
            "estimated_annual_km": round(km_per_year, 0),
        },
        "breakdown": {
            "calendar_loss_pct": phys["calendar_loss_pct"],
            "cycle_loss_pct": phys["cycle_loss_pct"],
            "fast_charge_loss_pct": phys["fast_charge_loss_pct"],
            "thermal_loss_pct": phys["thermal_loss_pct"],
            "total_loss_pct": phys["total_loss_pct"],
        },
        "trajectory": trajectory,
        "recommendations": recommendations,
        "presets": PRESETS,
    }


if __name__ == "__main__":
    print("Training EV Calculator model...")
    train_and_save_model()
