# VoltGuard AI — FastAPI Backend Service

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Uvicorn](https://img.shields.io/badge/ASGI-Uvicorn-purple.svg)](https://www.uvicorn.org/)

High-performance RESTful microservice for battery State of Health (SOH) forecasting, physics-informed EV health estimation, Remaining Useful Life (RUL) calculation, and SHAP explainability.

**Live Backend**: [https://ev-battery-degradation-1dir.onrender.com](https://ev-battery-degradation-1dir.onrender.com)  
**Swagger API Docs**: [https://ev-battery-degradation-1dir.onrender.com/docs](https://ev-battery-degradation-1dir.onrender.com/docs)

---

## API Endpoints

### 1. Real-World EV Calculator
- `GET /api/ev-calculator/presets`: Returns 2-wheeler & 4-wheeler presets (*Tesla, Tata, MG, Ather, Ola, etc.*).
- `POST /api/ev-calculator/predict`: Predicts SOH, usable capacity (kWh), remaining range, lifespan to 80% EOL, and AI battery care tips.

### 2. Laboratory Cycler Diagnostics
- `GET /api/health`: Health check and available datasets list.
- `GET /api/datasets`: Dataset metadata and configuration.
- `GET /api/datasets/{dataset}/batteries`: Batteries with initial/latest SOH.
- `GET /api/datasets/{dataset}/fleet`: Fleet SOH trajectory overview.
- `GET /api/batteries/{id}/summary`: Summary metrics for a cell.
- `GET /api/batteries/{id}/history`: Full cycle history with downsampling.
- `POST /api/predict/soh`: SOH forecast with physics-constrained what-if simulation.
- `POST /api/predict/rul`: RUL estimation with user-defined EOL threshold.
- `GET /api/models/performance`: Leave-One-Battery-Out cross-validation benchmarks.
- `GET /api/explainability`: Global SHAP importance values.
- `GET /api/explainability/{id}`: Local SHAP explanations for a cycle.

---

## Local Development

```bash
# Activate virtualenv
.\.venv\Scripts\activate

# Run Uvicorn dev server
uvicorn backend.app.main:app --reload --port 8000
```
API runs on `http://127.0.0.1:8000`.
