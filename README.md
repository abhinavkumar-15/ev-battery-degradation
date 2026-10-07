# VoltGuard AI — EV Battery Degradation & Remaining Useful Life Prediction

[![CI](https://github.com/abhinavkumar-15/ev-battery-degradation/actions/workflows/ci.yml/badge.svg)](https://github.com/abhinavkumar-15/ev-battery-degradation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC.svg)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Physics-informed system for lithium-ion battery health diagnostics: **State of Health (SOH)** estimation, **degradation rate tracking**, **multi-horizon future SOH forecasting**, **Remaining Useful Life (RUL)** prediction, **Real-World EV SOH & Lifespan Calculator**, and **SHAP explainability** served via a **FastAPI backend** and an interactive **React dashboard**.

---

### 🌐 Live Deployments
- 🚀 **Live Web App**: [https://ev-battery-degradation.vercel.app](https://ev-battery-degradation.vercel.app)
- ⚡ **Backend API**: [https://ev-battery-degradation-1dir.onrender.com](https://ev-battery-degradation-1dir.onrender.com)
- 📖 **Interactive Swagger Docs**: [https://ev-battery-degradation-1dir.onrender.com/docs](https://ev-battery-degradation-1dir.onrender.com/docs)

---

## Key Highlights

- **Real-World EV SOH Calculator**: Instant SOH, usable capacity, remaining range, and lifespan to 80% EOL estimation for consumer EVs (2-wheelers & 4-wheelers: *Tesla Model 3/Y, Tata Nexon EV, MG ZS EV, Hyundai Ioniq 5, Ather 450X, Ola S1 Pro, TVS iQube, Custom*) based on odometer km, vehicle age, charging frequency habits, DC fast-charging ratio, and climate.
- **Leakage-Free Validation**: Evaluated strictly via **Leave-One-Battery-Out (LOBO)** cross-validation with trailing-window features and inner group-based hyperparameter tuning. Random temporal row splitting is avoided to prevent inflated metrics.
- **Multi-Horizon Forecasting**: Predicts future SOH at $h \in \{5, 10, 25, 50\}$ cycles ahead with empirical $90\%$ residual confidence bands.
- **Direct & Extrapolated RUL**: Predicts Remaining Useful Life until the battery drops below the End-of-Life threshold (default: $80\%$ SOH) using both ML models and physical trajectory extrapolation.
- **Physics-Constrained What-If Sandbox**: Simulates the effects of temperature shifts ($\Delta T$) and discharge load scaling ($C$-rate) using Arrhenius thermal acceleration and $C$-rate power laws.
- **Interpretable ML**: Global & local feature attributions via Tree/Linear SHAP with scientific disclaimers separating correlation from physical causation.
- **Dual Laboratory Dataset Support**:
  - **NASA Li-ion Dataset**: Real experimental aging tests from NASA Ames Prognostics Center of Excellence.
  - **Synthetic Demo Dataset**: Built-in 16-cell, 2,440-cycle dataset for instant local execution and smoke testing without external downloads.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Data Ingestion                              │
│   • NASA Li-ion .mat Cycler Datasets                             │
│   • Synthetic 16-cell Cycler Sample                              │
│   • EV Fleet Physics Simulator (Arrhenius + EFC + DCFC + DOD)   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│              Processing & Feature Engineering Pipeline           │
│  • Cycle cleaning, step segmentation & SOH calculation           │
│  • 18+ Trailing features (impedance Re/Rct, voltage/temp slopes) │
│  • Collinearity filter & support range metadata                  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│              Model Training & LOBO Evaluation                    │
│  • Leave-One-Battery-Out cross-validation                        │
│  • Models: Linear Regression, Random Forest, GBDT, Quantile GBR  │
│  • Baselines: Persistence, Linear Extrapolation                  │
│  • Quantile Regressors (10th & 90th percentile uncertainty)      │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Artifacts & Explainability                     │
│  • Saved joblib pipelines with feature bounds                    │
│  • Global & local Tree/Linear SHAP values                        │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
       ┌─────────────────────────┐ ┌─────────────────────────┐
       │   FastAPI Backend API   │ │   React 18 Dashboard    │
       │  • Port 8000            │ │  • Port 5173            │
       │  • Swagger docs (/docs) │ │  • Multi-page analytics │
       │  • EV Calculator Engine │ │  • SOH projection chart │
       └─────────────────────────┘ └─────────────────────────┘
```

---

## Dashboard Pages

| Page | Description |
| :--- | :--- |
| **Fleet Overview** | Single-pane health gauge, fleet distribution, multi-cell trajectory comparisons, and KPI cards. |
| **EV SOH Calculator** | Consumer calculator for real-world EVs with presets (*Tesla, Tata, MG, Ather, Ola*), sliders for km, age, charging frequency, DC fast charging %, climate, degradation factor breakdown, and AI battery care tips. |
| **Battery Analysis** | Cycle-by-cycle history exploring voltage profiles, current, thermal rise, and internal resistance. |
| **Prediction Sandbox** | Multi-horizon forecast fan chart with empirical confidence intervals and interactive what-if sliders. |
| **Explainability (SHAP)** | Global feature importance rankings and local waterfall charts for any cycle. |
| **Dataset Explorer** | Tabular raw feature browser with correlation heatmaps and CSV export. |
| **Model Performance** | LOBO benchmark tables, error vs. horizon curves, actual vs. predicted scatters, and residual diagnostics. |
| **Method & Architecture** | Complete scientific documentation, terminology, formulas, and limitations. |

---

## API Reference

The FastAPI service exposes RESTful JSON endpoints:

### EV SOH Calculator
- `GET /api/ev-calculator/presets`: Returns 2-wheeler and 4-wheeler presets.
- `POST /api/ev-calculator/predict`: Computes SOH %, usable kWh, range, lifespan to 80% EOL, degradation breakdown, and AI battery care advice.

### Laboratory Cycler Diagnostics
- `GET /api/health`: Health status and list of trained datasets.
- `GET /api/datasets`: Available dataset metadata.
- `GET /api/datasets/{dataset}/batteries`: Battery summary list with initial/latest SOH.
- `GET /api/batteries/{id}/summary`: High-level metrics for a specific cell.
- `GET /api/batteries/{id}/history`: Full cycle time-series with pagination and downsampling.
- `POST /api/predict/soh`: SOH forecast with physics-constrained what-if adjustments.
- `POST /api/predict/rul`: RUL estimation with user-defined EOL threshold.
- `GET /api/models/performance`: LOBO validation benchmark metrics.
- `GET /api/explainability`: Global SHAP feature importance.
- `GET /api/explainability/{id}`: Local SHAP waterfall for a specific cycle.

---

## Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm

### 2. Clone Repository
```bash
git clone https://github.com/abhinavkumar-15/ev-battery-degradation.git
cd ev-battery-degradation
```

### 3. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the synthetic training pipeline (instant smoke test)
python -m src.pipeline --dataset demo

# Train the EV Calculator model
python -m src.models.ev_calculator

# Start FastAPI server
uvicorn backend.app.main:app --reload --port 8000
```
Backend API will be running at [http://localhost:8000](http://localhost:8000) (Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs)).

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Running on Real NASA Battery Data

To train on experimental laboratory data:
```bash
# Download NASA battery aging dataset (.mat files)
python -m src.data.download_nasa

# Process features and train LOBO models
python -m src.pipeline --dataset nasa
```

---

## Testing & Quality Assurance

```bash
# Run unit & integration test suite
pytest -v

# Run frontend production build test
cd frontend
npm run build
```

---

## License
MIT License. Created by Abhinav Kumar.
