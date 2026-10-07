# VoltGuard AI — EV Battery Degradation & Remaining Useful Life Prediction

[![CI](https://github.com/Abhinavkumar003/ev-battery-degradation/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhinavkumar003/ev-battery-degradation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC.svg)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning system for lithium-ion battery health diagnostics: **State of Health (SOH)** estimation, **degradation rate tracking**, **multi-horizon future SOH forecasting**, **Remaining Useful Life (RUL)** prediction, and **SHAP explainability** served via a **FastAPI backend** and an interactive **React dashboard**.

---

## Key Highlights

- **Leakage-Free Validation**: Evaluated strictly via **Leave-One-Battery-Out (LOBO)** cross-validation with trailing-window features and inner group-based hyperparameter tuning. Random temporal row splitting is avoided to prevent inflated metrics.
- **Multi-Horizon Forecasting**: Predicts future SOH at $h \in \{5, 10, 25, 50\}$ cycles ahead with empirical $90\%$ residual confidence bands.
- **Direct & Extrapolated RUL**: Predicts Remaining Useful Life until the battery drops below the End-of-Life threshold (default: $80\%$ SOH) using both ML models and physical trajectory extrapolation.
- **Physics-Constrained What-If Sandbox**: Simulates the effects of temperature shifts ($\Delta T$) and discharge load scaling ($C$-rate) using Arrhenius thermal acceleration and $C$-rate power laws, with bounds clamped to the training support.
- **Interpretable ML**: Global & local feature attributions via Tree/Linear SHAP with scientific disclaimers separating correlation from physical causation.
- **Dual Dataset Support**:
  - **NASA Li-ion Dataset**: Real experimental aging tests from NASA Ames Prognostics Center of Excellence.
  - **Synthetic Demo Dataset**: Built-in 16-cell, 2,440-cycle dataset for instant local execution and smoke testing without external downloads.

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│               Raw Cycling Data                         │
│   (NASA .mat files / Synthetic Demo 16-cell CSV)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│           Data Processing & Feature Pipeline           │
│  • Cleaning, step segmentation & SOH computation       │
│  • 18+ Trailing features (OLS slopes, rolling means,   │
│    impedance Re/Rct, cumulative throughput)            │
│  • Collinearity filter (unsupervised duplicate prune)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          Model Training & LOBO Evaluation              │
│  • Leave-One-Battery-Out cross-validation              │
│  • Models: Linear Regression, Random Forest, XGBoost   │
│  • Baselines: Persistence, Linear Extrapolation        │
│  • Inner GroupKFold hyperparameter tuning              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Artifacts & Explainability               │
│  • Saved joblib pipelines with training ranges         │
│  • Global & local Tree/Linear SHAP values              │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│   FastAPI Backend API   │ │   React 18 Dashboard    │
│  • Port 8000            │ │  • Port 5173            │
│  • Swagger docs (/docs) │ │  • Multi-page analytics │
│  • What-if validation   │ │  • Dark/Light theme     │
└─────────────────────────┘ └─────────────────────────┘
```

---

## Dashboard Walkthrough

| Feature View | Screenshot |
| :--- | :--- |
| **Fleet Overview** | Single-pane health gauge, fleet distribution, multi-cell trajectory comparisons, and KPI cards. |
| **Battery Analysis** | Cycle-by-cycle history exploring voltage profiles, current, thermal rise, and impedance metrics. |
| **Prediction Sandbox** | Multi-horizon forecast fan chart with empirical confidence intervals and interactive what-if sliders. |
| **Explainability (SHAP)**| Global feature importance rankings and local waterfall charts for any cycle. |
| **Model Performance** | LOBO benchmark tables, error-vs-horizon curves, residual histograms, and parity plots. |
| **Dataset Explorer** | Data quality reports, outlier flags, descriptive statistics, and CSV data export. |

---

## Project Structure

```
ev-battery-degradation/
├── backend/                  # FastAPI Application
│   └── app/
│       ├── api/routes.py     # REST API endpoints
│       ├── schemas/          # Pydantic request/response models
│       ├── services/store.py # Artifact & dataset loader
│       └── main.py           # FastAPI application entrypoint
├── data/
│   ├── raw/                  # Downloaded NASA .mat files (git-ignored)
│   ├── processed/            # Generated modelling tables (git-ignored)
│   ├── sample/               # Committed synthetic demo sample dataset
│   └── README.md             # Dataset acquisition & convention guide
├── docs/                     # Documentation & screenshots
│   ├── architecture.md       # Detailed technical design
│   └── methodology.md        # Mathematical formulation & leakage prevention
├── frontend/                 # React 18 + Vite + Tailwind CSS Dashboard
│   ├── src/
│   │   ├── pages/            # Overview, Analysis, Prediction, Explainability, etc.
│   │   ├── charts/           # Recharts components (fan chart, residuals, etc.)
│   │   ├── components/       # Reusable UI widgets & inputs
│   │   └── services/api.js   # Frontend API client
│   └── package.json
├── models/                   # Serialized ML pipelines & metadata (.joblib)
│   ├── demo/                 # Demo models (5, 10, 25, 50 horizons + RUL)
│   └── nasa/                 # NASA trained models
├── outputs/                  # Evaluation outputs
│   ├── figures/              # Generated publication-quality PNG charts
│   ├── metrics/              # JSON performance metrics & run metadata
│   └── predictions/          # Held-out predictions CSV
├── src/                      # Core ML Library
│   ├── data/adapters/        # NASA, Demo, and Generic CSV data adapters
│   ├── preprocessing/        # Data cleaning, outlier filtering, step pairing
│   ├── features/             # Trailing-window feature engineering & targets
│   ├── models/               # Model zoo, training pipeline, LOBO evaluation, RUL
│   ├── explainability/       # Tree & Linear SHAP explainability
│   ├── visualization/        # Matplotlib plot generation
│   ├── config.py             # Global reproducible config & paths
│   └── pipeline.py           # End-to-end CLI training orchestrator
├── tests/                    # Pytest test suite (28 unit & integration tests)
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

---

## Quick Start (Local Setup)

### 1. Prerequisites
- **Python**: $\ge 3.10$ (developed and tested on $3.12$)
- **Node.js**: $\ge 18$

### 2. Clone & Install

```bash
# Clone the repository
git clone https://github.com/Abhinavkumar003/ev-battery-degradation.git
cd ev-battery-degradation

# Set up Python virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Run Pipeline (Demo Dataset)

Train models, evaluate LOBO folds, compute SHAP values, and export metrics:

```bash
# Train synthetic demo pipeline across all horizons (h = 5, 10, 25, 50) and RUL
python -m src.pipeline --dataset demo

# Or run the quick smoke test (headline horizon h=10 only)
python -m src.pipeline --dataset demo --quick
```

### 4. Start the Application

Open two terminal tabs:

**Terminal 1 — Backend API:**
```bash
uvicorn backend.app.main:app --reload --port 8000
```
* Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

**Terminal 2 — Frontend Dashboard:**
```bash
cd frontend
npm run dev
```
* Dashboard URL: [http://localhost:5173/](http://localhost:5173/)

---

## Training on Real NASA Data

1. Download the NASA PCoE dataset from the [NASA Data Repository](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets).
2. Extract the `.mat` files (`B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`) into `data/raw/nasa/`.
3. Run the training pipeline:
   ```bash
   python -m src.pipeline --dataset nasa
   ```
4. Refresh your dashboard and switch the dataset dropdown to **NASA** to explore real experimental degradation.

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health and available datasets status |
| `GET` | `/api/datasets` | List all available datasets and their metadata |
| `GET` | `/api/datasets/{id}/batteries` | List all cells in a dataset with summary stats |
| `GET` | `/api/datasets/{id}/fleet` | Downsampled multi-cell SOH trajectories for fleet comparisons |
| `GET` | `/api/datasets/{id}/summary` | Aggregate statistics and data quality report |
| `GET` | `/api/batteries/{id}/summary` | Current SOH, capacity, degradation rate, and status |
| `GET` | `/api/batteries/{id}/history` | Historical cycle records (voltage, temperature, capacity, etc.) |
| `POST`| `/api/predict/soh` | Multi-horizon SOH forecast with confidence intervals |
| `POST`| `/api/predict/degradation` | Forecast degradation rate (% per cycle) |
| `POST`| `/api/predict/rul` | Remaining Useful Life prediction (ML + trajectory extrapolation) |
| `GET` | `/api/scenario/limits` | Permissible what-if temperature and current scaling bounds |
| `GET` | `/api/models` | List active model architectures, parameters, and training ranges |
| `GET` | `/api/models/performance` | LOBO evaluation metrics (MAE, RMSE, $R^2$, per-battery scores) |
| `GET` | `/api/models/horizons` | Error-vs-horizon comparison curves |
| `GET` | `/api/explainability` | Global SHAP feature importance rankings |
| `GET` | `/api/explainability/{id}` | Local SHAP waterfall attributions for a specific cell and cycle |

---

## Deployment & Hosting Guide

### Option 1: Split Hosting (Recommended — Free Tier)

* **Backend on [Render](https://render.com/)**:
  - Service Type: **Web Service**
  - Build Command: `pip install -r requirements.txt && python -m src.pipeline --dataset demo`
  - Start Command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
* **Frontend on [Vercel](https://vercel.com/)**:
  - Root Directory: `frontend`
  - Build Command: `npm run build`
  - Output Directory: `dist`
  - Environment Variable: `VITE_API_BASE=https://your-backend-app.onrender.com`

### Option 2: Docker Container (Unified)

Deploy a single container to Fly.io, AWS App Runner, or Hugging Face Spaces:

```dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Testing

The project includes a comprehensive test suite covering data adapters, feature leakage prevention, LOBO evaluation, API route responses, and input boundary conditions:

```bash
pytest -v
```

```
tests/test_api.py ......................... [ 25%]
tests/test_features_and_validation.py ..... [ 42%]
tests/test_nasa_adapter.py ................ [ 57%]
tests/test_targets_and_cleaning.py ........ [ 78%]
tests/test_v2_features.py ................. [100%]

======================== 28 passed in 5.2s ========================
```

---

## Scientific & Methodological Disclaimers

1. **Laboratory vs. Real-World EV Behavior**: Laboratory cycling under constant ambient chamber temperatures and constant-current discharge does not capture complex dynamic drive cycles, regenerative braking spikes, or calendar aging in varying weather conditions.
2. **Explainability $\neq$ Causation**: SHAP values indicate feature contributions to the fitted statistical model, not proof of underlying physical electro-chemical degradation mechanisms.
3. **Model Selection**: Held-out scores are computed across individual cells to prevent temporal leakage; however, real-world fleet deployments should calibrate with cell chemistry specifications.

---

## Citation & Attribution

- NASA Ames Prognostics Center of Excellence: *B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.*

---

## License

This project is licensed under the [MIT License](LICENSE).
