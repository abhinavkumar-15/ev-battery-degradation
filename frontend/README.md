# VoltGuard AI — Frontend Dashboard

[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4.0-38B2AC.svg)](https://tailwindcss.com)

A modern, responsive dashboard for EV battery State of Health (SOH) forecasting, real-world vehicle health calculation, Remaining Useful Life (RUL) estimation, and SHAP explainability.

**Live Application**: [https://ev-battery-degradation.vercel.app](https://ev-battery-degradation.vercel.app)

---

## Features

- **EV SOH & Range Calculator**: Real-world consumer calculator with 2-wheeler and 4-wheeler presets (*Tesla, Tata, MG, Hyundai, Ather, Ola, TVS, Custom*), interactive odometer/age/charging frequency sliders, degradation factor breakdown, and actionable AI care tips.
- **Fleet Overview**: SOH distribution gauge, fleet health summaries, and multi-cell trajectory comparisons.
- **Battery Analysis**: Deep-dive into cycling history (voltage, current, temperature, impedance $R_e$/$R_{ct}$, coulombic efficiency) with interactive charting and cell-to-cell comparisons.
- **Prediction & What-If Sandbox**: Multi-horizon SOH forecast fan charts with empirical 90% confidence bands ($5$ to $50$ cycles ahead), RUL estimates, and physics-constrained what-if slider simulations (temperature and discharge rate).
- **Explainability (SHAP)**: Global feature importance rankings and local waterfall contributions for individual prediction cycles with clear causal disclaimer badges.
- **Model Performance**: Held-out Leave-One-Battery-Out (LOBO) benchmark curves, MAE/RMSE across horizons, and residual diagnostics.
- **Dataset Explorer**: Data quality summaries, statistics, and CSV download capabilities.
- **Theme Support**: Seamless dark/light mode toggle with persistence.

---

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Development Mode
```bash
npm run dev
```
Runs the development server on [http://localhost:5173/](http://localhost:5173/). API requests to `/api/*` are automatically proxied to the local backend on `http://127.0.0.1:8000`.

### 3. Production Build
```bash
npm run build
```
Generates optimized static assets inside `dist/`.

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE` | `""` (empty string) | Base URL of the backend API. In local dev, leave empty to use Vite's proxy. In production (e.g. Vercel), set to your hosted backend URL (`https://ev-battery-degradation-1dir.onrender.com`). |

---

## Deployment (Vercel)

1. Connect your repository to **Vercel**.
2. Set **Root Directory** to `frontend`.
3. Set **Build Command** to `npm run build`.
4. Set **Output Directory** to `dist`.
5. Add `VITE_API_BASE` = `https://ev-battery-degradation-1dir.onrender.com`.
