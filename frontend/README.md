# VoltGuard AI — Frontend Dashboard

A modern, responsive dashboard for EV battery State of Health (SOH) forecasting, Remaining Useful Life (RUL) estimation, and SHAP explainability.

Built with **React 18**, **Vite**, **Tailwind CSS v4**, **Recharts**, and **Lucide Icons**.

---

## Features

- **Overview Dashboard**: SOH distribution gauge, fleet health summaries, and multi-cell trajectory comparisons.
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
| `VITE_API_BASE` | `""` (empty string) | Base URL of the backend API. In local dev, leave empty to use Vite's proxy. In production (e.g. Vercel), set to your hosted backend URL (e.g. `https://voltguard-api.onrender.com`). |

---

## Deployment (Vercel / Netlify)

1. Connect your repository to **Vercel** or **Netlify**.
2. Set **Root Directory** to `frontend`.
3. Set **Build Command** to `npm run build`.
4. Set **Output Directory** to `dist`.
5. Add the environment variable `VITE_API_BASE` pointing to your deployed backend URL.
