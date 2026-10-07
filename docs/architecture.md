# Architecture

```
 raw files ──► DatasetAdapter (nasa | generic_csv | demo) ──► Common Battery Data Format (cycle table)
                                                                    │
                      clean_cycle_table ◄───────────────────────────┘   (quality report, nothing dropped silently)
                            │
        targets (SOH, loss, rates, RUL)  ──►  history features (trailing windows)  ──►  modelling table
                            │
        leave-one-battery-out  ──► Linear / RandomForest / XGBoost (+ no-model references), GroupKFold tuning
                            │
   metrics JSON · predictions CSV · figures · SHAP importance · joblib artifacts (pipeline + features + ranges + residual band)
                            │
              FastAPI (backend/app) loads artifacts only ──► React dashboard (frontend/src)
```

* `src/data/adapters/` – one adapter per dataset; add a new dataset by emitting the columns in `src/data/schema.py`.
* `src/models/` – zoo (estimators as sklearn Pipelines), tuning, forecasting, RUL, validation, inference helpers.
* `backend/app/` – `api/routes.py` (endpoints), `services/store.py` (artifact/table loading), `schemas/` (request validation).
* Artifacts: `models/<dataset>/forecast_h{H}.joblib`, `models/<dataset>/rul.joblib`; metrics in `outputs/metrics/<dataset>/`.
* Resumable training: `src/models/cache.py` caches finished held-out folds so long runs can be split across sessions.
* Every API response and UI card is tagged `measured`, `derived` or `predicted`; demo data is labelled in the UI banner and API payloads.
