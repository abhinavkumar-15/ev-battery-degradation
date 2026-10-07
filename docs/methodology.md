# Methodology

## Targets
* **SOH (%)** = capacity ÷ reference capacity × 100. Reference = dataset's documented rated capacity (NASA: 2 Ah). If none exists, `first_valid` (first measured capacity of each cell) is used and reported in the quality report; it is never chosen silently.
* **Capacity loss** = 100 − SOH.
* **Degradation rate** = (SOH[t−1] − SOH[t]) / (cycle[t] − cycle[t−1]), in %SOH/cycle; trailing means over 10/25/50 cycles smooth measurement noise (they only look backwards).
* **RUL** = first cycle at which observed SOH ≤ EOL threshold (default 80 %) − current cycle. Cells that never reach the threshold are right-censored and excluded from RUL training/scoring.

## Features (all trailing-window; see `src/features/history.py::FEATURE_DOCS`)
Why each family exists: *cycle / cumulative throughput / equivalent full cycles* are ageing clocks (degradation is cumulative);
*temperature and current statistics* represent operating stress (reaction rates and lithium-plating/SEI growth depend on both);
*Re/Rct* are the resistance growth indicators that accompany ageing; *SOH slope and rolling degradation rate* encode the recent trajectory (the strongest practical predictor of near-term SOH);
*coulombic efficiency* flags side-reaction losses. Features missing from a dataset are dropped, never invented.
Deliberately **not** created (documented in `EXCLUDED_FEATURES`): capacity slopes (exactly collinear with SOH slopes), `cycles_since_reference_point` (= cycle), depth of discharge (fixed cut-off ⇒ ≈ SOH), temperature rise (identical to range with the stored statistics), DC internal resistance (not measured).

## Forecasting formulation
At cycle *t* the model sees only features of cycles ≤ *t* and predicts **ΔSOH = SOH[t+h] − SOH[t]**; forecast = SOH[t] + ΔSOH. Predicting the delta stops the model getting credit for the trivially known current level (we report R² on both absolute SOH and Δ).
Reference points always reported next to the ML models: *persistence* (SOH stays flat) and *linear extrapolation* of the last-10-cycle slope. A model that cannot beat extrapolation adds no value.

## Validation and temporal leakage
Rows of one battery are strongly autocorrelated and targets look forward in time. A random row split would train on cycle 101 and test on cycle 100 — the test point can be interpolated from its neighbours, giving optimistic scores unrelated to forecasting a *new* cell.
Defences: (1) trailing-only features (unit-tested: truncating the future leaves earlier features unchanged); (2) **leave-one-battery-out** evaluation — each battery is held out entirely; (3) hyper-parameter tuning with GroupKFold by battery inside each training fold; (4) scalers/imputers live inside the sklearn Pipeline so they are fitted on training rows only; (5) `chronological_split` (70/15/15 per battery) is available for single-cell forecasting.
Caveat: the "best model" is chosen on the same LOBO scores that are reported, so they are mildly optimistic. A final untouched test battery set needs more cells than NASA's core four provide.

## Feature screening
Features that are exact duplicates (|Pearson r| ≥ 0.99999, e.g. capacity ≡ SOH × reference/100) are removed automatically — unsupervised, earlier features win — because perfectly collinear inputs make ordinary least squares pick huge offsetting coefficients (which we observed) and make SHAP credit arbitrary. The dropped pairs are stored in `run_meta.json` and shown in the UI. Looser thresholds (0.999) were tried and rejected because they also removed legitimate history features. Near-collinear features that remain (e.g. ambient vs rolling temperature) can still receive offsetting SHAP credit.

## What-if scenarios
A scenario shifts a whole feature family together — *all* temperature features by Δ°C, or *all* discharge-current features by a scale factor — so the input vector stays internally consistent (overriding one feature alone produced nonsensical forecasts, including SOH *rising*). Shifts are limited to what every model's training data covers (`/api/scenario/limits`), and requests outside it are rejected. SOH forecasts are constrained to be non-increasing (disclosed in the API response). A scenario is a statistical what-if on the fitted model, **not a physical simulation**.

## RUL estimation
Two independent estimators: direct ML regression of RUL on the cycle-*t* features, and trajectory extrapolation (SOH[t] − threshold) ÷ trailing degradation rate (no ML). The direct model is tied to the threshold used at training time; the API reports extrapolation only for other thresholds. RUL is an estimate with substantial uncertainty — not a physical measurement.

## Uncertainty
SOH forecasts carry an empirical 5–95 % residual band computed from held-out-battery residuals. It is a coarse, marginal band (not conditional on the input and not guaranteed coverage on unseen cells). RUL has no interval yet.

## Explainability
Tree SHAP (or linear SHAP) on the final model; global importance = mean |SHAP|, local = per-prediction contributions, dependence plot from a sample. SHAP explains the *model*; correlated features share credit and the data are observational, so statements are "associated with the prediction", never "causes degradation". Causal claims require controlled experiments.

## Known limitations
Laboratory cells ≠ real EVs (drive-cycle loads, thermal management, calendar ageing, mixed chemistries); few NASA cells ⇒ high variance across held-out cells; capacity-regeneration spikes are noise for a smooth model; model applies to conditions represented in training data.
