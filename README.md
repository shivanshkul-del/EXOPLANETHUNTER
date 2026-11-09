# EXOPLANETHUNTER
# 🛰️ AI Powered Exoplanet Hunter
**Interactive Exoplanet Transit Observatory & Mission Control Simulator**

**Team Members:* Shivansh Kulshrestha and Samay Gurjar

## Overview
**AI Powered Exoplanet Hunter** combines an ESP32-S3 microcontroller, a light-dependent resistor (LDR) optical photometer, a motorized planetary transit rig, and an integrated Python Streamlit Mission Control dashboard (`live_dashboard_nasa_plus.py`) to simulate how space telescopes (Kepler, TESS, JWST) detect transiting exoplanets and forecast orbital timing using Machine Learning.

## Key Features (`live_dashboard_nasa_plus.py`)
- **Live 3-Trace Light Curve:** Plots real-time star brightness (%), dynamic median baseline estimate (%), and active transit detection threshold (%) over a customizable 100–1,800 reading window.
- **3-Reading Transit Detector & Mission Timeline:** Requires 3 consecutive low readings to trigger a candidate dip and 3 consecutive recovered readings to confirm a completed transit, logging every stage (`STARTUP`, `CALIBRATION`, `CANDIDATE DIP`, `TRANSIT DETECTED`, `INTERVAL MEASURED`) to a live timestamped Mission Timeline.
- **Photometric Math, Habitable Zone & Candidate Scoring:**
  - Calculates transit depth, duration, and planet-to-star radius ratio (`Rp/Rs = sqrt(depth)`).
  - Computes an illustrative solar-scaled Habitable Zone band (`0.95 * sqrt(L)` to `1.67 * sqrt(L)` AU).
  - Evaluates each event with a 0–100 **Candidate Review Score** weighted by transit depth (45%), duration (20%), and interval repeatability (35%).
- **Random Forest Interval Predictor & Diagnostics:** Loads a pre-trained Scikit-Learn `RandomForestRegressor` (`n_estimators=100`, `max_depth=3`, `min_samples_leaf=2`) using the latest 3 transit intervals to forecast the next transit with a live countdown timer, model diagnostics panel, MAE/MAPE telemetry, and a live **Predicted vs. Actual** line chart.
- **Session Reports & Multi-Run Comparison:** Generates downloadable Markdown (`.md`) session reports, CSV event exports, and read-only statistical comparisons across saved experimental CSV runs.

## Repository Structure
- `live_dashboard_nasa_plus.py` — Flagship single-page Streamlit Mission Control dashboard with full habitability, timeline, and Predicted-vs-Actual analytics.
- `live_dashboard_nasa_upgrade.py` — Streamlined version of the live Streamlit dashboard.
- `train_ai_final.py` — Applies a 3-point rolling median filter to experimental CSV runs, constructs 3-lag sliding windows, and trains/serializes the Random Forest model.
- `validate_ai.py` — Performs Leave-One-Recording-Out cross-validation across three experimental runs (`data/run_*.csv`), benchmarking ML MAE against a median baseline.
- `analyse_transit.py` — Offline Matplotlib light-curve analysis script (80th-percentile baseline, dip depth, and radius ratio).
- `models/transit_interval_model_final.joblib` — Pre-trained Random Forest model loaded by the dashboard.
- `data/` — Recorded experimental sensor datasets (`time_ms`, `light_mV`, `brightness_pct`).

## Quick Start
1. Install dependencies:
   ```bash
   pip install streamlit streamlit-autorefresh pandas numpy scikit-learn joblib pyserial matplotlib

2. Launch the primary Mission Control dashboard:

```Bash
python -m streamlit run live_dashboard_nasa_plus.py
```
3. Run offline light-curve analysis or ML cross-validation (no hardware required):

```Bash
python analyse_transit.py data/run_20261026_170039.csv
python validate_ai.py
