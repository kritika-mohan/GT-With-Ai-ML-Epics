# AI-ML Assisted Game-Theoretic Framework for Sustainable Energy Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](dashboard/app.py)

**University Group Research Project**  
*A Community Case Study Using Government Open Energy Data*

---

## Executive Summary

This research project develops an end-to-end Machine Learning and Game-Theoretic framework to forecast national electricity demand and renewable energy generation, optimize Time-of-Use ($ToU$) pricing, and incentivize community demand response.

- **Open Data Source:** Grid-India hourly electricity demand, solar, and wind generation (January 2024 – June 2025; 12,984 hourly telemetry records).
- **Core ML Architecture:** Gradient Boosted Decision Trees (**XGBoost** and **LightGBM**) evaluated against Random Forest, Linear Regression, and Naive Persistence baselines under strict chronological partitioning (80% train / 20% test).
- **Game-Theoretic Formulation:** Outputs feed into a **Stackelberg Leader-Follower Game** where the electricity provider (Leader) sets dynamic pricing signals and consumers (Followers) shift flexible demand to absorb solar peaks and minimize net load burden.
- **Interactive Dashboard:** Full Streamlit web application with real-time inference, Duck Curve visualization, model explainability, and game simulation.

---

## Project Architecture

```
GT-With-Ai-ML-Epics/
│
├── dashboard/
│   └── app.py                                         # Interactive Streamlit Web Dashboard
│
├── data/
│   ├── raw/                                           # Grid-India Excel Workbooks
│   │   └── January 2024 - June 2025.xlsx              # Consolidated hourly dataset (12,984 rows)
│   ├── processed/
│   │   ├── energy_data_cleaned.csv                    # Cleaned time-series with Net Load
│   │   └── energy_data_full_features.csv              # 27 engineered lag/cyclical/rolling features
│   └── weather/                                       # NASA POWER weather telemetry
│
├── models/
│   ├── best_model.pkl                                 # Trained champion XGBoost Regressor (1.8 MB)
│   └── model_metadata.json                            # Model features, metrics, training parameters
│
├── notebooks/
│   ├── 01_Energy_Data_Analysis_and_ML_Forecasting.ipynb # Executed Jupyter Notebook (with all outputs & plots)
│   ├── build_notebook.py                              # Notebook generator script
│   └── run_notebook.py                                # Automated headless notebook execution runner
│
├── results/
│   ├── model_comparison.csv                           # Final benchmark comparison across all models
│   ├── ml_forecast_for_game_theory.csv                # Interface feed for Stackelberg Game Theory
│   ├── ml_forecast_sample_1week.csv                   # 168-hour sample for quick inspection
│   └── plots/
│       ├── 01_time_series_overview.png                # Full 18-month time series trend
│       ├── 02_hourly_patterns.png                     # Diurnal demand and renewable profiles
│       ├── 05_correlation_heatmap.png                 # Pearson correlation matrix
│       ├── 13_model_comparison.png                    # Benchmark comparison bar charts
│       ├── 14_best_model_full_analysis.png            # Actual vs. predicted and residual error curves
│       ├── 15_feature_importance.png                  # Top predictive feature rankings
│       └── 16_game_theory_interface.png               # Probabilistic dispatch signals for Game Theory
│
└── README.md                                          # This documentation
```

---

## Model Performance Benchmark

Models were trained on the earliest **10,252 hours** (80%) and evaluated on the subsequent **2,564 hours** (20%) of unseen future grid data:

| Rank | Model Architecture | MAE (MW) | RMSE (MW) | MAPE (%) | $R^2$ Score | Operational Note |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | **LightGBM Regressor** | **2,090.20** | **3,444.22** | **1.061%** | **0.9382** | Fastest training convergence |
| 2 | **XGBoost Regressor** | **2,105.77** | **3,444.06** | **1.068%** | **0.9382** | Selected champion for deployment |
| 3 | **Random Forest Regressor** | 2,414.98 | 3,771.32 | 1.220% | 0.9258 | Robust non-linear ensemble |
| 4 | **Linear Regression (Standardized)**| 2,927.77 | 4,556.19 | 1.478% | 0.8918 | Captures rolling mean trend |
| 5 | **Naive Persistence Baseline** | 3,738.41 | 5,370.46 | 1.866% | 0.8496 | Benchmark reference ($y_t = y_{t-1}$) |

> **Key Takeaway:** Both gradient boosted models achieve exceptional precision on the national grid with **~1.06% MAPE** (average deviation of ~2,000 MW on a 200,000 MW grid) and explain over **93.8% of variance**.

---

## Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/kritika-mohan/GT-With-Ai-ML-Epics.git
cd GT-With-Ai-ML-Epics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard Locally
```bash
streamlit run dashboard/app.py
```

### 4. Run or Inspect the Jupyter Notebook
Open the fully executed notebook directly in Jupyter, VS Code, or GitHub:
```bash
jupyter notebook notebooks/01_Energy_Data_Analysis_and_ML_Forecasting.ipynb
```
Or execute headlessly from the terminal:
```bash
python notebooks/run_notebook.py
```

---

## Deploying to Render (Render.com)

This repository is pre-configured for **1-click / zero-config deployment on Render** as a Free Web Service.

### Option A: Automatic Blueprint Deploy (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Blueprint**.
2. Connect your GitHub repository `kritika-mohan/GT-With-Ai-ML-Epics`.
3. Render will automatically detect [`render.yaml`](render.yaml) and configure the build and start commands.
4. Click **Apply** to deploy.

### Option B: Manual Web Service Setup
1. On Render, click **New +** → **Web Service**.
2. Select your repository: `kritika-mohan/GT-With-Ai-ML-Epics`.
3. Fill in the deployment settings:
   - **Name:** `energy-optimization-dashboard`
   - **Language / Environment:** `Python 3`
   - **Region:** Any (e.g. `Oregon (US West)`)
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
4. Click **Create Web Service**. Render will install dependencies, load the serialized model (`models/best_model.pkl`), and host the live interactive dashboard on a public `onrender.com` URL.

---

## Dashboard Capabilities

The interactive Streamlit dashboard (`dashboard/app.py`) provides four analytical modules:

1. **Live Model Inference Engine:** Interactively adjust time of day, season, and past demand lags to trigger real-time predictions from `models/best_model.pkl`.
2. **Forecasting & Dispatch Explorer:** Explore the actual vs. predicted timeline, 95% uncertainty confidence bands, and the Duck Curve phenomenon.
3. **Model Benchmark & Explainability:** Side-by-side metric comparisons and feature importance rankings.
4. **Stackelberg Game Theory Simulator:** Adjust peak, off-peak, and solar valley electricity tariffs along with consumer price elasticity ($\epsilon$) to visualize how price signals shave peak grid demand and abate carbon emissions.

---

## Academic Citation & Acknowledgments

- **Dataset:** Grid-India (2024). *Electricity Demand, Solar and Wind Generation Data of India at Hourly Interval*. Mendeley Data, V2. [https://doi.org/10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2)
- **Meteorological Data:** NASA Langley Research Center POWER Project. [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
