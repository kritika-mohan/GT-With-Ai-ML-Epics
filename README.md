# AI-ML Assisted Game-Theoretic Framework for Sustainable Energy Optimization

## Project Overview

This university research project combines **Machine Learning (ML) forecasting** with **Game Theory (Stackelberg Leader-Follower framework)** to optimize electricity pricing, consumption, energy allocation, and sustainability using real Indian energy grid data.

**Dataset:** Electricity Demand, Solar and Wind Generation Data of India at Hourly Interval  
**Source:** [Mendeley Data – y58jknpgs8/2](https://data.mendeley.com/datasets/y58jknpgs8/2)  
**Weather Data:** [NASA POWER API](https://power.larc.nasa.gov/)

---

## Project Structure

```
AI_Energy_Optimization/
│
├── data/
│   ├── raw/                   ← Place downloaded Excel file here
│   ├── processed/             ← Cleaned datasets (auto-generated)
│   └── weather/               ← NASA POWER weather data (auto-fetched)
│
├── notebooks/
│   ├── 01_Energy_Data_Analysis_and_ML_Forecasting.ipynb   ← Main ML Notebook
│   └── 02_Stackelberg_Game_Theory_Optimization.ipynb      ← Game Theory Notebook (Phase 2)
│
├── models/                    ← Saved trained ML models
├── results/
│   └── plots/                 ← All EDA and forecast visualizations
├── dashboard/                 ← (Future) Interactive dashboard
│
└── README.md                  ← This file
```

---

## How to Run the Project

### Step 1: Download the Dataset

1. Go to: https://data.mendeley.com/datasets/y58jknpgs8/2
2. Download the Excel file (expected name: `January 2024 - June 2025.xlsx`)
3. Place the file inside: `AI_Energy_Optimization/data/raw/`

### Step 2: Install Required Python Libraries

Open a terminal/command prompt and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm requests joblib openpyxl
```

### Step 3: Open Jupyter Notebook

```bash
cd AI_Energy_Optimization
jupyter notebook notebooks/01_Energy_Data_Analysis_and_ML_Forecasting.ipynb
```

### Step 4: Run All Cells

In Jupyter Notebook, go to: **Kernel → Restart & Run All**

The notebook will:
- Load and inspect the dataset
- Clean and explore the data
- Engineer features
- Train multiple ML models
- Compare model performance
- Generate forecasts
- Export results for Game Theory integration

---

## Python Version & Library Requirements

| Library       | Version (Tested) | Purpose                        |
|---------------|-----------------|--------------------------------|
| Python        | 3.8+            | Core language                  |
| pandas        | 2.x             | Data manipulation              |
| numpy         | 1.x / 2.x       | Numerical computation          |
| matplotlib    | 3.x             | Visualizations                 |
| seaborn       | 0.13.x          | Statistical visualizations     |
| scikit-learn  | 1.x             | ML models & metrics            |
| xgboost       | 2.x             | Gradient boosted trees         |
| lightgbm      | 4.x             | Light gradient boosting        |
| requests      | 2.x             | NASA POWER API calls           |
| joblib        | 1.x             | Model saving/loading           |
| openpyxl      | 3.x             | Excel file reading             |

---

## Key Outputs

| Output                                      | Location                              |
|---------------------------------------------|---------------------------------------|
| Cleaned dataset CSV                         | `data/processed/`                     |
| EDA visualizations                          | `results/plots/`                      |
| Model comparison table                      | `results/model_comparison.csv`        |
| Actual vs Predicted plots                   | `results/plots/`                      |
| Forecast results for Game Theory            | `results/ml_forecast_for_game_theory.csv` |
| Trained ML model                            | `models/best_model.pkl`               |

---

## Research Team

University Group Research Project  
Topic: AI-ML Assisted Game-Theoretic Framework for Sustainable Energy Optimization  
Data: Grid-India hourly electricity data (January 2024 – June 2025)

---

## Important Notes

- **Do NOT modify files in `data/raw/`** — the notebook works from the original Excel file.
- The notebook handles **missing data, outliers, and time-series gaps** automatically.
- If the Excel file is not found, the notebook will display a clear error message with instructions.
- Synthetic data is used **only** for Game Theory simulation parameters (pricing/consumer response), and this is explicitly labeled in the notebook.
