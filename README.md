# FSE 570 Capstone — Update 1 (Chicago Traffic Crashes)

This repository contains the Update‑1 analysis workflow for the capstone project:
**Severity‑Driven Traffic Risk Modeling and Optimization for Urban Safety**.

Update‑1 delivers baseline severity models, EDA, spatial hot‑spot clustering,
and a severity‑weighted intersection risk index, along with a quad chart PDF.

## What’s Included in this
- Baseline crash‑severity models (LR/RF/XGBoost or fallback)
- EDA dashboard (time, severity distribution, conditions)
- Spatial hot‑spot clustering (DBSCAN)
- Severity‑weighted intersection risk index
- Temporal trend preview
- Quad chart (template‑styled) for Check‑in 2

## Data
Place the Kaggle CSVs in the same folder as the scripts:
- `Traffic_Crashes_-_Crashes.csv`
- `Traffic_Crashes_-_People.csv`
- `Traffic_Crashes_-_Vehicles.csv`

Note: The current Kaggle release does **not** include `CRASH_RECORD_ID`
in People/Vehicles, so those tables can’t be merged. The pipeline uses
`MOST_SEVERE_INJURY` from the Crashes file to derive severity.

## Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost shap   # optional but recommended
```

## Run Update‑1
```bash
PYTHONUNBUFFERED=1 python3 /Users/snadimi3/Downloads/Capstone/update1_analysis.py
MPLCONFIGDIR=/tmp python3 /Users/snadimi3/Downloads/Capstone/make_quad_chart.py
```

## Outputs
All outputs are written to:
`/Users/snadimi3/Downloads/Capstone/outputs`

Key files:
- `eda_dashboard.png`
- `baseline_model_curves.png`
- `spatial_hotspots.png`
- `temporal_trend.png`
- `top_risk_intersections.csv`
- `top_clusters.csv`
- `baseline_model_metrics.csv`
- `quad_chart_update1.pdf`
- `quad_chart_update1.png`

## Notes and Defaults
- Downsamples to 100,000 rows for fast Update‑1 runs.
- DBSCAN clustering uses haversine distance with 250m radius and Chicago
  geographic bounds to avoid outlier distortion.
- If XGBoost is not installed, the script falls back to
  `HistGradientBoostingClassifier`.

## Scripts
- `update1_analysis.py` — EDA + baseline models + spatial analysis
- `make_quad_chart.py` — Template‑styled quad chart generator
