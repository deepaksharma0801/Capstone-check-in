"""
FSE 570 Capstone – Status Report 1
Project: Severity-Driven Traffic Risk Modeling and Optimization for Urban Safety
Update 1: Data Integration, Feature Engineering & Baseline Models (Weeks 5–7)

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
    pip install geopandas folium statsmodels pysal libpysal esda
    pip install imbalanced-learn

Datasets (download from Kaggle – Chicago Traffic Crashes):
    Traffic_Crashes_-_Crashes.csv
    Traffic_Crashes_-_People.csv
    Traffic_Crashes_-_Vehicles.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = "./"             # default: current folder with CSVs
OUTPUT_DIR = "./outputs/"
RANDOM_STATE = 42
MAX_ROWS = 100000   # downsample for faster Update-1 runs
TOP_N_CATS = 20     # cap categories to reduce one-hot dimensionality
FORCE_LOAD_PEOPLE_VEHICLES = True  # set True to load full People/Vehicles even if not joinable

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR) + "/" if DATA_DIR == "./" else DATA_DIR
OUTPUT_DIR = str((BASE_DIR / "outputs").resolve()) + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA LOADING & MERGING
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Datasets")
print("=" * 60)

crashes  = pd.read_csv(DATA_DIR + "Traffic_Crashes_-_Crashes.csv",   low_memory=False)

# Read only headers first for People/Vehicles to avoid heavy load if not joinable
people_cols   = pd.read_csv(DATA_DIR + "Traffic_Crashes_-_People.csv",   nrows=0).columns
vehicles_cols = pd.read_csv(DATA_DIR + "Traffic_Crashes_-_Vehicles.csv", nrows=0).columns

people = None
vehicles = None
if FORCE_LOAD_PEOPLE_VEHICLES or ("CRASH_RECORD_ID" in people_cols):
    people = pd.read_csv(DATA_DIR + "Traffic_Crashes_-_People.csv", low_memory=False)
if FORCE_LOAD_PEOPLE_VEHICLES or ("CRASH_RECORD_ID" in vehicles_cols):
    vehicles = pd.read_csv(DATA_DIR + "Traffic_Crashes_-_Vehicles.csv", low_memory=False)

print(f"  Crashes  : {crashes.shape}")
if people is None:
    print("  People   : skipped (no CRASH_RECORD_ID join key in this release)")
else:
    print(f"  People   : {people.shape}")
if vehicles is None:
    print("  Vehicles : skipped (no CRASH_RECORD_ID join key in this release)")
else:
    print(f"  Vehicles : {vehicles.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Feature Engineering")
print("=" * 60)

# --- Temporal features ---
crashes["CRASH_DATE"] = pd.to_datetime(crashes["CRASH_DATE"], errors="coerce")
crashes["HOUR"]       = crashes["CRASH_DATE"].dt.hour
crashes["DAY_OF_WEEK"]= crashes["CRASH_DATE"].dt.dayofweek      # 0=Mon
crashes["MONTH"]      = crashes["CRASH_DATE"].dt.month
crashes["YEAR"]       = crashes["CRASH_DATE"].dt.year
crashes["IS_WEEKEND"] = crashes["DAY_OF_WEEK"].isin([5, 6]).astype(int)
crashes["TIME_OF_DAY"]= pd.cut(
    crashes["HOUR"],
    bins=[-1, 6, 12, 18, 21, 24],
    labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"]
)
date_min = crashes["CRASH_DATE"].min()
date_max = crashes["CRASH_DATE"].max()

# --- Severity target (KABCO-based binary) ---
# Prefer People dataset if join key exists; otherwise fall back to MOST_SEVERE_INJURY in Crashes.
INJURY_SEVERITY_MAP = {
    "FATAL"                    : 5,
    "INCAPACITATING INJURY"    : 4,
    "NONINCAPACITATING INJURY" : 3,
    "REPORTED, NOT EVIDENT"    : 2,
    "NO INDICATION OF INJURY"  : 1,
}

use_people_severity = (people is not None) and ("CRASH_RECORD_ID" in people.columns) and ("CRASH_RECORD_ID" in crashes.columns)
if use_people_severity:
    people["KABCO"] = people["INJURY_CLASSIFICATION"].map(INJURY_SEVERITY_MAP).fillna(1)
    severity_agg = (
        people.groupby("CRASH_RECORD_ID")["KABCO"]
        .max()
        .reset_index()
        .rename(columns={"KABCO": "MAX_KABCO"})
    )
    severity_agg["SEVERITY_BINARY"] = (severity_agg["MAX_KABCO"] >= 3).astype(int)
    df = crashes.merge(severity_agg, on="CRASH_RECORD_ID", how="left")
    print(f"  After crash+severity merge : {df.shape}")
else:
    # Fallback for this Kaggle release: no CRASH_RECORD_ID in People/Vehicles
    df = crashes.copy()
    if "MOST_SEVERE_INJURY" not in df.columns:
        raise KeyError("MOST_SEVERE_INJURY not found in crashes; cannot derive severity.")
    df["MAX_KABCO"] = df["MOST_SEVERE_INJURY"].map(INJURY_SEVERITY_MAP).fillna(1)
    df["SEVERITY_BINARY"] = (df["MAX_KABCO"] >= 3).astype(int)
    print("  Using MOST_SEVERE_INJURY from crashes for severity target.")

# Keep relevant columns for baseline model
MODEL_COLS = [
    "CRASH_RECORD_ID", "HOUR", "DAY_OF_WEEK", "MONTH", "YEAR",
    "IS_WEEKEND", "TIME_OF_DAY",
    "LIGHTING_CONDITION", "WEATHER_CONDITION", "ROADWAY_SURFACE_COND",
    "TRAFFICWAY_TYPE", "PRIM_CONTRIBUTORY_CAUSE",
    "POSTED_SPEED_LIMIT", "NUM_UNITS",
    "HIT_AND_RUN_I", "LATITUDE", "LONGITUDE",
    "MAX_KABCO", "SEVERITY_BINARY"
]
df = df[[c for c in MODEL_COLS if c in df.columns]]

# Clean-up
df["HIT_AND_RUN_I"] = (df.get("HIT_AND_RUN_I", "N") == "Y").astype(int)
df["POSTED_SPEED_LIMIT"] = pd.to_numeric(df["POSTED_SPEED_LIMIT"], errors="coerce")
df.dropna(subset=["SEVERITY_BINARY", "LATITUDE", "LONGITUDE"], inplace=True)

# Reduce high-cardinality categories
for c in ["LIGHTING_CONDITION", "WEATHER_CONDITION", "ROADWAY_SURFACE_COND", "TRAFFICWAY_TYPE", "PRIM_CONTRIBUTORY_CAUSE"]:
    if c in df.columns:
        top_vals = df[c].value_counts().head(TOP_N_CATS).index
        df[c] = df[c].where(df[c].isin(top_vals), other="OTHER")

# Downsample for performance if needed
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Downsampled to {MAX_ROWS} rows for Update‑1 runtime.")

print(f"  Final modelling frame      : {df.shape}")
print(f"  Severe crash rate          : {df['SEVERITY_BINARY'].mean():.2%}")
print(f"  Date range                 : {date_min.date()} → {date_max.date()}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Chicago Traffic Crashes – EDA Dashboard", fontsize=16, fontweight="bold")

# (a) Hourly distribution
ax = axes[0, 0]
hour_counts = df.groupby(["HOUR", "SEVERITY_BINARY"]).size().unstack(fill_value=0)
hour_counts.plot(kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"], width=0.8)
ax.set_title("Crash Count by Hour of Day")
ax.set_xlabel("Hour"); ax.set_ylabel("Count")
ax.legend(["Non-Severe", "Severe"])
ax.tick_params(axis="x", rotation=45)

# (b) Monthly trend
ax = axes[0, 1]
monthly = df.groupby(["YEAR", "MONTH"])["SEVERITY_BINARY"].agg(["sum","count"])
monthly["severe_rate"] = monthly["sum"] / monthly["count"]
# Plot last 3 years
for yr in sorted(df["YEAR"].unique())[-3:]:
    sub = monthly.xs(yr, level="YEAR")
    ax.plot(sub.index, sub["severe_rate"], marker="o", label=str(yr))
ax.set_title("Severe Crash Rate by Month")
ax.set_xlabel("Month"); ax.set_ylabel("Severe Rate")
ax.legend(); ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

# (c) Lighting condition
ax = axes[0, 2]
if "LIGHTING_CONDITION" in df.columns:
    top_light = df["LIGHTING_CONDITION"].value_counts().head(6)
    top_light.plot(kind="barh", ax=ax, color="#6C5CE7")
    ax.set_title("Crashes by Lighting Condition")
    ax.set_xlabel("Count")

# (d) Speed limit vs severity
ax = axes[1, 0]
df_speed = df[df["POSTED_SPEED_LIMIT"].between(5, 80)]
ax.boxplot(
    [df_speed[df_speed["SEVERITY_BINARY"]==0]["POSTED_SPEED_LIMIT"].dropna(),
     df_speed[df_speed["SEVERITY_BINARY"]==1]["POSTED_SPEED_LIMIT"].dropna()],
    labels=["Non-Severe", "Severe"], patch_artist=True,
    boxprops=dict(facecolor="#FDCB6E")
)
ax.set_title("Posted Speed Limit vs Severity")
ax.set_ylabel("Speed Limit (mph)")

# (e) Hit-and-run by time of day
ax = axes[1, 1]
if "TIME_OF_DAY" in df.columns:
    hr_tod = df.groupby("TIME_OF_DAY")["HIT_AND_RUN_I"].mean().sort_values(ascending=False)
    hr_tod.plot(kind="bar", ax=ax, color="#E17055")
    ax.set_title("Hit-and-Run Rate by Time of Day")
    ax.set_ylabel("Rate"); ax.tick_params(axis="x", rotation=30)

# (f) Class balance
ax = axes[1, 2]
sev_counts = df["SEVERITY_BINARY"].value_counts()
ax.pie(sev_counts, labels=["Non-Severe","Severe"], autopct="%1.1f%%",
       colors=["#00B894","#D63031"], startangle=90)
ax.set_title("Severity Class Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "eda_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ EDA dashboard saved → outputs/eda_dashboard.png")

# ─────────────────────────────────────────────
# 4. PREPROCESSING & CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("\n" + "=" * 60)
print("STEP 4 — Preprocessing & Imbalance Handling")
print("=" * 60)

CAT_FEATURES = ["LIGHTING_CONDITION", "WEATHER_CONDITION",
                "ROADWAY_SURFACE_COND", "TRAFFICWAY_TYPE",
                "PRIM_CONTRIBUTORY_CAUSE"]
NUM_FEATURES = ["HOUR", "DAY_OF_WEEK", "MONTH", "IS_WEEKEND",
                "POSTED_SPEED_LIMIT", "NUM_UNITS"]

# Keep only available columns
CAT_FEATURES = [c for c in CAT_FEATURES if c in df.columns]
NUM_FEATURES = [c for c in NUM_FEATURES if c in df.columns]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

X = df[ALL_FEATURES].copy()
y = df["SEVERITY_BINARY"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"  Train : {X_train.shape}, Test : {X_test.shape}")
print(f"  Severe in train : {y_train.mean():.2%}, test : {y_test.mean():.2%}")

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_FEATURES),
    ("cat", cat_pipe, CAT_FEATURES)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Class weights for imbalance
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}
print(f"  Class weights → 0:{cw[0]:.2f}, 1:{cw[1]:.2f}")

# ─────────────────────────────────────────────
# 5. BASELINE MODELS
# ─────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

print("\n" + "=" * 60)
print("STEP 5 — Baseline Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        class_weight=class_weight, max_iter=500, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight=class_weight,
        n_jobs=-1, random_state=RANDOM_STATE),
}
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, scale_pos_weight=cw[1]/cw[0],
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1
    )
else:
    models["HistGradientBoosting (fallback)"] = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=200, random_state=RANDOM_STATE
    )

results = {}
X_train_dense = None
X_test_dense = None
for name, model in models.items():
    X_fit = X_train_proc
    X_eval = X_test_proc
    if "HistGradientBoosting" in name:
        if X_train_dense is None:
            X_train_dense = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else X_train_proc
            X_test_dense = X_test_proc.toarray() if hasattr(X_test_proc, "toarray") else X_test_proc
        X_fit = X_train_dense
        X_eval = X_test_dense
    model.fit(X_fit, y_train)
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)
    results[name] = {"model": model, "y_prob": y_prob, "ROC-AUC": roc, "PR-AUC": pr}
    print(f"\n  ── {name} ──")
    print(f"     ROC-AUC : {roc:.4f}   PR-AUC : {pr:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Non-Severe","Severe"], digits=3))

# ROC + PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Baseline Model Comparison – ROC & Precision-Recall", fontweight="bold")
colors = ["#0984E3", "#00B894", "#E17055"]
for (name, res), col in zip(results.items(), colors):
    RocCurveDisplay.from_predictions(y_test, res["y_prob"], name=f"{name} (AUC={res['ROC-AUC']:.3f})", ax=ax1, color=col)
    PrecisionRecallDisplay.from_predictions(y_test, res["y_prob"], name=f"{name} (AP={res['PR-AUC']:.3f})", ax=ax2, color=col)
ax1.set_title("ROC Curves"); ax2.set_title("Precision-Recall Curves")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "baseline_model_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ Model curves saved → outputs/baseline_model_curves.png")

# ─────────────────────────────────────────────
# 6. SHAP FEATURE IMPORTANCE (XGBoost)
# ─────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    try:
        import shap

        print("\n" + "=" * 60)
        print("STEP 6 — SHAP Feature Importance (XGBoost)")
        print("=" * 60)

        xgb_model = results["XGBoost"]["model"]

        # Feature names after OHE
        ohe_features = list(preprocessor.named_transformers_["cat"]["ohe"]
                             .get_feature_names_out(CAT_FEATURES))
        feature_names = NUM_FEATURES + ohe_features

        # Use TreeExplainer for speed
        explainer  = shap.TreeExplainer(xgb_model)
        shap_sample = X_test_proc[:2000]          # sample for speed
        if hasattr(shap_sample, "toarray"):
            shap_sample_dense = shap_sample.toarray()
        else:
            shap_sample_dense = shap_sample
        shap_values = explainer.shap_values(shap_sample_dense)

        # Bar summary (top 20 features)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, shap_sample_dense,
            feature_names=feature_names,
            max_display=20, plot_type="bar", show=False
        )
        plt.title("SHAP Feature Importance – XGBoost Severity Model", fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + "shap_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ SHAP plot saved → outputs/shap_feature_importance.png")
    except Exception as e:
        print("  SHAP skipped:", e)
else:
    print("\n" + "=" * 60)
    print("STEP 6 — SHAP Feature Importance (skipped: XGBoost not installed)")
    print("=" * 60)

# ─────────────────────────────────────────────
# 7. SPATIAL HOT-SPOT ANALYSIS (DBSCAN)
# ─────────────────────────────────────────────
from sklearn.cluster import DBSCAN

print("\n" + "=" * 60)
print("STEP 7 — Spatial Hot-Spot Analysis (DBSCAN)")
print("=" * 60)

severe_df = df[df["SEVERITY_BINARY"] == 1].dropna(subset=["LATITUDE","LONGITUDE"])
# Filter to realistic Chicago bounds and drop zeros/outliers
severe_df = severe_df[
    (severe_df["LATITUDE"].between(41.60, 42.10)) &
    (severe_df["LONGITUDE"].between(-87.95, -87.45))
].copy()
coords = severe_df[["LATITUDE","LONGITUDE"]].values

# DBSCAN using haversine (meters -> radians)
eps_m = 250  # radius in meters
coords_rad = np.radians(coords)
db = DBSCAN(
    eps=eps_m / 6371000.0,
    min_samples=10,
    metric="haversine",
    n_jobs=-1
).fit(coords_rad)
severe_df = severe_df.copy()
severe_df["CLUSTER"] = db.labels_

n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
noise_pct   = (db.labels_ == -1).mean()
print(f"  DBSCAN clusters found : {n_clusters}")
print(f"  Noise points          : {noise_pct:.1%}")

# Cluster summary
cluster_summary = (
    severe_df[severe_df["CLUSTER"] >= 0]
    .groupby("CLUSTER")
    .agg(Count=("CLUSTER","count"),
         Lat=("LATITUDE","mean"),
         Lon=("LONGITUDE","mean"),
         AvgKABCO=("MAX_KABCO","mean"))
    .sort_values("Count", ascending=False)
    .head(15)
)
print("\n  Top 10 Severe Crash Hot-Spot Clusters:")
print(cluster_summary.head(10).to_string())
cluster_summary.head(10).to_csv(OUTPUT_DIR + "top_clusters.csv", index=True)

# Scatter plot of clusters
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    severe_df["LONGITUDE"], severe_df["LATITUDE"],
    c=severe_df["CLUSTER"], cmap="tab20",
    s=3, alpha=0.5
)
plt.title("Severe Crash Spatial Clusters – Chicago (DBSCAN)", fontweight="bold")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.xlim(-87.95, -87.45)
plt.ylim(41.60, 42.10)
plt.colorbar(scatter, label="Cluster ID")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "spatial_hotspots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Spatial hot-spot map saved → outputs/spatial_hotspots.png")

# ─────────────────────────────────────────────
# 8. SEVERITY-WEIGHTED INTERSECTION RISK INDEX
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Severity-Weighted Risk Index by Intersection")
print("=" * 60)

# Round coordinates to ~100 m grid as proxy for intersection
df["LAT_ROUND"] = df["LATITUDE"].round(3)
df["LON_ROUND"] = df["LONGITUDE"].round(3)

# KABCO weights: Fatal=5, Incap=4, NonIncap=3, etc.
risk_idx = (
    df.groupby(["LAT_ROUND","LON_ROUND"])
    .agg(
        CrashCount   = ("SEVERITY_BINARY","count"),
        SevereCount  = ("SEVERITY_BINARY","sum"),
        AvgKABCO     = ("MAX_KABCO","mean"),
        HitRunRate   = ("HIT_AND_RUN_I","mean")
    )
    .reset_index()
)
# Weighted index: severe count × mean KABCO
risk_idx["RISK_INDEX"] = risk_idx["SevereCount"] * risk_idx["AvgKABCO"]
risk_idx = risk_idx.sort_values("RISK_INDEX", ascending=False).reset_index(drop=True)

print("\n  Top 10 Highest Risk Intersections (proxy by grid):")
print(risk_idx.head(10).to_string(index=False))
risk_idx.head(10).to_csv(OUTPUT_DIR + "top_risk_intersections.csv", index=False)

# ─────────────────────────────────────────────
# 9. HIT-AND-RUN BASELINE CLASSIFIER
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Hit-and-Run Baseline Classifier")
print("=" * 60)

y_hnr = df["HIT_AND_RUN_I"].values
X_hnr = df[ALL_FEATURES].copy()

X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
    X_hnr, y_hnr, test_size=0.2, stratify=y_hnr, random_state=RANDOM_STATE
)

cw_hnr = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr_h)
prep_hnr = ColumnTransformer([
    ("num", num_pipe, NUM_FEATURES),
    ("cat", cat_pipe, CAT_FEATURES)
])

X_tr_h_proc = prep_hnr.fit_transform(X_tr_h)
X_te_h_proc = prep_hnr.transform(X_te_h)

if XGBOOST_AVAILABLE:
    hnr_model = XGBClassifier(
        n_estimators=150, scale_pos_weight=cw_hnr[1]/cw_hnr[0],
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    hnr_model.fit(X_tr_h_proc, y_tr_h)
    hnr_prob = hnr_model.predict_proba(X_te_h_proc)[:,1]
    print(f"  Hit-and-Run XGBoost  ROC-AUC : {roc_auc_score(y_te_h, hnr_prob):.4f}")
    print(f"  Hit-and-Run XGBoost  PR-AUC  : {average_precision_score(y_te_h, hnr_prob):.4f}")
else:
    hnr_model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=200, random_state=RANDOM_STATE
    )
    X_tr_h_dense = X_tr_h_proc.toarray() if hasattr(X_tr_h_proc, "toarray") else X_tr_h_proc
    X_te_h_dense = X_te_h_proc.toarray() if hasattr(X_te_h_proc, "toarray") else X_te_h_proc
    hnr_model.fit(X_tr_h_dense, y_tr_h)
    hnr_prob = hnr_model.predict_proba(X_te_h_dense)[:,1]
    print(f"  Hit-and-Run HGB (fallback) ROC-AUC : {roc_auc_score(y_te_h, hnr_prob):.4f}")
    print(f"  Hit-and-Run HGB (fallback) PR-AUC  : {average_precision_score(y_te_h, hnr_prob):.4f}")

# ─────────────────────────────────────────────
# 10. TEMPORAL TREND (ARIMA PREVIEW)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Temporal Trend Preview")
print("=" * 60)

monthly_severe = (
    df.groupby(["YEAR","MONTH"])["SEVERITY_BINARY"]
    .agg(["sum","count"])
    .reset_index()
)
monthly_severe["Date"] = pd.to_datetime(
    monthly_severe["YEAR"].astype(str) + "-" + monthly_severe["MONTH"].astype(str),
    format="%Y-%m"
)
monthly_severe = monthly_severe.sort_values("Date")

plt.figure(figsize=(14, 4))
plt.plot(monthly_severe["Date"], monthly_severe["sum"],
         color="#6C5CE7", linewidth=1.5, label="Severe Crashes/Month")
plt.fill_between(monthly_severe["Date"], monthly_severe["sum"], alpha=0.15, color="#6C5CE7")
plt.title("Monthly Severe Crash Counts – Chicago (Full History)", fontweight="bold")
plt.xlabel("Date"); plt.ylabel("Severe Crashes")
plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR + "temporal_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Temporal trend saved → outputs/temporal_trend.png")
print("\n  Note: Full S-ARIMA & gradient-boosting forecasting in Update 2 (Weeks 9–11).")

# ─────────────────────────────────────────────
# 11. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Baseline Model Results")
print("=" * 60)
summary = pd.DataFrame([
    {"Model": k, "ROC-AUC": f"{v['ROC-AUC']:.4f}", "PR-AUC": f"{v['PR-AUC']:.4f}"}
    for k, v in results.items()
])
print(summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR + "baseline_model_metrics.csv", index=False)

# ─────────────────────────────────────────────
# 12. SAVE COMPACT SUMMARY FOR QUAD CHART
# ─────────────────────────────────────────────
import json
summary_dict = {
    "dataset_shapes": {
        "crashes": list(crashes.shape),
        "people": None if people is None else list(people.shape),
        "vehicles": None if vehicles is None else list(vehicles.shape),
        "model_frame": list(df.shape),
    },
    "date_range": {
        "min": str(date_min.date()) if pd.notnull(date_min) else None,
        "max": str(date_max.date()) if pd.notnull(date_max) else None,
    },
    "severe_rate": float(df["SEVERITY_BINARY"].mean()),
    "models": summary.to_dict(orient="records"),
    "top_risk_intersections": risk_idx.head(5).to_dict(orient="records"),
}
with open(OUTPUT_DIR + "update1_summary.json", "w") as f:
    json.dump(summary_dict, f, indent=2)

print("\n✅  Update 1 analysis complete.")
print("    Outputs saved in:", OUTPUT_DIR)
