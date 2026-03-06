"""
Generate a 1-page quad chart PDF for Update 1.
Run after update1_analysis.py so outputs/update1_summary.json exists.
"""

from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "update1_summary.json"

def wrap(text, width=60):
    return "\n".join(textwrap.wrap(text, width=width))

def format_bullets(lines, width=70):
    wrapped = []
    for line in lines:
        parts = textwrap.wrap(line, width=width)
        if not parts:
            continue
        wrapped.append(f"- {parts[0]}")
        for p in parts[1:]:
            wrapped.append(f"  {p}")
    return "\n".join(wrapped)

summary = {}
if SUMMARY_PATH.exists():
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

severe_rate = summary.get("severe_rate")
date_range = summary.get("date_range", {})
models = summary.get("models", [])
top_risk = summary.get("top_risk_intersections", [])

model_lines = []
for m in models:
    model_lines.append(f"{m['Model']}: ROC-AUC {m['ROC-AUC']}, PR-AUC {m['PR-AUC']}")

best_model = None
if models:
    best_model = sorted(models, key=lambda x: float(x["ROC-AUC"]), reverse=True)[0]

top_risk_short = []
for r in top_risk[:3]:
    top_risk_short.append(f"({r['LAT_ROUND']}, {r['LON_ROUND']})")

top_risk_lines = []
for r in top_risk:
    top_risk_lines.append(
        f"({r['LAT_ROUND']}, {r['LON_ROUND']}): Risk {r['RISK_INDEX']:.1f}"
    )

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.06, wspace=0.25, hspace=0.30)
fig.text(0.03, 0.95, "Check In 2 - Quad Chart", fontsize=22, fontweight="bold", ha="left", va="top")

left_color = "#B22222"   # firebrick
right_color = "#B8860B"  # dark goldenrod

accomplishments = [
    "Built baseline severity classifiers to establish a performance benchmark.",
    "Engineered temporal, roadway, and environmental features and validated key fields.",
    "Produced an EDA dashboard to summarize patterns in time, conditions, and severity.",
    "Mapped severe‑crash hot spots using DBSCAN clustering to reveal spatial concentration.",
    "Created a severity‑weighted intersection risk index to prioritize candidate locations.",
    "Generated a temporal trend preview and a baseline hit‑and‑run model.",
]

next_tasks = [
    "Tune severity models and interpret drivers with SHAP for explainability.",
    "Compute spatial autocorrelation and equity exposure metrics.",
    "Forecast severe‑crash timing windows and seasonal peaks.",
    "Model interaction effects across weather, lighting, and contributory causes.",
    "Define budget‑constrained optimization scenarios for intervention selection.",
]

activities_left = [
    "Finalize the quad chart PDF and select key figures for submission.",
    "Record the Zoom presentation with shared speaking roles.",
    "Upload the video first, then the quad chart PDF.",
    "Document dataset limitations and integration constraints across tables.",
]

activities_right = [
    "Validate models using temporal splits and cross‑validation.",
    "Refine clustering sensitivity and confirm stability of hot‑spot areas.",
    "Prepare report‑ready figures and narrative summary for Checkpoint 2.",
]

axes[0, 0].axis("off")
axes[0, 0].text(0, 1, "Latest Accomplishments", color=left_color, fontsize=13, fontweight="bold", va="top")
axes[0, 0].text(0, 0.92, format_bullets(accomplishments), color=left_color, fontsize=10, va="top")

axes[0, 1].axis("off")
axes[0, 1].text(0, 1, "Major Next Tasks", color=right_color, fontsize=13, fontweight="bold", va="top")
axes[0, 1].text(0, 0.92, format_bullets(next_tasks), color=right_color, fontsize=10, va="top")

axes[1, 0].axis("off")
axes[1, 0].text(0, 1, "Major Activities Remaining", color=right_color, fontsize=13, fontweight="bold", va="top")
axes[1, 0].text(0, 0.92, format_bullets(activities_left), color=right_color, fontsize=10, va="top")

axes[1, 1].axis("off")
axes[1, 1].text(0, 1, "Major Activities Remaining", color=left_color, fontsize=13, fontweight="bold", va="top")
axes[1, 1].text(0, 0.92, format_bullets(activities_right), color=left_color, fontsize=10, va="top")

OUTPUT_DIR.mkdir(exist_ok=True)
pdf_path = OUTPUT_DIR / "quad_chart_update1.pdf"
png_path = OUTPUT_DIR / "quad_chart_update1.png"
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
