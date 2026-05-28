#!/usr/bin/env python3
"""Build notebooks/02_biooptima_results.ipynb from the focused presentation plan."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_biooptima_results.ipynb"


def md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


COLAB_SETUP = r'''# Colab only — skip when running locally in Jupyter
import os
import sys

try:
    from google.colab import drive
except ImportError:
    print("Not Colab — skip this cell.")
else:
    drive.mount("/content/drive")
    MY_DRIVE = "/content/drive/MyDrive"
    PROJECT_ROOT = ""  # e.g. "/content/drive/MyDrive/major_project/webapp"

    def _is_root(p):
        return os.path.isdir(p) and os.path.isdir(os.path.join(p, "src"))

    if not PROJECT_ROOT or not _is_root(PROJECT_ROOT):
        from collections import deque

        found, q = None, deque([(MY_DRIVE, 0)])
        while q and not found:
            p, d = q.popleft()
            if d > 5:
                continue
            if _is_root(p):
                found = p
                break
            if os.path.isdir(p):
                for name in os.listdir(p):
                    c = os.path.join(p, name)
                    if not name.startswith(".") and os.path.isdir(c):
                        q.append((c, d + 1))
        if not found:
            raise FileNotFoundError(
                "Set PROJECT_ROOT to your Drive folder containing src/"
            )
        PROJECT_ROOT = found
        print("Auto-detected:", PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    print("OK — Drive project:", os.getcwd())
'''

COLAB_LOCAL = r'''# Copy project to Colab local disk (fixes Errno 107 on import shap)
import os
import shutil
import sys

try:
    import google.colab  # noqa: F401
except ImportError:
    print("Not Colab — skip.")
else:
    DRIVE_ROOT = os.getcwd()
    LOCAL = "/content/biooptima"
    if os.path.abspath(DRIVE_ROOT) != os.path.abspath(LOCAL):
        if os.path.isdir(LOCAL):
            shutil.rmtree(LOCAL, ignore_errors=True)
        print("Copying to /content/biooptima (1–3 min)...")
        shutil.copytree(
            DRIVE_ROOT, LOCAL, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "node_modules", ".ipynb_checkpoints"),
        )
        os.chdir(LOCAL)
        sys.path.insert(0, LOCAL)
    print("Working directory:", os.getcwd())
'''

PIP_INSTALL = r'''# Install packages on Colab (harmless skip locally if already installed)
try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    import subprocess
    import sys

    pkgs = [
        "xgboost>=2.0,<4",
        "shap>=0.43,<1",
        "imbalanced-learn>=0.12,<0.14",
        "seaborn>=0.13,<1",
        "scikit-learn>=1.3,<1.6",
        "joblib>=1.3,<2",
        "PyYAML>=6,<7",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
    )
    print("Installed:", ", ".join(pkgs))
    print("If imports fail below, use Runtime → Restart session, then Run all from the top.")
else:
    print("Not Colab — using your local environment (pip install skipped).")
'''

IMPORTS = r'''import json, sys, time, warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    mean_squared_error, r2_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path.cwd()
if not (ROOT / "src").is_dir() and (ROOT.parent / "src").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as feat_mod
from src import health_utils, simulator

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", palette="muted")
%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True})

SEED = 42
V_LIQ = 100.0
CSV_PATH = ROOT / "data" / "generated" / "biogas_training_merged.csv"

READABLE = {
    "cow_frac": "Cow dung fraction", "rice_frac": "Rice straw fraction",
    "wheat_frac": "Wheat straw fraction", "food_frac": "Food waste fraction",
    "press_frac": "Press mud fraction", "poultry_frac": "Poultry litter fraction",
    "temperature_C": "Temperature (C)", "HRT_days": "HRT (days)",
    "OLR": "Organic loading rate", "CN_ratio": "Blended C/N ratio",
    "VS_total": "Total VS loaded (OLR x HRT)", "temp_HRT": "Temperature x HRT",
    "lipid_frac_mix": "Blended lipid fraction", "protein_frac_mix": "Blended protein fraction",
    "COD_proxy": "COD proxy (weighted)", "OLR_x_food_frac": "OLR x food fraction",
    "protein_to_CN": "Protein / C:N ratio", "OLR_per_HRT": "OLR / HRT",
    "lipid_x_temp": "Lipid fraction x Temperature",
}

print("Project root:", ROOT)
print("Training CSV:", CSV_PATH.name)
'''

PREFLIGHT = r'''# Required files (fail fast with a clear message)
_required = [
    ("Training CSV", CSV_PATH),
    ("Literature points", ROOT / "data" / "validation" / "literature_points.csv"),
    ("Substrate YAMLs", ROOT / "data" / "substrates"),
]
_optional = [
    ("Model comparison (Section 4)", ROOT / "models" / "model_comparison.json"),
]

missing = [name for name, p in _required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required paths: " + ", ".join(missing)
        + "\nUpload the full webapp folder to Drive (src/, data/, models/)."
    )

for name, p in _optional:
    print(("OK" if p.exists() else "SKIP (optional)"), name)

if not (ROOT / "data" / "substrates").is_dir():
    raise FileNotFoundError("data/substrates/ folder required for feature engineering.")
'''

LOAD_CSV = r'''df = pd.read_csv(CSV_PATH)
if "OLR" not in df.columns and "OLR_used" in df.columns:
    df["OLR"] = df["OLR_used"]
df = df[df["converged"] == True].copy()
df = df[df["stability_label"] >= 0].copy()
if len(df) < 100:
    raise ValueError(f"Too few rows after filtering ({len(df)}). Check CSV path and converged column.")

print(f"Dataset: {len(df)} converged rows, {df.shape[1]} columns")
print(f"Temperature range: {df['temperature_C'].min():.1f} - {df['temperature_C'].max():.1f} C")
print(f"HRT range: {df['HRT_days'].min():.1f} - {df['HRT_days'].max():.1f} d")
print(f"OLR range: {df['OLR'].min():.2f} - {df['OLR'].max():.2f} kg VS/m3/d")
df.head()
'''

HISTOGRAMS = r'''fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label, color in zip(
    axes,
    ["temperature_C", "HRT_days", "OLR"],
    ["Temperature (C)", "HRT (days)", "OLR (kg VS/m3/d)"],
    ["#e74c3c", "#2ecc71", "#3498db"],
):
    ax.hist(df[col], bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(label)
plt.suptitle(f"Training data coverage ({len(df)} ADM1 runs)", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
'''

STABILITY = r'''counts = df["stability_label"].value_counts().sort_index()
labels_map = {0: "Stable", 1: "Warning", 2: "Critical"}
colors = ["#2ecc71", "#e67e22", "#e74c3c"]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    [labels_map.get(k, f"Class {k}") for k in counts.index],
    counts.values,
    color=[colors[min(int(k), len(colors) - 1)] for k in counts.index],
    edgecolor="white",
)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            str(v), ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")
ax.set_title("Stability class distribution (significant imbalance)")
plt.tight_layout()
plt.show()
if len(counts) >= 2:
    print(f"Ratio (majority:minority): {counts.iloc[0]}:{counts.iloc[1]} = {counts.iloc[0]/max(counts.iloc[1],1):.0f}:1")
'''

FEATURES = r'''X_df = feat_mod.dataframe_features(df)
print(f"Feature matrix: {X_df.shape[0]} rows x {X_df.shape[1]} features\n")
for i, col in enumerate(feat_mod.FEATURE_COLUMNS, 1):
    print(f"  {i:2d}. {col:22s}  --  {READABLE.get(col, col)}")
'''

CORR = r'''corr_df = X_df.copy()
corr_df["CH4_yield"] = df["q_ch4_avg"].values
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True, linewidths=0.5,
            annot_kws={"size": 7})
ax.set_title("Feature correlation matrix", fontsize=13)
plt.tight_layout()
plt.show()
'''

SCATTERS = r'''fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(df["OLR"], df["q_ch4_avg"], alpha=0.15, s=8, color="#3498db")
axes[0].set_xlabel("OLR (kg VS/m3/d)")
axes[0].set_ylabel("CH4 yield (m3/d)")
axes[0].set_title("OLR vs CH4 yield")

axes[1].scatter(df["temperature_C"], df["q_ch4_avg"], alpha=0.15, s=8, color="#e74c3c")
axes[1].set_xlabel("Temperature (C)")
axes[1].set_ylabel("CH4 yield (m3/d)")
axes[1].set_title("Temperature vs CH4 yield")

axes[2].scatter(X_df["food_frac"], df["q_ch4_avg"], alpha=0.15, s=8, color="#2ecc71")
axes[2].set_xlabel("Food waste fraction")
axes[2].set_ylabel("CH4 yield (m3/d)")
axes[2].set_title("Food waste fraction vs CH4 yield")

plt.suptitle("Physical sanity: training data follows expected AD trends", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
'''

MODEL_COMP = r'''comp_path = ROOT / "models" / "model_comparison.json"
models = ["xgboost", "lightgbm", "random_forest", "stacking"]
nice_names = ["XGBoost", "LightGBM", "Random Forest", "Stacking"]

if comp_path.is_file():
    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    r2_vals = [comp["regressor"][m]["cv_r2_mean"] for m in models]
    f1_vals = [comp["classifier"][m]["cv_f1_macro_mean"] for m in models]
    best_reg = max(models, key=lambda m: comp["regressor"][m]["cv_r2_mean"])
    best_clf = max(models, key=lambda m: comp["classifier"][m]["cv_f1_macro_mean"])
    cv_source = "model_comparison.json (from scripts/train_models.py on same CSV)"
else:
    print("Note: model_comparison.json not found — showing XGBoost hold-out metrics only.")
    print("Run: python scripts/train_models.py --csv data/generated/biogas_training_merged.csv")
    comp = None
    r2_vals = [np.nan] * len(models)
    f1_vals = [np.nan] * len(models)
    best_reg = best_clf = "xgboost"
    cv_source = "n/a"

if comp is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(models))

    axes[0].bar(x, r2_vals, color="#3498db", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(nice_names, rotation=15, ha="right")
    axes[0].set_ylabel("CV R2")
    axes[0].set_title("Yield (regression)")
    axes[0].set_ylim(0.85, 1.01)
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, f1_vals, color="#e67e22", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nice_names, rotation=15, ha="right")
    axes[1].set_ylabel("CV F1 (macro)")
    axes[1].set_title("Health (classification)")
    axes[1].set_ylim(0.7, 1.05)
    for i, v in enumerate(f1_vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Model comparison: 5-fold cross-validation", fontsize=13)
    plt.tight_layout()
    plt.show()

    print(f"Best regressor by CV R2: {best_reg} ({comp['regressor'][best_reg]['cv_r2_mean']:.4f})")
    print(f"Best classifier by CV F1: {best_clf} ({comp['classifier'][best_clf]['cv_f1_macro_mean']:.4f})")
    print("Source:", cv_source)
else:
    print("Section 4 chart skipped — upload models/model_comparison.json for the 4-model CV plot.")
'''

TRAIN_YIELD = r'''X = X_df.values
y_yield = df["q_ch4_avg"].astype(float).values
y_health = df["stability_label"].astype(int).values

try:
    X_train, X_test, y_tr, y_te, h_tr, h_te = train_test_split(
        X, y_yield, y_health, test_size=0.15, random_state=SEED, stratify=y_health)
except ValueError:
    X_train, X_test, y_tr, y_te, h_tr, h_te = train_test_split(
        X, y_yield, y_health, test_size=0.15, random_state=SEED)

reg = XGBRegressor(
    n_estimators=400, learning_rate=0.06, max_depth=6,
    subsample=0.85, colsample_bytree=0.85, random_state=SEED, n_jobs=-1)
reg.fit(X_train, y_tr)
pred = reg.predict(X_test)

yield_r2 = r2_score(y_te, pred)
yield_rmse = np.sqrt(mean_squared_error(y_te, pred))
print(f"Yield model hold-out R2  = {yield_r2:.4f}")
print(f"Yield model hold-out RMSE = {yield_rmse:.4f} m3/d")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
'''

SCATTER_PRED = r'''fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_te, pred, alpha=0.4, s=15, color="#3498db", edgecolors="none")
lims = [min(y_te.min(), pred.min()) - 5, max(y_te.max(), pred.max()) + 5]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual CH4 yield (m3/d)", fontsize=12)
ax.set_ylabel("Predicted CH4 yield (m3/d)", fontsize=12)
ax.set_title("Yield model: predicted vs actual (hold-out)", fontsize=13)
ax.text(0.05, 0.92, f"R2 = {yield_r2:.4f}\nRMSE = {yield_rmse:.2f} m3/d",
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.legend(loc="lower right")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
'''

TRAIN_HEALTH = r'''from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

clf_base = XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.9, colsample_bytree=0.85, random_state=SEED, n_jobs=-1,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss")

kn = min(3, int(np.bincount(h_tr).min()) - 1) if len(np.unique(h_tr)) > 1 else 0
if kn >= 1:
    clf_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=SEED, k_neighbors=kn)),
        ("clf", clf_base)])
    clf_pipe.fit(X_train, h_tr)
    hp = clf_pipe.predict(X_test)
    if getattr(hp, "ndim", 1) == 2:
        hp = np.argmax(hp, axis=1)
    hp = np.asarray(hp).astype(int)
    clf_final = clf_pipe
else:
    clf_base.fit(X_train, h_tr)
    hp = clf_base.predict(X_test)
    if getattr(hp, "ndim", 1) == 2:
        hp = np.argmax(hp, axis=1)
    hp = np.asarray(hp).astype(int)
    clf_final = clf_base

all_labels = sorted(np.unique(y_health))
label_names = {0: "Stable", 1: "Warning", 2: "Critical"}
target_names = [label_names.get(l, f"Class {l}") for l in all_labels]
health_f1 = f1_score(h_te, hp, average="macro", labels=all_labels, zero_division=0)
health_acc = accuracy_score(h_te, hp)
print(classification_report(h_te, hp, digits=3, labels=all_labels,
                            target_names=target_names, zero_division=0))
print(f"Hold-out F1 (macro) = {health_f1:.4f}")
print(f"Hold-out accuracy   = {health_acc:.4f}")
'''

CONFUSION = r'''cm = confusion_matrix(h_te, hp, labels=all_labels)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=target_names, yticklabels=target_names)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Health classifier: confusion matrix (hold-out)", fontsize=13)
plt.tight_layout()
plt.show()
'''

SAVE_MODELS = r'''mdir = ROOT / "models"
mdir.mkdir(parents=True, exist_ok=True)

joblib.dump(reg, mdir / "yield_predictor.pkl")
joblib.dump(clf_final, mdir / "health_classifier.pkl")

bg = pd.DataFrame(X_train, columns=feat_mod.FEATURE_COLUMNS).sample(
    n=min(100, len(X_train)), random_state=SEED)
joblib.dump(bg, mdir / "X_background.pkl")

if hasattr(reg, "get_booster"):
    reg.get_booster().save_model(str(mdir / "yield_predictor.json"))
if hasattr(clf_base, "get_booster"):
    clf_base.get_booster().save_model(str(mdir / "health_classifier.json"))

# eval_metrics.json — always reflects THIS notebook run (used by app + literature)
_training_ranges = {
    "temperature_C": [float(df["temperature_C"].min()), float(df["temperature_C"].max())],
    "HRT_days": [float(df["HRT_days"].min()), float(df["HRT_days"].max())],
    "OLR": [float(df["OLR"].min()), float(df["OLR"].max())],
}
eval_metrics = {
    "yield_r2": float(yield_r2),
    "yield_rmse": float(yield_rmse),
    "health_f1_macro": float(health_f1),
    "health_accuracy": float(health_acc),
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(X.shape[1]),
    "feature_names": list(feat_mod.FEATURE_COLUMNS),
    "best_regressor": "xgboost",
    "best_classifier": "xgboost",
    "training_ranges": _training_ranges,
    "stability_distribution": df["stability_label"].value_counts().sort_index().to_dict(),
    "notebook_run": True,
}
_comp_path = ROOT / "models" / "model_comparison.json"
if _comp_path.is_file():
    _comp_save = json.loads(_comp_path.read_text(encoding="utf-8"))
    eval_metrics["yield_cv_r2_mean"] = float(_comp_save["regressor"]["xgboost"]["cv_r2_mean"])
    eval_metrics["health_cv_f1_macro_mean"] = float(_comp_save["classifier"]["xgboost"]["cv_f1_macro_mean"])
(mdir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

print("Saved to models/:")
for f in sorted(mdir.glob("*")):
    if f.is_file():
        print(f"  {f.name:35s}  {f.stat().st_size / 1024:.0f} KB")
'''

SHAP = r'''import shap

explainer = shap.TreeExplainer(reg)
bg_sample = pd.DataFrame(X_test[:200], columns=feat_mod.FEATURE_COLUMNS)
shap_values = explainer(bg_sample)

readable = [READABLE.get(c, c) for c in feat_mod.FEATURE_COLUMNS]
shap_values.feature_names = readable

plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("SHAP feature importance (yield model)", fontsize=13)
plt.tight_layout()
plt.show()
'''

LITERATURE = r'''lit_df = pd.read_csv(ROOT / "data" / "validation" / "literature_points.csv").copy()

if "reg" not in globals():
    raise NameError("Run the model training cells first (reg is not defined).")

em_path = ROOT / "models" / "eval_metrics.json"
if em_path.is_file():
    em = json.loads(em_path.read_text(encoding="utf-8"))
    tr = em.get("training_ranges")
    if tr:
        ranges = {k: (float(v[0]), float(v[1])) for k, v in tr.items()}
    else:
        ranges = {
            "temperature_C": (float(df["temperature_C"].min()), float(df["temperature_C"].max())),
            "HRT_days": (float(df["HRT_days"].min()), float(df["HRT_days"].max())),
            "OLR": (float(df["OLR"].min()), float(df["OLR"].max())),
        }
else:
    ranges = {
        "temperature_C": (float(df["temperature_C"].min()), float(df["temperature_C"].max())),
        "HRT_days": (float(df["HRT_days"].min()), float(df["HRT_days"].max())),
        "OLR": (float(df["OLR"].min()), float(df["OLR"].max())),
    }
print("Literature in-range uses training ranges:", ranges)


def _in_range(val, bounds):
    return bounds[0] <= val <= bounds[1]


preds_raw, in_range = [], []
for _, row in lit_df.iterrows():
    temp = float(row["temperature_C"])
    hrt = float(row["HRT_days"])
    olr = float(row["OLR"])
    feat_row = {
        "cow_frac": float(row["cow_frac"]),
        "rice_frac": float(row["rice_frac"]),
        "wheat_frac": float(row["wheat_frac"]),
        "food_frac": float(row["food_frac"]),
        "press_frac": float(row["press_frac"]),
        "poultry_frac": float(row["poultry_frac"]),
        "temperature_C": temp,
        "HRT_days": hrt,
        "OLR": olr,
    }
    X_lit = feat_mod.feature_matrix(feat_row)
    yhat = float(reg.predict(X_lit)[0]) / V_LIQ
    preds_raw.append(yhat)
    ok = (
        _in_range(temp, ranges["temperature_C"])
        and _in_range(hrt, ranges["HRT_days"])
        and _in_range(olr, ranges["OLR"])
    )
    in_range.append(ok)

lit_df["predicted"] = preds_raw
lit_df["in_range"] = in_range
obs = lit_df["observed_specific_yield"].values
pred_lit = np.array(preds_raw, dtype=float)

ir = lit_df[lit_df["in_range"]].copy()
ir_mask = lit_df["in_range"].astype(bool)

a_corr, b_corr = 1.0, 0.0
if len(ir) >= 3:
    A = np.column_stack([ir["predicted"].values, np.ones(len(ir))])
    coef, _, _, _ = np.linalg.lstsq(A, ir["observed_specific_yield"].values, rcond=None)
    if len(coef) >= 2:
        a_corr, b_corr = float(coef[0]), float(coef[1])
else:
    lv_path = ROOT / "models" / "literature_validation.json"
    if lv_path.is_file():
        lv = json.loads(lv_path.read_text(encoding="utf-8"))
        bc = lv.get("overall_in_range", {}).get("bias_correction")
        if bc:
            a_corr = float(bc.get("a", 1.0))
            b_corr = float(bc.get("b", 0.0))
    print(f"Warning: only {len(ir)} in-range points; using bias a={a_corr:.4f}, b={b_corr:.4f}")

lit_df["predicted_corrected"] = a_corr * pred_lit + b_corr

if len(ir) >= 1:
    denom = np.maximum(ir["observed_specific_yield"].values, 1e-6)
    mape_before = float(np.mean(np.abs(ir["observed_specific_yield"].values - ir["predicted"].values) / denom * 100))
    mape_after = float(
        np.mean(np.abs(ir["observed_specific_yield"].values - (a_corr * ir["predicted"].values + b_corr)) / denom * 100)
    )
else:
    mape_before = float("nan")
    mape_after = float("nan")

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    lit_df.loc[ir_mask, "observed_specific_yield"],
    lit_df.loc[ir_mask, "predicted_corrected"],
    s=60, color="#2ecc71", edgecolors="black", linewidths=0.5,
    label=f"In-range (n={ir_mask.sum()})", zorder=3,
)
ax.scatter(
    lit_df.loc[~ir_mask, "observed_specific_yield"],
    lit_df.loc[~ir_mask, "predicted_corrected"],
    s=60, color="#e74c3c", edgecolors="black", linewidths=0.5,
    marker="x", label=f"Excluded (n={(~ir_mask).sum()})", zorder=3,
)
lims = [0, max(obs.max(), pred_lit.max()) * 1.15 + 0.05]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Observed specific yield (m3 CH4 / m3 / d)", fontsize=12)
ax.set_ylabel("Predicted specific yield (corrected)", fontsize=12)
ax.set_title("Literature validation: observed vs predicted", fontsize=13)
ax.text(
    0.05, 0.92,
    f"In-range MAPE = {mape_after:.1f}% (was {mape_before:.1f}%)\n"
    f"Bias correction: y = {a_corr:.3f}x + {b_corr:.3f}",
    transform=ax.transAxes, fontsize=11, verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
ax.legend(loc="lower right")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

print(f"{'Substrate':<45s}  {'Obs':>6s}  {'Pred':>6s}  {'Corr':>6s}  {'Err%':>5s}  {'Range':>5s}")
print("-" * 90)
for _, row in lit_df.iterrows():
    flag = "OK" if row["in_range"] else "OUT"
    err = abs(row["observed_specific_yield"] - row["predicted"]) / max(row["observed_specific_yield"], 1e-6) * 100
    print(
        f"{row['substrate_desc'][:44]:<45s}  {row['observed_specific_yield']:6.3f}  "
        f"{row['predicted']:6.3f}  {row['predicted_corrected']:6.3f}  {err:5.1f}  {flag:>5s}"
    )
print("-" * 90)
print(f"\nIn-range: {ir_mask.sum()}/{len(lit_df)} points")
print(f"MAPE before correction: {mape_before:.1f}%")
print(f"MAPE after correction:  {mape_after:.1f}%")
print(f"Bias correction: a={a_corr:.4f}, b={b_corr:.4f}")

lit_result = {
    "n_points_total": int(len(lit_df)),
    "n_points_in_range": int(ir_mask.sum()),
    "n_points_excluded": int((~ir_mask).sum()),
    "training_ranges": {k: list(v) for k, v in ranges.items()},
    "overall_in_range": {
        "n": int(ir_mask.sum()),
        "mape_pct": round(mape_before, 2) if np.isfinite(mape_before) else None,
        "mape_pct_corrected": round(mape_after, 2) if np.isfinite(mape_after) else None,
        "bias_correction": {"a": round(a_corr, 4), "b": round(b_corr, 4)},
    },
    "per_point": [
        {
            "substrate": row["substrate_desc"],
            "observed": round(float(row["observed_specific_yield"]), 4),
            "predicted": round(float(row["predicted"]), 4),
            "predicted_corrected": round(float(row["predicted_corrected"]), 4),
            "in_training_range": bool(row["in_range"]),
        }
        for _, row in lit_df.iterrows()
    ],
    "unit": "m3_CH4_per_m3_reactor_per_day",
    "V_liq_model": V_LIQ,
}
(ROOT / "models" / "literature_validation.json").write_text(
    json.dumps(lit_result, indent=2), encoding="utf-8"
)
print("Saved models/literature_validation.json")
'''

LIT_TABLE = ""

ADM1 = r'''MIX = np.array([0.5, 0.1, 0.05, 0.15, 0.1, 0.1], dtype=float)
MIX /= MIX.sum()

try:
    t0 = time.perf_counter()
    result = simulator.run_from_hrt_mix(MIX, temp_C=35.0, HRT=25.0, sim_days=30)
    elapsed = time.perf_counter() - t0
except Exception as exc:
    print("ADM1 demo skipped:", exc)
    print("Ensure pyADM1ODE/ is on Drive. ML sections above are still valid.")
    result = None

if result is not None:
    print(f"Simulation time: {elapsed:.1f} s")
    print(f"Converged: {result['converged']}")
    print(f"Achieved OLR: {result.get('OLR', result.get('OLR_used', 0)):.2f} kg VS/m3/d")
    print(f"CH4 (last-week avg): {result['q_ch4_avg']:.2f} m3/d")
    print(f"Final pH: {result['pH_final']:.2f}")
    print(f"Final VFA: {result['VFA_final']:.2f} kg/m3")
    stability = health_utils.classify_stability(result["pH_final"], result["VFA_final"])
    print(f"Stability: {health_utils.stability_name(stability)}")

ts = result.get("time_series") if result else None
if ts is not None and len(ts) > 0:
    t = ts["time_d"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(t, ts["pH"], color="#e74c3c")
    axes[0, 0].set_ylabel("pH")
    axes[0, 0].set_title("pH over time")
    axes[0, 0].axhline(6.8, ls="--", color="orange", alpha=0.5, label="Warning threshold")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, ts["VFA"], color="#e67e22")
    axes[0, 1].set_ylabel("VFA (kg/m3)")
    axes[0, 1].set_title("Volatile fatty acids")

    axes[1, 0].plot(t, ts["q_ch4"], color="#2ecc71")
    axes[1, 0].set_ylabel("CH4 (m3/d)")
    axes[1, 0].set_title("Methane production")
    axes[1, 0].set_xlabel("Time (days)")

    axes[1, 1].plot(t, ts["S_nh3"], color="#9b59b6")
    axes[1, 1].set_ylabel("NH3 (kg/m3)")
    axes[1, 1].set_title("Free ammonia")
    axes[1, 1].set_xlabel("Time (days)")

    plt.suptitle("ADM1 simulation: cow-dung-heavy mix, 35 C, HRT 25 d", fontsize=13)
    plt.tight_layout()
    plt.show()
else:
    print("(Time series not available — check PyADM1ODE installation)")
'''

SUMMARY_DYNAMIC = r'''from IPython.display import Markdown, display

# Hold-out = this notebook run. CV = from model_comparison.json if present.
_cv_path = ROOT / "models" / "model_comparison.json"
if _cv_path.is_file():
    _comp = json.loads(_cv_path.read_text(encoding="utf-8"))
    cv_yield_r2 = float(_comp["regressor"]["xgboost"]["cv_r2_mean"])
    cv_health_f1 = float(_comp["classifier"]["xgboost"]["cv_f1_macro_mean"])
    cv_note = "5-fold CV from model_comparison.json"
else:
    cv_yield_r2 = cv_health_f1 = float("nan")
    cv_note = "CV not available (add model_comparison.json)"

_n_lit = len(lit_df)
_n_in = int(lit_df["in_range"].sum())
_n_ex = _n_lit - _n_in

summary_rows = [
    ("Training dataset", f"{len(df):,} converged ADM1 rows"),
    ("Features", f"{X_df.shape[1]} engineered inputs"),
    ("Model trained here", "XGBoost (yield + health)"),
    ("Yield hold-out R2 (this run)", f"{yield_r2:.4f}"),
    ("Yield hold-out RMSE (this run)", f"{yield_rmse:.4f} m3/d"),
    ("Yield 5-fold CV R2", f"{cv_yield_r2:.4f}" if cv_yield_r2 == cv_yield_r2 else "n/a"),
    ("Health hold-out F1 macro (this run)", f"{health_f1:.4f}"),
    ("Health hold-out accuracy (this run)", f"{health_acc:.4f}"),
    ("Health 5-fold CV F1 macro", f"{cv_health_f1:.4f}" if cv_health_f1 == cv_health_f1 else "n/a"),
    ("Literature points", f"{_n_lit} total, {_n_in} in-range, {_n_ex} excluded"),
    ("Literature MAPE in-range (uncorrected)", f"{mape_before:.1f}%"),
    ("Literature MAPE in-range (corrected)", f"{mape_after:.1f}%"),
    ("Bias correction (in-range fit)", f"y = {a_corr:.4f} * pred + {b_corr:.4f}"),
    ("Training T / HRT / OLR", (
        f"{ranges['temperature_C'][0]:.1f}-{ranges['temperature_C'][1]:.1f} C, "
        f"{ranges['HRT_days'][0]:.1f}-{ranges['HRT_days'][1]:.1f} d, "
        f"{ranges['OLR'][0]:.2f}-{ranges['OLR'][1]:.2f} kg VS/m3/d"
    )),
]

md_lines = ["| Metric | Value |", "|--------|-------|"]
for label, value in summary_rows:
    md_lines.append(f"| {label} | {value} |")

display(Markdown("\n".join(md_lines)))
print(cv_note)
print("Models saved to models/ — ready for the Streamlit app.")
'''


def main() -> None:
    cells = [
        md(
            "# BioOptima — Results & Model Evaluation\n\n"
            "This notebook documents the end-to-end modeling workflow for BioOptima: "
            "**data overview → model training → evaluation → SHAP explainability → "
            "literature validation → ADM1 physics demonstration**.\n\n"
            "**Colab:** mount Drive → **copy to local** → pip → imports (in that order). "
            "If `import shap` fails with *Transport endpoint is not connected*, re-run from the top after copy-to-local. "
            "Restart runtime after pip if needed, then **Run all**.\n"
        ),
        code(COLAB_SETUP),
        code(COLAB_LOCAL),
        code(PIP_INSTALL),
        code(IMPORTS),
        code(PREFLIGHT),
        md(
            "---\n\n"
            "## 1. Training data overview\n\n"
            "The surrogate models in this project are trained on data generated by the "
            "**ADM1 physics engine (PyADM1ODE)**. This section loads the merged training "
            "dataset and examines its coverage across temperature, HRT, OLR, and stability classes.\n"
        ),
        code(LOAD_CSV),
        code(HISTOGRAMS),
        code(STABILITY),
        md(
            "---\n\n"
            "## 2. Feature engineering\n\n"
            "Raw simulation inputs are transformed into engineered features that encode process "
            "knowledge, including blended substrate properties and interaction terms. "
            "This section presents the feature set and correlation structure used for model training.\n"
        ),
        code(FEATURES),
        code(CORR),
        md(
            "---\n\n"
            "## 3. Physical sanity — do the trends make sense?\n\n"
            "Before training, the dataset should reflect physically plausible anaerobic digestion behavior. "
            "These plots verify that the simulated data follows expected trends between operating conditions, "
            "feed composition, and methane production.\n"
        ),
        code(SCATTERS),
        md(
            "---\n\n"
            "## 4. Model comparison\n\n"
            "Multiple machine-learning algorithms are compared using **5-fold cross-validation** "
            "to identify a suitable regressor and classifier for deployment. The comparison shown "
            "here is loaded from the saved training pipeline outputs.\n"
        ),
        code(MODEL_COMP),
        md(
            "---\n\n"
            "## 5. Train best model (XGBoost) and evaluate on hold-out\n\n"
            "This section trains the selected model on an 85/15 train-test split and evaluates "
            "hold-out performance. The resulting regression and classification metrics are produced "
            "by the current notebook run.\n"
        ),
        code(TRAIN_YIELD),
        code(SCATTER_PRED),
        code(TRAIN_HEALTH),
        code(CONFUSION),
        code(SAVE_MODELS),
        md(
            "---\n\n"
            "## 6. SHAP explainability\n\n"
            "**SHAP (SHapley Additive exPlanations)** decomposes predictions into feature-level contributions. "
            "This section provides interpretability by showing which variables most strongly influence methane-yield predictions.\n"
        ),
        code(SHAP),
        md(
            "---\n\n"
            "## 7. Literature validation (external)\n\n"
            "To assess external credibility, model predictions are compared with published co-digestion studies. "
            "Points outside the training domain are identified and excluded from in-range error metrics, "
            "and a linear bias correction is fitted on the in-range subset.\n"
        ),
        code(LITERATURE),
        md(
            "---\n\n"
            "## 8. ADM1 physics demo\n\n"
            "The full **ADM1** model (via PyADM1ODE) is the physics foundation of the BioOptima pipeline "
            "and also powers the Live Simulator page in the web app. This section runs one baseline case "
            "to demonstrate dynamic process behavior and time-series outputs.\n"
        ),
        code(ADM1),
        md(
            "---\n\n"
            "## 9. Results summary\n\n"
            "This section consolidates key metrics generated in the cells above. "
            "All values are computed dynamically from the current notebook run and are not hardcoded.\n"
        ),
        code(SUMMARY_DYNAMIC),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
