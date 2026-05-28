#!/usr/bin/env python3
"""Export required Chapter 4 output images into one folder."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "required_images"
OUT.mkdir(parents=True, exist_ok=True)


def _save(name: str) -> None:
    p = OUT / name
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved: {p.relative_to(ROOT)}")


def _load_training():
    from src import features as feat_mod

    csv_path = ROOT / "data" / "generated" / "biogas_training_merged.csv"
    df = pd.read_csv(csv_path)
    if "OLR" not in df.columns and "OLR_used" in df.columns:
        df["OLR"] = df["OLR_used"]
    df = df[(df["converged"] == True) & (df["stability_label"] >= 0)].copy()  # noqa: E712

    X_df = feat_mod.dataframe_features(df)
    X = X_df.values
    y_yield = df["q_ch4_avg"].astype(float).values
    y_health = df["stability_label"].astype(int).values
    return df, X_df, X, y_yield, y_health


def export_model_comparison():
    p = ROOT / "models" / "model_comparison.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    models = ["xgboost", "lightgbm", "random_forest", "stacking"]
    names = ["XGBoost", "LightGBM", "Random Forest", "Stacking"]
    r2 = [data["regressor"][m]["cv_r2_mean"] for m in models]
    f1 = [data["classifier"][m]["cv_f1_macro_mean"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(names, r2, color="#3e95cd")
    axes[0].set_title("Yield (regression)")
    axes[0].set_ylabel("CV R²")
    axes[0].tick_params(axis="x", rotation=18)
    for i, v in enumerate(r2):
        axes[0].text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(names, f1, color="#e67e22")
    axes[1].set_title("Health (classification)")
    axes[1].set_ylabel("CV F1 (macro)")
    axes[1].tick_params(axis="x", rotation=18)
    for i, v in enumerate(f1):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("Model comparison: 5-fold cross-validation", y=1.02)
    plt.tight_layout()
    _save("fig_4_2_model_comparison.png")
    plt.close(fig)


def export_yield_and_learning(df: pd.DataFrame, X: np.ndarray, y_yield: np.ndarray):
    reg = XGBRegressor(
        n_estimators=400,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    X_train, X_test, y_tr, y_te = train_test_split(X, y_yield, test_size=0.15, random_state=42)
    reg.fit(X_train, y_tr)
    pred = reg.predict(X_test)
    r2 = r2_score(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y_te, pred, s=16, alpha=0.35, color="#2d6a4f", edgecolors="none")
    lo = min(float(y_te.min()), float(pred.min())) - 5
    hi = max(float(y_te.max()), float(pred.max())) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual CH₄ yield (m³/d)")
    ax.set_ylabel("Predicted CH₄ yield (m³/d)")
    ax.set_title("Yield model: predicted vs actual")
    ax.text(0.05, 0.92, f"R²={r2:.4f}\nRMSE={rmse:.2f}", transform=ax.transAxes, fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    _save("fig_4_3_yield_scatter.png")
    plt.close(fig)

    em = json.loads((ROOT / "models" / "eval_metrics.json").read_text(encoding="utf-8"))
    lc = em.get("learning_curve") or []
    if lc:
        fracs = [it["fraction"] for it in lc]
        vals = [it["r2"] for it in lc]
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.plot(fracs, vals, "o-", color="#2d6a4f", lw=2)
        ax.set_xlabel("Training fraction")
        ax.set_ylabel("Hold-out R²")
        ax.set_title("Learning curve (yield regressor)")
        ax.set_ylim(0.98, 1.002)
        plt.tight_layout()
        _save("fig_4_3_learning_curve.png")
        plt.close(fig)


def export_health(df: pd.DataFrame, X: np.ndarray, y_health: np.ndarray):
    # Class distribution
    counts = pd.Series(y_health).value_counts().sort_index()
    label_map = {0: "Stable", 1: "Warning", 2: "Critical"}
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [label_map.get(int(k), str(k)) for k in counts.index]
    ax.bar(x, counts.values, color=["#2ecc71", "#f39c12", "#e74c3c"][: len(x)])
    ax.set_title("Stability class distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _save("fig_4_4_class_distribution.png")
    plt.close(fig)

    # Confusion matrix (hold-out)
    X_train, X_test, h_tr, h_te = train_test_split(X, y_health, test_size=0.15, random_state=42)
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, h_tr)
    hp = np.asarray(clf.predict(X_test))
    if hp.ndim == 2:
        hp = np.argmax(hp, axis=1)
    hp = hp.astype(int).ravel()
    h_te = np.asarray(h_te).astype(int).ravel()
    labels = sorted(np.unique(np.concatenate([h_te, hp])))
    names = [label_map.get(int(l), f"C{l}") for l in labels]
    f1 = f1_score(h_te, hp, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(h_te, hp, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=names, yticklabels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (macro F1={f1:.3f})")
    plt.tight_layout()
    _save("fig_4_4_confusion_matrix.png")
    plt.close(fig)


def export_shap(X_df: pd.DataFrame):
    try:
        from src import features as feat_mod
        from src.shap_viz import plot_beeswarm, plot_waterfall
    except Exception as exc:  # pragma: no cover
        print("skip SHAP export:", exc)
        return

    reg = joblib.load(ROOT / "models" / "yield_predictor.pkl")
    bg = pd.DataFrame(X_df.values[:200], columns=feat_mod.FEATURE_COLUMNS)
    fig = plot_beeswarm(reg, bg)
    fig.suptitle("SHAP beeswarm (yield model)", y=0.99, fontsize=11)
    plt.tight_layout()
    _save("fig_4_5_shap_beeswarm.png")
    plt.close(fig)

    rep_row = {
        "cow_frac": 0.45,
        "rice_frac": 0.08,
        "wheat_frac": 0.05,
        "food_frac": 0.22,
        "press_frac": 0.12,
        "poultry_frac": 0.08,
        "temperature_C": 35.0,
        "HRT_days": 25.0,
        "OLR": 2.5,
    }
    x = feat_mod.feature_matrix(rep_row)
    fig = plot_waterfall(reg, x, feature_names=list(feat_mod.FEATURE_COLUMNS))
    fig.suptitle("SHAP waterfall (representative input)", y=0.99, fontsize=10)
    plt.tight_layout()
    _save("fig_4_5_shap_waterfall.png")
    plt.close(fig)


def export_literature():
    lit = json.loads((ROOT / "models" / "literature_validation.json").read_text(encoding="utf-8"))
    pts = pd.DataFrame(lit["per_point"])
    inr = pts[pts["in_training_range"]]
    if inr.empty:
        return

    # before correction
    fig, ax = plt.subplots(figsize=(6.3, 6.3))
    ax.scatter(inr["observed"], inr["predicted"], s=55, color="#3498db", edgecolors="black", linewidths=0.3)
    lo = 0.0
    hi = max(float(inr["observed"].max()), float(inr["predicted"].max())) * 1.12
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed specific yield")
    ax.set_ylabel("Predicted (uncorrected)")
    ax.set_title("Literature parity before correction")
    ax.set_aspect("equal")
    plt.tight_layout()
    _save("fig_4_6_literature_before_correction.png")
    plt.close(fig)

    # after correction
    fig, ax = plt.subplots(figsize=(6.3, 6.3))
    ax.scatter(inr["observed"], inr["predicted_corrected"], s=55, color="#2d6a4f", edgecolors="black", linewidths=0.3)
    ex = pts[~pts["in_training_range"]]
    if not ex.empty:
        ax.scatter(ex["observed"], ex["predicted_corrected"], s=45, marker="x", color="#8d99ae", label="Out-of-range")
        ax.legend(loc="upper left", fontsize=8)
    hi = max(float(inr["observed"].max()), float(inr["predicted_corrected"].max())) * 1.12
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed specific yield")
    ax.set_ylabel("Predicted (bias-corrected)")
    ax.set_title("Literature parity after correction")
    ax.set_aspect("equal")
    plt.tight_layout()
    _save("fig_4_6_literature_after_correction.png")
    plt.close(fig)


def export_adm1():
    from src import simulator

    mix = np.array([0.5, 0.1, 0.05, 0.15, 0.1, 0.1], dtype=float)
    mix /= mix.sum()
    try:
        out = simulator.run_from_hrt_mix(mix, temp_C=35.0, HRT=25.0, sim_days=30)
    except Exception as exc:
        print("skip ADM1 export:", exc)
        return
    ts = out.get("time_series")
    if ts is None or len(ts) == 0:
        return
    t = ts["time_d"]
    plots = [
        ("ph", ts["pH"], "pH", "#e74c3c"),
        ("vfa", ts["VFA"], "VFA (kg HAc-eq/m³)", "#e67e22"),
        ("ch4", ts["q_ch4"], "CH₄ flow (m³/d)", "#2d6a4f"),
    ]
    for key, y, ylab, color in plots:
        fig, ax = plt.subplots(figsize=(8.6, 3.9))
        ax.plot(t, y, color=color, lw=1.6)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(ylab)
        ax.set_title(f"ADM1 time series: {ylab}")
        plt.tight_layout()
        _save(f"fig_4_7_adm1_{key}.png")
        plt.close(fig)


def write_readme():
    txt = """Required output images generated for report/presentation.

4.2
- fig_4_2_model_comparison.png

4.3
- fig_4_3_yield_scatter.png
- fig_4_3_learning_curve.png

4.4
- fig_4_4_confusion_matrix.png
- fig_4_4_class_distribution.png

4.5
- fig_4_5_shap_beeswarm.png
- fig_4_5_shap_waterfall.png

4.6
- fig_4_6_literature_before_correction.png
- fig_4_6_literature_after_correction.png

4.7
- fig_4_7_adm1_ph.png
- fig_4_7_adm1_vfa.png
- fig_4_7_adm1_ch4.png

4.8 app page screenshots should be captured manually from Streamlit.
"""
    (OUT / "README.txt").write_text(txt, encoding="utf-8")


def main():
    sns.set_theme(style="whitegrid")
    print("output folder:", OUT)
    df, X_df, X, y_yield, y_health = _load_training()
    export_model_comparison()
    export_yield_and_learning(df, X, y_yield)
    export_health(df, X, y_health)
    export_shap(X_df)
    export_literature()
    export_adm1()
    write_readme()
    print("done.")


if __name__ == "__main__":
    main()

