#!/usr/bin/env python3
"""Train BioOptima yield + health models with multi-model comparison and k-fold CV.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --tune --n-trials 60
    python scripts/train_models.py --csv data/generated/biogas_training_data.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as feat_mod  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

REGRESSOR_NAMES = ("xgboost", "lightgbm", "random_forest", "stacking")
CLASSIFIER_NAMES = ("xgboost", "lightgbm", "random_forest", "stacking")


def _load_data(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "OLR" not in df.columns and "OLR_used" in df.columns:
        df["OLR"] = df["OLR_used"]
    df = df[df["converged"] == True].copy()  # noqa: E712
    df = df[df["stability_label"] >= 0].copy()
    if len(df) < 30:
        raise SystemExit(f"Need >=30 converged rows, got {len(df)}.")
    X = feat_mod.dataframe_features(df)
    y_yield = df["q_ch4_avg"].astype(float).values
    y_health = df["stability_label"].astype(int).values
    return df, X.values, y_yield, y_health


def _make_regressor(name: str, seed: int, params: Optional[Dict[str, Any]] = None):
    p = params or {}
    if name == "xgboost":
        defaults = dict(
            n_estimators=400,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
        )
        defaults.update(p)
        return XGBRegressor(**defaults)
    if name == "lightgbm":
        import lightgbm as lgb

        defaults = dict(
            n_estimators=400,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(p)
        return lgb.LGBMRegressor(**defaults)
    if name == "random_forest":
        defaults = dict(n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1)
        defaults.update(p)
        return RandomForestRegressor(**defaults)
    if name == "stacking":
        estimators = [
            ("xgb", _make_regressor("xgboost", seed, p)),
            ("lgb", _make_regressor("lightgbm", seed, {})),
            ("rf", _make_regressor("random_forest", seed, {})),
        ]
        return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1)
    raise ValueError(name)


def _make_classifier(name: str, seed: int, params: Optional[Dict[str, Any]] = None):
    p = params or {}
    common_clf = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    if name == "xgboost":
        defaults = dict(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            **common_clf,
        )
        defaults.update(p)
        return XGBClassifier(**defaults)
    if name == "lightgbm":
        import lightgbm as lgb

        defaults = dict(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            objective="multiclass",
            num_class=3,
            verbose=-1,
        )
        defaults.update(p)
        return lgb.LGBMClassifier(**defaults)
    if name == "random_forest":
        defaults = dict(n_estimators=300, max_depth=10, random_state=seed, n_jobs=-1, class_weight="balanced")
        defaults.update(p)
        return RandomForestClassifier(**defaults)
    if name == "stacking":
        estimators = [
            ("xgb", _make_classifier("xgboost", seed, p)),
            ("lgb", _make_classifier("lightgbm", seed, {})),
            ("rf", _make_classifier("random_forest", seed, {})),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=500, random_state=seed),
            n_jobs=-1,
        )
    raise ValueError(name)


def _cv_regressor(model, X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_binned = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    r2s, rmses = [], []
    for tr, te in cv.split(X, y_binned):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], pred))))
    return {"cv_r2_mean": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)), "cv_rmse_mean": float(np.mean(rmses))}


def _safe_predict(model, X: np.ndarray) -> np.ndarray:
    """Predict class labels, handling XGBoost returning probabilities when num_class > actual classes."""
    pred = model.predict(X)
    if pred.ndim == 2:
        pred = np.argmax(pred, axis=1)
    return pred.astype(int)


def _cv_classifier(model, X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> Dict[str, float]:
    n_classes = len(np.unique(y))
    min_class_count = int(np.bincount(y).min()) if n_classes > 1 else len(y)
    effective_splits = min(n_splits, min_class_count) if n_classes > 1 else 2

    if n_classes < 2 or min_class_count < 2:
        print(f"  WARNING: only {n_classes} class(es) present (min count={min_class_count}); skipping CV, returning majority-class baseline.")
        majority_acc = float(np.bincount(y).max()) / len(y)
        return {"cv_f1_macro_mean": 1.0 / n_classes, "cv_f1_macro_std": 0.0, "cv_accuracy_mean": majority_acc}

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    all_labels = sorted(np.unique(y))
    f1s, accs = [], []
    for tr, te in cv.split(X, y):
        X_tr, y_tr = X[tr], y[tr]
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            kn = min(3, int(np.bincount(y_tr).min()) - 1)
            if kn >= 1:
                pipe = ImbPipeline([("smote", SMOTE(random_state=seed, k_neighbors=kn)), ("clf", model)])
                pipe.fit(X_tr, y_tr)
                pred = _safe_predict(pipe, X[te])
            else:
                raise ValueError("k_neighbors < 1")
        except Exception:  # noqa: BLE001
            model.fit(X_tr, y_tr)
            pred = _safe_predict(model, X[te])
        f1s.append(f1_score(y[te], pred, average="macro", labels=all_labels, zero_division=0))
        accs.append(accuracy_score(y[te], pred))
    return {"cv_f1_macro_mean": float(np.mean(f1s)), "cv_f1_macro_std": float(np.std(f1s)), "cv_accuracy_mean": float(np.mean(accs))}


def _learning_curve_regressor(model, X: np.ndarray, y: np.ndarray, seed: int) -> List[Dict[str, float]]:
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    out = []
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for f in fracs:
        n_use = max(30, int(n * f))
        tr_idx = idx[:n_use]
        te_idx = idx[n_use:] if n_use < n else idx[int(n * 0.85) :]
        if len(te_idx) < 10:
            te_idx = idx[int(n * 0.85) :]
        m = _make_regressor("xgboost", seed)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        out.append({"fraction": f, "n_train": n_use, "r2": float(r2_score(y[te_idx], pred))})
    return out


def _tune_regressor(X_train, y_train, seed: int, n_trials: int = 60):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        reg = XGBRegressor(**params, random_state=seed, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_binned = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
        scores = cross_val_score(reg, X_train, y_train, cv=cv.split(X_train, y_binned), scoring="r2")
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def _tune_classifier(X_train, y_train, seed: int, n_trials: int = 60):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        clf = XGBClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/generated/biogas_training_data.csv")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true", help="Optuna HP search for XGBoost before comparison")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()
    csv_path = ROOT / args.csv
    if not csv_path.is_file():
        raise SystemExit(f"Missing training CSV: {csv_path}")

    df, X, y_yield, y_health = _load_data(csv_path)
    print(f"Loaded {len(df)} converged rows from {csv_path.name}")
    print("Stability distribution:", df["stability_label"].value_counts().sort_index().to_dict())
    if "OLR" in df.columns:
        print(f"OLR range: {df['OLR'].min():.2f} – {df['OLR'].max():.2f}")

    try:
        X_train, X_test, y_tr, y_te, h_tr, h_te = train_test_split(
            X, y_yield, y_health,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y_health,
        )
    except ValueError:
        X_train, X_test, y_tr, y_te, h_tr, h_te = train_test_split(
            X, y_yield, y_health,
            test_size=args.test_size,
            random_state=args.seed,
        )

    mdir = ROOT / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    bg = pd.DataFrame(X_train, columns=feat_mod.FEATURE_COLUMNS).sample(
        n=min(100, len(X_train)), random_state=args.seed
    )
    joblib.dump(bg, mdir / "X_background.pkl")

    best_reg_params: Dict[str, Any] = {}
    best_clf_params: Dict[str, Any] = {}
    if args.tune:
        print("Tuning XGBoost regressor...")
        best_reg_params = _tune_regressor(X_train, y_tr, args.seed, args.n_trials)
        print("Tuning XGBoost classifier...")
        best_clf_params = _tune_classifier(X_train, h_tr, args.seed, args.n_trials)

    # --- Model comparison: regressors ---
    reg_comparison: Dict[str, Any] = {}
    best_reg_name = None
    best_reg_cv = -1e9
    for name in REGRESSOR_NAMES:
        print(f"CV regressor: {name}...")
        params = best_reg_params if name in ("xgboost", "stacking") else {}
        model = _make_regressor(name, args.seed, params)
        metrics = _cv_regressor(model, X, y_yield, args.seed, args.cv_folds)
        reg_comparison[name] = metrics
        if metrics["cv_r2_mean"] > best_reg_cv:
            best_reg_cv = metrics["cv_r2_mean"]
            best_reg_name = name

    # --- Model comparison: classifiers ---
    clf_comparison: Dict[str, Any] = {}
    best_clf_name = None
    best_clf_cv = -1e9
    n_health_classes = len(np.unique(y_health))
    min_health_count = int(np.bincount(y_health).min()) if n_health_classes > 1 else len(y_health)
    skip_clf_comparison = n_health_classes < 2 or min_health_count < 5

    if skip_clf_comparison:
        print(f"WARNING: Only {n_health_classes} health class(es) with min count {min_health_count}.")
        print("  Skipping multi-model classifier CV (not enough minority samples).")
        print("  Using XGBoost classifier with majority-class fallback.")
        best_clf_name = "xgboost"
        best_clf_cv = float(np.bincount(y_health).max()) / len(y_health)
        clf_comparison["xgboost"] = {"cv_f1_macro_mean": best_clf_cv, "cv_f1_macro_std": 0.0, "cv_accuracy_mean": best_clf_cv}
    else:
        for name in CLASSIFIER_NAMES:
            print(f"CV classifier: {name}...")
            params = best_clf_params if name in ("xgboost", "stacking") else {}
            model = _make_classifier(name, args.seed, params)
            metrics = _cv_classifier(model, X, y_health, args.seed, args.cv_folds)
            clf_comparison[name] = metrics
            if metrics["cv_f1_macro_mean"] > best_clf_cv:
                best_clf_cv = metrics["cv_f1_macro_mean"]
                best_clf_name = name

    print(f"\nBest regressor by CV R²: {best_reg_name} ({best_reg_cv:.4f})")
    print(f"Best classifier by CV F1-macro: {best_clf_name} ({best_clf_cv:.4f})")

    # Fit winners on train, evaluate on hold-out
    reg = _make_regressor(best_reg_name or "xgboost", args.seed, best_reg_params)
    reg.fit(X_train, y_tr)
    pred = reg.predict(X_test)
    yield_r2 = float(r2_score(y_te, pred))
    yield_rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    print(f"Hold-out yield R²={yield_r2:.4f} RMSE={yield_rmse:.4f}")

    clf = _make_classifier(best_clf_name or "xgboost", args.seed, best_clf_params)
    all_health_labels = sorted(np.unique(y_health))
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        kn = min(3, int(np.bincount(h_tr).min()) - 1) if len(np.unique(h_tr)) > 1 else 0
        if kn >= 1:
            clf_pipe = ImbPipeline([
                ("smote", SMOTE(random_state=args.seed, k_neighbors=kn)),
                ("clf", clf),
            ])
            clf_pipe.fit(X_train, h_tr)
            hp = _safe_predict(clf_pipe, X_test)
            clf_final = clf_pipe
        else:
            raise ValueError("Too few minority samples for SMOTE")
    except Exception as exc:  # noqa: BLE001
        print(f"SMOTE unavailable ({exc}); fitting classifier without oversampling.")
        clf.fit(X_train, h_tr)
        hp = _safe_predict(clf, X_test)
        clf_final = clf

    f1_mac = float(f1_score(h_te, hp, average="macro", labels=all_health_labels, zero_division=0))
    acc = float(accuracy_score(h_te, hp))
    print(classification_report(h_te, hp, digits=3, labels=all_health_labels, zero_division=0))
    print(f"Hold-out health F1-macro={f1_mac:.4f} accuracy={acc:.4f}")

    joblib.dump(reg, mdir / "yield_predictor.pkl")
    joblib.dump(clf_final, mdir / "health_classifier.pkl")
    if hasattr(reg, "get_booster"):
        reg.get_booster().save_model(str(mdir / "yield_predictor.json"))
    if hasattr(clf, "get_booster"):
        clf.get_booster().save_model(str(mdir / "health_classifier.json"))

    learning_curve = _learning_curve_regressor(reg, X, y_yield, args.seed)

    training_ranges = {
        "temperature_C": [float(df["temperature_C"].min()), float(df["temperature_C"].max())],
        "HRT_days": [float(df["HRT_days"].min()), float(df["HRT_days"].max())],
        "OLR": [float(df["OLR"].min()), float(df["OLR"].max())] if "OLR" in df.columns else [0.5, 8.0],
    }

    metrics = {
        "yield_r2": yield_r2,
        "yield_rmse": yield_rmse,
        "yield_cv_r2_mean": best_reg_cv,
        "health_f1_macro": f1_mac,
        "health_accuracy": acc,
        "health_cv_f1_macro_mean": best_clf_cv,
        "health_classification_report": classification_report(h_te, hp, digits=3, output_dict=True),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "feature_names": list(feat_mod.FEATURE_COLUMNS),
        "best_regressor": best_reg_name,
        "best_classifier": best_clf_name,
        "regressor_comparison": reg_comparison,
        "classifier_comparison": clf_comparison,
        "learning_curve": learning_curve,
        "training_ranges": training_ranges,
        "stability_distribution": df["stability_label"].value_counts().sort_index().to_dict(),
        "tuned": args.tune,
        "sampling_strategy": "HRT-first (see sweep.py)",
    }
    (mdir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (mdir / "model_comparison.json").write_text(
        json.dumps({"regressor": reg_comparison, "classifier": clf_comparison}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved models, eval_metrics.json, model_comparison.json to {mdir}")


if __name__ == "__main__":
    main()
