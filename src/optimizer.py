"""
Optuna-based feedstock optimization using trained ML yield surrogate.

Yield predictions use literature bias correction (calibration.py).
Health/stability uses rule-based pH/VFA estimates from operating parameters,
not the ML classifier (training data had no Warning/Critical samples).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import joblib
import numpy as np
import optuna

from . import calibration as cal_mod
from . import features as feat_mod
from . import feedstock_db
from . import health_utils

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _models_dir(root: Optional[Path] = None) -> Path:
    base = root or feedstock_db.project_root()
    return base / "models"


def load_yield_model(root: Optional[Path] = None):
    p = _models_dir(root) / "yield_predictor.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"Missing model: {p}")
    return joblib.load(p)


def load_health_model(root: Optional[Path] = None):
    """Optional legacy classifier; predictions use rule-based health instead."""
    p = _models_dir(root) / "health_classifier.pkl"
    if not p.is_file():
        return None
    return joblib.load(p)


def _predict_yield_health(
    yield_m,
    row: Mapping[str, float],
    *,
    bias_a: float,
    bias_b: float,
    v_liq: float = cal_mod.V_LIQ_DEFAULT,
) -> Tuple[float, int]:
    X = feat_mod.feature_matrix(row)
    y_raw = float(yield_m.predict(X)[0])
    yhat = cal_mod.apply_yield_correction(y_raw, bias_a, bias_b, v_liq=v_liq)
    hhat = cal_mod.predict_health_from_row(row)
    return yhat, hhat


def optimize_feedstock(
    available_kg: Mapping[str, float],
    temp_C: float,
    *,
    HRT_days: float = 25.0,
    OLR: float = 2.5,
    V_liq: float = 100.0,
    n_trials: int = 200,
    root: Optional[Path] = None,
    yield_model=None,
    health_model=None,
) -> Dict[str, Any]:
    """
    Maximize bias-corrected predicted CH4 yield under rule-based health penalty.

    ``available_kg`` keys: cow_dung, rice_straw, wheat_straw, food_waste, press_mud, poultry_litter.
    Returns recommended mass fractions, predicted yield, health label, C/N, improvement vs baseline.
    """
    _ = health_model  # rule-based health; ML classifier not used
    root = root or feedstock_db.project_root()
    ym = yield_model or load_yield_model(root)
    bias_a, bias_b = cal_mod.load_bias_correction(root)

    keys = ["cow_dung", "rice_straw", "wheat_straw", "food_waste", "press_mud", "poultry_litter"]
    avail = np.array([float(available_kg.get(k, 0.0)) for k in keys], dtype=float)
    tot = float(avail.sum())
    if tot <= 1e-9:
        return {
            "mix": {k: 0.0 for k in keys},
            "yield": 0.0,
            "health_label": "N/A",
            "cn_ratio": 25.0,
            "improvement_pct": 0.0,
            "error": "no_feedstock",
        }

    mass0 = avail / tot
    baseline_row = {
        "cow_frac": float(mass0[0]),
        "rice_frac": float(mass0[1]),
        "wheat_frac": float(mass0[2]),
        "food_frac": float(mass0[3]),
        "press_frac": float(mass0[4]),
        "poultry_frac": float(mass0[5]),
        "temperature_C": float(temp_C),
        "HRT_days": float(HRT_days),
        "OLR": float(OLR),
    }
    y0, h0 = _predict_yield_health(ym, baseline_row, bias_a=bias_a, bias_b=bias_b, v_liq=V_liq)

    def objective(trial: optuna.Trial) -> float:
        raw = []
        for i in range(6):
            raw.append(trial.suggest_float(f"m{i}", 0.0, max(avail[i], 1e-6)))
        r = np.array(raw, dtype=float)
        if r.sum() <= 0:
            return 0.0
        use = np.minimum(r, avail)
        mf = use / use.sum()
        row = {
            "cow_frac": float(mf[0]),
            "rice_frac": float(mf[1]),
            "wheat_frac": float(mf[2]),
            "food_frac": float(mf[3]),
            "press_frac": float(mf[4]),
            "poultry_frac": float(mf[5]),
            "temperature_C": float(temp_C),
            "HRT_days": float(HRT_days),
            "OLR": float(OLR),
        }
        yhat, hhat = _predict_yield_health(ym, row, bias_a=bias_a, bias_b=bias_b, v_liq=V_liq)
        if hhat == 2:
            return 0.0
        if hhat == 1:
            yhat *= 0.7
        return yhat

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    raw_best = np.array([best.params[f"m{i}"] for i in range(6)], dtype=float)
    use_best = np.minimum(raw_best, avail)
    mf_best = use_best / use_best.sum()

    row_best = {
        "cow_frac": float(mf_best[0]),
        "rice_frac": float(mf_best[1]),
        "wheat_frac": float(mf_best[2]),
        "food_frac": float(mf_best[3]),
        "press_frac": float(mf_best[4]),
        "poultry_frac": float(mf_best[5]),
        "temperature_C": float(temp_C),
        "HRT_days": float(HRT_days),
        "OLR": float(OLR),
    }
    y_best, h_best = _predict_yield_health(ym, row_best, bias_a=bias_a, bias_b=bias_b, v_liq=V_liq)

    improvement = 0.0 if y0 <= 1e-6 else (y_best - y0) / y0 * 100.0
    cn = feedstock_db.blended_cn_ratio(list(mf_best))

    return {
        "mix": dict(zip(keys, [float(x) for x in mf_best])),
        "yield": y_best,
        "health_label": health_utils.stability_name(h_best),
        "health_class": h_best,
        "cn_ratio": cn,
        "improvement_pct": float(improvement),
        "baseline_yield": y0,
        "baseline_health_class": h0,
        "error": None,
        "yield_bias_corrected": True,
    }


def predict_yield_health(
    row: Mapping[str, float],
    *,
    root: Optional[Path] = None,
    yield_model=None,
    health_model=None,
    apply_bias_correction: bool = True,
    v_liq: float = cal_mod.V_LIQ_DEFAULT,
) -> Tuple[float, int]:
    _ = health_model
    root = root or feedstock_db.project_root()
    ym = yield_model or load_yield_model(root)
    bias_a, bias_b = cal_mod.load_bias_correction(root) if apply_bias_correction else (1.0, 0.0)
    return _predict_yield_health(ym, row, bias_a=bias_a, bias_b=bias_b, v_liq=v_liq)
