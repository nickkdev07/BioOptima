#!/usr/bin/env python3
"""Validate trained BioOptima models against published literature data points.

Literature values are specific yield (m3 CH4 / m3 reactor / d).
Model predictions use V_liq=100 m3; divide absolute yield by 100 for comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as feat_mod  # noqa: E402

V_LIQ = 100.0

# Training ranges (HRT-first sweep defaults); overridden from eval_metrics.json if present
DEFAULT_TRAINING_RANGES = {
    "temperature_C": (25.0, 45.0),
    "HRT_days": (15.0, 40.0),
    "OLR": (0.5, 8.0),
}


def _load_training_ranges(mdir: Path) -> Dict[str, Tuple[float, float]]:
    p = mdir / "eval_metrics.json"
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        tr = data.get("training_ranges")
        if tr:
            return {k: (float(v[0]), float(v[1])) for k, v in tr.items()}
    return {k: (float(v[0]), float(v[1])) for k, v in DEFAULT_TRAINING_RANGES.items()}


def _in_range(val: float, bounds: Tuple[float, float]) -> bool:
    return bounds[0] <= val <= bounds[1]


def _exclusion_reasons(
    temp: float,
    hrt: float,
    olr: float,
    ranges: Dict[str, Tuple[float, float]],
) -> List[str]:
    """Return human-readable labels for parameters outside training range."""
    reasons: List[str] = []
    if not _in_range(temp, ranges["temperature_C"]):
        reasons.append(f"T={temp:.0f}C")
    if not _in_range(hrt, ranges["HRT_days"]):
        reasons.append(f"HRT={hrt:.0f}d")
    if not _in_range(olr, ranges["OLR"]):
        reasons.append(f"OLR={olr:.1f}")
    return reasons


def _bias_correction_loo(observed: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """Fit y_obs = a * y_pred + b with leave-one-out style on full set (simple OLS)."""
    if len(observed) < 3:
        return 1.0, 0.0
    A = np.column_stack([predicted, np.ones(len(predicted))])
    coef, _, _, _ = np.linalg.lstsq(A, observed, rcond=None)
    return float(coef[0]), float(coef[1])


def _category(row: pd.Series) -> str:
    fracs = [
        float(row["cow_frac"]),
        float(row["rice_frac"]),
        float(row["wheat_frac"]),
        float(row["food_frac"]),
        float(row["press_frac"]),
        float(row["poultry_frac"]),
    ]
    names = ["cow", "rice", "wheat", "food", "press", "poultry"]
    dom = names[int(np.argmax(fracs))]
    mono = max(fracs) >= 0.95
    return f"{'mono' if mono else 'co'}-{dom}"


def main() -> None:
    lit_csv = ROOT / "data" / "validation" / "literature_points.csv"
    if not lit_csv.is_file():
        raise SystemExit(f"Missing literature CSV: {lit_csv}")

    mdir = ROOT / "models"
    reg = joblib.load(mdir / "yield_predictor.pkl")
    print(f"Loaded yield model from {mdir / 'yield_predictor.pkl'}")

    df = pd.read_csv(lit_csv)
    ranges = _load_training_ranges(mdir)
    print(f"Training ranges: {ranges}\n")

    preds_raw = []
    in_training_mask = []
    for _, row in df.iterrows():
        temp = float(row["temperature_C"])
        hrt = float(row["HRT_days"])
        olr = float(row["OLR"])
        ok = (
            _in_range(temp, ranges["temperature_C"])
            and _in_range(hrt, ranges["HRT_days"])
            and _in_range(olr, ranges["OLR"])
        )
        in_training_mask.append(ok)
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
        X = feat_mod.feature_matrix(feat_row)
        yhat_abs = float(reg.predict(X)[0])
        preds_raw.append(yhat_abs / V_LIQ)

    df["predicted_specific_yield"] = preds_raw
    obs_arr = df["observed_specific_yield"].values
    pred_arr = np.array(preds_raw)
    a_all, b_all = _bias_correction_loo(obs_arr, pred_arr) if len(obs_arr) >= 3 else (1.0, 0.0)
    df["predicted_specific_yield_corrected"] = a_all * pred_arr + b_all
    df["in_training_range"] = in_training_mask
    df["category"] = df.apply(_category, axis=1)
    df["abs_error"] = np.abs(df["observed_specific_yield"] - df["predicted_specific_yield"])
    df["pct_error"] = df["abs_error"] / np.maximum(df["observed_specific_yield"], 1e-6) * 100.0

    excluded = df[~df["in_training_range"]].copy()
    df_eval = df[df["in_training_range"]].copy()

    print("Excluded (outside training range):")
    for _, row in excluded.iterrows():
        t, h, o = float(row["temperature_C"]), float(row["HRT_days"]), float(row["OLR"])
        reasons = _exclusion_reasons(t, h, o, ranges)
        print(f"  {row['substrate_desc'][:50]:50s}  {', '.join(reasons)}")
    print()

    def _metrics(sub: pd.DataFrame, label: str) -> Dict[str, Any]:
        if len(sub) == 0:
            return {"n": 0}
        obs = sub["observed_specific_yield"].values
        pred = sub["predicted_specific_yield"].values
        mape = float(sub["pct_error"].mean())
        ss_res = float(np.sum((obs - pred) ** 2))
        ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        a, b = _bias_correction_loo(obs, pred)
        pred_corr = a * pred + b
        mape_corr = float(np.mean(np.abs(obs - pred_corr) / np.maximum(obs, 1e-6) * 100.0))
        return {
            "label": label,
            "n": int(len(sub)),
            "mape_pct": round(mape, 2),
            "r2": round(r2, 4),
            "bias_correction": {"a": round(a, 4), "b": round(b, 4)},
            "mape_pct_corrected": round(mape_corr, 2),
        }

    overall = _metrics(df_eval, "in_range")
    mono = _metrics(df_eval[df_eval["category"].str.startswith("mono")], "mono_digestion")
    co = _metrics(df_eval[df_eval["category"].str.startswith("co")], "co_digestion")

    print("Per-point results (in-range only, m3 CH4 / m3 reactor / d):")
    print("-" * 110)
    for _, row in df_eval.iterrows():
        print(
            f"  {row['substrate_desc']:45s}  "
            f"obs={row['observed_specific_yield']:6.3f}  "
            f"pred={row['predicted_specific_yield']:6.3f}  "
            f"err={row['pct_error']:5.1f}%  [{row['category']}]"
        )

    print("-" * 110)
    print(f"In-range MAPE = {overall.get('mape_pct', 'n/a')}%  R² = {overall.get('r2', 'n/a')}")
    if overall.get("bias_correction"):
        print(f"After bias correction: MAPE = {overall.get('mape_pct_corrected')}%")

    by_cat: Dict[str, Any] = {}
    for cat in df_eval["category"].unique():
        by_cat[cat] = _metrics(df_eval[df_eval["category"] == cat], cat)

    result = {
        "n_points_total": int(len(df)),
        "n_points_in_range": int(len(df_eval)),
        "n_points_excluded": int(len(excluded)),
        "training_ranges": {k: list(v) for k, v in ranges.items()},
        "overall_in_range": overall,
        "mono_digestion": mono,
        "co_digestion": co,
        "by_category": by_cat,
        "excluded_points": [
            {
                "substrate": row["substrate_desc"],
                "temperature_C": float(row["temperature_C"]),
                "HRT_days": float(row["HRT_days"]),
                "OLR": float(row["OLR"]),
                "reason": (
                    "outside training range: "
                    + ", ".join(
                        _exclusion_reasons(
                            float(row["temperature_C"]),
                            float(row["HRT_days"]),
                            float(row["OLR"]),
                            ranges,
                        )
                    )
                ),
            }
            for _, row in excluded.iterrows()
        ],
        "per_point": [
            {
                "source": row["source"],
                "substrate": row["substrate_desc"],
                "observed": round(float(row["observed_specific_yield"]), 4),
                "predicted": round(float(row["predicted_specific_yield"]), 4),
                "predicted_corrected": round(float(row["predicted_specific_yield_corrected"]), 4),
                "pct_error": round(float(row["pct_error"]), 1),
                "in_training_range": bool(row["in_training_range"]),
                "category": row["category"],
            }
            for _, row in df.iterrows()
        ],
        "unit": "m3_CH4_per_m3_reactor_per_day",
        "V_liq_model": V_LIQ,
        # Legacy keys for Model Insights page
        "mape_pct": overall.get("mape_pct", float("nan")),
        "r2": overall.get("r2", float("nan")),
        "n_points": int(len(df_eval)),
    }
    out = mdir / "literature_validation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
