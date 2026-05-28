"""
Literature bias correction and rule-based digester health from operating parameters.

Yield correction fits observed = a * predicted_specific + b on in-range literature
points (see models/literature_validation.json). Health uses estimated pH/VFA from
OLR, HRT, and feed mix, then standard AD thresholds (health_utils).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np

from . import feedstock_db
from . import health_utils

V_LIQ_DEFAULT = 100.0

# UI slider bounds aligned with achieved training sweep (see eval_metrics.json)
UI_OLR_MIN = 1.75
UI_OLR_MAX = 6.0
UI_HRT_MIN = 10.0
UI_HRT_MAX = 50.0
UI_TEMP_MIN = 25.0
UI_TEMP_MAX = 42.0


def _models_dir(root: Path | None = None) -> Path:
    return (root or feedstock_db.project_root()) / "models"


def load_bias_correction(root: Path | None = None) -> Tuple[float, float]:
    """Return (a, b) for specific-yield correction; identity if missing."""
    p = _models_dir(root) / "literature_validation.json"
    if not p.is_file():
        return 1.0, 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        overall = data.get("overall_in_range") or {}
        bc = overall.get("bias_correction") or {}
        a = float(bc.get("a", 1.0))
        b = float(bc.get("b", 0.0))
        if a <= 0:
            return 1.0, 0.0
        return a, b
    except (json.JSONDecodeError, TypeError, ValueError):
        return 1.0, 0.0


def apply_yield_correction(
    y_abs_m3_per_d: float,
    a: float,
    b: float,
    *,
    v_liq: float = V_LIQ_DEFAULT,
) -> float:
    """Apply literature bias correction on specific yield, return absolute m³/d."""
    y_spec = float(y_abs_m3_per_d) / max(v_liq, 1e-6)
    y_spec_c = a * y_spec + b
    return max(0.0, y_spec_c * v_liq)


def estimate_ph_vfa_from_row(row: Mapping[str, float]) -> Tuple[float, float]:
    """
    Heuristic steady-state pH and VFA (kg HAc-eq/m³) from operating parameters.

    Used when lab readings are unavailable; tuned so high OLR / low HRT / high
    food fraction map toward Warning/Critical bands in health_utils.
    """
    olr = float(row.get("OLR", row.get("OLR_used", 2.5)))
    hrt = float(row.get("HRT_days", 25.0))
    food = float(row.get("food_frac", 0.0))
    press = float(row.get("press_frac", 0.0))
    poultry = float(row.get("poultry_frac", 0.0))
    ligno = float(row.get("rice_frac", 0.0)) + float(row.get("wheat_frac", 0.0))
    temp = float(row.get("temperature_C", 35.0))

    vfa = (
        0.32
        + 0.20 * max(0.0, olr - 2.0)
        + 0.50 * food
        + 0.22 * press
        + 0.18 * poultry
        + 0.10 * ligno
    )
    if hrt < 16.0:
        vfa += 0.28 * (16.0 - hrt) / 16.0
    if olr > 4.5:
        vfa += 0.25 * (olr - 4.5)
    if olr > 5.5:
        vfa += 0.35 * (olr - 5.5)
    if temp < 28.0:
        vfa += 0.12

    ph = 7.28 - 0.14 * max(0.0, vfa - 0.45) - 0.07 * max(0.0, olr - 3.5)
    if hrt < 14.0:
        ph -= 0.12
    if temp < 28.0:
        ph -= 0.06

    return float(np.clip(ph, 5.5, 8.2)), float(np.clip(vfa, 0.1, 4.5))


def predict_health_from_row(row: Mapping[str, float]) -> int:
    """Rule-based stability class 0/1/2 from estimated pH/VFA."""
    ph, vfa = estimate_ph_vfa_from_row(row)
    return health_utils.classify_stability(ph, vfa)
