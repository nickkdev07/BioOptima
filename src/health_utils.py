"""
Digester health labels (from final pH / VFA) and rule-based early warning.

VFA from PyADM1ODE is kg HAc-eq/m³ (Schlattmann 2011). Thresholds below are
aligned with typical process bands (~1.5 ≈ 1500 mg/L order of magnitude).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# 0 Stable, 1 Warning, 2 Critical
STABILITY_LABELS = ("Stable", "Warning", "Critical")


def classify_stability(pH: float, vfa_kg_m3: float) -> int:
    """Return stability class index 0/1/2."""
    if pH < 6.5 or vfa_kg_m3 > 3.0:
        return 2
    if pH < 6.8 or pH > 7.8 or vfa_kg_m3 > 1.5:
        return 1
    return 0


def stability_name(idx: int) -> str:
    return STABILITY_LABELS[int(idx)] if 0 <= int(idx) < 3 else "Unknown"


def health_score_from_readings(
    pH: float,
    vfa_kg_m3: float,
    nh3_kg_m3: float,
    *,
    health_class: Optional[int] = None,
) -> float:
    """Map readings + optional classifier output to 0–100 score."""
    base = 100.0
    if health_class == 2:
        return 5.0
    if health_class == 1:
        base = 55.0
    if pH < 6.5 or pH > 8.0:
        base -= 25.0
    elif pH < 6.8 or pH > 7.8:
        base -= 15.0
    if vfa_kg_m3 > 3.0:
        base -= 35.0
    elif vfa_kg_m3 > 1.5:
        base -= 20.0
    if nh3_kg_m3 > 0.08:
        base -= 15.0
    return float(max(0.0, min(100.0, base)))


def early_warning(
    current_health: int,
    vfa_slope: float,
    ph_slope: float,
    current_vfa: float,
    current_ph: float,
) -> Dict[str, Any]:
    """
    Rule-based risk score and messages (replaces a flawed third ML model).

    vfa_slope: kg HAc-eq/m³ per day (from last ~5 days of simulation).
    ph_slope: pH units per day.
    """
    risk_score = 0
    warnings: List[str] = []

    if int(current_health) == 2:
        risk_score = 100
        warnings.append("Critical: process indicators suggest acidification or severe stress. Reduce loading and seek expert advice.")
    elif int(current_health) == 1:
        risk_score += 50
        warnings.append("Warning: digester is drifting from optimal conditions.")

    if vfa_slope > 0.15:
        risk_score += 25
        warnings.append("VFA is rising quickly — reduce organic loading (~15–25%).")
    if ph_slope < -0.04:
        risk_score += 25
        warnings.append("pH is falling — check alkalinity and feeding consistency.")
    if current_vfa > 2.0 and current_ph < 6.85:
        risk_score += 20
        warnings.append("Combined low pH and elevated VFA — risk of further acidification.")

    risk_score = min(100, risk_score)
    if risk_score > 70:
        level = "Critical"
    elif risk_score > 30:
        level = "Warning"
    else:
        level = "Stable"

    return {"risk_score": risk_score, "warnings": warnings, "level": level}


def infer_readings_from_quick_mode(
    gas_drop: bool,
    bad_smell: bool,
    foam: bool,
    feed_change: bool,
    season_temp: str,
) -> Dict[str, float]:
    """Map qualitative symptoms to approximate pH / VFA / NH3 for classifier."""
    pH = 7.1
    vfa = 0.4
    nh3 = 0.02
    if gas_drop:
        vfa += 0.5
        pH -= 0.15
    if bad_smell:
        vfa += 0.35
        nh3 += 0.02
    if foam:
        vfa += 0.25
        pH -= 0.08
    if feed_change:
        vfa += 0.2
        pH -= 0.05
    if season_temp == "cold":
        pH -= 0.05
        vfa += 0.1
    elif season_temp == "hot":
        nh3 += 0.03
    return {"pH": pH, "VFA": vfa, "S_nh3": nh3}
