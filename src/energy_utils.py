"""Energy / LPG equivalents (CH₄ flow in m³/d at normal conditions per model output)."""

from __future__ import annotations

# Plan: 1 m³ CH₄ ≈ 0.75 kg LPG equivalent (order-of-magnitude for farmer messaging)
CH4_M3_TO_LPG_KG = 0.75
LPG_CYLINDER_KG = 14.2
INR_PER_CYLINDER = 900.0

# Illustrative CO2-eq avoided per m³ CH₄ used as fuel (vs fossil substitution; order-of-magnitude).
CO2_AVOIDED_PER_M3_CH4 = 2.0
# Rough annual CO2 uptake per mature tree (illustrative).
TREE_CO2_PER_YEAR = 22.0


def lpg_kg_from_ch4_m3(ch4_m3: float) -> float:
    return float(ch4_m3) * CH4_M3_TO_LPG_KG


def lpg_cylinders_from_ch4_m3(ch4_m3: float) -> float:
    denom = max(LPG_CYLINDER_KG, 1e-9)
    return lpg_kg_from_ch4_m3(ch4_m3) / denom


def inr_savings_from_ch4_m3(ch4_m3: float, *, inr_per_cylinder: float | None = None) -> float:
    rate = float(INR_PER_CYLINDER if inr_per_cylinder is None else inr_per_cylinder)
    return lpg_cylinders_from_ch4_m3(ch4_m3) * rate


def co2_avoided_from_ch4_m3(ch4_m3: float) -> float:
    """Return kg CO2-eq avoided per day for a daily CH₄ flow rate (m³/d)."""
    return float(ch4_m3) * CO2_AVOIDED_PER_M3_CH4


def trees_equivalent(co2_kg: float, *, years: float = 1.0) -> float:
    """Trees needed to absorb ``co2_kg`` over ``years`` at TREE_CO2_PER_YEAR per tree."""
    denom = max(float(TREE_CO2_PER_YEAR) * float(years), 1e-9)
    return float(co2_kg) / denom


def monthly_savings(ch4_m3_per_day: float, *, inr_per_cylinder: float | None = None) -> float:
    return inr_savings_from_ch4_m3(ch4_m3_per_day, inr_per_cylinder=inr_per_cylinder) * 30.0


def annual_savings(ch4_m3_per_day: float, *, inr_per_cylinder: float | None = None) -> float:
    return inr_savings_from_ch4_m3(ch4_m3_per_day, inr_per_cylinder=inr_per_cylinder) * 365.0
