#!/usr/bin/env python3
"""Mirror Feed Optimizer page — same inputs as verify_tool_sanity farm scenario."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import energy_utils, optimizer
from src.calibration import V_LIQ_DEFAULT
from src.model_loader import load_models
from src.ui_helpers import DISPLAY_NAMES, FEEDSTOCK_KEYS

# Farm scenario (same as verify_tool_sanity.py)
COW, RICE, WHEAT, FOOD, PRESS, POULTRY = 400, 80, 40, 120, 0, 30
TEMP, HRT, OLR = 35.0, 25.0, 3.0

ym, hm = load_models()
avail = {
    "cow_dung": COW,
    "rice_straw": RICE,
    "wheat_straw": WHEAT,
    "food_waste": FOOD,
    "press_mud": PRESS,
    "poultry_litter": POULTRY,
}

res = optimizer.optimize_feedstock(
    avail, TEMP, HRT_days=HRT, OLR=OLR, n_trials=200, yield_model=ym, health_model=hm
)

print("=" * 60)
print("FEED OPTIMIZER — UI mirror (values shown on screen)")
print("=" * 60)
print("\nEnter these in the app:\n")
print(f"  Cow dung: {COW} kg/d   Rice straw: {RICE}   Wheat straw: {WHEAT}")
print(f"  Food waste: {FOOD}   Press mud: {PRESS}   Poultry litter: {POULTRY}")
print(f"  Temperature: {TEMP} C   HRT: {HRT} d   OLR: {OLR} kg VS/m3/d")
print("\nThen click: Find best mix\n")
print("-" * 60)
if res.get("error"):
    print("ERROR:", res["error"])
    sys.exit(1)

m = res["mix"]
daily_ch4 = float(res["yield"])
spec = daily_ch4 / V_LIQ_DEFAULT

print("Metrics row 1:")
print(f"  Predicted CH4:     {daily_ch4:.2f} m3/d  ({res['improvement_pct']:+.1f}% vs baseline)")
print(f"  Predicted stability: {res['health_label']}")
print(f"  C/N (proxy):       {res['cn_ratio']:.1f}")

cyl = energy_utils.lpg_cylinders_from_ch4_m3(daily_ch4)
inr = energy_utils.inr_savings_from_ch4_m3(daily_ch4)
co2 = daily_ch4 * 2.0
print("\nMetrics row 2:")
print(f"  LPG equivalent:  {cyl:.2f} cyl./day")
print(f"  Rupee value:     Rs {inr:,.0f}/day")
print(f"  CO2 avoided:     {co2:.1f} kg/day")

print("\nRecommended mix (pie chart %):")
for k in FEEDSTOCK_KEYS:
    pct = m[k] * 100
    if pct > 0.5:
        print(f"  {DISPLAY_NAMES[k]}: {pct:.1f}%")

print(f"\nBaseline (equal split of available mass): {res['baseline_yield']:.2f} m3/d")
print(f"Specific yield (V_liq=100): {spec:.2f} m3/m3 reactor/day")
print("=" * 60)
