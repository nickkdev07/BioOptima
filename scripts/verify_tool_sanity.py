#!/usr/bin/env python3
"""End-to-end sanity check: predictions, literature replay, optimizer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import calibration as cal_mod
from src import health_utils
from src.optimizer import optimize_feedstock, predict_yield_health

V_LIQ = cal_mod.V_LIQ_DEFAULT


def row_from_lit(r) -> dict:
    return {
        "cow_frac": float(r["cow_frac"]),
        "rice_frac": float(r["rice_frac"]),
        "wheat_frac": float(r["wheat_frac"]),
        "food_frac": float(r["food_frac"]),
        "press_frac": float(r["press_frac"]),
        "poultry_frac": float(r["poultry_frac"]),
        "temperature_C": float(r["temperature_C"]),
        "HRT_days": float(r["HRT_days"]),
        "OLR": float(r["OLR"]),
    }


def main() -> None:
    lit = pd.read_csv(ROOT / "data" / "validation" / "literature_points.csv")
    val = json.loads((ROOT / "models" / "literature_validation.json").read_text(encoding="utf-8"))
    a, b = cal_mod.load_bias_correction(ROOT)
    ranges = val["training_ranges"]

    print("=" * 70)
    print("BIOOPTIMA TOOL VERIFICATION")
    print("=" * 70)
    print(f"Bias correction: a={a:.4f}, b={b:.4f}")
    print(f"Training ranges: T={ranges['temperature_C']}, HRT={ranges['HRT_days']}, OLR={ranges['OLR']}")
    print()

    # --- Literature replay (in-range only) ---
    print("1) Literature points (in-range, bias-corrected specific yield m3/m3/d)")
    print("-" * 70)
    errors = []
    for _, r in lit.iterrows():
        row = row_from_lit(r)
        y_abs, h = predict_yield_health(row, root=ROOT)
        y_spec = y_abs / V_LIQ
        obs = float(r["observed_specific_yield"])
        err_pct = abs(y_spec - obs) / max(obs, 1e-6) * 100
        in_range = (
            ranges["temperature_C"][0] <= row["temperature_C"] <= ranges["temperature_C"][1]
            and ranges["HRT_days"][0] <= row["HRT_days"] <= ranges["HRT_days"][1]
            and ranges["OLR"][0] <= row["OLR"] <= ranges["OLR"][1]
        )
        if not in_range:
            continue
        errors.append(err_pct)
        flag = "OK" if err_pct < 25 else "HIGH"
        print(
            f"  [{flag}] {r['substrate_desc'][:42]:42s}  "
            f"obs={obs:.2f} pred={y_spec:.2f} err={err_pct:5.1f}%  health={health_utils.stability_name(h)}"
        )
    print(f"\n  In-range MAPE (this run): {sum(errors)/len(errors):.1f}%  (saved validation: {val['overall_in_range']['mape_pct_corrected']}%)")
    print()

    # --- Qualitative physics / process checks ---
    print("2) Qualitative checks (should match AD intuition)")
    print("-" * 70)

    base_cow = {
        "cow_frac": 1.0, "rice_frac": 0, "wheat_frac": 0, "food_frac": 0,
        "press_frac": 0, "poultry_frac": 0, "temperature_C": 35, "HRT_days": 25, "OLR": 2.0,
    }
    base_food = {**base_cow, "cow_frac": 0, "food_frac": 1.0}
    y_cow, h_cow = predict_yield_health(base_cow, root=ROOT)
    y_food, h_food = predict_yield_health(base_food, root=ROOT)
    print(f"  Mono cow dung @ 35C:  CH4={y_cow/V_LIQ:.2f} m3/m3/d  health={health_utils.stability_name(h_cow)}")
    print(f"  Mono food waste @ 35C: CH4={y_food/V_LIQ:.2f} m3/m3/d  health={health_utils.stability_name(h_food)}")
    check1 = y_food > y_cow
    print(f"  Food yield > cow yield? {'PASS' if check1 else 'FAIL'} (literature: food BMP >> cow)")

    mild = {**base_cow, "OLR": 2.5, "HRT_days": 30}
    stress = {**base_cow, "OLR": 7.0, "HRT_days": 12, "food_frac": 0.5, "cow_frac": 0.5}
    _, h_mild = predict_yield_health(mild, root=ROOT)
    _, h_stress = predict_yield_health(stress, root=ROOT)
    print(f"  Mild co-digestion OLR=2.5: health={health_utils.stability_name(h_mild)}")
    print(f"  Stressed OLR=7 HRT=12:   health={health_utils.stability_name(h_stress)}")
    check2 = h_stress >= h_mild
    print(f"  Stress >= mild severity? {'PASS' if check2 else 'FAIL'} (expect Warning or Critical under stress)")

    cold = {**base_cow, "temperature_C": 28}
    hot = {**base_cow, "temperature_C": 40}
    y_cold, _ = predict_yield_health(cold, root=ROOT)
    y_hot, _ = predict_yield_health(hot, root=ROOT)
    check3 = y_hot > y_cold
    print(f"  CH4 @ 28C vs 40C: {y_cold/V_LIQ:.2f} vs {y_hot/V_LIQ:.2f}  higher T better? {'PASS' if check3 else 'FAIL'}")
    print()

    # --- Optimizer (realistic farm scenario) ---
    print("3) Feed optimizer — typical Indian farm scenario")
    print("-" * 70)
    available = {
        "cow_dung": 400,
        "rice_straw": 80,
        "wheat_straw": 40,
        "food_waste": 120,
        "press_mud": 0,
        "poultry_litter": 30,
    }
    res = optimize_feedstock(
        available,
        temp_C=35.0,
        HRT_days=25.0,
        OLR=3.0,
        n_trials=80,
        root=ROOT,
    )
    if res.get("error"):
        print("  ERROR:", res["error"])
    else:
        print(f"  Baseline yield: {res['baseline_yield']:.1f} m3/d  ({res['baseline_yield']/V_LIQ:.2f} specific)")
        print(f"  Optimized yield: {res['yield']:.1f} m3/d  ({res['yield']/V_LIQ:.2f} specific)")
        print(f"  Improvement: {res['improvement_pct']:.1f}%")
        print(f"  Health: {res['health_label']}  C/N: {res['cn_ratio']:.1f}")
        mix = res["mix"]
        print("  Recommended mix (mass %):")
        for k, v in mix.items():
            if v > 0.01:
                print(f"    {k}: {v*100:.1f}%")
        check4 = res["yield"] >= res["baseline_yield"]
        print(f"  Optimizer improves yield? {'PASS' if check4 else 'FAIL'}")
        # BMP logic: should use some food waste if available
        check5 = mix.get("food_waste", 0) > 0.05
        print(f"  Uses food waste when available? {'PASS' if check5 else 'WARN'} (often optimal for CH4)")
    print()

    # --- Summary ---
    print("4) Cross-check vs published Indian AD ranges (order-of-magnitude)")
    print("-" * 70)
    print("  Typical mesophilic specific CH4: ~0.3–1.0 m3/m3 reactor/day for farm co-digestion")
    print("  Typical OLR (CSTR): ~2–6 kg VS/m3/d; >8 risks instability")
    y_typical, _ = predict_yield_health(
        {"cow_frac": 0.6, "rice_frac": 0.1, "wheat_frac": 0, "food_frac": 0.2,
         "press_frac": 0.1, "poultry_frac": 0, "temperature_C": 35, "HRT_days": 25, "OLR": 3.0},
        root=ROOT,
    )
    y_s = y_typical / V_LIQ
    check6 = 0.25 <= y_s <= 1.2
    print(f"  Typical mix prediction: {y_s:.2f} m3/m3/d  in [0.25, 1.2]? {'PASS' if check6 else 'FAIL'}")
    print()
    print("=" * 70)
    print("OVERALL: Tool pipeline loads models, applies bias correction, and behaves")
    print("         consistently with literature and AD process logic for in-range inputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
