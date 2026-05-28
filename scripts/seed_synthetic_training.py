"""
Semi-empirical synthetic training data (fallback when PyADM1ODE sweep is unavailable).

Uses literature-grounded formulas instead of arbitrary coefficients:
  - BMP-weighted methane yield with Arrhenius temperature correction
  - First-order HRT saturation kinetics
  - Sigmoid OLR inhibition above ~4.5 kg VS/m3/d
  - pH and VFA driven by C/N and OLR stress
  - NH3 driven by protein fraction and temperature

Generates 2000 rows in < 2 seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feedstock_db, health_utils

BMP_VALUES = {sid: feedstock_db.FEEDSTOCK_META[sid]["bmp"] for sid in feedstock_db.SUBSTRATE_IDS}
BMP_ORDERED = np.array([BMP_VALUES[sid] for sid in feedstock_db.SUBSTRATE_IDS])
CN_ORDERED = np.array([feedstock_db.FEEDSTOCK_META[sid]["cn"] for sid in feedstock_db.SUBSTRATE_IDS])

PROTEIN_FRAC = np.array([feedstock_db.BIOCHEM_VS[sid][1] for sid in feedstock_db.SUBSTRATE_IDS])
LIPID_FRAC = np.array([feedstock_db.BIOCHEM_VS[sid][2] for sid in feedstock_db.SUBSTRATE_IDS])


def main() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    V_liq = 100.0
    rows = []

    alpha = np.array([2.0, 1.0, 1.0, 1.0, 0.5, 0.8])

    for i in range(n):
        fracs = rng.dirichlet(alpha)
        temp = float(rng.uniform(25, 45))
        hrt = float(rng.uniform(10, 50))
        olr = float(rng.uniform(1.0, 7.0))

        bmp_mix = float(fracs @ BMP_ORDERED)
        cn_mix = float(fracs @ CN_ORDERED)
        prot_mix = float(fracs @ PROTEIN_FRAC)
        lip_mix = float(fracs @ LIPID_FRAC)

        k_hrt = 8.0
        hrt_eff = 1.0 - np.exp(-hrt / k_hrt)

        temp_factor = np.exp(0.069 * (temp - 35.0))

        olr_penalty = 1.0 / (1.0 + np.exp(3.0 * (olr - 4.5)))

        q_ch4 = (bmp_mix / 1000.0) * olr * V_liq * hrt_eff * temp_factor * olr_penalty
        q_ch4 *= (1.0 + rng.normal(0, 0.08))
        q_ch4 = max(0.01, q_ch4)

        ph_base = 7.2
        ph_cn_shift = -0.3 * max(0, 15.0 - cn_mix) / 15.0
        ph_olr_shift = -0.4 * max(0, olr - 3.5)
        ph_prot_shift = -0.15 * max(0, prot_mix - 0.25)
        pH = ph_base + ph_cn_shift + ph_olr_shift + ph_prot_shift + rng.normal(0, 0.08)
        pH = float(np.clip(pH, 5.5, 8.3))

        vfa_base = 0.25
        vfa_olr = 0.6 * max(0, olr - 2.5) ** 1.3
        vfa_food = 0.4 * fracs[3] * olr
        vfa_lip = 0.3 * lip_mix * max(0, olr - 2.0)
        vfa = vfa_base + vfa_olr + vfa_food + vfa_lip + rng.exponential(0.08)
        vfa = float(np.clip(vfa, 0.05, 6.0))

        nh3_base = 0.015
        nh3_prot = 0.12 * prot_mix
        nh3_temp = 0.002 * max(0, temp - 35)
        nh3 = nh3_base + nh3_prot + nh3_temp + rng.exponential(0.005)
        nh3 = float(np.clip(nh3, 0.005, 0.15))

        vfa_slope = float(rng.normal(0.01 * max(0, olr - 3.0), 0.03))
        ph_slope = float(rng.normal(-0.005 * max(0, olr - 3.5), 0.01))

        stab = health_utils.classify_stability(pH, vfa)

        rows.append(
            {
                "idx": i,
                "cow_frac": float(fracs[0]),
                "rice_frac": float(fracs[1]),
                "wheat_frac": float(fracs[2]),
                "food_frac": float(fracs[3]),
                "press_frac": float(fracs[4]),
                "poultry_frac": float(fracs[5]),
                "temperature_C": temp,
                "HRT_days": hrt,
                "OLR": olr,
                "converged": True,
                "q_ch4_avg": float(q_ch4),
                "pH_final": pH,
                "VFA_final": vfa,
                "S_nh3_final": nh3,
                "vfa_slope_last5d": vfa_slope,
                "ph_slope_last5d": ph_slope,
                "first_instability_day": -1,
                "sim_error": np.nan,
                "stability_label": stab,
            }
        )

    out = ROOT / "data" / "generated" / "biogas_training_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({n} rows, semi-empirical fallback)")


if __name__ == "__main__":
    main()
