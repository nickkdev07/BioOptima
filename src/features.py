"""
Feature engineering for BioOptima ML (aligned with training CSV columns).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from . import feedstock_db

FEATURE_COLUMNS: List[str] = [
    "cow_frac",
    "rice_frac",
    "wheat_frac",
    "food_frac",
    "press_frac",
    "poultry_frac",
    "temperature_C",
    "HRT_days",
    "OLR",
    "CN_ratio",
    "VS_total",
    "temp_HRT",
    "lipid_frac_mix",
    "protein_frac_mix",
    "COD_proxy",
    "OLR_x_food_frac",
    "protein_to_CN",
    "OLR_per_HRT",
    "lipid_x_temp",
]


def _mix_biochem(mass_fracs: Sequence[float]) -> tuple[float, float, float]:
    """Return (lipid_frac, protein_frac, cod_proxy) weighted by mass fractions."""
    mf = np.array(mass_fracs[: len(feedstock_db.SUBSTRATE_IDS)], dtype=float)
    if mf.size < len(feedstock_db.SUBSTRATE_IDS):
        mf = np.pad(mf, (0, len(feedstock_db.SUBSTRATE_IDS) - mf.size))
    s = float(mf.sum())
    if s <= 0:
        return 0.1, 0.2, 0.0
    mf = mf / s
    lip = pr = cod = 0.0
    for i, sid in enumerate(feedstock_db.SUBSTRATE_IDS):
        fch, fpr, fli, fxi = feedstock_db.BIOCHEM_VS[sid]
        lip += mf[i] * fli
        pr += mf[i] * fpr
        cod += mf[i] * (
            fch * feedstock_db.COD_CH
            + fpr * feedstock_db.COD_PR
            + fli * feedstock_db.COD_LI
            + fxi * feedstock_db.COD_XI
        )
    return float(lip), float(pr), float(cod)


def row_dict_to_features(row: Mapping[str, float]) -> Dict[str, float]:
    """Build feature dict from one sweep / UI row."""
    mass_fracs = [
        float(row.get("cow_frac", row.get("cow_dung_frac", 0.0))),
        float(row.get("rice_frac", row.get("rice_straw_frac", 0.0))),
        float(row.get("wheat_frac", row.get("wheat_straw_frac", 0.0))),
        float(row.get("food_frac", row.get("food_waste_frac", 0.0))),
        float(row.get("press_frac", row.get("press_mud_frac", 0.0))),
        float(row.get("poultry_frac", row.get("poultry_litter_frac", 0.0))),
    ]
    temp = float(row.get("temperature_C", 35.0))
    hrt = float(row.get("HRT_days", 25.0))
    olr = float(row.get("OLR", row.get("OLR_used", 2.0)))
    cn = feedstock_db.blended_cn_ratio(mass_fracs)
    lip, pr, cod_p = _mix_biochem(mass_fracs)
    vs_total = olr * hrt
    cn_safe = max(cn, 1e-6)
    return {
        "cow_frac": mass_fracs[0],
        "rice_frac": mass_fracs[1],
        "wheat_frac": mass_fracs[2],
        "food_frac": mass_fracs[3],
        "press_frac": mass_fracs[4],
        "poultry_frac": mass_fracs[5],
        "temperature_C": temp,
        "HRT_days": hrt,
        "OLR": olr,
        "CN_ratio": cn,
        "VS_total": vs_total,
        "temp_HRT": temp * hrt,
        "lipid_frac_mix": lip,
        "protein_frac_mix": pr,
        "COD_proxy": cod_p,
        "OLR_x_food_frac": olr * mass_fracs[3],
        "protein_to_CN": pr / cn_safe,
        "OLR_per_HRT": olr / max(hrt, 1e-6),
        "lipid_x_temp": lip * temp,
    }


def dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature frame from a sweep dataframe."""
    out = pd.DataFrame({c: np.zeros(len(df), dtype=float) for c in FEATURE_COLUMNS})
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        fd = row_dict_to_features(r)
        for c in FEATURE_COLUMNS:
            out.at[i, c] = fd[c]
    return out


def feature_matrix(row: Mapping[str, float]) -> np.ndarray:
    d = row_dict_to_features(row)
    return np.array([[d[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
