"""
Indian feedstock library and helpers to map user inputs to PyADM1ODE flows.

COD conversion (reference, for documentation / derived features):
  carbohydrates 1.07 kg COD / kg
  proteins      1.42 kg COD / kg
  lipids        2.88 kg COD / kg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Substrate YAML stems (order fixed for Q vectors)
SUBSTRATE_IDS: Tuple[str, ...] = (
    "bio_cow_dung",
    "bio_rice_straw",
    "bio_wheat_straw",
    "bio_food_waste",
    "bio_press_mud",
    "bio_poultry_litter",
)

# Mid-range literature values for UI / C:N display
FEEDSTOCK_META: Dict[str, Dict[str, float]] = {
    "bio_cow_dung": {"cn": 18.0, "bmp": 200.0, "ts_pct_fm": 20.0, "vs_of_ts": 0.775},
    "bio_rice_straw": {"cn": 50.0, "bmp": 230.0, "ts_pct_fm": 90.0, "vs_of_ts": 0.825},
    "bio_wheat_straw": {"cn": 90.0, "bmp": 240.0, "ts_pct_fm": 89.0, "vs_of_ts": 0.84},
    "bio_food_waste": {"cn": 17.0, "bmp": 380.0, "ts_pct_fm": 25.0, "vs_of_ts": 0.90},
    "bio_press_mud": {"cn": 20.0, "bmp": 260.0, "ts_pct_fm": 72.0, "vs_of_ts": 0.75},
    "bio_poultry_litter": {"cn": 7.5, "bmp": 280.0, "ts_pct_fm": 62.0, "vs_of_ts": 0.685},
}

COD_CH = 1.07
COD_PR = 1.42
COD_LI = 2.88
COD_XI = 1.42

# Plan biochemical fractions of VS (f_ch, f_pr, f_li, f_xi) — used for weighted C:N proxy
BIOCHEM_VS: Dict[str, Tuple[float, float, float, float]] = {
    "bio_cow_dung": (0.35, 0.20, 0.10, 0.35),
    "bio_rice_straw": (0.65, 0.05, 0.02, 0.28),
    "bio_wheat_straw": (0.60, 0.05, 0.02, 0.33),
    "bio_food_waste": (0.50, 0.25, 0.15, 0.10),
    "bio_press_mud": (0.40, 0.12, 0.08, 0.40),
    "bio_poultry_litter": (0.25, 0.35, 0.10, 0.30),
}

# Approximate fresh-matter bulk density (kg/m³) for plant-sizing / volume estimates (UI only).
APPROX_DENSITY: Dict[str, float] = {
    "bio_cow_dung": 1000.0,
    "bio_rice_straw": 150.0,
    "bio_wheat_straw": 150.0,
    "bio_food_waste": 900.0,
    "bio_press_mud": 700.0,
    "bio_poultry_litter": 600.0,
}

PRACTICAL_TIPS: Dict[str, str] = {
    "bio_cow_dung": (
        "Stable base feed; co-digest with straw or press mud to lift gas yield. "
        "Year-round on dairy farms; little pre-treatment beyond mixing."
    ),
    "bio_rice_straw": (
        "High C/N — pair with dung, food waste, or poultry litter. "
        "Chop or soak to improve handling; seasonal post-harvest availability."
    ),
    "bio_wheat_straw": (
        "Similar to rice straw; blend with nitrogen-rich streams. "
        "Mechanical size reduction helps hydrolysis."
    ),
    "bio_food_waste": (
        "High degradability — limit share if VFA spikes; good with dung for buffering. "
        "Remove plastics; consider hygiene and rapid feeding."
    ),
    "bio_press_mud": (
        "Sugar-mill residue; often good with straw for C/N balance. "
        "Watch seasonal supply; can be fibrous — mix well."
    ),
    "bio_poultry_litter": (
        "High ammonia risk — dilute with dung or straw, avoid overload. "
        "Monitor free NH₃; good N source for high-carbon feeds."
    ),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def substrate_yaml_paths(
    root: Optional[Path] = None,
    *,
    subdir: Optional[str] = None,
) -> List[Path]:
    """Return ordered YAML paths; ``subdir`` e.g. ``low_olr`` uses ``data/substrates/low_olr/``."""
    base = (root or project_root()) / "data" / "substrates"
    if subdir:
        base = base / subdir
    return [base / f"{sid}.yaml" for sid in SUBSTRATE_IDS]


def blended_cn_ratio(mass_fractions: Sequence[float]) -> float:
    """mass_fractions aligned with SUBSTRATE_IDS (simple mass-weighted C/N)."""
    if len(mass_fractions) != len(SUBSTRATE_IDS):
        raise ValueError("mass_fractions must match SUBSTRATE_IDS length")
    w = sum(mass_fractions)
    if w <= 0:
        return 25.0
    num = sum(
        float(mf) * FEEDSTOCK_META[sid]["cn"] for mf, sid in zip(mass_fractions, SUBSTRATE_IDS) if mf > 0
    )
    return float(num / w)


def masses_kg_to_mass_fractions(
    cow_kg: float,
    rice_kg: float,
    wheat_kg: float,
    food_kg: float,
    press_kg: float,
    poultry_kg: float,
) -> List[float]:
    masses = [cow_kg, rice_kg, wheat_kg, food_kg, press_kg, poultry_kg]
    tot = sum(masses)
    if tot <= 0:
        return [0.0] * len(SUBSTRATE_IDS)
    return [float(m) / tot for m in masses]


def estimate_daily_volume_m3(masses_kg_per_day: Mapping[str, float]) -> float:
    """
    Sum daily volumetric feed (m³/d) from named masses using ``APPROX_DENSITY``.

    Keys: ``cow_dung``, ``rice_straw``, ``wheat_straw``, ``food_waste``,
    ``press_mud``, ``poultry_litter`` (or ``bio_*`` substrate ids).
    """
    key_map = {
        "cow_dung": "bio_cow_dung",
        "rice_straw": "bio_rice_straw",
        "wheat_straw": "bio_wheat_straw",
        "food_waste": "bio_food_waste",
        "press_mud": "bio_press_mud",
        "poultry_litter": "bio_poultry_litter",
    }
    vol = 0.0
    for k, m in masses_kg_per_day.items():
        sid = key_map.get(str(k), str(k))
        if sid not in APPROX_DENSITY:
            continue
        rho = max(APPROX_DENSITY[sid], 1e-6)
        vol += float(m) / rho
    return float(vol)


def estimate_daily_vs_kg(masses_kg_per_day: Mapping[str, float]) -> float:
    """Total volatile solids (kg VS/d) from daily fresh-matter masses using ``FEEDSTOCK_META``."""
    key_map = {
        "cow_dung": "bio_cow_dung",
        "rice_straw": "bio_rice_straw",
        "wheat_straw": "bio_wheat_straw",
        "food_waste": "bio_food_waste",
        "press_mud": "bio_press_mud",
        "poultry_litter": "bio_poultry_litter",
    }
    total = 0.0
    for k, m in masses_kg_per_day.items():
        sid = key_map.get(str(k), str(k))
        if sid not in FEEDSTOCK_META:
            continue
        meta = FEEDSTOCK_META[sid]
        ts_frac = float(meta["ts_pct_fm"]) / 100.0
        vs = float(m) * ts_frac * float(meta["vs_of_ts"])
        total += vs
    return float(total)


def build_Q_from_olr_and_fractions(
    fs,
    OLR: float,
    V_liq: float,
    vs_fractions: Sequence[float],
    simba_q_convention: bool = False,
) -> List[float]:
    """
    Build per-substrate Q [m³/d] (length 10, padded) from organic loading rate.

    OLR: kg VS / (m³ reactor) / day
    vs_fractions: non-negative, will be normalized to sum to 1 over positive entries.
    """
    import numpy as np

    vf = np.array(vs_fractions[: len(SUBSTRATE_IDS)], dtype=float)
    if vf.size < len(SUBSTRATE_IDS):
        vf = np.pad(vf, (0, len(SUBSTRATE_IDS) - vf.size))
    s = float(vf.sum())
    if s <= 0:
        return [0.0] * 10
    vf = vf / s

    target_vs_rate = float(OLR) * float(V_liq)  # kg VS / d
    Q: List[float] = []
    for i, sid in enumerate(SUBSTRATE_IDS):
        vs_per_m3 = float(fs.vs_content(i))
        if vs_per_m3 <= 1e-9:
            Q.append(0.0)
            continue
        vs_i = target_vs_rate * float(vf[i])
        q_i = vs_i / vs_per_m3
        Q.append(float(q_i))

    # Pad to 10 slots for Digester
    while len(Q) < 10:
        Q.append(0.0)
    return Q[:10]


def build_Q_from_hrt_fractions(
    fs,
    HRT: float,
    V_liq: float,
    vs_fractions: Sequence[float],
) -> Tuple[List[float], float]:
    """
    Build per-substrate Q [m³/d] from target HRT; derive achieved OLR from flows.

    Q_total = V_liq / HRT. Volumetric split follows normalized VS-fraction weights
    (vf_i / vs_content(i)). Achieved OLR = sum(Q_i * vs_content(i)) / V_liq [kg VS/m³/d].
    """
    import numpy as np

    vf = np.array(vs_fractions[: len(SUBSTRATE_IDS)], dtype=float)
    if vf.size < len(SUBSTRATE_IDS):
        vf = np.pad(vf, (0, len(SUBSTRATE_IDS) - vf.size))
    s = float(vf.sum())
    if s <= 0 or HRT <= 0:
        return [0.0] * 10, 0.0
    vf = vf / s

    q_total = float(V_liq) / float(HRT)
    weights = []
    for i in range(len(SUBSTRATE_IDS)):
        vs_per_m3 = float(fs.vs_content(i))
        if vs_per_m3 <= 1e-9 or vf[i] <= 0:
            weights.append(0.0)
        else:
            weights.append(float(vf[i]) / vs_per_m3)
    wsum = float(sum(weights))
    if wsum <= 1e-15:
        return [0.0] * 10, 0.0

    Q: List[float] = []
    vs_load = 0.0
    for i in range(len(SUBSTRATE_IDS)):
        q_i = q_total * weights[i] / wsum
        Q.append(float(q_i))
        vs_load += q_i * float(fs.vs_content(i))
    while len(Q) < 10:
        Q.append(0.0)
    achieved_olr = vs_load / float(V_liq)
    return Q[:10], float(achieved_olr)


def build_Q_from_olr_hrt_fractions(
    fs,
    OLR: float,
    HRT: float,
    V_liq: float,
    vs_fractions: Sequence[float],
) -> Tuple[List[float], float]:
    """
    Scale volumetric flows so hydraulic retention time matches ``HRT`` [d].

    Returns (Q padded to length 10, achieved OLR [kg VS/m³/d] after scaling).
    """
    Q = build_Q_from_olr_and_fractions(fs, OLR, V_liq, vs_fractions)
    ssum = float(sum(Q[:10]))
    if ssum <= 1e-15 or HRT <= 0:
        return Q, float(OLR)
    hrt0 = float(V_liq) / ssum
    scale = hrt0 / float(HRT)
    Q2 = [float(q) * scale for q in Q[:10]]
    while len(Q2) < 10:
        Q2.append(0.0)
    achieved_olr = float(OLR) * scale
    return Q2, achieved_olr


def build_Q_from_daily_masses_kg(
    fs,
    masses_kg_per_day: Mapping[str, float],
    V_liq: float,
    simba_q_convention: bool = False,
) -> List[float]:
    """Convert named daily mass feeds (kg FM/d) to Q [m³/d] using fresh-matter density."""
    key_map = {
        "cow_dung": "bio_cow_dung",
        "rice_straw": "bio_rice_straw",
        "wheat_straw": "bio_wheat_straw",
        "food_waste": "bio_food_waste",
        "press_mud": "bio_press_mud",
        "poultry_litter": "bio_poultry_litter",
    }
    mass_by_sid = {sid: 0.0 for sid in SUBSTRATE_IDS}
    for k, v in masses_kg_per_day.items():
        sid = key_map.get(k, k)
        if sid in mass_by_sid:
            mass_by_sid[sid] += float(v)

    Q: List[float] = []
    for i, sid in enumerate(SUBSTRATE_IDS):
        m = mass_by_sid[sid]
        rho = max(float(fs.densities[i]), 400.0)
        Q.append(float(m) / rho if m > 0 else 0.0)
    while len(Q) < 10:
        Q.append(0.0)
    return Q[:10]


def export_feedstock_library_json(path: Optional[Path] = None) -> Path:
    """Write combined metadata to data/feedstock_library.json."""
    import json

    root = project_root()
    out = path or (root / "data" / "feedstock_library.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "substrate_ids": list(SUBSTRATE_IDS),
        "meta": FEEDSTOCK_META,
        "biochem_vs_fractions": {k: list(v) for k, v in BIOCHEM_VS.items()},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
