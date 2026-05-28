"""
PyADM1ODE wrapper — single-digester simulation for BioOptima training & validation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import feedstock_db

_REPO = feedstock_db.project_root()
_PYADM = _REPO / "pyADM1ODE"
if _PYADM.is_dir() and str(_PYADM) not in sys.path:
    sys.path.insert(0, str(_PYADM))

# Index for free ammonia in ADM1da state vector (41 states)
_S_NH3_IDX = 36


def _get_feedstock(sim_days: int = 60, *, substrates_subdir: Optional[str] = None) -> Any:
    from pyadm1 import Feedstock

    paths = feedstock_db.substrate_yaml_paths(_REPO, subdir=substrates_subdir)
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing substrate file: {p}")
    return Feedstock(
        paths,
        feeding_freq=24,
        total_simtime=max(sim_days + 5, 60),
        simba_q_convention=False,
    )


def run_simulation(
    Q_substrates: Sequence[float],
    *,
    temp_C: float = 35.0,
    V_liq: float = 100.0,
    V_gas: float = 20.0,
    sim_days: int = 30,
    k_L_a: float = 200.0,
) -> Dict[str, Any]:
    """
    Run a single digester simulation.

    Q_substrates: up to 10 entries [m³/d] per substrate slot (BioOptima uses first 6).

    Returns keys per revised plan including time_series and slopes.
    """
    from pyadm1 import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    out: Dict[str, Any] = {
        "q_ch4_avg": 0.0,
        "pH_final": 7.0,
        "S_ac_final": 0.0,
        "S_nh3_final": 0.0,
        "VFA_final": 0.0,
        "time_series": pd.DataFrame(),
        "vfa_slope_last5d": 0.0,
        "ph_slope_last5d": 0.0,
        "first_instability_day": -1,
        "converged": False,
        "HRT_final": 0.0,
        "error": None,
    }

    Q = list(Q_substrates[:10]) if len(Q_substrates) >= 10 else list(Q_substrates) + [0.0] * (10 - len(Q_substrates))
    if sum(abs(q) for q in Q[:6]) < 1e-12:
        out["error"] = "zero_flow"
        return out

    T_ad = 273.15 + float(temp_C)

    try:
        fs = _get_feedstock(sim_days=sim_days)
        plant = BiogasPlant("BioOptima")
        cfg = PlantConfigurator(plant, fs)
        cfg.add_digester(
            digester_id="main_digester",
            V_liq=float(V_liq),
            V_gas=float(V_gas),
            T_ad=float(T_ad),
            Q_substrates=Q,
            k_L_a=float(k_L_a),
        )
        plant.initialize()
        results = plant.simulate(duration=float(sim_days), dt=1.0, save_interval=1.0)
    except Exception as exc:  # noqa: BLE001
        out["converged"] = False
        out["error"] = str(exc)
        return out

    if not results:
        out["error"] = "no_results"
        return out

    times: List[float] = []
    q_ch4: List[float] = []
    pH: List[float] = []
    vfa: List[float] = []
    s_nh3: List[float] = []
    sac: List[float] = []
    hrt: List[float] = []

    for r in results:
        dig = r["components"].get("main_digester")
        if dig is None:
            continue
        times.append(float(r["time"]))
        q_ch4.append(float(dig.get("Q_ch4", 0.0)))
        pH.append(float(dig.get("pH", 7.0)))
        vfa.append(float(dig.get("VFA", 0.0)))
        hrt.append(float(dig.get("HRT", 0.0)))
        st = dig.get("state_out")
        if isinstance(st, (list, tuple)) and len(st) > _S_NH3_IDX:
            s_nh3.append(float(st[_S_NH3_IDX]))
            try:
                from pyadm1.core.adm1 import _IDX_S_AC

                sac.append(float(st[_IDX_S_AC]))
            except Exception:  # noqa: BLE001
                sac.append(vfa[-1])
        else:
            s_nh3.append(0.0)
            sac.append(vfa[-1])

    ts = pd.DataFrame(
        {
            "time_d": times,
            "q_ch4": q_ch4,
            "pH": pH,
            "VFA": vfa,
            "S_nh3": s_nh3,
            "S_ac": sac,
            "HRT": hrt,
        }
    )

    out["converged"] = True
    out["time_series"] = ts
    out["pH_final"] = float(ts["pH"].iloc[-1])
    out["VFA_final"] = float(ts["VFA"].iloc[-1])
    out["S_ac_final"] = float(ts["S_ac"].iloc[-1])
    out["S_nh3_final"] = float(ts["S_nh3"].iloc[-1])
    out["HRT_final"] = float(ts["HRT"].iloc[-1])

    tail = min(7, len(ts))
    out["q_ch4_avg"] = float(ts["q_ch4"].iloc[-tail:].mean())

    # Slopes over last 5 saved days (1 d interval)
    if len(ts) >= 2:
        n = min(5, len(ts) - 1)
        sub = ts.iloc[-(n + 1) :]
        x = np.arange(len(sub), dtype=float)
        if len(sub) >= 2:
            out["vfa_slope_last5d"] = float(np.polyfit(x, sub["VFA"].values, 1)[0])
            out["ph_slope_last5d"] = float(np.polyfit(x, sub["pH"].values, 1)[0])

    unstable = (ts["pH"] < 6.5) | (ts["VFA"] > 4.0)
    if unstable.any():
        out["first_instability_day"] = int(ts.loc[unstable, "time_d"].iloc[0])

    return out


def run_from_olr_mix(
    vs_fractions: Sequence[float],
    *,
    temp_C: float,
    OLR: float,
    V_liq: float = 100.0,
    HRT: Optional[float] = None,
    sim_days: int = 30,
) -> Dict[str, Any]:
    """Convenience: OLR-driven mix (vs_fractions length 6, sums ~1). Optional HRT scales flows."""
    fs = _get_feedstock(sim_days=sim_days)
    if HRT is not None and float(HRT) > 0:
        Q, olr_used = feedstock_db.build_Q_from_olr_hrt_fractions(fs, OLR, HRT, V_liq, vs_fractions)
    else:
        Q = feedstock_db.build_Q_from_olr_and_fractions(fs, OLR, V_liq, vs_fractions)
        olr_used = float(OLR)
    out = run_simulation(Q, temp_C=temp_C, V_liq=V_liq, sim_days=sim_days)
    out["OLR_used"] = olr_used
    return out


def run_from_hrt_mix(
    vs_fractions: Sequence[float],
    *,
    temp_C: float,
    HRT: float,
    V_liq: float = 100.0,
    sim_days: int = 30,
    substrates_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """HRT-first mix: flows set by HRT, achieved OLR returned as ``OLR``."""
    fs = _get_feedstock(sim_days=sim_days, substrates_subdir=substrates_subdir)
    Q, olr_achieved = feedstock_db.build_Q_from_hrt_fractions(fs, HRT, V_liq, vs_fractions)
    out = run_simulation(Q, temp_C=temp_C, V_liq=V_liq, sim_days=sim_days)
    out["OLR"] = olr_achieved
    out["OLR_used"] = olr_achieved  # backward compatibility
    return out


def run_from_daily_masses(
    masses_kg_per_day: Dict[str, float],
    *,
    temp_C: float,
    V_liq: float = 100.0,
    sim_days: int = 30,
) -> Dict[str, Any]:
    fs = _get_feedstock(sim_days=sim_days)
    Q = feedstock_db.build_Q_from_daily_masses_kg(fs, masses_kg_per_day, V_liq)
    return run_simulation(Q, temp_C=temp_C, V_liq=V_liq, sim_days=sim_days)
