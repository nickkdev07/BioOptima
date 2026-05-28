"""
Latin Hypercube + Dirichlet sampling; parallel PyADM1ODE runs for training CSV.

Sampling strategy (HRT-first): sample temperature and HRT, build flows so HRT is
exact, derive OLR from mix composition. Reject/resample until OLR is in [0.5, 8.0]
to align with UI slider range.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pyDOE2 import lhs
except ImportError:  # pragma: no cover
    lhs = None  # type: ignore[misc, assignment]

# Achieved OLR band aligned with Streamlit sliders (0.5–6.0) plus modest margin
OLR_MIN = 0.5
OLR_MAX = 8.0

DIRICHLET_ALPHA = np.array([1.5, 1.0, 1.0, 1.2, 0.7, 0.8], dtype=float)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_path() -> None:
    r = _repo_root()
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))


def _sample_params(
    n_samples: int,
    seed: int,
    temp_range: Tuple[float, float] = (25.0, 45.0),
    hrt_range: Tuple[float, float] = (10.0, 50.0),
    olr_min: float = OLR_MIN,
    olr_max: float = OLR_MAX,
    max_attempts_per_sample: int = 200,
    dirichlet_alpha: Optional[np.ndarray] = None,
    substrates_subdir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    HRT-first LHS: sample temp + HRT, Dirichlet mix; estimate OLR via feedstock
    metadata proxy and accept only if OLR in [olr_min, olr_max].
    """
    _ensure_project_path()
    from src import feedstock_db

    rng = np.random.default_rng(seed)
    alpha = dirichlet_alpha if dirichlet_alpha is not None else DIRICHLET_ALPHA
    if lhs is None:
        u = rng.random((n_samples * 3, 2))
    else:
        try:
            u = lhs(2, samples=n_samples * 3, random_state=seed)
        except TypeError:
            u = lhs(2, samples=n_samples * 3)

    t_lo, t_hi = temp_range
    h_lo, h_hi = hrt_range
    rows: List[Dict[str, Any]] = []
    ui = 0
    attempts = 0
    max_total = n_samples * max_attempts_per_sample

    # Lightweight OLR estimate without loading PyADM1ODE (for rejection sampling)
    root = _repo_root()
    try:
        from pyadm1 import Feedstock

        paths = feedstock_db.substrate_yaml_paths(root, subdir=substrates_subdir)
        fs_est = Feedstock(paths, feeding_freq=24, total_simtime=60, simba_q_convention=False)
        use_fs = True
    except Exception:  # noqa: BLE001
        use_fs = False
        fs_est = None

    V_liq_est = 100.0

    while len(rows) < n_samples and attempts < max_total:
        attempts += 1
        temp = t_lo + float(u[ui % len(u), 0]) * (t_hi - t_lo)
        hrt = h_lo + float(u[ui % len(u), 1]) * (h_hi - h_lo)
        ui += 1
        frac = rng.dirichlet(alpha)

        if use_fs and fs_est is not None:
            _, olr_est = feedstock_db.build_Q_from_hrt_fractions(fs_est, hrt, V_liq_est, frac)
        else:
            # Fallback: BMP-weighted VS proxy
            bmp = np.array([feedstock_db.FEEDSTOCK_META[s]["bmp"] for s in feedstock_db.SUBSTRATE_IDS])
            vs_proxy = float(frac @ bmp) / 250.0
            olr_est = (V_liq_est / hrt) * vs_proxy * 0.15

        if olr_est < olr_min or olr_est > olr_max:
            if ui >= len(u):
                u = rng.random((max(n_samples, 100), 2))
            continue

        rows.append(
            {
                "idx": len(rows),
                "cow_frac": float(frac[0]),
                "rice_frac": float(frac[1]),
                "wheat_frac": float(frac[2]),
                "food_frac": float(frac[3]),
                "press_frac": float(frac[4]),
                "poultry_frac": float(frac[5]),
                "temperature_C": temp,
                "HRT_days": hrt,
            }
        )

    if len(rows) < n_samples:
        raise RuntimeError(
            f"Only generated {len(rows)}/{n_samples} samples within OLR [{olr_min}, {olr_max}]. "
            "Widen OLR bounds or increase max_attempts_per_sample."
        )
    return rows


def _run_one(
    params: Dict[str, Any],
    sim_days: int,
    V_liq: float,
    *,
    substrates_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Worker: HRT-first simulation."""
    _ensure_project_path()
    from src import health_utils, simulator

    vf = [
        params["cow_frac"],
        params["rice_frac"],
        params["wheat_frac"],
        params["food_frac"],
        params["press_frac"],
        params["poultry_frac"],
    ]
    out = simulator.run_from_hrt_mix(
        vf,
        temp_C=params["temperature_C"],
        HRT=params["HRT_days"],
        V_liq=V_liq,
        sim_days=sim_days,
        substrates_subdir=substrates_subdir,
    )
    row = dict(params)
    olr_achieved = float(out.get("OLR", out.get("OLR_used", 0.0)))
    row["OLR"] = olr_achieved
    row["OLR_used"] = olr_achieved
    row["converged"] = bool(out.get("converged", False))
    row["q_ch4_avg"] = float(out.get("q_ch4_avg", 0.0))
    row["pH_final"] = float(out.get("pH_final", 7.0))
    row["VFA_final"] = float(out.get("VFA_final", 0.0))
    row["S_nh3_final"] = float(out.get("S_nh3_final", 0.0))
    row["vfa_slope_last5d"] = float(out.get("vfa_slope_last5d", 0.0))
    row["ph_slope_last5d"] = float(out.get("ph_slope_last5d", 0.0))
    row["first_instability_day"] = int(out.get("first_instability_day", -1))
    row["sim_error"] = out.get("error")
    if row["converged"]:
        row["stability_label"] = health_utils.classify_stability(row["pH_final"], row["VFA_final"])
    else:
        row["stability_label"] = -1
    return row


def generate_dataset(
    n_samples: int,
    out_csv: Path,
    *,
    sim_days: int = 30,
    V_liq: float = 100.0,
    seed: int = 42,
    workers: Optional[int] = None,
    chunk_flush: int = 50,
    progress_bar: bool = False,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    temp_range: Tuple[float, float] = (25.0, 45.0),
    hrt_range: Tuple[float, float] = (10.0, 50.0),
    olr_min: float = OLR_MIN,
    olr_max: float = OLR_MAX,
    substrates_subdir: Optional[str] = None,
    dirichlet_alpha: Optional[np.ndarray] = None,
) -> Path:
    _ensure_project_path()
    from src import feedstock_db

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feedstock_db.export_feedstock_library_json(_repo_root() / "data" / "feedstock_library.json")
    params_list = _sample_params(
        n_samples,
        seed,
        temp_range=temp_range,
        hrt_range=hrt_range,
        olr_min=olr_min,
        olr_max=olr_max,
        dirichlet_alpha=dirichlet_alpha,
        substrates_subdir=substrates_subdir,
    )
    n_workers = workers or max(1, (os.cpu_count() or 4) - 1)
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    def _hook(done: int, row: Dict[str, Any]) -> None:
        if progress_hook is None:
            return
        progress_hook(
            {
                "done": done,
                "total": n_samples,
                "row": row,
                "elapsed_s": float(time.perf_counter() - t0),
            }
        )

    if n_workers <= 1:
        seq = params_list
        if progress_bar:
            try:
                from tqdm.auto import tqdm

                seq = tqdm(params_list, desc="ADM1 sweep", unit="sim")  # type: ignore[assignment]
            except ImportError:
                pass
        for p in seq:
            rows.append(_run_one(p, sim_days, V_liq, substrates_subdir=substrates_subdir))
            _hook(len(rows), rows[-1])
            if len(rows) % chunk_flush == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(_run_one, p, sim_days, V_liq, substrates_subdir=substrates_subdir): p["idx"]
                for p in params_list
            }
            tmp: Dict[int, Dict[str, Any]] = {}
            done = 0
            fut_iter = as_completed(futs)
            if progress_bar:
                try:
                    from tqdm.auto import tqdm

                    fut_iter = tqdm(fut_iter, total=n_samples, desc="ADM1 sweep", unit="sim")  # type: ignore[assignment]
                except ImportError:
                    pass
            for fut in fut_iter:
                r = fut.result()
                tmp[int(r["idx"])] = r
                done += 1
                _hook(done, r)
            rows = [tmp[i] for i in range(n_samples)]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="BioOptima LHS sweep for training data (HRT-first)")
    ap.add_argument("--n", type=int, default=3000, help="number of samples")
    ap.add_argument("--out", type=str, default="data/generated/biogas_training_data.csv")
    ap.add_argument("--sim-days", type=int, default=30)
    ap.add_argument("--V-liq", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--progress", action="store_true", help="show tqdm progress bar")
    args = ap.parse_args()
    root = _repo_root()
    out = root / args.out
    generate_dataset(
        args.n,
        out,
        sim_days=args.sim_days,
        V_liq=args.V_liq,
        seed=args.seed,
        workers=args.workers,
        progress_bar=args.progress,
    )
    print(f"Wrote {out} ({args.n} rows)")


if __name__ == "__main__":
    main()
