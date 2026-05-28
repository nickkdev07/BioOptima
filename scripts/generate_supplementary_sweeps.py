#!/usr/bin/env python3
"""
Generate supplementary ADM1 training CSVs (run locally or on Colab).

Modes:
  low_olr      — diluted substrates, OLR 0.5–2.5, HRT 25–60 d (1200 samples)
  instability  — short HRT + food-heavy mixes, HRT 10–18 d (500 samples)

After Colab runs, merge with:
  python scripts/merge_training_csvs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sweep import generate_dataset


PRESETS = {
    "low_olr": {
        "n": 1200,
        "out": "data/generated/biogas_training_low_olr.csv",
        "seed": 101,
        "temp_range": (25.0, 42.0),
        "hrt_range": (25.0, 60.0),
        "olr_min": 0.5,
        "olr_max": 2.5,
        "substrates_subdir": "low_olr",
        "dirichlet_alpha": np.array([1.5, 1.0, 1.0, 1.2, 0.7, 0.8], dtype=float),
    },
    "instability": {
        "n": 500,
        "out": "data/generated/biogas_training_instability.csv",
        "seed": 202,
        "temp_range": (30.0, 42.0),
        "hrt_range": (10.0, 18.0),
        "olr_min": 0.5,
        "olr_max": 99.0,
        "substrates_subdir": None,
        "dirichlet_alpha": np.array([0.3, 0.3, 0.3, 3.5, 0.6, 0.8], dtype=float),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description="BioOptima supplementary ADM1 sweeps")
    ap.add_argument(
        "--mode",
        choices=list(PRESETS.keys()),
        required=True,
        help="low_olr or instability preset",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--create-low-olr-yaml", action="store_true", help="write low_olr YAMLs first")
    args = ap.parse_args()

    if args.create_low_olr_yaml and args.mode == "low_olr":
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "create_low_olr_substrates.py")], check=True)

    cfg = PRESETS[args.mode]
    out = ROOT / cfg["out"]
    generate_dataset(
        cfg["n"],
        out,
        seed=cfg["seed"],
        workers=args.workers,
        progress_bar=args.progress,
        temp_range=cfg["temp_range"],
        hrt_range=cfg["hrt_range"],
        olr_min=cfg["olr_min"],
        olr_max=cfg["olr_max"],
        substrates_subdir=cfg["substrates_subdir"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
    )
    print(f"Wrote {out} ({cfg['n']} rows, mode={args.mode})")


if __name__ == "__main__":
    main()
