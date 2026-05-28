#!/usr/bin/env python3
"""Merge main + supplementary sweep CSVs into one training file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUTS = [
    "data/generated/biogas_training_data.csv",
    "data/generated/biogas_training_low_olr.csv",
    "data/generated/biogas_training_instability.csv",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated/biogas_training_merged.csv")
    ap.add_argument("--inputs", nargs="*", default=None, help="CSV paths to merge (existing only)")
    args = ap.parse_args()

    paths = [ROOT / p for p in (args.inputs or DEFAULT_INPUTS)]
    frames = []
    for p in paths:
        if p.is_file():
            df = pd.read_csv(p)
            df["sweep_source"] = p.name
            frames.append(df)
            print(f"  + {len(df)} rows from {p.name}")
        else:
            print(f"  skip (missing): {p}")

    if not frames:
        raise SystemExit("No input CSVs found.")

    merged = pd.concat(frames, ignore_index=True)
    if "idx" in merged.columns:
        merged["idx"] = range(len(merged))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {out} ({len(merged)} rows)")
    if "OLR" in merged.columns and "stability_label" in merged.columns:
        conv = merged[merged["converged"] == True] if "converged" in merged.columns else merged
        print(f"  OLR: {conv['OLR'].min():.2f} – {conv['OLR'].max():.2f}")
        print(f"  stability_label: {conv['stability_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
