#!/usr/bin/env python3
"""Create low-TS substrate YAMLs (data/substrates/low_olr/) for OLR 0.5–2.5 sweeps."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feedstock_db

TS_SCALE = 0.55  # ~half TS → lower OLR at same HRT
SCALED_KEYS = ("TS", "NH4", "KS43", "FFS")


def _scale_yaml(text: str, factor: float) -> str:
    out = text
    for key in SCALED_KEYS:
        pat = rf"^({key}:\s+)([\d.]+)"
        out = re.sub(pat, lambda m: f"{m.group(1)}{float(m.group(2)) * factor:.4g}", out, flags=re.M)
    if "name:" in out:
        out = re.sub(r"^name: (.+)$", r"name: \1 (low OLR sweep)", out, flags=re.M)
    return out


def main() -> None:
    src_dir = ROOT / "data" / "substrates"
    dst_dir = src_dir / "low_olr"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for sid in feedstock_db.SUBSTRATE_IDS:
        src = src_dir / f"{sid}.yaml"
        dst = dst_dir / f"{sid}.yaml"
        text = src.read_text(encoding="utf-8")
        header = (
            f"# Low-TS variant for supplementary sweep (TS × {TS_SCALE}).\n"
            f"# Targets OLR 0.5–2.5 kg VS/m³/d with HRT 25–60 d.\n"
        )
        dst.write_text(header + _scale_yaml(text, TS_SCALE), encoding="utf-8")
        print(f"Wrote {dst}")
    print("Done. Run: python scripts/generate_supplementary_sweeps.py --mode low_olr")
