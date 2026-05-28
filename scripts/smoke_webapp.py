from __future__ import annotations

import json
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("[SMOKE] root:", ROOT)

# Core modules that the Streamlit app relies on
from src import calibration, health_utils, model_loader, optimizer  # noqa: E402

# Verify required artifacts exist
csv_path = ROOT / "data" / "generated" / "biogas_training_merged.csv"
mdir = ROOT / "models"
lit_path = ROOT / "data" / "validation" / "literature_points.csv"

print("[SMOKE] CSV exists:", csv_path.is_file(), csv_path)
print("[SMOKE] Literature CSV exists:", lit_path.is_file(), lit_path)
print("[SMOKE] Models dir exists:", mdir.is_dir(), mdir)
print("[SMOKE] Model files:", sorted([p.name for p in mdir.glob("*") if p.is_file()]))

eval_metrics = json.loads((mdir / "eval_metrics.json").read_text(encoding="utf-8"))
literature_validation = json.loads(
    (mdir / "literature_validation.json").read_text(encoding="utf-8")
)
print("[SMOKE] eval_metrics yield_r2:", eval_metrics.get("yield_r2"))
print("[SMOKE] eval_metrics health_f1_macro:", eval_metrics.get("health_f1_macro"))
print(
    "[SMOKE] literature points:",
    literature_validation.get("n_points_total"),
    "in_range:",
    literature_validation.get("n_points_in_range"),
)

# Load models (used by the web app)
yield_model, health_model = model_loader.load_models()
print("[SMOKE] yield model loaded:", yield_model is not None)
print("[SMOKE] health model loaded:", health_model is not None)

# Run a single optimizer prediction for sanity
# optimizer expects keys: cow_dung, rice_straw, wheat_straw, food_waste, press_mud, poultry_litter
avail = {
    "cow_dung": 100.0,
    "rice_straw": 100.0,
    "wheat_straw": 100.0,
    "food_waste": 100.0,
    "press_mud": 100.0,
    "poultry_litter": 100.0,
}
params = {"temperature_C": 35.0, "HRT_days": 25.0, "OLR": 3.0}

opt = optimizer.optimize_feedstock(
    avail,
    params["temperature_C"],
    HRT_days=params["HRT_days"],
    OLR=params["OLR"],
    yield_model=yield_model,
    health_model=health_model,
    n_trials=50,
)
print("[SMOKE] optimizer keys:", sorted(opt.keys()))
print("[SMOKE] recommended mix sum:", sum(opt["mix"].values()))
print("[SMOKE] predicted yield (m3/d):", opt.get("yield"))
print("[SMOKE] health class:", opt.get("health_label"))

# Health rule-based sanity check
row = {
    "temperature_C": params["temperature_C"],
    "HRT_days": params["HRT_days"],
    "OLR": params["OLR"],
    # uniform fractions across 6 feed types
    "cow_frac": 1.0 / 6,
    "rice_frac": 1.0 / 6,
    "wheat_frac": 1.0 / 6,
    "food_frac": 1.0 / 6,
    "press_frac": 1.0 / 6,
    "poultry_frac": 1.0 / 6,
}

ph_est, vfa_est = calibration.estimate_ph_vfa_from_row(row)
health_label = calibration.predict_health_from_row(row)

print("[SMOKE] estimated pH:", round(ph_est, 3), "estimated VFA:", round(vfa_est, 3))
print(
    "[SMOKE] health label:",
    health_label,
    "name:",
    health_utils.stability_name(health_label),
)

print("[SMOKE] OK: webapp pipeline smoke test passed.")

