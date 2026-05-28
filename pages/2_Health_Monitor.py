import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import health_utils, optimizer
from src.model_loader import load_models
from src.calibration import UI_HRT_MAX, UI_HRT_MIN, UI_OLR_MAX, UI_OLR_MIN
from src.ui_helpers import (
    HEALTH_PRESETS,
    apply_health_preset,
    digester_schematic_figure,
    mark_workflow,
    metric_row,
    reading_zone_figure,
    render_page_shell,
    status_banner,
)

st.set_page_config(page_title="Health Monitor", page_icon="❤️", layout="wide")
render_page_shell(page_title="Health Monitor")

st.title("Digester health")
st.markdown("Use **lab readings** if you have them, or **quick symptoms** for guidance.")

ym, hm = load_models()
if ym is None:
    st.error("Trained models are missing. Follow the README first.")
    st.stop()

mark_workflow("health")

with st.expander("Load example readings", expanded=False):
    hp1, hp2, hp3, hp4 = st.columns(4)
    for col, key in zip((hp1, hp2, hp3, hp4), ("stable", "stressed", "stable_quick", "stressed_quick")):
        with col:
            if st.button(HEALTH_PRESETS[key]["label"], use_container_width=True, key=f"hp_{key}"):
                apply_health_preset(key)
                st.rerun()

vfa_slope = 0.0
ph_slope = 0.0

mode = st.radio(
    "How do you want to input?",
    ("Lab readings (advanced)", "Quick symptoms"),
    horizontal=True,
    key="health_mode",
)

if mode.startswith("Lab"):
    st.subheader("Process readings")
    c1, c2, c3 = st.columns(3)
    with c1:
        pH = st.slider("pH", 5.0, 8.5, key="health_pH", step=0.05)
    with c2:
        vfa = st.slider("VFA (kg HAc-eq/m³)", 0.0, 5.0, key="health_vfa", step=0.05)
    with c3:
        nh3 = st.slider("Free NH₃ (kg/m³)", 0.0, 0.15, key="health_nh3", step=0.005)
    st.subheader("Operating parameters (for CH₄ estimate)")
    c4, c5, c6 = st.columns(3)
    with c4:
        temp = st.slider("Temperature (°C)", 20.0, 45.0, 35.0)
    with c5:
        olr = st.slider("OLR (kg VS/m³/d)", UI_OLR_MIN, UI_OLR_MAX, 2.5, 0.1)
    with c6:
        hrt = st.slider("HRT (days)", UI_HRT_MIN, UI_HRT_MAX, 25.0)
    mf = [1 / 6] * 6
    with st.expander("Optional: recent trends (for early-warning rules)", expanded=False):
        vfa_slope = st.slider(
            "VFA trend (kg HAc-eq/m³ per day)", -0.2, 0.3, key="health_vfa_slope", step=0.01,
        )
        ph_slope = st.slider(
            "pH trend (units per day)", -0.15, 0.1, key="health_ph_slope", step=0.005,
        )
else:
    gas_drop = st.checkbox("Gas production has dropped recently", key="health_gas_drop")
    bad_smell = st.checkbox("Bad or sour smell from the digester", key="health_bad_smell")
    foam = st.checkbox("Foam or unusually thick digestate", key="health_foam")
    feed_change = st.checkbox("Large change in feed recipe in the last few days", key="health_feed_change")
    season_temp = st.radio("Season (ambient)", ("cold", "warm", "hot"), horizontal=True, key="health_season")
    rd = health_utils.infer_readings_from_quick_mode(gas_drop, bad_smell, foam, feed_change, season_temp)
    pH, vfa, nh3 = rd["pH"], rd["VFA"], rd["S_nh3"]
    temp, olr, hrt = 35.0, 2.5, 25.0
    mf = [1 / 6] * 6
    if gas_drop or feed_change:
        vfa_slope = 0.05
    if gas_drop:
        ph_slope = -0.03

    st.subheader("Inferred lab-equivalent readings")
    metric_row([
        {"label": "pH (estimated)", "value": f"{pH:.2f}"},
        {"label": "VFA (estimated)", "value": f"{vfa:.2f} kg/m³"},
        {"label": "Free NH₃ (estimated)", "value": f"{nh3:.3f} kg/m³"},
    ])
    st.caption("Values above are derived from your symptom checkboxes — switch to Lab mode for real measurements.")

row = {
    "cow_frac": mf[0], "rice_frac": mf[1], "wheat_frac": mf[2],
    "food_frac": mf[3], "press_frac": mf[4], "poultry_frac": mf[5],
    "temperature_C": temp, "HRT_days": hrt, "OLR": olr,
}
yhat, hhat = optimizer.predict_yield_health(row, yield_model=ym, health_model=hm)
stab = health_utils.classify_stability(pH, vfa)
score = health_utils.health_score_from_readings(pH, vfa, nh3, health_class=stab)
ew = health_utils.early_warning(stab, vfa_slope, ph_slope, vfa, pH)
hl = health_utils.stability_name(stab)

st.divider()
schem_col, banner_col = st.columns([1, 2])
with schem_col:
    st.plotly_chart(digester_schematic_figure(stab, title=f"Tank — {hl}"), use_container_width=True)
with banner_col:
    status_banner(hl)
    metric_row([
        {"label": "Health score (0–100)", "value": f"{score:.0f}"},
        {"label": "Predicted CH₄ (operating point)", "value": f"{yhat:.2f} m³/d"},
        {
            "label": "Stability from OLR/HRT/mix",
            "value": health_utils.stability_name(hhat),
            "help": "From operating parameters — not the lab reading classifier.",
        },
    ])

st.subheader("Reading vs safe bands")
st.plotly_chart(reading_zone_figure(pH, vfa, nh3), use_container_width=True)

st.subheader("Early warning (rules)")
if ew["level"] == "Critical" or ew["risk_score"] >= 70:
    st.error(f"**{ew['level']}** — risk score **{ew['risk_score']}** / 100")
    if stab >= 1:
        st.markdown("**Suggested action:** Reduce OLR, dilute feed, or pause high-VFA substrates until pH recovers.")
else:
    st.warning(f"**{ew['level']}** — risk score **{ew['risk_score']}** / 100")
for w in ew["warnings"]:
    st.markdown(f"- {w}")
