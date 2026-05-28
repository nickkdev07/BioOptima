import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import energy_utils, optimizer
from src.calibration import UI_HRT_MAX, UI_HRT_MIN, UI_OLR_MAX, UI_OLR_MIN, UI_TEMP_MAX, UI_TEMP_MIN
from src.model_loader import load_models
from src.ui_helpers import (
    DEMO_PRESETS,
    DISPLAY_NAMES,
    FEEDSTOCK_HELP,
    FEEDSTOCK_KEYS,
    apply_feed_preset,
    ch4_gauge_figure,
    get_feed_masses_from_session,
    init_session_defaults,
    instant_baseline_ch4,
    mark_workflow,
    metric_row,
    mix_comparison_table_figure,
    mix_labels_for_keys,
    mix_pie_figure,
    render_page_shell,
    status_banner,
    warn_if_out_of_training_range,
)

st.set_page_config(page_title="Feed Optimizer", page_icon="🌾", layout="wide")
render_page_shell(page_title="Feed Optimizer")

st.title("Feed Optimizer")
st.markdown(
    "Enter **how much of each waste you have per day** (kg/d), set digester conditions, "
    "then get the **best co-digestion mix** and methane forecast."
)

ym, hm = load_models()
if ym is None:
    st.error("Trained models are missing. Follow the README to build `models/*.pkl`.")
    st.stop()

init_session_defaults()

preset_cols = st.columns(len(DEMO_PRESETS))
for col, (key, preset) in zip(preset_cols, DEMO_PRESETS.items()):
    with col:
        if st.button(preset["label"], use_container_width=True, key=f"preset_{key}"):
            apply_feed_preset(key)
            st.rerun()

input_col, live_col = st.columns([2, 1])

with input_col:
    st.subheader("1. Available feed (kg/day)")
    col1, col2, col3 = st.columns(3)
    with col1:
        cow = st.number_input(
            DISPLAY_NAMES["cow_dung"] + " (kg/d)",
            0.0, 5000.0, key="feed_cow_dung", step=10.0, help=FEEDSTOCK_HELP["cow_dung"],
        )
        rice = st.number_input(
            DISPLAY_NAMES["rice_straw"] + " (kg/d)",
            0.0, 2000.0, key="feed_rice_straw", step=10.0, help=FEEDSTOCK_HELP["rice_straw"],
        )
    with col2:
        wheat = st.number_input(
            DISPLAY_NAMES["wheat_straw"] + " (kg/d)",
            0.0, 2000.0, key="feed_wheat_straw", step=10.0, help=FEEDSTOCK_HELP["wheat_straw"],
        )
        food = st.number_input(
            DISPLAY_NAMES["food_waste"] + " (kg/d)",
            0.0, 1000.0, key="feed_food_waste", step=5.0, help=FEEDSTOCK_HELP["food_waste"],
        )
    with col3:
        press = st.number_input(
            DISPLAY_NAMES["press_mud"] + " (kg/d)",
            0.0, 2000.0, key="feed_press_mud", step=5.0, help=FEEDSTOCK_HELP["press_mud"],
        )
        poultry = st.number_input(
            DISPLAY_NAMES["poultry_litter"] + " (kg/d)",
            0.0, 2000.0, key="feed_poultry_litter", step=5.0, help=FEEDSTOCK_HELP["poultry_litter"],
        )

    st.subheader("2. Digester conditions")
    temp = st.slider(
        "Digester temperature (°C)",
        UI_TEMP_MIN, UI_TEMP_MAX, key="feed_temp",
        help="Mesophilic digesters are often run near 35 °C.",
    )
    with st.expander("Hydraulic retention and organic loading", expanded=False):
        hrt = st.slider("Hydraulic retention time (days)", UI_HRT_MIN, UI_HRT_MAX, key="feed_hrt")
        olr = st.slider(
            "Organic loading rate (kg VS per m³ reactor per day)",
            UI_OLR_MIN, UI_OLR_MAX, key="feed_olr", step=0.1,
        )
        if olr > 4.0:
            st.warning("High loading — monitor VFA and pH closely.")

    warn_if_out_of_training_range(temperature_C=temp, HRT_days=hrt, OLR=olr)

    if st.button("Find best mix", type="primary"):
        total_m = cow + rice + wheat + food + press + poultry
        if total_m <= 0:
            st.error("Enter at least one positive feed rate.")
        else:
            with st.spinner("Searching for a high-yield, safer mix…"):
                avail = get_feed_masses_from_session()
                res = optimizer.optimize_feedstock(
                    avail, temp, HRT_days=hrt, OLR=olr,
                    yield_model=ym, health_model=hm,
                )
            if res.get("error"):
                st.error(res["error"])
            else:
                st.session_state["opt_result"] = res
                st.session_state["opt_context"] = {
                    "temp_C": temp, "HRT_days": hrt, "OLR": olr, "masses_kg": avail,
                }
                mark_workflow("feed")
                st.toast("Optimization complete!", icon="✅")
                st.success("Updated recommendation — see results below.")

with live_col:
    st.subheader("Live preview")
    avail = get_feed_masses_from_session()
    total_kg = sum(avail.values())
    st.metric("Total feed available", f"{total_kg:.0f} kg/d")
    baseline = instant_baseline_ch4(avail, temp, hrt, olr, yield_model=ym)
    if baseline is not None and total_kg > 0:
        st.metric(
            "Baseline CH₄ (equal mix)",
            f"{baseline:.2f} m³/d",
            help="Surrogate prediction if you used all feedstocks in proportion to availability — before Optuna.",
        )
    elif total_kg <= 0:
        st.caption("Enter at least one positive feed rate to preview yield.")
    preset_name = st.session_state.get("feed_preset_name")
    if preset_name and preset_name in DEMO_PRESETS:
        st.caption(f"Preset: **{DEMO_PRESETS[preset_name]['label']}**")

res = st.session_state.get("opt_result")
ctx = st.session_state.get("opt_context") or {}
if res and not res.get("error"):
    st.divider()
    st.subheader("Results")
    status_banner(res["health_label"])

    labels = mix_labels_for_keys(FEEDSTOCK_KEYS)
    vals = [res["mix"][k] * 100.0 for k in FEEDSTOCK_KEYS]
    daily_ch4 = float(res["yield"])
    baseline_y = res.get("baseline_yield")

    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(ch4_gauge_figure(daily_ch4), use_container_width=True)
    with g2:
        metric_row([
            {"label": "Predicted CH₄", "value": f"{daily_ch4:.2f} m³/d",
             "delta": f"{res['improvement_pct']:+.1f}% vs availability mix"},
            {"label": "C/N (proxy)", "value": f"{res['cn_ratio']:.1f}"},
        ])
        if baseline_y is not None:
            st.caption(f"Availability mix alone would yield about **{float(baseline_y):.2f} m³/d** at the same T/HRT/OLR.")
        cyl_d = energy_utils.lpg_cylinders_from_ch4_m3(daily_ch4)
        inr_d = energy_utils.inr_savings_from_ch4_m3(daily_ch4)
        metric_row([
            {"label": "LPG equivalent", "value": f"{cyl_d:.2f} cyl./day"},
            {"label": "Rupee value (approx.)", "value": f"₹{inr_d:,.0f}/day"},
        ])

    masses = ctx.get("masses_kg") or {}
    avail_pct = [0.0] * len(FEEDSTOCK_KEYS)
    if masses:
        tot = sum(float(masses.get(k, 0)) for k in FEEDSTOCK_KEYS) or 1.0
        avail_pct = [100.0 * float(masses.get(k, 0)) / tot for k in FEEDSTOCK_KEYS]

    cmp_rows = []
    for i, k in enumerate(FEEDSTOCK_KEYS):
        if avail_pct[i] >= 0.5 or vals[i] >= 0.5:
            cmp_rows.append({
                "Feedstock": labels[i],
                "Availability %": round(avail_pct[i], 1),
                "Recommended %": round(vals[i], 1),
                "Change": round(vals[i] - avail_pct[i], 1),
            })
    if cmp_rows:
        st.plotly_chart(
            mix_comparison_table_figure(cmp_rows),
            use_container_width=True,
            key="mix_cmp_table",
        )

    st.plotly_chart(
        mix_pie_figure(labels, vals, title="Recommended mix"),
        use_container_width=True,
        key="mix_cmp_pie",
    )

else:
    st.info("Set your daily feed rates and press **Find best mix** to see the recommended co-digestion blend.")
