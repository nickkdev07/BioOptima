"""Live PyADM1ODE digester simulation — time-series charts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import health_utils, simulator
from src.ui_helpers import (
    DISPLAY_NAMES,
    FEEDSTOCK_HELP,
    FEEDSTOCK_KEYS,
    get_feed_masses_from_session,
    init_session_defaults,
    metric_row,
    render_page_shell,
)

st.set_page_config(page_title="Live Simulator", page_icon="🔬", layout="wide")
render_page_shell(page_title="Live Simulator")

st.title("Live digester simulator")
st.markdown(
    "Runs the **full ADM1** model (PyADM1ODE). This can take **15–60 seconds** depending on your machine."
)
st.warning("First run may be slower while the model initializes.")
st.caption("**Physics deep-dive** — optional ADM1 validation after using the Feed Optimizer.")

init_session_defaults()
ctx = st.session_state.get("opt_context") or {}
if ctx.get("masses_kg") and st.button("Load feed from last optimizer run", type="secondary"):
    for k in FEEDSTOCK_KEYS:
        st.session_state[f"sim_{k}"] = float(ctx["masses_kg"].get(k, 0.0))
    if "temp_C" in ctx:
        st.session_state["sim_temp"] = float(ctx["temp_C"])
    st.toast("Loaded optimizer feed masses.", icon="🌾")
    st.rerun()

for k in FEEDSTOCK_KEYS:
    sk = f"sim_{k}"
    if sk not in st.session_state:
        st.session_state[sk] = float(get_feed_masses_from_session().get(k, 200.0 if k == "cow_dung" else 0.0))
if "sim_temp" not in st.session_state:
    st.session_state["sim_temp"] = 35.0
if "sim_V_liq" not in st.session_state:
    st.session_state["sim_V_liq"] = 100.0
if "sim_days" not in st.session_state:
    st.session_state["sim_days"] = 30

st.subheader("Feed (kg fresh matter per day)")
col1, col2, col3 = st.columns(3)
masses: dict[str, float] = {}
with col1:
    masses["cow_dung"] = st.slider(
        DISPLAY_NAMES["cow_dung"], 0.0, 2000.0, key="sim_cow_dung", step=10.0, help=FEEDSTOCK_HELP["cow_dung"],
    )
    masses["rice_straw"] = st.slider(
        DISPLAY_NAMES["rice_straw"], 0.0, 1500.0, key="sim_rice_straw", step=10.0, help=FEEDSTOCK_HELP["rice_straw"],
    )
with col2:
    masses["wheat_straw"] = st.slider(
        DISPLAY_NAMES["wheat_straw"], 0.0, 1500.0, key="sim_wheat_straw", step=10.0, help=FEEDSTOCK_HELP["wheat_straw"],
    )
    masses["food_waste"] = st.slider(
        DISPLAY_NAMES["food_waste"], 0.0, 800.0, key="sim_food_waste", step=5.0, help=FEEDSTOCK_HELP["food_waste"],
    )
with col3:
    masses["press_mud"] = st.slider(
        DISPLAY_NAMES["press_mud"], 0.0, 1500.0, key="sim_press_mud", step=5.0, help=FEEDSTOCK_HELP["press_mud"],
    )
    masses["poultry_litter"] = st.slider(
        DISPLAY_NAMES["poultry_litter"], 0.0, 1500.0, key="sim_poultry_litter", step=5.0,
        help=FEEDSTOCK_HELP["poultry_litter"],
    )

temp = st.slider("Digester temperature (°C)", 25.0, 42.0, key="sim_temp")

with st.expander("Reactor and simulation length", expanded=False):
    V_liq = st.slider("Liquid volume V_liq (m³)", 10.0, 200.0, key="sim_V_liq", step=5.0)
    sim_days = st.slider("Simulation length (days)", 14, 60, key="sim_days", step=1)

if st.button("Run simulation", type="primary"):
    total_m = sum(masses.get(k, 0.0) for k in FEEDSTOCK_KEYS)
    if total_m <= 0:
        st.error("Set at least one positive feed rate.")
    else:
        progress = st.progress(0, text="Building feed map…")
        progress.progress(25, text="Integrating ADM1 ODEs…")
        with st.spinner("Running ADM1… please wait."):
            payload = {k: masses[k] for k in FEEDSTOCK_KEYS}
            out = simulator.run_from_daily_masses(
                payload, temp_C=float(temp), V_liq=float(V_liq), sim_days=int(sim_days),
            )
        progress.progress(90, text="Extracting time series…")
        progress.progress(100, text="Done")
        st.session_state["sim_result"] = out
        st.session_state["sim_input"] = {
            "masses_kg": dict(payload), "temp_C": float(temp),
            "V_liq": float(V_liq), "sim_days": int(sim_days),
        }
        st.toast("Simulation complete.", icon="🔬")

res = st.session_state.get("sim_result")
if isinstance(res, dict) and res:
    if res.get("error"):
        st.error(f"Simulation error: **{res['error']}**")
    else:
        ts = res.get("time_series")
        if ts is None or getattr(ts, "empty", True):
            st.warning("No time series returned.")
        else:
            if not res.get("converged", True):
                st.warning("Run finished with **converged = False** — interpret curves cautiously.")
            fid = int(res.get("first_instability_day", -1))
            sim_d = int((st.session_state.get("sim_input") or {}).get("sim_days", len(ts)))
            tail = min(7, len(ts))
            q_avg = float(ts["q_ch4"].iloc[-tail:].mean())
            ph_f = float(ts["pH"].iloc[-1])
            vfa_f = float(ts["VFA"].iloc[-1])
            stab = health_utils.classify_stability(ph_f, vfa_f)
            stab_name = health_utils.stability_name(stab)

            st.info(
                f"**Run summary:** {sim_d} days simulated · avg CH₄ (last week) **{q_avg:.2f} m³/d** · "
                f"final pH **{ph_f:.2f}** · final VFA **{vfa_f:.2f}** kg/m³ · status **{stab_name}**"
            )
            if fid >= 0:
                st.warning(f"**Instability band** first crossed near **day {fid}** (pH < 6.5 or high VFA).")

            tcol = "time_d" if "time_d" in ts.columns else ts.columns[0]
            t_arr = ts[tcol].values
            n = len(t_arr)
            t_startup_end = float(t_arr[int(n * 0.2)]) if n > 5 else float(t_arr[-1]) * 0.2
            t_steady_end = float(t_arr[int(n * 0.7)]) if n > 5 else float(t_arr[-1]) * 0.7

            def _add_phases(fig: go.Figure, *, rows: int = 1) -> None:
                for row in range(1, rows + 1):
                    fig.add_vrect(x0=0, x1=t_startup_end, fillcolor="#d8f5a2", opacity=0.15, line_width=0, row=row, col=1)
                    fig.add_vrect(
                        x0=t_startup_end, x1=t_steady_end, fillcolor="#b7e4c7", opacity=0.12, line_width=0,
                        row=row, col=1,
                    )
                    if fid >= 0:
                        fig.add_vrect(
                            x0=float(fid), x1=float(t_arr[-1]),
                            fillcolor="#ffccd5", opacity=0.2, line_width=0, row=row, col=1,
                        )

            tab_ch4, tab_ph_vfa, tab_nh3 = st.tabs(["CH₄ flow", "pH & VFA", "Free NH₃"])

            with tab_ch4:
                fig_ch4 = go.Figure()
                fig_ch4.add_trace(go.Scatter(x=ts[tcol], y=ts["q_ch4"], line=dict(color="#2d6a4f"), name="CH₄"))
                fig_ch4.update_layout(title="CH₄ flow (m³/d)", xaxis_title="Time (days)", height=380)
                st.plotly_chart(fig_ch4, use_container_width=True)

            with tab_ph_vfa:
                fig_pv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                       subplot_titles=("pH", "VFA (kg HAc-eq/m³)"))
                fig_pv.add_trace(go.Scatter(x=ts[tcol], y=ts["pH"], line=dict(color="#1d3557")), row=1, col=1)
                fig_pv.add_trace(go.Scatter(x=ts[tcol], y=ts["VFA"], line=dict(color="#bc6c25")), row=2, col=1)
                _add_phases(fig_pv, rows=2)
                fig_pv.update_xaxes(title_text="Time (days)", row=2, col=1)
                fig_pv.update_layout(height=480, showlegend=False)
                st.plotly_chart(fig_pv, use_container_width=True)

            with tab_nh3:
                fig_nh3 = go.Figure()
                fig_nh3.add_trace(go.Scatter(x=ts[tcol], y=ts["S_nh3"], line=dict(color="#6a4c93"), name="NH₃"))
                fig_nh3.update_layout(title="Free NH₃ (kg/m³)", xaxis_title="Time (days)", height=380)
                st.plotly_chart(fig_nh3, use_container_width=True)

            st.caption("Green tint = startup; mid green = steady-state; pink = stress band (if detected).")

            metric_row([
                {"label": "Avg CH₄ (last week)", "value": f"{q_avg:.2f} m³/d"},
                {"label": "Final pH", "value": f"{ph_f:.2f}"},
                {"label": "Final VFA", "value": f"{vfa_f:.2f} kg/m³"},
                {"label": "Stability (pH/VFA)", "value": stab_name},
            ])

            with st.expander("Inputs used for this run"):
                st.json(st.session_state.get("sim_input") or {"masses_kg": masses})
else:
    st.info("Adjust feeds and press **Run simulation** to see curves.")
