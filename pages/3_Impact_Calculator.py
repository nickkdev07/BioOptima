"""Impact Calculator: financial savings and environmental benefit from CH₄ output."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import energy_utils
from src.ui_helpers import (
    ch4_from_session_state,
    mark_workflow,
    metric_row,
    render_page_shell,
)

st.set_page_config(page_title="Impact Calculator", page_icon="📊", layout="wide")
render_page_shell(page_title="Impact Calculator")

st.title("Impact calculator")
st.markdown("See **financial** and **environmental** impact from your biogas output (illustrative).")

mark_workflow("impact")

default_ch4 = ch4_from_session_state()
has_optimizer_ch4 = default_ch4 is not None

if has_optimizer_ch4:
    st.info(f"Using **{default_ch4:.2f} m³/d CH₄** from your last **Feed Optimizer** run.")
else:
    st.warning(
        "No Feed Optimizer result in this session. "
        "Run the optimizer first for a linked CH₄ value, or adjust the slider below."
    )
    st.page_link("pages/1_Feed_Optimizer.py", label="Feed Optimizer")

ch4_default = float(default_ch4) if default_ch4 is not None else 5.0
prev_ch4 = st.session_state.get("_impact_prev_ch4")
ch4 = st.slider("Daily CH₄ production (m³/d)", 0.0, 50.0, min(ch4_default, 50.0), 0.5, key="impact_ch4_slider")
if prev_ch4 is not None and abs(ch4 - prev_ch4) > 0.01:
    delta_m3 = ch4 - prev_ch4
    st.metric(
        "CH₄ rate",
        f"{ch4:.2f} m³/d",
        delta=f"{delta_m3:+.2f} m³/d vs last adjustment",
        delta_color="normal" if delta_m3 >= 0 else "inverse",
    )
st.session_state["_impact_prev_ch4"] = ch4

monthly_inr = energy_utils.monthly_savings(ch4)
co2_year = energy_utils.co2_avoided_from_ch4_m3(ch4) * 365.0
trees = energy_utils.trees_equivalent(co2_year, years=1.0)

st.success(
    f"**At a glance:** ~**₹{monthly_inr:,.0f}/month** in LPG savings and "
    f"~**{co2_year:,.0f} kg CO₂-eq/year** avoided (illustrative)."
)

tab_fin, tab_env = st.tabs(["Financial", "Environmental"])

with tab_fin:
    st.markdown("Rough **rupee savings** vs LPG and simple **payback** on plant cost.")

    capital = st.number_input("Plant capital cost (₹)", min_value=0, value=500_000, step=25_000)
    inr_cyl = st.number_input(
        "LPG cylinder price (₹ / 14.2 kg)",
        min_value=1.0, value=float(energy_utils.INR_PER_CYLINDER), step=25.0,
    )
    monthly_opex = st.number_input("Monthly operating cost (₹)", min_value=0.0, value=2000.0, step=100.0)
    fertilizer_on = st.checkbox("Include digestate / fertilizer revenue", value=True)
    fertilizer_monthly = st.number_input(
        "Monthly fertilizer revenue (₹)", min_value=0.0, value=500.0, step=50.0, disabled=not fertilizer_on,
    )

    daily_inr = energy_utils.inr_savings_from_ch4_m3(ch4, inr_per_cylinder=inr_cyl)
    monthly_rev = energy_utils.monthly_savings(ch4, inr_per_cylinder=inr_cyl)
    annual_rev = energy_utils.annual_savings(ch4, inr_per_cylinder=inr_cyl)
    fert = fertilizer_monthly if fertilizer_on else 0.0
    monthly_net = monthly_rev - monthly_opex + fert
    annual_net = monthly_net * 12.0
    cyl_d = energy_utils.lpg_cylinders_from_ch4_m3(ch4)

    metric_row([
        {"label": "LPG equivalent", "value": f"{cyl_d:.2f} cyl./day"},
        {"label": "Value (approx.)", "value": f"₹{daily_inr:,.0f}/day"},
        {"label": "Monthly savings (gas)", "value": f"₹{monthly_rev:,.0f}"},
        {"label": "Yearly savings (gas)", "value": f"₹{annual_rev:,.0f}"},
    ])
    metric_row([
        {"label": "Monthly net (after OpEx ± fertilizer)", "value": f"₹{monthly_net:,.0f}"},
        {"label": "Annual net", "value": f"₹{annual_net:,.0f}"},
    ])

    if annual_net > 0 and capital > 0:
        payback_years = float(capital) / annual_net
        st.metric("Simple payback (years)", f"{payback_years:.1f}",
                  help=f"About {payback_years * 12:.0f} months at constant net cash flow.")
    elif capital <= 0:
        st.caption("Enter a positive capital cost to estimate payback.")
    else:
        st.warning("Annual net cash flow is not positive — payback undefined with these inputs.")

    fig_bar = go.Figure(
        data=[
            go.Bar(
                name="Monthly breakdown (₹)",
                x=["Gas revenue", "Operating cost", "Fertilizer", "Net"],
                y=[monthly_rev, -monthly_opex, fert, monthly_net],
                marker_color=["#2d6a4f", "#bc6c25", "#40916c", "#1d3557"],
            )
        ]
    )
    fig_bar.update_layout(
        title="Monthly cash flow (illustrative)",
        yaxis_title="₹",
        height=360,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    months = 60
    cum = []
    balance = -float(capital)
    for m in range(1, months + 1):
        balance += monthly_net
        cum.append({"month": m, "cumulative_inr": balance})
    cdf = pd.DataFrame(cum)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["month"], y=cdf["cumulative_inr"], mode="lines",
                             line=dict(color="#2d6a4f"), name="Cumulative cash"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven")
    fig.update_layout(title="Cumulative cash position (60 months)", xaxis_title="Month", yaxis_title="₹", height=420)
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    balance_y = -float(capital)
    for y in range(1, 6):
        rev_y = monthly_rev * 12
        opex_y = monthly_opex * 12
        fert_y = fert * 12 if fertilizer_on else 0.0
        net_y = rev_y - opex_y + fert_y
        balance_y += net_y
        rows.append({
            "Year": y, "Gas revenue (₹)": round(rev_y, 0), "OpEx (₹)": round(opex_y, 0),
            "Fertilizer (₹)": round(fert_y, 0), "Net (₹)": round(net_y, 0),
            "Cumulative (₹)": round(balance_y, 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Figures are order-of-magnitude; real projects need local quotes.")

with tab_env:
    st.markdown("Illustrative **climate and fuel substitution** from the same CH₄ rate.")

    scale_label = st.selectbox("Show impact for", ("1 month", "6 months", "1 year", "5 years"))
    mult = {"1 month": 30, "6 months": 183, "1 year": 365, "5 years": 365 * 5}[scale_label]

    co2_per_day = energy_utils.co2_avoided_from_ch4_m3(ch4)
    co2_period = co2_per_day * float(mult)
    lpg_kg = energy_utils.lpg_kg_from_ch4_m3(ch4) * float(mult)

    ctx = st.session_state.get("opt_context") or {}
    masses = ctx.get("masses_kg") or {}
    waste_tonnes = sum(float(v) for v in masses.values()) * float(mult) / 1000.0 if masses else None

    metric_row([
        {"label": f"CO₂ avoided ({scale_label})", "value": f"{co2_period:,.0f} kg"},
        {"label": "LPG displaced (approx.)", "value": f"{lpg_kg:,.0f} kg"},
        {"label": "Trees equivalent (1 yr uptake)", "value": f"{trees:.1f} trees"},
    ])
    if waste_tonnes is not None and waste_tonnes > 0:
        metric_row([{
            "label": "Organic feed processed",
            "value": f"{waste_tonnes:,.1f} t",
            "help": "From last Feed Optimizer daily masses × selected period.",
        }])

    env_fig = go.Figure(
        data=[
            go.Bar(
                x=["CO₂ avoided (1 yr)", "Trees equivalent"],
                y=[co2_year, trees * 21.0],
                marker_color=["#2d6a4f", "#40916c"],
                text=[f"{co2_year:,.0f} kg", f"{trees:.0f} trees"],
                textposition="outside",
            )
        ]
    )
    env_fig.update_layout(
        title="Environmental indicators (illustrative scale)",
        yaxis_title="kg CO₂-eq / tree-proxy units",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(env_fig, use_container_width=True)

    gauge_max = max(5000.0, co2_year * 1.2)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=co2_year,
        title={"text": "CO₂ avoided per year (kg CO₂-eq)"},
        gauge={
            "axis": {"range": [0, gauge_max]}, "bar": {"color": "#2d6a4f"},
            "steps": [
                {"range": [0, gauge_max * 0.33], "color": "#d8f5a2"},
                {"range": [gauge_max * 0.33, gauge_max * 0.66], "color": "#fff3bf"},
                {"range": [gauge_max * 0.66, gauge_max], "color": "#ffccd5"},
            ],
        },
    ))
    fig_g.update_layout(height=320)
    st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("SDG alignment (illustrative)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**SDG 7 — Affordable energy**")
        st.caption("Biogas can substitute LPG or firewood for cooking.")
    with c2:
        st.markdown("**SDG 12 — Responsible consumption**")
        st.caption("Diverts organic residues toward productive use.")
    with c3:
        st.markdown("**SDG 13 — Climate action**")
        st.caption("Methane capture and fossil substitution reduce GHG intensity.")

    share = (
        f"My biogas plant model suggests about {co2_year:,.0f} kg CO₂-equivalent avoided per year "
        f"(illustrative), similar to ~{trees:.0f} mature trees over one year, "
        f"from about {ch4:.1f} m³ methane per day."
    )
    st.text_area("Share your impact (copy text)", share, height=100)
    st.caption("Educational approximations — not certified carbon credits.")
