"""Shared Streamlit UI helpers for BioOptima pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .calibration import UI_HRT_MAX, UI_HRT_MIN, UI_OLR_MAX, UI_OLR_MIN, UI_TEMP_MAX, UI_TEMP_MIN

# Internal API keys -> user-facing labels
DISPLAY_NAMES: Dict[str, str] = {
    "cow_dung": "Cow dung",
    "rice_straw": "Rice straw",
    "wheat_straw": "Wheat straw",
    "food_waste": "Food waste",
    "press_mud": "Press mud",
    "poultry_litter": "Poultry litter",
}

# Slider keys order for Scenario / Feed mix display
FEEDSTOCK_KEYS: Tuple[str, ...] = (
    "cow_dung",
    "rice_straw",
    "wheat_straw",
    "food_waste",
    "press_mud",
    "poultry_litter",
)

FEEDSTOCK_HELP: Dict[str, str] = {
    "cow_dung": "Typical BMP ~150–250 Nm³ CH₄/t VS; C/N ~16–20.",
    "rice_straw": "Lignocellulosic; BMP ~180–280 Nm³/t VS; higher C/N.",
    "wheat_straw": "Similar to rice straw; BMP ~190–290 Nm³/t VS.",
    "food_waste": "High degradability; BMP ~300–450 Nm³/t VS; watch acidification.",
    "press_mud": "Sugar-industry residue; BMP ~200–320 Nm³/t VS.",
    "poultry_litter": "High N; BMP ~200–350 Nm³/t VS; ammonia risk.",
}

# Demo feed presets (kg/d + operating point)
DEMO_PRESETS: Dict[str, Dict[str, Any]] = {
    "punjab_dairy": {
        "label": "Punjab dairy",
        "masses": {
            "cow_dung": 400.0,
            "rice_straw": 50.0,
            "wheat_straw": 30.0,
            "food_waste": 80.0,
            "press_mud": 40.0,
            "poultry_litter": 0.0,
        },
        "temp_C": 35.0,
        "HRT_days": 25.0,
        "OLR": 2.5,
    },
    "straw_rich": {
        "label": "Straw-heavy",
        "masses": {
            "cow_dung": 150.0,
            "rice_straw": 200.0,
            "wheat_straw": 150.0,
            "food_waste": 30.0,
            "press_mud": 0.0,
            "poultry_litter": 0.0,
        },
        "temp_C": 35.0,
        "HRT_days": 30.0,
        "OLR": 2.0,
    },
    "food_waste_heavy": {
        "label": "Food-waste mix",
        "masses": {
            "cow_dung": 100.0,
            "rice_straw": 20.0,
            "wheat_straw": 0.0,
            "food_waste": 250.0,
            "press_mud": 80.0,
            "poultry_litter": 20.0,
        },
        "temp_C": 37.0,
        "HRT_days": 22.0,
        "OLR": 3.2,
    },
}

HEALTH_PRESETS: Dict[str, Dict[str, Any]] = {
    "stable": {
        "label": "Stable plant",
        "mode": "lab",
        "pH": 7.2,
        "vfa": 0.4,
        "nh3": 0.02,
        "vfa_slope": 0.0,
        "ph_slope": 0.0,
    },
    "stressed": {
        "label": "Stressed plant",
        "mode": "lab",
        "pH": 6.4,
        "vfa": 3.2,
        "nh3": 0.06,
        "vfa_slope": 0.08,
        "ph_slope": -0.04,
    },
    "stable_quick": {
        "label": "Stable (symptoms)",
        "mode": "quick",
        "gas_drop": False,
        "bad_smell": False,
        "foam": False,
        "feed_change": False,
        "season_temp": "warm",
    },
    "stressed_quick": {
        "label": "Stressed (symptoms)",
        "mode": "quick",
        "gas_drop": True,
        "bad_smell": True,
        "foam": False,
        "feed_change": True,
        "season_temp": "cold",
    },
}

SIDEBAR_TOOLS: Tuple[Tuple[str, str], ...] = (
    ("pages/1_Feed_Optimizer.py", "Feed Optimizer"),
    ("pages/2_Health_Monitor.py", "Health Monitor"),
    ("pages/3_Impact_Calculator.py", "Impact Calculator"),
    ("pages/4_Live_Simulator.py", "Live Simulator"),
    ("pages/5_Model_Insights.py", "Model Insights"),
    ("pages/6_About.py", "About"),
)

_DEFAULT_FEED = DEMO_PRESETS["punjab_dairy"]


def init_session_defaults() -> None:
    """Initialize workflow / feed widget keys once per session."""
    if "workflow_flags" not in st.session_state:
        st.session_state["workflow_flags"] = {}
    if "feed_preset_name" not in st.session_state:
        st.session_state["feed_preset_name"] = None
    for k in FEEDSTOCK_KEYS:
        key = f"feed_{k}"
        if key not in st.session_state:
            st.session_state[key] = float(_DEFAULT_FEED["masses"].get(k, 0.0))
    for sk, val in (("feed_temp", 35.0), ("feed_hrt", 25.0), ("feed_olr", 2.5)):
        if sk not in st.session_state:
            st.session_state[sk] = val
    if "health_mode" not in st.session_state:
        st.session_state["health_mode"] = "Lab readings (advanced)"
    for hk, hv in (
        ("health_pH", 7.1),
        ("health_vfa", 0.5),
        ("health_nh3", 0.02),
        ("health_vfa_slope", 0.0),
        ("health_ph_slope", 0.0),
        ("health_gas_drop", False),
        ("health_bad_smell", False),
        ("health_foam", False),
        ("health_feed_change", False),
        ("health_season", "warm"),
    ):
        if hk not in st.session_state:
            st.session_state[hk] = hv


def inject_app_css() -> None:
    """Shared card and workflow styling."""
    st.markdown(
        """
        <style>
        .bio-hero {
            background: linear-gradient(135deg, #d8f5a2 0%, #b7e4c7 45%, #dcefe0 100%);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #95d5b2;
        }
        .bio-hero h1 { margin: 0 0 0.35rem 0; color: #1b4332; font-size: 1.75rem; }
        .bio-hero p { margin: 0; color: #2d6a4f; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #b7e4c7;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            box-shadow: 0 1px 2px rgba(27, 67, 50, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mark_workflow(step_id: str) -> None:
    init_session_defaults()
    flags = st.session_state["workflow_flags"]
    flags[step_id] = True


def workflow_step_done(step_id: str) -> bool:
    init_session_defaults()
    flags = st.session_state.get("workflow_flags") or {}
    if step_id == "feed":
        res = st.session_state.get("opt_result")
        return bool(isinstance(res, dict) and not res.get("error"))
    return bool(flags.get(step_id))


def apply_feed_preset(preset_key: str) -> None:
    """Load example feed masses and operating point into session widget keys."""
    init_session_defaults()
    preset = DEMO_PRESETS.get(preset_key)
    if not preset:
        return
    for k, v in preset["masses"].items():
        st.session_state[f"feed_{k}"] = float(v)
    st.session_state["feed_temp"] = float(preset["temp_C"])
    st.session_state["feed_hrt"] = float(preset["HRT_days"])
    st.session_state["feed_olr"] = float(preset["OLR"])
    st.session_state["feed_preset_name"] = preset_key


def apply_health_preset(preset_key: str) -> None:
    """Load example health readings into session widget keys."""
    preset = HEALTH_PRESETS.get(preset_key)
    if not preset:
        return
    if preset.get("mode") == "lab":
        st.session_state["health_mode"] = "Lab readings (advanced)"
        st.session_state["health_pH"] = float(preset["pH"])
        st.session_state["health_vfa"] = float(preset["vfa"])
        st.session_state["health_nh3"] = float(preset["nh3"])
        st.session_state["health_vfa_slope"] = float(preset.get("vfa_slope", 0.0))
        st.session_state["health_ph_slope"] = float(preset.get("ph_slope", 0.0))
    else:
        st.session_state["health_mode"] = "Quick symptoms"
        st.session_state["health_gas_drop"] = bool(preset.get("gas_drop", False))
        st.session_state["health_bad_smell"] = bool(preset.get("bad_smell", False))
        st.session_state["health_foam"] = bool(preset.get("foam", False))
        st.session_state["health_feed_change"] = bool(preset.get("feed_change", False))
        st.session_state["health_season"] = preset.get("season_temp", "warm")


def get_feed_masses_from_session() -> Dict[str, float]:
    init_session_defaults()
    return {k: float(st.session_state.get(f"feed_{k}", 0.0)) for k in FEEDSTOCK_KEYS}


def instant_baseline_ch4(
    available_kg: Mapping[str, float],
    temp_C: float,
    HRT_days: float,
    OLR: float,
    *,
    yield_model: Any,
) -> Optional[float]:
    """Predict CH₄ for equal-fraction availability mix (no Optuna)."""
    from . import optimizer

    keys = list(FEEDSTOCK_KEYS)
    avail = [float(available_kg.get(k, 0.0)) for k in keys]
    tot = sum(avail)
    if tot <= 1e-9 or yield_model is None:
        return None
    mf = [a / tot for a in avail]
    row = {
        "cow_frac": mf[0],
        "rice_frac": mf[1],
        "wheat_frac": mf[2],
        "food_frac": mf[3],
        "press_frac": mf[4],
        "poultry_frac": mf[5],
        "temperature_C": float(temp_C),
        "HRT_days": float(HRT_days),
        "OLR": float(OLR),
    }
    yhat, _ = optimizer.predict_yield_health(row, yield_model=yield_model)
    return float(yhat)


def render_page_shell(*, page_title: str) -> None:
    """CSS + sidebar."""
    init_session_defaults()
    inject_app_css()
    render_sidebar(page_title=page_title)


def render_sidebar(*, page_title: Optional[str] = None) -> None:
    """Branded sidebar with plant snapshot and page links."""
    init_session_defaults()
    with st.sidebar:
        st.markdown("### BioOptima")
        if page_title:
            st.caption(page_title)

        plant = plant_status_from_session()
        if plant["ready"]:
            st.metric("Last CH₄", f"{plant['ch4_m3']:.2f} m³/d" if plant.get("ch4_m3") else "—")
            st.caption(f"Stability: {plant.get('health_label', '—')}")
        else:
            st.caption("Run Feed Optimizer to see plant snapshot.")

        st.divider()
        for path, title in SIDEBAR_TOOLS:
            st.page_link(path, label=title)


def status_badge(level: int | str) -> None:
    """Render a compact status line for stability class."""
    if isinstance(level, str):
        label = level
        idx = {"Stable": 0, "Warning": 1, "Critical": 2}.get(label, -1)
    else:
        idx = int(level)
        label = ("Stable", "Warning", "Critical")[idx] if 0 <= idx <= 2 else "Unknown"
    if idx == 2:
        st.error(f"**{label}**")
    elif idx == 1:
        st.warning(f"**{label}**")
    else:
        st.success(f"**{label}**")


def metric_row(items: Sequence[Dict[str, Any]]) -> None:
    """Render a row of ``st.metric`` from dicts with keys: label, value, delta (optional)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                item["label"],
                item["value"],
                delta=item.get("delta"),
                delta_color=item.get("delta_color", "normal"),
                help=item.get("help"),
            )


def mix_labels_for_keys(keys: Sequence[str]) -> List[str]:
    return [DISPLAY_NAMES.get(k, k) for k in keys]


def reading_zone_figure(ph: float, vfa: float, nh3: float) -> go.Figure:
    """
    Three strip charts: shaded risk bands plus a marker at your reading (no fake time series).
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(
            "pH (green ~ 6.8–7.8)",
            "VFA — green below 1.5, red above 3 (kg HAc-eq/m³)",
            "Free NH₃ — green below 0.04 (kg/m³)",
        ),
    )

    def rects(subplot_row: int, ranges: Sequence[Tuple[float, float, str]]) -> None:
        for x0, x1, color in ranges:
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=0.35,
                y1=0.65,
                fillcolor=color,
                opacity=0.55,
                layer="below",
                line_width=0,
                row=subplot_row,
                col=1,
            )

    rects(1, [(5, 6.5, "#ffccd5"), (6.5, 6.8, "#fff3bf"), (6.8, 7.8, "#d8f5a2"), (7.8, 8.2, "#fff3bf"), (8.2, 8.5, "#ffccd5")])
    fig.add_trace(
        go.Scatter(x=[ph], y=[0.5], mode="markers", marker=dict(size=16, color="#1b4332", line=dict(width=2, color="white")), showlegend=False),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[5, 8.5], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    rects(2, [(0, 1.5, "#d8f5a2"), (1.5, 3.0, "#fff3bf"), (3.0, 4.0, "#ffccd5")])
    fig.add_trace(
        go.Scatter(x=[vfa], y=[0.5], mode="markers", marker=dict(size=16, color="#1b4332", line=dict(width=2, color="white")), showlegend=False),
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[0, 4], row=2, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=2, col=1)

    rects(3, [(0, 0.04, "#d8f5a2"), (0.04, 0.08, "#fff3bf"), (0.08, 0.12, "#ffccd5")])
    fig.add_trace(
        go.Scatter(x=[nh3], y=[0.5], mode="markers", marker=dict(size=16, color="#1b4332", line=dict(width=2, color="white")), showlegend=False),
        row=3,
        col=1,
    )
    fig.update_xaxes(range=[0, 0.12], row=3, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=3, col=1)

    for row, labels in [
        (1, [(6.2, "Low"), (7.3, "Safe"), (8.0, "High")]),
        (2, [(0.8, "Safe"), (2.2, "Warn"), (3.5, "Critical")]),
        (3, [(0.02, "Safe"), (0.06, "Warn"), (0.10, "High")]),
    ]:
        for x, txt in labels:
            fig.add_annotation(x=x, y=0.82, text=txt, showarrow=False, font=dict(size=9, color="#495057"), row=row, col=1)

    fig.update_layout(height=400, margin=dict(t=48, b=24, l=48, r=24), showlegend=False)
    return fig


def ch4_from_session_state() -> Optional[float]:
    """
    Daily CH₄ (m³/d) from the last Feed Optimizer run, if present in ``st.session_state``.
    """
    res = st.session_state.get("opt_result")
    if not isinstance(res, dict) or res.get("error"):
        return None
    y = res.get("yield")
    if y is None:
        return None
    try:
        return float(y)
    except (TypeError, ValueError):
        return None


def load_training_ranges() -> Dict[str, Tuple[float, float]]:
    """Load feature ranges from ``models/eval_metrics.json`` (set at train time)."""
    root = Path(__file__).resolve().parents[1]
    p = root / "models" / "eval_metrics.json"
    defaults = {
        "temperature_C": (25.0, 45.0),
        "HRT_days": (15.0, 40.0),
        "OLR": (0.5, 8.0),
    }
    if not p.is_file():
        return defaults
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        tr = data.get("training_ranges") or {}
        out = dict(defaults)
        for k, v in tr.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[k] = (float(v[0]), float(v[1]))
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return defaults


def load_eval_summary() -> Optional[Dict[str, Any]]:
    """Return eval metrics dict if ``models/eval_metrics.json`` exists."""
    p = Path(__file__).resolve().parents[1] / "models" / "eval_metrics.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def warn_if_out_of_training_range(
    *,
    temperature_C: Optional[float] = None,
    HRT_days: Optional[float] = None,
    OLR: Optional[float] = None,
) -> None:
    """Show Streamlit warnings for inputs outside training data range."""
    import streamlit as st

    ranges = load_training_ranges()
    msgs: List[str] = []
    if temperature_C is not None:
        lo, hi = ranges["temperature_C"]
        if temperature_C < lo or temperature_C > hi:
            msgs.append(f"Temperature {temperature_C:.1f} °C is outside training range [{lo:.0f}, {hi:.0f}] °C.")
    if HRT_days is not None:
        lo, hi = ranges["HRT_days"]
        if HRT_days < lo or HRT_days > hi:
            msgs.append(f"HRT {HRT_days:.1f} d is outside training range [{lo:.0f}, {hi:.0f}] d.")
    if OLR is not None:
        lo, hi = ranges["OLR"]
        if OLR < lo or OLR > hi:
            msgs.append(f"OLR {OLR:.2f} is outside training range [{lo:.1f}, {hi:.1f}] kg VS/m³/d.")
        if OLR < UI_OLR_MIN:
            msgs.append(
                f"OLR {OLR:.2f} is below the model training floor (~{UI_OLR_MIN:.1f} kg VS/m³/d). "
                "Predictions are extrapolated."
            )
    for m in msgs:
        st.warning(m + " Predictions may be less reliable.")


def health_delta_color(from_idx: int, to_idx: int) -> Literal["normal", "inverse"]:
    """Green if health improves or stays good; red if worsens."""
    if to_idx < from_idx:
        return "normal"
    if to_idx > from_idx:
        return "inverse"
    return "normal"


_HEALTH_COLORS = {0: "#2d6a4f", 1: "#bc6c25", 2: "#9d0208"}
_HEALTH_BG = {0: "#d8f5a2", 1: "#fff3bf", 2: "#ffccd5"}


def digester_schematic_figure(health_idx: int, *, title: str = "Digester status") -> go.Figure:
    """Simple 2D tank schematic colored by stability class (0=Stable, 1=Warning, 2=Critical)."""
    fill = _HEALTH_BG.get(int(health_idx), "#dcefe0")
    border = _HEALTH_COLORS.get(int(health_idx), "#2d6a4f")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.15, y0=0.1, x1=0.85, y1=0.75, fillcolor=fill, line=dict(color=border, width=3))
    fig.add_shape(type="rect", x0=0.2, y0=0.75, x1=0.8, y1=0.88, fillcolor="#b7e4c7", line=dict(color="#2d6a4f", width=2))
    fig.add_annotation(x=0.5, y=0.42, text="Liquid<br>digestate", showarrow=False, font=dict(size=14, color="#1b4332"))
    fig.add_annotation(x=0.5, y=0.81, text="Gas headspace", showarrow=False, font=dict(size=11, color="#1b4332"))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=220,
        margin=dict(t=40, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def status_banner(health_label: str) -> None:
    """Large colored verdict banner for digester health."""
    idx = {"Stable": 0, "Warning": 1, "Critical": 2}.get(health_label, -1)
    if idx == 2:
        st.error(f"**Plant status: {health_label}** — immediate attention recommended.")
    elif idx == 1:
        st.warning(f"**Plant status: {health_label}** — monitor VFA and pH closely.")
    elif idx == 0:
        st.success(f"**Plant status: {health_label}** — operating within safe bands.")
    else:
        st.info(f"**Plant status: {health_label}**")


def ch4_gauge_figure(ch4_m3: float, *, max_ch4: float = 15.0) -> go.Figure:
    """Plotly gauge for daily methane output."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(ch4_m3),
            number={"suffix": " m³/d"},
            title={"text": "Predicted CH₄"},
            gauge={
                "axis": {"range": [0, max_ch4]},
                "bar": {"color": "#2d6a4f"},
                "steps": [
                    {"range": [0, max_ch4 * 0.33], "color": "#ffccd5"},
                    {"range": [max_ch4 * 0.33, max_ch4 * 0.66], "color": "#fff3bf"},
                    {"range": [max_ch4 * 0.66, max_ch4], "color": "#d8f5a2"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=48, b=24))
    return fig


def mix_comparison_table_figure(rows: Sequence[Dict[str, Any]], *, title: str = "Availability vs recommended") -> go.Figure:
    """Plotly table with aligned columns (replaces wide Streamlit dataframe)."""
    if not rows:
        return go.Figure()
    n = len(rows)
    row_colors = ["#f4faf4" if i % 2 == 0 else "#ffffff" for i in range(n)]
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[220, 110, 110, 90],
                header=dict(
                    values=["Feedstock", "Availability %", "Recommended %", "Change (pp)"],
                    fill_color="#2d6a4f",
                    font=dict(color="white", size=13),
                    align=["left", "right", "right", "right"],
                ),
                cells=dict(
                    values=[
                        [r["Feedstock"] for r in rows],
                        [f"{float(r['Availability %']):.1f}" for r in rows],
                        [f"{float(r['Recommended %']):.1f}" for r in rows],
                        [f"{float(r['Change']):+.1f}" for r in rows],
                    ],
                    fill_color=[row_colors, row_colors, row_colors, row_colors],
                    align=["left", "right", "right", "right"],
                    font=dict(size=12, color="#1b4332"),
                    height=28,
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14)),
        margin=dict(l=0, r=0, t=36, b=0),
        height=52 + 30 * n,
    )
    return fig


def mix_pie_figure(
    labels: Sequence[str],
    values_pct: Sequence[float],
    *,
    title: str = "Recommended mix",
    min_pct: float = 1.0,
) -> go.Figure:
    """Donut chart; drops near-zero slices so labels do not overlap."""
    lab_f: List[str] = []
    val_f: List[float] = []
    for lab, v in zip(labels, values_pct):
        if float(v) >= min_pct:
            lab_f.append(str(lab))
            val_f.append(float(v))
    if not lab_f:
        i = max(range(len(values_pct)), key=lambda j: float(values_pct[j]))
        lab_f, val_f = [str(labels[i])], [float(values_pct[i])]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=lab_f,
                values=val_f,
                hole=0.45,
                textinfo="label+percent",
                textposition="auto",
                insidetextorientation="horizontal",
                marker=dict(line=dict(color="white", width=2)),
                sort=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(t=48, b=24, l=24, r=24),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02),
    )
    return fig


def plant_status_from_session() -> Dict[str, Any]:
    """Snapshot of last optimizer run for the home dashboard."""
    res = st.session_state.get("opt_result")
    if not isinstance(res, dict) or res.get("error"):
        return {"ready": False, "message": "No optimization run yet — start with **Feed Optimizer**."}
    ch4 = res.get("yield")
    try:
        ch4_f = float(ch4) if ch4 is not None else None
    except (TypeError, ValueError):
        ch4_f = None
    return {
        "ready": True,
        "ch4_m3": ch4_f,
        "health_label": res.get("health_label", "—"),
        "improvement_pct": res.get("improvement_pct"),
        "cn_ratio": res.get("cn_ratio"),
        "message": None,
    }


def progress_metric_bar(label: str, value: float, *, max_val: float = 1.0, help_text: Optional[str] = None) -> None:
    """Display a metric with a progress bar (for R², F1 on home page)."""
    pct = min(max(float(value) / max_val, 0.0), 1.0)
    st.metric(label, f"{value:.3f}")
    st.progress(pct, text=help_text or f"{pct * 100:.0f}% of target ({max_val:.2f})")
