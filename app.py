"""
BioOptima — Biogas Digital Twin (Streamlit home).

Use the sidebar to open other tools. Model loading lives in ``src.model_loader`` so
pages never import this module (avoids duplicate ``set_page_config`` / home UI bleed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_loader import load_models
from src.ui_helpers import (
    inject_app_css,
    load_eval_summary,
    plant_status_from_session,
    progress_metric_bar,
    render_sidebar,
)

st.set_page_config(
    page_title="BioOptima",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_app_css()
render_sidebar(page_title="Home")

st.markdown(
    """
    <div class="bio-hero">
        <h1>BioOptima</h1>
        <p>Biogas digital twin for Indian co-digestion — optimize feed, check health, simulate digestion, and quantify impact.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

ym, _ = load_models()
if ym is None:
    st.warning("**Status: setup needed** — trained models are missing under `models/`. See the README.")
else:
    st.success("**Status: ready** — prediction models are loaded.")

st.page_link("pages/1_Feed_Optimizer.py", label="Open Feed Optimizer", use_container_width=True)

plant = plant_status_from_session()
if plant["ready"]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Last optimized CH₄", f"{plant['ch4_m3']:.2f} m³/d" if plant.get("ch4_m3") else "—")
    c2.metric("Predicted stability", str(plant.get("health_label", "—")))
    imp = plant.get("improvement_pct")
    c3.metric("Yield vs baseline", f"{imp:+.1f}%" if imp is not None else "—")
else:
    st.info(plant["message"])

ev = load_eval_summary()
if ev:
    st.subheader("Model accuracy (training)")
    cv_r2 = ev.get("yield_cv_r2_mean", ev.get("yield_r2"))
    cv_f1 = ev.get("health_cv_f1_macro_mean", ev.get("health_f1_macro"))
    col1, col2 = st.columns(2)
    with col1:
        if cv_r2 is not None:
            progress_metric_bar("Yield model (5-fold CV R²)", float(cv_r2), max_val=1.0)
    with col2:
        if cv_f1 is not None:
            progress_metric_bar("Health model (5-fold CV F1)", float(cv_f1), max_val=1.0)
    st.caption(
        f"Deployed: **{ev.get('best_regressor', 'xgboost')}** regressor · "
        f"**{ev.get('best_classifier', 'xgboost')}** classifier · "
        "CH₄ uses literature bias correction; stability uses rule-based pH/VFA."
    )

st.caption("Use the **sidebar** to switch pages. Typical order: Feed Optimizer → Health Monitor → Impact Calculator.")
