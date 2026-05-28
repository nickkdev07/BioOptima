"""SHAP explainability + literature validation for trained models."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import features as feat_mod
from src.model_loader import load_models
from src.calibration import UI_HRT_MAX, UI_HRT_MIN, UI_OLR_MAX, UI_OLR_MIN, UI_TEMP_MAX, UI_TEMP_MIN
from src.shap_viz import plot_beeswarm, plot_waterfall, shap_tree_estimator
from src.ui_helpers import (
    FEEDSTOCK_KEYS,
    render_page_shell,
)

_SHAP_INTERP = {
    "yield": (
        "**How to read this (methane yield):** Features farther right push predicted CH₄ **up**; left pushes it **down**. "
        "In AD terms, higher **food waste** or **press mud** fractions often raise degradable COD; **OLR** and **HRT** "
        "capture loading vs retention; **temperature** reflects mesophilic vs stressed thermophilic-like conditions."
    ),
    "health": (
        "**How to read this (stability classifier):** SHAP for the **Warning** class (index 1). "
        "The live app uses **rule-based** pH/VFA health on the Health Monitor page — this tab explains the trained ML classifier only."
    ),
}

st.set_page_config(page_title="Model Insights", page_icon="🔍", layout="wide")
render_page_shell(page_title="Model Insights")

st.title("Model insights")
st.caption("SHAP explainability and literature validation — first beeswarm may take ~30 s; results are cached.")

ym, hm = load_models()
mdir = ROOT / "models"

if ym is None:
    st.error("Models not found. Train first with `python scripts/train_models.py`.")
    st.stop()

bg_path = mdir / "X_background.pkl"
has_bg = bg_path.is_file()

tab_yield, tab_health, tab_lit = st.tabs(["Yield model (SHAP)", "Health model (SHAP)", "Literature validation"])


def _shap_defaults() -> tuple[list[float], float, float, float]:
    """Fractions + T/HRT/OLR from last optimizer run when available."""
    res = st.session_state.get("opt_result")
    ctx = st.session_state.get("opt_context") or {}
    if isinstance(res, dict) and res.get("mix"):
        mix = res["mix"]
        fracs = [float(mix.get(k, 0.0)) for k in FEEDSTOCK_KEYS]
        return (
            fracs,
            float(ctx.get("temp_C", 35.0)),
            float(ctx.get("HRT_days", 25.0)),
            float(ctx.get("OLR", 2.5)),
        )
    return [0.4, 0.1, 0.1, 0.2, 0.1, 0.1], 35.0, 25.0, 2.5


@st.cache_data(show_spinner=False)
def _cached_beeswarm(
    model_tag: str,
    bg_mtime: float,
    class_index: Optional[int],
    class_label: Optional[str],
) -> bytes:
    """Cache beeswarm as PNG bytes via matplotlib figure (pickle-friendly return: use figure recreation)."""
    import joblib

    ym_local = joblib.load(mdir / "yield_predictor.pkl") if model_tag == "yield" else joblib.load(mdir / "health_classifier.pkl")
    bg = joblib.load(bg_path)
    fig = plot_beeswarm(ym_local, bg, class_index=class_index, class_label=class_label)
    import io
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _shap_input_widgets(label: str) -> dict:
    fracs, t0, h0, o0 = _shap_defaults()
    c1, c2, c3 = st.columns(3)
    with c1:
        cow = st.slider("Cow dung frac", 0.0, 1.0, fracs[0], 0.05, key=f"{label}_cow")
        rice = st.slider("Rice straw frac", 0.0, 1.0, fracs[1], 0.05, key=f"{label}_rice")
    with c2:
        wheat = st.slider("Wheat straw frac", 0.0, 1.0, fracs[2], 0.05, key=f"{label}_wheat")
        food = st.slider("Food waste frac", 0.0, 1.0, fracs[3], 0.05, key=f"{label}_food")
    with c3:
        press = st.slider("Press mud frac", 0.0, 1.0, fracs[4], 0.05, key=f"{label}_press")
        poultry = st.slider("Poultry litter frac", 0.0, 1.0, fracs[5], 0.05, key=f"{label}_poultry")
    total = cow + rice + wheat + food + press + poultry
    fracs_n = [cow / total, rice / total, wheat / total, food / total, press / total, poultry / total] if total > 0 else [1 / 6] * 6
    temp = st.slider("Temperature (°C)", UI_TEMP_MIN, UI_TEMP_MAX, t0, key=f"{label}_temp")
    hrt = st.slider("HRT (days)", UI_HRT_MIN, UI_HRT_MAX, h0, key=f"{label}_hrt")
    olr = st.slider("OLR (kg VS/m³/d)", UI_OLR_MIN, UI_OLR_MAX, o0, 0.1, key=f"{label}_olr")
    return {
        "cow_frac": fracs_n[0], "rice_frac": fracs_n[1], "wheat_frac": fracs_n[2],
        "food_frac": fracs_n[3], "press_frac": fracs_n[4], "poultry_frac": fracs_n[5],
        "temperature_C": temp, "HRT_days": hrt, "OLR": olr,
    }


def _render_shap_tab(model: Any, label: str, *, class_index: int | None = None, class_label: str | None = None) -> None:
    if not has_bg:
        st.warning(
            "SHAP background file (`models/X_background.pkl`) not found. "
            "Re-run `python scripts/train_models.py` to generate it."
        )
        return

    try:
        import shap  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("Install SHAP: `pip install shap matplotlib`")
        return

    import joblib

    bg = joblib.load(bg_path)
    fn = list(bg.columns) if hasattr(bg, "columns") else None

    st.markdown(f"**Global feature importance ({label})**")
    model_tag = label
    try:
        with st.spinner("Computing SHAP values (cached after first run)…"):
            png_bytes = _cached_beeswarm(
                model_tag,
                bg_path.stat().st_mtime,
                class_index,
                class_label,
            )
        st.image(png_bytes, use_container_width=True)
    except Exception as exc:
        try:
            with st.spinner("Computing SHAP values…"):
                fig = plot_beeswarm(model, bg, class_index=class_index, class_label=class_label)
            pad_l, plot_c, pad_r = st.columns([1, 5, 1])
            with plot_c:
                st.pyplot(fig, use_container_width=False, clear_figure=True)
            plt.close(fig)
        except Exception as exc2:
            st.error(f"Could not render SHAP beeswarm: {exc2} ({exc})")
            return

    with st.expander("How to read this chart", expanded=False):
        st.markdown(_SHAP_INTERP.get(label, ""))

    with st.expander("Explain one prediction (waterfall)", expanded=False):
        st.caption("Adjust mix and operating point, then view a single forecast breakdown.")
        row = _shap_input_widgets(label)
        X_single = feat_mod.feature_matrix(row)
        try:
            with st.spinner("Building explanation…"):
                fig_w = plot_waterfall(
                    model,
                    X_single,
                    class_index=class_index,
                    feature_names=fn,
                )
            w_l, w_c, w_r = st.columns([1, 5, 1])
            with w_c:
                st.pyplot(fig_w, use_container_width=False, clear_figure=True)
            plt.close(fig_w)
        except Exception as exc:
            st.error(f"Could not render waterfall plot: {exc}")


with tab_yield:
    _render_shap_tab(ym, "yield")

with tab_health:
    if hm is None:
        st.info("No health classifier saved. The app uses rule-based stability in production.")
    elif not hasattr(shap_tree_estimator(hm), "get_booster"):
        st.warning("Health model is not a tree model — SHAP tree plots are skipped.")
    else:
        _render_shap_tab(hm, "health", class_index=1, class_label="Warning")

with tab_lit:
    lit_json = mdir / "literature_validation.json"
    lit_csv = ROOT / "data" / "validation" / "literature_points.csv"
    if lit_json.is_file():
        data = json.loads(lit_json.read_text(encoding="utf-8"))
        overall = data.get("overall_in_range") or {}
        raw_mape = float(overall.get("mape_pct", data.get("mape_pct", 0)))
        corr_mape = float(overall.get("mape_pct_corrected", 0))
        bc = overall.get("bias_correction") or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAPE (raw, in-range)", f"{raw_mape:.1f}%")
        if bc:
            m2.metric("MAPE (bias-corrected)", f"{corr_mape:.1f}%",
                      delta=f"{corr_mape - raw_mape:+.1f} pp", delta_color="inverse")
        m3.metric("R² (in-range)", f"{overall.get('r2', data.get('r2', 0)):.4f}")
        m4.metric("Points validated", f"{data.get('n_points_in_range', data.get('n_points', '—'))}")
        st.caption(
            "Bias correction aligns the surrogate with published Indian co-digestion yields — "
            "Bias correction aligns predictions with published Indian co-digestion yields."
        )
        if data.get("n_points_excluded", 0) > 0:
            st.caption(f"{data['n_points_excluded']} point(s) excluded from in-range metrics.")
        pts = pd.DataFrame(data["per_point"])
        ba = float(bc.get("a", 1.0))
        bb = float(bc.get("b", 0.0))
        if ba > 0 and "predicted" in pts.columns:
            pts = pts.copy()
            pts["predicted_corrected"] = ba * pts["predicted"] + bb
        in_rng = pts.get("in_training_range", pd.Series([True] * len(pts)))
        fig = go.Figure()
        if "in_training_range" in pts.columns:
            ex = pts[~in_rng]
            inn = pts[in_rng]
            if len(ex):
                fig.add_trace(go.Scatter(
                    x=ex["observed"], y=ex["predicted"], mode="markers",
                    name="Out of training range",
                    marker=dict(color="#94a3b8", size=10, symbol="circle-open"),
                    text=ex["substrate"],
                    hovertemplate="%{text}<br>obs=%{x:.3f} pred=%{y:.3f}<extra></extra>",
                ))
            if len(inn):
                ycol = "predicted_corrected" if "predicted_corrected" in inn.columns else "predicted"
                fig.add_trace(go.Scatter(
                    x=inn["observed"], y=inn[ycol], mode="markers",
                    name="In range (bias-corrected)" if ycol == "predicted_corrected" else "In range",
                    marker=dict(color="#2d6a4f", size=11),
                    text=inn["substrate"],
                    hovertemplate="%{text}<br>obs=%{x:.3f} pred=%{y:.3f}<extra></extra>",
                ))
        else:
            fig.add_trace(go.Scatter(x=pts["observed"], y=pts["predicted"], mode="markers"))
        all_y = list(pts["observed"]) + list(pts.get("predicted_corrected", pts["predicted"]))
        lo = min(min(all_y), min(pts["observed"])) * 0.8
        hi = max(max(all_y), max(pts["observed"])) * 1.2
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"))
        fig.update_layout(
            title="Published vs BioOptima specific CH₄ yield",
            xaxis_title="Published (m³ CH₄/m³ reactor/d)",
            yaxis_title="BioOptima (m³ CH₄/m³ reactor/d)",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pts, use_container_width=True, hide_index=True)
    elif lit_csv.is_file():
        st.info("Run `python scripts/validate_against_literature.py` after training.")
    else:
        st.info("No literature validation data found.")
