"""Cached ML model loading for Streamlit (no UI at import time)."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st


@st.cache_resource
def load_models():
    """Load yield regressor; health classifier optional (app uses rule-based health)."""
    mdir = Path(__file__).resolve().parents[1] / "models"
    yp = mdir / "yield_predictor.pkl"
    hp = mdir / "health_classifier.pkl"
    if not yp.is_file():
        return None, None
    ym = joblib.load(yp)
    hm = joblib.load(hp) if hp.is_file() else None
    return ym, hm
