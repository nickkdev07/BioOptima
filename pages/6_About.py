"""About BioOptima: science, model stack, glossary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.ui_helpers import render_page_shell

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
render_page_shell(page_title="About")

st.title("About anaerobic digestion & BioOptima")

with st.expander("Suggested workflow", expanded=False):
    st.markdown(
        """
        1. **Feed Optimizer** — Daily feed rates or a preset → **Find best mix**.
        2. **Health Monitor** — Lab readings or symptoms → zone chart and early warnings.
        3. **Impact Calculator** — Financial and environmental impact from optimizer CH₄.
        4. **Model Insights** — SHAP explainability and literature validation (optional).
        5. **Live Simulator** — ADM1 time series (~30–60 s per run, optional).
        """
    )

st.markdown("### System architecture")
st.markdown(
    """
```mermaid
flowchart TB
  subgraph data [Training data]
    ADM1[PyADM1ODE sweeps]
    CSV[biogas_training_data.csv]
    ADM1 --> CSV
  end
  subgraph ml [Surrogates]
    XGB[XGBoost yield + health]
    SHAP[SHAP explainability]
    CSV --> XGB
    XGB --> SHAP
  end
  subgraph app [Streamlit app]
    Opt[Feed Optimizer]
    Health[Health Monitor]
    Sim[Live Simulator]
    Impact[Impact Calculator]
    XGB --> Opt
    XGB --> Health
    ADM1 --> Sim
    Opt --> Impact
  end
```
"""
)

with st.expander("How anaerobic digestion works (4 stages)", expanded=False):
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("**1. Hydrolysis**")
        st.caption("Polymers → sugars, amino acids, fatty acids.")
    with h2:
        st.markdown("**2. Acidogenesis**")
        st.caption("Fermentation to VFAs, CO₂, H₂ — can lower pH if unbalanced.")
    with h3:
        st.markdown("**3. Acetogenesis**")
        st.caption("VFAs → acetate, CO₂, H₂ — often rate-limiting.")
    with h4:
        st.markdown("**4. Methanogenesis**")
        st.caption("CH₄ from acetate or CO₂ + H₂.")

st.markdown("### About ADM1")
st.markdown(
    """
    **Anaerobic Digestion Model No. 1 (ADM1)** is an IWA-standard structured model of single-tank digestion.
    **PyADM1ODE** provides ODE-based physics used for training sweeps and the **Live Simulator**.

    **BioOptima** uses PyADM1ODE for labeled data, **XGBoost** surrogates for fast what-if answers,
    and **Optuna** for feed mix optimization in the web app.
    """
)

st.markdown("### How BioOptima fits together")
st.code(
    """
    PyADM1ODE (LHS sweeps)  -->  training dataset (yield + stability)
              |
              v
         XGBoost surrogates  -->  fast predictions (Feed Optimizer, Health)
              |
              v
         Optuna optimization  -->  recommended mixes
              |
              v
         Streamlit app  -->  optimize · diagnose · simulate · impact
    """,
    language="text",
)

nb = ROOT / "notebooks" / "01_biooptima_full_analysis.ipynb"
if nb.is_file():
    st.markdown(f"**Full analysis notebook:** `{nb.name}` — physics check, EDA, training, and literature validation.")

st.divider()
with st.expander("Glossary", expanded=False):
    gloss = pd.DataFrame({
        "Term": ["OLR", "HRT", "VFA", "BMP", "C/N ratio", "ADM1", "Surrogate model", "Bias correction"],
        "Definition": [
            "Organic loading rate — kg VS per m³ liquid per day.",
            "Hydraulic retention time — average days in the digester.",
            "Volatile fatty acids — stress indicator when high.",
            "Biochemical methane potential — Nm³ CH₄ per tonne VS.",
            "Carbon-to-nitrogen ratio; co-digestion often targets ~20–30.",
            "Anaerobic Digestion Model No. 1 — multi-species biokinetic model.",
            "Fast ML model approximating slower physics simulations.",
            "Linear adjustment of CH₄ predictions to match literature validation points.",
        ],
    })
    q = st.text_input("Search glossary", placeholder="e.g. OLR, VFA, ADM1")
    if q.strip():
        mask = gloss.apply(lambda row: q.lower() in str(row["Term"]).lower() or q.lower() in str(row["Definition"]).lower(), axis=1)
        gloss = gloss[mask]
    st.dataframe(gloss, use_container_width=True, hide_index=True)

st.divider()
st.markdown("### Team & references")
st.markdown(
    """
    **BioOptima** — biogas digital twin for Indian feedstocks (BTech major project).

    **References:** Batstone et al. (ADM1); PyADM1ODE. Validate against **your own plant data** before operational use.
    """
)
