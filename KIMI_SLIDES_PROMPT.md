# Kimi Slides Master Prompt for BioOptima

Copy everything inside the prompt block below and paste it into Kimi Slides.  
Attach:
- `major project report (22112054).pdf` (or converted report PDF)
- images from `outputs/required_images/`

---

## Prompt to paste in Kimi Slides

Create a **14–16 slide viva presentation** for my B.Tech major project using the attached report PDF and images.

### 0) Project identity (must be used exactly)
- **Title:** BioOptima: A Machine-Learning-Augmented Digital Twin for Anaerobic Digestion Optimization
- **Student:** Nikhil Kumar (22112054)
- **Department:** Chemical Engineering, NIT Jalandhar
- **Use case:** Decision support for anaerobic digestion plants (Indian context), especially mixed-feed farm/cluster operation.

### 1) Inputs and image usage rules
I am giving you:
1. Project report PDF (final source of truth for text and claims)
2. Output images folder (`required_images`) with named figures

**Rules**
- Do **not** use all images automatically.
- Use only context-relevant images on each slide.
- Do not fabricate data/plots/references.
- If an image listed for a slide is missing, continue with text and available visuals.
- Keep slide text concise and viva-friendly.

### 2) Core technical understanding you must reflect
This project is a **hybrid digital twin**:
- **Mechanistic layer:** ADM1 (PyADM1ODE) for realistic digestion physics and dynamic behavior.
- **Data-driven layer:** XGBoost surrogates for fast prediction.
- **Optimization layer:** feed-mix recommendation under operating constraints.
- **Validation layer:** literature parity + bias correction.
- **Deployment:** Streamlit app (Feed Optimizer, Health Monitor, Impact Calculator, Live Simulator, Model Insights, About).

### 3) Must-include quantitative anchors
Use these values in the deck (from project artifacts/report):
- Dataset scenarios: **4700**
- Engineered features: **19**
- Yield model CV R² (XGBoost): **~0.998070**
- Health classifier CV macro F1: **~0.831764**
- Literature validation: **21 points total, 19 in-range, 2 excluded**
- MAPE before correction: **~10.7%**
- MAPE after correction: **~8.5%**

### 4) India + Punjab section (mandatory)
Add specific slide(s) on:
- Current biogas/CBG push in India and Punjab
- Why many plants underperform in practice:
  - inconsistent feedstock supply/quality
  - inadequate process monitoring
  - unstable loading and recipe shifts
  - O&M and economics constraints
- Explain clearly how BioOptima addresses this:
  - feed optimization
  - early-warning style health view
  - interpretable model outputs
  - ADM1-backed credibility

Use conservative, defensible wording. Do not overclaim.

### 5) Slide-by-slide full content plan (follow strictly)

#### Slide 1 — Title
- BioOptima project title
- Student, roll number, institute, department, guide
- One-line subtitle: “Hybrid ADM1 + ML digital twin for practical biogas decision support”

#### Slide 2 — Problem Motivation
- AD plants handle variable feedstocks and changing conditions.
- Manual decisions can reduce methane yield and destabilize digester health.
- Need: fast, reliable, explainable decision support for operators.

#### Slide 3 — India & Punjab Context
- India and Punjab are pushing biogas/CBG, but many plants face utilization gaps.
- Operational bottlenecks often prevent full methane potential.
- Project relevance: improve plant-level performance without heavy new instrumentation.

#### Slide 4 — Why Plants Underperform
- Feed recipe changes not optimized scientifically.
- Health stress indicators (pH/VFA trends) not used proactively.
- Physics-only simulation is accurate but too slow for frequent what-if decisions.
- Conclusion: need a hybrid framework.

#### Slide 5 — Why ADM1 Matters
- ADM1 captures core digestion stages and process interactions.
- Provides mechanistic realism for pH, VFA, methane trends.
- Acts as high-fidelity backbone for generating trustworthy training labels.

#### Slide 6 — Why ML Surrogates Were Needed
- Operational decisions need near-instant response.
- Surrogates approximate ADM1 behavior at low latency.
- Result: real-time usability with physics-informed training origin.

#### Slide 7 — BioOptima Architecture
- Offline: scenario generation -> ADM1 simulation -> feature engineering -> model training
- Online: user inputs -> surrogate prediction -> optimization/health/impact dashboards
- Mention explainability + literature validation modules.

#### Slide 8 — Dataset and Features
- 4700 converged scenarios from ADM1-based pipeline
- 19 engineered features (fractions, operating variables, interaction/proxy terms)
- Stability class imbalance exists; motivates macro F1 focus for classifier evaluation.

#### Slide 9 — Model Comparison (Yield + Classifier context)
- Show comparative performance of candidate regressors/classifiers.
- State clearly: XGBoost selected for yield due to best CV R².
- Mention classifier comparison briefly and rationale for selected approach.

**Use image:** `fig_4_2_model_comparison.png`

#### Slide 10 — Yield Model Performance
- Hold-out predicted vs actual parity indicates strong fit.
- Learning curve shows improving generalization with data scale.
- Key point: surrogate is accurate enough for interactive optimization use.

**Use images:**
- `fig_4_3_yield_scatter.png`
- `fig_4_3_learning_curve.png`

#### Slide 11 — Health Classifier Performance
- Primary metric for generalization: 5-fold CV macro F1 ~0.831764.
- Macro F1 used because class distribution is imbalanced.
- Include class distribution evidence; confusion matrix optional if layout stays clean.

**Prefer image:** `fig_4_4_class_distribution.png`  
**Optional image:** `fig_4_4_confusion_matrix.png`

#### Slide 12 — SHAP Explainability
- Show which engineered features most influence methane prediction.
- Link SHAP trends to AD process logic (loading stress, composition effects, nutrient balance).
- Explain one representative prediction with waterfall plot.

**Use images:**
- `fig_4_5_shap_beeswarm.png`
- `fig_4_5_shap_waterfall.png`

#### Slide 13 — Literature Validation
- External plausibility check on curated published points.
- Distinguish in-range vs excluded points.
- Bias correction improves agreement (MAPE ~10.7% to ~8.5%).

**Use images:**
- `fig_4_6_literature_before_correction.png`
- `fig_4_6_literature_after_correction.png`

#### Slide 14 — ADM1 Dynamic Demonstration
- Show time-series realism for pH, VFA, CH4.
- Explain that this reinforces mechanistic credibility behind surrogate framework.

**Use images:**
- `fig_4_7_adm1_ph.png`
- `fig_4_7_adm1_vfa.png`
- `fig_4_7_adm1_ch4.png`

#### Slide 15 — Web Application & Practical Use
- Pages: Feed Optimizer, Health Monitor, Impact Calculator, Live Simulator, Model Insights.
- Workflow example:
  1) enter available feed and operating conditions  
  2) get recommended blend + predicted methane  
  3) inspect health risk/early warning view  
  4) quantify financial/environmental impact
- Practical value: more stable and data-informed plant operation.

#### Slide 16 — Conclusion and Future Scope
- Summary:
  - Hybrid ADM1 + ML approach works for fast and credible decision support.
  - Strong yield prediction, usable classifier signal, improved literature parity after correction.
  - Deployable web interface for practical adoption.
- Future work:
  - broaden feedstock/operating coverage
  - plant-specific recalibration with field data
  - richer uncertainty and monitoring integration
- End with: “Thank you — Questions”

### 6) Design instructions (strict)
- Style: bright, clean, professional, human-made.
- Avoid AI-generic visual style.
- Use 2–3 consistent colors, clear fonts, balanced white space.
- No flashy animations or overdesigned icon packs.
- Keep each slide easy to present verbally (short bullets, clear hierarchy).

### 7) Language style
- Simple technical English.
- Natural phrasing suitable for viva.
- No exaggerated claims.
- No AI-sounding filler text.

### 8) Final output requirement
- Produce an editable PPT with coherent storytelling.
- Ensure figures appear on logically matching slides only.
- Prioritize clarity, credibility, and presentation flow over visual excess.

---

## Local image folder reference

Use images from:
- `outputs/required_images/`

Expected figure names:
- `fig_4_2_model_comparison.png`
- `fig_4_3_yield_scatter.png`
- `fig_4_3_learning_curve.png`
- `fig_4_4_class_distribution.png`
- `fig_4_4_confusion_matrix.png`
- `fig_4_5_shap_beeswarm.png`
- `fig_4_5_shap_waterfall.png`
- `fig_4_6_literature_before_correction.png`
- `fig_4_6_literature_after_correction.png`
- `fig_4_7_adm1_ph.png`
- `fig_4_7_adm1_vfa.png`
- `fig_4_7_adm1_ch4.png`

