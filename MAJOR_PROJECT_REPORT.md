# BioOptima: A Machine-Learning-Augmented Digital Twin for Anaerobic Digestion Optimization

## Major Project Report

Submitted in partial fulfilment of the requirement for the award of degree of

**BACHELOR OF TECHNOLOGY in Chemical Engineering**

---

### Submitted by:
**Nikhil Kumar (22112054)**

### Under the guidance of:
**Dr. Anurag Kumar Tiwari**  
Assistant Professor

---

**DEPARTMENT OF CHEMICAL ENGINEERING**  
**DR. B. R. AMBEDKAR NATIONAL INSTITUTE OF TECHNOLOGY**  
Jalandhar, Punjab  
May 2026

---

## Candidate Declaration

I hereby declare that the work which is being presented in the major project report titled **"BioOptima: A Machine-Learning-Augmented Digital Twin for Anaerobic Digestion Optimization"** submitted towards the partial fulfilment of the requirements for the award of degree of Bachelor of Technology in Chemical Engineering at Dr. B. R. Ambedkar National Institute of Technology, Jalandhar is an authentic record of our work carried out from July 2025 to May 2026 under the supervision of Dr. Anurag Kumar Tiwari, Assistant Professor, Department of Chemical Engineering, NIT Jalandhar. The matter embodied in this project report has not been submitted by me for any other degree or diploma.

Place: NIT Jalandhar  
Date:

**Nikhil Kumar (22112054)**

---

## Certificate

This is to certify that the above statement made by the candidate is correct to the best of my knowledge.

Date:

**Dr. Anurag Kumar Tiwari** (Assistant Professor)  
**Dr. J. K. Ratan** (Head of Department)

---

## Acknowledgement

The satisfaction that accompanies the successful completion of any work would be incomplete unless we mention the names of the people who made it possible by providing constant guidance and encouragement.

I express my sincere gratitude to Dr. Anurag Kumar Tiwari, Assistant Professor, Department of Chemical Engineering, NIT Jalandhar for constant support, guidance, and encouragement. The timely suggestions and discussions helped in shaping the project and improving its technical quality.

I am also thankful to the Head of Department and the entire faculty and staff of the Department of Chemical Engineering for valuable advice and suggestions during this work.

**Nikhil Kumar (22112054)**

---

## Abstract

Farm-scale anaerobic digestion (AD) plants often operate with mixed, variable feedstocks and limited sensor availability. This makes it difficult to answer practical questions such as: (i) how to blend available wastes to maximize methane while maintaining digester stability, (ii) how to interpret health risk using limited lab measurements, and (iii) how to support decisions with credible process dynamics without running slow mechanistic simulations repeatedly.

This project presents BioOptima, a machine-learning-augmented biogas digital twin built for Indian digester conditions. The mechanistic physics layer uses ADM1 (Anaerobic Digestion Model No. 1) via a Python implementation (PyADM1ODE) to generate training labels and to run time-series simulations on demand. Surrogate models (XGBoost) are trained on engineered feedstock fractions and operating variables to predict methane yield and stability classes with low inference time. An Optuna-based optimizer uses bias-corrected surrogate predictions to compute recommended co-digestion mixes within operational constraints.

The trained methane surrogate achieves a 5-fold CV mean R^2 of 0.998070 and RMSE of 1.1628 (from `models/eval_metrics.json`). The stability classifier achieves a 5-fold CV mean macro F1 of 0.831764. Literature validation is performed using 21 curated data points; mean absolute percentage error (MAPE) improves from 10.7% to 8.5% after applying a linear bias correction. The final system is deployed as a Streamlit multi-page web application providing feed optimization, health monitoring, a live ADM1 simulator, impact calculation, and model insights with SHAP explainability.

---

## List of Abbreviations

AD: Anaerobic Digestion  
ADM1: Anaerobic Digestion Model No. 1  
VFA: Volatile Fatty Acids  
CH4: Methane  
HRT: Hydraulic Retention Time  
OLR: Organic Loading Rate  
ML: Machine Learning  
SHAP: SHapley Additive exPlanations  
LHS: Latin Hypercube Sampling  
SMOTE: Synthetic Minority Over-sampling Technique  
Optuna: Hyperparameter optimization framework  

---

## Table of Contents

1. Introduction
2. Literature Review
3. Methodology
4. Results and Discussion
5. Conclusion and Future Work
6. References
Appendix A. Code and supporting material (placeholders)

---

## List of Figures (Placeholders)

- Figure 1: BioOptima architecture (physics + surrogate + web app)
- Figure 2: Feature engineering overview (19 features)
- Figure 3: Model comparison (yield surrogate)
- Figure 4: Confusion matrix for stability classes
- Figure 5: SHAP feature importance (yield)
- Figure 6: Literature validation scatter (before/after bias correction)
- Figure 7: Live ADM1 simulation time series (pH, VFA, CH4)
- Figure 8-13: Streamlit page screenshots (Feed Optimizer, Health Monitor, Impact Calculator, Live Simulator, Model Insights, About)

---

## List of Tables (Placeholders)

- Table 1: Training dataset summary (size, features, parameter ranges)
- Table 2: Model comparison metrics (yield and health)
- Table 3: Literature validation summary (MAPE before/after correction)

---

## Chapter 1: Introduction

### 1.1 Background

Anaerobic digestion (AD) is a biological process in which microorganisms convert organic matter into biogas, mainly methane (CH4), under oxygen-free conditions. In many rural and farm-based settings, AD units are fed with mixed wastes such as cow dung, crop residues (rice and wheat straw), food waste, press mud, and poultry litter. Operational success depends strongly on maintaining digester stability: methane-producing microbes require favorable pH conditions and limited accumulation of intermediate acids.

In practice, operators face two constraints. First, feedstock composition changes daily and seasonally, which alters kinetic behavior and the risk of acidification. Second, laboratory measurements (such as VFA or pH) are not available frequently for all plants. Without a tool that connects feed composition and operating conditions to both methane production and health risk, plant performance can degrade unexpectedly.

### 1.2 Motivation

Mechanistic models such as ADM1 can simulate AD dynamics credibly, but running ADM1 for every “what-if” decision is computationally slow for interactive use. Conversely, simplified equations can be fast but may not represent complex interactions among feed composition, inhibition mechanisms, and dynamic acid accumulation.

Therefore, a hybrid approach is required: accurate mechanistic modeling for training and on-demand validation, combined with fast machine-learning surrogates for real-time decision support.

### 1.3 Objectives

The main objectives of this project are:

1. Build a data pipeline that uses ADM1-based simulation to generate training labels and scenarios for Indian feedstocks.
2. Train surrogate models that predict methane yield and stability-related risk efficiently.
3. Implement a feed optimization workflow that recommends co-digestion mixes using bias-corrected surrogate predictions and operational constraints.
4. Deploy the workflow as a user-friendly Streamlit web application with interactive pages and clear visual outputs.
5. Provide explainability (via SHAP) and an external literature plausibility check (with bias correction).

### 1.4 Scope and Limitations

Scope:
- Six Indian feedstocks are modeled as input building blocks.
- The app provides methane prediction, stability guidance, live ADM1 simulation on demand, and impact estimation.

Limitations:
- Surrogate accuracy is constrained to the parameter ranges used during training.
- Literature validation uses curated specific-yield values; uncertainty is expected due to differences in reactor design and reported conditions.
- Uncertainty quantification is not included in the UI by design (project requirement).

### 1.5 Organization of the Report

Chapter 2 summarizes the scientific background and the literature supporting ADM1, co-digestion, and machine-learning surrogates. Chapter 3 details the methodology: data generation, feature engineering, surrogate training, literature validation with bias correction, and optimization. Chapter 4 presents metrics and discusses results, including SHAP explainability and Streamlit outputs. Chapter 5 provides conclusions and future work directions.

---

## Chapter 2: Literature Review

### 2.1 Anaerobic Digestion fundamentals and ADM1

ADM1 (Anaerobic Digestion Model No. 1) is a structured dynamic model that represents the biochemical stages of AD (hydrolysis, acidogenesis, acetogenesis, and methanogenesis) along with physicochemical processes such as ion equilibria and gas-liquid transfer. Because ADM1 provides a consistent mechanistic framework, it serves as a reference model for generating labels and for interpreting how operating changes affect process dynamics.

### 2.2 Co-digestion of Indian feedstocks

Co-digestion refers to simultaneously processing multiple organic feedstocks. Compared to mono-digestion, co-digestion can improve nutrient balance (e.g., carbon to nitrogen ratio), enhance buffering capacity, and dilute inhibitory compounds. However, it can also introduce new risks if mixing ratios and loading conditions are not controlled.

In this project, the feedstock set includes cow dung, rice straw, wheat straw, food waste, press mud, and poultry litter. These represent a practical variety of rural availability and provide a wide range of carbohydrate/protein/lipid characteristics and inhibition potentials.

### 2.3 Machine learning surrogates for process simulation

Machine learning models are increasingly used in bioprocess contexts to enable fast prediction, pattern recognition, and optimization. However, surrogate models must be trained on representative data and should be used with care when extrapolating beyond the training distribution.

This project uses gradient-boosted decision trees (XGBoost) to learn nonlinear relationships between engineered features (feed fractions and operating variables) and mechanistic simulation outputs.

### 2.4 XGBoost and explainability with SHAP

XGBoost offers strong performance for tabular regression and classification tasks and is efficient enough for interactive web applications. Since chemical engineering decision-making benefits from interpretability, SHAP is employed to quantify the contribution of each engineered feature to predicted methane yield.

### 2.5 Digital twins in bioprocess engineering

Digital twins are software-based representations that support monitoring and decision-making by combining physics-driven models with data-driven components. In the context of AD, a hybrid twin can use mechanistic simulation as the high-fidelity reference while surrogates provide fast updates and optimization in near real-time.

### 2.6 Gaps in existing work and project motivation

Common limitations identified across AD modeling efforts include:
- reliance on slow mechanistic simulations for interactive exploration,
- insufficient handling of mixed feedstock variability,
- limited explainability for operators,
- challenges in connecting external experimental/literature data to model predictions without calibration.

BioOptima addresses these gaps by combining ADM1-based data generation, explainable surrogate learning, optimization, and literature validation with bias correction.

---

## Chapter 3: Methodology

### 3.1 Overall architecture

The BioOptima workflow contains an offline training pipeline and an online Streamlit application:

1. Offline (data generation and training)
   - Load Indian substrate parameters (YAML files).
   - Generate operating scenarios using LHS and feed-mix constraints via Dirichlet sampling.
   - Run ADM1 simulation to obtain methane and stability-related labels.
   - Engineer features (19 features) and train surrogate models.

2. Online (real-time decision support)
   - Load trained surrogates from `models/`.
   - Accept operator inputs (feed masses, temperature, HRT, OLR).
   - Predict methane yield with bias correction and provide stability guidance.
   - Provide a live ADM1 simulation page for time-series behavior.

### 3.2 Feedstock characterization (YAML and physical meaning)

Six Indian feedstocks are represented as building blocks. Each substrate YAML includes mechanistic parameters required by PyADM1ODE. These parameters encode physical/biochemical properties that affect degradation pathways and methane formation.

The user-facing part of BioOptima uses feed fractions and computes engineered proxies such as:
- carbon-nitrogen (C/N) ratio proxy,
- total VS turnover proxy,
- lipid and protein mix proxies,
- interaction features capturing coupled effects (e.g., food fraction x loading).

### 3.3 ADM1 simulation and data generation

The training dataset is generated by running ADM1 scenarios with PyADM1ODE. The scenario sampling strategy includes:
- Feed mix sampling using Dirichlet distribution so that mixture fractions sum to 1.
- Operating-variable sampling using LHS across temperature, HRT, and OLR ranges.

The final training dataset used for the current deployed models contains:
- `n_total = 4700` scenarios
- `n_features = 19` engineered features

Parameter ranges used for training ranges (from `models/eval_metrics.json` and `models/literature_validation.json`):
- Temperature: 25.01 to 44.99 C
- HRT: 10.00 to 49.99 days
- OLR: 1.74 to 14.90 kg VS/m3/d

### 3.4 Feature engineering

Feature vector dimensionality is 19, built from six feed fraction inputs, operating variables, derived physicochemical proxies, and interaction terms. The feature names (from `models/eval_metrics.json`) are:

`cow_frac, rice_frac, wheat_frac, food_frac, press_frac, poultry_frac, temperature_C, HRT_days, OLR, CN_ratio, VS_total, temp_HRT, lipid_frac_mix, protein_frac_mix, COD_proxy, OLR_x_food_frac, protein_to_CN, OLR_per_HRT, lipid_x_temp`

These features allow the surrogate model to approximate mechanistic relationships while remaining compatible with fast inference.

### 3.5 Model training

Two supervised learning tasks are used:

1. Yield regression:
   - Target: mean methane flow rate over the final saved window from ADM1
   - Model: XGBoost regressor

2. Stability classification:
   - Target labels: stability class generated from mechanistic final pH/VFA thresholds
   - Model: XGBoost classifier with class imbalance handling (SMOTE used in training pipeline)

Model selection is based on 5-fold cross-validation metrics.

### 3.6 SHAP explainability

SHAP values are computed for the yield surrogate to identify which engineered features most strongly influence predicted methane yield. This supports model transparency in the Streamlit “Model Insights” page.

### 3.7 Literature validation and bias correction

To check plausibility against external reported values, the trained yield surrogate is evaluated on curated literature points stored as 21 entries in:
`data/validation/literature_points.csv`

In that validation:
- raw predictions are compared to observed specific methane yield values,
- linear bias correction is applied to reduce systematic under/over-prediction:

`predicted_corrected = a * predicted + b`

For the current tuned model, the in-range bias correction parameters and MAPE are:
- In-range MAPE before correction: 10.7%
- In-range MAPE after correction: 8.5%
- Bias correction: a = 1.1493, b = -0.0344

### 3.8 Feed optimization (Optuna)

The feed optimizer is implemented using Optuna to search recommended feed compositions under practical availability constraints. The optimization variables are the maximum masses drawn from the available feedstocks, subject to fraction normalization constraints.

Key details (from `src/optimizer.py`):
- optimization trials: `n_trials = 200`
- health penalty uses rule-based stability evaluation (not the ML classifier in production inference)
- critical health class results in objective 0, and warning penalizes yield by a factor of 0.7.

The result includes:
- recommended feed mix fractions,
- bias-corrected predicted methane yield,
- health label and improvement percentage compared to the availability mix.

### 3.9 Web application design (Streamlit pages)

The Streamlit application exposes the workflow through six pages:

1. `pages/1_Feed_Optimizer.py` - recommended co-digestion mix and methane outputs
2. `pages/2_Health_Monitor.py` - stability guidance from readings and risk bands
3. `pages/3_Impact_Calculator.py` - economics and environmental indicators from predicted methane
4. `pages/4_Live_Simulator.py` - live ADM1 time-series simulation
5. `pages/5_Model_Insights.py` - SHAP explainability and literature validation scatter
6. `pages/6_About.py` - scientific background and workflow description

The app loads surrogates using cached model loading for fast response and provides screenshot-friendly visualizations.

---

## Chapter 4: Results and Discussion

### 4.1 Dataset overview

The surrogate models are trained on a mechanistic ADM1-generated dataset that includes:
- `4700` total scenarios
- `19` engineered features per sample

The stability labels show class imbalance, with most scenarios belonging to the “Critical” category, making macro-averaged F1 the more meaningful metric for classifier performance.

### 4.2 Model comparison results (yield)

Model comparison (from `models/eval_metrics.json`) indicates the best regressor is XGBoost, with:
- XGBoost mean CV R^2 = 0.998070
- LightGBM mean CV R^2 = 0.997737
- Random Forest mean CV R^2 = 0.993895
- Stacking CV R^2 = 0.909999

### 4.3 Yield model performance

The yield surrogate is evaluated using:
- scatter plots (predicted vs true methane),
- learning curve behavior,
- RMSE and R^2 values.

Expected figures to insert (generate in **`notebooks/03_report_figures_colab.ipynb`** → `models/report_figures/`):
- `<!-- SCREENSHOT: fig_4_3_yield_scatter.png -->`
- `<!-- SCREENSHOT: fig_4_3_learning_curve.png -->`

### 4.4 Health classifier performance

The stability classifier metrics (from `models/eval_metrics.json`) are reported as:
- 5-fold CV macro F1 mean: 0.831764
- (include test macro F1 if shown in notebook outputs)

Expected figures (`03_report_figures_colab.ipynb`):
- `<!-- SCREENSHOT: fig_4_4_confusion_matrix.png -->`
- `<!-- SCREENSHOT: fig_4_4_class_distribution.png -->`

### 4.5 SHAP analysis

SHAP analysis identifies the most influential engineered features for methane prediction. Interpretation should link features back to known process behavior. For example, interaction terms involving food fraction and loading are expected to reflect acidification sensitivity, while C/N and lipid/protein proxies represent nutrient balance and degradability.

Expected figures (`03_report_figures_colab.ipynb`):
- `<!-- SCREENSHOT: fig_4_5_shap_beeswarm.png -->`
- `<!-- SCREENSHOT: fig_4_5_shap_waterfall.png -->`

### 4.6 Literature validation

Literature validation is performed using 21 curated points (19 in-range; 2 excluded as out-of-training-range). For in-range points:
- MAPE before correction: 10.7%
- MAPE after correction: 8.5%

Expected figures (`03_report_figures_colab.ipynb`):
- `<!-- SCREENSHOT: fig_4_6_literature_before_correction.png -->`
- `<!-- SCREENSHOT: fig_4_6_literature_after_correction.png -->`

### 4.7 ADM1 live simulation demonstration

The Live Simulator page demonstrates credible mechanistic dynamics (time series of pH, VFA, methane, and related intermediates). This is used to validate that the surrogate framework corresponds to realistic process behavior.

Expected figures (`03_report_figures_colab.ipynb` for ADM1 plots; or Streamlit Live Simulator tabs):
- `<!-- SCREENSHOT: fig_4_7_adm1_ph.png -->`
- `<!-- SCREENSHOT: fig_4_7_adm1_vfa.png -->`
- `<!-- SCREENSHOT: fig_4_7_adm1_ch4.png -->`

### 4.8 Web application demonstration

Insert screenshot proofs for each Streamlit page:
- `<!-- SCREENSHOT: Feed Optimizer page -->`
- `<!-- SCREENSHOT: Health Monitor page -->`
- `<!-- SCREENSHOT: Impact Calculator page -->`
- `<!-- SCREENSHOT: Live Simulator page -->`
- `<!-- SCREENSHOT: Model Insights page -->`
- `<!-- SCREENSHOT: About page -->`

---

## Chapter 5: Conclusion and Future Work

### 5.1 Summary of contributions

BioOptima integrates mechanistic ADM1 simulation with fast machine learning surrogates and an optimization layer to provide practical decision support for AD plant operators. The final tool is deployed as a Streamlit web application with explainable predictions and literature validation.

### 5.2 Key findings

1. Methane surrogate performance is strong: mean CV R^2 is approximately 0.998070.
2. Stability classification achieves 5-fold CV macro F1 mean of 0.831764, supporting stability-aware feed recommendations.
3. Literature validation improves with bias correction: in-range MAPE decreases from 10.7% to 8.5%.

### 5.3 Limitations

1. Surrogate accuracy depends on training coverage; out-of-range conditions are not guaranteed.
2. Literature points include experimental uncertainty and heterogeneous reactor configurations.
3. The app uses rule-based health logic for real-time decisions; additional ML models may be explored for future research.

### 5.4 Future work

- Expand the feedstock library and regenerate training scenarios.
- Add more operating range coverage, including thermophilic regimes if required by available literature.
- Improve the health monitoring layer using additional interpretable features and/or enhanced rule calibration.
- Extend the optimization to include policy constraints (e.g., minimum dung inclusion) and multi-objective settings (methane + stability risk).

---

## Chapter 6: References

> Note: Format each entry according to the citation style required by your institute (IEEE/APA/etc.). The list below includes core references used in the report planning.

### ADM1 and bioprocess modeling

Batstone, D.J., Keller, J., Angelidaki, I., Kalyuzhnyi, S.V., Pavlostathis, S.G., Rozzi, A., Sanders, W.T.M., Siegrist, H.A., & Vavilin, V.A. (2002). The IWA Anaerobic Digestion Model No. 1 (ADM1). *Water Science and Technology*, 45(10), 65-73. https://doi.org/10.2166/wst.2002.0292

Schlattmann, M. (2011). Weiterentwicklung des "Anaerobic Digestion Model (ADM1)" zur Anwendung auf landwirtschaftliche Substrate. PhD dissertation, Technische Universitat Munchen. https://nbn-resolving.org/urn:nbn:de:bvb:91-diss-20110728-1071150-1-0

### Software implementations of ADM1 (Python)

Sadrimajd, P., Mannion, P., Howley, E., & Lens, P.N.L. (2021). PyADM1: a Python implementation of Anaerobic Digestion Model No. 1. *bioRxiv*. https://doi.org/10.1101/2021.03.03.433746

Gaida, D. (2014). Dynamic real-time substrate feed optimization of anaerobic co-digestion plants. PhD thesis, Universiteit Leiden.

Gaida, D. PyADM1ODE / PyADM1: Advanced Biogas Plant Simulation Framework (software repository). https://github.com/dgaida/PyADM1ODE

### Machine learning, explainability, optimization

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*. https://doi.org/10.1145/2939672.2939785

Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. https://doi.org/10.1613/JAIR.953

Lundberg, S.M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017 (NIPS 30)*, 4765-4774.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-Generation Hyperparameter Optimization Framework. *KDD 2019*. https://doi.org/10.1145/3292500.3330701

### Digital twin and web app framework

Treuille, A., & Kelly, A. (2019). Streamlit, a new app framework for machine learning tools. *NeurIPS 2019 Demonstration*.

Lu, Y., Liu, C., Wang, K.I.-K., Huang, H., & Xu, X. (2019). Digital Twin-driven smart manufacturing: Connotation, reference model, applications and research issues. *Robotics and Computer-Integrated Manufacturing*, 61, 101837. https://doi.org/10.1016/j.rcim.2019.101837

### Domain reviews used in Chapter 2

Rabii, A., Aldin, S., Dahman, Y., & Elbeshbishy, E. (2019). A Review on Anaerobic Co-Digestion with a Focus on the Microbial Populations and the Effect of Multi-Stage Digester Configuration. *Energies*, 12(6), 1106. https://doi.org/10.3390/en12061106

Rutland, H., You, J., Liu, H., Bull, L., & Reynolds, D. (2023). A Systematic Review of Machine-Learning Solutions in Anaerobic Digestion. *Bioengineering*, 10(12), 1410. https://doi.org/10.3390/bioengineering10121410

### Literature validation sources (exact identifiers from dataset)

The 21 curated literature points used for external validation are referenced by their dataset `source` labels in:
`data/validation/literature_points.csv`

Exact labels used:
MDPI Bioengineering 2022, Table 2
Frontiers Energy Res. 2025, Table 3
IJERT 2020, Table 1
Springer AAER 2023, Table 2
Indian J. Anim. Sci. 2019, Fig 3
Waste Mgmt 2021, Table 4
Sugar Tech 2020, Table 3
Poultry Sci. 2018, Table 2
Bioresource Tech. 2019, Fig 5
Environ. Sci. Pollut. Res. 2022
Renew. Energy 2021, Table 5

---

## Appendix A: Code, model outputs, and supporting screenshots (Placeholders)

### A.1 Colab notebook content (code + outputs)

`<!-- BEGIN PASTE: Colab notebook cells and outputs (02_biooptima_results.ipynb) -->`

Instructions for insertion:
- Paste only the code cells used for your main figures and outputs.
- Include the outputs corresponding to: data overview, model comparison plots, SHAP plots, literature validation scatter, and ADM1 demo plots.
- Keep variable settings and parameter ranges visible in screenshots (temperature, HRT, OLR, sample size).

`<!-- END PASTE: Colab notebook cells and outputs -->`

### A.2 Streamlit proof screenshots

`<!-- BEGIN PASTE: Streamlit page screenshots -->`

- Feed Optimizer: one realistic farm case with recommended mix and predicted methane.
- Health Monitor: one stable and one critical example with risk band visualization.
- Impact Calculator: LPG/cost/CO2 outputs derived from predicted methane.
- Live Simulator: pH/VFA/CH4 time series.
- Model Insights: SHAP summary and literature validation plot.

`<!-- END PASTE: Streamlit page screenshots -->`

### A.3 Software and libraries used

Insert the relevant software stack and versions from `requirements.txt`.

`<!-- BEGIN PASTE: requirements.txt summary + versions -->`

`<!-- END PASTE: requirements.txt summary + versions -->`

