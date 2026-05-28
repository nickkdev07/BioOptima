# BioOptima — Biogas digital twin

Indian farm-scale **feed optimization**, **health insight**, and **short-term CH₄ outlook** using PyADM1ODE simulations (training data) and lightweight XGBoost surrogates for the Streamlit app.

For a beginner-friendly project narrative, demo script, and viva FAQ, see **[BIOOPTIMA_PROJECT_GUIDE.md](BIOOPTIMA_PROJECT_GUIDE.md)**.

## Quick start

```bash
cd "d:\major project\webapp"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes **shap** and **streamlit-shap** for the Model Insights page. Always start the app with the venv active:

```bash
.venv\Scripts\streamlit run app.py
```

### 1) Training data

**Option A — full PyADM1ODE physics sweep (recommended, best fidelity)**

Run locally (takes ~8 hours with 1 worker):

```bash
python -m src.sweep --n 1000 --workers 1
```

Or use the Colab notebook `notebooks/00_generate_sweep_colab.ipynb` (~4-5 hours with 2 workers). Download the resulting CSV to `data/generated/biogas_training_data.csv`.

**Option B — semi-empirical fallback (fast, for CI / UI testing)**

```bash
python scripts/seed_synthetic_training.py
```

Generates 2000 rows in < 2 seconds using literature-grounded formulas (Arrhenius, first-order HRT, sigmoid OLR inhibition). Not used for final shipped models.

### 2) Train models

```bash
python scripts/train_models.py --csv data/generated/biogas_training_data.csv
```

Add `--tune` for Optuna hyperparameter search (~2-3 min):

```bash
python scripts/train_models.py --tune --n-trials 60
```

Writes `models/yield_predictor.pkl`, `models/health_classifier.pkl`, `models/eval_metrics.json`, and `models/X_background.pkl` (for SHAP).

### 3) Validate against published literature

```bash
python scripts/validate_against_literature.py
```

Runs the trained model on ~21 curated data points from peer-reviewed Indian AD papers and saves `models/literature_validation.json`.

### 4) Run the app

```bash
streamlit run app.py
```

## Architecture

- **Simulation:** `src/simulator.py` wraps `pyADM1ODE` (`BiogasPlant` + single digester).
- **Feed mapping:** `src/feedstock_db.py` — six Indian substrates, VS-based flows from OLR/HRT.
- **Sweep:** `src/sweep.py` — Dirichlet mix + Latin hypercube on temperature, HRT, OLR.
- **Features:** `src/features.py` — 19 engineered features including domain interaction terms (OLR×food, protein/CN, OLR/HRT, lipid×temp).
- **ML:** `scripts/train_models.py` — XGBoost regressor (CH₄) + classifier (stability) with early stopping and optional Optuna tuning.
- **Optimization:** `src/optimizer.py` — Optuna over available mass with health penalty.
- **Rules:** `src/health_utils.py` — early-warning heuristics (no third ML model).
- **Explainability:** SHAP beeswarm + waterfall plots on the Model Insights page.
- **Validation:** Literature data points in `data/validation/literature_points.csv`.
- **App:** `app.py` + 6 focused pages — Feed Optimizer, Health Monitor, Impact Calculator, Live Simulator, Model Insights, About.

## Notebooks

| Notebook | Purpose |
|---|---|
| `00_generate_sweep_colab.ipynb` | ADM1 sweeps on Google Colab (main + supplementary CSVs) |
| `01_biooptima_full_analysis.ipynb` | Deep analysis (physics check → EDA → full ML pipeline) |
| `02_biooptima_results.ipynb` | **Presentation notebook** — focused figures, train models, SHAP, literature validation, ADM1 demo (screenshots / viva) |

## Streamlit Cloud

Push this repo; set **Main file** to `app.py`. Training artifacts (`models/*.pkl`, `models/eval_metrics.json`) must be committed or built in CI before deploy. PyADM1ODE is **not** required on Cloud if models are present.

## Licence

PyADM1ODE is bundled under its own licence in `pyADM1ODE/`. BioOptima application code is provided for the academic project.
