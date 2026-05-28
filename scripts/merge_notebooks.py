#!/usr/bin/env python3
"""Merge notebooks 01–03 into one analysis notebook."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

COLAB_CELL = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Colab only — skip when running locally in Jupyter\n",
        "import os\n",
        "import sys\n",
        "\n",
        "try:\n",
        "    from google.colab import drive\n",
        "except ImportError:\n",
        "    print(\"Not Colab — skip this cell.\")\n",
        "else:\n",
        "    drive.mount(\"/content/drive\")\n",
        "    MY_DRIVE = \"/content/drive/MyDrive\"\n",
        "    PROJECT_ROOT = \"\"  # e.g. \"/content/drive/MyDrive/major_project\"\n",
        "\n",
        "    def _is_root(p):\n",
        "        return os.path.isdir(p) and os.path.isdir(os.path.join(p, \"src\"))\n",
        "\n",
        "    if not PROJECT_ROOT or not _is_root(PROJECT_ROOT):\n",
        "        from collections import deque\n",
        "\n",
        "        found, q = None, deque([(MY_DRIVE, 0)])\n",
        "        while q and not found:\n",
        "            p, d = q.popleft()\n",
        "            if d > 5:\n",
        "                continue\n",
        "            if _is_root(p):\n",
        "                found = p\n",
        "                break\n",
        "            if os.path.isdir(p):\n",
        "                for name in os.listdir(p):\n",
        "                    c = os.path.join(p, name)\n",
        "                    if not name.startswith(\".\") and os.path.isdir(c):\n",
        "                        q.append((c, d + 1))\n",
        "        if not found:\n",
        "            raise FileNotFoundError(\n",
        "                \"Set PROJECT_ROOT to your Drive folder containing src/\"\n",
        "            )\n",
        "        PROJECT_ROOT = found\n",
        "        print(\"Auto-detected:\", PROJECT_ROOT)\n",
        "    os.chdir(PROJECT_ROOT)\n",
        "    if PROJECT_ROOT not in sys.path:\n",
        "        sys.path.insert(0, PROJECT_ROOT)\n",
        "    print(\"OK — CWD:\", os.getcwd())\n",
    ],
    "outputs": [],
    "execution_count": None,
}

IMPORTS_CELL = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import importlib.util\n",
        "import json\n",
        "import sys\n",
        "import time\n",
        "from pathlib import Path\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import shap\n",
        "from sklearn.metrics import (\n",
        "    classification_report,\n",
        "    confusion_matrix,\n",
        "    f1_score,\n",
        "    mean_squared_error,\n",
        "    r2_score,\n",
        "    accuracy_score,\n",
        ")\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "ROOT = Path.cwd()\n",
        "if not (ROOT / \"src\").is_dir() and (ROOT.parent / \"src\").is_dir():\n",
        "    ROOT = ROOT.parent\n",
        "if str(ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(ROOT))\n",
        "\n",
        "from src import features as feat_mod\n",
        "from src import health_utils\n",
        "from src import simulator\n",
        "from src.calibration import (\n",
        "    UI_HRT_MAX,\n",
        "    UI_HRT_MIN,\n",
        "    UI_OLR_MAX,\n",
        "    UI_OLR_MIN,\n",
        "    UI_TEMP_MAX,\n",
        "    UI_TEMP_MIN,\n",
        ")\n",
        "\n",
        "_spec = importlib.util.spec_from_file_location(\n",
        "    \"train_models\", ROOT / \"scripts\" / \"train_models.py\"\n",
        ")\n",
        "tm = importlib.util.module_from_spec(_spec)\n",
        "_spec.loader.exec_module(tm)\n",
        "\n",
        "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
        "%matplotlib inline\n",
        "plt.rcParams.update({\"figure.figsize\": (10, 4), \"axes.grid\": True})\n",
        "\n",
        "SEED = 42\n",
        "CV_FOLDS = 5\n",
        "V_LIQ = 100.0\n",
        "SKIP_STACKING_CV = True  # merge stacking metrics from model_comparison.json\n",
        "CSV_PATH = ROOT / \"data\" / \"generated\" / \"biogas_training_merged.csv\"\n",
        "\n",
        "print(\"Project root:\", ROOT)\n",
        "print(\"Training CSV:\", CSV_PATH.name)\n",
    ],
    "outputs": [],
    "execution_count": None,
}

PART_HEADERS = [
    (
        "# BioOptima — Full analysis notebook\n\n"
        "Single notebook for thesis / report: **physics check → EDA → ML training & literature validation**.\n\n"
        "| Part | Content |\n"
        "|------|----------|\n"
        "| **1** | ADM1 sanity check (baseline, stress, temperature, feed mix) |\n"
        "| **2** | Exploratory data analysis on merged training CSV (~4700 rows) |\n"
        "| **3** | Model comparison, SHAP, learning curve, literature parity |\n\n"
        "**Colab:** run the mount cell, then **Run all**. **Local:** skip mount cell, open from project root or `notebooks/`.\n\n"
        "Sweep generation stays in `00_generate_sweep_colab.ipynb` (GPU/time on Colab)."
    ),
    (
        "---\n\n"
        "# Part 1 — ADM1 physics sanity check\n\n"
        "Verify **PyADM1ODE** before trusting sweep data. Each run ~15–60 s.\n\n"
        "Checks: convergence, overload (VFA/pH), temperature response, substrate quality."
    ),
    (
        "---\n\n"
        "# Part 2 — Exploratory data analysis\n\n"
        "Merged training file: `biogas_training_merged.csv` (main + low-OLR + instability sweeps)."
    ),
    (
        "---\n\n"
        "# Part 3 — Model training & literature validation\n\n"
        "Mirrors `scripts/train_models.py` with figures. Deploy models with:\n"
        "`python scripts/train_models.py --csv data/generated/biogas_training_merged.csv`"
    ),
]


def _load(name: str) -> dict:
    return json.loads((NB_DIR / name).read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _is_setup_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = _src(cell)
    markers = (
        "from google.colab import drive",
        "ROOT = Path.cwd()",
        "import importlib.util",
        "from src import simulator",
        "from src import features",
        "sys.path.insert(0",
    )
    hits = sum(1 for m in markers if m in src)
    return hits >= 2 and len(src) < 2500


def _skip_intro(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    src = _src(cell)
    return src.startswith("# 0") and len(src) < 800


def _cells_from(nb: dict, skip_first_n: int = 0) -> list:
    out = []
    for i, c in enumerate(nb["cells"]):
        if i < skip_first_n:
            continue
        if _is_setup_cell(c) or _skip_intro(c):
            continue
        out.append(deepcopy(c))
    return out


def main() -> None:
    nb01 = _load("01_benchmark_and_validate.ipynb")
    nb02 = _load("02_eda.ipynb")
    nb03 = _load("03_model_training.ipynb")

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [PART_HEADERS[0]]},
        COLAB_CELL,
        IMPORTS_CELL,
        {"cell_type": "markdown", "metadata": {}, "source": [PART_HEADERS[1]]},
        *_cells_from(nb01, 0),
        {"cell_type": "markdown", "metadata": {}, "source": [PART_HEADERS[2]]},
        *_cells_from(nb02, 0),
        {"cell_type": "markdown", "metadata": {}, "source": [PART_HEADERS[3]]},
        *_cells_from(nb03, 0),
    ]

    # Patch Part 2 load cell to use CSV_PATH
    for c in cells:
        if c.get("cell_type") == "code" and "biogas_training" in _src(c):
            src = _src(c)
            if "read_csv" in src and "csv_path" in src:
                c["source"] = [
                    "raw = pd.read_csv(CSV_PATH)\n",
                    "if \"OLR\" not in raw.columns and \"OLR_used\" in raw.columns:\n",
                    "    raw[\"OLR\"] = raw[\"OLR_used\"]\n",
                    "\n",
                    "df = raw[raw[\"converged\"] == True].copy()\n",
                    "df = df[df[\"stability_label\"] >= 0].copy()\n",
                    "print(f\"Total rows: {len(raw)}, converged & valid: {len(df)}\")\n",
                    "if \"OLR\" in df.columns:\n",
                    "    print(f\"OLR range: {df['OLR'].min():.2f} – {df['OLR'].max():.2f}\")\n",
                    "df.head()\n",
                ]

    # Patch Part 3 load cell
    for c in cells:
        if c.get("cell_type") == "code" and "tm._load_data" in _src(c):
            c["source"] = [
                "df, X, y_yield, y_health = tm._load_data(CSV_PATH)\n",
                "X_df = pd.DataFrame(X, columns=feat_mod.FEATURE_COLUMNS)\n",
                "print(f\"Usable rows: {len(df)}\")\n",
                "print(\"Stability:\", df[\"stability_label\"].value_counts().sort_index().to_dict())\n",
                "if \"OLR\" in df.columns:\n",
                "    print(f\"OLR: {df['OLR'].min():.2f} – {df['OLR'].max():.2f}\")\n",
            ]

    # Remove duplicate import/json in part 3 if still present
    for c in cells:
        if c.get("cell_type") == "code" and _src(c).strip().startswith("import json") and "SKIP_STACKING" in _src(c):
            c["source"] = ["# Constants set in shared imports cell above.\n"]

    out_path = NB_DIR / "01_biooptima_full_analysis.ipynb"
    merged = {
        "cells": cells,
        "metadata": nb03.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(cells)} cells)")

    # Deprecation notice at top of legacy notebooks (content kept for reference)
    stub = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **Deprecated** — use **`01_biooptima_full_analysis.ipynb`** (Parts 1–3 combined). "
            "Sweep data: `00_generate_sweep_colab.ipynb`.\n"
        ],
    }
    for old in ("01_benchmark_and_validate.ipynb", "02_eda.ipynb", "03_model_training.ipynb"):
        p = NB_DIR / old
        nb = _load(old)
        if not _src(nb["cells"][0]).startswith("> **Deprecated**"):
            nb["cells"].insert(0, stub)
            p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
            print(f"Tagged {old}")


if __name__ == "__main__":
    main()
