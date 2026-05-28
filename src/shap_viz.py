"""SHAP plotting helpers for BioOptima Model Insights (matplotlib + TreeExplainer)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap

# Readable labels for SHAP axes (matches FEATURE_COLUMNS order/names)
FEATURE_LABELS: dict[str, str] = {
    "cow_frac": "Cow dung share",
    "rice_frac": "Rice straw share",
    "wheat_frac": "Wheat straw share",
    "food_frac": "Food waste share",
    "press_frac": "Press mud share",
    "poultry_frac": "Poultry litter share",
    "temperature_C": "Temperature (°C)",
    "HRT_days": "HRT (days)",
    "OLR": "OLR",
    "CN_ratio": "C/N ratio",
    "VS_total": "Total VS",
    "temp_HRT": "Temp × HRT",
    "lipid_frac_mix": "Lipid fraction",
    "protein_frac_mix": "Protein fraction",
    "COD_proxy": "COD proxy",
    "OLR_x_food_frac": "OLR × food share",
    "protein_to_CN": "Protein / C/N",
    "OLR_per_HRT": "OLR / HRT",
    "lipid_x_temp": "Lipid × temp",
}


def shap_tree_estimator(model: Any) -> Any:
    """Use the fitted XGBoost estimator (unwrap SMOTE pipeline if needed)."""
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        return model.named_steps["clf"]
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def _feature_names(bg: Any) -> List[str]:
    if hasattr(bg, "columns"):
        return list(bg.columns)
    return [f"f{i}" for i in range(np.asarray(bg).shape[1])]


def _display_names(names: Sequence[str]) -> List[str]:
    return [FEATURE_LABELS.get(str(n), str(n)) for n in names]


def _background_matrix(bg: Any, max_rows: int = 64) -> np.ndarray:
    X = bg.values if hasattr(bg, "values") else np.asarray(bg, dtype=float)
    if len(X) > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X = X[idx]
    return np.asarray(X, dtype=float)


def _fig_height(n_features: int, *, base: float = 1.8, per_feat: float = 0.16, cap: float = 4.8) -> float:
    return min(cap, base + per_feat * n_features)


def plot_beeswarm(
    model: Any,
    background: Any,
    *,
    class_index: Optional[int] = None,
    class_label: Optional[str] = None,
    max_background: int = 64,
) -> plt.Figure:
    """Compact global SHAP beeswarm (TreeExplainer)."""
    est = shap_tree_estimator(model)
    X = _background_matrix(background, max_rows=max_background)
    names = _feature_names(background)
    display = _display_names(names)
    n_feat = X.shape[1]

    explainer = shap.TreeExplainer(est)
    explanation = explainer(X)

    if class_index is not None and len(explanation.shape) == 3:
        plot_exp = explanation[:, :, class_index]
        title_suffix = f" — {class_label or f'class {class_index}'}"
    else:
        plot_exp = explanation
        title_suffix = ""

    if hasattr(plot_exp, "feature_names"):
        plot_exp.feature_names = display

    h = _fig_height(n_feat)
    fig = plt.figure(figsize=(7.2, h), dpi=96)
    shap.plots.beeswarm(plot_exp, show=False, max_display=14)
    fig.suptitle(f"Feature impact{title_suffix}", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.22, right=0.88, top=0.92, bottom=0.12)
    return fig


def plot_waterfall(
    model: Any,
    x_row: np.ndarray,
    *,
    class_index: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Compact single-row waterfall plot."""
    est = shap_tree_estimator(model)
    x = np.asarray(x_row, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    explainer = shap.TreeExplainer(est)
    explanation = explainer(x)

    if class_index is not None and len(explanation.shape) == 3:
        row_exp = explanation[0, :, class_index]
    else:
        row_exp = explanation[0]

    if feature_names is not None and hasattr(row_exp, "feature_names"):
        row_exp.feature_names = _display_names(feature_names)

    fig = plt.figure(figsize=(6.8, 3.0), dpi=96)
    shap.plots.waterfall(row_exp, show=False, max_display=12)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    return fig
