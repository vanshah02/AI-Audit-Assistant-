# src/explainability/shap_explain.py
"""Utilities for computing SHAP explanations for the ML‑based detectors.

We provide a thin wrapper that fits a ``shap.TreeExplainer`` to an
``IsolationForestDetector`` (tree‑based) and a ``shap.KernelExplainer`` to the
auto‑encoder (a black‑box regressor). The function returns a DataFrame where each
row corresponds to a transaction and each column is a feature contribution.
"""

import pandas as pd
import shap
import numpy as np
from ..detection.models import IsolationForestDetector, AutoencoderDetector


def _fit_shap_tree(model: IsolationForestDetector, X: pd.DataFrame):
    # IsolationForest uses an ensemble of trees; TreeExplainer works directly.
    explainer = shap.TreeExplainer(model.model)
    # shap values shape: (n_samples, n_features)
    shap_vals = explainer.shap_values(X)
    return pd.DataFrame(shap_vals, index=X.index, columns=X.columns)


def _fit_shap_kernel(model: AutoencoderDetector, X: pd.DataFrame):
    # Autoencoder is an MLPRegressor – use KernelExplainer with a background
    # sample (the mean of the data) for speed.
    background = X.mean().values.reshape(1, -1)
    # The model's predict returns reconstructed rows; we treat reconstruction
    # error as the target for SHAP.
    def predict_fn(data):
        recon = model.model.predict(pd.DataFrame(data, columns=X.columns))
        # Return mean squared error per row
        return np.mean((data - recon) ** 2, axis=1)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(X, nsamples=100)  # limit samples for speed
    return pd.DataFrame(shap_vals, index=X.index, columns=X.columns)


def compute_shap_values(
    iso_detector: IsolationForestDetector,
    ae_detector: AutoencoderDetector,
    X: pd.DataFrame,
) -> dict:
    """Compute SHAP contributions for both detectors.

    Returns a dict with keys ``"iso"`` and ``"ae"`` mapping to DataFrames of the
    same shape as ``X``.
    """
    iso_shap = _fit_shap_tree(iso_detector, X)
    ae_shap = _fit_shap_kernel(ae_detector, X)
    return {"iso": iso_shap, "ae": ae_shap}
