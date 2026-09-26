# src/detection/models.py
"""Detection models and scoring utilities.
Implements:
- IsolationForest anomaly detector
- Simple autoencoder (MLP) for reconstruction error
- Normalization of scores to comparable percentile ranks
- Combined score calculation with explicit weights
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# --------------------------
# Isolation Forest wrapper
# --------------------------
class IsolationForestDetector:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)
        # IsolationForest returns negative scores (the lower, the more abnormal)
        raw_scores = -self.model.decision_function(X).reshape(-1, 1)
        self.scaler.fit(raw_scores)
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsolationForestDetector not fitted")
        raw = -self.model.decision_function(X).reshape(-1, 1)
        # Normalize to [0,1]
        return self.scaler.transform(raw).flatten()

# --------------------------
# Autoencoder wrapper (MLPRegressor as a simple proxy)
# --------------------------
class AutoencoderDetector:
    def __init__(self, hidden_layer_sizes=(64, 32, 64), max_iter=200, random_state=42):
        # MLPRegressor learns to reconstruct its input
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
        )
        self.scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        # Train to predict each column from itself
        self.model.fit(X, X)
        recon = self.model.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1).reshape(-1, 1)
        self.scaler.fit(errors)
        self.fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("AutoencoderDetector not fitted")
        recon = self.model.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1).reshape(-1, 1)
        # Higher reconstruction error → more abnormal, normalize to [0,1]
        return self.scaler.transform(errors).flatten()

# --------------------------
# Score combination utilities
# --------------------------
def combine_scores(isolation_score: np.ndarray, autoencoder_score: np.ndarray,
                   weight_iso: float = 0.6, weight_ae: float = 0.4) -> np.ndarray:
    """Combine two normalized anomaly scores.
    Both inputs are assumed to be in the range [0, 1].
    The weights must sum to 1 (or will be normalized).
    """
    w = np.array([weight_iso, weight_ae], dtype=float)
    w = w / w.sum()
    combined = w[0] * isolation_score + w[1] * autoencoder_score
    return combined
