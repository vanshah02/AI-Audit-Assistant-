# src/detection/scoring.py
"""Scoring utilities for the anomaly detectors.

The MVP requires **normalization** of the raw IsolationForest and Autoencoder
outputs onto a common scale before combining them with explicit weights.  A
reference population (e.g., historic transactions) can be supplied; if none is
provided we fall back to min‑max scaling based on the current dataset.

The public API consists of:

* ``ReferenceStats`` – simple container for min/max values of each detector.
* ``ScoreNormalizer`` – fits a ``MinMaxScaler`` using the reference stats and
  transforms raw scores to the interval ``[0, 1]``.
* ``combine_scores`` – weighted combination (default 0.6 IsolationForest,
  0.4 Autoencoder).  We also expose ``DEFAULT_WEIGHTS`` for callers.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Reference statistics container – in a real system these would be derived from a
# large historical baseline.  For the MVP we allow the caller to pass ``None``
# and the normalizer will fit on‑the‑fly using the current batch.
# ---------------------------------------------------------------------------
@dataclass
class ReferenceStats:
    iso_min: float = 0.0
    iso_max: float = 1.0
    ae_min: float = 0.0
    ae_max: float = 1.0

    @classmethod
    def from_batch(cls, iso_scores: np.ndarray, ae_scores: np.ndarray) -> "ReferenceStats":
        return cls(
            iso_min=float(np.min(iso_scores)),
            iso_max=float(np.max(iso_scores)),
            ae_min=float(np.min(ae_scores)),
            ae_max=float(np.max(ae_scores)),
        )

# ---------------------------------------------------------------------------
# Normalizer – fits a separate MinMax scaler for each detector so that the
# resulting values are comparable regardless of the underlying algorithm.
# ---------------------------------------------------------------------------
class ScoreNormalizer:
    def __init__(self, reference: ReferenceStats | None = None):
        self.reference = reference
        self.iso_scaler = MinMaxScaler()
        self.ae_scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, iso_scores: np.ndarray, ae_scores: np.ndarray) -> "ScoreNormalizer":
        if self.reference is None:
            # Use the batch itself as the reference population
            self.reference = ReferenceStats.from_batch(iso_scores, ae_scores)
        # Fit scalers on the *reference* min/max values to guarantee identical
        # transforms for future batches.
        self.iso_scaler.fit(np.array([[self.reference.iso_min], [self.reference.iso_max]]))
        self.ae_scaler.fit(np.array([[self.reference.ae_min], [self.reference.ae_max]]))
        self.fitted = True
        return self

    def transform(self, iso_scores: np.ndarray, ae_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("ScoreNormalizer must be fitted before transform")
        iso_norm = self.iso_scaler.transform(iso_scores.reshape(-1, 1)).flatten()
        ae_norm = self.ae_scaler.transform(ae_scores.reshape(-1, 1)).flatten()
        return iso_norm, ae_norm

# ---------------------------------------------------------------------------
# Combination helper – explicit weighting scheme (documented default).
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {"iso": 0.6, "ae": 0.4}


def combine_scores(
    iso_norm: np.ndarray,
    ae_norm: np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """Return a weighted sum of the two normalized score vectors.

    The function normalises the supplied ``weights`` dict so they sum to ``1``.
    ``weights`` keys must be ``"iso"`` and ``"ae"``.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w_iso = float(weights.get("iso", 0.0))
    w_ae = float(weights.get("ae", 0.0))
    total = w_iso + w_ae
    if total == 0:
        raise ValueError("At least one weight must be non‑zero")
    w_iso, w_ae = w_iso / total, w_ae / total
    return w_iso * iso_norm + w_ae * ae_norm
