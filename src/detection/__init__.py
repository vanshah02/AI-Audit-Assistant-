# src/detection/__init__.py
"""Detection package exposing ready‑to‑use detectors.
"""

from .models import IsolationForestDetector, AutoencoderDetector, combine_scores
from .rules import duplicate_detection, split_invoice_detection
