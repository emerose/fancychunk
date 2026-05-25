"""Shared numpy type aliases used across stages.

Centralizing these prevents subtle drift (e.g., ``NDArray[np.float64]``
vs. ``NDArray[np.floating]``) across the per-stage modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
