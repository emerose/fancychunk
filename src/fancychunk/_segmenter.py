"""Sentence-segmentation model interface and default implementations.

Two segmenters are bundled:

* :class:`SaTSegmenter` (the default) wraps a Segment Any Text (SaT)
  model from `wtpsplit-lite` and returns per-character boundary
  probabilities exactly as SPEC-CHUNK-106 prescribes. The 408 MB
  ``sat-3l-sm`` weights download lazily on first call so importing
  ``fancychunk`` stays cheap.
* :func:`punctuation_segmenter` is a no-dependencies fallback that
  marks ``.``/``!``/``?`` followed by whitespace or end-of-document.

Either is a valid SentenceSegmenter; callers may pass their own
through the keyword-only ``segmenter`` parameter on
``split_sentences``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from wtpsplit_lite import SaT


Vector = NDArray[np.float64]


class SentenceSegmenter(Protocol):
    """Callable mapping a document to a per-character boundary probability.

    The returned array has length ``len(document)`` and dtype
    convertible to ``float64``.
    """

    def __call__(self, document: str) -> Vector: ...


_TERMINATORS = frozenset({".", "!", "?"})
_DEFAULT_BOUNDARY_PROB = 0.9


def punctuation_segmenter(document: str) -> Vector:
    """Rule-based fallback segmenter.

    Assigns ``_DEFAULT_BOUNDARY_PROB`` at every character that is a
    sentence-final terminator (``.!?``) followed by whitespace or end
    of document; ``0.0`` elsewhere. Crude but adequate for tests that
    don't need real model output.
    """
    n = len(document)
    probs: Vector = np.zeros(n, dtype=np.float64)
    for i, ch in enumerate(document):
        if ch in _TERMINATORS:
            if i == n - 1 or document[i + 1].isspace():
                probs[i] = _DEFAULT_BOUNDARY_PROB
    return probs


_DEFAULT_SAT_MODEL = "sat-3l-sm"


class SaTSegmenter:
    """SPEC-CHUNK-106 segmenter backed by wtpsplit-lite's SaT model.

    The 408 MB ``sat-3l-sm`` weights are downloaded by Hugging Face on
    first use (subsequent calls use the cache); the import itself is
    cheap because the model is only loaded on the first
    ``__call__``. Instances are reusable and thread-safe for read
    (wtpsplit-lite's ONNX backend is itself reentrant).
    """

    def __init__(self, model_name: str = _DEFAULT_SAT_MODEL) -> None:
        self.model_name: str = model_name
        self._sat: SaT | None = None

    def _ensure_loaded(self) -> SaT:
        if self._sat is None:
            # Local import keeps ``import fancychunk`` lightweight even
            # when the SaT weights aren't yet cached.
            from wtpsplit_lite import SaT as _SaT

            self._sat = _SaT(self.model_name)
        return self._sat

    def __call__(self, document: str) -> Vector:
        sat = self._ensure_loaded()
        raw = sat.predict_proba(document)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != len(document):
            raise RuntimeError(
                f"SaT returned shape {arr.shape}; expected ({len(document)},)"
            )
        return arr


# Module-level default singleton (lazy weight load on first call).
_default_segmenter: SaTSegmenter | None = None


def get_default_segmenter() -> SaTSegmenter:
    """Return the process-wide default segmenter (the SaT singleton)."""
    global _default_segmenter
    if _default_segmenter is None:
        _default_segmenter = SaTSegmenter()
    return _default_segmenter


def make_segmenter(
    segmenter: SentenceSegmenter | None,
) -> Callable[[str], Vector]:
    """Resolve ``segmenter`` to a callable, defaulting to SaT."""
    if segmenter is None:
        return get_default_segmenter()
    return segmenter
