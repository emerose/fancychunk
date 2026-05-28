"""Bundled sentence segmenters for fancychunk.

Factories returning a configured :class:`SaTSegmenter` for each
recommended Segment Any Text (SaT) checkpoint, plus the rule-based
fallback. Pass the result anywhere a segmenter is accepted —
``split_sentences(doc, segmenter=...)``,
``chunk_document(..., segmenter=...)``,
``chunk_documents(..., segmenter=...)``.

The three SaT checkpoints trade speed for segmentation quality. Numbers
below are RTX 3090, batched (the ``chunk_documents`` production path),
``weighting="hat"``; see ``benchmarks/sat-model-selection.md`` for the
full quality/throughput data.

* :func:`sat_3l` — ``sat-3l-sm``. Fastest (~3.2× :func:`sat_9l`), but
  mis-segments some scientific-prose constructs: abbreviation
  references (``Tab. TABREF21``, ``Eq. EQREF9``) and years before a
  capitalised word (``SemEval-2014 Task``).
* :func:`sat_9l` — ``sat-9l-sm``. **The default.** Artifact-free like
  ``sat-12l-sm`` and tracks its boundaries closely (corpus F1 0.97),
  at ~1.3× the throughput.
* :func:`sat_12l` — ``sat-12l-sm``. Highest quality, slowest.
* :func:`punctuation` — the zero-dependency rule-based fallback
  (``.!?`` followed by whitespace); no model download.

All SaT factories forward keyword arguments to :class:`SaTSegmenter`,
so e.g. ``sat_9l(device="cuda")`` or ``sat_12l(ort_kwargs=...)`` work.
Weights download lazily on first use (Hugging Face cache).
"""

from __future__ import annotations

from typing import Any

from ._segmenter import SaTSegmenter, SentenceSegmenter, punctuation_segmenter


def sat_3l(**kwargs: Any) -> SaTSegmenter:
    """``sat-3l-sm`` — fastest SaT checkpoint, lower scientific-prose quality."""
    return SaTSegmenter("sat-3l-sm", **kwargs)


def sat_9l(**kwargs: Any) -> SaTSegmenter:
    """``sat-9l-sm`` — the default: artifact-free, ~1.3× faster than 12l."""
    return SaTSegmenter("sat-9l-sm", **kwargs)


def sat_12l(**kwargs: Any) -> SaTSegmenter:
    """``sat-12l-sm`` — highest-quality SaT checkpoint, slowest."""
    return SaTSegmenter("sat-12l-sm", **kwargs)


def sat_default(**kwargs: Any) -> SaTSegmenter:
    """The default SaT segmenter (currently ``sat-9l-sm``) — equivalent
    to what ``split_sentences`` uses when no ``segmenter`` is passed."""
    return SaTSegmenter(**kwargs)  # SaTSegmenter() applies the default model


def punctuation() -> SentenceSegmenter:
    """The rule-based fallback segmenter (no model download)."""
    return punctuation_segmenter


__all__ = [
    "sat_3l",
    "sat_9l",
    "sat_12l",
    "sat_default",
    "punctuation",
    "SaTSegmenter",
    "punctuation_segmenter",
]
