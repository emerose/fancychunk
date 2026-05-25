"""Stage 1 — sentence splitting (SPEC-CHUNK-1xx).

Public entry point: ``split_sentences``.

Internals:

* ``_default_known_overrides`` builds the SPEC-CHUNK-108 heading
  override vector (a per-character vector where finite positions force
  the boundary probability and ``NaN`` defers to the predicted value).
* ``_merge_known`` combines predicted and known vectors.
* ``_whitespace_trailing`` applies the SPEC-CHUNK-109 rule to bias
  boundaries onto trailing whitespace.
* ``_dp_split`` solves the constrained maximization (SPEC-CHUNK-110).
"""

from __future__ import annotations

import types
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from . import _constants as C
from ._markdown import heading_spans
from ._segmenter import SentenceSegmenter, make_segmenter
from ._telemetry import get_tracer
from ._typing import Vector
from .errors import UnsplittableDocumentError, ValidationError

_KnownArg = Vector | Callable[[str], Vector] | None


def _describe_callable(obj: object) -> str:
    """Best-effort human-readable name for a callable, used in trace attrs."""
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return type(obj).__name__


def split_sentences(
    document: str,
    min_len: int = 4,
    max_len: int | None = None,
    known_boundary_probas: _KnownArg = None,
    *,
    segmenter: SentenceSegmenter | None = None,
) -> list[str]:
    """Partition ``document`` into a list of sentences.

    See ``docs/specs/01-sentence-splitting.md`` for the contract. The
    optional keyword-only ``segmenter`` parameter swaps the default
    rule-based punctuation segmenter for any callable that returns a
    per-character probability vector (SPEC-CHUNK-106).
    """
    if min_len < 0:
        raise ValidationError("min_len must be non-negative")
    if max_len is not None and max_len <= 0:
        raise ValidationError("max_len must be positive when set")

    with get_tracer().start_as_current_span("fancychunk.split_sentences") as span:
        span.set_attribute("fancychunk.document.length", len(document))
        span.set_attribute("fancychunk.min_len", min_len)
        if max_len is not None:
            span.set_attribute("fancychunk.max_len", max_len)
        resolved = make_segmenter(segmenter)
        span.set_attribute("fancychunk.segmenter", _describe_callable(resolved))

        # SPEC-CHUNK-117 — empty document.
        if not document:
            span.set_attribute("fancychunk.sentences.count", 0)
            span.set_attribute("fancychunk.short_circuit", "empty")
            return []

        # SPEC-CHUNK-101 — every sentence must contain a non-whitespace
        # character. A document consisting only of whitespace cannot
        # produce any conforming sentence, so it is treated the same as
        # the empty document.
        if not document.strip():
            span.set_attribute("fancychunk.sentences.count", 0)
            span.set_attribute("fancychunk.short_circuit", "whitespace_only")
            return []

        # SPEC-CHUNK-114 — short-circuit when document is no longer than min_len.
        if len(document) <= min_len:
            span.set_attribute("fancychunk.sentences.count", 1)
            span.set_attribute("fancychunk.short_circuit", "below_min_len")
            return [document]

        n = len(document)
        predicted = np.asarray(resolved(document), dtype=np.float64)
        if predicted.shape != (n,):
            raise ValidationError(
                f"segmenter returned shape {predicted.shape}; expected ({n},)"
            )

        known = _resolve_known(known_boundary_probas, document, n)
        merged = _merge_known(predicted, known)
        final = _whitespace_trailing(document, merged)

        boundaries = _dp_split(final, min_len=min_len, max_len=max_len)
        out = _slice(document, boundaries)
        span.set_attribute("fancychunk.sentences.count", len(out))
        return out




def _resolve_known(
    known_boundary_probas: _KnownArg, document: str, n: int
) -> Vector:
    if known_boundary_probas is None:
        return _default_known_overrides(document, n)
    if callable(known_boundary_probas):
        result = np.asarray(known_boundary_probas(document), dtype=np.float64)
    else:
        result = np.asarray(known_boundary_probas, dtype=np.float64)
    if result.shape != (n,):
        raise ValidationError(
            f"known_boundary_probas has shape {result.shape}; expected ({n},)"
        )
    return result


def _default_known_overrides(document: str, n: int) -> Vector:
    """SPEC-CHUNK-108 — force standalone heading sentences."""
    known = np.full(n, np.nan, dtype=np.float64)
    for span in heading_spans(document):
        if span.last < span.first:
            # Empty heading body (`# \n` and similar): skip the
            # in-body and last-position writes.
            continue
        if span.first > 0:
            known[span.first - 1] = 1.0
        for k in range(span.first, span.last):
            known[k] = 0.0
        known[span.last] = 1.0
    return known


def _merge_known(predicted: Vector, known: Vector) -> Vector:
    out = predicted.copy()
    mask = np.isfinite(known)
    out[mask] = known[mask]
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _whitespace_trailing(document: str, p: Vector) -> Vector:
    """SPEC-CHUNK-109 — over every maximal whitespace run ``[i, j)``,
    pin position ``j-1`` to the maximum probability of the *extended
    run* and every earlier position (including ``i-1`` when present)
    to the minimum of the same set.

    The extended run ``[i', j)`` is ``[i-1, j)`` when ``i > 0`` and
    ``[i, j)`` otherwise; including the preceding non-whitespace
    position is what carries SPEC-CHUNK-108's heading-end probability
    across the trailing newline so the heading sentence consumes its
    blank-line tail.
    """
    p = p.copy()
    n = len(document)
    i = 0
    while i < n:
        if not document[i].isspace():
            i += 1
            continue
        j = i
        while j < n and document[j].isspace():
            j += 1
        include_prev = i > 0
        if not include_prev:
            i = j
            continue
        # Extended run: [i-1, j). Min/max include position i-1.
        values = [float(x) for x in p[i:j]]
        values.append(float(p[i - 1]))
        mn = min(values)
        mx = max(values)
        p[i - 1] = mn
        for k in range(i, j - 1):
            p[k] = mn
        if j - 1 >= i:
            p[j - 1] = mx
        i = j
    return p


def _dp_split(
    p: Vector, min_len: int, max_len: int | None
) -> list[int]:
    """Return the list of boundary indices (each in ``[0, N-1)``).

    Maximizes ``Σ (p[k] - BOUNDARY_SCORE_THRESHOLD)`` over chosen
    boundaries (SPEC-CHUNK-110), enforcing
    ``min_len <= len(sentence) <= max_len`` for every sentence and
    tie-breaking by smallest predecessor index (SPEC-CHUNK-113).
    Raises ``UnsplittableDocumentError`` when the length constraints
    are infeasible.

    The recurrence is
    ``dp_score[i] = max_{j ∈ [i - max_eff, i - min_len]} dp_score[j] + score_at_i``
    where ``score_at_i = p[i-1] - threshold`` for ``i < n`` and ``0``
    for ``i == n``. Because ``score_at_i`` is a constant for the
    transition into state ``i``, the argmax over the valid ``j``
    window is independent of the score and can be computed with a
    single ``numpy`` slice + ``np.argmax`` per state.
    """
    n = len(p)
    if n == 0:
        return []
    threshold = C.BOUNDARY_SCORE_THRESHOLD
    max_eff = n if max_len is None else max_len

    neg_inf = -np.inf
    dp_score: NDArray[np.float64] = np.full(n + 1, neg_inf, dtype=np.float64)
    dp_prev: NDArray[np.int64] = np.full(n + 1, -1, dtype=np.int64)
    dp_score[0] = 0.0

    for i in range(1, n + 1):
        lo = max(0, i - max_eff)
        hi = i - min_len
        if hi < 0:
            continue
        if lo > hi:
            continue
        score_at_i = (float(p[i - 1]) - threshold) if i < n else 0.0
        window = dp_score[lo : hi + 1]
        # np.argmax returns the smallest index on ties — matches
        # SPEC-CHUNK-113's smallest-predecessor rule.
        local_argmax = int(np.argmax(window))
        best_base = float(window[local_argmax])
        if best_base == neg_inf:
            continue
        dp_score[i] = best_base + score_at_i
        dp_prev[i] = lo + local_argmax

    if dp_score[n] == neg_inf:
        raise UnsplittableDocumentError(
            "no sentence partition satisfies the configured length constraints"
        )

    boundaries: list[int] = []
    i = n
    while i > 0:
        j = int(dp_prev[i])
        if i < n:
            boundaries.append(i - 1)
        i = j
    boundaries.reverse()
    return boundaries


def _slice(document: str, boundaries: list[int]) -> list[str]:
    """Slice ``document`` by ``boundaries`` (each is the last index of
    a sentence)."""
    out: list[str] = []
    start = 0
    for k in boundaries:
        out.append(document[start : k + 1])
        start = k + 1
    out.append(document[start:])
    return out
