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

import re
import types
from collections import deque
from typing import Callable

import numpy as np
from numpy.typing import NDArray

_WHITESPACE_RUN = re.compile(r"\s+")

# A digit immediately followed by a whitespace run and then a
# non-whitespace character. Used to detect the "<numeral> <Capital>"
# segmentation artifact (see ``_suppress_numeral_boundary_artifacts``).
_NUMERAL_WS_NONSPACE = re.compile(r"[0-9]\s+\S")

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
        tracer = get_tracer()
        with tracer.start_as_current_span("fancychunk.sentences.segmenter"):
            predicted = np.asarray(resolved(document), dtype=np.float64)
        if predicted.shape != (n,):
            raise ValidationError(
                f"segmenter returned shape {predicted.shape}; expected ({n},)"
            )

        with tracer.start_as_current_span("fancychunk.sentences.numeral_artifact"):
            predicted = _suppress_numeral_boundary_artifacts(document, predicted)
        with tracer.start_as_current_span("fancychunk.sentences.heading_override"):
            known = _resolve_known(known_boundary_probas, document, n)
        with tracer.start_as_current_span("fancychunk.sentences.merge"):
            merged = _merge_known(predicted, known)
        with tracer.start_as_current_span("fancychunk.sentences.whitespace_trailing"):
            final = _whitespace_trailing(document, merged)

        with tracer.start_as_current_span("fancychunk.sentences.dp"):
            boundaries = _dp_split(final, min_len=min_len, max_len=max_len)
        with tracer.start_as_current_span("fancychunk.sentences.slice"):
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
    """SPEC-CHUNK-108 — force standalone heading sentences.

    Uses slice assignment instead of a per-character loop over heading
    bodies; heading-heavy documents see ~3× less time in this function.
    """
    known = np.full(n, np.nan, dtype=np.float64)
    for span in heading_spans(document):
        if span.last < span.first:
            # Empty heading body (`# \n` and similar): skip the
            # in-body and last-position writes.
            continue
        if span.first > 0:
            known[span.first - 1] = 1.0
        if span.last > span.first:
            known[span.first : span.last] = 0.0
        known[span.last] = 1.0
    return known


def _suppress_numeral_boundary_artifacts(
    document: str, predicted: Vector
) -> Vector:
    """Correct a SaT segmentation artifact: a numeral (typically a
    year) directly followed by whitespace and a *capitalized* word is
    assigned a spuriously high boundary probability — e.g. the final
    ``4`` of ``SemEval-2014`` in ``"...SemEval-2014 Task 4..."`` scores
    ~0.5, well above ``BOUNDARY_SCORE_THRESHOLD``, so the DP breaks the
    sentence mid-phrase. This pattern (``"WMT 2016 Task"``,
    ``"ICLR 2020 Workshop"``, …) is common in scientific writing.

    A genuine sentence end at a number would sit on the *terminating
    punctuation* (``"...in 2014. Later, we..."`` — the boundary is on
    the ``.``), never on the digit itself, so forcing the predicted
    probability to ``0`` at the digit position removes the artifact
    without suppressing any legitimate break. The rule fires only when
    the following token is capitalized, matching the model's own
    trigger — a numeral followed by a lowercase word already scores
    ~0, so the override is a no-op there.

    Applied to the *predicted* vector before the heading override and
    merge, so an explicit caller-supplied ``known_boundary_probas``
    (SPEC-CHUNK-107) still takes precedence at these positions.
    """
    out = predicted
    copied = False
    for m in _NUMERAL_WS_NONSPACE.finditer(document):
        # ``m`` spans ``<digit><whitespace...><non-space>``; the final
        # character of the match is the next token's first character.
        if not document[m.end() - 1].isupper():
            continue
        if not copied:
            out = predicted.copy()
            copied = True
        out[m.start()] = 0.0
    return out


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

    Implementation: ``re.finditer`` is the C-implemented regex engine
    discovering whitespace runs; per-run min/max use numpy slice
    operations rather than Python char-by-char iteration.
    """
    n = len(document)
    if n == 0:
        return p.copy()
    p = p.copy()

    for match in _WHITESPACE_RUN.finditer(document):
        run_start, run_end = match.span()
        if run_start == 0:
            # Run flanked by EOS on the left; nothing to attach.
            continue
        # Extended run: [run_start-1, run_end). Min/max include the
        # preceding non-whitespace position. ``tolist`` + Python
        # ``min``/``max`` beats numpy's ``.min()``/``.max()`` per-call
        # overhead for the 2-3 element windows typical here.
        ext_lo = run_start - 1
        values = p[ext_lo:run_end].tolist()
        mn = min(values)
        mx = max(values)
        p[ext_lo : run_end - 1] = mn
        p[run_end - 1] = mx
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
    ``dp_score[i] = max_{j ∈ [lo, hi]} dp_score[j] + score_at_i``
    where ``lo = max(0, i - max_eff)``, ``hi = i - min_len``, and
    ``score_at_i = p[i-1] - threshold`` for ``i < n`` (``0`` at the
    final state). Both ``lo`` and ``hi`` advance monotonically with
    ``i`` — i.e. this is a sliding window — so the argmax can be
    maintained in **amortized O(1)** per state using a monotonic
    deque, giving O(N) total work versus the naive O(N × max_len).

    Deque invariant: indices are stored in increasing order; the
    ``dp_score`` values at those indices are non-increasing
    front-to-back. New indices pop strictly-smaller predecessors off
    the back (the ``<`` ensures earlier equal-score indices stay,
    preserving SPEC-CHUNK-113's smallest-predecessor tie-break). The
    front is therefore always the argmax with smallest-index ties.
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

    dq: deque[int] = deque()
    last_added = -1  # largest j currently considered for the deque

    for i in range(1, n + 1):
        lo = max(0, i - max_eff)
        hi = i - min_len
        if hi < 0:
            continue

        # Extend the deque rightward to include j = last_added+1..hi.
        # By the time we process state i, dp_score[j] for j ≤ i-1 has
        # been finalized in earlier iterations.
        while last_added < hi:
            last_added += 1
            j = last_added
            jval = float(dp_score[j])
            while dq and float(dp_score[dq[-1]]) < jval:
                dq.pop()
            dq.append(j)

        # Shrink the deque leftward: drop indices below the new lo.
        while dq and dq[0] < lo:
            dq.popleft()

        if not dq:
            continue
        best_prev = dq[0]
        best_base = float(dp_score[best_prev])
        if best_base == neg_inf:
            continue

        score_at_i = (float(p[i - 1]) - threshold) if i < n else 0.0
        dp_score[i] = best_base + score_at_i
        dp_prev[i] = best_prev

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
