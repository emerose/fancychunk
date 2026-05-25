"""Stage 2 — chunklet grouping (SPEC-CHUNK-2xx).

Public entry point: ``split_chunklets``.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from . import _constants as C
from ._markdown import compute_line_starts, line_of_offset, openers_by_line
from ._telemetry import get_tracer
from ._typing import Vector
from .errors import OversizedSentenceError, ValidationError


def split_chunklets(
    sentences: list[str],
    max_size: int = C.DEFAULT_MAX_SIZE_CHARS,
    boundary_cost: Callable[[Vector], float] | None = None,
    statement_cost: Callable[[float], float] | None = None,
) -> list[str]:
    """Group consecutive ``sentences`` into chunklets.

    Implements ``docs/specs/02-chunklet-grouping.md``.
    """
    if max_size <= 0:
        raise ValidationError("max_size must be positive")

    with get_tracer().start_as_current_span("fancychunk.split_chunklets") as span:
        span.set_attribute("fancychunk.sentences.count", len(sentences))
        span.set_attribute("fancychunk.max_size", max_size)
        span.set_attribute(
            "fancychunk.custom_costs",
            boundary_cost is not None or statement_cost is not None,
        )

        # SPEC-CHUNK-260 — empty input.
        if not sentences:
            span.set_attribute("fancychunk.chunklets.count", 0)
            span.set_attribute("fancychunk.short_circuit", "empty")
            return []
        # SPEC-CHUNK-261 — single-sentence input passes through unchanged.
        if len(sentences) == 1:
            span.set_attribute("fancychunk.chunklets.count", 1)
            span.set_attribute("fancychunk.short_circuit", "single_sentence")
            return [sentences[0]]

        # SPEC-CHUNK-263 — reject if any sentence alone exceeds max_size.
        for idx, s in enumerate(sentences):
            if len(s) > max_size:
                raise OversizedSentenceError(
                    f"sentence {idx} has length {len(s)} > max_size {max_size}"
                )

        bcost = boundary_cost if boundary_cost is not None else _default_boundary_cost
        scost = statement_cost if statement_cost is not None else _default_statement_cost

        probas = _per_sentence_boundary_probas(sentences)
        statements = _statement_counts([_word_count(s) for s in sentences])

        chunklets = _dp_partition(sentences, max_size, probas, statements, bcost, scost)
        span.set_attribute("fancychunk.chunklets.count", len(chunklets))
        return chunklets


def _default_boundary_cost(p: Vector) -> float:
    """SPEC-CHUNK-220 — ``(1 - p[0]) + sum(p[1:])``."""
    if p.size == 0:
        return 0.0
    return float((1.0 - p[0]) + np.sum(p[1:]))


def _default_statement_cost(s: float) -> float:
    """SPEC-CHUNK-221 — quadratic-deviation cost."""
    denom = math.sqrt(max(s, C.STATEMENT_COST_FLOOR))
    return C.STATEMENT_COST_SCALE * (s - C.TARGET_STATEMENTS_PER_CHUNKLET) ** 2 / denom


def _word_count(sentence: str) -> int:
    """Whitespace-separated token count."""
    return len(sentence.split())


def _statement_counts(word_counts: list[int]) -> list[float]:
    """SPEC-CHUNK-230 — convert per-sentence word counts to statement
    counts via a piecewise-linear function anchored at the document's
    word-count quartiles.
    """
    arr = np.asarray(word_counts, dtype=np.float64)
    q25 = float(np.percentile(arr, 25, method="linear"))
    q75 = float(np.percentile(arr, 75, method="linear"))
    q25 = max(q25, C.MIN_Q25_WORDS)
    q75 = max(q75, q25 + C.MIN_Q25_WORDS)

    out: list[float] = []
    for n in word_counts:
        if n <= q25:
            out.append(C.STATEMENTS_AT_Q25 * n / q25)
        else:
            out.append(
                C.STATEMENTS_AT_Q25
                + C.QUARTILE_GAP_STATEMENTS * (n - q25) / (q75 - q25)
            )
    return out


def _per_sentence_boundary_probas(sentences: list[str]) -> Vector:
    """SPEC-CHUNK-240 + SPEC-CHUNK-241 — per-sentence boundary probability.

    Determined from the *joined* document: each sentence's probability
    depends on the Markdown tokens that open on the source line of
    that sentence's first non-whitespace character. Then SPEC-CHUNK-241
    suppresses non-zero values inside contiguous runs, keeping only
    the strongest in each run.
    """
    document = "".join(sentences)
    line_starts = compute_line_starts(document)
    openers = openers_by_line(document)

    probas = np.zeros(len(sentences), dtype=np.float64)
    pos = 0
    for i, s in enumerate(sentences):
        # Find the first non-whitespace character in the sentence.
        offset = pos
        while offset < pos + len(s) and document[offset].isspace():
            offset += 1
        if offset == pos + len(s):
            # All-whitespace sentence — fall back to its start position.
            offset = pos
        line = line_of_offset(line_starts, offset)
        types = openers.get(line, set())
        probas[i] = _strength_for_openers(types)
        pos += len(s)

    # SPEC-CHUNK-241 — suppress consecutive non-zero values.
    _suppress_consecutive_nonzeros(probas)
    return probas


def _strength_for_openers(types: set[str]) -> float:
    """SPEC-CHUNK-240 — strongest applicable token-type weight."""
    if "heading_open" in types:
        return C.BOUNDARY_STRENGTH_HEADING
    if "blockquote_open" in types:
        return C.BOUNDARY_STRENGTH_BLOCKQUOTE
    if "bullet_list_open" in types or "ordered_list_open" in types:
        # List-opening shadows the accompanying paragraph_open.
        return C.BOUNDARY_STRENGTH_LIST
    if "paragraph_open" in types:
        return C.BOUNDARY_STRENGTH_PARAGRAPH
    return 0.0


def _suppress_consecutive_nonzeros(probas: Vector) -> None:
    """Mutate ``probas`` in place: within each maximal run of non-zero
    entries keep only the (earliest) maximum, zero the rest.
    """
    n = len(probas)
    i = 0
    while i < n:
        if probas[i] == 0.0:
            i += 1
            continue
        j = i
        while j < n and probas[j] != 0.0:
            j += 1
        # Run [i, j). Find earliest argmax.
        max_val = float(probas[i])
        max_idx = i
        for k in range(i + 1, j):
            if float(probas[k]) > max_val:
                max_val = float(probas[k])
                max_idx = k
        for k in range(i, j):
            if k != max_idx:
                probas[k] = 0.0
        i = j


def _dp_partition(
    sentences: list[str],
    max_size: int,
    probas: Vector,
    statements: list[float],
    boundary_cost: Callable[[Vector], float],
    statement_cost: Callable[[float], float],
) -> list[str]:
    """SPEC-CHUNK-210 / -251 — minimize total cost via 1-D DP.

    ``dp[i]`` is the minimum cost over partitions of ``sentences[:i]``;
    transition tries every prior cut ``j`` such that the chunklet
    ``sentences[j:i]`` fits in ``max_size`` characters. Ties resolve to
    the smallest predecessor index (SPEC-CHUNK-251).

    For the *default* cost functions the chunklet cost is a closed
    form over cumulative-sum arrays, so the inner ``j`` loop is a
    single numpy expression. For *custom* cost callables the inner
    loop calls them per candidate ``j``.
    """
    n = len(sentences)
    lengths_np: NDArray[np.int64] = np.fromiter(
        (len(s) for s in sentences), dtype=np.int64, count=n
    )
    cum_len_np: NDArray[np.int64] = np.concatenate(
        ([np.int64(0)], np.cumsum(lengths_np))
    )
    stmt_np: NDArray[np.float64] = np.asarray(statements, dtype=np.float64)
    cum_stmt_np: NDArray[np.float64] = np.concatenate(
        ([np.float64(0.0)], np.cumsum(stmt_np))
    )
    cum_proba_np: NDArray[np.float64] = np.concatenate(
        ([np.float64(0.0)], np.cumsum(probas))
    )

    using_defaults = (
        boundary_cost is _default_boundary_cost
        and statement_cost is _default_statement_cost
    )

    inf = np.inf
    dp_cost: NDArray[np.float64] = np.full(n + 1, inf, dtype=np.float64)
    dp_prev: NDArray[np.int64] = np.full(n + 1, -1, dtype=np.int64)
    dp_cost[0] = 0.0

    for i in range(1, n + 1):
        # Find smallest j such that the chunklet sentences[j:i] fits.
        # Equivalent to cum_len[i] - cum_len[j] <= max_size, i.e.
        # cum_len[j] >= cum_len[i] - max_size.
        threshold_val = int(cum_len_np[i]) - max_size
        # j must lie in [j_lo, i-1].
        j_lo = int(np.searchsorted(cum_len_np, threshold_val, side="left"))
        j_hi = i  # exclusive
        if j_lo >= j_hi:
            continue
        if using_defaults:
            costs = _vectorized_default_cost(
                cum_proba_np, cum_stmt_np, probas, j_lo, j_hi, i
            )
            candidates = dp_cost[j_lo:j_hi] + costs
        else:
            candidates = np.empty(j_hi - j_lo, dtype=np.float64)
            for idx, j in enumerate(range(j_lo, j_hi)):
                chunklet_probs = probas[j:i]
                chunklet_stmt = float(cum_stmt_np[i] - cum_stmt_np[j])
                candidates[idx] = (
                    float(dp_cost[j])
                    + boundary_cost(chunklet_probs)
                    + statement_cost(chunklet_stmt)
                )
        # np.argmin returns smallest index on ties → smallest j wins.
        local_argmin = int(np.argmin(candidates))
        dp_cost[i] = float(candidates[local_argmin])
        dp_prev[i] = j_lo + local_argmin

    # Unreachable in practice: every sentence's length is ≤ max_size
    # (validated above), so the per-sentence chunklet partition is
    # always feasible and dp_cost[n] is finite. Kept as an assertion
    # so a future bug surfaces as an internal error rather than a
    # confusing user-facing OversizedSentenceError.
    assert np.isfinite(dp_cost[n]), "internal: chunklet DP left dp_cost[n] non-finite"

    cuts: list[int] = []
    i = n
    while i > 0:
        j = int(dp_prev[i])
        cuts.append(j)
        i = j
    cuts.reverse()
    cuts.append(n)

    out: list[str] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        out.append("".join(sentences[a:b]))
    return out


def _vectorized_default_cost(
    cum_proba: NDArray[np.float64],
    cum_stmt: NDArray[np.float64],
    probas: Vector,
    j_lo: int,
    j_hi: int,  # exclusive
    i: int,
) -> NDArray[np.float64]:
    """Default ``boundary_cost`` + ``statement_cost`` evaluated for
    every candidate predecessor ``j`` in ``[j_lo, j_hi)`` against the
    fixed right endpoint ``i``, all in one shot.

    Boundary cost: ``(1 - probas[j]) + (cum_proba[i] - cum_proba[j+1])``.
    Statement cost: the closed form from SPEC-CHUNK-221, applied to
    the stmt-sum ``cum_stmt[i] - cum_stmt[j]``.
    """
    j_idx = np.arange(j_lo, j_hi, dtype=np.int64)
    p_at_j = probas[j_idx]
    sum_after_j = cum_proba[i] - cum_proba[j_idx + 1]
    bcost = (1.0 - p_at_j) + sum_after_j

    stmts = cum_stmt[i] - cum_stmt[j_idx]
    clamped = np.maximum(stmts, C.STATEMENT_COST_FLOOR)
    scost = (
        C.STATEMENT_COST_SCALE
        * (stmts - C.TARGET_STATEMENTS_PER_CHUNKLET) ** 2
        / np.sqrt(clamped)
    )
    return bcost + scost
