"""Stage 3 — semantic chunking (SPEC-CHUNK-3xx).

Public entry point: ``split_chunks``.
"""

from __future__ import annotations

import math

import numpy as np
from markdown_it import MarkdownIt
from numpy.typing import NDArray

from . import _constants as C
from .errors import (
    OptimizationFailedError,
    OversizedChunkletError,
    ValidationError,
    ZeroNormEmbeddingError,
)

Matrix = NDArray[np.floating]
Vector = NDArray[np.floating]


def split_chunks(
    chunklets: list[str],
    chunklet_embeddings: Matrix,
    max_size: int = C.DEFAULT_MAX_SIZE_CHARS,
) -> tuple[list[str], list[Matrix]]:
    """Partition ``chunklets`` (with embeddings) into chunks.

    Implements ``docs/specs/03-semantic-chunking.md``.
    """
    if max_size <= 0:
        raise ValidationError("max_size must be positive")

    emb = np.asarray(chunklet_embeddings)
    if emb.ndim != 2:
        raise ValidationError("chunklet_embeddings must be a 2-D matrix")
    if emb.shape[0] != len(chunklets):
        raise ValidationError(
            f"chunklet_embeddings has {emb.shape[0]} rows but chunklets has "
            f"{len(chunklets)} entries"
        )

    # SPEC-CHUNK-340 — empty input.
    if not chunklets:
        return [], []

    # SPEC-CHUNK-342 — zero-norm embedding.
    norms = np.linalg.norm(emb, axis=1)
    if np.any(norms == 0):
        raise ZeroNormEmbeddingError(
            "one or more chunklet embeddings have L2 norm 0"
        )

    # SPEC-CHUNK-341 — oversized chunklet.
    lengths = [len(c) for c in chunklets]
    for idx, ln in enumerate(lengths):
        if ln > max_size:
            raise OversizedChunkletError(
                f"chunklet {idx} has length {ln} > max_size {max_size}"
            )

    # SPEC-CHUNK-340 — single chunklet.
    if len(chunklets) == 1:
        return [chunklets[0]], [emb]

    # SPEC-CHUNK-340 — total fits.
    if sum(lengths) <= max_size:
        return ["".join(chunklets)], [emb]

    sim = _partition_similarities(emb, chunklets, lengths)
    chunks, splits = _solve_partition(chunklets, lengths, sim, max_size)
    # Build chunk_embeddings from the original input rows.
    chunk_embeddings: list[Matrix] = []
    for a, b in splits:
        chunk_embeddings.append(emb[a:b])
    return chunks, chunk_embeddings


def _partition_similarities(
    emb: Matrix, chunklets: list[str], lengths: list[int]
) -> Vector:
    """SPEC-CHUNK-320 — return ``sim[i]`` for each partition point ``i``
    in ``[0, N-2]`` (between chunklet ``i`` and chunklet ``i+1``).
    """
    n = emb.shape[0]
    # Step 1 — unit-normalize.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / norms

    # Step 2 — discourse-vector removal (with safeguards).
    corrected = _discourse_corrected(unit, lengths)

    # Step 3 — base partition similarity.
    epsilon = float(np.finfo(corrected.dtype).eps)
    floor = math.sqrt(epsilon)
    sim = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        dot = float(np.dot(corrected[i], corrected[i + 1]))
        val = (dot + 1.0) / 2.0
        sim[i] = max(val, floor)

    # Step 4 — heading-aware modification.
    _apply_heading_modification(sim, chunklets, floor)
    return sim


def _discourse_corrected(unit: Matrix, lengths: list[int]) -> Matrix:
    """SPEC-CHUNK-321 — project out the discourse vector when feasible.

    Falls back to the unit-normalized embeddings if there are fewer
    than two "typical" chunklets, or if the projection would zero any
    row (within machine epsilon).
    """
    n = unit.shape[0]
    if n < 2:
        return unit

    lens = np.asarray(lengths, dtype=np.float64)
    q_lower = float(np.percentile(lens, C.TYPICAL_CHUNKLET_LOWER_QUANTILE * 100, method="linear"))
    q_upper = float(np.percentile(lens, C.TYPICAL_CHUNKLET_UPPER_QUANTILE * 100, method="linear"))
    typical_mask = (lens >= q_lower) & (lens <= q_upper)
    typical_count = int(typical_mask.sum())
    if typical_count < 2:
        return unit

    typical_rows = unit[typical_mask]
    mean = typical_rows.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm == 0:
        return unit
    discourse = mean / mean_norm

    projections = unit @ discourse
    corrected = unit - np.outer(projections, discourse)
    epsilon = float(np.finfo(corrected.dtype).eps)
    row_norms = np.linalg.norm(corrected, axis=1)
    if np.any(row_norms < epsilon):
        return unit

    corrected = corrected / row_norms[:, np.newaxis]
    return corrected


# Module-level parser; CommonMark parsing is reentrant.
_MD_PARSER = MarkdownIt("commonmark")


def _is_heading(chunklet: str) -> bool:
    """SPEC-CHUNK-322 — chunklet is a heading iff its full block-level
    structure consists of exactly one heading element.

    Delegates to ``markdown-it-py`` so ATX and Setext forms are both
    recognised through the same parser stages 1 and 2 already use.
    A chunklet whose block-level tokens contain anything other than a
    single ``heading_open`` (e.g., a heading followed by body text,
    leading paragraph text, or no headings at all) returns ``False``.
    """
    block_opens = [t for t in _MD_PARSER.parse(chunklet) if t.type.endswith("_open")]
    return len(block_opens) == 1 and block_opens[0].type == "heading_open"


def _apply_heading_modification(sim: Vector, chunklets: list[str], floor: float) -> None:
    n = len(chunklets)
    previous_is_heading = False
    for i in range(n):
        if _is_heading(chunklets[i]):
            if i >= 1 and not previous_is_heading:
                sim[i - 1] = max(sim[i - 1] / C.HEADING_SPLIT_BEFORE_DIVISOR, floor)
            if i <= n - 2:
                sim[i] = C.HEADING_SPLIT_AFTER_FORBID
            previous_is_heading = True
        else:
            previous_is_heading = False


def _solve_partition(
    chunklets: list[str], lengths: list[int], sim: Vector, max_size: int
) -> tuple[list[str], list[tuple[int, int]]]:
    """SPEC-CHUNK-310/-311 — minimize total partition similarity under the
    covering constraint that every chunk fits in ``max_size``.

    Returns ``(chunks, ranges)`` where ``ranges[i] = (a, b)`` is the
    half-open chunklet range of ``chunks[i]``.

    ``dp_cost[i] = min_{j} dp_cost[j] + (sim[j-1] if j > 0 else 0)``
    over all ``j`` such that the chunk ``chunklets[j:i]`` fits in
    ``max_size``. With ``cum_len`` available, the feasibility window
    ``[j_lo, i)`` is a single ``np.searchsorted``; the argmin over
    the window is a single ``np.argmin`` (smallest-index tie-break
    matches SPEC-CHUNK-251).
    """
    n = len(chunklets)
    lengths_np: NDArray[np.int64] = np.asarray(lengths, dtype=np.int64)
    cum_len_np: NDArray[np.int64] = np.concatenate(
        ([np.int64(0)], np.cumsum(lengths_np))
    )
    # transition_to_j[j] = sim[j-1] for 1 <= j <= n-1; transition for
    # j=0 is 0 (no preceding chunk) and for j=n is irrelevant (no
    # state has predecessor n). sim has length n-1.
    transition_to_j: NDArray[np.float64] = np.zeros(n + 1, dtype=np.float64)
    if n >= 2:
        transition_to_j[1:n] = sim.astype(np.float64)

    inf = np.inf
    dp_cost: NDArray[np.float64] = np.full(n + 1, inf, dtype=np.float64)
    dp_prev: NDArray[np.int64] = np.full(n + 1, -1, dtype=np.int64)
    dp_cost[0] = 0.0

    for i in range(1, n + 1):
        threshold_val = int(cum_len_np[i]) - max_size
        j_lo = int(np.searchsorted(cum_len_np, threshold_val, side="left"))
        j_hi = i  # exclusive
        if j_lo >= j_hi:
            continue
        candidates = dp_cost[j_lo:j_hi] + transition_to_j[j_lo:j_hi]
        local_argmin = int(np.argmin(candidates))
        cand_cost = float(candidates[local_argmin])
        if not np.isfinite(cand_cost):
            continue
        dp_cost[i] = cand_cost
        dp_prev[i] = j_lo + local_argmin

    if not np.isfinite(dp_cost[n]):
        raise OptimizationFailedError(
            "covering constraint is infeasible; this should have been caught earlier"
        )

    cuts: list[int] = []
    i = n
    while i > 0:
        j = int(dp_prev[i])
        cuts.append(j)
        i = j
    cuts.reverse()
    cuts.append(n)

    chunks: list[str] = []
    ranges: list[tuple[int, int]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        chunks.append("".join(chunklets[a:b]))
        ranges.append((a, b))
    return chunks, ranges
