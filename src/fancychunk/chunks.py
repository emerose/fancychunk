"""Stage 3 — semantic chunking (SPEC-CHUNK-3xx).

Public entry point: ``split_chunks``. Public data type:
:class:`Chunk` (a frozen dataclass with the chunk text plus
optional character-offset metadata).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from markdown_it import MarkdownIt
from numpy.typing import NDArray

from . import _constants as C
from ._telemetry import get_tracer
from ._typing import Matrix, Vector
from .errors import (
    OversizedChunkletError,
    ValidationError,
    ZeroNormEmbeddingError,
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document with optional metadata.

    Attributes
    ----------
    text:
        The chunk's content. Always present.
    start, end:
        Character offsets into the source the chunk was produced
        from, with half-open ``[start, end)`` semantics so
        ``source[chunk.start:chunk.end] == chunk.text``. ``None``
        when not computed (e.g. a hand-constructed :class:`Chunk`).
        For :func:`split_chunks` the "source" is
        ``"".join(chunklets)``; for :func:`chunk_document` that
        equals the original document (chunklets round-trip per
        SPEC-CHUNK-300).
    heading_path:
        Markdown heading stack **in scope at the chunk's start** —
        a tuple of full heading-line strings (``"# Top"``,
        ``"## **Bold** Sub"``, …) preserving inline formatting and
        the ``#`` markers (the marker count encodes the heading
        level). Trailing whitespace and newlines are stripped from
        each entry. Empty tuple means "computed, no heading in
        scope" (e.g. the first chunk before any heading appears).
        ``None`` means "not computed" — a hand-constructed
        :class:`Chunk` won't have it populated; :func:`split_chunks`
        and :func:`chunk_document` always do.

    New optional metadata fields may be added in future releases.
    Adding a field is non-breaking: existing keyword-construction
    calls still work, and existing field accesses still resolve.
    """

    text: str
    start: int | None = None
    end: int | None = None
    heading_path: tuple[str, ...] | None = None

    def __str__(self) -> str:
        """``str(chunk)`` returns the chunk text."""
        return self.text


class ChunkletEmbedder(Protocol):
    """Caller-supplied object producing one pooled vector per chunklet.

    Used by :func:`split_chunks` to compute the cosine signal driving
    the semantic-split decision. Independent of the
    :class:`SegmentEmbedder` protocol (used by
    :func:`embed_with_late_chunking`); a concrete class may satisfy
    both — the bundled embedders do.

    Methods
    -------
    embed_chunklets(chunklets) -> Matrix
        Async — return a 2-D matrix with one row per chunklet, in
        order. Each row must have nonzero L2 norm (SPEC-CHUNK-342).
    """

    async def embed_chunklets(self, chunklets: list[str]) -> Matrix: ...


async def split_chunks(
    chunklets: list[str],
    embedder: ChunkletEmbedder,
    max_size: int = C.DEFAULT_MAX_SIZE_CHARS,
) -> list[Chunk]:
    """Partition ``chunklets`` into chunks.

    Implements ``docs/specs/03-semantic-chunking.md``.

    ``embedder`` is required. The caller picks the embedder
    explicitly — see ``fancychunk.embedders`` for the bundled
    choices (``qwen3_600m()`` is the recommended default for most
    uses). The embedder drives the *split decision* only; its output
    is not returned. For storage embeddings the caller indexes
    against, use :func:`embed_with_late_chunking` on the chunks
    returned here — or use :func:`chunk_document` to do both in one
    call.

    Pass ``embedder=embedders.noop()`` for a no-model-download
    structural-only split (uniform cosine signal, heading-aware
    boundaries only).

    On the trivial-input short-circuit paths (SPEC-CHUNK-340) the
    embedder argument is required for signature consistency but is
    not invoked.
    """
    if max_size <= 0:
        raise ValidationError("max_size must be positive")

    with get_tracer().start_as_current_span("fancychunk.split_chunks") as span:
        span.set_attribute("fancychunk.chunklets.count", len(chunklets))
        span.set_attribute("fancychunk.max_size", max_size)

        # SPEC-CHUNK-340 — empty input. No embedder call.
        if not chunklets:
            span.set_attribute("fancychunk.chunks.count", 0)
            span.set_attribute("fancychunk.short_circuit", "empty")
            return []

        # SPEC-CHUNK-341 — oversized chunklet (validate before embed
        # so a bad input doesn't trigger a model load).
        lengths = [len(c) for c in chunklets]
        for idx, ln in enumerate(lengths):
            if ln > max_size:
                raise OversizedChunkletError(
                    f"chunklet {idx} has length {ln} > max_size {max_size}"
                )

        # SPEC-CHUNK-340 — single chunklet. No embedder call.
        # heading_path is empty () because nothing precedes chunk 0.
        if len(chunklets) == 1:
            span.set_attribute("fancychunk.chunks.count", 1)
            span.set_attribute("fancychunk.short_circuit", "single_chunklet")
            return [
                Chunk(
                    text=chunklets[0],
                    start=0,
                    end=lengths[0],
                    heading_path=(),
                )
            ]

        # SPEC-CHUNK-340 — total fits. No embedder call.
        total_len = sum(lengths)
        if total_len <= max_size:
            span.set_attribute("fancychunk.chunks.count", 1)
            span.set_attribute("fancychunk.short_circuit", "total_fits")
            return [
                Chunk(
                    text="".join(chunklets),
                    start=0,
                    end=total_len,
                    heading_path=(),
                )
            ]

        # Multi-chunklet, multi-chunk case: embedder drives the
        # partition decision.
        emb = np.asarray(await embedder.embed_chunklets(chunklets))
        if emb.ndim != 2:
            raise ValidationError(
                "embedder.embed_chunklets must return a 2-D matrix"
            )
        if emb.shape[0] != len(chunklets):
            raise ValidationError(
                f"embedder returned {emb.shape[0]} rows but chunklets has "
                f"{len(chunklets)} entries"
            )
        # SPEC-CHUNK-342 — zero-norm embedding.
        if np.any(np.linalg.norm(emb, axis=1) == 0):
            raise ZeroNormEmbeddingError(
                "one or more chunklet embeddings have L2 norm 0"
            )
        span.set_attribute("fancychunk.embedding.dim", int(emb.shape[1]))

        tracer = get_tracer()
        with tracer.start_as_current_span("fancychunk.chunks.partition_similarities"):
            sim = _partition_similarities(emb, chunklets, lengths)
        with tracer.start_as_current_span("fancychunk.chunks.dp"):
            chunks = _solve_partition(chunklets, lengths, sim, max_size)
        span.set_attribute("fancychunk.chunks.count", len(chunks))
        return chunks


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

    # Step 4 — heading-aware modification (in place).
    _apply_heading_modification_inplace(sim, chunklets, floor)
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


def _apply_heading_modification_inplace(
    sim: Vector, chunklets: list[str], floor: float
) -> None:
    n = len(chunklets)
    # Compute the heading flag once per chunklet; markdown-it parsing
    # is reentrant but not free, and the loop below reads each value
    # once.
    is_heading_flags = [_is_heading(c) for c in chunklets]
    previous_is_heading = False
    for i in range(n):
        if is_heading_flags[i]:
            if i >= 1 and not previous_is_heading:
                sim[i - 1] = max(sim[i - 1] / C.HEADING_SPLIT_BEFORE_DIVISOR, floor)
            if i <= n - 2:
                sim[i] = C.HEADING_SPLIT_AFTER_FORBID
            previous_is_heading = True
        else:
            previous_is_heading = False


def _solve_partition(
    chunklets: list[str], lengths: list[int], sim: Vector, max_size: int
) -> list[Chunk]:
    """SPEC-CHUNK-310/-311 — minimize total partition similarity under the
    covering constraint that every chunk fits in ``max_size``.

    Returns the partitioned chunks (each chunk is the concatenation
    of one or more contiguous chunklets).

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
        dp_cost[i] = float(candidates[local_argmin])
        dp_prev[i] = j_lo + local_argmin

    # Unreachable in practice: every chunklet ≤ max_size (validated
    # above), so the per-chunklet partition is always feasible.
    assert np.isfinite(dp_cost[n]), "internal: chunks DP left dp_cost[n] non-finite"

    cuts: list[int] = []
    i = n
    while i > 0:
        j = int(dp_prev[i])
        cuts.append(j)
        i = j
    cuts.reverse()
    cuts.append(n)

    # Each chunk spans chunklets[a:b]; its character range in
    # ``"".join(chunklets)`` is ``[cum_len_np[a], cum_len_np[b])``.
    # Build chunks first, then populate heading_path in a second
    # pass (the heading scan needs the chunk text to be assembled).
    bare_chunks = [
        Chunk(
            text="".join(chunklets[a:b]),
            start=int(cum_len_np[a]),
            end=int(cum_len_np[b]),
        )
        for a, b in zip(cuts[:-1], cuts[1:])
    ]
    # Local import to avoid a cycle (headings → chunks).
    from .headings import heading_paths
    from dataclasses import replace

    paths = heading_paths(bare_chunks)
    return [replace(c, heading_path=p) for c, p in zip(bare_chunks, paths)]
