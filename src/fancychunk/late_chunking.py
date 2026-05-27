"""Stage 4 — late chunking (SPEC-CHUNK-4xx).

Public entry point: :func:`embed_with_late_chunking`.

The library owns the *late-chunking algorithm* — segment planning with
backward preamble, optional heading-stack prepend, per-chunk
mean-pool, preamble discard, optional normalization — and nothing
else. Tokenization, special-token policy, and the choice of method
for mapping joined-input tokens back to their per-unit boundaries are
the caller's responsibility, encapsulated behind the
:class:`SegmentEmbedder` protocol.

See ``examples/embedders/`` for reference adapters over MLX, HuggingFace
transformers, and a remote HTTP service.
"""

from __future__ import annotations

import asyncio
import math
from typing import Protocol

import numpy as np

from . import _constants as C
from ._telemetry import get_tracer
from ._typing import Matrix
from .chunks import Chunk
from .errors import ChunkExceedsContextError, ValidationError


class SegmentEmbedder(Protocol):
    """Caller-supplied object that embeds multi-text segments and
    returns per-token outputs aligned to per-text boundaries.

    The "texts" passed to :meth:`count_tokens` and :meth:`embed_segment`
    are whatever contiguous text units the late-chunking algorithm is
    sliding windows over — chunks for :func:`embed_with_late_chunking`
    (one heading-prepend text plus one entry per chunk in the
    segment). Implementations choose their own tokenization library,
    special-token policy (SPEC-CHUNK-420), and method for mapping
    joined-input tokens back to source units (sentinel-token,
    offset-based, or anything else that satisfies the contract below).

    Attributes
    ----------
    n_ctx:
        Maximum number of tokens the embedder accepts in one segment.

    Methods
    -------
    count_tokens(texts) -> list[int]
        Async — per-text token count for budget planning. May be
        approximate — subword merges across boundaries can shift
        counts by ±1, and the largest-remainder safety net in
        :func:`embed_with_late_chunking` absorbs that drift. Used only
        for segment construction.
    embed_segment(texts) -> (per_token_embeddings, per_text_counts)
        Async — embed ``texts`` joined into a single contextualized
        segment. Return the per-token embedding matrix and the per-text
        token allocation. The allocation must conserve the matrix's
        row count: ``sum(per_text_counts) == per_token_embeddings.shape[0]``.
        Any leading or trailing special tokens (``[CLS]``, ``[SEP]``,
        BOS, EOS) the embedder injects are the implementation's concern
        — typically absorbed into the first and last texts' allocations
        per SPEC-CHUNK-420 option (b).
    """

    n_ctx: int

    async def count_tokens(self, texts: list[str]) -> list[int]: ...

    async def embed_segment(
        self, texts: list[str]
    ) -> tuple[Matrix, list[int]]: ...


async def embed_with_late_chunking(
    chunks: list[Chunk],
    embedder: SegmentEmbedder,
    max_tokens_per_segment: int | None = None,
    preamble_fraction: float = C.DEFAULT_PREAMBLE_FRACTION,
    normalize: bool = True,
    include_headings: bool = True,
) -> Matrix:
    """Compute one context-aware embedding per chunk.

    Implements ``docs/specs/04-late-chunking.md``.

    Internally treats chunks as the sliding-window unit: each segment
    holds as many adjacent chunks as fit in ``max_tokens_per_segment``,
    with ``preamble_fraction`` of the budget reserved for warm-up
    context (heading-stack prepend + backward-walk preamble). The
    embedder runs over the joined segment text; tokens are mean-pooled
    within each chunk's allocation. Preamble token embeddings (heading
    + backward-walk) are discarded.

    Parameters
    ----------
    chunks:
        Ordered list of chunks (typically the first element of
        :func:`split_chunks`'s output, *before* any
        :func:`enrich_with_headings` post-processing — that helper is
        for stored-text breadcrumbs, not embedding-time context).
    embedder:
        Caller-supplied :class:`SegmentEmbedder`. See
        ``examples/embedders/`` for reference adapters.
    max_tokens_per_segment:
        Optional cap on the per-segment token budget. Defaults to
        ``embedder.n_ctx`` (the whole context window). Must not exceed
        ``embedder.n_ctx``.
    preamble_fraction:
        Fraction of the segment budget reserved for the preamble
        (heading prepend + backward-walk context). Default ``0.382``
        (inverse golden ratio); operating band roughly ``[0.25, 0.45]``.
        Pass ``0.0`` to disable late chunking and recover naive
        per-segment embedding (useful for ablation).
    normalize:
        Whether to L2-normalize each output row. Default ``True``.
    include_headings:
        When ``True`` (the default), each segment is prefixed with the
        Markdown heading stack in scope at the segment's first content
        chunk, giving the embedder document-outline context (SPEC-CHUNK-470).
        The heading-stack tokens count against the preamble budget;
        their embeddings are discarded after pooling, same as preamble
        text. Pass ``False`` for non-markdown inputs or for ablation.

    Returns
    -------
    NDArray
        Matrix of shape ``(len(chunks), D)`` where ``D`` is the
        embedder's hidden size.
    """
    if preamble_fraction < 0 or preamble_fraction >= 1:
        raise ValidationError("preamble_fraction must be in [0, 1)")
    if max_tokens_per_segment is not None and max_tokens_per_segment <= 0:
        raise ValidationError("max_tokens_per_segment must be positive")
    n_ctx = int(embedder.n_ctx)
    if max_tokens_per_segment is not None and max_tokens_per_segment > n_ctx:
        raise ValidationError(
            f"max_tokens_per_segment ({max_tokens_per_segment}) exceeds "
            f"embedder.n_ctx ({n_ctx})"
        )

    with get_tracer().start_as_current_span(
        "fancychunk.embed_with_late_chunking"
    ) as span:
        span.set_attribute("fancychunk.chunks.count", len(chunks))
        span.set_attribute("fancychunk.preamble_fraction", preamble_fraction)
        span.set_attribute("fancychunk.normalize", normalize)
        span.set_attribute("fancychunk.include_headings", include_headings)
        span.set_attribute("fancychunk.embedder.n_ctx", n_ctx)
        span.set_attribute("fancychunk.embedder", type(embedder).__name__)

        if not chunks:
            span.set_attribute("fancychunk.segments.count", 0)
            dim = await _infer_dim(embedder)
            return np.zeros((0, dim), dtype=np.float64)

        # Extract chunk texts once — the embedder protocol works in
        # strings, and downstream slicing / tokenization needs them
        # repeatedly. Chunk metadata (start/end) is not used here;
        # callers preserve it on the input list for their own use.
        chunk_texts = [c.text for c in chunks]

        budget = (
            max_tokens_per_segment if max_tokens_per_segment is not None else n_ctx
        )
        preamble_budget = math.floor(preamble_fraction * budget)
        span.set_attribute("fancychunk.budget", budget)
        span.set_attribute("fancychunk.preamble_budget", preamble_budget)

        # SPEC-CHUNK-470 — compute per-chunk heading prepends once.
        # Empty strings indicate "no heading in scope" (or
        # include_headings=False); those segments skip the prepend.
        if include_headings:
            from .headings import heading_paths

            heading_prepends = heading_paths(chunks)
        else:
            heading_prepends = [""] * len(chunks)

        # SPEC-CHUNK-411 — caller-supplied per-chunk counts for budget
        # planning. Isolated-tokenization counts; SPEC-CHUNK-412's
        # largest-remainder safety net absorbs any drift between these
        # and the joined-input counts the embedder ultimately sees.
        iso_token_counts = await embedder.count_tokens(chunk_texts)
        if len(iso_token_counts) != len(chunks):
            raise ValidationError(
                f"embedder.count_tokens returned {len(iso_token_counts)} "
                f"counts; expected {len(chunks)}"
            )

        # SPEC-CHUNK-451 — refuse early if any chunk exceeds n_ctx.
        for idx, count in enumerate(iso_token_counts):
            if count > n_ctx:
                raise ChunkExceedsContextError(
                    f"chunk {idx} tokenizes to {count} tokens > "
                    f"embedder.n_ctx {n_ctx}"
                )

        # Pre-compute heading-token counts so segment planning can
        # reserve them against the preamble budget without re-tokenizing
        # per segment. ``count_tokens`` accepts a list, so one batched
        # call is enough; entries for empty heading strings get 0.
        nonempty_paths = [p for p in heading_prepends if p]
        nonempty_counts = (
            await embedder.count_tokens(nonempty_paths) if nonempty_paths else []
        )
        if len(nonempty_counts) != len(nonempty_paths):
            raise ValidationError(
                f"embedder.count_tokens returned {len(nonempty_counts)} "
                f"counts for {len(nonempty_paths)} heading prepends"
            )
        heading_token_counts: list[int] = []
        nec_iter = iter(nonempty_counts)
        for p in heading_prepends:
            heading_token_counts.append(next(nec_iter) if p else 0)

        segments = _build_segments(
            iso_token_counts=iso_token_counts,
            heading_token_counts=heading_token_counts,
            n_ctx=n_ctx,
            budget=budget,
            preamble_budget=preamble_budget,
        )
        span.set_attribute("fancychunk.segments.count", len(segments))

        n = len(chunks)
        out: Matrix | None = None
        filled = np.zeros(n, dtype=bool)

        # Build the per-segment input lists, then launch every
        # segment's embed_segment call concurrently. Segments have no
        # data dependencies on each other (heading prepends are
        # pre-computed; pooling is local), so this is safe — and for
        # remote/parallel embedders it overlaps the network/GPU time.
        # For the bundled embedders the internal lock serializes to
        # the device anyway; the gather is harmless there.
        segment_inputs: list[list[str]] = []
        for seg_start, content_start, _seg_end in segments:
            heading_text = heading_prepends[content_start]
            seg_chunk_texts = chunk_texts[seg_start:_seg_end]
            if heading_text:
                segment_inputs.append([heading_text] + seg_chunk_texts)
            else:
                segment_inputs.append(seg_chunk_texts)

        segment_results = await asyncio.gather(
            *(embedder.embed_segment(texts) for texts in segment_inputs)
        )

        for seg, segment_texts, (token_embeddings, per_text_counts) in zip(
            segments, segment_inputs, segment_results
        ):
            seg_start, content_start, seg_end = seg
            heading_text = heading_prepends[content_start]
            heading_tokens_est = heading_token_counts[content_start]

            if len(per_text_counts) != len(segment_texts):
                raise ValidationError(
                    f"embed_segment returned {len(per_text_counts)} counts "
                    f"for {len(segment_texts)} input texts"
                )
            total_tokens = int(token_embeddings.shape[0])
            if sum(per_text_counts) != total_tokens:
                # SPEC-CHUNK-412 — apportion via largest remainder when
                # counts don't conserve the matrix row count.
                per_text_counts = _largest_remainder(total_tokens, per_text_counts)
            if out is None:
                dim = int(token_embeddings.shape[1])
                out = np.zeros((n, dim), dtype=np.float64)

            cursor = 0
            # Heading prepend (if present) consumes the first slice of
            # rows; we discard them — they're preamble context.
            if heading_text:
                heading_count = per_text_counts[0]
                cursor += heading_count
                chunk_counts = per_text_counts[1:]
                # Useful telemetry: which chunk index this prepend
                # served. (Span attribute would be per-segment; skip.)
                _ = heading_tokens_est  # silence unused-var lint
            else:
                chunk_counts = per_text_counts

            for local_idx, count in enumerate(chunk_counts):
                chunk_global_idx = seg_start + local_idx
                if count <= 0:
                    if not filled[chunk_global_idx]:
                        raise ValidationError(
                            f"chunk {chunk_global_idx} received zero "
                            "tokens; cannot mean-pool"
                        )
                    cursor += count
                    continue
                if (
                    content_start <= chunk_global_idx < seg_end
                    and not filled[chunk_global_idx]
                ):
                    pooled = token_embeddings[
                        cursor : cursor + count
                    ].mean(axis=0)
                    out[chunk_global_idx] = pooled
                    filled[chunk_global_idx] = True
                cursor += count

        if out is None:
            # ``segments`` is non-empty when ``chunks`` is non-empty
            # (the greedy planner guarantees progress); reaching here
            # would be an internal bug.
            raise ValidationError("no segments produced for non-empty input")

        if not np.all(filled):
            missing = np.where(~filled)[0].tolist()
            raise ValidationError(
                f"chunks {missing} did not receive content embeddings"
            )

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            out = out / norms
        span.set_attribute("fancychunk.embedding.dim", int(out.shape[1]))
        return out


async def _infer_dim(embedder: SegmentEmbedder) -> int:
    """Probe the embedder for its hidden size by embedding a one-text segment."""
    mat, _ = await embedder.embed_segment(["a"])
    return int(mat.shape[1])


def _build_segments(
    iso_token_counts: list[int],
    heading_token_counts: list[int],
    n_ctx: int,
    budget: int,
    preamble_budget: int,
) -> list[tuple[int, int, int]]:
    """SPEC-CHUNK-411 + SPEC-CHUNK-470 — greedy segment construction
    with backward preamble and heading prepend.

    Returns a list of ``(segment_start, content_start, segment_end)``
    triples covering every chunk in exactly one segment's content
    range. The heading prepend for ``content_start`` consumes
    ``heading_token_counts[content_start]`` tokens out of the preamble
    budget *before* the backward walk runs.
    """
    n = len(iso_token_counts)
    segments: list[tuple[int, int, int]] = []
    content_start = 0
    while content_start < n:
        heading_tokens = heading_token_counts[content_start]
        # Heading prepend comes out of the preamble budget first; what
        # remains is available for the backward-walk preamble text.
        backward_budget = max(0, preamble_budget - heading_tokens)
        if content_start == 0:
            segment_start = 0
            backward_tokens = 0
        else:
            backward_tokens = 0
            segment_start = content_start
            for k in range(content_start - 1, -1, -1):
                if backward_tokens + iso_token_counts[k] > backward_budget:
                    break
                backward_tokens += iso_token_counts[k]
                segment_start = k

        preamble_tokens = heading_tokens + backward_tokens
        unused_preamble = preamble_budget - preamble_tokens
        content_budget = (budget - preamble_budget) + unused_preamble

        content_tokens = 0
        segment_end = content_start
        for k in range(content_start, n):
            chunk_tokens = iso_token_counts[k]
            if content_tokens + chunk_tokens > content_budget:
                if segment_end == content_start:
                    # Progress guarantee: include at least one content
                    # chunk even if it overshoots the content budget,
                    # shrinking the backward-walk preamble if necessary
                    # to fit n_ctx. The heading prepend cannot be
                    # shrunk — it's all-or-nothing semantically.
                    while (
                        preamble_tokens + chunk_tokens > n_ctx
                        and segment_start < content_start
                    ):
                        preamble_tokens -= iso_token_counts[segment_start]
                        segment_start += 1
                    if preamble_tokens + chunk_tokens > n_ctx:
                        # Last resort: drop the heading prepend.
                        if heading_tokens > 0:
                            preamble_tokens -= heading_tokens
                            heading_tokens = 0
                    if preamble_tokens + chunk_tokens > n_ctx:
                        raise ChunkExceedsContextError(
                            f"chunk {k} ({chunk_tokens} tokens) plus "
                            f"unshrinkable preamble exceeds n_ctx {n_ctx}"
                        )
                    content_tokens += chunk_tokens
                    segment_end = k + 1
                break
            content_tokens += chunk_tokens
            segment_end = k + 1
        if segment_end == content_start:
            raise ValidationError("segment construction made no progress")
        segments.append((segment_start, content_start, segment_end))
        content_start = segment_end
    return segments


def _largest_remainder(total: int, counts: list[int]) -> list[int]:
    """SPEC-CHUNK-412 — apportion ``total`` tokens to per-text counts
    using the largest-remainder method when the embedder-reported
    counts don't add up to the actual matrix row count.

    Floors each text's share at 1 (SPEC-CHUNK-452 option a) to avoid
    mean-pooling zero vectors when apportionment would otherwise
    allocate 0 rows to a text with positive isolated count.
    """
    sum_counts = sum(counts) or 1
    floats = [total * c / sum_counts for c in counts]
    floors = [math.floor(f) for f in floats]
    remainder = total - sum(floors)
    fractions = sorted(
        range(len(counts)), key=lambda i: floats[i] - floors[i], reverse=True
    )
    for i in fractions[:remainder]:
        floors[i] += 1
    deficit = 0
    for i, (orig, share) in enumerate(zip(counts, floors)):
        if orig > 0 and share <= 0:
            floors[i] = 1
            deficit += 1
    if deficit:
        for i in fractions:
            if deficit == 0:
                break
            if floors[i] > 1:
                floors[i] -= 1
                deficit -= 1
    if sum(floors) != total:
        raise ValidationError(
            f"per-text token apportionment ({sum(floors)}) does not "
            f"sum to embedder output rows ({total})"
        )
    return floors
