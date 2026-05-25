"""Stage 4 — late chunking (SPEC-CHUNK-4xx).

Public entry point: :func:`embed_with_late_chunking`.

The library owns the *late-chunking algorithm* — segment planning with
backward preamble, mean-pool per sentence, preamble discard, optional
normalization — and nothing else. Tokenization, special-token policy,
and the choice of sentence-to-token alignment method (sentinel-token,
offset-based, custom) are the caller's responsibility, encapsulated
behind the :class:`SegmentEmbedder` protocol.

See ``examples/embedders/`` for reference adapters over MLX, HuggingFace
transformers, and a remote HTTP service.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from . import _constants as C
from ._telemetry import get_tracer
from ._typing import Matrix
from .errors import SentenceExceedsContextError, ValidationError


class SegmentEmbedder(Protocol):
    """Caller-supplied object that embeds multi-sentence segments and
    returns per-token outputs aligned to per-sentence boundaries.

    Implementations choose their own tokenization library, special-token
    policy (SPEC-CHUNK-420), and method for mapping joined-input tokens
    back to their source sentences (sentinel-token, offset-based, or
    anything else that satisfies the contract below).

    Attributes
    ----------
    n_ctx:
        Maximum number of tokens the embedder accepts in one segment.

    Methods
    -------
    count_tokens(sentences) -> list[int]
        Per-sentence token count for budget planning. May be approximate
        — subword merges across sentence boundaries can shift counts by
        ±1, and the largest-remainder safety net in
        :func:`embed_with_late_chunking` absorbs that drift. Used only
        for segment construction.
    embed_segment(sentences) -> (per_token_embeddings, per_sentence_counts)
        Embed ``sentences`` joined into a single contextualized
        segment. Return the per-token embedding matrix and the
        per-sentence token allocation. The allocation must conserve
        the matrix's row count: ``sum(per_sentence_counts) ==
        per_token_embeddings.shape[0]``. Any leading or trailing
        special tokens (``[CLS]``, ``[SEP]``, BOS, EOS) the embedder
        injects are the implementation's concern — typically absorbed
        into the first and last sentences' allocations per
        SPEC-CHUNK-420 option (b).
    """

    n_ctx: int

    def count_tokens(self, sentences: list[str]) -> list[int]: ...

    def embed_segment(
        self, sentences: list[str]
    ) -> tuple[Matrix, list[int]]: ...


def embed_with_late_chunking(
    sentences: list[str],
    embedder: SegmentEmbedder,
    max_tokens_per_segment: int | None = None,
    preamble_fraction: float = C.DEFAULT_PREAMBLE_FRACTION,
    normalize: bool = True,
) -> Matrix:
    """Compute per-sentence embeddings with late-chunking context.

    Implements ``docs/specs/04-late-chunking.md``.

    Parameters
    ----------
    sentences:
        Ordered list of sentences (typically the output of stage 1).
    embedder:
        Caller-supplied :class:`SegmentEmbedder`. See
        ``examples/embedders/`` for reference adapters.
    max_tokens_per_segment:
        Optional cap on the per-segment token budget. Defaults to
        ``embedder.n_ctx`` (the whole context window). Must not
        exceed ``embedder.n_ctx``.
    preamble_fraction:
        Fraction of the segment budget reserved for preamble context.
        Default ``0.382`` (inverse golden ratio); operating band roughly
        ``[0.25, 0.45]``. Pass ``0.0`` to disable late chunking and
        recover naive per-segment embedding (useful for ablation).
    normalize:
        Whether to L2-normalize each output row. Default ``True``.

    Returns
    -------
    NDArray
        Matrix of shape ``(len(sentences), D)`` where ``D`` is the
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
        span.set_attribute("fancychunk.sentences.count", len(sentences))
        span.set_attribute("fancychunk.preamble_fraction", preamble_fraction)
        span.set_attribute("fancychunk.normalize", normalize)
        span.set_attribute("fancychunk.embedder.n_ctx", n_ctx)
        span.set_attribute("fancychunk.embedder", type(embedder).__name__)

        if not sentences:
            span.set_attribute("fancychunk.segments.count", 0)
            dim = _infer_dim(embedder)
            return np.zeros((0, dim), dtype=np.float64)

        budget = (
            max_tokens_per_segment if max_tokens_per_segment is not None else n_ctx
        )
        preamble_budget = math.floor(preamble_fraction * budget)
        span.set_attribute("fancychunk.budget", budget)
        span.set_attribute("fancychunk.preamble_budget", preamble_budget)

        # SPEC-CHUNK-411 — caller-supplied per-sentence counts for
        # budget planning. These are isolated-tokenization counts and
        # may drift from what embed_segment actually sees in the joined
        # input; the drift is absorbed by the largest-remainder safety
        # net (SPEC-CHUNK-412) downstream.
        iso_token_counts = embedder.count_tokens(sentences)
        if len(iso_token_counts) != len(sentences):
            raise ValidationError(
                f"embedder.count_tokens returned {len(iso_token_counts)} "
                f"counts; expected {len(sentences)}"
            )

        # SPEC-CHUNK-451 — refuse early if any sentence exceeds n_ctx.
        for idx, count in enumerate(iso_token_counts):
            if count > n_ctx:
                raise SentenceExceedsContextError(
                    f"sentence {idx} tokenizes to {count} tokens > "
                    f"embedder.n_ctx {n_ctx}"
                )

        segments = _build_segments(
            iso_token_counts=iso_token_counts,
            n_ctx=n_ctx,
            budget=budget,
            preamble_budget=preamble_budget,
        )
        span.set_attribute("fancychunk.segments.count", len(segments))

        n = len(sentences)
        # Infer dim from the first segment's first embed call (avoids
        # an extra round trip when ``embedder.embed_segment`` is
        # expensive).
        out: Matrix | None = None
        filled = np.zeros(n, dtype=bool)

        for seg in segments:
            seg_start, content_start, seg_end = seg
            seg_sentences = sentences[seg_start:seg_end]
            token_embeddings, per_sentence_counts = embedder.embed_segment(
                seg_sentences
            )
            if len(per_sentence_counts) != len(seg_sentences):
                raise ValidationError(
                    f"embed_segment returned {len(per_sentence_counts)} counts "
                    f"for {len(seg_sentences)} sentences"
                )
            total_tokens = int(token_embeddings.shape[0])
            if sum(per_sentence_counts) != total_tokens:
                # SPEC-CHUNK-412 — apportion via largest remainder when
                # counts don't conserve the matrix row count.
                per_sentence_counts = _largest_remainder(
                    total_tokens, per_sentence_counts
                )
            if out is None:
                dim = int(token_embeddings.shape[1])
                out = np.zeros((n, dim), dtype=np.float64)

            cursor = 0
            for local_idx, count in enumerate(per_sentence_counts):
                sentence_global_idx = seg_start + local_idx
                if count <= 0:
                    if not filled[sentence_global_idx]:
                        raise ValidationError(
                            f"sentence {sentence_global_idx} received zero "
                            "tokens; cannot mean-pool"
                        )
                    cursor += count
                    continue
                if (
                    content_start <= sentence_global_idx < seg_end
                    and not filled[sentence_global_idx]
                ):
                    pooled = token_embeddings[
                        cursor : cursor + count
                    ].mean(axis=0)
                    out[sentence_global_idx] = pooled
                    filled[sentence_global_idx] = True
                cursor += count

        if out is None:
            # ``segments`` is non-empty when ``sentences`` is non-empty
            # (the greedy planner guarantees progress); reaching here
            # would be an internal bug.
            raise ValidationError("no segments produced for non-empty input")

        if not np.all(filled):
            missing = np.where(~filled)[0].tolist()
            raise ValidationError(
                f"sentences {missing} did not receive content embeddings"
            )

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            out = out / norms
        span.set_attribute("fancychunk.embedding.dim", int(out.shape[1]))
        return out


def _infer_dim(embedder: SegmentEmbedder) -> int:
    """Probe the embedder for its hidden size by embedding a one-sentence segment."""
    mat, _ = embedder.embed_segment(["a"])
    return int(mat.shape[1])


def _build_segments(
    iso_token_counts: list[int],
    n_ctx: int,
    budget: int,
    preamble_budget: int,
) -> list[tuple[int, int, int]]:
    """SPEC-CHUNK-411 — greedy segment construction with backward preamble.

    Returns a list of ``(segment_start, content_start, segment_end)``
    triples covering every sentence in exactly one segment's content
    range.
    """
    n = len(iso_token_counts)
    segments: list[tuple[int, int, int]] = []
    content_start = 0
    while content_start < n:
        if content_start == 0:
            segment_start = 0
            preamble_tokens = 0
        else:
            preamble_tokens = 0
            segment_start = content_start
            for k in range(content_start - 1, -1, -1):
                if preamble_tokens + iso_token_counts[k] > preamble_budget:
                    break
                preamble_tokens += iso_token_counts[k]
                segment_start = k

        unused_preamble = preamble_budget - preamble_tokens
        content_budget = (budget - preamble_budget) + unused_preamble

        content_tokens = 0
        segment_end = content_start
        for k in range(content_start, n):
            sentence_tokens = iso_token_counts[k]
            if content_tokens + sentence_tokens > content_budget:
                if segment_end == content_start:
                    # Progress guarantee: include at least one content
                    # sentence even if it overshoots the content budget,
                    # shrinking the preamble if necessary to fit n_ctx.
                    while (
                        preamble_tokens + sentence_tokens > n_ctx
                        and segment_start < content_start
                    ):
                        preamble_tokens -= iso_token_counts[segment_start]
                        segment_start += 1
                    if preamble_tokens + sentence_tokens > n_ctx:
                        raise SentenceExceedsContextError(
                            f"sentence {k} ({sentence_tokens} tokens) plus "
                            f"unshrinkable preamble exceeds n_ctx {n_ctx}"
                        )
                    content_tokens += sentence_tokens
                    segment_end = k + 1
                break
            content_tokens += sentence_tokens
            segment_end = k + 1
        if segment_end == content_start:
            raise ValidationError("segment construction made no progress")
        segments.append((segment_start, content_start, segment_end))
        content_start = segment_end
    return segments


def _largest_remainder(total: int, counts: list[int]) -> list[int]:
    """SPEC-CHUNK-412 — apportion ``total`` tokens to sentences using
    the largest-remainder method when the embedder-reported counts
    don't add up to the actual matrix row count.

    Floors each sentence's share at 1 (SPEC-CHUNK-452 option a) to
    avoid mean-pooling zero vectors when apportionment would otherwise
    allocate 0 rows to a sentence with positive isolated count.
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
            f"per-sentence token apportionment ({sum(floors)}) does not "
            f"sum to embedder output rows ({total})"
        )
    return floors
