"""Stage 4 — late chunking (SPEC-CHUNK-4xx).

Public entry point: ``embed_with_late_chunking``.

Sentence-to-token alignment uses the sentinel-token method described
in SPEC-CHUNK-420. If a sentinel character collision is detected
(SPEC-CHUNK-421), the implementation falls back to a per-sentence
isolated tokenization with the largest-remainder allocation safety net
(SPEC-CHUNK-412).
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import numpy as np

from . import _constants as C
from ._typing import Matrix
from .errors import SentenceExceedsContextError, ValidationError

_DEFAULT_SENTINEL = "⊕"  # CIRCLED PLUS


class TokenLevelEmbedder(Protocol):
    """Embedder protocol used by late chunking (SPEC-CHUNK-04
    §Embedder contract)."""

    n_ctx: int

    def tokenize(self, text: str) -> list[int]: ...  # pragma: no cover
    def detokenize(self, tokens: list[int]) -> str: ...  # pragma: no cover
    def embed(self, text: str) -> Matrix: ...  # pragma: no cover


def embed_with_late_chunking(
    sentences: list[str],
    embedder: TokenLevelEmbedder,
    max_tokens_per_segment: int | None = None,
    preamble_fraction: float = C.DEFAULT_PREAMBLE_FRACTION,
    normalize: bool = True,
    *,
    sentinel: str = _DEFAULT_SENTINEL,
) -> Matrix:
    """Compute per-sentence embeddings with late-chunking context.

    Implements ``docs/specs/04-late-chunking.md``.
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
    if not sentences:
        return np.zeros((0, _infer_dim(embedder)), dtype=np.float64)

    budget = max_tokens_per_segment if max_tokens_per_segment is not None else n_ctx
    preamble_budget = math.floor(preamble_fraction * budget)

    # Pre-compute isolated tokenization counts (used for segment construction).
    iso_token_counts = [len(embedder.tokenize(s)) for s in sentences]
    # SPEC-CHUNK-451 — refuse early if any sentence exceeds n_ctx.
    for idx, count in enumerate(iso_token_counts):
        if count > n_ctx:
            raise SentenceExceedsContextError(
                f"sentence {idx} tokenizes to {count} tokens > embedder.n_ctx {n_ctx}"
            )

    use_sentinel = _can_use_sentinel(sentences, sentinel)

    segments = _build_segments(
        iso_token_counts=iso_token_counts,
        n_ctx=n_ctx,
        budget=budget,
        preamble_budget=preamble_budget,
    )

    n = len(sentences)
    dim = _infer_dim(embedder)
    out = np.zeros((n, dim), dtype=np.float64)
    filled = np.zeros(n, dtype=bool)

    for seg in segments:
        seg_start, content_start, seg_end = seg
        seg_sentences = sentences[seg_start:seg_end]
        joined, per_sentence_counts = _encode_segment(
            embedder=embedder,
            seg_sentences=seg_sentences,
            sentinel=sentinel if use_sentinel else None,
        )
        token_embeddings = embedder.embed(joined)
        if token_embeddings.shape[0] != sum(per_sentence_counts):
            per_sentence_counts = _largest_remainder(
                token_embeddings.shape[0], per_sentence_counts
            )
        cursor = 0
        for local_idx, count in enumerate(per_sentence_counts):
            sentence_global_idx = seg_start + local_idx
            if count <= 0:
                # SPEC-CHUNK-452 — raise on zero-allocation.
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
                pooled = token_embeddings[cursor : cursor + count].mean(axis=0)
                out[sentence_global_idx] = pooled
                filled[sentence_global_idx] = True
            cursor += count

    if not np.all(filled):
        missing = np.where(~filled)[0].tolist()
        raise ValidationError(
            f"sentences {missing} did not receive content embeddings"
        )

    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        out = out / norms
    return out


def _infer_dim(embedder: TokenLevelEmbedder) -> int:
    """Probe the embedder for its hidden size by embedding a short string.

    Embedders are required to be deterministic; a single probe is
    sufficient.
    """
    probe = embedder.embed("a")
    return int(probe.shape[1])


def _can_use_sentinel(sentences: Sequence[str], sentinel: str) -> bool:
    """SPEC-CHUNK-421 — refuse the sentinel method if the character
    appears in any input sentence."""
    if sentinel == "":
        return False
    return not any(sentinel in s for s in sentences)


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
        # Backward preamble walk.
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

        # Forward content walk.
        content_tokens = 0
        segment_end = content_start
        for k in range(content_start, n):
            sentence_tokens = iso_token_counts[k]
            if content_tokens + sentence_tokens > content_budget:
                # Progress guarantee — include at least one content sentence.
                if segment_end == content_start:
                    # Try to fit by shrinking the preamble.
                    while (
                        preamble_tokens + sentence_tokens > n_ctx
                        and segment_start < content_start
                    ):
                        preamble_tokens -= iso_token_counts[segment_start]
                        segment_start += 1
                    if preamble_tokens + sentence_tokens > n_ctx:
                        # SPEC-CHUNK-451 should have caught this; reraise.
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
            # No progress: defensive backstop (the progress guarantee above
            # should make this unreachable in practice).
            raise ValidationError("segment construction made no progress")
        segments.append((segment_start, content_start, segment_end))
        content_start = segment_end
    return segments


def _encode_segment(
    embedder: TokenLevelEmbedder,
    seg_sentences: list[str],
    sentinel: str | None,
) -> tuple[str, list[int]]:
    """Return ``(joined_text, per_sentence_token_counts)`` for the segment.

    When ``sentinel`` is not None, joins sentences with the sentinel
    character and derives per-sentence counts from sentinel-token
    positions; otherwise falls back to isolated tokenization.
    """
    if sentinel is None or len(seg_sentences) < 2:
        # Fallback: concatenate and use per-sentence isolated counts.
        joined = "".join(seg_sentences)
        counts = [len(embedder.tokenize(s)) for s in seg_sentences]
        return joined, counts

    joined = sentinel.join(seg_sentences)
    tokens = embedder.tokenize(joined)
    sentinel_ids = _discover_sentinel_ids(embedder, sentinel)
    sentinel_positions: list[int] = [
        idx for idx, tok in enumerate(tokens) if tok in sentinel_ids
    ]
    expected_separators = len(seg_sentences) - 1
    if len(sentinel_positions) != expected_separators:
        # Fall back when sentinel detection didn't land on the expected count.
        joined = "".join(seg_sentences)
        counts = [len(embedder.tokenize(s)) for s in seg_sentences]
        return joined, counts

    counts: list[int] = []
    last = -1
    for pos in sentinel_positions:
        counts.append(pos - last)
        last = pos
    counts.append(len(tokens) - 1 - last)
    return joined, counts


def _discover_sentinel_ids(
    embedder: TokenLevelEmbedder, sentinel: str
) -> set[int]:
    """SPEC-CHUNK-421 — collect token IDs that decode to a string
    containing the sentinel character.
    """
    probe = f"a{sentinel}b{sentinel}c"
    tokens = embedder.tokenize(probe)
    ids: set[int] = set()
    for tok in tokens:
        try:
            text = embedder.detokenize([tok])
        except Exception:
            continue
        if sentinel in text:
            ids.add(tok)
    return ids


def _largest_remainder(total: int, counts: list[int]) -> list[int]:
    """SPEC-CHUNK-412 — apportion ``total`` tokens to sentences using
    the largest-remainder method when the isolated counts don't add up.

    Floors each sentence's share at 1 (SPEC-CHUNK-452 option a) to
    avoid mean-pooling zero vectors when the apportionment would
    otherwise allocate 0 rows to a sentence.
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
    # Floor at 1 per sentence with original positive isolated count.
    deficit = 0
    for i, (orig, share) in enumerate(zip(counts, floors)):
        if orig > 0 and share <= 0:
            floors[i] = 1
            deficit += 1
    if deficit:
        # Borrow from the most overshot positions until total balances.
        for i in fractions:
            if deficit == 0:
                break
            if floors[i] > 1:
                floors[i] -= 1
                deficit -= 1
    # Round-trip invariant: apportionment must conserve the total. If
    # this assertion ever fires it means the borrow loop above ran
    # out of donors before zeroing the deficit (pathological input
    # like ``total == 1`` with many positive-count sentences); promote
    # to a typed error rather than silently overshooting the buffer.
    if sum(floors) != total:
        raise ValidationError(
            f"per-sentence token apportionment ({sum(floors)}) does not "
            f"sum to embedder output rows ({total})"
        )
    return floors
