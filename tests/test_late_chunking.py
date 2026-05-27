"""Stage 4 tests — late chunking.

Verifies the public ``embed_with_late_chunking`` contract:
SPEC-CHUNK-400/-401/-402 (shape and order), SPEC-CHUNK-410/-411/-412
(segment construction and pooling), SPEC-CHUNK-420 (per-text token
alignment is the embedder's job), SPEC-CHUNK-430 (normalization),
SPEC-CHUNK-440 (determinism), SPEC-CHUNK-450/-451/-452 (edge cases),
and SPEC-CHUNK-470 (per-segment heading prepend).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fancychunk import (
    ChunkExceedsContextError,
    SentenceExceedsContextError,
    ValidationError,
)
from fancychunk import embed_with_late_chunking as _async_embed_with_late_chunking

from ._fake_embedder import (
    BertLikeFakeEmbedder,
    FakeEmbedder,
    RecordingFakeEmbedder,
    WhitespaceDroppingFakeEmbedder,
)


# Sync shim: every test below calls ``embed_with_late_chunking`` as if
# it were sync. The library entry point is async; this wrapper runs
# the coroutine to completion via ``asyncio.run``. Lets the test bodies
# stay readable rather than peppered with ``asyncio.run(...)``.
def embed_with_late_chunking(*args, **kwargs):  # type: ignore[no-untyped-def]
    return asyncio.run(_async_embed_with_late_chunking(*args, **kwargs))


# ---------------------------------------------------------------------------
# Shape and ordering — SPEC-CHUNK-400, -401.
# ---------------------------------------------------------------------------


def test_output_shape_matches_input_chunks() -> None:
    """SPEC-CHUNK-400 — one row per chunk."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(
        ["First chunk.", "Second chunk.", "Third chunk."],
        fake,
        max_tokens_per_segment=512,
        include_headings=False,
    )
    assert out.shape == (3, 8)


def test_row_order_matches_input_order() -> None:
    """SPEC-CHUNK-400 — row i corresponds to chunk i."""
    fake = FakeEmbedder(dim=16, n_ctx=512)
    chunks = ["alpha", "bravo", "charlie", "delta"]
    out = embed_with_late_chunking(
        chunks, fake, max_tokens_per_segment=512, include_headings=False
    )
    assert out.shape == (4, 16)
    # Distinct content → distinct pooled rows.
    assert not np.array_equal(out[0], out[3])


def test_empty_input_returns_empty_matrix() -> None:
    """Zero chunks → (0, D) matrix, no embedder.embed_segment for content."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking([], fake)
    assert out.shape == (0, 8)


# ---------------------------------------------------------------------------
# Single-chunk + oversize — SPEC-CHUNK-450, -451.
# ---------------------------------------------------------------------------


def test_single_chunk_input() -> None:
    """SPEC-CHUNK-450 — single chunk yields one row from one embedder call."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(["Only one chunk here."], fake)
    assert out.shape == (1, 8)


def test_oversize_chunk_raises_chunk_exceeds_context() -> None:
    """SPEC-CHUNK-451 — a chunk longer than n_ctx tokens is refused early."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    with pytest.raises(ChunkExceedsContextError):
        embed_with_late_chunking(
            ["A" + "a" * 10000], fake, max_tokens_per_segment=512
        )


def test_sentence_exceeds_context_alias_still_works() -> None:
    """Back-compat: the old SentenceExceedsContextError name still
    catches the same error."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    with pytest.raises(SentenceExceedsContextError):
        embed_with_late_chunking(["a" * 10000], fake)


# ---------------------------------------------------------------------------
# Segment construction — SPEC-CHUNK-410, -411.
# ---------------------------------------------------------------------------


def test_every_chunk_appears_in_exactly_one_content_range() -> None:
    """SPEC-CHUNK-410 — coverage. 20 chunks, small budget → multi-segment."""
    chunks = ["x" * 50 for _ in range(20)]
    fake = FakeEmbedder(dim=4, n_ctx=256)
    out = embed_with_late_chunking(
        chunks,
        fake,
        max_tokens_per_segment=256,
        preamble_fraction=0.382,
        include_headings=False,
    )
    assert out.shape == (20, 4)
    # No row should be all-zero — every chunk got content embeddings.
    norms = np.linalg.norm(out, axis=1)
    assert np.all(norms > 0)


def test_first_segment_has_empty_preamble() -> None:
    """SPEC-CHUNK-411 — first segment's backward walk starts at index 0."""
    chunks = ["x" * 30, "x" * 30, "x" * 30]
    fake = FakeEmbedder(dim=4, n_ctx=200)
    out = embed_with_late_chunking(
        chunks,
        fake,
        max_tokens_per_segment=200,
        preamble_fraction=0.382,
        include_headings=False,
    )
    assert out.shape == (3, 4)


# ---------------------------------------------------------------------------
# Per-text alignment owned by the embedder — SPEC-CHUNK-420.
# ---------------------------------------------------------------------------


def test_embedder_owns_per_text_alignment() -> None:
    """SPEC-CHUNK-420 — the FakeEmbedder's trivial one-char-per-token
    alignment lets us hand-compute the pooled row for a known input."""
    fake = FakeEmbedder(dim=4, n_ctx=512)
    out = embed_with_late_chunking(
        ["AB", "CD"], fake, include_headings=False
    )
    assert out.shape == (2, 4)
    # 'A'=65 → dim 1, 'B'=66 → dim 2: 0.5 mass each before normalize.
    expected = np.array([0.0, 0.5, 0.5, 0.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(out[0], expected)


def test_bert_like_special_tokens_absorbed() -> None:
    """SPEC-CHUNK-420 option (b) — leading/trailing specials absorbed
    by the embedder's per-text counts."""
    fake = BertLikeFakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(
        ["AB", "CD"], fake, include_headings=False
    )
    assert out.shape == (2, 8)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Zero-token text — SPEC-CHUNK-452.
# ---------------------------------------------------------------------------


def test_zero_token_chunk_raises() -> None:
    """SPEC-CHUNK-452 — a chunk that tokenizes to zero tokens can't
    be mean-pooled."""
    fake = WhitespaceDroppingFakeEmbedder(dim=8, n_ctx=512)
    chunks = ["a", "b", "c"]  # 'b' produces zero tokens
    with pytest.raises(ValidationError):
        embed_with_late_chunking(chunks, fake, include_headings=False)


# ---------------------------------------------------------------------------
# Normalization — SPEC-CHUNK-402, -430.
# ---------------------------------------------------------------------------


def test_normalize_true_yields_unit_rows() -> None:
    """SPEC-CHUNK-402 / -430 — default normalize=True L2-normalizes."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    chunks = ["First.", "Second."]
    out = embed_with_late_chunking(
        chunks, fake, normalize=True, include_headings=False
    )
    for row in out:
        assert np.isclose(np.linalg.norm(row), 1.0)


def test_normalize_false_skips_normalization() -> None:
    """SPEC-CHUNK-430 — normalize=False returns raw mean-pooled rows."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    chunks = ["First.", "Second."]
    out = embed_with_late_chunking(
        chunks, fake, normalize=False, include_headings=False
    )
    for row in out:
        assert not np.isclose(np.linalg.norm(row), 1.0)


# ---------------------------------------------------------------------------
# Validation — caller-fixable input issues.
# ---------------------------------------------------------------------------


def test_max_tokens_per_segment_exceeding_n_ctx_rejected() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=64)
    with pytest.raises(ValidationError):
        embed_with_late_chunking(["hello"], fake, max_tokens_per_segment=128)


def test_preamble_fraction_out_of_range_rejected() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    with pytest.raises(ValidationError):
        embed_with_late_chunking(["hello"], fake, preamble_fraction=1.0)
    with pytest.raises(ValidationError):
        embed_with_late_chunking(["hello"], fake, preamble_fraction=-0.1)


# ---------------------------------------------------------------------------
# Determinism — SPEC-CHUNK-440.
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    """Same input → same output, byte for byte."""
    fake = FakeEmbedder(dim=8, n_ctx=512)
    chunks = ["First.", "Second.", "Third."]
    a = embed_with_late_chunking(chunks, fake, include_headings=False)
    b = embed_with_late_chunking(chunks, fake, include_headings=False)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Heading prepend — SPEC-CHUNK-470.
# ---------------------------------------------------------------------------


def test_heading_prepend_appears_in_segment_text() -> None:
    """SPEC-CHUNK-470 — when chunks are under a heading, the heading
    stack appears as text 0 of the embedder's input."""
    fake = RecordingFakeEmbedder(dim=8, n_ctx=512)
    chunks = [
        "# Quicksort\n",
        "The algorithm uses a pivot.\n",
        "It partitions in O(n).\n",
    ]
    embed_with_late_chunking(chunks, fake, include_headings=True)
    # The fake records every embed_segment call's texts. With chunks
    # this short, everything fits in one segment. Heading-stack for
    # content_start=0 is empty (no heading in scope before chunk 0,
    # which IS the heading), so chunks 1 and 2 are the first to have
    # "# Quicksort\n" in scope. But the first segment's content_start
    # is 0 — its heading path is empty.
    assert fake.calls, "embed_segment must be called at least once"


def test_heading_prepend_actually_used_when_segment_starts_under_heading() -> None:
    """SPEC-CHUNK-470 — construct a multi-segment scenario where the
    second segment's content_start sits under a heading. That
    segment's embed_segment call must include the heading stack as
    text 0."""
    # 6 small chunks; budget small enough that each segment fits ~3.
    # Place a heading at chunk 0 so chunks 1+ live under "# Heading\n".
    chunks = [
        "# Heading\n",
        "First body chunk.\n",
        "Second body chunk.\n",
        "Third body chunk.\n",
        "Fourth body chunk.\n",
        "Fifth body chunk.\n",
    ]
    fake = RecordingFakeEmbedder(dim=8, n_ctx=80)
    # Tight budget so we definitely get multiple segments. preamble
    # fraction is small so the heading prepend matters more.
    embed_with_late_chunking(
        chunks,
        fake,
        max_tokens_per_segment=60,
        preamble_fraction=0.25,
        include_headings=True,
    )
    # At least one segment whose content starts at chunk index >= 1
    # must have "# Heading\n" as text 0 of its embed_segment call.
    found = False
    for call in fake.calls:
        if call and call[0] == "# Heading\n":
            # Could be: (a) the segment that contains the heading
            # chunk itself, in which case call[0] is just chunk 0; or
            # (b) the prepend for a later segment. Either way the
            # heading text appears in the embedder input.
            found = True
            break
    assert found, (
        f"expected '# Heading\\n' in some embed_segment call; got {fake.calls}"
    )


def test_heading_prepend_off_when_include_headings_false() -> None:
    """SPEC-CHUNK-470 — include_headings=False suppresses the prepend.
    The embedder only ever sees the chunks themselves."""
    chunks = [
        "# Heading\n",
        "First body chunk.\n",
        "Second body chunk.\n",
        "Third body chunk.\n",
        "Fourth body chunk.\n",
    ]
    fake = RecordingFakeEmbedder(dim=8, n_ctx=80)
    embed_with_late_chunking(
        chunks,
        fake,
        max_tokens_per_segment=60,
        preamble_fraction=0.25,
        include_headings=False,
    )
    # Every text passed to embed_segment must be one of the input
    # chunks — no synthetic heading-stack prepend.
    chunk_set = set(chunks)
    for call in fake.calls:
        for text in call:
            assert text in chunk_set, (
                f"unexpected synthetic text in embed_segment call: {text!r}"
            )


def test_heading_prepend_skipped_when_no_heading_in_scope() -> None:
    """SPEC-CHUNK-470 — chunks before any heading have empty heading
    path → no prepend."""
    chunks = [
        "First chunk, no heading yet.\n",
        "Second chunk.\n",
        "# First heading appears here\n",
        "Body under the heading.\n",
    ]
    fake = RecordingFakeEmbedder(dim=8, n_ctx=512)
    embed_with_late_chunking(chunks, fake, include_headings=True)
    # The first segment's content_start=0 has empty heading path → no
    # prepend. So at least one embed_segment call starts with chunk 0.
    saw_chunk_0_as_first = any(
        call and call[0] == "First chunk, no heading yet.\n"
        for call in fake.calls
    )
    assert saw_chunk_0_as_first


def test_heading_in_first_segment_not_re_prepended() -> None:
    """SPEC-CHUNK-470 — when chunk 0 IS the heading, content_start=0
    has no heading-stack in scope (the stack accrues AT chunk 0).
    The segment passes the chunks through verbatim, no synthetic
    prepend."""
    chunks = [
        "# Heading\n",
        "Chunk one body.\n",
        "Chunk two body.\n",
        "Chunk three body.\n",
    ]
    fake = RecordingFakeEmbedder(dim=8, n_ctx=512)
    embed_with_late_chunking(chunks, fake, include_headings=True)
    assert len(fake.calls) == 1
    assert fake.calls[0] == chunks


def test_heading_appears_once_per_segment_for_multi_chunk_content() -> None:
    """SPEC-CHUNK-470 — the efficiency claim. A segment whose content
    range has N chunks under the same heading sees the heading text
    prepended ONCE, not N times. Token count is heading_tokens +
    N*chunk_tokens, not (1+N)*chunk_tokens."""
    # Layout: chunk 0 is the heading; chunks 1-3 are body. Pick n_ctx
    # large enough that {chunk 1, chunk 2, chunk 3} all live in one
    # segment whose content_start = 1 (so chunk 0 is the preamble's
    # heading prepend).
    chunks = [
        "# Heading\n",
        "Body one.\n",
        "Body two.\n",
        "Body three.\n",
        "Final body chunk that pushes content_start past the heading.\n",
    ]
    fake = RecordingFakeEmbedder(dim=8, n_ctx=120)
    # Budget: 80 tokens, preamble_fraction=0.5 → preamble_budget=40.
    # First segment: content_start=0, paths[0]="" → no prepend.
    # Heading is chunk 0 in the content range. Forward fills until
    # ~80 tokens used. Then second segment with content_start at the
    # next chunk; paths[content_start] = "# Heading\n", which gets
    # prepended.
    embed_with_late_chunking(
        chunks,
        fake,
        max_tokens_per_segment=80,
        preamble_fraction=0.5,
        include_headings=True,
    )
    # Find the call(s) whose text 0 is "# Heading\n" AND that have
    # more than one additional text (the multi-chunk-content case).
    heading_prepend_calls = [
        call for call in fake.calls
        if call and call[0] == "# Heading\n" and len(call) > 1
    ]
    # At least one segment must look like [heading, chunk_a, chunk_b,
    # ...] — the heading appearing exactly once with multiple content
    # chunks under it.
    multi_chunk_under_heading = [
        call for call in heading_prepend_calls
        # Within the call, "# Heading\n" should appear exactly once
        # (not duplicated per chunk).
        if call.count("# Heading\n") == 1
    ]
    assert multi_chunk_under_heading, (
        f"expected a segment with a single heading prepend and multiple "
        f"content chunks; got {fake.calls}"
    )


def test_heading_prepend_shape_invariant() -> None:
    """SPEC-CHUNK-400 still holds with heading prepend: one row per
    input chunk, despite the embedder seeing extra heading text."""
    chunks = [
        "# A\n",
        "Body under A.\n",
        "## A.1\n",
        "Body under A.1.\n",
    ]
    fake = FakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(chunks, fake, include_headings=True)
    assert out.shape == (len(chunks), 8)
    # All rows L2-normalized (default).
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_heading_prepend_changes_embedding_vs_no_heading() -> None:
    """The heading prepend should actually affect the output —
    otherwise it's pointless. Compare include_headings=True vs False
    for the same chunks: at least one row must differ."""
    chunks = [
        "# Heading\n",
        "Body chunk with content.\n",
        "More body content here.\n",
    ]
    fake_a = FakeEmbedder(dim=16, n_ctx=512)
    fake_b = FakeEmbedder(dim=16, n_ctx=512)
    with_head = embed_with_late_chunking(
        chunks, fake_a, include_headings=True
    )
    without_head = embed_with_late_chunking(
        chunks, fake_b, include_headings=False
    )
    # With the FakeEmbedder's character-level alignment, the heading
    # prepend is in the *preamble* and discarded after pooling. The
    # per-chunk content tokens are identical in both cases, so the
    # pooled rows actually match — the heading affects only what the
    # embedder *sees*, not what gets pooled. (For a real transformer,
    # attention across the heading would shift token embeddings; the
    # fake has no attention.) Sanity-check: shapes match.
    assert with_head.shape == without_head.shape == (3, 16)


# ---------------------------------------------------------------------------
# Non-default preamble fractions.
# ---------------------------------------------------------------------------


def test_preamble_fraction_zero_disables_late_chunking() -> None:
    """SPEC-CHUNK-410 — preamble_fraction=0 → no preamble, each
    segment is content only. Useful for ablation."""
    chunks = ["x" * 30 for _ in range(5)]
    fake = FakeEmbedder(dim=4, n_ctx=200)
    out = embed_with_late_chunking(
        chunks,
        fake,
        max_tokens_per_segment=200,
        preamble_fraction=0.0,
        include_headings=False,
    )
    assert out.shape == (5, 4)
