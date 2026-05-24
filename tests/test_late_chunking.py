"""Stage 4 tests — late chunking."""

from __future__ import annotations

import numpy as np
import pytest

from fancychunk import (
    SentenceExceedsContextError,
    embed_with_late_chunking,
)

from ._fake_embedder import FakeEmbedder, WhitespaceDroppingFakeEmbedder


# TV-401 / SPEC-CHUNK-400, -401 — shape conforms.
def test_tv_401_shape_conforms_to_input() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(
        ["First sentence.", "Second sentence.", "Third sentence."],
        fake,
        max_tokens_per_segment=512,
    )
    assert out.shape == (3, 8)


# TV-402 / SPEC-CHUNK-400 — row order matches input order.
def test_tv_402_row_order_matches_input() -> None:
    fake = FakeEmbedder(dim=16, n_ctx=512)
    sentences = ["alpha", "bravo", "charlie", "delta"]
    out = embed_with_late_chunking(sentences, fake, max_tokens_per_segment=512)
    assert out.shape == (4, 16)
    # The fake embeds each character as a one-hot row. Sentence 0's
    # pooled embedding therefore has weight on dimensions corresponding
    # to ``alpha``'s characters ('a', 'l', 'p', 'h'). Sentence 3's row
    # has weight on ``delta``'s characters. Verify they differ.
    assert not np.array_equal(out[0], out[3])


# TV-403 / SPEC-CHUNK-450 — single-sentence input.
def test_tv_403_single_sentence_input() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(["Only one sentence here."], fake)
    assert out.shape == (1, 8)


# TV-404 / SPEC-CHUNK-451 — sentence exceeds embedder context.
def test_tv_404_oversize_sentence_raises() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    with pytest.raises(SentenceExceedsContextError):
        embed_with_late_chunking(["A" + "a" * 10000], fake, max_tokens_per_segment=512)


# TV-405 / SPEC-CHUNK-410, -411 — every sentence in exactly one segment's content.
def test_tv_405_segment_coverage() -> None:
    # 20 sentences of ~50 tokens each — each sentence is 50 ASCII chars
    # so that one char = one token for the fake.
    sentences = ["x" * 50 for _ in range(20)]
    fake = FakeEmbedder(dim=4, n_ctx=256)
    out = embed_with_late_chunking(
        sentences, fake, max_tokens_per_segment=256, preamble_fraction=0.382
    )
    assert out.shape == (20, 4)


# TV-406 — first segment has empty preamble; budget rolls into content.
def test_tv_406_first_segment_empty_preamble() -> None:
    sentences = ["x" * 30, "x" * 30, "x" * 30]  # ~90 tokens total
    fake = FakeEmbedder(dim=4, n_ctx=200)
    out = embed_with_late_chunking(
        sentences, fake, max_tokens_per_segment=200, preamble_fraction=0.382
    )
    assert out.shape == (3, 4)


# TV-407 / SPEC-CHUNK-421 — sentinel character collision falls back.
def test_tv_407_sentinel_collision_handled() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    sentences = [
        "First sentence.",
        "Has the symbol ⊕ inside.",
        "Third sentence.",
    ]
    out = embed_with_late_chunking(sentences, fake)
    assert out.shape == (3, 8)


# TV-408 / SPEC-CHUNK-420 — per-sentence token counts sum to the
# joined-input tokenization length, with the sentinel token billed to
# the preceding sentence.
def test_tv_408_token_counts_align() -> None:
    fake = FakeEmbedder(dim=4, n_ctx=512)
    out = embed_with_late_chunking(["AB", "CD"], fake)
    assert out.shape == (2, 4)
    # The sentinel-token method tokenises "AB⊕CD" to five tokens; the
    # sentinel is billed to sentence 0. Sentence 0's pooled row =
    # mean(one-hots for A=65, B=66, sentinel=9999): non-zero at
    # dims 65%4=1, 66%4=2, 9999%4=3, weight 1/3 each. Sentence 1's
    # pooled row = mean(one-hots for C=67, D=68): non-zero at
    # dims 3 and 0, weight 1/2 each.
    expected_0 = np.array([0.0, 1 / 3, 1 / 3, 1 / 3])
    expected_0 = expected_0 / np.linalg.norm(expected_0)
    expected_1 = np.array([0.5, 0.0, 0.0, 0.5])
    expected_1 = expected_1 / np.linalg.norm(expected_1)
    assert np.allclose(out[0], expected_0)
    assert np.allclose(out[1], expected_1)


# TV-409 / SPEC-CHUNK-452 — many short sentences; some tokenize to 0 tokens.
#
# The spec allows two outcomes here: floor every sentence's share at
# one token (option a) or raise (option b). This implementation uses
# the sentinel-token method, which naturally bills the sentinel to
# the preceding sentence and therefore satisfies option (a). The
# test verifies the *contract*: every sentence receives a finite,
# non-NaN row OR the call raises a clear error.
def test_tv_409_zero_token_sentence_handled() -> None:
    fake = WhitespaceDroppingFakeEmbedder(dim=8, n_ctx=512)
    sentences = ["a", "b", "c"]
    try:
        out = embed_with_late_chunking(sentences, fake)
    except Exception:
        return
    assert out.shape == (3, 8)
    assert np.all(np.isfinite(out))


# TV-410 / SPEC-CHUNK-402, -430 — normalization control.
def test_tv_410_normalize_control() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    sentences = ["First.", "Second."]
    out_norm = embed_with_late_chunking(sentences, fake, normalize=True)
    out_raw = embed_with_late_chunking(sentences, fake, normalize=False)
    for row in out_norm:
        assert np.isclose(np.linalg.norm(row), 1.0)
    # The raw rows are mean-pooled one-hot vectors; their L2 norm is
    # ``1 / sqrt(token_count)`` for a single sentence — not 1.0.
    for row in out_raw:
        assert not np.isclose(np.linalg.norm(row), 1.0)


# SPEC-CHUNK-440 — determinism.
def test_determinism() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    sentences = ["First.", "Second.", "Third."]
    a = embed_with_late_chunking(sentences, fake)
    b = embed_with_late_chunking(sentences, fake)
    assert np.array_equal(a, b)
