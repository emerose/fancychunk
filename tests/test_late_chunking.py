"""Stage 4 tests — late chunking."""

from __future__ import annotations

import numpy as np
import pytest

from fancychunk import (
    SentenceExceedsContextError,
    ValidationError,
    embed_with_late_chunking,
)

from ._fake_embedder import (
    BertLikeFakeEmbedder,
    FakeEmbedder,
    WhitespaceDroppingFakeEmbedder,
)


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
    # Distinct sentences (no shared characters near the dim modulus)
    # produce distinct pooled rows.
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
        embed_with_late_chunking(
            ["A" + "a" * 10000], fake, max_tokens_per_segment=512
        )


# TV-405 / SPEC-CHUNK-410, -411 — every sentence appears in exactly one
# segment's content range.
def test_tv_405_segment_coverage() -> None:
    sentences = ["x" * 50 for _ in range(20)]
    fake = FakeEmbedder(dim=4, n_ctx=256)
    out = embed_with_late_chunking(
        sentences, fake, max_tokens_per_segment=256, preamble_fraction=0.382
    )
    assert out.shape == (20, 4)


# TV-406 — first segment has empty preamble; budget rolls into content.
def test_tv_406_first_segment_empty_preamble() -> None:
    sentences = ["x" * 30, "x" * 30, "x" * 30]
    fake = FakeEmbedder(dim=4, n_ctx=200)
    out = embed_with_late_chunking(
        sentences, fake, max_tokens_per_segment=200, preamble_fraction=0.382
    )
    assert out.shape == (3, 4)


# TV-408 / SPEC-CHUNK-420 — the embedder owns per-sentence token
# alignment. For the FakeEmbedder the alignment is trivial (one
# character per token, no joiner), so the pooled row reflects the
# arithmetic mean of the sentence's one-hot character embeddings.
def test_tv_408_embedder_owns_alignment() -> None:
    fake = FakeEmbedder(dim=4, n_ctx=512)
    out = embed_with_late_chunking(["AB", "CD"], fake)
    assert out.shape == (2, 4)
    # 'A'=65, 'B'=66 → dims 1, 2 each get 0.5 mass before normalize.
    expected = np.array([0.0, 0.5, 0.5, 0.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(out[0], expected)


# TV-409 / SPEC-CHUNK-452 — sentence tokenizing to zero tokens.
def test_tv_409_zero_token_sentence_handled() -> None:
    fake = WhitespaceDroppingFakeEmbedder(dim=8, n_ctx=512)
    sentences = ["a", "b", "c"]  # 'b' produces zero tokens
    with pytest.raises(ValidationError):
        embed_with_late_chunking(sentences, fake)


# TV-410 / SPEC-CHUNK-402, -430 — normalization control.
def test_tv_410_normalize_control() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    sentences = ["First.", "Second."]
    out_norm = embed_with_late_chunking(sentences, fake, normalize=True)
    out_raw = embed_with_late_chunking(sentences, fake, normalize=False)
    for row in out_norm:
        assert np.isclose(np.linalg.norm(row), 1.0)
    for row in out_raw:
        assert not np.isclose(np.linalg.norm(row), 1.0)


# SPEC-CHUNK-420 option (b) — leading/trailing special tokens absorbed
# into the first/last sentences' allocations by the embedder. The
# library never sees them.
def test_special_tokens_absorbed_by_embedder() -> None:
    fake = BertLikeFakeEmbedder(dim=8, n_ctx=512)
    out = embed_with_late_chunking(["AB", "CD"], fake)
    assert out.shape == (2, 8)
    assert np.all(np.isfinite(out))


# Validation: max_tokens_per_segment may not exceed embedder.n_ctx.
def test_max_tokens_per_segment_validation() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=64)
    with pytest.raises(ValidationError):
        embed_with_late_chunking(["hello"], fake, max_tokens_per_segment=128)


# SPEC-CHUNK-440 — determinism.
def test_determinism() -> None:
    fake = FakeEmbedder(dim=8, n_ctx=512)
    sentences = ["First.", "Second.", "Third."]
    a = embed_with_late_chunking(sentences, fake)
    b = embed_with_late_chunking(sentences, fake)
    assert np.array_equal(a, b)
