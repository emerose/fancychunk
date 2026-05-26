"""Tests for ``chunk_document`` — the one-call composed pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fancychunk import chunk_document
from fancychunk.embedders import noop

from ._fake_embedder import FakeEmbedder


def test_chunk_document_returns_chunks_and_vectors() -> None:
    """Basic contract: ``(chunks, vectors)``, chunks reconstruct the
    document, vectors have one row per chunk."""
    doc = (
        "# Introduction\n\n"
        "First paragraph here. Second sentence.\n\n"
        "## Methods\n\n"
        "Body of methods section.\n"
    )
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    chunks, vectors = chunk_document(doc, embedder)

    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert "".join(chunks) == doc  # SPEC-CHUNK-300 round-trip
    assert vectors.shape == (len(chunks), 8)


def test_chunk_document_vectors_are_l2_normalized() -> None:
    """Late-chunking default is normalize=True, so each row has unit
    L2 norm."""
    doc = "First sentence. Second sentence.\n\nThird sentence.\n"
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    chunks, vectors = chunk_document(doc, embedder)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_chunk_document_empty_input() -> None:
    """Empty document: zero chunks, zero-row matrix of the embedder's
    output dim."""
    embedder = FakeEmbedder(dim=4, n_ctx=512)
    chunks, vectors = chunk_document("", embedder)
    assert chunks == []
    assert vectors.shape == (0, 4)


def test_chunk_document_single_short_input() -> None:
    """One-sentence document → one chunk → one vector."""
    doc = "Just one sentence here.\n"
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    chunks, vectors = chunk_document(doc, embedder)
    assert len(chunks) == 1
    assert vectors.shape == (1, 8)
    assert "".join(chunks) == doc


def test_chunk_document_reuses_embedder_instance() -> None:
    """The same embedder instance drives split_chunks (via
    embed_chunklets) and embed_with_late_chunking (via
    count_tokens + embed_segment). One instance = one model load
    when the embedder is real."""

    class CallCountingFake(FakeEmbedder):
        embed_chunklets_calls: int = 0
        embed_segment_calls: int = 0

        def embed_chunklets(self, chunklets):  # type: ignore[no-untyped-def]
            self.embed_chunklets_calls += 1
            return super().embed_chunklets(chunklets)

        def embed_segment(self, texts):  # type: ignore[no-untyped-def]
            self.embed_segment_calls += 1
            return super().embed_segment(texts)

    # Document large enough to force a multi-chunk partition so
    # split_chunks actually invokes the embedder.
    doc = (
        "# Top\n\n"
        + "A" * 1500
        + ". "
        + "B" * 1500
        + ". "
        + "C" * 1500
        + ".\n"
    )
    embedder = CallCountingFake(dim=8, n_ctx=2048)
    _, _ = chunk_document(doc, embedder, max_size=2048)
    # split_chunks calls embed_chunklets exactly once.
    assert embedder.embed_chunklets_calls == 1
    # embed_with_late_chunking calls embed_segment at least once
    # (possibly more if the document needs multiple segments).
    assert embedder.embed_segment_calls >= 1


def test_chunk_document_with_noop_embedder_works() -> None:
    """``noop()`` satisfies the Embedder protocol; chunk_document
    runs end-to-end against it (vectors are constant but the
    function doesn't crash)."""
    doc = "# Heading\n\nFirst body. Second body.\n"
    chunks, vectors = chunk_document(doc, noop())
    assert "".join(chunks) == doc
    assert vectors.shape[0] == len(chunks)
    # noop's late-chunking output is constants → all rows identical.
    if len(chunks) > 1:
        assert np.allclose(vectors[0], vectors[1])


def test_chunk_document_respects_max_size() -> None:
    """Every chunk in the output is at most ``max_size`` characters."""
    doc = (
        ". ".join("Body sentence " + "x" * 200 for _ in range(20)) + ".\n"
    )
    embedder = FakeEmbedder(dim=8, n_ctx=4096)
    chunks, _ = chunk_document(doc, embedder, max_size=2048)
    for c in chunks:
        assert len(c) <= 2048


def test_chunk_document_determinism() -> None:
    """Same input + same embedder → same output."""
    doc = "First. Second. Third sentence here.\n\nAnother paragraph.\n"
    chunks_a, vectors_a = chunk_document(doc, FakeEmbedder(dim=8, n_ctx=512))
    chunks_b, vectors_b = chunk_document(doc, FakeEmbedder(dim=8, n_ctx=512))
    assert chunks_a == chunks_b
    assert np.array_equal(vectors_a, vectors_b)
