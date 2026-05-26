"""Tests for the bundled fancychunk.embedders module.

Two layers:

* **Fast tests** (always run): import the module, sanity-check the
  factory functions return objects with the expected protocol surface,
  verify MRL truncation logic, and check backend selection.
* **Integration tests** (gated by ``FANCYCHUNK_TEST_USE_EMBEDDERS=1``):
  download + load each model, end-to-end through ``embed_chunklets``
  and ``embed_segment``, verify shapes and round-trip with
  ``split_chunks`` and ``embed_with_late_chunking``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Importing the embedders module is cheap (lazy model load).
from fancychunk.embedders import (
    NoopSegmentEmbedder,
    PooledSegmentEmbedder,
    bge_m3,
    noop,
    qwen3_4b,
    qwen3_8b,
    qwen3_600m,
)


# ---------------------------------------------------------------------------
# Fast tests (no model load).
# ---------------------------------------------------------------------------


def _mlx_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError:
        return False
    return True


def test_model_named_factories_return_pooled_segment_embedder() -> None:
    assert isinstance(bge_m3(), PooledSegmentEmbedder)
    assert isinstance(qwen3_600m(), PooledSegmentEmbedder)
    assert isinstance(qwen3_4b(), PooledSegmentEmbedder)
    assert isinstance(qwen3_8b(), PooledSegmentEmbedder)


def test_factories_pick_correct_pooling() -> None:
    assert bge_m3().pooling == "cls"
    assert qwen3_600m().pooling == "last_token"
    assert qwen3_4b().pooling == "last_token"
    assert qwen3_8b().pooling == "last_token"


def test_model_factories_pick_mlx_on_apple_silicon() -> None:
    if _mlx_available():
        assert bge_m3().model_id.startswith("mlx-community/")
        assert qwen3_600m().model_id.startswith("mlx-community/")
        assert qwen3_4b().model_id.startswith("mlx-community/")
        assert qwen3_8b().model_id.startswith("mlx-community/")
    else:
        assert bge_m3().model_id == "BAAI/bge-m3"
        assert qwen3_600m().model_id == "Qwen/Qwen3-Embedding-0.6B"
        assert qwen3_4b().model_id == "Qwen/Qwen3-Embedding-4B"
        assert qwen3_8b().model_id == "Qwen/Qwen3-Embedding-8B"


def test_qwen_models_default_to_native_dim() -> None:
    """Native dim by default; MRL truncation only when explicitly requested."""
    assert qwen3_600m().output_dim is None  # native 1024
    assert qwen3_4b().output_dim is None  # native 2560
    assert qwen3_8b().output_dim is None  # native 4096


def test_qwen_models_accept_mrl_dim() -> None:
    assert qwen3_4b(dim=1024).output_dim == 1024
    assert qwen3_4b(dim=2560).output_dim == 2560
    assert qwen3_8b(dim=1024).output_dim == 1024
    assert qwen3_8b(dim=4096).output_dim == 4096


def test_qwen3_4b_rejects_out_of_range_dim() -> None:
    with pytest.raises(ValueError):
        qwen3_4b(dim=0)
    with pytest.raises(ValueError):
        qwen3_4b(dim=4096)  # > native 2560 for the 4B model


def test_qwen3_8b_rejects_out_of_range_dim() -> None:
    with pytest.raises(ValueError):
        qwen3_8b(dim=0)
    with pytest.raises(ValueError):
        qwen3_8b(dim=8192)  # > native 4096 for the 8B model


def test_bge_m3_and_qwen3_600m_have_no_mrl_truncation() -> None:
    assert bge_m3().output_dim is None
    assert qwen3_600m().output_dim is None


def test_factories_return_fresh_instances_each_call() -> None:
    """No caching. Each factory call returns a new embedder
    instance; the caller manages lifecycle. (Pass the same instance
    twice to reuse weights, or call the factory twice to load the
    model twice — the caller's choice.)"""
    a = qwen3_600m()
    b = qwen3_600m()
    assert a is not b


# ----- noop -----


def test_noop_returns_noop_segment_embedder() -> None:
    assert isinstance(noop(), NoopSegmentEmbedder)


def test_noop_embed_chunklets_constant_unit_vectors() -> None:
    e = noop()
    chunklets = ["alpha", "beta gamma", "delta"]
    emb = e.embed_chunklets(chunklets)
    assert emb.shape == (3, e.embedding_dim)
    # All rows identical.
    assert np.allclose(emb[0], emb[1])
    assert np.allclose(emb[1], emb[2])
    # Unit norm — satisfies SPEC-CHUNK-342.
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_noop_embed_segment_satisfies_row_conservation() -> None:
    e = noop()
    texts = ["one", "two three", "four five six seven"]
    mat, counts = e.embed_segment(texts)
    assert mat.shape[1] == e.embedding_dim
    assert sum(counts) == mat.shape[0]
    assert len(counts) == len(texts)


def test_noop_with_split_chunks_structural_only() -> None:
    """Passing noop() to split_chunks reproduces the structural-only
    path: every chunklet has the same vector, so the cosine signal
    is uniform and only the heading-aware modification affects where
    splits land."""
    from fancychunk import split_chunks

    chunklets = [
        "# Heading\n",
        "First paragraph.\n",
        "Second paragraph.\n",
        "Third paragraph.\n",
    ]
    chunks = split_chunks(chunklets, noop(), max_size=2048)
    assert "".join(chunks) == "".join(chunklets)


# ---------------------------------------------------------------------------
# Integration tests (gated). Each test is decorated individually so the
# fast tests above always run.
# ---------------------------------------------------------------------------


_requires_models = pytest.mark.skipif(
    os.environ.get("FANCYCHUNK_TEST_USE_EMBEDDERS") != "1",
    reason="set FANCYCHUNK_TEST_USE_EMBEDDERS=1 to download + run real models",
)


SAMPLE_CHUNKLETS = [
    "# Quicksort\n\nQuicksort is a divide-and-conquer algorithm.\n",
    "It selects a pivot and partitions around it.\n",
    "Random pivots give O(n log n) expected time.\n",
]

SAMPLE_SENTENCES = [
    "Quicksort uses a pivot.",
    "It partitions the array around that pivot.",
    "Random pivots give O(n log n) expected time.",
]


@_requires_models
@pytest.mark.parametrize(
    "factory,expected_dim",
    [
        (bge_m3, 1024),
        (qwen3_600m, 1024),
        (qwen3_4b, 2560),
        (lambda: qwen3_4b(dim=1024), 1024),
        (lambda: qwen3_4b(dim=512), 512),
        (qwen3_8b, 4096),
        (lambda: qwen3_8b(dim=1024), 1024),
    ],
)
def test_embed_chunklets_shape_and_norm(factory, expected_dim: int) -> None:
    embedder = factory()
    emb = embedder.embed_chunklets(SAMPLE_CHUNKLETS)
    assert emb.shape == (len(SAMPLE_CHUNKLETS), expected_dim)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_embed_chunklets_handles_empty_list() -> None:
    embedder = qwen3_600m()
    emb = embedder.embed_chunklets([])
    assert emb.shape == (0, embedder.embedding_dim)


@_requires_models
@pytest.mark.parametrize("factory", [bge_m3, qwen3_600m, qwen3_4b])
def test_embedders_implement_segment_embedder_protocol(factory) -> None:
    embedder = factory()
    counts = embedder.count_tokens(SAMPLE_SENTENCES)
    assert len(counts) == len(SAMPLE_SENTENCES)
    assert all(c > 0 for c in counts)
    mat, per_text = embedder.embed_segment(SAMPLE_SENTENCES)
    assert mat.ndim == 2
    assert sum(per_text) == mat.shape[0]
    assert len(per_text) == len(SAMPLE_SENTENCES)


@_requires_models
def test_qwen3_600m_end_to_end_with_split_chunks() -> None:
    from fancychunk import split_chunklets, split_chunks, split_sentences
    from fancychunk._segmenter import punctuation_segmenter

    embedder = qwen3_600m()
    doc = "\n\n".join(["# Heading\n", "First paragraph. ", "Second paragraph."])
    sentences = split_sentences(doc, max_len=2048, segmenter=punctuation_segmenter)
    chunklets = split_chunklets(sentences, max_size=2048)
    chunks = split_chunks(chunklets, embedder, max_size=2048)
    assert "".join(chunks) == "".join(chunklets)


@_requires_models
def test_qwen3_600m_end_to_end_with_late_chunking() -> None:
    from fancychunk import embed_with_late_chunking, split_sentences
    from fancychunk._segmenter import punctuation_segmenter

    embedder = qwen3_600m()
    doc = "# Heading\n\nFirst sentence. Second sentence. Third sentence.\n"
    sentences = split_sentences(doc, max_len=2048, segmenter=punctuation_segmenter)
    # embed_with_late_chunking takes chunks now; use the sentences
    # directly as a single-chunk approximation for this smoke test.
    emb = embed_with_late_chunking(sentences, embedder, include_headings=False)
    assert emb.shape[0] == len(sentences)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_qwen3_4b_mrl_truncation_changes_output() -> None:
    full = qwen3_4b()  # native 2560
    truncated = qwen3_4b(dim=512)
    full_out = full.embed_chunklets(SAMPLE_CHUNKLETS)
    trunc_out = truncated.embed_chunklets(SAMPLE_CHUNKLETS)
    assert full_out.shape == (len(SAMPLE_CHUNKLETS), 2560)
    assert trunc_out.shape == (len(SAMPLE_CHUNKLETS), 512)
    expected = full_out[:, :512]
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(trunc_out, expected, atol=1e-4)
