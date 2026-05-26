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
    PooledSegmentEmbedder,
    fast,
    fastest,
    high,
    medium,
)


# ---------------------------------------------------------------------------
# Fast tests (no model load).
# ---------------------------------------------------------------------------


def test_factories_return_pooled_segment_embedder() -> None:
    assert isinstance(fastest(), PooledSegmentEmbedder)
    assert isinstance(fast(), PooledSegmentEmbedder)
    assert isinstance(medium(), PooledSegmentEmbedder)
    assert isinstance(high(), PooledSegmentEmbedder)


def test_factories_pick_correct_pooling() -> None:
    assert fastest().pooling == "cls"
    assert fast().pooling == "last_token"
    assert medium().pooling == "last_token"
    assert high().pooling == "last_token"


def test_factories_use_mlx_on_apple_silicon_with_mlx_embeddings() -> None:
    try:
        import mlx_embeddings  # noqa: F401

        mlx_available = True
    except ImportError:
        mlx_available = False
    on_apple_silicon = sys.platform == "darwin"

    if on_apple_silicon and mlx_available:
        assert fastest().model_id.startswith("mlx-community/")
        assert fast().model_id.startswith("mlx-community/")
        assert medium().model_id.startswith("mlx-community/")
        assert high().model_id.startswith("mlx-community/")
    else:
        assert fastest().model_id == "BAAI/bge-m3"
        assert fast().model_id == "Qwen/Qwen3-Embedding-0.6B"
        assert medium().model_id == "Qwen/Qwen3-Embedding-4B"
        assert high().model_id == "Qwen/Qwen3-Embedding-8B"


def test_medium_defaults_to_1024_dim() -> None:
    assert medium().output_dim == 1024
    assert medium(dim=512).output_dim == 512
    assert medium(dim=2560).output_dim == 2560


def test_medium_rejects_out_of_range_dim() -> None:
    with pytest.raises(ValueError):
        medium(dim=0)
    with pytest.raises(ValueError):
        medium(dim=4096)  # > native 2560 for the 4B model


def test_high_defaults_to_1024_dim() -> None:
    assert high().output_dim == 1024
    assert high(dim=512).output_dim == 512
    assert high(dim=4096).output_dim == 4096


def test_high_rejects_out_of_range_dim() -> None:
    with pytest.raises(ValueError):
        high(dim=0)
    with pytest.raises(ValueError):
        high(dim=8192)  # > native 4096 for the 8B model


def test_fastest_and_fast_have_no_mrl_truncation() -> None:
    assert fastest().output_dim is None
    assert fast().output_dim is None


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
        (fastest, 1024),
        (fast, 1024),
        (lambda: medium(dim=1024), 1024),
        (lambda: medium(dim=512), 512),
        (lambda: high(dim=1024), 1024),
    ],
)
def test_embed_chunklets_shape_and_norm(factory, expected_dim: int) -> None:
    embedder = factory()
    emb = embedder.embed_chunklets(SAMPLE_CHUNKLETS)
    assert emb.shape == (len(SAMPLE_CHUNKLETS), expected_dim)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_embed_chunklets_handles_empty_list() -> None:
    embedder = fast()
    emb = embedder.embed_chunklets([])
    assert emb.shape == (0, embedder.embedding_dim)


@_requires_models
@pytest.mark.parametrize("factory", [fastest, fast, lambda: medium()])
def test_embedders_implement_segment_embedder_protocol(factory) -> None:
    embedder = factory()
    counts = embedder.count_tokens(SAMPLE_SENTENCES)
    assert len(counts) == len(SAMPLE_SENTENCES)
    assert all(c > 0 for c in counts)
    mat, per_sentence = embedder.embed_segment(SAMPLE_SENTENCES)
    assert mat.ndim == 2
    assert sum(per_sentence) == mat.shape[0]
    assert len(per_sentence) == len(SAMPLE_SENTENCES)


@_requires_models
def test_fast_end_to_end_with_split_chunks() -> None:
    from fancychunk import split_chunklets, split_chunks, split_sentences
    from fancychunk._segmenter import punctuation_segmenter

    embedder = fast()
    doc = "\n\n".join(["# Heading\n", "First paragraph. ", "Second paragraph."])
    sentences = split_sentences(doc, max_len=2048, segmenter=punctuation_segmenter)
    chunklets = split_chunklets(sentences, max_size=2048)
    embeddings = embedder.embed_chunklets(chunklets)
    chunks, _ = split_chunks(chunklets, embeddings, max_size=2048)
    assert "".join(chunks) == "".join(chunklets)


@_requires_models
def test_fast_end_to_end_with_late_chunking() -> None:
    from fancychunk import embed_with_late_chunking, split_sentences
    from fancychunk._segmenter import punctuation_segmenter

    embedder = fast()
    doc = "# Heading\n\nFirst sentence. Second sentence. Third sentence.\n"
    sentences = split_sentences(doc, max_len=2048, segmenter=punctuation_segmenter)
    emb = embed_with_late_chunking(sentences, embedder)
    assert emb.shape[0] == len(sentences)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_medium_mrl_truncation_changes_output() -> None:
    full = medium(dim=2560)
    truncated = medium(dim=512)
    full_out = full.embed_chunklets(SAMPLE_CHUNKLETS)
    trunc_out = truncated.embed_chunklets(SAMPLE_CHUNKLETS)
    assert full_out.shape == (len(SAMPLE_CHUNKLETS), 2560)
    assert trunc_out.shape == (len(SAMPLE_CHUNKLETS), 512)
    expected = full_out[:, :512]
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(trunc_out, expected, atol=1e-4)
