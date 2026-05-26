"""Tests for the bundled fancychunk.embedders module.

Two layers:

* **Fast tests** (always run): import the module, sanity-check the
  factory functions return objects with the expected protocol surface,
  and verify MRL truncation logic on a synthetic matrix without
  loading any model.
* **Integration tests** (gated by ``FANCYCHUNK_TEST_USE_EMBEDDERS=1``):
  download + load each model, end-to-end through ``embed_chunklets``
  and ``embed_segment``, verify shapes and round-trip with
  ``split_chunks`` and ``embed_with_late_chunking``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Importing the embedders module is cheap (lazy model load).
from fancychunk.embedders import (
    PooledSegmentEmbedder,
    default,
    fast,
    high_quality,
)


# ---------------------------------------------------------------------------
# Fast tests (no model load).
# ---------------------------------------------------------------------------


def test_factories_return_pooled_segment_embedder() -> None:
    d = default()
    f = fast()
    hq = high_quality()
    assert isinstance(d, PooledSegmentEmbedder)
    assert isinstance(f, PooledSegmentEmbedder)
    assert isinstance(hq, PooledSegmentEmbedder)


def test_factories_carry_expected_model_ids() -> None:
    assert default().model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert fast().model_id == "BAAI/bge-m3"
    assert high_quality().model_id == "Qwen/Qwen3-Embedding-4B"


def test_factories_pick_correct_pooling() -> None:
    assert default().pooling == "last_token"
    assert fast().pooling == "cls"
    assert high_quality().pooling == "last_token"


def test_high_quality_defaults_to_1024_dim() -> None:
    assert high_quality().output_dim == 1024
    assert high_quality(dim=512).output_dim == 512
    assert high_quality(dim=2560).output_dim == 2560


def test_high_quality_rejects_out_of_range_dim() -> None:
    with pytest.raises(ValueError):
        high_quality(dim=0)
    with pytest.raises(ValueError):
        high_quality(dim=4096)  # > native 2560


def test_default_and_fast_have_no_mrl_truncation() -> None:
    assert default().output_dim is None
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
        (default, 1024),
        (fast, 1024),
        (lambda: high_quality(dim=1024), 1024),
        (lambda: high_quality(dim=512), 512),
    ],
)
def test_embed_chunklets_shape_and_norm(factory, expected_dim: int) -> None:
    embedder = factory()
    emb = embedder.embed_chunklets(SAMPLE_CHUNKLETS)
    assert emb.shape == (len(SAMPLE_CHUNKLETS), expected_dim)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_embed_chunklets_handles_empty_list() -> None:
    embedder = default()
    emb = embedder.embed_chunklets([])
    assert emb.shape == (0, embedder.embedding_dim)


@_requires_models
@pytest.mark.parametrize("factory", [default, fast, lambda: high_quality()])
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
def test_default_end_to_end_with_split_chunks() -> None:
    from fancychunk import split_chunklets, split_chunks, split_sentences
    from fancychunk._segmenter import punctuation_segmenter

    embedder = default()
    doc = "\n\n".join(["# Heading\n", "First paragraph. ", "Second paragraph."])
    sentences = split_sentences(doc, max_len=2048, segmenter=punctuation_segmenter)
    chunklets = split_chunklets(sentences, max_size=2048)
    embeddings = embedder.embed_chunklets(chunklets)
    chunks, _ = split_chunks(chunklets, embeddings, max_size=2048)
    assert "".join(chunks) == "".join(chunklets)


@_requires_models
def test_default_end_to_end_with_late_chunking() -> None:
    from fancychunk import embed_with_late_chunking, split_sentences
    from fancychunk._segmenter import punctuation_segmenter

    embedder = default()
    doc = "# Heading\n\nFirst sentence. Second sentence. Third sentence.\n"
    sentences = split_sentences(doc, max_len=2048, segmenter=punctuation_segmenter)
    emb = embed_with_late_chunking(sentences, embedder)
    assert emb.shape[0] == len(sentences)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_mrl_truncation_changes_output() -> None:
    full = high_quality(dim=2560)
    truncated = high_quality(dim=512)
    full_out = full.embed_chunklets(SAMPLE_CHUNKLETS)
    trunc_out = truncated.embed_chunklets(SAMPLE_CHUNKLETS)
    assert full_out.shape == (len(SAMPLE_CHUNKLETS), 2560)
    assert trunc_out.shape == (len(SAMPLE_CHUNKLETS), 512)
    # Truncated output should equal the first 512 dims of full, re-normalized.
    expected = full_out[:, :512]
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(trunc_out, expected, atol=1e-4)
