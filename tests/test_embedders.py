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

import asyncio
import concurrent.futures
import os
import sys
import threading
import time

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
    emb = asyncio.run(e.embed_chunklets(chunklets))
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
    mat, counts = asyncio.run(e.embed_segment(texts))
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
    chunks = asyncio.run(split_chunks(chunklets, noop(), max_size=2048))
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
    emb = asyncio.run(embedder.embed_chunklets(SAMPLE_CHUNKLETS))
    assert emb.shape == (len(SAMPLE_CHUNKLETS), expected_dim)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


@_requires_models
def test_embed_chunklets_handles_empty_list() -> None:
    embedder = qwen3_600m()
    emb = asyncio.run(embedder.embed_chunklets([]))
    assert emb.shape == (0, embedder.embedding_dim)


@_requires_models
@pytest.mark.parametrize("factory", [bge_m3, qwen3_600m, qwen3_4b])
def test_embedders_implement_segment_embedder_protocol(factory) -> None:
    embedder = factory()
    counts = asyncio.run(embedder.count_tokens(SAMPLE_SENTENCES))
    assert len(counts) == len(SAMPLE_SENTENCES)
    assert all(c > 0 for c in counts)
    mat, per_text = asyncio.run(embedder.embed_segment(SAMPLE_SENTENCES))
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
    chunks = asyncio.run(split_chunks(chunklets, embedder, max_size=2048))
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
    emb = asyncio.run(
        embed_with_late_chunking(sentences, embedder, include_headings=False)
    )
    assert emb.shape[0] == len(sentences)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Thread-safety (fast — uses monkey-patched loader, no model download).
# ---------------------------------------------------------------------------


class _StubModel:
    """Stand-in for a torch/MLX model. Records concurrent forward calls so
    tests can assert the embedder's lock serializes them."""

    class _Config:
        hidden_size = 8

    def __init__(self) -> None:
        self.config = _StubModel._Config()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self._counter_lock = threading.Lock()

    def __call__(self, *args, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("stub model not meant to be invoked directly")

    def _enter(self) -> None:
        with self._counter_lock:
            self.in_flight += 1
            self.calls += 1
            if self.in_flight > self.max_in_flight:
                self.max_in_flight = self.in_flight

    def _exit(self) -> None:
        with self._counter_lock:
            self.in_flight -= 1


class _StubTokenizer:
    model_max_length = 512

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(s)


def _install_stub(emb: PooledSegmentEmbedder, load_delay: float = 0.0):
    """Replace ``_load_torch`` with a stub that simulates a slow loader.
    Returns a counter dict the test can read to verify load-once."""
    state = {"load_count": 0}
    load_lock = threading.Lock()

    def fake_load() -> None:
        time.sleep(load_delay)
        with load_lock:
            state["load_count"] += 1
        emb._backend = "torch"
        emb._device = "cpu"
        emb._tokenizer = _StubTokenizer()
        emb._model = _StubModel()

    emb._resolve_backend = lambda: "torch"  # type: ignore[method-assign]
    emb._load_torch = fake_load  # type: ignore[method-assign]
    return state


def test_concurrent_first_call_does_not_double_load() -> None:
    """Lazy load is guarded — N threads racing on a fresh embedder
    see exactly one weight load."""
    emb = PooledSegmentEmbedder(model_id="fake", pooling="mean")
    state = _install_stub(emb, load_delay=0.05)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: emb._ensure_loaded(), range(8)))

    assert state["load_count"] == 1


def test_concurrent_count_tokens_serialized() -> None:
    """``count_tokens`` and the lazy load share one lock; gathered
    coroutines complete without crashing and the model loads once.
    Post-migration, ``count_tokens`` is async and offloads to a
    worker thread via ``asyncio.to_thread``; ``gather`` exercises the
    same code path multi-segment late chunking will use."""
    emb = PooledSegmentEmbedder(model_id="fake", pooling="mean")
    state = _install_stub(emb, load_delay=0.01)
    inputs = [[f"sentence {i}"] for i in range(32)]

    async def _drive() -> list[list[int]]:
        return await asyncio.gather(*(emb.count_tokens(inp) for inp in inputs))

    results = asyncio.run(_drive())

    assert state["load_count"] == 1
    assert len(results) == 32
    assert all(len(r) == 1 and r[0] == len(inp[0]) for r, inp in zip(results, inputs))


def test_embed_segment_serializes_forward_calls() -> None:
    """``embed_segment`` holds the instance lock across the forward
    pass — gathered coroutines never enter the model concurrently."""

    class _FakeTokenizer:
        model_max_length = 512

        def __call__(self, joined, **kwargs):
            return {
                "input_ids": np.array([[0] * max(1, len(joined))]),
                "attention_mask": np.array([[1] * max(1, len(joined))]),
                "offset_mapping": np.array(
                    [[(i, i + 1) for i in range(max(1, len(joined)))]]
                ),
            }

        def encode(self, s, add_special_tokens=False):
            return [0] * len(s)

    emb = PooledSegmentEmbedder(model_id="fake", pooling="mean")
    emb._resolve_backend = lambda: "torch"  # type: ignore[method-assign]

    def fake_load() -> None:
        emb._backend = "torch"
        emb._device = "cpu"
        emb._tokenizer = _FakeTokenizer()
        emb._model = _StubModel()

    emb._load_torch = fake_load  # type: ignore[method-assign]

    def fake_forward(ids, attention_mask) -> np.ndarray:
        emb._model._enter()
        try:
            time.sleep(0.01)
        finally:
            emb._model._exit()
        n = ids.shape[1]
        return np.zeros((n, 8), dtype=np.float64)

    emb._forward_torch_per_token = fake_forward  # type: ignore[method-assign]

    inputs = [[f"text {i}"] for i in range(16)]

    async def _drive():
        return await asyncio.gather(*(emb.embed_segment(inp) for inp in inputs))

    results = asyncio.run(_drive())

    assert len(results) == 16
    assert emb._model.calls == 16
    assert emb._model.max_in_flight == 1


# ---------------------------------------------------------------------------
# Thread-safety (gated — real model load).
# ---------------------------------------------------------------------------


@_requires_models
def test_concurrent_embed_segment_is_safe() -> None:
    """End-to-end: gather many concurrent embed_segment coroutines
    against a real embedder. Should not crash, hang, or return
    malformed output."""
    emb = qwen3_600m()
    texts = [[f"sentence number {i}"] for i in range(16)]

    async def _drive():
        return await asyncio.gather(*(emb.embed_segment(t) for t in texts))

    results = asyncio.run(_drive())
    assert len(results) == 16
    for mat, counts in results:
        assert mat.ndim == 2
        assert sum(counts) == mat.shape[0]


@_requires_models
def test_qwen3_4b_mrl_truncation_changes_output() -> None:
    full = qwen3_4b()  # native 2560
    truncated = qwen3_4b(dim=512)
    full_out = asyncio.run(full.embed_chunklets(SAMPLE_CHUNKLETS))
    trunc_out = asyncio.run(truncated.embed_chunklets(SAMPLE_CHUNKLETS))
    assert full_out.shape == (len(SAMPLE_CHUNKLETS), 2560)
    assert trunc_out.shape == (len(SAMPLE_CHUNKLETS), 512)
    expected = full_out[:, :512]
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(trunc_out, expected, atol=1e-4)
