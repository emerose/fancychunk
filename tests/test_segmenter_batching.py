"""Tests for SaTSegmenter device configuration and batched inference.

The model itself is heavy (~408 MB download, ~500 MB resident), so
these tests stub ``wtpsplit_lite.SaT`` in ``sys.modules`` to capture
constructor / batch arguments without invoking real inference. The
genuine SaT path is exercised separately by ``tests/test_sat.py``,
gated on ``FANCYCHUNK_TEST_USE_SAT=1``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

from fancychunk import (
    BatchSentenceSegmenter,
    SaTSegmenter,
    ValidationError,
    chunk_document,
    chunk_documents,
    precomputed_segmenter,
    split_sentences,
)

from ._fake_embedder import FakeEmbedder


class _StubSaT:
    """Stand-in for ``wtpsplit_lite.SaT`` that records inputs and
    returns synthetic probability arrays.

    ``predict_proba`` returns a generator over per-text arrays for a
    list input, and a single array for a string input — matching the
    real API."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.calls: list[list[str]] = []
        # 2-D thread safety for the global init counter:
        self.lock = threading.Lock()

    def predict_proba(self, text_or_texts: str | list[str]):
        if isinstance(text_or_texts, str):
            with self.lock:
                self.calls.append([text_or_texts])
            return self._proba_for(text_or_texts)

        texts = list(text_or_texts)
        with self.lock:
            self.calls.append(texts)

        def _gen():
            for t in texts:
                yield self._proba_for(t)

        return _gen()

    @staticmethod
    def _proba_for(text: str) -> np.ndarray:
        """Synthetic boundaries: 1.0 at every '.' followed by whitespace."""
        n = len(text)
        out = np.zeros(n, dtype=np.float32)
        for i, ch in enumerate(text):
            if ch == "." and (i == n - 1 or text[i + 1].isspace()):
                out[i] = 1.0
        return out


def _install_stub_sat(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Replace ``wtpsplit_lite.SaT`` with a recording stub.

    Returns a dict capturing the live stub state — ``instances`` is a
    list populated as ``SaT(...)`` is called.
    """
    state: dict[str, Any] = {"instances": []}

    def _factory(model_name: str, **kwargs: Any) -> _StubSaT:
        sat = _StubSaT(model_name, **kwargs)
        state["instances"].append(sat)
        return sat

    fake_module = types.ModuleType("wtpsplit_lite")
    fake_module.SaT = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wtpsplit_lite", fake_module)
    return state


# ---------------------------------------------------------------------------
# Device → ort_providers resolution.
# ---------------------------------------------------------------------------


def test_device_cpu_forces_cpu_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter(device="cpu")
    assert seg.ort_providers == ["CPUExecutionProvider"]
    seg("Hello world. Second sentence.")
    sat = state["instances"][0]
    assert sat.init_kwargs.get("ort_providers") == ["CPUExecutionProvider"]


def test_device_cuda_requests_cuda_with_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter(device="cuda")
    # CPU fallback so a misconfigured GPU box still runs.
    assert seg.ort_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    seg("Hello world. Second sentence.")
    sat = state["instances"][0]
    assert sat.init_kwargs.get("ort_providers") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_device_gpu_is_alias_for_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter(device="gpu")
    assert seg.ort_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    seg("doc.")
    sat = state["instances"][0]
    assert sat.init_kwargs.get("ort_providers") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_device_auto_defers_to_wtpsplit(monkeypatch: pytest.MonkeyPatch) -> None:
    """``device='auto'`` (the default) must NOT pass ``ort_providers=``
    so wtpsplit-lite's own auto-detect runs."""
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter()
    assert seg.ort_providers is None
    seg("doc.")
    sat = state["instances"][0]
    assert "ort_providers" not in sat.init_kwargs


def test_explicit_ort_providers_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter(
        ort_providers=["ROCMExecutionProvider", "CPUExecutionProvider"]
    )
    assert seg.ort_providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    seg("doc.")
    sat = state["instances"][0]
    assert sat.init_kwargs.get("ort_providers") == [
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_device_and_ort_providers_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="device|providers"):
        SaTSegmenter(device="cuda", ort_providers=["CPUExecutionProvider"])


def test_unknown_device_raises() -> None:
    with pytest.raises(ValidationError, match="unknown device"):
        SaTSegmenter(device="tpu")


def test_ort_kwargs_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter(ort_kwargs={"sess_options": "stub"})
    seg("doc.")
    sat = state["instances"][0]
    assert sat.init_kwargs.get("ort_kwargs") == {"sess_options": "stub"}


# ---------------------------------------------------------------------------
# Batched inference.
# ---------------------------------------------------------------------------


def test_satsegmenter_is_batch_segmenter_protocol() -> None:
    """SaTSegmenter must satisfy the BatchSentenceSegmenter protocol."""
    seg = SaTSegmenter()
    assert isinstance(seg, BatchSentenceSegmenter)


def test_predict_proba_batch_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty list short-circuits with no SaT load and no call."""
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter()
    assert seg.predict_proba_batch([]) == []
    assert state["instances"] == []


def test_predict_proba_batch_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output is one vector per doc, in input order, matching per-doc
    calls. The stub uses one batched call regardless of list size."""
    state = _install_stub_sat(monkeypatch)
    docs = [
        "First sentence. Second sentence.",
        "Another doc. With two.",
        "Third.",
    ]
    seg = SaTSegmenter()
    batched = seg.predict_proba_batch(docs)
    assert len(batched) == len(docs)
    for doc, vec in zip(docs, batched):
        assert vec.shape == (len(doc),)
        # The synthetic stub returns 1.0 at end-of-sentence period.
        # Verify those positions match.
        expected = _StubSaT._proba_for(doc).astype(np.float64)
        assert np.allclose(vec, expected)

    # One batched call, not three per-doc calls.
    sat = state["instances"][0]
    assert len(sat.calls) == 1
    assert sat.calls[0] == docs


def test_predict_proba_batch_handles_empty_and_whitespace_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty/whitespace-only docs get zero-filled vectors of the right
    shape — the downstream split_sentences short-circuits them before
    using the values, but the protocol shape must hold."""
    state = _install_stub_sat(monkeypatch)
    docs = ["First. Second.", "", "   ", "Third."]
    seg = SaTSegmenter()
    out = seg.predict_proba_batch(docs)
    assert len(out) == 4
    assert out[0].shape == (len(docs[0]),)
    assert out[1].shape == (0,)
    assert out[2].shape == (3,)
    assert np.allclose(out[2], 0.0)
    assert out[3].shape == (len(docs[3]),)

    # Only non-empty docs are forwarded to the underlying model.
    sat = state["instances"][0]
    assert len(sat.calls) == 1
    assert sat.calls[0] == ["First. Second.", "Third."]


def test_predict_proba_batch_all_empty_skips_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_stub_sat(monkeypatch)
    seg = SaTSegmenter()
    out = seg.predict_proba_batch(["", "  ", "\n\n"])
    assert [v.shape[0] for v in out] == [0, 2, 2]
    # Model is never loaded when every doc is empty/whitespace —
    # we skip _ensure_loaded entirely.
    assert state["instances"] == []


# ---------------------------------------------------------------------------
# precomputed_segmenter helper.
# ---------------------------------------------------------------------------


def test_precomputed_segmenter_round_trips_in_split_sentences() -> None:
    doc = "First sentence. Second sentence. Third one."
    n = len(doc)
    probas = np.zeros(n, dtype=np.float64)
    # Set boundary at the first period (index 14) — '.' before space.
    probas[14] = 1.0
    out = split_sentences(doc, segmenter=precomputed_segmenter(probas))
    assert "".join(out) == doc
    # The precomputed signal should produce at least two sentences.
    assert len(out) >= 2


# ---------------------------------------------------------------------------
# chunk_documents + segmenter_batch_size.
# ---------------------------------------------------------------------------


def test_chunk_documents_uses_batched_segmenter_when_requested() -> None:
    """When segmenter_batch_size is set, the segmenter's
    predict_proba_batch is invoked instead of the per-doc __call__."""
    calls: dict[str, int] = {"single": 0, "batch": 0}

    class CountingSeg:
        def __call__(self, document: str) -> np.ndarray:
            calls["single"] += 1
            return np.zeros(len(document), dtype=np.float64)

        def predict_proba_batch(
            self, documents: list[str]
        ) -> list[np.ndarray]:
            calls["batch"] += 1
            return [np.zeros(len(d), dtype=np.float64) for d in documents]

    docs = [
        "First doc. With sentence.",
        "Second one here. Two sentences.",
        "Third short.",
    ]
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    asyncio.run(
        chunk_documents(
            docs,
            embedder,
            segmenter=CountingSeg(),
            segmenter_batch_size=2,
        )
    )
    # 3 docs at batch size 2 → 2 batched calls; zero per-doc calls.
    assert calls["batch"] == 2
    assert calls["single"] == 0


def test_chunk_documents_batched_matches_serial_output() -> None:
    """Batched chunk_documents must produce identical chunks to the
    serial path (modulo unrelated nondeterminism)."""

    class CountingSeg:
        """Wraps punctuation_segmenter so we can use the same logic
        in both __call__ and predict_proba_batch paths."""

        def __init__(self) -> None:
            from fancychunk import punctuation_segmenter

            self._impl = punctuation_segmenter

        def __call__(self, document: str) -> np.ndarray:
            return self._impl(document)

        def predict_proba_batch(
            self, documents: list[str]
        ) -> list[np.ndarray]:
            return [self._impl(d) for d in documents]

    docs = [
        "# Heading\n\nFirst doc. With body content here. And more.\n",
        "Second doc with multiple sentences. Two. Three.\n",
        "Short third.\n",
    ]
    seg = CountingSeg()

    embedder_a = FakeEmbedder(dim=8, n_ctx=512)
    serial = asyncio.run(
        chunk_documents(docs, embedder_a, segmenter=seg)
    )
    embedder_b = FakeEmbedder(dim=8, n_ctx=512)
    batched = asyncio.run(
        chunk_documents(
            docs, embedder_b, segmenter=seg, segmenter_batch_size=2
        )
    )
    assert len(serial) == len(batched)
    for (cs, vs), (cb, vb) in zip(serial, batched):
        assert cs == cb
        assert np.array_equal(vs, vb)


def test_chunk_documents_rejects_batch_size_with_non_batchable_segmenter() -> None:
    """A custom segmenter without predict_proba_batch can't use
    segmenter_batch_size — raise ValidationError early, not silently
    fall back."""
    from fancychunk import punctuation_segmenter

    with pytest.raises(ValidationError, match="predict_proba_batch"):
        asyncio.run(
            chunk_documents(
                ["doc."],
                FakeEmbedder(dim=8, n_ctx=512),
                segmenter=punctuation_segmenter,
                segmenter_batch_size=4,
            )
        )


def test_chunk_documents_rejects_invalid_segmenter_batch_size() -> None:
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    with pytest.raises(ValidationError, match="segmenter_batch_size"):
        asyncio.run(
            chunk_documents(["a."], embedder, segmenter_batch_size=0)
        )
    with pytest.raises(ValidationError, match="segmenter_batch_size"):
        asyncio.run(
            chunk_documents(["a."], embedder, segmenter_batch_size=-1)
        )


def test_chunk_documents_empty_with_batch_size_is_noop() -> None:
    """Empty input + batch size → no error, no segmenter calls."""
    calls: dict[str, int] = {"batch": 0}

    class CountingSeg:
        def __call__(self, document: str) -> np.ndarray:
            return np.zeros(len(document), dtype=np.float64)

        def predict_proba_batch(
            self, documents: list[str]
        ) -> list[np.ndarray]:
            calls["batch"] += 1
            return [np.zeros(len(d), dtype=np.float64) for d in documents]

    embedder = FakeEmbedder(dim=8, n_ctx=512)
    results = asyncio.run(
        chunk_documents(
            [], embedder, segmenter=CountingSeg(), segmenter_batch_size=4
        )
    )
    assert results == []
    assert calls["batch"] == 0


def test_chunk_document_accepts_segmenter_override() -> None:
    """Per-doc chunk_document also takes segmenter= so per-call
    callers can use a custom-configured (e.g. GPU) SaTSegmenter."""
    seen: list[str] = []

    def recording_seg(document: str) -> np.ndarray:
        seen.append(document)
        # Use punctuation segmenter logic for consistent output.
        from fancychunk import punctuation_segmenter

        return punctuation_segmenter(document)

    doc = "First sentence. Second sentence.\n"
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    chunks, vectors = asyncio.run(
        chunk_document(doc, embedder, segmenter=recording_seg)
    )
    assert "".join(c.text for c in chunks) == doc
    assert seen == [doc]


def test_chunk_documents_batched_with_max_concurrency() -> None:
    """Batched + concurrent are independent dials — both can be set."""

    class CountingSeg:
        def __init__(self) -> None:
            from fancychunk import punctuation_segmenter

            self._impl = punctuation_segmenter
            self.batch_calls = 0

        def __call__(self, document: str) -> np.ndarray:
            return self._impl(document)

        def predict_proba_batch(
            self, documents: list[str]
        ) -> list[np.ndarray]:
            self.batch_calls += 1
            return [self._impl(d) for d in documents]

    docs = [f"Doc number {i}. Body here.\n" for i in range(5)]
    seg = CountingSeg()
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    results = asyncio.run(
        chunk_documents(
            docs,
            embedder,
            segmenter=seg,
            segmenter_batch_size=2,
            max_concurrency=2,
        )
    )
    assert len(results) == 5
    # ceil(5/2) = 3 batched segmenter calls.
    assert seg.batch_calls == 3
