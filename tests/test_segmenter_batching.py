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

    def predict_proba(
        self, text_or_texts: str | list[str], **kwargs: Any
    ):
        # The segmenter passes inference params (stride/block_size/
        # weighting); record them so tests can assert they're forwarded.
        self.predict_kwargs = kwargs
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
# chunk_document / chunk_documents segmenter wiring.
# ---------------------------------------------------------------------------


def test_chunk_document_accepts_segmenter_override() -> None:
    """Per-doc chunk_document takes segmenter= so per-call callers can
    use a custom-configured (e.g. GPU) SaTSegmenter. Structure-first
    only invokes the segmenter on the fallback path, so the document
    here is built to overflow ``max_size`` and force that path."""
    seen: list[str] = []

    def recording_seg(document: str) -> np.ndarray:
        seen.append(document)
        # Use punctuation segmenter logic for consistent output.
        from fancychunk import punctuation_segmenter

        return punctuation_segmenter(document)

    # No headings + over max_size → a single fallback unit covering the
    # whole document, so the segmenter sees exactly the full text. Use a
    # small max_size and a roomy n_ctx so the fallback chunks fit late
    # chunking's per-segment context budget.
    doc = "First sentence here. Second sentence here. " * 40
    embedder = FakeEmbedder(dim=8, n_ctx=4096)
    chunks, vectors = asyncio.run(
        chunk_document(doc, embedder, max_size=512, segmenter=recording_seg)
    )
    assert "".join(c.text for c in chunks) == doc
    assert seen == [doc]


def test_chunk_document_fitting_doc_skips_segmenter() -> None:
    """A document that fits ``max_size`` is emitted structurally with no
    segmenter call at all."""
    calls: list[str] = []

    def recording_seg(document: str) -> np.ndarray:
        calls.append(document)
        return np.zeros(len(document), dtype=np.float64)

    doc = "# Title\n\nShort body that comfortably fits.\n"
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    chunks, _ = asyncio.run(
        chunk_document(doc, embedder, segmenter=recording_seg)
    )
    assert "".join(c.text for c in chunks) == doc
    assert calls == []


def test_chunk_documents_empty_is_noop() -> None:
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    assert asyncio.run(chunk_documents([], embedder)) == []


def test_chunk_documents_with_max_concurrency() -> None:
    """max_concurrency caps fan-in without changing per-doc output."""
    docs = [f"# Doc {i}\n\nBody number {i}. More body.\n" for i in range(5)]
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    results = asyncio.run(
        chunk_documents(docs, embedder, max_concurrency=2)
    )
    assert len(results) == 5
    for doc, (chunks, _) in zip(docs, results):
        assert "".join(c.text for c in chunks) == doc


def test_chunk_documents_rejects_invalid_max_concurrency() -> None:
    embedder = FakeEmbedder(dim=8, n_ctx=512)
    with pytest.raises(ValidationError, match="max_concurrency"):
        asyncio.run(chunk_documents(["a."], embedder, max_concurrency=0))


# ---------------------------------------------------------------------------
# Fast token_to_char_probs monkey-patch.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    cls_token = "<s>"
    sep_token = "</s>"
    pad_token = "<pad>"


def _make_token_inputs(
    text: str,
    spans: list[tuple[int, int]],
    cols: int = 4,
    rng_seed: int = 0,
) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    """Build a (tokens, token_logits, offsets) triple where the first
    and last tokens are specials (CLS / SEP) and the middle ones cover
    ``spans``."""
    rng = np.random.default_rng(rng_seed)
    tokens = ["<s>", *[f"tok{i}" for i in range(len(spans))], "</s>"]
    offsets = [(0, 0), *spans, (0, 0)]
    token_logits = rng.standard_normal((len(tokens), cols))
    return tokens, token_logits, offsets


def test_fast_token_to_char_probs_matches_upstream_reference() -> None:
    """The vectorised postprocess must produce the same output as the
    upstream Python loop on realistic token offsets."""
    from fancychunk._segmenter import _fast_token_to_char_probs
    from wtpsplit_lite._utils import token_to_char_probs as upstream_ttc

    text = "The cat sat on the mat. A bird flew."  # len 36
    spans = [(0, 3), (4, 7), (8, 11), (12, 14), (15, 18), (19, 22)]
    spans += [(23, 24), (25, 26), (27, 31), (32, 36)]
    tokens, logits, offsets = _make_token_inputs(text, spans, cols=5)
    tok = _FakeTokenizer()
    ref = upstream_ttc(text, tokens, logits, tok, offsets)
    fast = _fast_token_to_char_probs(text, tokens, logits, tok, offsets)
    assert ref.shape == fast.shape
    assert np.allclose(ref, fast, equal_nan=True)


def test_fast_token_to_char_probs_handles_all_specials() -> None:
    """A degenerate input with only special tokens must produce an
    all--inf output (one row per character)."""
    from fancychunk._segmenter import _fast_token_to_char_probs

    text = "abc"
    tokens = ["<s>", "</s>"]
    logits = np.zeros((2, 3))
    offsets = [(0, 0), (0, 0)]
    out = _fast_token_to_char_probs(text, tokens, logits, _FakeTokenizer(), offsets)
    assert out.shape == (3, 3)
    assert np.all(np.isneginf(out))


def test_fast_token_to_char_probs_skips_out_of_bounds() -> None:
    """A token whose offset.end falls past the end of ``text`` must
    not raise — the bounds-safe replacement silently drops it."""
    from fancychunk._segmenter import _fast_token_to_char_probs

    text = "abcd"
    tokens = ["<s>", "ab", "ef", "</s>"]
    # second token "ef" has end=10, past text length 4 — must be skipped.
    offsets = [(0, 0), (0, 2), (4, 10), (0, 0)]
    logits = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = _fast_token_to_char_probs(text, tokens, logits, _FakeTokenizer(), offsets)
    assert out.shape == (4, 3)
    # row at index 1 (end-1 of token "ab" = 2-1) should match token 1's logits.
    assert np.array_equal(out[1], logits[1])
    # all other rows stay at -inf (out-of-bounds token dropped).
    expected_neginf = np.array([True, False, True, True])
    assert np.array_equal(np.all(np.isneginf(out), axis=1), expected_neginf)


def test_install_fast_postprocess_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling the installer twice must not double-patch or break the
    binding. Also covers the env-var kill switch."""
    import fancychunk._segmenter as fcs
    import wtpsplit_lite._sat as wsat

    original = wsat.token_to_char_probs
    monkeypatch.setattr(fcs, "_fast_postprocess_installed", False)
    try:
        fcs._install_fast_postprocess()
        assert wsat.token_to_char_probs is fcs._fast_token_to_char_probs
        # Idempotent — second call is a no-op.
        fcs._install_fast_postprocess()
        assert wsat.token_to_char_probs is fcs._fast_token_to_char_probs
    finally:
        wsat.token_to_char_probs = original
        monkeypatch.setattr(fcs, "_fast_postprocess_installed", False)


def test_install_fast_postprocess_respects_env_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fancychunk._segmenter as fcs
    import wtpsplit_lite._sat as wsat

    original = wsat.token_to_char_probs
    monkeypatch.setattr(fcs, "_fast_postprocess_installed", False)
    monkeypatch.setenv(fcs._FAST_POSTPROCESS_DISABLE_ENV, "1")
    try:
        fcs._install_fast_postprocess()
        # Kill switch set → upstream binding unchanged.
        assert wsat.token_to_char_probs is original
        # Flag still flipped so we don't re-check the env on every call.
        assert fcs._fast_postprocess_installed is True
    finally:
        wsat.token_to_char_probs = original
        monkeypatch.setattr(fcs, "_fast_postprocess_installed", False)


# ---------------------------------------------------------------------------
# wants_batching() — SaTSegmenter capability hint.
# ---------------------------------------------------------------------------


def test_wants_batching_true_for_explicit_cuda() -> None:
    seg = SaTSegmenter(device="cuda")
    assert seg.wants_batching() is True


def test_wants_batching_false_for_explicit_cpu() -> None:
    seg = SaTSegmenter(device="cpu")
    assert seg.wants_batching() is False


def test_wants_batching_true_for_explicit_gpu_alias() -> None:
    assert SaTSegmenter(device="gpu").wants_batching() is True


def test_wants_batching_respects_explicit_ort_providers() -> None:
    # ROCm — counts as GPU.
    seg = SaTSegmenter(ort_providers=["ROCMExecutionProvider", "CPUExecutionProvider"])
    assert seg.wants_batching() is True
    # Custom CPU-only list.
    seg = SaTSegmenter(ort_providers=["CPUExecutionProvider"])
    assert seg.wants_batching() is False


def test_wants_batching_auto_peeks_at_available_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``device="auto"`` consults onnxruntime.get_available_providers."""
    import onnxruntime as ort

    seg = SaTSegmenter(device="auto")
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    assert seg.wants_batching() is False
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert seg.wants_batching() is True
