"""Tracing-contract tests.

These tests install an in-memory OpenTelemetry SDK exporter and verify
that every public stage emits a span with the documented name and
attribute set. They run quickly because no real exporter is involved.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from fancychunk import (
    Chunk,
    heading_paths,
    split_chunklets,
    split_sentences,
)
from fancychunk import embed_with_late_chunking as _async_embed_with_late_chunking
from fancychunk import split_chunks as _async_split_chunks

from ._fake_embedder import FakeEmbedder


# Sync shims — see test_late_chunking.py for rationale.
def embed_with_late_chunking(chunks, *args, **kwargs):  # type: ignore[no-untyped-def]
    if chunks and isinstance(chunks[0], str):
        chunks = [Chunk(text=s) for s in chunks]
    return asyncio.run(_async_embed_with_late_chunking(chunks, *args, **kwargs))


def split_chunks(*args, **kwargs):  # type: ignore[no-untyped-def]
    return asyncio.run(_async_split_chunks(*args, **kwargs))


def heading_paths_via_wrap(texts):  # type: ignore[no-untyped-def]
    """Wrap raw strings as Chunks and call the real heading_paths."""
    return heading_paths([Chunk(text=t) for t in texts])


@pytest.fixture
def captured_spans() -> tuple[InMemorySpanExporter, TracerProvider]:
    """Provide an in-memory exporter wired into a fresh TracerProvider.

    The fixture replaces the global tracer provider; tests use the
    returned exporter to inspect spans and the provider to flush.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # ``set_tracer_provider`` warns on double-set, but tests are
    # process-local. Use the override directly.
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    return exporter, provider


def _attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def test_split_sentences_emits_span(captured_spans) -> None:
    exporter, provider = captured_spans
    split_sentences("Hello. World.", max_len=128)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "fancychunk.split_sentences" in names
    span = next(s for s in spans if s.name == "fancychunk.split_sentences")
    attrs = _attrs(span)
    assert attrs["fancychunk.document.length"] == 13
    assert attrs["fancychunk.min_len"] == 4
    assert attrs["fancychunk.max_len"] == 128
    assert attrs["fancychunk.sentences.count"] >= 1
    assert "fancychunk.segmenter" in attrs


def test_split_sentences_short_circuits_recorded(captured_spans) -> None:
    exporter, provider = captured_spans
    split_sentences("")
    split_sentences("   \n   ")
    split_sentences("ab")
    provider.force_flush()
    short_circuits = [
        _attrs(s).get("fancychunk.short_circuit")
        for s in exporter.get_finished_spans()
        if s.name == "fancychunk.split_sentences"
    ]
    assert "empty" in short_circuits
    assert "whitespace_only" in short_circuits
    assert "below_min_len" in short_circuits


def test_split_chunklets_emits_span(captured_spans) -> None:
    exporter, provider = captured_spans
    split_chunklets(["a sentence. ", "another. "])
    provider.force_flush()
    span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "fancychunk.split_chunklets"
    )
    attrs = _attrs(span)
    assert attrs["fancychunk.sentences.count"] == 2
    assert attrs["fancychunk.max_size"] == 2048
    assert attrs["fancychunk.custom_costs"] is False
    assert "fancychunk.chunklets.count" in attrs


class _FixedEmbedder:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    async def embed_chunklets(self, chunklets: list[str]) -> np.ndarray:
        return self.matrix


def test_split_chunks_emits_span_short_circuit(captured_spans) -> None:
    """SPEC-CHUNK-340 — single-chunklet input short-circuits before
    invoking the embedder. The span records the short-circuit branch
    but does NOT carry an embedding.dim attribute (the embedder was
    never called)."""
    from fancychunk.embedders import noop

    exporter, provider = captured_spans
    chunks = split_chunks(["a chunklet."], noop())
    assert [c.text for c in chunks] == ["a chunklet."]
    provider.force_flush()
    span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "fancychunk.split_chunks"
    )
    attrs = _attrs(span)
    assert attrs["fancychunk.chunklets.count"] == 1
    assert attrs["fancychunk.chunks.count"] == 1
    assert attrs["fancychunk.short_circuit"] == "single_chunklet"
    assert "fancychunk.embedding.dim" not in attrs


def test_split_chunks_emits_span_multi_chunk(captured_spans) -> None:
    """Multi-chunk path actually invokes the embedder, so the span
    carries the embedding.dim attribute."""
    exporter, provider = captured_spans
    # Three 1000-char chunklets > max_size=2048 → forced split,
    # embedder gets called.
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    chunks = split_chunks(
        ["a" * 1000, "b" * 1000, "c" * 1000],
        _FixedEmbedder(matrix),
        max_size=2048,
    )
    assert len(chunks) >= 2
    provider.force_flush()
    span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "fancychunk.split_chunks"
    )
    attrs = _attrs(span)
    assert attrs["fancychunk.chunklets.count"] == 3
    assert attrs["fancychunk.embedding.dim"] == 2
    # Multi-chunk path does NOT set short_circuit.
    assert "fancychunk.short_circuit" not in attrs


def test_embed_with_late_chunking_emits_span(captured_spans) -> None:
    exporter, provider = captured_spans
    fake = FakeEmbedder(dim=8, n_ctx=512)
    embed_with_late_chunking(["first.", "second."], fake, include_headings=False)
    provider.force_flush()
    span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "fancychunk.embed_with_late_chunking"
    )
    attrs = _attrs(span)
    assert attrs["fancychunk.chunks.count"] == 2
    assert attrs["fancychunk.embedder"] == "FakeEmbedder"
    assert attrs["fancychunk.embedding.dim"] == 8
    assert attrs["fancychunk.segments.count"] >= 1
    assert attrs["fancychunk.normalize"] is True
    assert attrs["fancychunk.include_headings"] is False


def test_heading_paths_emits_span(captured_spans) -> None:
    exporter, provider = captured_spans
    heading_paths_via_wrap(["# Title\n", "Body.\n"])
    provider.force_flush()
    span = next(
        s for s in exporter.get_finished_spans() if s.name == "fancychunk.heading_paths"
    )
    attrs = _attrs(span)
    assert attrs["fancychunk.chunks.count"] == 2
    assert attrs["fancychunk.paths.non_empty"] == 1
