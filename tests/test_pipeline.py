"""Cross-stage integration tests — SPEC-CHUNK-9xx invariants."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fancychunk import (
    heading_paths,
    split_chunklets,
    split_chunks,
    split_sentences,
)
from fancychunk.embedders import noop


class _PreCookedEmbedder:
    """Wraps a precomputed matrix to satisfy ChunkletEmbedder."""

    def __init__(self, matrix: NDArray[np.floating]) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def embed_chunklets(
        self, chunklets: list[str]
    ) -> NDArray[np.float64]:
        assert len(chunklets) == self.matrix.shape[0]
        return self.matrix


# SPEC-CHUNK-900 — concatenation round-trip through all three stages.
def test_concatenation_round_trip_all_stages() -> None:
    document = (
        "# Introduction\n\n"
        "First paragraph here, with two sentences. Second sentence.\n\n"
        "## Methods\n\n"
        "Body of methods section spans several sentences. "
        "Another one. And a third.\n\n"
        "## Results\n\n"
        "Findings as a paragraph. Details follow.\n"
    )
    sentences = split_sentences(document, max_len=2048)
    assert "".join(sentences) == document

    chunklets = split_chunklets(sentences, max_size=2048)
    assert "".join(chunklets) == "".join(sentences)

    # Pre-compute synthetic embeddings: deterministic, nonzero norm.
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(len(chunklets), 8))
    # Make sure no row has near-zero norm.
    emb = emb + 0.01 * np.sign(emb)
    chunks, chunk_embeddings = split_chunks(
        chunklets, _PreCookedEmbedder(emb), max_size=2048
    )
    assert "".join(chunks) == "".join(chunklets)
    rows = np.vstack(chunk_embeddings)
    assert np.array_equal(rows, emb)


# SPEC-CHUNK-901 — determinism across runs.
def test_determinism_across_runs() -> None:
    document = "First sentence here. Second sentence. Third one.\n"
    a = split_sentences(document)
    b = split_sentences(document)
    assert a == b


# SPEC-CHUNK-902 — size limits are upper bounds.
def test_size_limits_are_upper_bounds() -> None:
    document = "a" * 500 + ". " + "b" * 500 + ". "
    sentences = split_sentences(document, max_len=400)
    for s in sentences:
        assert len(s) <= 400


# SPEC-CHUNK-903 — trivial-input short-circuits.
def test_trivial_input_short_circuits() -> None:
    # Stage 1.
    assert split_sentences("") == []
    assert split_sentences("ab") == ["ab"]
    # Stage 3 — empty input still short-circuits before any embedder call.
    chunks, emb = split_chunks([])
    assert chunks == [] and emb == []


def test_heading_paths_consistent_with_pipeline() -> None:
    document = (
        "# Top\n\nIntro.\n\n## Sub\n\nDetails. More details.\n\n# Other\n\nFinal.\n"
    )
    sentences = split_sentences(document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(chunklets), 8)) + 0.01
    chunks, _ = split_chunks(chunklets, _PreCookedEmbedder(emb), max_size=2048)
    paths = heading_paths(chunks)
    assert len(paths) == len(chunks)
    assert paths[0] == ""


# Pipeline composes cleanly through noop() for a no-model-download path.
def test_pipeline_with_noop_embedder() -> None:
    document = (
        "# Top\n\nFirst paragraph. Second sentence.\n\n"
        "## Sub\n\nMore body text here.\n"
    )
    sentences = split_sentences(document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    chunks, _ = split_chunks(chunklets, noop(), max_size=2048)
    assert "".join(chunks) == "".join(chunklets)
