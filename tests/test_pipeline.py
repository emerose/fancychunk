"""Cross-stage integration tests — SPEC-CHUNK-9xx invariants."""

from __future__ import annotations

import asyncio

import numpy as np
from numpy.typing import NDArray

from fancychunk import (
    heading_paths,
    split_chunklets,
    split_sentences,
)
from fancychunk import split_chunks as _async_split_chunks
from fancychunk.embedders import noop


# Sync shim — see test_late_chunking.py for rationale.
def split_chunks(*args, **kwargs):  # type: ignore[no-untyped-def]
    return asyncio.run(_async_split_chunks(*args, **kwargs))


class _PreCookedEmbedder:
    """Wraps a precomputed matrix to satisfy ChunkletEmbedder."""

    def __init__(self, matrix: NDArray[np.floating]) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)

    async def embed_chunklets(
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
    chunks = split_chunks(chunklets, _PreCookedEmbedder(emb), max_size=2048)
    assert "".join(c.text for c in chunks) == "".join(chunklets)


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
    # Stage 3 — empty input short-circuits; the embedder is required
    # for signature consistency but is not invoked on this path.
    assert split_chunks([], noop()) == []


def _para(tag: str, n: int) -> str:
    body = " ".join(
        f"{tag} sentence number {i} carries a fair amount of real textual content here."
        for i in range(n)
    )
    return body + "\n\n"


# Regression — a chunk's last non-whitespace line is never a Markdown
# heading (Issues 3 + 4). A one-line multi-sentence paragraph immediately
# followed by a heading is the structure that used to strand the heading
# at a chunk's tail: before the SPEC-CHUNK-240 block-opener guard, every
# interior sentence of a one-line paragraph inherited paragraph strength,
# SPEC-CHUNK-241 collapsed the document to a single surviving boundary,
# and the heading got swallowed into the preceding body chunklet's tail.
def test_heading_never_stranded_at_chunk_tail() -> None:
    from fancychunk.chunks import _is_heading

    document = (
        _para("Intro", 10)
        + "## Section A\n\n"
        + _para("A", 12)
        + "## Section B\n\n"
        + _para("B", 12)
        + "## Section C\n\n"
        + _para("C", 12)
    )
    sentences = split_sentences(document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=500)
    # The partition is driven by structural boundaries, not embedding
    # values, so the invariant must hold for every embedding seed.
    for seed in range(8):
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=(len(chunklets), 8)) + 0.05
        chunks = split_chunks(chunklets, _PreCookedEmbedder(emb), max_size=500)
        for c in chunks:
            last_line = c.text.rstrip().split("\n")[-1]
            assert not _is_heading(last_line + "\n"), (seed, repr(c.text[-50:]))
        # Issue 4 — no undersized stub chunk (< 40% of target).
        assert all(len(c.text) >= 0.4 * 500 for c in chunks), [
            len(c.text) for c in chunks
        ]


# Regression — the abstract is kept whole and the break lands at the
# ``## Introduction`` heading rather than mid-abstract-paragraph
# (Issue 2). Uses the deterministic no-embeddings structural path so the
# outcome depends only on size + heading-aware boundary preference.
def test_abstract_kept_whole_break_at_introduction() -> None:
    abstract = " ".join(
        f"Abstract sentence {i} states a finding clearly." for i in range(8)
    )
    intro = " ".join(
        f"Introduction sentence {i} gives background detail." for i in range(8)
    )
    document = f"## Abstract\n\n{abstract}\n\n## Introduction\n\n{intro}\n"
    sentences = split_sentences(document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=480)
    chunks = split_chunks(chunklets, noop(), max_size=480)
    texts = [c.text for c in chunks]
    # The whole abstract body lives in one chunk (no mid-paragraph cut).
    assert any(abstract in t for t in texts)
    # A chunk begins exactly at the Introduction heading.
    assert any(t.lstrip().startswith("## Introduction") for t in texts)


def test_heading_paths_consistent_with_pipeline() -> None:
    document = (
        "# Top\n\nIntro.\n\n## Sub\n\nDetails. More details.\n\n# Other\n\nFinal.\n"
    )
    sentences = split_sentences(document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(chunklets), 8)) + 0.01
    chunks = split_chunks(chunklets, _PreCookedEmbedder(emb), max_size=2048)
    paths = heading_paths(chunks)
    assert len(paths) == len(chunks)
    assert paths[0] == ()  # empty path (no heading in scope before chunk 0)


# Pipeline composes cleanly through noop() for a no-model-download path.
def test_pipeline_with_noop_embedder() -> None:
    document = (
        "# Top\n\nFirst paragraph. Second sentence.\n\n"
        "## Sub\n\nMore body text here.\n"
    )
    sentences = split_sentences(document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    chunks = split_chunks(chunklets, noop(), max_size=2048)
    assert "".join(c.text for c in chunks) == "".join(chunklets)
