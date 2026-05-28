"""Stage 3 tests — semantic chunking."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray

from fancychunk import (
    Chunk,
    OversizedChunkletError,
    ZeroNormEmbeddingError,
    split_chunks,
)
from fancychunk.embedders import noop


def _texts(chunks: list[Chunk]) -> list[str]:
    """Extract the ``.text`` of each chunk — most assertions in this
    file compare against raw strings, so this is a readability hack."""
    return [c.text for c in chunks]


class _FixedEmbedder:
    """Test-only ChunkletEmbedder that returns a pre-baked matrix.

    Lets tests pin a specific embedding pattern (heading boundaries,
    discourse-vector edge cases, identical-vector short-circuits)
    without depending on a real model.
    """

    def __init__(self, matrix: NDArray[np.floating]) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)

    async def embed_chunklets(
        self, chunklets: list[str]
    ) -> NDArray[np.float64]:
        assert len(chunklets) == self.matrix.shape[0], (
            f"_FixedEmbedder configured for {self.matrix.shape[0]} chunklets, "
            f"got {len(chunklets)}"
        )
        return self.matrix


class _RaisingEmbedder:
    """Test-only ChunkletEmbedder that raises if invoked.

    Used to assert the short-circuit paths in SPEC-CHUNK-340 don't
    call the embedder at all.
    """

    async def embed_chunklets(self, chunklets: list[str]) -> NDArray[np.float64]:
        raise AssertionError(
            "embedder.embed_chunklets must not be called on the short-circuit path"
        )


# SPEC-CHUNK-300 — round-trip across the partition.
def test_partition_preserves_concatenation() -> None:
    chunklets = ["a" * 1000, "b" * 1000, "c" * 1000, "d" * 1000]
    matrix = np.eye(4)
    chunks = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix)))
    assert "".join(_texts(chunks)) == "".join(chunklets)


# SPEC-CHUNK-301, -311 — hard size constraint.
def test_size_constraint_forces_split() -> None:
    chunklets = ["a" * 1000, "b" * 1000, "c" * 1000]
    matrix = np.tile([[1.0, 0.0]], (3, 1))
    chunks = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix)))
    for c in chunks:
        assert len(c.text) <= 2048
    assert "".join(_texts(chunks)) == "".join(chunklets)


# SPEC-CHUNK-340 — empty input. No embedder call.
def test_empty_input_skips_embedder() -> None:
    chunks = asyncio.run(split_chunks([], _RaisingEmbedder()))
    assert chunks == []


# SPEC-CHUNK-340 — single chunklet. No embedder call.
def test_single_chunklet_skips_embedder() -> None:
    chunks = asyncio.run(split_chunks(["Single chunklet content."], _RaisingEmbedder()))
    assert _texts(chunks) == ["Single chunklet content."]


# SPEC-CHUNK-340 — total fits in max_size. No embedder call.
def test_total_fits_skips_embedder() -> None:
    chunks = asyncio.run(split_chunks(
        ["one ", "two ", "three"], _RaisingEmbedder()
    ))
    assert _texts(chunks) == ["one two three"]


# SPEC-CHUNK-340 — short-circuits hold for any embedder argument
# (the embedder is never invoked on these paths). We use the noop
# embedder here since it's the cheapest valid choice.
def test_short_circuit_with_noop_embedder() -> None:
    assert asyncio.run(split_chunks([], noop())) == []
    assert _texts(asyncio.run(split_chunks(["only one"], noop()))) == ["only one"]
    assert _texts(asyncio.run(split_chunks(["one ", "two ", "three"], noop()))) == [
        "one two three"
    ]


# Identical embeddings, total fits: short-circuit before similarity.
def test_identical_short_circuit() -> None:
    chunklets = ["x" * 100] * 10
    chunks = asyncio.run(split_chunks(chunklets, _RaisingEmbedder()))
    assert _texts(chunks) == ["x" * 1000]


# SPEC-CHUNK-322 — no split immediately after a heading.
def test_no_split_after_heading() -> None:
    # Use a pure heading chunklet plus body chunklets that together
    # exceed max_size so the short-circuit does not apply and the
    # heading-aware modification runs.
    chunklets = ["# Heading\n\n", "x" * 800, "y" * 800, "z" * 800]
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
    )
    chunks = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048))
    # The heading must not be its own standalone chunk.
    assert chunks[0].text != "# Heading\n\n"


# SPEC-CHUNK-322 — encourage split before heading.
def test_split_before_heading() -> None:
    chunklets = ["a" * 900, "b" * 900, "## Subhead\n\n", "c" * 900]
    matrix = np.array(
        [
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
        ]
    )
    chunks = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048))
    assert len(chunks) == 2
    assert chunks[1].text.startswith("## Subhead")


# SPEC-CHUNK-342 — zero-norm embedding rejected.
def test_zero_norm_rejected() -> None:
    # Need 3+ chunklets so we hit the multi-chunk path where the
    # embedder is actually invoked (short-circuits skip it now).
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    emb = _FixedEmbedder(matrix)
    with pytest.raises(ZeroNormEmbeddingError):
        asyncio.run(split_chunks(["a" * 1000, "b" * 1000, "c" * 1000], emb, max_size=2048))


# SPEC-CHUNK-341 — oversized chunklet rejected.
def test_oversized_chunklet_rejected() -> None:
    emb = _FixedEmbedder(np.eye(2))
    with pytest.raises(OversizedChunkletError):
        asyncio.run(split_chunks(["a" * 3000, "b"], emb, max_size=2048))


# SPEC-CHUNK-321 — discourse-vector fallback when projection would zero rows.
def test_discourse_fallback() -> None:
    matrix = np.tile([[1.0, 0.0]], (5, 1))
    chunks = asyncio.run(split_chunks(
        ["x" * 1000] * 5, _FixedEmbedder(matrix), max_size=2048
    ))
    for c in chunks:
        assert len(c.text) <= 2048
    assert "".join(_texts(chunks)) == "".join(["x" * 1000] * 5)


# SPEC-CHUNK-330 — determinism.
def test_determinism() -> None:
    chunklets = ["a" * 900, "b" * 900, "## Heading\n", "c" * 900]
    matrix = np.array([[1.0, 0.1], [1.0, 0.2], [1.0, 0.05], [1.0, 0.1]])
    out_a = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048))
    out_b = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048))
    assert out_a == out_b


# Structural-only path: pass embedders.noop() instead of a real model.
# Equivalent to the legacy "no embeddings supplied" behavior — the
# similarity term is uniform across partition points, so only the
# heading-aware modification shapes where splits land.
def test_split_chunks_with_noop_embedder() -> None:
    chunklets = ["a" * 1000, "b" * 1000, "c" * 1000]
    chunks = asyncio.run(split_chunks(chunklets, noop(), max_size=2048))
    assert "".join(_texts(chunks)) == "".join(chunklets)
    assert all(len(c.text) <= 2048 for c in chunks)


def test_split_chunks_with_noop_prefers_heading_split() -> None:
    # 4 chunklets totalling > max_size, forcing one or more splits.
    # The heading chunklet at index 2 makes the partition point at
    # position 1 (between non-heading-1 and heading) cheapest. The DP
    # should prefer that split over splitting between two body
    # chunklets.
    chunklets = ["a" * 900, "b" * 900, "## Subhead\n\n", "c" * 900]
    chunks = asyncio.run(split_chunks(chunklets, noop(), max_size=2048))
    assert len(chunks) == 2
    assert chunks[1].text.startswith("## Subhead")


# SPEC-CHUNK-322 — heading detection accepts ATX and Setext forms,
# rejects heading-plus-body and non-heading content.
@pytest.mark.parametrize(
    "chunklet,expected",
    [
        ("# ATX H1\n", True),
        ("## ATX H2\n\n", True),
        ("###### ATX H6\n", True),
        ("Heading text\n============\n\n", True),  # Setext H1
        ("Heading text\n============", True),  # Setext H1, no trailing newline
        ("H2 form\n--------\n", True),  # Setext H2
        ("Title 1\nTitle 2\n========", True),  # multi-line Setext text
        ("Heading\n=====\n\nBody.\n", False),  # heading + body
        ("Just a paragraph.\n", False),
        ("Text\n# Heading\n", False),  # body before heading
        ("", False),
        ("####### Seven hashes\n", False),  # > 6 hashes is not a heading
    ],
)
def test_is_heading_atx_and_setext(chunklet: str, expected: bool) -> None:
    from fancychunk.chunks import _is_heading

    assert _is_heading(chunklet) is expected


# SPEC-CHUNK-322 — heading-aware modification fires for Setext exactly
# as it does for ATX. Same shape as the ATX test above: a Setext
# heading buried between large body chunklets pulls the cheap split
# to its left edge and forbids the split to its right.
def test_setext_heading_pulls_split_before() -> None:
    chunklets = [
        "a" * 900,
        "b" * 900,
        "Subhead\n=======\n\n",
        "c" * 900,
    ]
    matrix = np.array(
        [
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
        ]
    )
    chunks = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048))
    assert len(chunks) == 2
    assert chunks[1].text.startswith("Subhead\n=======")


# ---------------------------------------------------------------------------
# SPEC-CHUNK-323 — small-chunk badness (front-matter + general).
# ---------------------------------------------------------------------------


def _is_heading_only_text(text: str) -> bool:
    """A chunk whose every block-level element is a Markdown heading."""
    from markdown_it import MarkdownIt

    opens = [t.type for t in MarkdownIt("commonmark").parse(text) if t.type.endswith("_open")]
    return bool(opens) and all(t == "heading_open" for t in opens)


# SPEC-CHUNK-323 — a leading front-matter chunk (title + abstract, with
# no body section) is bundled forward into the first body section
# rather than emitted standalone. This is the deterministic, model-free
# analogue of the qasper "1903.09588" repro: the bare similarity DP
# isolates the front-matter chunklet, the small-chunk badness term in
# the objective makes the DP bundle it forward instead.
def test_front_matter_chunk_is_bundled_forward() -> None:
    from fancychunk.chunks import _dp_cuts, _partition_similarities

    front = "# Title\n\n## Abstract\n\n" + "a" * 579  # title + abstract head + body
    intro = "## Introduction\n\n" + "b" * 583
    body1 = "c" * 600
    body2 = "d" * 600
    chunklets = [front, intro, body1, body2]
    lengths = [len(c) for c in chunklets]

    # Embeddings: the front matter is dissimilar from the body, which is
    # internally uniform — so the cheapest single split lands right
    # after the front matter, isolating it.
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # With the front-matter badness disabled (front_matter_eligible=False),
    # the bare similarity DP isolates the front matter as chunk 0.
    cum = np.concatenate(([0], np.cumsum(lengths)))
    sim = _partition_similarities(matrix, chunklets, lengths)
    bare = _dp_cuts(
        cum,
        sim,
        2048,
        len(chunklets),
        front_matter_eligible=False,
    )
    assert bare[:2] == [0, 1]

    # split_chunks (with the badness term) bundles it forward.
    chunks = asyncio.run(
        split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048)
    )
    assert "".join(_texts(chunks)) == "".join(chunklets)
    assert all(len(c.text) <= 2048 for c in chunks)
    # The first chunk is no longer front-matter-only: it carries the
    # first body section with it.
    assert "## Introduction" in chunks[0].text
    # And no chunk is solely title/heading + abstract with no body.
    assert chunks[0].text != front


# SPEC-CHUNK-323 — a genuinely *tiny* body chunk is merged forward even
# when it is a distinct topic (a ~20-char fragment is a poor retrieval
# unit), but a moderately-short distinct section is kept. The cutoff is
# not a fixed size: it scales with how distinct the neighbours are.
def test_tiny_body_chunk_merges_but_short_section_kept() -> None:
    # Three chunklets where isolating the lead chunk ([c0][c1+c2]) and
    # merging it ([c0+c1][c2]) are *both* feasible single cuts, so the
    # small-chunk badness — not cut-minimization — decides. The lead
    # chunk is a distinct topic; c1 and c2 are the same topic.
    body1 = "b" * 700
    body2 = "b" * 700
    max_size = 1410  # general badness ceiling = 0.2 * 1410 ≈ 282 chars
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    # A 20-char distinct lead chunk is below the ceiling -> merged
    # forward even though it is a distinct topic.
    tiny = ["t" * 20, body1, body2]
    tiny_chunks = asyncio.run(
        split_chunks(tiny, _FixedEmbedder(matrix), max_size=max_size)
    )
    assert "".join(_texts(tiny_chunks)) == "".join(tiny)
    assert tiny_chunks[0].text != "t" * 20  # not left standalone

    # A 400-char distinct lead chunk is above the ceiling -> kept as its
    # own chunk (the semantic split is honoured).
    short = ["s" * 400, body1, body2]
    short_chunks = asyncio.run(
        split_chunks(short, _FixedEmbedder(matrix), max_size=max_size)
    )
    assert "".join(_texts(short_chunks)) == "".join(short)
    assert short_chunks[0].text == "s" * 400  # distinct short section kept


# SPEC-CHUNK-322/-323 — the chunker does not emit a chunk that is only a
# heading/title. This is guaranteed by SPEC-CHUNK-322 (the split-after-
# heading cost is pinned to the maximum, so a heading is never
# voluntarily isolated) with the general small-chunk term as backstop;
# there is no dedicated heading-only rule.
def test_no_chunk_is_heading_only() -> None:
    chunklets = ["x" * 1000, "## Lonely Heading\n\n", "y" * 1000, "z" * 1000]
    matrix = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    )
    chunks = asyncio.run(
        split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048)
    )
    assert "".join(_texts(chunks)) == "".join(chunklets)
    for c in chunks:
        assert not _is_heading_only_text(c.text), (
            f"emitted a heading-only chunk: {c.text!r}"
        )


# SPEC-CHUNK-322 — a trailing heading rides along with the preceding
# body rather than being split off into its own chunk.
def test_trailing_heading_only_chunk_merges_backward() -> None:
    chunklets = ["x" * 1000, "y" * 1000, "z" * 1000, "## Trailing Head\n"]
    matrix = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    chunks = asyncio.run(
        split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048)
    )
    assert "".join(_texts(chunks)) == "".join(chunklets)
    for c in chunks:
        assert not _is_heading_only_text(c.text)
    # The trailing heading rode along with preceding body.
    assert chunks[-1].text.endswith("## Trailing Head\n")
    assert len(chunks[-1].text) > len("## Trailing Head\n")


# ---------------------------------------------------------------------------
# SPEC-CHUNK-324 — paragraph-boundary preference.
# ---------------------------------------------------------------------------


# SPEC-CHUNK-324 — the penalty mechanism: a mid-paragraph partition point
# is penalized; a paragraph (blank-line) break is not.
def test_paragraph_preference_mechanism() -> None:
    from fancychunk.chunks import _apply_paragraph_preference_inplace, _ends_block

    assert _ends_block("text.\n\n") and not _ends_block("text. ")
    chunklets = ["a. ", "b.\n\n", "c. ", "d"]  # mid, block, mid (3 points)
    sim = np.array([0.5, 0.5, 0.5])
    _apply_paragraph_preference_inplace(sim, chunklets)
    assert sim[0] == 0.75  # mid-paragraph -> +0.25
    assert sim[1] == 0.5  # paragraph break -> unchanged
    assert sim[2] == 0.75  # mid-paragraph -> +0.25


# SPEC-CHUNK-324 — with a uniform (noop) similarity signal, the forced
# cut lands at the paragraph break rather than the mid-paragraph point.
def test_paragraph_preference_routes_cut_to_block_break() -> None:
    c0 = "x" * 900  # ends mid-paragraph
    c1 = "y" * 898 + "\n\n"  # ends a paragraph (blank line)
    c2 = "z" * 900
    chunks = asyncio.run(split_chunks([c0, c1, c2], noop(), max_size=2048))
    assert "".join(_texts(chunks)) == c0 + c1 + c2
    assert len(chunks) == 2
    # The cut is after the paragraph break (c0+c1), not mid-paragraph (c0).
    assert chunks[0].text == c0 + c1


# ---------------------------------------------------------------------------
# Chunk metadata — start/end character offsets.
# ---------------------------------------------------------------------------


def test_chunk_offsets_index_into_joined_chunklets() -> None:
    """For every chunk produced from chunklets, ``joined[start:end] == text``
    where joined is ``"".join(chunklets)``."""
    chunklets = ["a" * 1000, "b" * 1000, "c" * 1000, "d" * 1000]
    matrix = np.eye(4)
    chunks = asyncio.run(split_chunks(chunklets, _FixedEmbedder(matrix)))
    joined = "".join(chunklets)
    for c in chunks:
        assert c.start is not None and c.end is not None
        assert joined[c.start : c.end] == c.text


def test_chunk_offsets_short_circuit_single_chunklet() -> None:
    chunks = asyncio.run(
        split_chunks(["Lone chunklet."], _RaisingEmbedder())
    )
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == len("Lone chunklet.")


def test_chunk_offsets_short_circuit_total_fits() -> None:
    chunklets = ["abc", "def", "ghi"]
    chunks = asyncio.run(split_chunks(chunklets, _RaisingEmbedder()))
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == sum(len(c) for c in chunklets)


def test_chunk_offsets_are_contiguous_and_cover_input() -> None:
    """Adjacent chunks meet at a shared offset; first starts at 0;
    last ends at total length."""
    chunklets = ["abc" * 200, "def" * 200, "ghi" * 200, "jkl" * 200]
    matrix = np.eye(4)
    chunks = asyncio.run(
        split_chunks(chunklets, _FixedEmbedder(matrix), max_size=1500)
    )
    assert chunks[0].start == 0
    assert chunks[-1].end == sum(len(c) for c in chunklets)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end == nxt.start


def test_chunk_str_returns_text() -> None:
    """``str(chunk)`` returns chunk.text — usability check."""
    c = Chunk(text="hello", start=0, end=5)
    assert str(c) == "hello"


def test_chunk_is_hashable() -> None:
    """frozen dataclass means we can put Chunks in sets / dict keys."""
    a = Chunk(text="hi", start=0, end=2)
    b = Chunk(text="hi", start=0, end=2)
    c = Chunk(text="bye", start=0, end=3)
    assert {a, b, c} == {a, c}


def test_split_chunks_populates_heading_path() -> None:
    """Chunks from split_chunks carry ``heading_path`` populated —
    tuple of full markdown heading lines in scope at the chunk's start."""
    chunklets = [
        "# Top\n\n",
        "a" * 900,
        "## Sub\n\n",
        "b" * 900,
        "c" * 900,
    ]
    matrix = np.eye(5)
    chunks = asyncio.run(
        split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048)
    )
    # All chunks have heading_path populated (not None).
    assert all(c.heading_path is not None for c in chunks)
    # First chunk: nothing in scope before it.
    assert chunks[0].heading_path == ()
    # Some later chunk starts under "# Top" or "# Top, ## Sub" depending on
    # where the splits land. Just verify a non-empty path appears at some
    # point and that '#' markers are preserved.
    has_top = any("# Top" in p for c in chunks for p in (c.heading_path or ()))
    assert has_top


def test_split_chunks_heading_path_short_circuit() -> None:
    """Short-circuit paths (single chunklet, total fits) still
    populate heading_path with the empty tuple (no heading before chunk 0)."""
    single = asyncio.run(split_chunks(["# Top\n\nBody.\n"], _RaisingEmbedder()))
    assert single[0].heading_path == ()

    fits = asyncio.run(
        split_chunks(["# Top\n", "Body.\n"], _RaisingEmbedder(), max_size=1000)
    )
    assert fits[0].heading_path == ()


def test_split_chunks_heading_path_preserves_level_via_markers() -> None:
    """Skipped levels (H1 then H3) are visible in the tuple via the
    '#' marker count — ``("# H1", "### H3")`` not the misleading
    ``("H1", "H3")``. We size the body so the H3 heading lands in
    one chunk and additional H3-scoped body lands in a later chunk,
    putting both H1 and H3 in scope at that later chunk's start."""
    chunklets = [
        "# H1\n\n",
        "a" * 900,
        "### H3\n\n",
        "b" * 900,
        "c" * 900,
        "d" * 900,
        "e" * 900,
    ]
    matrix = np.eye(7)
    chunks = asyncio.run(
        split_chunks(chunklets, _FixedEmbedder(matrix), max_size=2048)
    )
    # Find a chunk whose path includes both H1 and H3.
    deep_paths = [c.heading_path for c in chunks if c.heading_path and len(c.heading_path) == 2]
    assert deep_paths, "expected at least one chunk under both H1 and H3"
    p = deep_paths[0]
    assert p[0] == "# H1"
    assert p[1] == "### H3"
