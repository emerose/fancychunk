"""Stage 5 tests — contextual chunk headings."""

from __future__ import annotations

from fancychunk import Chunk, enrich_with_headings, heading_paths


def _wrap(texts: list[str]) -> list[Chunk]:
    """Wrap raw strings as Chunks for the typed API. Tests in this
    file are about heading behavior, not chunk metadata, so we don't
    bother computing offsets."""
    return [Chunk(text=t) for t in texts]


# TV-501 / SPEC-CHUNK-540 — empty input.
def test_tv_501_empty_input() -> None:
    assert heading_paths([]) == []


# TV-502 / SPEC-CHUNK-541 — document without headings.
def test_tv_502_no_headings() -> None:
    chunks = _wrap([
        "First paragraph.\n\n",
        "Second paragraph.\n",
        "Third paragraph.\n",
    ])
    assert heading_paths(chunks) == [(), (), ()]


# TV-503 / SPEC-CHUNK-510, -511 — simple linear heading structure.
def test_tv_503_linear_heading_structure() -> None:
    chunks = _wrap([
        "# Introduction\n\nOpening text.\n\n",
        "## Background\n\nMore detail.\n\n",
        "Continuing background.\n",
        "## Method\n\nDescription.\n",
    ])
    assert heading_paths(chunks) == [
        (),
        ("# Introduction",),
        ("# Introduction", "## Background"),
        ("# Introduction", "## Background"),
    ]


# TV-504 / SPEC-CHUNK-520 — stack reset when heading level rises.
def test_tv_504_stack_reset() -> None:
    chunks = _wrap([
        "# A\n\n## A.1\n\n### A.1.x\n\nContent.\n",
        "Next chunk content.\n",
        "# B\n\nB content.\n",
        "More B content.\n",
    ])
    assert heading_paths(chunks) == [
        (),
        ("# A", "## A.1", "### A.1.x"),
        ("# A", "## A.1", "### A.1.x"),
        ("# B",),
    ]


# TV-505 / SPEC-CHUNK-502, -542 — first chunk introduces the first heading.
def test_tv_505_first_chunk_starts_with_heading() -> None:
    chunks = _wrap(["# Title\n\nBody text.\n\n", "More body text.\n"])
    assert heading_paths(chunks) == [(), ("# Title",)]


# TV-506 / SPEC-CHUNK-543 — heading levels skipped. Tuple form preserves
# level info via the ``#`` marker count, so ``("# H1", "### H3")`` keeps
# the gap visible.
def test_tv_506_levels_skipped() -> None:
    chunks = _wrap(["# H1\n\n### H3\n\nContent under H3.\n", "More content.\n"])
    assert heading_paths(chunks) == [(), ("# H1", "### H3")]


# TV-507 / SPEC-CHUNK-544 — seven or more '#' is not a heading.
def test_tv_507_seven_hashes_not_a_heading() -> None:
    chunks = _wrap(["####### Not a heading\n\nBody text.\n", "More text.\n"])
    assert heading_paths(chunks) == [(), ()]


# TV-508 — heading text preserves inline markdown (bold, code, etc.)
# but is stripped of trailing whitespace and newlines.
def test_tv_508_preserves_inline_markdown() -> None:
    chunks = _wrap([
        "# Title with **bold** and `code`  \n\nBody.\n",
        "More body.\n",
    ])
    assert heading_paths(chunks) == [
        (),
        ("# Title with **bold** and `code`",),  # trailing 2 spaces stripped
    ]


# TV-509 — render_heading_path joins entries with newlines for a
# Markdown preamble suitable for the embedder.
def test_tv_509_render_heading_path() -> None:
    from fancychunk.headings import render_heading_path

    assert render_heading_path(()) == ""
    assert render_heading_path(("# Top",)) == "# Top\n"
    assert render_heading_path(("# Top", "## Sub")) == "# Top\n## Sub\n"


# SPEC-CHUNK-530 — determinism.
def test_determinism() -> None:
    chunks = _wrap(["# A\n\nText.\n", "More.\n"])
    assert heading_paths(chunks) == heading_paths(chunks)


# enrich_with_headings — convenience that prepends the heading path
# rendered as Markdown onto each chunk's text in one call. Chunks
# whose path is empty pass through unchanged; chunks with a path get
# the rendered path prepended on .text. Metadata (start/end/heading_path)
# is preserved.
def test_enrich_with_headings_prepends_path() -> None:
    chunks = _wrap([
        "# Intro\n\nOpening text.\n",
        "Continuation.\n",
        "## Methods\n\nDetails.\n",
    ])
    enriched = enrich_with_headings(chunks)
    assert len(enriched) == 3
    # First chunk: heading is inside the chunk itself; path at chunk
    # start is empty, so the chunk is returned unchanged.
    assert enriched[0] == chunks[0]
    # Second chunk: lives under "# Intro", path is non-empty.
    assert enriched[1].text.startswith("# Intro")
    assert enriched[1].text.endswith(chunks[1].text)
    # Third chunk: same — path at its start is ("# Intro",).
    assert enriched[2].text.startswith("# Intro")
    assert enriched[2].text.endswith(chunks[2].text)


def test_enrich_with_headings_empty_input() -> None:
    assert enrich_with_headings([]) == []


def test_enrich_with_headings_no_headings_passthrough() -> None:
    chunks = _wrap(["First.\n", "Second.\n", "Third.\n"])
    assert enrich_with_headings(chunks) == chunks


def test_enrich_with_headings_preserves_metadata() -> None:
    """Metadata (start/end/heading_path) is preserved from input through
    enrichment, even though len(.text) no longer equals end-start after
    prepending."""
    chunks = [
        Chunk(text="# Intro\n\nBody.\n", start=0, end=16, heading_path=()),
        Chunk(
            text="More body.\n",
            start=16,
            end=27,
            heading_path=("# Intro",),
        ),
    ]
    enriched = enrich_with_headings(chunks)
    # First chunk: empty path, passes through with metadata intact.
    assert enriched[0].start == 0 and enriched[0].end == 16
    assert enriched[0].heading_path == ()
    # Second chunk: text is enriched, but metadata still points at the
    # original source range and heading_path is unchanged.
    assert enriched[1].start == 16 and enriched[1].end == 27
    assert enriched[1].heading_path == ("# Intro",)
    assert enriched[1].text.endswith("More body.\n")


def test_enrich_with_headings_uses_populated_field_when_present() -> None:
    """When ``chunk.heading_path`` is already populated (typical for
    chunks from ``split_chunks`` / ``chunk_document``), ``enrich_with_headings``
    uses it directly without re-scanning."""
    chunks = [
        # Hand-set heading_path that doesn't match what a scan would
        # produce — to prove the populated field wins.
        Chunk(text="Body.\n", heading_path=("# Custom",)),
    ]
    enriched = enrich_with_headings(chunks)
    assert enriched[0].text == "# Custom\nBody.\n"


def test_enrich_with_headings_falls_back_to_scan_when_unpopulated() -> None:
    """For hand-constructed chunks without heading_path, fall back to
    a fresh scan via heading_paths()."""
    chunks = _wrap(["# Intro\n\nText.\n", "Body.\n"])
    # No heading_path populated on input; enrich_with_headings should
    # compute one via heading_paths.
    enriched = enrich_with_headings(chunks)
    assert enriched[0] == chunks[0]
    assert enriched[1].text == "# Intro\nBody.\n"
