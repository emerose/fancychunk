"""Tests for the experimental structure-first spike.

These cover the invariants the spike must not regress: covering,
round-trip, fitting sections emitted with no model call, oversized
sections falling back to the semantic split, and bare/front-matter
headings never stranded. The fallback path uses ``punctuation_segmenter``
so no SaT model loads in CI.
"""

from __future__ import annotations

import asyncio
import re

from fancychunk import punctuation_segmenter
from fancychunk.embedders import noop
from fancychunk.structure_first import (
    _heading_level,
    plan_units,
    split_chunks_structure_first,
)


class _RaisingEmbedder:
    """Fails loudly if the slow embedder is ever invoked."""

    async def embed_chunklets(self, chunklets: list[str]):
        raise AssertionError("embedder must not be called for fitting sections")


def _run(doc: str, embedder, **kw):
    return asyncio.run(
        split_chunks_structure_first(doc, embedder, **kw)
    )


def test_heading_level_unifies_hashes_and_colon_path() -> None:
    assert _heading_level("# Title") == 1
    assert _heading_level("## Abstract") == 2
    assert _heading_level("### Sub") == 3
    # Qasper-style flat ## with a ::: hierarchy path.
    assert _heading_level("## Method ::: Sub") == 3
    assert _heading_level("## Method ::: Sub ::: Step") == 4
    assert _heading_level("plain prose line") == 0


def test_roundtrip_and_covering() -> None:
    doc = (
        "# Title\n\n## Abstract\n\nShort abstract.\n\n"
        "## Section\n\n" + ("word " * 600) + "\n\n"
        "## Tail\n\nTiny tail.\n"
    )
    chunks = _run(doc, noop(), max_size=2048, segmenter=punctuation_segmenter)
    assert "".join(c.text for c in chunks) == doc
    assert all(len(c.text) <= 2048 for c in chunks)


def test_fitting_sections_skip_the_embedder() -> None:
    # Every section comfortably fits, so no fallback → no model call.
    doc = (
        "# Title\n\n## Abstract\n\nShort abstract body.\n\n"
        "## One\n\nFirst section body.\n\n"
        "## Two\n\nSecond section body.\n"
    )
    chunks = _run(doc, _RaisingEmbedder(), max_size=2048)
    assert "".join(c.text for c in chunks) == doc
    # Each chunk after the merged front-matter starts at a heading.
    for c in chunks:
        assert c.text.lstrip().startswith("#")


def test_oversized_section_falls_back() -> None:
    big = "Sentence number one. " * 300  # well over 2048 chars
    doc = f"# Title\n\n## Big\n\n{big}\n"
    chunks = _run(doc, noop(), max_size=2048, segmenter=punctuation_segmenter)
    assert "".join(c.text for c in chunks) == doc
    assert all(len(c.text) <= 2048 for c in chunks)
    # The oversized section must have been split into more than one chunk.
    assert len(chunks) >= 2


def test_bare_front_matter_heading_merged_forward() -> None:
    # "# Title" has no body of its own before "## Abstract" → must not
    # be stranded as its own chunk.
    doc = "# Title\n\n## Abstract\n\nThe abstract text.\n\n## Body\n\nMore.\n"
    chunks = _run(doc, _RaisingEmbedder(), max_size=2048)
    heading_re = re.compile(r"#{1,6}(\s|$)")
    for c in chunks:
        nonblank = [ln for ln in c.text.splitlines() if ln.strip()]
        # No chunk is only heading line(s) with no body content.
        only_headings = all(heading_re.match(ln) for ln in nonblank)
        assert not only_headings, f"stranded heading chunk: {c.text!r}"
    # The first chunk carries the title glued to the abstract.
    assert chunks[0].text.startswith("# Title")
    assert "abstract text" in chunks[0].text.lower()


def test_no_headings_is_single_fallback_unit() -> None:
    doc = "Just some prose with no headings at all. Another sentence."
    units = plan_units(doc, max_size=2048)
    assert len(units) == 1
    # Fits, so even the no-heading doc needs no model here.
    assert units[0].needs_model is False
    chunks = _run(doc, _RaisingEmbedder(), max_size=2048)
    assert "".join(c.text for c in chunks) == doc


def test_empty_document() -> None:
    assert _run("", _RaisingEmbedder()) == []
    assert plan_units("", max_size=2048) == []
