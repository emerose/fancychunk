"""Tests for structure-first chunking.

These cover the invariants the partitioner must hold: covering,
round-trip, fitting sections emitted with no model call, oversized
sections falling back to the semantic split, bare/front-matter
headings never stranded, and the minimum-size merge. The fallback
path uses ``punctuation_segmenter`` so no SaT model loads in CI.
"""

from __future__ import annotations

import asyncio
import re

from fancychunk import punctuation_segmenter
from fancychunk.embedders import noop
from fancychunk.structure_first import (
    _fold_parent_intros,
    _heading_level,
    _merge_small_units,
    _Unit,
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


def _make_doc(sections: list[tuple[str, int]]) -> str:
    """Build a doc from (heading, body-word-count) pairs."""
    parts = []
    for heading, words in sections:
        body = ("word " * words).strip() if words else ""
        parts.append(f"{heading}\n\n{body}\n\n")
    return "".join(parts)


def test_small_units_merge_forward_to_clear_floor() -> None:
    # A thin section followed by another section merges forward; the
    # combined span stays a single unit and no thin chunk survives.
    doc = _make_doc(
        [("## Tiny", 5), ("## Next", 30), ("## After", 200)]
    )
    min_size = 200
    units = plan_units(doc, max_size=2048, min_size=min_size)
    # The first (tiny) unit absorbed the next until it cleared the floor.
    assert all(
        (u.end - u.start) >= min_size or u is units[-1] for u in units
    )
    # Covering preserved.
    assert units[0].start == 0
    assert units[-1].end == len(doc)
    for a, b in zip(units, units[1:]):
        assert a.end == b.start


def test_thin_trailing_unit_glues_backward() -> None:
    # A thin trailing section that cannot grow forward (no next) glues
    # backward into its predecessor.
    doc = _make_doc([("## Body", 300), ("## Acknowledgments", 3)])
    min_size = 400
    units = plan_units(doc, max_size=8192, min_size=min_size)
    assert len(units) == 1
    assert units[0].start == 0
    assert units[0].end == len(doc)


def test_no_chunk_below_floor_when_mergeable() -> None:
    doc = _make_doc(
        [
            ("## Methodology", 4),
            ("## Quotes Fixing", 3),
            ("## Recaser", 6),
            ("## Main", 250),
            ("## Acknowledgments", 6),
        ]
    )
    min_size = 300
    chunks = _run(
        doc, noop(), max_size=2048, min_size=min_size,
        segmenter=punctuation_segmenter,
    )
    assert "".join(c.text for c in chunks) == doc
    assert all(len(c.text) <= 2048 for c in chunks)
    # Only the very last chunk may fall below the floor (leftover tail);
    # everything else must clear it.
    for c in chunks[:-1]:
        assert len(c.text) >= min_size, f"thin chunk survived: {c.text!r}"


def test_no_heading_only_chunk_after_merge() -> None:
    doc = _make_doc(
        [("## A", 3), ("## B", 4), ("## C", 250)]
    )
    chunks = _run(doc, noop(), max_size=2048, min_size=400,
                  segmenter=punctuation_segmenter)
    heading_re = re.compile(r"#{1,6}(\s|$)")
    for c in chunks:
        nonblank = [ln for ln in c.text.splitlines() if ln.strip()]
        only_headings = all(heading_re.match(ln) for ln in nonblank)
        assert not only_headings, f"heading-only chunk: {c.text!r}"


def test_merge_never_exceeds_max_size() -> None:
    # Two units that individually fit but would overflow if joined must
    # stay separate even though the first is below the floor.
    a = _Unit(0, 100, needs_model=False)
    b = _Unit(100, 100 + 2000, needs_model=False)
    merged = _merge_small_units([a, b], max_size=2048, min_size=500)
    # 100 + 2000 = 2100 > 2048 → cannot merge.
    assert merged == [a, b]


def test_merge_stops_at_floor_no_packing() -> None:
    # Floor 500. The first two thin units (400 each) merge to clear the
    # floor, then STOP — the third unit is already above the floor (800)
    # so it is not packed onto the merged span.
    units = [
        _Unit(0, 400, needs_model=False),
        _Unit(400, 800, needs_model=False),
        _Unit(800, 1600, needs_model=False),
    ]
    merged = _merge_small_units(units, max_size=2048, min_size=500)
    assert merged == [
        _Unit(0, 800, needs_model=False),
        _Unit(800, 1600, needs_model=False),
    ]


def test_merge_disabled_with_zero_min_size() -> None:
    units = [
        _Unit(0, 50, needs_model=False),
        _Unit(50, 100, needs_model=False),
    ]
    assert _merge_small_units(units, max_size=2048, min_size=0) == units


# --- SPEC-CHUNK-632 — parent-intro fold ---------------------------------


def test_small_parent_intro_folds_into_oversized_first_child() -> None:
    # A short section lead-in sitting before an oversized first
    # subsection must not strand as a thin chunk; it folds into the
    # first child and rides with that child's first chunk.
    intro = (
        "In this section, we first discuss our experiments on the "
        "word recognition systems."
    )
    assert len(intro) < 716  # below the default floor
    child1 = ("Word recognition sentence here. " * 80).strip()  # oversized
    child2 = ("Recaser sentence here too. " * 90).strip()  # oversized
    doc = (
        f"## Experiments\n\n{intro}\n\n"
        f"### Word Recognition\n\n{child1}\n\n"
        f"### Recaser\n\n{child2}\n"
    )
    intro_span_end = doc.index("### Word Recognition")
    units = plan_units(doc, max_size=2048)
    # The intro is no longer a standalone unit ending at the child heading.
    assert not any(
        u.start == 0 and u.end == intro_span_end for u in units
    ), "parent intro stranded as its own unit"
    # The first unit folds the intro into the first child.
    first = units[0]
    assert first.start == 0
    assert "### Word Recognition" in doc[first.start : first.end]

    chunks = _run(doc, noop(), max_size=2048, segmenter=punctuation_segmenter)
    assert "".join(c.text for c in chunks) == doc
    intro_chunks = [c for c in chunks if intro in c.text]
    assert len(intro_chunks) == 1, "intro split across chunks"
    # It rides with the first child's first chunk (carries its heading).
    assert "### Word Recognition" in intro_chunks[0].text


def test_small_parent_intro_folds_into_fitting_first_child() -> None:
    # Small parent intro + a first child that fits → one combined direct
    # unit (no model call), within max_size.
    intro = "Brief lead-in to the section."
    child1 = ("First child body sentence. " * 15).strip()  # fits
    child2 = ("Second child sentence here. " * 120).strip()  # oversized
    doc = (
        f"## Parent\n\n{intro}\n\n"
        f"### First\n\n{child1}\n\n"
        f"### Second\n\n{child2}\n"
    )
    units = plan_units(doc, max_size=2048, min_size=300)
    combined = units[0]
    assert combined.start == 0
    assert combined.needs_model is False
    assert (combined.end - combined.start) <= 2048
    text = doc[combined.start : combined.end]
    assert intro in text
    assert "### First" in text
    assert "### Second" not in text


def test_large_parent_intro_stays_standalone() -> None:
    # A parent intro that already clears the floor is a fine standalone
    # chunk — it must NOT be folded into the child.
    intro = ("This intro is long enough to stand on its own here. " * 16).strip()
    assert len(intro) >= 716
    child = ("Child body sentence here. " * 120).strip()  # oversized
    doc = f"## Parent\n\n{intro}\n\n### Child\n\n{child}\n"
    intro_end = doc.index("### Child")
    units = plan_units(doc, max_size=2048)
    assert any(
        u.start == 0 and u.end == intro_end for u in units
    ), "large parent intro was folded but should stay standalone"


def test_bare_parent_heading_handled_by_bare_merge_not_fold() -> None:
    # A bare parent heading (no lead-in prose) before an oversized child
    # is handled by the bare-heading merge, not the intro fold — and not
    # double-handled.
    child = ("Child sentence here now. " * 130).strip()  # oversized
    doc = f"## Parent\n\n### Child\n\n{child}\n"
    chunks = _run(doc, noop(), max_size=2048, segmenter=punctuation_segmenter)
    assert "".join(c.text for c in chunks) == doc
    heading_re = re.compile(r"#{1,6}(\s|$)")
    for c in chunks:
        nonblank = [ln for ln in c.text.splitlines() if ln.strip()]
        only_headings = all(heading_re.match(ln) for ln in nonblank)
        assert not only_headings, f"stranded heading chunk: {c.text!r}"
    assert chunks[0].text.lstrip().startswith("## Parent")
    assert "### Child" in chunks[0].text


def test_sibling_sections_are_not_folded() -> None:
    # The fold is strictly parent → own-first-child. Two sibling
    # top-level sections (no fold_forward flag) are left untouched even
    # when the first is tiny.
    abstract = _Unit(0, 100, needs_model=False, fold_forward=False)
    intro = _Unit(100, 5000, needs_model=True, fold_forward=False)
    assert _fold_parent_intros(
        [abstract, intro], max_size=2048, min_size=716
    ) == [abstract, intro]

    # End-to-end: an abstract that clears the floor stays its own chunk,
    # never folded into the following oversized introduction.
    abstract_body = ("Abstract sentence here. " * 45).strip()  # >= 716, fits
    intro_body = ("Introduction sentence here. " * 120).strip()  # oversized
    doc = (
        f"## Abstract\n\n{abstract_body}\n\n"
        f"## Introduction\n\n{intro_body}\n"
    )
    chunks = _run(doc, noop(), max_size=2048, segmenter=punctuation_segmenter)
    abstract_chunks = [c for c in chunks if "## Abstract" in c.text]
    assert len(abstract_chunks) == 1
    assert "## Introduction" not in abstract_chunks[0].text
