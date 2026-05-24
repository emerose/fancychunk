"""Stage 5 tests — contextual chunk headings."""

from __future__ import annotations

from fancychunk import heading_paths


# TV-501 / SPEC-CHUNK-540 — empty input.
def test_tv_501_empty_input() -> None:
    assert heading_paths([]) == []


# TV-502 / SPEC-CHUNK-541 — document without headings.
def test_tv_502_no_headings() -> None:
    chunks = [
        "First paragraph.\n\n",
        "Second paragraph.\n",
        "Third paragraph.\n",
    ]
    assert heading_paths(chunks) == ["", "", ""]


# TV-503 / SPEC-CHUNK-510, -511 — simple linear heading structure.
def test_tv_503_linear_heading_structure() -> None:
    chunks = [
        "# Introduction\n\nOpening text.\n\n",
        "## Background\n\nMore detail.\n\n",
        "Continuing background.\n",
        "## Method\n\nDescription.\n",
    ]
    assert heading_paths(chunks) == [
        "",
        "# Introduction\n",
        "# Introduction\n\n## Background\n",
        "# Introduction\n\n## Background\n",
    ]


# TV-504 / SPEC-CHUNK-520 — stack reset when heading level rises.
def test_tv_504_stack_reset() -> None:
    chunks = [
        "# A\n\n## A.1\n\n### A.1.x\n\nContent.\n",
        "Next chunk content.\n",
        "# B\n\nB content.\n",
        "More B content.\n",
    ]
    assert heading_paths(chunks) == [
        "",
        "# A\n\n## A.1\n\n### A.1.x\n",
        "# A\n\n## A.1\n\n### A.1.x\n",
        "# B\n",
    ]


# TV-505 / SPEC-CHUNK-502, -542 — first chunk introduces the first heading.
def test_tv_505_first_chunk_starts_with_heading() -> None:
    chunks = ["# Title\n\nBody text.\n\n", "More body text.\n"]
    assert heading_paths(chunks) == ["", "# Title\n"]


# TV-506 / SPEC-CHUNK-543 — heading levels skipped.
def test_tv_506_levels_skipped() -> None:
    chunks = ["# H1\n\n### H3\n\nContent under H3.\n", "More content.\n"]
    assert heading_paths(chunks) == ["", "# H1\n\n### H3\n"]


# TV-507 / SPEC-CHUNK-544 — seven or more '#' is not a heading.
def test_tv_507_seven_hashes_not_a_heading() -> None:
    chunks = ["####### Not a heading\n\nBody text.\n", "More text.\n"]
    assert heading_paths(chunks) == ["", ""]


# TV-508 / SPEC-CHUNK-512, -513 — trailing whitespace preserved.
def test_tv_508_path_format_multi_line() -> None:
    chunks = ["# Title with trailing spaces  \n\nBody.\n", "More body.\n"]
    assert heading_paths(chunks) == ["", "# Title with trailing spaces  \n"]


# TV-509 — prepended path produces valid Markdown structure (sanity check).
def test_tv_509_prepended_path_is_valid_markdown() -> None:
    chunks = ["# Top\n\nIntro.\n\n", "## Sub\n\nDetails.\n\n", "Continued.\n"]
    paths = heading_paths(chunks)
    composed = [p + c for p, c in zip(paths, chunks)]
    # The composed chunks contain the headings; verify the path
    # doesn't introduce duplication when the chunk doesn't repeat it.
    assert composed[2].startswith("# Top\n\n## Sub\n")


# SPEC-CHUNK-530 — determinism.
def test_determinism() -> None:
    chunks = ["# A\n\nText.\n", "More.\n"]
    assert heading_paths(chunks) == heading_paths(chunks)
