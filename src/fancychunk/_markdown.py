"""Helpers that interpret a Markdown document via markdown-it-py.

Two queries the splitter stages need against the parsed token stream:

* ``heading_spans`` — character spans of ATX/Setext heading content, used
  by stage 1 to override boundary probabilities.
* ``token_openers_by_line`` — for each source line, which block-level
  token types open on that line, used by stage 2 to assign per-sentence
  boundary probabilities.

Both helpers operate on the original document string; line numbers from
markdown-it tokens are 0-indexed into a list of lines split on ``\n``.
"""

from __future__ import annotations

from dataclasses import dataclass

from markdown_it import MarkdownIt


@dataclass(frozen=True)
class HeadingSpan:
    """Character span of a heading's marker through its last
    non-whitespace text character.

    ``first`` is the position of the heading marker's first character
    (e.g. the ``#``); ``last`` is the index of the last non-whitespace
    character on the heading's text line(s). Both are inclusive.
    """

    first: int
    last: int


def _line_starts(document: str) -> list[int]:
    """Return the character offset of each line's first character.

    Index ``i`` is the offset of line ``i`` (0-indexed). An empty
    document yields a single offset 0.
    """
    starts: list[int] = [0]
    for idx, ch in enumerate(document):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


# Module-level CommonMark parser. ``markdown-it-py`` parsing is
# reentrant (``.parse()`` builds a fresh ``StateBlock`` per call;
# the rule list is read-only after init), so a single shared instance
# is safe across threads and avoids the per-call construction
# overhead. ``tests/test_markdown_concurrency.py`` regresses this if
# a future markdown-it-py release breaks the assumption.
_PARSER = MarkdownIt("commonmark")


def _parser() -> MarkdownIt:
    return _PARSER


def heading_spans(document: str) -> list[HeadingSpan]:
    """Return character spans for every ATX/Setext heading in ``document``.

    A heading's span runs from the first character of the heading
    marker (or, for Setext headings, the first character of the heading
    text) through the last non-whitespace character of the heading's
    text. Trailing whitespace and following blank lines are not part
    of the span.

    Headings with empty text bodies (e.g. ``# \\n``) are returned with
    ``first == last + 1`` (a degenerate span). Callers should detect
    this and apply the SPEC-CHUNK-108 edge case for empty headings.
    """
    if not document:
        return []
    md = _parser()
    tokens = md.parse(document)
    line_starts = _line_starts(document)
    n = len(document)

    spans: list[HeadingSpan] = []
    for tok in tokens:
        if tok.type != "heading_open":
            continue
        if tok.map is None:
            continue
        start_line, end_line = tok.map  # half-open
        # Character window: from the marker's start through end of the
        # heading content. For ATX, both start_line == end_line-1
        # typically; for Setext, end_line - 1 is the underline line.
        first = line_starts[start_line]
        # Trim leading whitespace on the start line to find the marker
        while first < n and document[first] in (" ", "\t"):
            first += 1
        # Span end: scan back from end_line-1's terminating newline
        # over whitespace to find last non-whitespace character.
        end_line_idx = end_line - 1
        if end_line_idx + 1 < len(line_starts):
            end_pos = line_starts[end_line_idx + 1] - 1  # newline index
        else:
            end_pos = n - 1
        # If line ends without newline, end_pos may equal n-1 already.
        # Walk back from end_pos to the last non-whitespace char.
        last = end_pos
        while last >= first and document[last] in (" ", "\t", "\n", "\r"):
            last -= 1
        spans.append(HeadingSpan(first=first, last=last))
    return spans


@dataclass(frozen=True)
class LineOpeners:
    """The set of block-level token types that open on a given source
    line (0-indexed).
    """

    line: int
    types: tuple[str, ...]


_RELEVANT_OPEN_TYPES = frozenset(
    {
        "heading_open",
        "blockquote_open",
        "paragraph_open",
        "bullet_list_open",
        "ordered_list_open",
    }
)


def openers_by_line(document: str) -> dict[int, set[str]]:
    """For each source line, which relevant block-level openers begin there.

    Only the token types listed in SPEC-CHUNK-240 are tracked. Lines
    with no relevant opener are simply absent from the result.
    """
    out: dict[int, set[str]] = {}
    if not document:
        return out
    md = _parser()
    tokens = md.parse(document)
    for tok in tokens:
        if tok.type not in _RELEVANT_OPEN_TYPES:
            continue
        if tok.map is None:
            continue
        line = tok.map[0]
        out.setdefault(line, set()).add(tok.type)
    return out


def line_of_offset(line_starts: list[int], offset: int) -> int:
    """Return the 0-indexed line containing ``offset``.

    ``line_starts`` must be sorted ascending (e.g. produced by
    :func:`compute_line_starts`).
    """
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def compute_line_starts(document: str) -> list[int]:
    """Public wrapper around the internal ``_line_starts`` helper."""
    return _line_starts(document)
