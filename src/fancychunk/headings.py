"""Stage 5 — contextual chunk headings (SPEC-CHUNK-5xx).

Public entry points:

* :func:`heading_paths` — for each chunk, the Markdown heading
  stack in scope at the chunk's start, as a tuple of full heading
  lines (e.g. ``("# Top", "## **Bold** Sub")``). Each entry preserves
  the ``#`` markers and inline markdown but is stripped of trailing
  whitespace / newlines. The marker count encodes heading level.
* :func:`enrich_with_headings` — convenience that prepends a
  rendered version of the heading path to each chunk's ``text``.
* :func:`render_heading_path` — convert a tuple-form heading path
  back to a single Markdown string. Used internally by late chunking
  and enrich; exposed for callers who want the same rendering.
"""

from __future__ import annotations

import re

from . import _constants as C
from ._telemetry import get_tracer
from .chunks import Chunk

# Anchoring is done manually by the scanner (only call ``.match`` at
# line-start positions), so this pattern does not include ``^``.
_HEADING_LINE_RE = re.compile(r"(#{1,6})(\s|$)")


def heading_paths(chunks: list[Chunk]) -> list[tuple[str, ...]]:
    """Return the per-chunk Markdown heading path.

    Each path is a tuple of full heading lines in scope at the
    chunk's start (e.g. ``("# Top", "## Sub")``). Marker count
    (``#`` repeats) encodes the level — so paths from documents with
    skipped levels (``# H1`` then ``### H3``) preserve that
    information as ``("# H1", "### H3")`` rather than the misleading
    ``("H1", "H3")``.

    Each entry is stripped of trailing whitespace and newlines.
    Empty tuple means "no heading in scope" — e.g. the first chunk
    before any heading appears.

    Implements ``docs/specs/05-contextual-headings.md``.
    """
    with get_tracer().start_as_current_span("fancychunk.heading_paths") as span:
        span.set_attribute("fancychunk.chunks.count", len(chunks))
        paths: list[tuple[str, ...]] = []
        stack: list[str | None] = [None] * C.MAX_HEADING_LEVELS

        for chunk in chunks:
            paths.append(_render_path(stack))
            _scan_and_update(chunk.text, stack)
        span.set_attribute(
            "fancychunk.paths.non_empty",
            sum(1 for p in paths if p),
        )
        return paths


def render_heading_path(path: tuple[str, ...]) -> str:
    """Render a tuple-form heading path to a single Markdown string.

    Joins entries with ``\\n`` and appends a trailing newline so the
    rendered form is suitable as a preamble before chunk text.
    Returns ``""`` for the empty path.
    """
    if not path:
        return ""
    return "\n".join(path) + "\n"


def _scan_and_update(chunk: str, stack: list[str | None]) -> None:
    """SPEC-CHUNK-511 — update ``stack`` with every heading line in ``chunk``.

    Captures the stripped heading line (``"# Heading"`` — no trailing
    whitespace or newline) into the stack at the heading's level.

    Line starts: the chunk's first character, plus every position
    immediately after ``\\n`` or ``\\r`` (treating ``\\r\\n``, ``\\n``,
    and bare-CR line endings uniformly).
    """
    i = 0
    n = len(chunk)
    while i < n:
        if i == 0 or chunk[i - 1] in "\n\r":
            m = _HEADING_LINE_RE.match(chunk, i)
            if m is not None:
                level = len(m.group(1))
                end_of_line = _find_line_end(chunk, i)
                if end_of_line == -1:
                    line = chunk[i:]
                    i = n
                else:
                    # Capture up through (but not including) the line
                    # terminator. Strip trailing whitespace from the
                    # captured line — the user has opted into losing
                    # trailing whitespace (it carried no semantic
                    # weight) in exchange for a cleaner tuple form.
                    line = chunk[i:end_of_line]
                    i = end_of_line + 1
                line = line.rstrip()
                stack[level - 1] = line
                for j in range(level, C.MAX_HEADING_LEVELS):
                    stack[j] = None
                continue
        i += 1


def _find_line_end(chunk: str, start: int) -> int:
    """Return the index of the line terminator at or after ``start``,
    or ``-1`` if the line runs to end of chunk. Recognizes ``\\n``,
    ``\\r``, and ``\\r\\n``.
    """
    n = len(chunk)
    k = start
    while k < n and chunk[k] not in "\n\r":
        k += 1
    if k == n:
        return -1
    # The returned index is the first character of the terminator.
    # The caller skips past the terminator (one or two chars for CRLF).
    if chunk[k] == "\r" and k + 1 < n and chunk[k + 1] == "\n":
        return k + 1
    return k


def _render_path(stack: list[str | None]) -> tuple[str, ...]:
    """Squeeze the level-indexed stack into a tuple, dropping None slots."""
    return tuple(s for s in stack if s is not None)


def enrich_with_headings(chunks: list[Chunk]) -> list[Chunk]:
    """Return chunks with their Markdown heading path prepended to
    ``text``, separated by a blank line. Chunks whose path is empty
    are returned unchanged. Metadata (``start`` / ``end`` /
    ``heading_path``) is preserved from the input — they still
    reference the original source's character offsets, even though
    ``len(chunk.text)`` no longer equals ``end - start`` after
    enrichment.

    The output preserves ``len(chunks)``; the i-th output element
    corresponds to the i-th input element. Uses the input's
    ``chunk.heading_path`` when populated; falls back to computing
    via :func:`heading_paths` when ``None``.

    Implements SPEC-CHUNK-520. The concatenation round-trip
    (SPEC-CHUNK-300) does **not** hold after enrichment — this
    helper deliberately breaks it in exchange for storage-time
    outline context.
    """
    from dataclasses import replace

    with get_tracer().start_as_current_span(
        "fancychunk.enrich_with_headings"
    ) as span:
        span.set_attribute("fancychunk.chunks.count", len(chunks))
        paths = resolve_heading_paths(chunks)
        out = [
            replace(c, text=(render_heading_path(p) + c.text)) if p else c
            for p, c in zip(paths, chunks)
        ]
        span.set_attribute(
            "fancychunk.paths.non_empty",
            sum(1 for p in paths if p),
        )
        return out


def resolve_heading_paths(chunks: list[Chunk]) -> list[tuple[str, ...]]:
    """Return heading paths for ``chunks``, preferring populated
    ``chunk.heading_path`` and falling back to a fresh
    :func:`heading_paths` scan when any chunk has ``None``."""
    if all(c.heading_path is not None for c in chunks):
        return [c.heading_path or () for c in chunks]
    return heading_paths(chunks)
