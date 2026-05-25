"""Stage 5 — contextual chunk headings (SPEC-CHUNK-5xx).

Public entry point: ``heading_paths``.
"""

from __future__ import annotations

import re

from . import _constants as C

# Anchoring is done manually by the scanner (only call ``.match`` at
# line-start positions), so this pattern does not include ``^``.
_HEADING_LINE_RE = re.compile(r"(#{1,6})(\s|$)")


def heading_paths(chunks: list[str]) -> list[str]:
    """Return the per-chunk Markdown heading path.

    Implements ``docs/specs/05-contextual-headings.md``.
    """
    paths: list[str] = []
    stack: list[str | None] = [None] * C.MAX_HEADING_LEVELS

    for chunk in chunks:
        paths.append(_render_path(stack))
        _scan_and_update(chunk, stack)
    return paths


def _scan_and_update(chunk: str, stack: list[str | None]) -> None:
    """SPEC-CHUNK-511 — update ``stack`` with every heading line in ``chunk``.

    Line starts: the chunk's first character, plus every position
    immediately after ``\\n`` or ``\\r`` (treating ``\\r\\n``, ``\\n``,
    and bare-CR line endings uniformly — the regex won't match a
    leading newline anyway, so a harmless double-line-start for CRLF
    is fine).
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
                    line = chunk[i : end_of_line + 1]
                    i = end_of_line + 1
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
    # Include the line terminator in the heading line. For CRLF,
    # include both characters.
    if chunk[k] == "\r" and k + 1 < n and chunk[k + 1] == "\n":
        return k + 1
    return k


def _render_path(stack: list[str | None]) -> str:
    parts = [s for s in stack if s is not None]
    if not parts:
        return ""
    return C.HEADING_PATH_SEPARATOR.join(parts)
