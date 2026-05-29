"""Spike: structure-first chunking (experimental).

An additive, opt-in alternative to the default semantic pipeline
(:func:`fancychunk.chunk_document` / :func:`fancychunk.split_chunks`).
It is **not** wired into any default; import it explicitly.

Hypothesis (see the spike PR): for documents *with* headings, honoring
the heading structure first is better for both latency and boundary
quality than letting the O(N^2) similarity DP decide every cut. The
slow models (SaT sentence segmentation, the chunklet embedder) only
need to run on the rare section that overflows ``max_size``.

Pipeline
--------
1. Parse the document into heading-delimited *segments* (a heading
   line plus the prose that follows it, until the next heading).
2. Recurse the implied heading tree top-down. For any node whose
   *entire subtree* already fits in ``max_size``, emit it directly as
   one chunk — **no SaT, no embedding call**. This is the latency win
   and the boundary win: the section becomes the primary unit, so a
   heading always lands at a chunk start (fixes "Observation C", where
   a whole-section-that-fits left the heading mid-chunk).
3. Only a section whose own heading+body still overflows ``max_size``
   falls back to the existing semantic split
   (``split_sentences`` → ``split_chunklets`` → ``split_chunks``) on
   *that span alone*.
4. A *bare* heading unit (a container/front-matter heading with no
   body of its own, e.g. ``# Title`` before ``## Abstract``) is merged
   forward into the following unit so a lone heading is never stranded.
   This is the only structural merge — genuinely short standalone
   sections are kept as-is (small chunks are not inherently bad).

Heading levels
--------------
``level(heading)`` = (count of leading ``#``) + (count of ``" ::: "``
separators in the heading text). This unifies two corpora:

* **Real Markdown** — hierarchy is the ``#`` count (``#`` → 1,
  ``##`` → 2, …), no ``:::`` present.
* **Qasper** — every section heading is rendered flat as ``##`` and the
  hierarchy is encoded in the *text* as a ``:::`` path
  (``## Methodology ::: Sub ::: Step``). The ``:::`` count recovers the
  real depth.

Invariants (same as the default pipeline)
-----------------------------------------
* Covering: every chunk ``len <= max_size``.
* Round-trip: ``"".join(c.text for c in chunks) == document``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from . import _constants as C
from ._segmenter import SentenceSegmenter
from .chunks import Chunk, ChunkletEmbedder, split_chunks
from .chunklets import split_chunklets
from .errors import ValidationError
from .sentences import split_sentences

# Line-start heading match. Mirrors ``headings._HEADING_LINE_RE`` (a
# bare line-anchored scan, not fence-aware) so structure-first agrees
# with the library's other heading handling.
_HEADING_RE = re.compile(r"(#{1,6})(?:\s|$)")
_SEP = " ::: "


@dataclass
class StructureFirstStats:
    """Optional instrumentation for the benchmark harness.

    Populated in place when passed to
    :func:`split_chunks_structure_first`. ``chars_direct`` /
    ``chars_fallback`` partition the document; their ratio is the
    fraction of work that skipped the slow models.
    """

    units_direct: int = 0
    units_fallback: int = 0
    chars_direct: int = 0
    chars_fallback: int = 0
    fallback_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def direct_fraction(self) -> float:
        total = self.chars_direct + self.chars_fallback
        return self.chars_direct / total if total else 1.0


@dataclass(frozen=True)
class _Segment:
    """A heading and the body that follows it, as a char span."""

    start: int
    end: int  # exclusive; end of body (start of next heading or EOF)
    level: int


@dataclass(frozen=True)
class _Unit:
    """A planned output span and whether it needs the slow models."""

    start: int
    end: int
    needs_model: bool


def _heading_level(line: str) -> int:
    """Level of a heading line, or 0 if the line is not a heading.

    ``#``-count plus ``:::``-separator count (see module docstring)."""
    m = _HEADING_RE.match(line)
    if m is None:
        return 0
    hashes = len(m.group(1))
    text = line[m.end() :]
    return hashes + text.count(_SEP)


def _segments(document: str) -> tuple[int, list[_Segment]]:
    """Split ``document`` into heading-delimited segments.

    Returns ``(preamble_end, segments)`` where ``[0, preamble_end)`` is
    any text before the first heading (possibly empty) and each segment
    covers a heading line through the char just before the next heading.
    """
    heading_offsets: list[tuple[int, int]] = []  # (line_start, level)
    pos = 0
    n = len(document)
    while pos < n:
        nl = document.find("\n", pos)
        line_end = n if nl == -1 else nl
        line = document[pos:line_end]
        level = _heading_level(line)
        if level:
            heading_offsets.append((pos, level))
        pos = line_end + 1

    if not heading_offsets:
        return n, []

    preamble_end = heading_offsets[0][0]
    segs: list[_Segment] = []
    for idx, (start, level) in enumerate(heading_offsets):
        end = (
            heading_offsets[idx + 1][0]
            if idx + 1 < len(heading_offsets)
            else n
        )
        segs.append(_Segment(start=start, end=end, level=level))
    return preamble_end, segs


def _plan_range(
    segs: list[_Segment],
    lo: int,
    hi: int,
    range_end: int,
    max_size: int,
    out: list[_Unit],
) -> None:
    """Recurse the heading forest ``segs[lo:hi]`` top-down, appending
    units to ``out``. ``range_end`` is the char offset where this forest
    range ends (the parent's subtree end, or the document end at the top
    level) — used to bound the last sibling's subtree."""
    i = lo
    while i < hi:
        # Extent of segment i's subtree: up to the next sibling-or-higher.
        j = i + 1
        while j < hi and segs[j].level > segs[i].level:
            j += 1
        subtree_start = segs[i].start
        subtree_end = segs[j].start if j < hi else range_end

        if subtree_end - subtree_start <= max_size:
            # Whole subtree fits — emit directly, no models.
            out.append(_Unit(subtree_start, subtree_end, needs_model=False))
        else:
            # Node's own heading+body (the prose before its first child).
            own_end = segs[i + 1].start if (i + 1) < j else subtree_end
            own_len = own_end - subtree_start
            if own_len <= max_size:
                out.append(_Unit(subtree_start, own_end, needs_model=False))
            else:
                out.append(_Unit(subtree_start, own_end, needs_model=True))
            # Recurse into the children, bounded by this subtree's end.
            _plan_range(segs, i + 1, j, subtree_end, max_size, out)
        i = j


def _is_bare_heading(document: str, unit: _Unit) -> bool:
    """True iff ``unit`` is only heading line(s) + whitespace — no body.

    Such container/front-matter headings are merged forward so a lone
    heading is never stranded at a chunk tail."""
    text = document[unit.start : unit.end]
    if not text.strip():
        return True
    for line in text.splitlines():
        if line.strip() and not _heading_level(line):
            return False
    return True


def _merge_bare_headings(
    document: str, units: list[_Unit], max_size: int
) -> list[_Unit]:
    """Merge each bare-heading unit forward into the next unit (or, if
    that would overflow / there is no next, backward), preserving the
    covering tiling. ``needs_model`` is OR-ed across a merge."""
    if not units:
        return units
    merged: list[_Unit] = []
    carry: _Unit | None = None  # a pending bare heading to prepend
    for unit in units:
        if carry is not None:
            # Gluing the heading onto a near-cap direct unit can push it
            # over max_size; route the combined span through the fallback
            # split in that case so covering still holds.
            overflow = (unit.end - carry.start) > max_size
            unit = _Unit(
                carry.start,
                unit.end,
                carry.needs_model or unit.needs_model or overflow,
            )
            carry = None
        if _is_bare_heading(document, unit) and (unit.end - unit.start) < max_size:
            # Defer: try to glue onto the following unit.
            carry = unit
            continue
        merged.append(unit)
    if carry is not None:
        # Trailing bare heading: glue backward if it fits, else keep.
        if merged and (merged[-1].end - merged[-1].start) + (
            carry.end - carry.start
        ) <= max_size:
            prev = merged[-1]
            merged[-1] = _Unit(
                prev.start, carry.end, prev.needs_model or carry.needs_model
            )
        else:
            merged.append(carry)
    return merged


def plan_units(document: str, max_size: int = C.DEFAULT_MAX_SIZE_CHARS) -> list[_Unit]:
    """Pure structure-only plan: the output spans and which need models.

    Shared by the measurement script and the splitter so both agree on
    what "already fits" means. No SaT, no embedding."""
    if max_size <= 0:
        raise ValidationError("max_size must be positive")
    n = len(document)
    if n == 0:
        return []

    preamble_end, segs = _segments(document)

    units: list[_Unit] = []
    # Preamble (text before the first heading, or the whole doc when it
    # has no headings at all).
    if preamble_end > 0:
        units.append(
            _Unit(0, preamble_end, needs_model=preamble_end > max_size)
        )
    if segs:
        _plan_range(segs, 0, len(segs), n, max_size, units)

    return _merge_bare_headings(document, units, max_size)


async def split_chunks_structure_first(
    document: str,
    embedder: ChunkletEmbedder,
    max_size: int = C.DEFAULT_MAX_SIZE_CHARS,
    *,
    segmenter: SentenceSegmenter | None = None,
    stats: StructureFirstStats | None = None,
) -> list[Chunk]:
    """Structure-first chunking of ``document`` (experimental).

    Sections that already fit ``max_size`` are emitted directly with no
    model calls; only an overflowing section falls back to the semantic
    pipeline (``split_sentences`` → ``split_chunklets`` → ``split_chunks``).

    ``embedder`` and ``segmenter`` are only used on the fallback path.
    Returns :class:`Chunk` objects with ``start`` / ``end`` offsets into
    ``document`` and a populated ``heading_path`` (same scan as the
    default pipeline). Satisfies the covering and round-trip invariants.
    """
    if max_size <= 0:
        raise ValidationError("max_size must be positive")
    if not document:
        return []

    units = plan_units(document, max_size)

    bare: list[Chunk] = []
    for unit in units:
        span = document[unit.start : unit.end]
        if not unit.needs_model:
            if stats is not None:
                stats.units_direct += 1
                stats.chars_direct += len(span)
            bare.append(Chunk(text=span, start=unit.start, end=unit.end))
            continue

        # Fallback: run the slow semantic split on this span alone, then
        # rebase offsets back into the full document.
        if stats is not None:
            stats.units_fallback += 1
            stats.chars_fallback += len(span)
            stats.fallback_spans.append((unit.start, unit.end))
        sentences = split_sentences(span, max_len=max_size, segmenter=segmenter)
        chunklets = split_chunklets(sentences, max_size=max_size)
        sub_chunks = await split_chunks(chunklets, embedder, max_size=max_size)
        for sc in sub_chunks:
            bare.append(
                Chunk(
                    text=sc.text,
                    start=unit.start + (sc.start or 0),
                    end=unit.start + (sc.end or len(sc.text)),
                )
            )

    # Populate heading_path with the same scan the default pipeline uses.
    from .headings import heading_paths

    paths = heading_paths(bare)
    return [replace(c, heading_path=p) for c, p in zip(bare, paths)]
