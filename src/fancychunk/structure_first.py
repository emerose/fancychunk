"""Structure-first chunking — the engine behind :func:`chunk_document`.

Honors a document's heading structure before reaching for the slow
models. For documents *with* headings this is both faster and produces
better boundaries than letting the O(N^2) similarity DP decide every
cut: the slow models (SaT sentence segmentation, the chunklet embedder)
only run on a section that overflows ``max_size``.

Implements spec [06-structural-chunking](../../docs/specs/06-structural-chunking.md)
(SPEC-CHUNK-600 through SPEC-CHUNK-640).

Pipeline
--------
1. Parse the document into heading-delimited *segments* (a heading
   line plus the prose that follows it, until the next heading).
2. Recurse the implied heading tree top-down. For any node whose
   *entire subtree* already fits in ``max_size``, emit it directly as
   one chunk — **no SaT, no embedding call**. This is the latency win
   and the boundary win: the section becomes the primary unit, so a
   heading always lands at a chunk start.
3. Only a section whose own heading+body still overflows ``max_size``
   falls back to the semantic split
   (``split_sentences`` → ``split_chunklets`` → ``split_chunks``) on
   *that span alone*.
4. A *bare* heading unit (a container/front-matter heading with no
   body of its own, e.g. ``# Title`` before ``## Abstract``) is merged
   forward into the following unit so a lone heading is never stranded.
5. A *small* parent-section intro (an overflowing parent's own
   heading+lead-in prose, emitted before its child subsections) is
   folded forward into its first child so a section lead-in is never
   severed from the section it introduces — see
   :func:`_fold_parent_intros`.
6. A unit below ``min_size`` is merged into a neighbor so the partition
   has no thin, fragmented chunks — see :func:`_merge_small_units`.
   Genuinely short standalone sections that clear the floor are kept
   as-is (small chunks are not inherently bad).

Heading levels
--------------
``level(heading)`` = (count of leading ``#``) + (count of ``" ::: "``
separators in the heading text). This unifies two heading conventions:

* **Real Markdown** — hierarchy is the ``#`` count (``#`` → 1,
  ``##`` → 2, …), no ``:::`` present.
* **Flat ``##`` with a ``:::`` path** — every section heading is
  rendered flat as ``##`` and the hierarchy is encoded in the *text*
  as a ``:::`` path (``## Methodology ::: Sub ::: Step``). The ``:::``
  count recovers the real depth. (This is how the Qasper corpus
  renders section hierarchy.)

Invariants (same as the default pipeline)
-----------------------------------------
* Covering: every chunk ``len <= max_size``.
* Round-trip: ``"".join(c.text for c in chunks) == document``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from . import _constants as C
from ._segmenter import SentenceSegmenter
from ._telemetry import get_tracer
from .chunks import Chunk, ChunkletEmbedder, split_chunks
from .chunklets import split_chunklets
from .errors import ValidationError
from .sentences import split_sentences

# Line-start heading match. Mirrors ``headings._HEADING_LINE_RE`` (a
# bare line-anchored scan, not fence-aware) so structure-first agrees
# with the library's other heading handling.
_HEADING_RE = re.compile(r"(#{1,6})(?:\s|$)")
_SEP = " ::: "

# Default chunk-size floor as a fraction of ``max_size``. A unit smaller
# than this is merged into a neighbor (forward first) to avoid thin,
# fragmented chunks — see ``_merge_small_units``.
_MIN_CHUNK_FRACTION = 0.35


@dataclass(frozen=True)
class _Segment:
    """A heading and the body that follows it, as a char span."""

    start: int
    end: int  # exclusive; end of body (start of next heading or EOF)
    level: int


@dataclass(frozen=True)
class _Unit:
    """A planned output span and whether it needs the slow models.

    ``fold_forward`` marks a unit emitted by :func:`_plan_range` for an
    overflowing parent's own heading+lead-in prose, when that parent has
    child subsections following it. A *small* such unit is folded into
    its first child by :func:`_fold_parent_intros` so a section lead-in
    is never stranded as a thin chunk."""

    start: int
    end: int
    needs_model: bool
    fold_forward: bool = False


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
            has_children = (i + 1) < j
            own_end = segs[i + 1].start if has_children else subtree_end
            own_len = own_end - subtree_start
            # The own-head+body unit can fold into its first child only
            # when there *is* a child (the unit it precedes in ``out``).
            out.append(
                _Unit(
                    subtree_start,
                    own_end,
                    needs_model=own_len > max_size,
                    fold_forward=has_children,
                )
            )
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
                fold_forward=unit.fold_forward,
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


def _fold_parent_intros(
    units: list[_Unit], max_size: int, min_size: int
) -> list[_Unit]:
    """Fold a small parent-section intro forward into its first child.

    SPEC-CHUNK-632. When an overflowing parent section emits its own
    heading + lead-in prose as a unit before recursing into its child
    subsections, :func:`_plan_range` flags that unit ``fold_forward``.
    A *short* such intro (below ``min_size``) sitting before an oversized
    first child would otherwise strand as a thin standalone chunk:
    :func:`_merge_small_units` cannot absorb it forward (the oversized
    child would push the combined span past ``max_size``), and gluing it
    backward into the previous sibling is topically wrong (a different
    section). The intro is the natural lead-in to the first subsection
    and shares the parent's heading context, so it belongs with it.

    This pass merges each ``fold_forward`` intro shorter than ``min_size``
    into the immediately-following unit (its first child) by extending
    that unit's start back to the intro's start. ``needs_model`` becomes
    ``True`` when the combined span overflows ``max_size`` (so the
    fallback splitter keeps the intro with the first child's first chunk)
    or the child already needed a model. The merge target keeps its own
    ``fold_forward`` flag, so a chain of nested overflowing parents folds
    correctly down to the first leaf.

    Args:
        units: The contiguous unit tiling, after :func:`_merge_bare_headings`.
        max_size: Hard upper bound on a unit's character span.
        min_size: Floor below which a ``fold_forward`` intro is folded.

    Returns:
        A new contiguous tiling with small parent intros folded into
        their first child. Intros at or above ``min_size`` are left as
        standalone chunks; non-``fold_forward`` units are untouched.
        Covering is preserved (only adjacent spans are joined).
    """
    if min_size <= 0 or not units:
        return units
    merged: list[_Unit] = []
    pending: _Unit | None = None  # a small parent intro awaiting its child
    for unit in units:
        if pending is not None:
            unit = _Unit(
                pending.start,
                unit.end,
                pending.needs_model
                or unit.needs_model
                or (unit.end - pending.start) > max_size,
                fold_forward=unit.fold_forward,
            )
            pending = None
        if unit.fold_forward and (unit.end - unit.start) < min_size:
            pending = unit
            continue
        merged.append(unit)
    if pending is not None:
        # A fold_forward unit always precedes its first child, so this is
        # unreachable; keep the span rather than drop it if it ever isn't.
        merged.append(pending)
    return merged


def _merge_small_units(
    units: list[_Unit], max_size: int, min_size: int
) -> list[_Unit]:
    """Merge sub-``min_size`` units into a neighbor to avoid thin chunks.

    Greedy forward pass: a unit below the floor absorbs the following
    unit(s) — the next sibling/child it introduces — until it clears
    ``min_size``, as long as the combined span stays within ``max_size``.
    A thin unit that cannot grow forward (the next unit would overflow)
    falls back to gluing *backward* into its predecessor when that fits.

    The merge only ever fires to clear the floor: it stops absorbing the
    moment a unit reaches ``min_size``, so distinct sections are never
    packed together up to the cap. ``needs_model`` is OR-ed across a
    merge. Covering is preserved because units are a contiguous tiling
    and a merge only joins adjacent spans."""
    if min_size <= 0 or not units:
        return units

    def _glue_back(merged: list[_Unit], unit: _Unit) -> None:
        """Append ``unit``, gluing it backward if it is thin and fits."""
        if (
            merged
            and (unit.end - unit.start) < min_size
            and (unit.end - merged[-1].start) <= max_size
        ):
            prev = merged[-1]
            merged[-1] = _Unit(
                prev.start, unit.end, prev.needs_model or unit.needs_model
            )
        else:
            merged.append(unit)

    merged: list[_Unit] = []
    cur: _Unit | None = None
    for unit in units:
        if cur is None:
            cur = unit
            continue
        if (cur.end - cur.start) < min_size and (unit.end - cur.start) <= max_size:
            # Absorb forward to clear the floor.
            cur = _Unit(cur.start, unit.end, cur.needs_model or unit.needs_model)
            continue
        _glue_back(merged, cur)
        cur = unit
    if cur is not None:
        _glue_back(merged, cur)
    return merged


def plan_units(
    document: str,
    max_size: int = C.DEFAULT_MAX_SIZE_CHARS,
    *,
    min_size: int | None = None,
) -> list[_Unit]:
    """Pure structure-only plan: the output spans and which need models.

    Shared by the measurement script and the splitter so both agree on
    what "already fits" means. No SaT, no embedding.

    ``min_size`` is the chunk-size floor below which a unit is merged into
    a neighbor (see :func:`_merge_small_units`); it defaults to
    ``_MIN_CHUNK_FRACTION * max_size``."""
    if max_size <= 0:
        raise ValidationError("max_size must be positive")
    if min_size is None:
        min_size = int(_MIN_CHUNK_FRACTION * max_size)
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

    units = _merge_bare_headings(document, units, max_size)
    units = _fold_parent_intros(units, max_size, min_size)
    return _merge_small_units(units, max_size, min_size)


async def split_chunks_structure_first(
    document: str,
    embedder: ChunkletEmbedder,
    max_size: int = C.DEFAULT_MAX_SIZE_CHARS,
    *,
    min_size: int | None = None,
    segmenter: SentenceSegmenter | None = None,
) -> list[Chunk]:
    """Structure-first chunking of ``document``.

    Sections that already fit ``max_size`` are emitted directly with no
    model calls; only an overflowing section falls back to the semantic
    pipeline (``split_sentences`` → ``split_chunklets`` → ``split_chunks``).

    Units smaller than ``min_size`` (default ``_MIN_CHUNK_FRACTION *
    max_size``) are merged into a neighbor to avoid thin chunks.

    ``embedder`` and ``segmenter`` are only used on the fallback path.
    Returns :class:`Chunk` objects with ``start`` / ``end`` offsets into
    ``document`` and a populated ``heading_path`` (same scan as the
    default pipeline). Satisfies the covering and round-trip invariants.
    """
    if max_size <= 0:
        raise ValidationError("max_size must be positive")
    if not document:
        return []

    with get_tracer().start_as_current_span(
        "fancychunk.split_chunks_structure_first"
    ) as span:
        span.set_attribute("fancychunk.document.chars", len(document))
        span.set_attribute("fancychunk.max_size", max_size)

        units = plan_units(document, max_size, min_size=min_size)

        units_direct = units_fallback = 0
        chars_direct = chars_fallback = 0

        bare: list[Chunk] = []
        for unit in units:
            span_text = document[unit.start : unit.end]
            if not unit.needs_model:
                units_direct += 1
                chars_direct += len(span_text)
                bare.append(
                    Chunk(text=span_text, start=unit.start, end=unit.end)
                )
                continue

            # Fallback: run the semantic split on this span alone, then
            # rebase offsets back into the full document.
            units_fallback += 1
            chars_fallback += len(span_text)
            sentences = split_sentences(
                span_text, max_len=max_size, segmenter=segmenter
            )
            chunklets = split_chunklets(sentences, max_size=max_size)
            sub_chunks = await split_chunks(
                chunklets, embedder, max_size=max_size
            )
            for sc in sub_chunks:
                bare.append(
                    Chunk(
                        text=sc.text,
                        start=unit.start + (sc.start or 0),
                        end=unit.start + (sc.end or len(sc.text)),
                    )
                )

        span.set_attribute("fancychunk.units.direct", units_direct)
        span.set_attribute("fancychunk.units.fallback", units_fallback)
        span.set_attribute("fancychunk.chars.direct", chars_direct)
        span.set_attribute("fancychunk.chars.fallback", chars_fallback)

        # Populate heading_path with the same scan the default pipeline uses.
        from .headings import heading_paths

        paths = heading_paths(bare)
        return [replace(c, heading_path=p) for c, p in zip(bare, paths)]
