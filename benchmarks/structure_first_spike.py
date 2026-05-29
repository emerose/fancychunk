"""Spike harness: structure-first vs. the default semantic pipeline.

Two subcommands:

* ``measure`` — pure structure-only pass over Qasper. Reports the
  fraction of total characters that live in heading-delimited sections
  already ``<= max_size``. That fraction is the rough upper bound on the
  latency win (those sections skip SaT + the embedder entirely). No
  models are loaded.

* ``compare`` — runs BOTH pipelines over a sample of papers and reports
  wall-clock latency, how often the slow models were skipped, chunk
  counts, the chunk-size distribution, and boundary quality (do chunks
  start at headings? any heading mid-chunk? covering / round-trip OK?).

Corpus: ``NomaDamas/qasper`` (``train``). Every section heading is
rendered flat as ``##`` with the hierarchy encoded as a ``:::`` path in
the heading text (see ``structure_first`` for how levels are recovered).

Usage:
    uv run --with datasets python -m benchmarks.structure_first_spike measure
    uv run --with datasets python -m benchmarks.structure_first_spike measure --num-papers 200
    uv run --with datasets python -m benchmarks.structure_first_spike compare \
        --paper 1908.05925 --paper 1909.13375 --paper 1910.07601 --embedder qwen3
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from ._qasper import _reconstruct_markdown

MAX_SIZE = 2048


# ---------------------------------------------------------------------------
# Corpus loading (NomaDamas/qasper, reusing the shared MD reconstruction)
# ---------------------------------------------------------------------------


def load_papers(
    num_papers: int | None,
    only_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(paper_id, markdown), ...]`` from NomaDamas/qasper train."""
    from datasets import load_dataset

    ds = load_dataset("NomaDamas/qasper", split="train")
    out: list[tuple[str, str]] = []
    for row in ds:
        pid = str(row.get("id", ""))
        if only_ids is not None and pid not in only_ids:
            continue
        md = _reconstruct_markdown(row)
        if not md.strip():
            continue
        out.append((pid, md))
        if only_ids is None and num_papers is not None and len(out) >= num_papers:
            break
        if only_ids is not None and len(out) >= len(only_ids):
            break
    return out


# ---------------------------------------------------------------------------
# measure — structure-only fit fraction
# ---------------------------------------------------------------------------


def cmd_measure(args: argparse.Namespace) -> None:
    from fancychunk.structure_first import plan_units

    papers = load_papers(args.num_papers)
    print(f"loaded {len(papers)} papers from NomaDamas/qasper train\n")

    total_chars = 0
    direct_chars = 0
    fallback_chars = 0
    per_paper_direct: list[float] = []
    n_units = 0
    n_fallback_units = 0
    docs_no_headings = 0
    docs_fully_direct = 0

    for _pid, md in papers:
        units = plan_units(md, MAX_SIZE)
        doc_total = len(md)
        doc_direct = sum(u.end - u.start for u in units if not u.needs_model)
        doc_fallback = doc_total - doc_direct
        total_chars += doc_total
        direct_chars += doc_direct
        fallback_chars += doc_fallback
        n_units += len(units)
        fb = sum(1 for u in units if u.needs_model)
        n_fallback_units += fb
        per_paper_direct.append(doc_direct / doc_total if doc_total else 1.0)
        if fb == 0:
            docs_fully_direct += 1
        # A doc with a single fallback unit covering everything = no usable headings.
        if len(units) == 1 and units[0].needs_model:
            docs_no_headings += 1

    print(f"{'corpus char fraction in already-fitting sections':<52}"
          f"{direct_chars / total_chars:>8.1%}")
    print(f"{'corpus char fraction needing the slow split':<52}"
          f"{fallback_chars / total_chars:>8.1%}")
    print()
    print(f"{'mean per-paper direct fraction':<52}"
          f"{statistics.fmean(per_paper_direct):>8.1%}")
    print(f"{'median per-paper direct fraction':<52}"
          f"{statistics.median(per_paper_direct):>8.1%}")
    print()
    print(f"{'total planned units':<52}{n_units:>8d}")
    print(f"{'units needing the slow split':<52}{n_fallback_units:>8d}"
          f"  ({n_fallback_units / n_units:.1%})")
    print(f"{'papers fully direct (no model needed at all)':<52}"
          f"{docs_fully_direct:>8d}  ({docs_fully_direct / len(papers):.1%})")
    print(f"{'papers with no usable headings (all fallback)':<52}"
          f"{docs_no_headings:>8d}  ({docs_no_headings / len(papers):.1%})")


# ---------------------------------------------------------------------------
# histogram — min-size merge off vs on (structure-only, no models)
# ---------------------------------------------------------------------------


def _size_buckets(sizes: list[int], max_size: int) -> list[tuple[str, int]]:
    edges = [0, 200, 400, 700, 1000, 1500, max_size + 1]
    labels = [
        f"{lo:>4}-{hi - 1:<4}" for lo, hi in zip(edges, edges[1:])
    ]
    counts = [0] * (len(edges) - 1)
    for s in sizes:
        for k in range(len(edges) - 1):
            if edges[k] <= s < edges[k + 1]:
                counts[k] += 1
                break
    return list(zip(labels, counts))


def cmd_histogram(args: argparse.Namespace) -> None:
    """Before/after chunk-size histogram for the min-size merge.

    Uses ``plan_units`` only (no SaT, no embedder). Thin chunks come from
    *direct* (already-fitting) units, which ``plan_units`` determines
    fully, so unit-span lengths are a faithful proxy for the chunk-size
    distribution where fragmentation lives."""
    from fancychunk.structure_first import plan_units

    only = set(args.paper) if args.paper else None
    papers = load_papers(args.num_papers if not only else None, only_ids=only)
    floor = int(0.35 * MAX_SIZE) if args.min_size is None else args.min_size
    print(f"histogram over {len(papers)} paper(s); max_size={MAX_SIZE}; "
          f"min_size floor (after) = {floor}\n")

    for pid, md in papers:
        before = [u.end - u.start for u in plan_units(md, MAX_SIZE, min_size=0)]
        after = [
            u.end - u.start for u in plan_units(md, MAX_SIZE, min_size=floor)
        ]
        thin_before = sum(1 for s in before if s < floor)
        thin_after = sum(1 for s in after if s < floor)
        print(f"=== {pid}  ({len(md)} chars) ===")
        print(f"  units            before {len(before):>3d}   after {len(after):>3d}")
        print(f"  under floor({floor:>4}) before {thin_before:>3d}   after {thin_after:>3d}")
        bb = dict(_size_buckets(before, MAX_SIZE))
        ab = dict(_size_buckets(after, MAX_SIZE))
        print(f"  {'size bucket':<12}{'before':>8}{'after':>8}")
        for label in bb:
            print(f"  {label:<12}{bb[label]:>8d}{ab[label]:>8d}")
        print()


# ---------------------------------------------------------------------------
# compare — both pipelines head to head
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    name: str
    latency_ms: float
    n_chunks: int
    sizes: list[int]
    chunks: list  # list[Chunk]
    skipped_models: bool
    direct_fraction: float | None = None


def _make_embedder(kind: str):
    from fancychunk import embedders

    if kind == "noop":
        return embedders.noop()
    if kind == "qwen3":
        return embedders.qwen3_600m()
    raise SystemExit(f"unknown embedder {kind!r} (use noop|qwen3)")


async def _run_current(md: str, embedder, max_size: int) -> RunResult:
    from fancychunk import split_chunklets, split_chunks, split_sentences

    t0 = time.perf_counter()
    sentences = split_sentences(md, max_len=max_size)
    chunklets = split_chunklets(sentences, max_size=max_size)
    chunks = await split_chunks(chunklets, embedder, max_size=max_size)
    dt = (time.perf_counter() - t0) * 1000
    return RunResult(
        name="current",
        latency_ms=dt,
        n_chunks=len(chunks),
        sizes=[len(c.text) for c in chunks],
        chunks=chunks,
        skipped_models=False,
    )


async def _run_structure_first(md: str, embedder, max_size: int) -> RunResult:
    from fancychunk.structure_first import (
        StructureFirstStats,
        split_chunks_structure_first,
    )

    stats = StructureFirstStats()
    t0 = time.perf_counter()
    chunks = await split_chunks_structure_first(
        md, embedder, max_size=max_size, stats=stats
    )
    dt = (time.perf_counter() - t0) * 1000
    return RunResult(
        name="structure-first",
        latency_ms=dt,
        n_chunks=len(chunks),
        sizes=[len(c.text) for c in chunks],
        chunks=chunks,
        skipped_models=stats.units_fallback == 0,
        direct_fraction=stats.direct_fraction,
    )


def _heading_at_start(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped.lstrip().startswith("#") and _first_line_is_heading(stripped)


def _first_line_is_heading(text: str) -> bool:
    import re

    return re.match(r"#{1,6}(\s|$)", text.splitlines()[0]) is not None if text else False


def _mid_chunk_headings(text: str) -> int:
    """Count heading lines that are NOT the chunk's first non-empty line."""
    import re

    lines = text.split("\n")
    count = 0
    seen_content = False
    for ln in lines:
        is_h = re.match(r"#{1,6}(\s|$)", ln) is not None
        if ln.strip():
            if is_h and seen_content:
                count += 1
            seen_content = True
    return count


def _boundary_report(name: str, res: RunResult, source: str, max_size: int) -> None:
    sizes = res.sizes
    roundtrip = "".join(c.text for c in res.chunks) == source
    covering = all(s <= max_size for s in sizes)
    heads_at_start = sum(1 for c in res.chunks if _heading_at_start(c.text))
    mid_headings = sum(_mid_chunk_headings(c.text) for c in res.chunks)
    tiny = sum(1 for s in sizes if s < 64)
    print(f"  [{name}]")
    print(f"    latency           {res.latency_ms:8.1f} ms"
          + (f"   (skipped models: {res.skipped_models})"
             if name == "structure-first" else ""))
    if res.direct_fraction is not None:
        print(f"    direct fraction   {res.direct_fraction:8.1%}  (chars emitted w/o models)")
    print(f"    chunks            {res.n_chunks:8d}")
    if sizes:
        print(f"    size  min/med/max {min(sizes):6d} /{int(statistics.median(sizes)):6d} /{max(sizes):6d}")
        print(f"    size  mean        {statistics.fmean(sizes):8.0f}")
    print(f"    heading@start     {heads_at_start:8d} / {res.n_chunks}")
    print(f"    heading mid-chunk {mid_headings:8d}")
    print(f"    tiny (<64 chars)  {tiny:8d}")
    print(f"    covering (<=max)  {str(covering):>8}")
    print(f"    round-trip OK     {str(roundtrip):>8}")


def cmd_compare(args: argparse.Namespace) -> None:
    only = set(args.paper) if args.paper else None
    papers = load_papers(args.num_papers if not only else None, only_ids=only)
    if only:
        found = {pid for pid, _ in papers}
        missing = only - found
        if missing:
            print(f"WARNING: paper ids not found: {sorted(missing)}")
    print(f"comparing on {len(papers)} paper(s); embedder={args.embedder}; "
          f"max_size={MAX_SIZE}\n")

    embedder = _make_embedder(args.embedder)

    async def _go() -> None:
        # Warm up the embedder once (model load not charged to per-paper times).
        if papers:
            warm = papers[0][1][:4000]
            await _run_current(warm, embedder, MAX_SIZE)
            await _run_structure_first(warm, embedder, MAX_SIZE)

        agg = {"current": [], "structure-first": []}
        for pid, md in papers:
            print(f"=== {pid}  ({len(md)} chars) ===")
            cur = await _run_current(md, embedder, MAX_SIZE)
            sf = await _run_structure_first(md, embedder, MAX_SIZE)
            _boundary_report("current", cur, md, MAX_SIZE)
            _boundary_report("structure-first", sf, md, MAX_SIZE)
            speedup = cur.latency_ms / sf.latency_ms if sf.latency_ms else float("inf")
            print(f"    -> structure-first speedup: {speedup:.1f}x\n")
            agg["current"].append(cur)
            agg["structure-first"].append(sf)

        if len(papers) > 1:
            print("=== aggregate ===")
            for name in ("current", "structure-first"):
                rs = agg[name]
                tot = sum(r.latency_ms for r in rs)
                chunks = sum(r.n_chunks for r in rs)
                print(f"  {name:<16} total {tot:8.1f} ms   chunks {chunks}")
            ct = sum(r.latency_ms for r in agg["current"])
            st = sum(r.latency_ms for r in agg["structure-first"])
            print(f"  overall speedup: {ct / st:.1f}x" if st else "")

    asyncio.run(_go())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="structure-only fit fraction")
    m.add_argument("--num-papers", type=int, default=200)
    m.set_defaults(func=cmd_measure)

    h = sub.add_parser("histogram", help="min-size merge off vs on (no models)")
    h.add_argument("--paper", action="append", help="restrict to these paper id(s)")
    h.add_argument("--num-papers", type=int, default=10)
    h.add_argument("--min-size", type=int, default=None, help="floor (default 0.35*max)")
    h.set_defaults(func=cmd_histogram)

    c = sub.add_parser("compare", help="run both pipelines head to head")
    c.add_argument("--paper", action="append", help="restrict to these paper id(s)")
    c.add_argument("--num-papers", type=int, default=10)
    c.add_argument("--embedder", choices=["noop", "qwen3"], default="qwen3")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
