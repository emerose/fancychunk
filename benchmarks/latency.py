"""Latency / throughput benchmark on Qasper.

For each chunker, process N papers end-to-end and report:

* total wall time
* per-doc mean / median / p95 in ms
* MB/sec throughput (input bytes / total time)
* mean chunks per doc

For fancychunk specifically, also break down the cost per pipeline
stage using the OpenTelemetry spans the library emits natively. This
gives you a "where does the time actually go" view that the other
chunkers can't easily provide.

Usage:
    .venv/bin/python -m benchmarks.latency --num-papers 50
    .venv/bin/python -m benchmarks.latency --chunker fancychunk-late

Note: weights load on first use. For accurate timings, the script
runs a warm-up pass (3 docs, not timed) before each chunker.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

from ._chunkers import Chunker, all_chunkers
from ._qasper import QasperPaper, load_qasper


@dataclass
class StageTimings:
    """Per-stage wall times collected from OpenTelemetry spans."""

    by_stage: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, span_name: str, duration_ms: float) -> None:
        self.by_stage[span_name].append(duration_ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, samples in self.by_stage.items():
            if not samples:
                continue
            out[stage] = {
                "n": len(samples),
                "mean_ms": statistics.fmean(samples),
                "median_ms": statistics.median(samples),
                "p95_ms": _percentile(samples, 95),
                "total_ms": sum(samples),
            }
        return out


@dataclass
class ChunkerLatency:
    name: str
    per_doc_ms: list[float] = field(default_factory=list)
    chunk_counts: list[int] = field(default_factory=list)
    bytes_processed: int = 0
    stage_timings: StageTimings = field(default_factory=StageTimings)


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _install_span_capture(stage_timings: StageTimings) -> object:
    """Install an in-memory OpenTelemetry exporter that records every
    fancychunk span duration. Returns the provider for cleanup; idle
    no-op for non-fancychunk chunkers (their spans, if any, just get
    recorded too)."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Replace the global provider; OK in a benchmark script.
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    return (provider, exporter, stage_timings)


def _drain_spans(captured: object) -> None:
    """Pop accumulated spans from the in-memory exporter and route
    their durations into the stage timings."""
    provider, exporter, stage_timings = captured  # type: ignore[misc]
    provider.force_flush()
    for span in exporter.get_finished_spans():
        # Span end/start times are in nanoseconds.
        duration_ms = (span.end_time - span.start_time) / 1_000_000
        stage_timings.add(span.name, duration_ms)
    exporter.clear()


async def _run_chunker(
    chunker: Chunker, papers: list[QasperPaper], warmup: int
) -> ChunkerLatency:
    """Process all papers through one chunker; record timings."""
    from tqdm import tqdm  # type: ignore[import-untyped]

    out = ChunkerLatency(name=chunker.name)
    captured = _install_span_capture(out.stage_timings)

    # Warm-up pass (model load, JIT, etc.) — not timed.
    for paper in papers[:warmup]:
        await chunker.achunk(paper.markdown)
    _drain_spans(captured)  # discard warmup spans

    # Timed pass.
    iterator = tqdm(papers, desc=chunker.name, leave=False)
    for paper in iterator:
        t0 = time.perf_counter()
        chunks, _vecs = await chunker.achunk(paper.markdown)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out.per_doc_ms.append(elapsed_ms)
        out.chunk_counts.append(len(chunks))
        out.bytes_processed += len(paper.markdown.encode("utf-8"))

    _drain_spans(captured)
    return out


async def run_benchmark(
    chunkers: list[Chunker],
    papers: list[QasperPaper],
    warmup: int = 3,
) -> dict[str, ChunkerLatency]:
    results: dict[str, ChunkerLatency] = {}
    for chunker in chunkers:
        results[chunker.name] = await _run_chunker(chunker, papers, warmup)
    return results


def print_summary(results: dict[str, ChunkerLatency]) -> None:
    header = (
        f"{'chunker':<28} {'N':>4} {'mean_ms':>9} {'median_ms':>10} "
        f"{'p95_ms':>9} {'total_s':>9} {'MB/s':>8} {'chunks/doc':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, lat in results.items():
        if not lat.per_doc_ms:
            continue
        total_s = sum(lat.per_doc_ms) / 1000
        mb = lat.bytes_processed / (1024 * 1024)
        mb_per_s = mb / total_s if total_s > 0 else float("inf")
        chunks_per_doc = (
            sum(lat.chunk_counts) / len(lat.chunk_counts)
            if lat.chunk_counts
            else 0.0
        )
        print(
            f"{name:<28} {len(lat.per_doc_ms):>4} "
            f"{statistics.fmean(lat.per_doc_ms):>9.1f} "
            f"{statistics.median(lat.per_doc_ms):>10.1f} "
            f"{_percentile(lat.per_doc_ms, 95):>9.1f} "
            f"{total_s:>9.1f} {mb_per_s:>8.2f} {chunks_per_doc:>11.1f}"
        )


def print_stage_breakdowns(results: dict[str, ChunkerLatency]) -> None:
    """Print per-stage timings for chunkers that emit OTel spans
    (fancychunk does natively; LangChain / chonkie don't unless the
    caller wraps them)."""
    for name, lat in results.items():
        summary = lat.stage_timings.summary()
        if not summary:
            continue
        # Filter to fancychunk-emitted spans for readability.
        fc_spans = {
            k: v for k, v in summary.items() if k.startswith("fancychunk.")
        }
        if not fc_spans:
            continue
        print(f"\n--- {name} per-stage breakdown (OpenTelemetry spans) ---")
        print(f"  {'span':<48} {'N':>5} {'mean_ms':>9} {'p95_ms':>9} {'total_ms':>10}")
        for span_name, stats in sorted(
            fc_spans.items(), key=lambda kv: -kv[1]["total_ms"]
        ):
            print(
                f"  {span_name:<48} {int(stats['n']):>5} "
                f"{stats['mean_ms']:>9.2f} {stats['p95_ms']:>9.2f} "
                f"{stats['total_ms']:>10.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-papers",
        type=int,
        default=50,
        help="Number of papers to time (default: 50).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
    )
    parser.add_argument(
        "--chunker",
        action="append",
        help="Run only the named chunker(s). Default: all 6.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warmup docs per chunker (model load etc.). Default 3.",
    )
    parser.add_argument(
        "--no-stage-breakdown",
        action="store_true",
        help="Skip the OpenTelemetry per-stage breakdown table.",
    )
    args = parser.parse_args()

    print(f"loading Qasper {args.split} split (limit {args.num_papers})…")
    papers = load_qasper(split=args.split, limit=args.num_papers)
    print(f"  loaded {len(papers)} papers")

    chunkers = all_chunkers()
    if args.chunker:
        wanted = set(args.chunker)
        chunkers = [c for c in chunkers if c.name in wanted]
        if not chunkers:
            available = ", ".join(c.name for c in all_chunkers())
            raise SystemExit(
                f"no chunker matched {sorted(wanted)}; available: {available}"
            )

    print(
        f"running {len(chunkers)} chunker(s) over {len(papers)} papers "
        f"(+{args.warmup} warmup each)…\n"
    )
    results = asyncio.run(run_benchmark(chunkers, papers, warmup=args.warmup))

    print_summary(results)
    if not args.no_stage_breakdown:
        print_stage_breakdowns(results)


if __name__ == "__main__":
    main()
