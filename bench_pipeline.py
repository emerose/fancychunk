"""Benchmark the chunking pipeline (sentences → chunklets → chunks) on a
variety of test documents, using OpenTelemetry span timings to attribute
work to specific phases.

Excludes ``embed_with_late_chunking`` per scope; uses the punctuation
segmenter (fast, deterministic, no model download) by default. Pass
``--use-sat`` to run against the SaT model instead — adds 408 MB
download on first run and ~10× per-call cost.

Run with:  PYENV_VERSION=3.12.8 python bench_pipeline.py
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from fancychunk import (
    punctuation_segmenter,
    split_chunklets,
    split_chunks,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Span collector: a tiny in-process exporter that just records durations.
# ---------------------------------------------------------------------------


@dataclass
class CollectedSpan:
    name: str
    duration_ns: int
    attributes: dict[str, object]


class CollectingExporter(SpanExporter):
    """Records every finished span's (name, duration_ns, attributes)."""

    def __init__(self) -> None:
        self.spans: list[CollectedSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            self.spans.append(
                CollectedSpan(
                    name=s.name,
                    duration_ns=(s.end_time or 0) - (s.start_time or 0),
                    attributes=dict(s.attributes or {}),
                )
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - protocol satisfaction
        pass


def install_collector() -> CollectingExporter:
    exporter = CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    return exporter


# ---------------------------------------------------------------------------
# Test document fixtures.
# ---------------------------------------------------------------------------


@dataclass
class Doc:
    name: str
    text: str
    notes: str = ""


def _para(words: int, seed: int) -> str:
    """Deterministic paragraph of ``words`` whitespace-separated words."""
    rng = np.random.default_rng(seed)
    tokens = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda".split()[
            int(rng.integers(0, 11))
        ]
        for _ in range(words)
    ]
    # Sprinkle sentence terminators.
    out: list[str] = []
    for i, t in enumerate(tokens):
        out.append(t)
        # Roughly one sentence every 12-18 words.
        if i and i % int(rng.integers(12, 18)) == 0:
            out[-1] = t + "."
    if out and not out[-1].endswith("."):
        out[-1] += "."
    return " ".join(out)


def make_documents() -> list[Doc]:
    docs: list[Doc] = []

    # Tiny: single short paragraph.
    docs.append(Doc("tiny_paragraph", _para(40, seed=1), "single short paragraph"))

    # Small: typical RAG chunk source (~1 KB).
    docs.append(
        Doc(
            "small_article",
            "# Quicksort\n\n"
            + _para(120, seed=2)
            + "\n\n## Pivots\n\n"
            + _para(80, seed=3)
            + "\n",
            "~1 KB with two H-level headings",
        )
    )

    # Medium: ~10 KB blog post.
    body = "# Sorting Algorithms\n\n"
    for i in range(8):
        body += f"## Section {i}\n\n{_para(180, seed=10 + i)}\n\n"
    docs.append(Doc("medium_blog", body, "~10 KB, 8 sections, mixed-length paragraphs"))

    # Large: ~100 KB long-form.
    long_body = "# Long-form Document\n\n"
    for i in range(40):
        long_body += f"## Section {i}\n\n{_para(220, seed=100 + i)}\n\n"
        long_body += _para(220, seed=200 + i) + "\n\n"
    docs.append(Doc("large_longform", long_body, "~100 KB, 40 sections"))

    # Heading-heavy: many small sections.
    heads = ""
    for i in range(60):
        heads += f"### Heading {i}\n\n{_para(15, seed=300 + i)}\n\n"
    docs.append(Doc("heading_heavy", heads, "60 small h3 sections"))

    # Lists.
    lists = "# Reference\n\n"
    for section in range(6):
        lists += f"## Item group {section}\n\n"
        for j in range(20):
            lists += f"- Bullet item {j} with a brief description.\n"
        lists += "\n"
    docs.append(Doc("list_heavy", lists, "6 sections of 20 bullets each"))

    # Code-heavy.
    code = "# API Reference\n\n"
    for fn in range(12):
        code += f"## function_{fn}\n\n{_para(40, seed=400 + fn)}\n\n"
        code += "```python\n"
        code += "def example():\n    return 42\n" * 8
        code += "```\n\n"
    docs.append(Doc("code_heavy", code, "12 functions w/ code fences"))

    # Long prose: single section, no internal headings, lots of sentences.
    docs.append(
        Doc(
            "long_prose",
            "# Essay\n\n" + "\n\n".join(_para(250, seed=500 + i) for i in range(8)) + "\n",
            "8 long paragraphs, no internal headings",
        )
    )

    return docs


# ---------------------------------------------------------------------------
# Benchmark runner.
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    doc_name: str
    spans: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))


def run_pipeline(doc: str, max_size: int = 2048) -> tuple[int, int, int]:
    """Run sentences → chunklets → chunks on ``doc``.

    Returns (n_sentences, n_chunklets, n_chunks). The pipeline uses
    synthetic random embeddings for stage 3 — we're benchmarking the
    chunking work, not embedding quality.
    """
    sents = split_sentences(doc, max_len=max_size, segmenter=punctuation_segmenter)
    chunklets = split_chunklets(sents, max_size=max_size)
    if not chunklets:
        return len(sents), 0, 0
    rng = np.random.default_rng(0)
    # Random unit-ish vectors with non-zero norm.
    emb = rng.normal(size=(len(chunklets), 16))
    emb = emb + 0.01 * np.sign(emb)
    chunks, _ = split_chunks(chunklets, emb, max_size=max_size)
    return len(sents), len(chunklets), len(chunks)


def benchmark(
    docs: Iterable[Doc], trials: int, warmup: int, max_size: int = 2048
) -> dict[str, TrialResult]:
    exporter = install_collector()
    results: dict[str, TrialResult] = {}

    for doc in docs:
        # Warmup (not recorded; helps amortize JIT / cache).
        for _ in range(warmup):
            run_pipeline(doc.text, max_size=max_size)
        exporter.spans.clear()
        gc.collect()

        n_sents = n_chunklets = n_chunks = 0
        for _ in range(trials):
            exporter.spans.clear()
            n_sents, n_chunklets, n_chunks = run_pipeline(
                doc.text, max_size=max_size
            )
            # Record this trial's spans.
            result = results.setdefault(doc.name, TrialResult(doc_name=doc.name))
            for s in exporter.spans:
                result.spans[s.name].append(s.duration_ns)
        # Bookkeeping for the summary table.
        result = results[doc.name]
        result.spans["__meta__/chars"].append(len(doc.text))
        result.spans["__meta__/sentences"].append(n_sents)
        result.spans["__meta__/chunklets"].append(n_chunklets)
        result.spans["__meta__/chunks"].append(n_chunks)

    return results


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


_STAGE_ORDER = [
    "fancychunk.split_sentences",
    "  fancychunk.sentences.segmenter",
    "  fancychunk.sentences.heading_override",
    "  fancychunk.sentences.merge",
    "  fancychunk.sentences.whitespace_trailing",
    "  fancychunk.sentences.dp",
    "  fancychunk.sentences.slice",
    "fancychunk.split_chunklets",
    "  fancychunk.chunklets.boundary_probas",
    "  fancychunk.chunklets.statement_counts",
    "  fancychunk.chunklets.dp",
    "fancychunk.split_chunks",
    "  fancychunk.chunks.partition_similarities",
    "  fancychunk.chunks.dp",
]


def _fmt_us(ns: float) -> str:
    if ns < 1000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1000:.1f}µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    return f"{ns / 1_000_000_000:.2f}s"


def report(results: dict[str, TrialResult], docs: list[Doc]) -> None:
    print()
    print("=" * 78)
    print(f"  Pipeline benchmark — {len(docs)} documents, punctuation segmenter")
    print("=" * 78)

    for doc in docs:
        result = results[doc.name]
        chars = int(result.spans["__meta__/chars"][0])
        sentences = int(result.spans["__meta__/sentences"][0])
        chunklets = int(result.spans["__meta__/chunklets"][0])
        chunks = int(result.spans["__meta__/chunks"][0])

        print()
        print(f"# {doc.name}  ({doc.notes})")
        print(
            f"  {chars:>7,} chars  →  {sentences:>4} sentences"
            f"  →  {chunklets:>3} chunklets  →  {chunks:>2} chunks"
        )

        # Aggregate per-span statistics.
        top_total = sum(result.spans.get("fancychunk.split_sentences", []))
        top_total += sum(result.spans.get("fancychunk.split_chunklets", []))
        top_total += sum(result.spans.get("fancychunk.split_chunks", []))

        print(
            f"  {'phase':<48} {'mean':>10} {'p50':>10} {'p95':>10} {'%':>7}"
        )
        for name in _STAGE_ORDER:
            key = name.strip()
            samples = result.spans.get(key, [])
            if not samples:
                continue
            mean = statistics.mean(samples)
            p50 = statistics.median(samples)
            p95 = (
                sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]
                if len(samples) >= 20
                else max(samples)
            )
            # % of total only for top-level stages and only meaningful
            # when a top_total exists.
            pct = ""
            if top_total > 0 and not name.startswith("  "):
                stage_total = sum(samples)
                pct = f"{100 * stage_total / top_total:>5.1f}%"
            print(
                f"  {name:<48} {_fmt_us(mean):>10} {_fmt_us(p50):>10} "
                f"{_fmt_us(p95):>10} {pct:>7}"
            )


def overall_summary(results: dict[str, TrialResult], trials: int) -> None:
    print()
    print("=" * 78)
    print(f"  Overall throughput summary (mean of {trials} trials per doc)")
    print("=" * 78)
    print(
        f"  {'doc':<22} {'chars':>8} {'pipeline':>12} {'throughput':>16}"
    )
    for name, result in results.items():
        chars = int(result.spans["__meta__/chars"][0])
        s1 = sum(result.spans.get("fancychunk.split_sentences", []))
        s2 = sum(result.spans.get("fancychunk.split_chunklets", []))
        s3 = sum(result.spans.get("fancychunk.split_chunks", []))
        n_trials = len(result.spans.get("fancychunk.split_sentences", [1])) or 1
        pipeline_ns = (s1 + s2 + s3) / n_trials
        chars_per_s = chars * 1e9 / pipeline_ns if pipeline_ns > 0 else 0
        print(
            f"  {name:<22} {chars:>8,} {_fmt_us(pipeline_ns):>12}"
            f" {chars_per_s / 1e6:>12,.2f} MB/s"
        )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="number of timed runs per document (default 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="warmup runs (not timed) per document (default 3)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="max_size for chunklets and chunks (default 2048)",
    )
    args = parser.parse_args()

    docs = make_documents()
    t0 = time.perf_counter()
    results = benchmark(docs, trials=args.trials, warmup=args.warmup, max_size=args.max_size)
    wall = time.perf_counter() - t0

    report(results, docs)
    overall_summary(results, args.trials)
    print(f"\ntotal wall time: {wall:.2f}s ({args.trials} trials × {len(docs)} docs)")


if __name__ == "__main__":
    main()
