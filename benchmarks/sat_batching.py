"""Microbenchmark: per-document vs batched SaT sentence segmentation.

Generates a synthetic corpus of short Markdown documents (the
small-doc / many-doc workload where SaT dominates the chunking budget)
and times three paths:

  1. ``SaTSegmenter.__call__`` per document (the legacy per-doc loop)
  2. ``SaTSegmenter.predict_proba_batch`` in slices of N documents
  3. ``chunk_documents(..., segmenter_batch_size=N)`` end-to-end
     (with ``--include-e2e``)

The win from batching is highly platform-dependent:

* **GPU (CUDAExecutionProvider).** Batching amortises per-call
  launch + memory-transfer cost. The headline number for this
  feature is the end-to-end ``chunk_documents`` improvement, not the
  SaT-only ratio: on RTX 3090 + sat-3l-sm we measure ~4.9× e2e over
  CPU from ``device="cuda"`` alone and ~6.6× e2e with
  ``segmenter_batch_size=64`` on top. The SaT-only batched-vs-serial
  ratio on the same device is ~2.2× (0.67 ms/doc batched, 1.45
  ms/doc serial). ``SaTSegmenter`` swaps in a vectorised
  ``token_to_char_probs`` on first load — without it the post-
  process loop alone was ~45% of the batched wall on this hardware.
* **CPU (CPUExecutionProvider).** Forward-pass FLOPs scale linearly
  with batch size, so there is no theoretical batch win — the CPU
  just does the same work serially. Observed numbers on Apple
  Silicon are 0.9–1.0×. The API still works; it just doesn't pay
  off until you put a GPU under it.

Run::

    python -m benchmarks.sat_batching                # quick: 200 docs
    python -m benchmarks.sat_batching --n-docs 1000  # the spec's number
    python -m benchmarks.sat_batching --batch-size 64
    python -m benchmarks.sat_batching --device cuda  # if onnxruntime-gpu
    python -m benchmarks.sat_batching --include-e2e  # full chunk_documents path

The model weights (~408 MB) download lazily on first call.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time

from fancychunk import SaTSegmenter, chunk_documents
from fancychunk.embedders import noop


_SENTENCES = [
    "The cat sat on the mat.",
    "A bird flew over the house.",
    "Two cars passed silently in the night.",
    "She read a long book by the fire.",
    "Mountains rose in the distance, blue against the sky.",
    "The quick brown fox jumps over the lazy dog.",
    "Coffee brewed slowly while the morning light crept in.",
    "Ancient walls held secrets nobody alive remembered.",
    "Quantum entanglement defies our classical intuitions about locality.",
    "Distributed systems trade consistency, availability, and partition tolerance.",
]


def _make_corpus(n_docs: int, target_chars: int, seed: int) -> list[str]:
    """Generate ``n_docs`` short Markdown documents of roughly
    ``target_chars`` characters each — matches the BeIR scifact
    profile mentioned in the spec (~1,500 chars / abstract)."""
    rng = random.Random(seed)
    docs: list[str] = []
    for i in range(n_docs):
        parts: list[str] = [f"# Document {i}\n\n"]
        running = len(parts[0])
        while running < target_chars:
            sent = rng.choice(_SENTENCES) + " "
            parts.append(sent)
            running += len(sent)
            if rng.random() < 0.1:
                parts.append("\n\n")
                running += 2
        docs.append("".join(parts).rstrip() + "\n")
    return docs


def _bench_serial(seg: SaTSegmenter, docs: list[str]) -> float:
    t0 = time.perf_counter()
    for d in docs:
        if not d.strip():
            continue
        _ = seg(d)
    return time.perf_counter() - t0


def _bench_batched(
    seg: SaTSegmenter, docs: list[str], batch_size: int
) -> float:
    t0 = time.perf_counter()
    for start in range(0, len(docs), batch_size):
        _ = seg.predict_proba_batch(docs[start : start + batch_size])
    return time.perf_counter() - t0


async def _bench_chunk_documents(
    docs: list[str],
    seg: SaTSegmenter,
    batch_size: int | None,
) -> float:
    embedder = noop()
    t0 = time.perf_counter()
    await chunk_documents(
        docs, embedder, segmenter=seg, segmenter_batch_size=batch_size
    )
    return time.perf_counter() - t0


def _stats(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], 0.0
    return statistics.median(samples), statistics.stdev(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-docs", type=int, default=200)
    parser.add_argument("--doc-chars", type=int, default=1500)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Documents per SaT forward batch.",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="SaT execution device (default: auto-detect).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Wall-clock samples per measurement.",
    )
    parser.add_argument(
        "--include-e2e", action="store_true",
        help="Also time chunk_documents end-to-end (with noop embedder).",
    )
    parser.add_argument(
        "--assert-speedup", type=float, default=None,
        help=(
            "Fail (exit 1) if batched speedup is below this ratio. "
            "Use with --device cuda; CPU does not see a batch win."
        ),
    )
    args = parser.parse_args()

    docs = _make_corpus(
        n_docs=args.n_docs, target_chars=args.doc_chars, seed=args.seed
    )
    total_chars = sum(len(d) for d in docs)
    print(
        f"corpus: {len(docs)} docs, "
        f"mean {total_chars / len(docs):.0f} chars, "
        f"total {total_chars / 1_000:.1f} kchars"
    )
    print(f"device: {args.device}, batch size: {args.batch_size}")

    seg = SaTSegmenter(device=args.device)
    print(f"resolved providers: {seg.ort_providers or '<wtpsplit auto-detect>'}")

    print("warming up SaT (may download 408 MB on first run)...", flush=True)
    _ = seg.predict_proba_batch(docs[: min(8, len(docs))])

    serial_samples = [_bench_serial(seg, docs) for _ in range(args.repeats)]
    batched_samples = [
        _bench_batched(seg, docs, args.batch_size)
        for _ in range(args.repeats)
    ]

    serial_med, serial_sd = _stats(serial_samples)
    batched_med, batched_sd = _stats(batched_samples)

    print()
    print(
        f"per-doc serial:       {serial_med * 1000:8.1f} ms  "
        f"± {serial_sd * 1000:.1f}   "
        f"({serial_med / len(docs) * 1000:.2f} ms/doc)"
    )
    print(
        f"batched (n={args.batch_size:>3}):       "
        f"{batched_med * 1000:8.1f} ms  ± {batched_sd * 1000:.1f}   "
        f"({batched_med / len(docs) * 1000:.2f} ms/doc)"
    )
    speedup = serial_med / batched_med if batched_med > 0 else float("inf")
    print(f"batched speedup:      {speedup:8.2f}×")

    if args.include_e2e:
        e2e_serial = asyncio.run(_bench_chunk_documents(docs, seg, None))
        e2e_batched = asyncio.run(
            _bench_chunk_documents(docs, seg, args.batch_size)
        )
        print()
        print(f"chunk_documents (no batch):  {e2e_serial * 1000:8.1f} ms")
        print(
            f"chunk_documents (batch={args.batch_size}): "
            f"{e2e_batched * 1000:8.1f} ms"
        )
        print(
            f"chunk_documents speedup:     "
            f"{e2e_serial / e2e_batched:8.2f}×"
        )

    if args.assert_speedup is not None:
        if speedup < args.assert_speedup:
            print(
                f"\nFAIL: batched speedup {speedup:.2f}× below target "
                f"{args.assert_speedup:.2f}×",
                file=sys.stderr,
            )
            return 1
        print(
            f"\nOK: batched speedup {speedup:.2f}× meets target "
            f"{args.assert_speedup:.2f}×"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
