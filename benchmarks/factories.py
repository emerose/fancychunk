"""Benchmark the four fancychunk.embedders factories on this machine.

Each factory is loaded once, then ``embed_chunklets`` is called over a
synthetic batch of chunklets. Measures load time, forward-pass mean +
p95, throughput, embedding dimension, and approximate resident
memory. Auto-uses MLX on Apple Silicon when available.
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import psutil

from fancychunk.embedders import fast, fastest, high, medium

FACTORIES = [
    ("fastest", fastest, "MTEB-Multi 59.5"),
    ("fast", fast, "MTEB-Multi 64.33"),
    ("medium(dim=1024)", lambda: medium(dim=1024), "MTEB-Multi 69.45"),
    ("high(dim=1024)", lambda: high(dim=1024), "MTEB-Multi 70.58"),
]

CHUNKLETS = [
    "Quicksort is a divide-and-conquer sorting algorithm.\n"
    "It selects a pivot and partitions around it.\n"
    "Random pivots give O(n log n) expected time.\n",
    "Merge sort divides the array, sorts halves recursively, then merges.\n"
    "Guaranteed O(n log n) worst case, but uses O(n) auxiliary memory.\n",
    "Heap sort builds a binary heap and extracts the maximum repeatedly.\n"
    "Runs in-place at O(n log n) but with higher constants than quicksort.\n",
]


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def main() -> None:
    print(f"  {'factory':<22} {'backend':>6} {'dim':>5} {'load':>8} {'mean':>10}"
          f" {'p95':>10} {'tok/s':>8} {'resident':>10}  {'notes'}")
    print("  " + "-" * 102)

    for name, factory, mteb_note in FACTORIES:
        gc.collect()
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        embedder = factory()
        # Force load by running an empty-ish embed call.
        _ = embedder.count_tokens(["warmup"])
        embedder.embed_chunklets(["warmup"])
        load_time = time.perf_counter() - t0
        backend = embedder._backend  # type: ignore[attr-defined]
        dim = embedder.embedding_dim

        # Token count via the model's tokenizer.
        total_tokens = sum(embedder.count_tokens(CHUNKLETS))

        # Benchmark loop.
        durations: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            embedder.embed_chunklets(CHUNKLETS)
            durations.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = float(np.mean(durations))
        p95_ms = float(np.percentile(durations, 95))
        tok_per_s = total_tokens * 1000.0 / mean_ms if mean_ms > 0 else 0.0
        rss_after = _rss_mb()

        print(
            f"  {name:<22} {backend:>6} {dim:>5} {load_time:>7.1f}s"
            f" {mean_ms:>8.1f}ms {p95_ms:>8.1f}ms {tok_per_s:>7,.0f}"
            f" {rss_after - rss_before:>8.0f}MB  {mteb_note}"
        )
        # Encourage GC of the previous model before loading the next.
        del embedder
        gc.collect()


if __name__ == "__main__":
    main()
