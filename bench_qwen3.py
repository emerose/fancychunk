"""Benchmark Qwen3-Embedding-8B (mxfp8) end-to-end through fancychunk's
late-chunking helper on an M2 / 24 GB MacBook Air.

Run with:  PYENV_VERSION=3.12.8 python bench_qwen3.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Make the reference adapter importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent / "examples" / "embedders"))

from qwen3_mlx import Qwen3MLXEmbedder  # noqa: E402

from fancychunk import embed_with_late_chunking, split_sentences  # noqa: E402

MODEL_ID = "mlx-community/Qwen3-Embedding-8B-mxfp8"


def main() -> None:
    print(f"loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    embedder = Qwen3MLXEmbedder(MODEL_ID)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    seed = """# Sorting Algorithms

Sorting algorithms put elements of a list in order. Efficient sorting is
important for optimizing other algorithms (such as search and merge
algorithms) that require input data to be in sorted lists.

## Quicksort

Quicksort is a divide-and-conquer algorithm. It works by selecting a
pivot element and partitioning the array around it.

Random pivots give expected O(n log n) performance.
Worst-case is still O(n^2) for adversarial inputs.

## Merge Sort

Merge sort is also divide-and-conquer. It divides the input into halves,
sorts them recursively, and merges them back.

Performance is O(n log n) worst-case, but requires O(n) auxiliary space.
"""
    doc = seed * 8

    print(f"\ndocument: {len(doc)} chars")

    sents = split_sentences(doc, max_len=2048)
    print(f"sentences: {len(sents)}")
    total_tokens = sum(embedder.count_tokens(sents))
    print(f"  total tokens (isolated): {total_tokens}")

    print("\nwarming up ...")
    _ = embedder.embed_segment(["warmup text here"])
    mx.eval(mx.array(0))

    print("\nrunning late chunking ...")
    times: list[float] = []
    for trial in range(3):
        gc.collect()
        t0 = time.perf_counter()
        out = embed_with_late_chunking(sents, embedder)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  trial {trial + 1}: {dt:.2f}s  ({total_tokens / dt:,.0f} tok/s)")

    print(f"\noutput shape: {out.shape}")
    print(f"output norms: {np.linalg.norm(out, axis=1).round(4)[:6]} ...")
    print(f"\nbest time: {min(times):.2f}s")
    print(f"best throughput: {total_tokens / min(times):,.0f} tok/s")


if __name__ == "__main__":
    main()
