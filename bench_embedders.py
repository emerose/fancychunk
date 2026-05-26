"""Benchmark candidate embedding models for the fancychunk "opinionated
default" decision.

Compares Qwen3-Embedding-0.6B/-4B against same-tier MTEB performers.
All models loaded via HuggingFace transformers for fairness (MLX
ecosystem only ships some of them); fp16 / bfloat16 on MPS where
available, else CPU.

Per model:
- Load time (one-shot)
- Forward-pass throughput on a fixed ~1.5 KB document
- Embedding dimension
- Approximate resident memory after load
- Published MTEB Multilingual score (cited, not measured)

Run:  PYENV_VERSION=3.12.8 python bench_embedders.py
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch  # type: ignore[import-untyped]
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Candidates.
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    hf_id: str
    params_m: int            # millions of parameters
    mteb_multi: float | None  # published MTEB Multilingual (Mean Task)
    mteb_eng: float | None   # published MTEB English v2
    notes: str = ""


CANDIDATES = [
    Candidate(
        name="Qwen3-Embedding-0.6B",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        params_m=596,
        mteb_multi=64.33,
        mteb_eng=70.70,
        notes="Qwen team's small embedder",
    ),
    Candidate(
        name="BGE-M3",
        hf_id="BAAI/bge-m3",
        params_m=568,
        mteb_multi=59.5,
        mteb_eng=63.5,
        notes="BAAI multilingual; dense+sparse+multi-vector",
    ),
    Candidate(
        name="multilingual-e5-large",
        hf_id="intfloat/multilingual-e5-large",
        params_m=560,
        mteb_multi=58.0,
        mteb_eng=63.5,
        notes="Microsoft multilingual; common baseline",
    ),
    Candidate(
        name="Qwen3-Embedding-4B",
        hf_id="Qwen/Qwen3-Embedding-4B",
        params_m=3600,
        mteb_multi=69.45,
        mteb_eng=74.60,
        notes="Qwen team's 4B tier",
    ),
]


# ---------------------------------------------------------------------------
# Test document.
# ---------------------------------------------------------------------------


DOC = (
    "Quicksort is a divide-and-conquer sorting algorithm.\n"
    "It works by selecting a pivot element from the array.\n"
    "Then it partitions the other elements into two sub-arrays — "
    "those less than the pivot, and those greater.\n"
    "The sub-arrays are then sorted recursively.\n"
    "\n"
    "Random pivots give expected O(n log n) performance.\n"
    "Worst-case is O(n^2) for adversarial inputs.\n"
    "In practice quicksort is often the fastest comparison sort.\n"
    "\n"
    "Merge sort, by contrast, has guaranteed O(n log n) worst-case "
    "complexity but uses O(n) auxiliary memory.\n"
    "Heap sort runs in-place at O(n log n) but with higher constant "
    "factors and worse cache behavior.\n"
)


# ---------------------------------------------------------------------------
# Per-model benchmark.
# ---------------------------------------------------------------------------


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resident_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1024 / 1024


@dataclass
class Result:
    candidate: Candidate
    load_time_s: float
    embed_dim: int
    n_tokens: int
    forward_ms_mean: float
    forward_ms_p95: float
    resident_after_load_mb: float
    resident_after_inference_mb: float


def bench(candidate: Candidate, trials: int = 5) -> Result | None:
    print(f"\n  {candidate.name}  ({candidate.hf_id})")
    device = _pick_device()
    print(f"    device: {device}")

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    mem_before = _resident_mb()

    t0 = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(candidate.hf_id)
        model = AutoModel.from_pretrained(
            candidate.hf_id, torch_dtype=torch.float16
        )
        model.eval()
        if device != "cpu":
            model = model.to(device)
    except Exception as e:
        print(f"    SKIP: load failed: {e}")
        return None
    load_time = time.perf_counter() - t0
    mem_after_load = _resident_mb()
    print(f"    loaded in {load_time:.1f}s  (+{mem_after_load - mem_before:,.0f} MB)")

    # Forward-pass benchmark.
    ids = tokenizer(DOC, return_tensors="pt", truncation=False)
    n_tokens = int(ids["input_ids"].shape[1])
    if device != "cpu":
        ids = {k: v.to(device) for k, v in ids.items()}

    # Warmup.
    with torch.no_grad():
        out = model(**ids)
    if device == "mps":
        torch.mps.synchronize()
    embed_dim = int(out.last_hidden_state.shape[-1])

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**ids)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    print(
        f"    forward pass: mean {mean_ms:.1f} ms / p95 {p95_ms:.1f} ms"
        f"  ({n_tokens / (mean_ms / 1000):,.0f} tok/s)"
    )

    mem_after_inference = _resident_mb()
    print(f"    resident after inference: {mem_after_inference:,.0f} MB")

    return Result(
        candidate=candidate,
        load_time_s=load_time,
        embed_dim=embed_dim,
        n_tokens=n_tokens,
        forward_ms_mean=mean_ms,
        forward_ms_p95=p95_ms,
        resident_after_load_mb=mem_after_load,
        resident_after_inference_mb=mem_after_inference,
    )


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def summary(results: list[Result]) -> None:
    print()
    print("=" * 110)
    print("  Summary")
    print("=" * 110)
    header = (
        f"  {'model':<26} {'params':>8} {'dim':>5} {'tokens':>7}"
        f" {'mean':>9} {'p95':>9} {'tok/s':>8}"
        f" {'MTEB-Mu':>8} {'MTEB-En':>8} {'RAM':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        c = r.candidate
        mteb_m = f"{c.mteb_multi:.2f}" if c.mteb_multi is not None else "—"
        mteb_e = f"{c.mteb_eng:.2f}" if c.mteb_eng is not None else "—"
        tok_per_s = r.n_tokens / (r.forward_ms_mean / 1000)
        ram_gb = r.resident_after_inference_mb / 1024
        print(
            f"  {c.name:<26} {c.params_m:>6}M  {r.embed_dim:>5} {r.n_tokens:>7}"
            f" {r.forward_ms_mean:>7.1f}ms {r.forward_ms_p95:>7.1f}ms"
            f" {tok_per_s:>7,.0f} {mteb_m:>8} {mteb_e:>8} {ram_gb:>6.1f}GB"
        )


def main() -> None:
    print(f"device pick: {_pick_device()}   "
          f"torch: {torch.__version__}  initial RSS: {_resident_mb():.0f} MB")
    results: list[Result] = []
    for c in CANDIDATES:
        r = bench(c)
        if r is not None:
            results.append(r)
        # Aggressively free memory between models — 4B+8B in 24 GB
        # is tight without this.
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    summary(results)


if __name__ == "__main__":
    main()
