"""Bundled embedders for fancychunk.

Two parallel sets of factory functions:

**Model-named factories** pin a specific model regardless of backend.
Use these when reproducibility across machines matters.

* :func:`bge_m3` — BGE-M3 (CLS pooling). 568M params, native 1024-dim.
* :func:`qwen3_600m` — Qwen3-Embedding-0.6B (last-token pooling).
  596M params, native 1024-dim. The MTEB leader at sub-1B.
* :func:`qwen3_4b` — Qwen3-Embedding-4B. 3.6B params, native 2560-dim.
  Pass ``dim=N`` to truncate to N leading dims via Matryoshka
  Representation Learning (MRL).
* :func:`qwen3_8b` — Qwen3-Embedding-8B. 7.6B params, native 4096-dim.
  Pass ``dim=N`` for MRL truncation.

**Tier-named factories** pick a hardware-appropriate model:

* :func:`default` — the recommended default. ``qwen3_600m()`` on MLX,
  ``qwen3_8b(dim=1024)`` on torch (CUDA / MPS / CPU).
* :func:`fastest` — throughput king. ``qwen3_600m()`` on MLX (the
  Qwen3-mxfp8 build outruns BGE-M3 on Apple Silicon), ``bge_m3()``
  everywhere else.
* :func:`fast` — best quality at sub-1B params. Always ``qwen3_600m()``.
* :func:`medium` — alias for ``qwen3_4b``.
* :func:`high` — alias for ``qwen3_8b``.

**Constant-output factory** for no-model-download splits:

* :func:`noop` — :class:`NoopSegmentEmbedder`. Returns constant per-
  chunklet vectors. ``split_chunks(chunklets, embedder=noop())``
  collapses the semantic-similarity term and leaves heading-aware
  structural boundaries as the only split signal.

All factories return objects that:

* Implement the :class:`fancychunk.SegmentEmbedder` protocol for
  ``embed_with_late_chunking`` (except ``noop()``, which technically
  satisfies the protocol but produces meaningless contextual
  embeddings).
* Provide ``embed_chunklets()`` for pooled per-chunklet embeddings
  ready to pass to :func:`split_chunks`.
* Auto-select an MLX backend on Apple Silicon when ``mlx_embeddings``
  is installed; fall back to torch everywhere else. BGE-M3 also has
  an MLX build; the Qwen3 family uses 8-bit microscaling (mxfp8).

Model weights download lazily on first call (Hugging Face cache).
Pre-warm in your build by calling ``embedder.embed_chunklets(["warmup"])``.

**All factory functions return process-wide singletons** (via
``functools.cache``). Calling ``embedders.default()`` twice — once
inside :func:`split_chunks` and once at the call site — returns the
*same* :class:`PooledSegmentEmbedder` instance, so the underlying
model weights load exactly once. Factory args participate in the
cache key, so ``qwen3_4b(dim=1024)`` and ``qwen3_4b(dim=2560)`` are
distinct entries. Call :func:`clear_cache` to drop *all* cached
singletons (or a single factory's ``.cache_clear()`` for
fine-grained control — but note that ``default()`` and
``fastest()`` dispatch to model-named factories whose caches are
separate, so use :func:`clear_cache` if you really want to free
VRAM in long-running services).
"""

from __future__ import annotations

import sys
from functools import cache

from ._noop import NoopSegmentEmbedder
from ._pooled import PooledSegmentEmbedder


def _mlx_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError:
        return False
    return True


def _pick_model_id(canonical: str, mlx_id: str) -> str:
    """Pick an MLX-community build on Apple Silicon (when mlx_embeddings
    is installed), else the canonical HuggingFace build."""
    return mlx_id if _mlx_available() else canonical


# ---------------------------------------------------------------------------
# Model-named factories — pin a specific model regardless of backend.
# ---------------------------------------------------------------------------


@cache
def bge_m3() -> PooledSegmentEmbedder:
    """BGE-M3 (CLS pooling).

    ~568M parameters, native 1024-dim, 8K context. MTEB Multilingual
    ~59.5 / English v2 ~63.5. Throughput king on torch (CUDA / MPS /
    CPU); on MLX, Qwen3-0.6B-mxfp8 is faster.
    """
    return PooledSegmentEmbedder(
        model_id=_pick_model_id("BAAI/bge-m3", "mlx-community/bge-m3-mlx-fp16"),
        pooling="cls",
    )


@cache
def qwen3_600m() -> PooledSegmentEmbedder:
    """Qwen3-Embedding-0.6B (last-token pooling).

    ~596M parameters, native 1024-dim, 32K context. MTEB Multilingual
    64.33 / English v2 70.70 — the strongest of the common
    600M-tier models. On MLX (mxfp8) it's both the quality and
    throughput leader at this size class.
    """
    return PooledSegmentEmbedder(
        model_id=_pick_model_id(
            "Qwen/Qwen3-Embedding-0.6B",
            "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
        ),
        pooling="last_token",
    )


@cache
def qwen3_4b(dim: int | None = None) -> PooledSegmentEmbedder:
    """Qwen3-Embedding-4B (last-token pooling).

    ~3.6B parameters, native 2560-dim. MTEB Multilingual 69.45 — about
    5 points above :func:`qwen3_600m`. ~5× slower per forward pass;
    ~4 GB resident on the MLX-mxfp8 build.

    ``dim=None`` (default) returns native 2560-dim. Pass ``dim=N`` to
    truncate to N leading dims via Matryoshka Representation Learning
    and re-L2-normalize; e.g. ``dim=1024`` for storage compatibility
    with :func:`qwen3_600m` / :func:`bge_m3`.
    """
    if dim is not None and (dim < 64 or dim > 2560):
        raise ValueError(
            f"dim must be in [64, 2560] for Qwen3-Embedding-4B; got {dim}"
        )
    return PooledSegmentEmbedder(
        model_id=_pick_model_id(
            "Qwen/Qwen3-Embedding-4B",
            "mlx-community/Qwen3-Embedding-4B-mxfp8",
        ),
        pooling="last_token",
        output_dim=dim,
    )


@cache
def qwen3_8b(dim: int | None = None) -> PooledSegmentEmbedder:
    """Qwen3-Embedding-8B (last-token pooling).

    ~7.6B parameters, native 4096-dim. MTEB Multilingual 70.58 /
    English v2 75.22 — #1 on the multilingual leaderboard at release.
    Tight on a 24 GB Mac (~7-8 GB resident on MLX-mxfp8); throughput
    is the limiting factor.

    ``dim=None`` (default) returns native 4096-dim. Pass ``dim=N`` for
    MRL truncation; ``dim=1024`` for storage compatibility with the
    smaller bundled embedders.
    """
    if dim is not None and (dim < 64 or dim > 4096):
        raise ValueError(
            f"dim must be in [64, 4096] for Qwen3-Embedding-8B; got {dim}"
        )
    return PooledSegmentEmbedder(
        model_id=_pick_model_id(
            "Qwen/Qwen3-Embedding-8B",
            "mlx-community/Qwen3-Embedding-8B-mxfp8",
        ),
        pooling="last_token",
        output_dim=dim,
    )


# ---------------------------------------------------------------------------
# Tier-named factories — pick a hardware-appropriate model.
# ---------------------------------------------------------------------------


@cache
def default() -> PooledSegmentEmbedder:
    """Hardware-aware recommended default.

    * On MLX (Apple Silicon with ``mlx_embeddings`` installed): returns
      :func:`qwen3_600m` at native 1024-dim — fast enough to keep
      interactive workflows snappy and the MTEB leader at the size
      class.
    * On torch (CUDA / MPS / CPU): returns :func:`qwen3_8b` with MRL
      truncation to 1024-dim — a discrete GPU swallows the per-pass
      cost (~44 ms on an RTX 3090) and the quality gain over the
      600M tier is substantial.

    Either way, the output is 1024-dim and L2-normalized.
    """
    if _mlx_available():
        return qwen3_600m()
    return qwen3_8b(dim=1024)


@cache
def fastest() -> PooledSegmentEmbedder:
    """Throughput king for the current backend.

    * On MLX: :func:`qwen3_600m` — Qwen3-0.6B-mxfp8 outruns BGE-M3 at
      fp16 on Apple Silicon.
    * On torch: :func:`bge_m3` — the encoder-vs-decoder gap reasserts
      itself; on an RTX 3090 BGE-M3 is ~2.3× faster than
      Qwen3-Embedding-0.6B.
    """
    if _mlx_available():
        return qwen3_600m()
    return bge_m3()


@cache
def fast() -> PooledSegmentEmbedder:
    """Best quality at sub-1B params. Always :func:`qwen3_600m`."""
    return qwen3_600m()


@cache
def medium(dim: int | None = None) -> PooledSegmentEmbedder:
    """Alias for :func:`qwen3_4b`. ``dim=None`` returns native 2560-dim."""
    return qwen3_4b(dim=dim)


@cache
def high(dim: int | None = None) -> PooledSegmentEmbedder:
    """Alias for :func:`qwen3_8b`. ``dim=None`` returns native 4096-dim."""
    return qwen3_8b(dim=dim)


# ---------------------------------------------------------------------------
# Constant-output factory.
# ---------------------------------------------------------------------------


@cache
def noop() -> NoopSegmentEmbedder:
    """Constant-output embedder for no-model-download splits.

    Returns identical unit vectors for every chunklet. With
    ``split_chunks(chunklets, embedder=noop())`` the semantic-
    similarity term collapses to a constant and only heading-aware
    structural boundaries shape where splits land — the same
    behavior the previous "no embeddings supplied" path produced.

    Late chunking with ``noop()`` is technically permitted (the
    protocol is satisfied) but produces meaningless contextual
    embeddings; the intended use is structural-only chunking.
    """
    return NoopSegmentEmbedder()


# ---------------------------------------------------------------------------
# Cache management.
# ---------------------------------------------------------------------------


def clear_cache() -> None:
    """Drop every cached embedder singleton.

    Frees the VRAM / RAM held by loaded models, at the cost of
    triggering a reload on the next factory call. Useful for tests
    that need fresh instances and for long-running services that
    want to reclaim model memory between bursts of work.
    """
    for fn in (
        bge_m3,
        qwen3_600m,
        qwen3_4b,
        qwen3_8b,
        default,
        fastest,
        fast,
        medium,
        high,
        noop,
    ):
        fn.cache_clear()


__all__ = [
    "PooledSegmentEmbedder",
    "NoopSegmentEmbedder",
    # Model-named factories.
    "bge_m3",
    "qwen3_600m",
    "qwen3_4b",
    "qwen3_8b",
    # Tier-named factories.
    "default",
    "fastest",
    "fast",
    "medium",
    "high",
    # Constant-output factory.
    "noop",
    # Cache management.
    "clear_cache",
]
