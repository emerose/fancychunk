"""Bundled embedders for fancychunk.

Five factories, each returning a fresh embedder instance. Pick the
one you want; pass it to ``split_chunks``,
``embed_with_late_chunking``, or ``chunk_document``.

* :func:`bge_m3` — BGE-M3 (CLS pooling). 568M params, native 1024-dim.
  Encoder; runs fastest on CUDA / MPS / CPU.
* :func:`qwen3_600m` — Qwen3-Embedding-0.6B (last-token pooling).
  596M params, native 1024-dim. MTEB Multilingual 64.33 — the
  leader at sub-1B. **Probably the right choice for most uses.**
* :func:`qwen3_4b` — Qwen3-Embedding-4B. 3.6B params, native 2560-dim.
  Pass ``dim=N`` to truncate via Matryoshka Representation Learning.
* :func:`qwen3_8b` — Qwen3-Embedding-8B. 7.6B params, native 4096-dim.
  Pass ``dim=N`` for MRL truncation. Tight on a 24 GB Mac.
* :func:`noop` — :class:`NoopSegmentEmbedder`. Returns constant
  per-chunklet vectors. Use ``split_chunks(chunklets, noop())`` for a
  no-model-download structural-only split.

Each factory returns a **fresh instance every call**; there is no
caching. Model weights download lazily on first call to
``embed_chunklets`` or ``embed_segment`` (Hugging Face cache, so
subsequent process runs hit the on-disk cache fast). Lifecycle is
the caller's responsibility — hold the embedder reference while you
need it, drop it to free memory.

The Apple Silicon MLX backend is auto-selected on macOS arm64 when
``mlx_embeddings`` is installed; the factories transparently pick
the MLX-community build of the same model.
"""

from __future__ import annotations

import sys

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
# Model-named factories.
# ---------------------------------------------------------------------------


def bge_m3() -> PooledSegmentEmbedder:
    """BGE-M3 (CLS pooling).

    ~568M parameters, native 1024-dim, 8K context. MTEB Multilingual
    ~59.5 / English v2 ~63.5. Throughput king on torch (CUDA / MPS /
    CPU); on MLX, :func:`qwen3_600m` is faster.
    """
    return PooledSegmentEmbedder(
        model_id=_pick_model_id("BAAI/bge-m3", "mlx-community/bge-m3-mlx-fp16"),
        pooling="cls",
    )


def qwen3_600m() -> PooledSegmentEmbedder:
    """Qwen3-Embedding-0.6B (last-token pooling).

    ~596M parameters, native 1024-dim, 32K context. MTEB Multilingual
    64.33 / English v2 70.70 — the strongest of the common
    600M-tier models. **Recommended default for most uses:** good
    quality, manageable resident memory (~0.5 GB on the MLX-mxfp8
    build, ~1 GB on torch), reasonable throughput.
    """
    return PooledSegmentEmbedder(
        model_id=_pick_model_id(
            "Qwen/Qwen3-Embedding-0.6B",
            "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
        ),
        pooling="last_token",
    )


def qwen3_4b(dim: int | None = None) -> PooledSegmentEmbedder:
    """Qwen3-Embedding-4B (last-token pooling).

    ~3.6B parameters, native 2560-dim. MTEB Multilingual 69.45 —
    about 5 points above :func:`qwen3_600m`. ~5× slower per forward
    pass; ~4 GB resident on the MLX-mxfp8 build.

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
# Constant-output factory.
# ---------------------------------------------------------------------------


def noop() -> NoopSegmentEmbedder:
    """Constant-output embedder for no-model-download splits.

    Returns identical unit vectors for every chunklet. With
    ``split_chunks(chunklets, noop())`` the semantic-similarity term
    collapses to a constant and only heading-aware structural
    boundaries shape where splits land — the same behavior the
    previous "no embeddings supplied" path produced.

    Late chunking with ``noop()`` is technically permitted (the
    protocol is satisfied) but produces meaningless contextual
    embeddings; the intended use is structural-only chunking.
    """
    return NoopSegmentEmbedder()


__all__ = [
    "PooledSegmentEmbedder",
    "NoopSegmentEmbedder",
    "bge_m3",
    "qwen3_600m",
    "qwen3_4b",
    "qwen3_8b",
    "noop",
]
