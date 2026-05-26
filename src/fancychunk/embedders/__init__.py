"""Opinionated embedder defaults for users who don't want to BYO.

Four named choices, in order of increasing quality and decreasing
throughput:

* :func:`fastest` → BGE-M3 (CLS pooling). ~568M params, the speed
  king. MTEB Multi ~59.5.
* :func:`fast` → Qwen3-Embedding-0.6B (last-token pooling). The
  recommended default — best quality at the 600M tier. MTEB Multi
  64.33.
* :func:`medium` → Qwen3-Embedding-4B + Matryoshka truncation to
  1024 dim. MTEB Multi 69.45, ~5× slower than ``fast``.
* :func:`high` → Qwen3-Embedding-8B + Matryoshka truncation to 1024
  dim. MTEB Multi 70.58, the leaderboard's #1 at its release.

All four return a :class:`PooledSegmentEmbedder` that:

* Implements the :class:`fancychunk.SegmentEmbedder` protocol for
  ``embed_with_late_chunking``.
* Provides ``embed_chunklets()`` for pooled per-chunklet embeddings
  ready to pass to :func:`split_chunks`.
* Auto-selects an MLX backend on Apple Silicon when ``mlx_embeddings``
  is installed (2-4× faster than torch + MPS); falls back to torch
  everywhere else. BGE-M3 has no MLX build, so :func:`fastest` always
  uses torch.

Requires the ``[embedders]`` extra:

.. code-block:: bash

    pip install 'fancychunk[embedders]'

(On macOS this also installs ``mlx`` and ``mlx_embeddings`` for the
fast Apple Silicon path; on Linux/Windows those are skipped.)

Usage:

.. code-block:: python

    from fancychunk import split_sentences, split_chunklets, split_chunks
    from fancychunk.embedders import fast

    embedder = fast()
    sentences = split_sentences(doc, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    embeddings = embedder.embed_chunklets(chunklets)
    chunks, _ = split_chunks(chunklets, embeddings, max_size=2048)

Model weights download lazily on first call (Hugging Face cache).
Pre-warm in your build by calling ``embedder.embed_chunklets(["warmup"])``.
"""

from __future__ import annotations

import sys

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


def fastest() -> PooledSegmentEmbedder:
    """Speed-optimized: BGE-M3.

    ~568M parameters, 1024-dim embeddings, 8K context, CLS pooling.
    MTEB Multilingual ~59.5 / English v2 ~63.5. Trades ~5 MTEB-Multi
    points for the speed. On Apple Silicon the MLX-fp16 build runs
    several times faster than the torch + MPS path.
    """
    return PooledSegmentEmbedder(
        model_id=_pick_model_id("BAAI/bge-m3", "mlx-community/bge-m3-mlx-fp16"),
        pooling="cls",
    )


def fast() -> PooledSegmentEmbedder:
    """The recommended default: Qwen3-Embedding-0.6B.

    ~596M parameters, 1024-dim embeddings, 32K context, last-token
    pooling. MTEB Multilingual 64.33 / English v2 70.70 — the
    strongest of the common 600M-tier models. ~1 GB resident on
    torch + MPS; ~600 MB on the MLX-mxfp8 build with markedly faster
    inference.
    """
    return PooledSegmentEmbedder(
        model_id=_pick_model_id(
            "Qwen/Qwen3-Embedding-0.6B",
            "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
        ),
        pooling="last_token",
    )


def medium(dim: int = 1024) -> PooledSegmentEmbedder:
    """Qwen3-Embedding-4B with Matryoshka truncation.

    ~3.6B parameters, native 2560-dim embeddings truncated to ``dim``
    via Matryoshka Representation Learning. MTEB Multilingual 69.45
    (full dim) — about 5 points above :func:`fast`. ~5× slower per
    forward pass; ~4 GB resident on the MLX-mxfp8 build.

    ``dim`` defaults to 1024 to match :func:`fast`'s output width so
    you can A/B test storage-compatibly; pass ``dim=2560`` for the
    full native width.
    """
    if dim < 64 or dim > 2560:
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


def high(dim: int = 1024) -> PooledSegmentEmbedder:
    """Top of the bundled options: Qwen3-Embedding-8B with MRL.

    ~7.6B parameters, native 4096-dim embeddings truncated to ``dim``
    via Matryoshka Representation Learning. MTEB Multilingual 70.58 /
    English v2 75.22 — #1 on the multilingual leaderboard at release.
    Tight on a 24 GB Mac (~7-8 GB resident on MLX-mxfp8); throughput
    is the limiting factor. ``dim`` defaults to 1024 to match the
    smaller factories' output width; pass ``dim=4096`` for native
    width.
    """
    if dim < 64 or dim > 4096:
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


__all__ = [
    "PooledSegmentEmbedder",
    "fastest",
    "fast",
    "medium",
    "high",
]
