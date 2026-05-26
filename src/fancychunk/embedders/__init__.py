"""Opinionated embedder defaults for users who don't want to BYO.

Three named choices covering the typical quality/speed trade-off:

* :func:`default` → Qwen3-Embedding-0.6B. Best-in-class MTEB at the
  600M parameter tier. ~1 GB resident, ~100ms per forward pass on
  Apple Silicon. **Recommended.**
* :func:`fast` → BGE-M3. ~2.5× faster than ``default`` per forward
  pass, ~5 MTEB points behind. Use when chunking latency dominates.
* :func:`high_quality` → Qwen3-Embedding-4B with Matryoshka truncation
  to 1024 dim. Highest quality available locally, ~5× slower than
  ``default``. ~7 GB resident. Use when retrieval quality is the
  binding constraint.

All three return a :class:`PooledSegmentEmbedder`, which provides:

* The :class:`fancychunk.SegmentEmbedder` protocol surface (for
  ``embed_with_late_chunking``).
* An ``embed_chunklets()`` convenience method returning pooled
  per-chunklet embeddings ready to pass to :func:`split_chunks`.

Requires the ``[embedders]`` extra:

.. code-block:: bash

    pip install 'fancychunk[embedders]'

Usage:

.. code-block:: python

    from fancychunk import split_sentences, split_chunklets, split_chunks
    from fancychunk.embedders import default

    embedder = default()
    sentences = split_sentences(doc, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    embeddings = embedder.embed_chunklets(chunklets)
    chunks, _ = split_chunks(chunklets, embeddings, max_size=2048)

Model weights download lazily on first call (Hugging Face cache).
Pre-warm in your build by calling ``embedder.embed_chunklets(["warmup"])``.
"""

from __future__ import annotations

from ._pooled import PooledSegmentEmbedder


def default() -> PooledSegmentEmbedder:
    """The recommended default: Qwen3-Embedding-0.6B.

    ~596M parameters, 1024-dim embeddings, 32K context. MTEB
    Multilingual 64.33 / English v2 70.70 — the strongest of the
    common 600M-tier models. About 100 ms per forward pass on Apple
    Silicon (fp16, MPS) with ~1 GB resident.
    """
    return PooledSegmentEmbedder(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        pooling="last_token",
    )


def fast() -> PooledSegmentEmbedder:
    """Speed-optimized: BGE-M3.

    ~568M parameters, 1024-dim embeddings, 8K context. MTEB
    Multilingual ~59.5 / English v2 ~63.5. About 2.5× faster than
    :func:`default` per forward pass; trades ~5 MTEB points for the
    latency. Use when you're chunking very large corpora and the
    embedding step is the bottleneck.
    """
    return PooledSegmentEmbedder(
        model_id="BAAI/bge-m3",
        pooling="cls",
    )


def high_quality(dim: int = 1024) -> PooledSegmentEmbedder:
    """Highest-quality bundled default: Qwen3-Embedding-4B with MRL.

    ~3.6B parameters, native 2560-dim embeddings truncated to ``dim``
    via Matryoshka Representation Learning. MTEB Multilingual 69.45
    (full dim) — about 5 points above :func:`default`. ~5× slower per
    forward pass; ~7 GB resident. ``dim`` defaults to 1024 to match
    :func:`default`'s output width so you can A/B test storage-
    compatibly; pass ``dim=2560`` for the full native width.
    """
    if dim < 64 or dim > 2560:
        raise ValueError(
            f"dim must be in [64, 2560] for Qwen3-Embedding-4B; got {dim}"
        )
    return PooledSegmentEmbedder(
        model_id="Qwen/Qwen3-Embedding-4B",
        pooling="last_token",
        output_dim=dim,
    )


__all__ = ["PooledSegmentEmbedder", "default", "fast", "high_quality"]
