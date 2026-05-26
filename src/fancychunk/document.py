"""High-level convenience: one call, document in → ``(chunks, vectors)``.

Public entry point: :func:`chunk_document`. Composes the three
stages plus late chunking under one signature, using a single
caller-supplied embedder for both the partition decision and the
storage embeddings.

For finer control (different embedders per stage, custom
``max_size`` per stage, structural-only mode, etc.) compose the
underlying primitives directly:

* :func:`split_sentences`
* :func:`split_chunklets`
* :func:`split_chunks`
* :func:`embed_with_late_chunking`
* :func:`enrich_with_headings` (optional, after embedding)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ._typing import Matrix
from .chunklets import split_chunklets
from .chunks import split_chunks
from .late_chunking import embed_with_late_chunking
from .sentences import split_sentences


class Embedder(Protocol):
    """Combined embedder protocol — satisfies both
    :class:`ChunkletEmbedder` (for :func:`split_chunks`) and
    :class:`SegmentEmbedder` (for :func:`embed_with_late_chunking`).

    Required by :func:`chunk_document`, which invokes both
    operations on the same embedder instance so the model loads
    exactly once.

    All bundled embedders (``fancychunk.embedders.bge_m3``,
    ``qwen3_600m``, ``qwen3_4b``, ``qwen3_8b``, ``noop``) satisfy
    this protocol. BYO embedders that previously implemented only
    the late-chunking ``SegmentEmbedder`` contract need to also
    expose ``embed_chunklets(chunklets) -> Matrix[N, D]`` —
    typically a thin batch loop over their pooled-output mode.
    """

    n_ctx: int

    def count_tokens(self, texts: list[str]) -> list[int]: ...

    def embed_segment(
        self, texts: list[str]
    ) -> tuple[Matrix, list[int]]: ...

    def embed_chunklets(self, chunklets: list[str]) -> Matrix: ...


def chunk_document(
    document: str,
    embedder: Embedder,
    max_size: int = 2048,
) -> tuple[list[str], NDArray[np.float64]]:
    """Chunk ``document`` and return ``(chunks, vectors)``.

    Composes the full pipeline:

    1. :func:`split_sentences(document, max_len=max_size)`
    2. :func:`split_chunklets(sentences, max_size=max_size)`
    3. :func:`split_chunks(chunklets, embedder, max_size=max_size)`
       — semantic split, driven by ``embedder.embed_chunklets``.
    4. :func:`embed_with_late_chunking(chunks, embedder)` — one
       context-aware embedding per chunk; ``include_headings=True``
       (the default) prepends the in-scope Markdown heading stack to
       each segment so the embedding sees the document outline.

    The returned chunks satisfy ``"".join(chunks) == document``
    (SPEC-CHUNK-300). The returned vectors are L2-normalized; row
    ``i`` is the storage embedding for ``chunks[i]``.

    The same ``embedder`` instance is used for both the partition
    decision and the late-chunking pass, so its model loads exactly
    once. Pick the embedder explicitly — ``fancychunk.embedders``
    has the bundled choices; ``qwen3_600m()`` is the recommended
    default for most uses.

    For storage-time heading-breadcrumb decoration, apply
    :func:`enrich_with_headings` to the returned chunks. That step
    does **not** affect ``vectors`` — late chunking already saw the
    headings in the document via its per-segment heading prepend
    (SPEC-CHUNK-470).
    """
    sentences = split_sentences(document, max_len=max_size)
    chunklets = split_chunklets(sentences, max_size=max_size)
    chunks = split_chunks(chunklets, embedder, max_size=max_size)
    vectors = embed_with_late_chunking(chunks, embedder)
    return chunks, vectors
