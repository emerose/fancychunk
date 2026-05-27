"""High-level convenience: one call, document in → ``(chunks, vectors)``.

Public entry points:

* :func:`chunk_document` — single document.
* :func:`chunk_documents` — batch of documents, embedded
  concurrently against one shared embedder.

Both compose the three stages plus late chunking under one
signature, using a single caller-supplied embedder for both the
partition decision and the storage embeddings.

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

import asyncio
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ._telemetry import get_tracer
from ._typing import Matrix
from .chunklets import split_chunklets
from .chunks import Chunk, split_chunks
from .errors import ValidationError
from .late_chunking import embed_with_late_chunking
from .sentences import split_sentences


class Embedder(Protocol):
    """Combined embedder protocol — satisfies both
    :class:`ChunkletEmbedder` (for :func:`split_chunks`) and
    :class:`SegmentEmbedder` (for :func:`embed_with_late_chunking`).

    All three methods are ``async``. Required by :func:`chunk_document`,
    which invokes both operations on the same embedder instance so the
    model loads exactly once.

    All bundled embedders (``fancychunk.embedders.bge_m3``,
    ``qwen3_600m``, ``qwen3_4b``, ``qwen3_8b``, ``noop``) satisfy
    this protocol. BYO embedders that previously implemented only
    the late-chunking ``SegmentEmbedder`` contract need to also
    expose ``async embed_chunklets(chunklets) -> Matrix[N, D]`` —
    typically a thin batch loop over their pooled-output mode.
    """

    n_ctx: int

    async def count_tokens(self, texts: list[str]) -> list[int]: ...

    async def embed_segment(
        self, texts: list[str]
    ) -> tuple[Matrix, list[int]]: ...

    async def embed_chunklets(self, chunklets: list[str]) -> Matrix: ...


async def chunk_document(
    document: str,
    embedder: Embedder,
    max_size: int = 2048,
) -> tuple[list[Chunk], NDArray[np.float64]]:
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
    chunks = await split_chunks(chunklets, embedder, max_size=max_size)
    vectors = await embed_with_late_chunking(chunks, embedder)
    return chunks, vectors


async def chunk_documents(
    documents: list[str],
    embedder: Embedder,
    max_size: int = 2048,
    max_concurrency: int | None = None,
) -> list[tuple[list[Chunk], NDArray[np.float64]]]:
    """Chunk a batch of documents concurrently against one embedder.

    Equivalent to
    ``await asyncio.gather(*[chunk_document(d, embedder, max_size)
    for d in documents])``, with optional concurrency capping via
    ``max_concurrency``. Returns one ``(chunks, vectors)`` tuple per
    input document, in input order.

    Concurrency notes:

    * With a bundled embedder (``PooledSegmentEmbedder``), the
      embedder's internal lock serializes access to the local device,
      so increasing concurrency above ~2-4 buys little — extra
      coroutines just queue. The win is overlap between one
      document's CPU work (sentence/chunklet/chunk DP) and another's
      embedder call.
    * With a remote / true-parallel embedder, scale up to the
      server's capacity; pass ``max_concurrency=N`` to cap fan-in if
      the server isn't ready for unbounded concurrent requests.
    * ``max_concurrency=None`` (the default) gathers all documents at
      once with no semaphore.

    Errors propagate via ``asyncio.gather``'s default semantics: the
    first exception aborts the batch and cancels in-flight siblings.
    Wrap individual documents in your own try/except if you need
    partial-results behavior.
    """
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValidationError("max_concurrency must be positive when set")

    with get_tracer().start_as_current_span("fancychunk.chunk_documents") as span:
        span.set_attribute("fancychunk.documents.count", len(documents))
        span.set_attribute("fancychunk.max_size", max_size)
        if max_concurrency is not None:
            span.set_attribute("fancychunk.max_concurrency", max_concurrency)

        if not documents:
            return []

        if max_concurrency is None:
            return list(
                await asyncio.gather(
                    *(
                        chunk_document(doc, embedder, max_size=max_size)
                        for doc in documents
                    )
                )
            )

        sem = asyncio.Semaphore(max_concurrency)

        async def _one(
            doc: str,
        ) -> tuple[list[Chunk], NDArray[np.float64]]:
            async with sem:
                return await chunk_document(doc, embedder, max_size=max_size)

        return list(await asyncio.gather(*(_one(d) for d in documents)))
