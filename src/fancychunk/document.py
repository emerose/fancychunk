"""High-level convenience: one call, document in → ``(chunks, vectors)``.

Public entry points:

* :func:`chunk_document` — single document.
* :func:`chunk_documents` — batch of documents, embedded
  concurrently against one shared embedder.

Both run structure-first chunking
(:func:`fancychunk.structure_first.split_chunks_structure_first`)
followed by a late-chunking embed pass, using a single
caller-supplied embedder for both the fallback partition decision
and the storage embeddings.

For finer control (different embedders per stage, custom
``max_size`` per stage, the older whole-document semantic split, or
embedding chunks in isolation instead of late chunking) compose the
underlying primitives directly:

* :func:`split_chunks_structure_first` (structure-first chunking)
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

from ._segmenter import SentenceSegmenter
from ._telemetry import get_tracer
from ._typing import Matrix
from .chunks import Chunk
from .errors import ValidationError
from .late_chunking import embed_with_late_chunking
from .structure_first import split_chunks_structure_first


class Embedder(Protocol):
    """Combined embedder protocol — satisfies both
    :class:`ChunkletEmbedder` (for :func:`split_chunks`) and
    :class:`SegmentEmbedder` (for :func:`embed_with_late_chunking`).

    All three methods are ``async``. Required by :func:`chunk_document`,
    which invokes both operations on the same embedder instance so the
    model loads exactly once.

    All bundled embedders (``fancychunk.embedders.bge_m3``,
    ``qwen3_600m``, ``qwen3_4b``, ``qwen3_8b``, ``jina_v3``,
    ``noop``) satisfy this protocol. BYO embedders that previously implemented only
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
    *,
    min_size: int | None = None,
    segmenter: SentenceSegmenter | None = None,
) -> tuple[list[Chunk], NDArray[np.float64]]:
    """Chunk ``document`` and return ``(chunks, vectors)``.

    Uses **structure-first** chunking (spec
    [06-structural-chunking](../../docs/specs/06-structural-chunking.md)):

    1. :func:`split_chunks_structure_first(document, embedder,
       max_size=max_size, min_size=min_size, segmenter=segmenter)` —
       the document's heading tree is the primary unit. A section whose
       whole subtree already fits ``max_size`` is emitted directly, with
       **no** SaT or embedder call; only an overflowing section falls
       back to the semantic split (``split_sentences`` →
       ``split_chunklets`` → ``split_chunks``) on that span alone. This
       lands headings at chunk starts and skips the slow models on
       already-fitting sections.
    2. :func:`embed_with_late_chunking(chunks, embedder)` — one
       context-aware embedding per chunk; ``include_headings=True``
       (the default) prepends the in-scope Markdown heading stack to
       each segment so the embedding sees the document outline.

    .. note::
       Step 2 uses **late chunking**, which is now considered
       experimental. Downstream RAG benchmarking found it did not beat
       plain isolated-chunk embedding, and it regressed on long
       documents with the bundled causal, last-token-pooled models
       (Qwen3). For the recommended plain path, run the structural
       split and embed the final chunks in isolation::

           from fancychunk.structure_first import split_chunks_structure_first
           chunks  = await split_chunks_structure_first(document, embedder, max_size=max_size)
           vectors = await embedder.embed_chunklets([c.text for c in chunks])

       To reach the older whole-document semantic split (no structural
       pass), compose the primitives directly: ``split_sentences`` →
       ``split_chunklets`` → ``split_chunks``.

    The returned chunks satisfy ``"".join(chunks) == document``
    (SPEC-CHUNK-300/900). The returned vectors are L2-normalized; row
    ``i`` is the storage embedding for ``chunks[i]``.

    The same ``embedder`` instance is used for both the fallback
    partition decision and the late-chunking pass, so its model loads
    exactly once. Pick the embedder explicitly — ``fancychunk.embedders``
    has the bundled choices; ``qwen3_600m()`` is the recommended
    default for most uses.

    ``min_size`` is the chunk-size floor below which a structural unit
    is merged into a neighbor to avoid thin chunks (SPEC-CHUNK-630). It
    defaults to ``0.35 * max_size``; pass ``0`` to disable merging.

    Pass ``segmenter=`` to override the SaT default (e.g. a CUDA-
    configured :class:`SaTSegmenter`, or
    :func:`punctuation_segmenter` for zero-dependency use). When
    ``segmenter`` is ``None`` the process-wide SaT singleton is used.
    The segmenter is only invoked on the fallback path.

    For storage-time heading-breadcrumb decoration, apply
    :func:`enrich_with_headings` to the returned chunks. That step
    does **not** affect ``vectors`` — late chunking already saw the
    headings in the document via its per-segment heading prepend
    (SPEC-CHUNK-470).
    """
    chunks = await split_chunks_structure_first(
        document,
        embedder,
        max_size=max_size,
        min_size=min_size,
        segmenter=segmenter,
    )
    vectors = await embed_with_late_chunking(chunks, embedder)
    return chunks, vectors


async def chunk_documents(
    documents: list[str],
    embedder: Embedder,
    max_size: int = 2048,
    max_concurrency: int | None = None,
    *,
    min_size: int | None = None,
    segmenter: SentenceSegmenter | None = None,
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
      document's CPU work (structural planning, sentence/chunklet/chunk
      DP on fallback spans) and another's embedder call.
    * With a remote / true-parallel embedder, scale up to the
      server's capacity; pass ``max_concurrency=N`` to cap fan-in if
      the server isn't ready for unbounded concurrent requests.
    * ``max_concurrency=None`` (the default) gathers all documents at
      once with no semaphore.

    Because :func:`chunk_document` is structure-first, SaT runs only on
    sections that overflow ``max_size``; a well-sectioned corpus skips
    the segmenter on most of its text. The slow models are no longer the
    dominant batch cost they were under whole-document segmentation, so
    this no longer cross-document-batches the segmenter — each document
    is segmented lazily on its own fallback spans.

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

        sem = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency is not None
            else None
        )

        async def _one(
            doc: str,
        ) -> tuple[list[Chunk], NDArray[np.float64]]:
            if sem is None:
                return await chunk_document(
                    doc,
                    embedder,
                    max_size=max_size,
                    min_size=min_size,
                    segmenter=segmenter,
                )
            async with sem:
                return await chunk_document(
                    doc,
                    embedder,
                    max_size=max_size,
                    min_size=min_size,
                    segmenter=segmenter,
                )

        return list(await asyncio.gather(*(_one(doc) for doc in documents)))
