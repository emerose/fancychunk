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

from ._segmenter import (
    BatchSentenceSegmenter,
    SentenceSegmenter,
    make_segmenter,
    precomputed_segmenter,
)
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
    *,
    segmenter: SentenceSegmenter | None = None,
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

    Pass ``segmenter=`` to override the SaT default (e.g. a CUDA-
    configured :class:`SaTSegmenter`, or
    :func:`punctuation_segmenter` for zero-dependency use). When
    ``segmenter`` is ``None`` the process-wide SaT singleton is used.

    For storage-time heading-breadcrumb decoration, apply
    :func:`enrich_with_headings` to the returned chunks. That step
    does **not** affect ``vectors`` — late chunking already saw the
    headings in the document via its per-segment heading prepend
    (SPEC-CHUNK-470).
    """
    sentences = split_sentences(document, max_len=max_size, segmenter=segmenter)
    chunklets = split_chunklets(sentences, max_size=max_size)
    chunks = await split_chunks(chunklets, embedder, max_size=max_size)
    vectors = await embed_with_late_chunking(chunks, embedder)
    return chunks, vectors


async def chunk_documents(
    documents: list[str],
    embedder: Embedder,
    max_size: int = 2048,
    max_concurrency: int | None = None,
    *,
    segmenter: SentenceSegmenter | None = None,
    segmenter_batch_size: int | None = None,
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

    Segmenter batching (``segmenter_batch_size``):

    Set ``segmenter_batch_size=N`` to pre-segment documents in groups
    of N. SaT runs in waves on a worker thread; each wave's
    downstream chunking/embedding tasks fire immediately so the next
    wave's forward pass overlaps with the current wave's downstream
    work. Measured numbers on this layout (RTX 3090, sat-3l-sm,
    1,000 × 1,500-char docs, ``embedders.noop()``): SaT-only batched
    vs serial on GPU is ~2.2× (0.67 ms/doc batched, 1.45 ms/doc
    serial); the full ``chunk_documents`` pipeline is 4.9× over CPU
    just from ``device="cuda"`` and 6.6× with batching on top.

    The segmenter must satisfy
    :class:`~fancychunk._segmenter.BatchSentenceSegmenter` (i.e. expose
    a ``predict_proba_batch`` method) — the bundled
    :class:`SaTSegmenter` does. Passing
    ``segmenter_batch_size`` with a non-batchable custom segmenter
    raises :class:`ValidationError`.

    CPU-only callers see no benefit (forward FLOPs scale linearly
    with batch size under ``CPUExecutionProvider``); leave
    ``segmenter_batch_size`` unset — the streaming overlap can
    actually make it slower because SaT waves serialise behind
    downstream work that would otherwise overlap per-doc.

    Errors propagate via ``asyncio.gather``'s default semantics: the
    first exception aborts the batch and cancels in-flight siblings.
    Wrap individual documents in your own try/except if you need
    partial-results behavior.
    """
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValidationError("max_concurrency must be positive when set")
    if segmenter_batch_size is not None and segmenter_batch_size <= 0:
        raise ValidationError(
            "segmenter_batch_size must be positive when set"
        )

    with get_tracer().start_as_current_span("fancychunk.chunk_documents") as span:
        span.set_attribute("fancychunk.documents.count", len(documents))
        span.set_attribute("fancychunk.max_size", max_size)
        if max_concurrency is not None:
            span.set_attribute("fancychunk.max_concurrency", max_concurrency)
        if segmenter_batch_size is not None:
            span.set_attribute(
                "fancychunk.segmenter_batch_size", segmenter_batch_size
            )

        if not documents:
            return []

        sem = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency is not None
            else None
        )

        async def _one(
            doc: str,
            seg: SentenceSegmenter | None,
        ) -> tuple[list[Chunk], NDArray[np.float64]]:
            if sem is None:
                return await chunk_document(
                    doc, embedder, max_size=max_size, segmenter=seg
                )
            async with sem:
                return await chunk_document(
                    doc, embedder, max_size=max_size, segmenter=seg
                )

        if segmenter_batch_size is None:
            return list(
                await asyncio.gather(
                    *(_one(doc, segmenter) for doc in documents)
                )
            )

        # Batched path: stream SaT waves and fire each wave's
        # chunk_document tasks immediately, so the next wave's
        # ``asyncio.to_thread`` runs concurrently with downstream
        # chunking/embedding for already-segmented docs.
        resolved = make_segmenter(segmenter)
        if not isinstance(resolved, BatchSentenceSegmenter):
            raise ValidationError(
                "segmenter_batch_size requires a segmenter that "
                "implements predict_proba_batch (BatchSentenceSegmenter); "
                "the bundled SaTSegmenter does. Got: "
                + type(resolved).__name__
            )
        batch_fn = resolved.predict_proba_batch

        tasks: list[asyncio.Task[tuple[list[Chunk], NDArray[np.float64]]]] = []
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "fancychunk.chunk_documents.presegment"
        ) as ps_span:
            ps_span.set_attribute(
                "fancychunk.segmenter_batch_size", segmenter_batch_size
            )
            ps_span.set_attribute(
                "fancychunk.documents.count", len(documents)
            )
            for start in range(0, len(documents), segmenter_batch_size):
                slice_docs = documents[start : start + segmenter_batch_size]
                vecs = await asyncio.to_thread(batch_fn, slice_docs)
                for offset, vec in enumerate(vecs):
                    i = start + offset
                    seg = (
                        precomputed_segmenter(vec)
                        if vec is not None
                        else None
                    )
                    tasks.append(
                        asyncio.create_task(_one(documents[i], seg))
                    )
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
        return list(results)
