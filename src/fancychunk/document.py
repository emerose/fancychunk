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
from ._typing import Matrix, Vector
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
    of N before the per-document downstream pipeline runs.
    wtpsplit-lite's SaT model can share one ONNX forward pass across
    many documents — for corpora of short documents this is the
    single largest win available, typically 3-10× the per-document
    SaT cost depending on length distribution and hardware.

    The segmenter must satisfy
    :class:`~fancychunk._segmenter.BatchSentenceSegmenter` (i.e. expose
    a ``predict_proba_batch`` method) — the bundled
    :class:`SaTSegmenter` does. Passing
    ``segmenter_batch_size`` with a non-batchable custom segmenter
    raises :class:`ValidationError`.

    Trade-off: all SaT inference happens up-front, so the embedder
    cannot start until the segmenter finishes. On corpora where the
    embedder dominates and SaT is already cheap (large documents,
    small batches, slow embedder), leaving
    ``segmenter_batch_size=None`` keeps the existing per-document
    overlap. The win is on small-document workloads where SaT is the
    bottleneck.

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

        per_doc_segmenters = await _maybe_presegment(
            documents=documents,
            segmenter=segmenter,
            segmenter_batch_size=segmenter_batch_size,
        )

        if max_concurrency is None:
            return list(
                await asyncio.gather(
                    *(
                        chunk_document(
                            doc,
                            embedder,
                            max_size=max_size,
                            segmenter=per_doc_segmenters[i],
                        )
                        for i, doc in enumerate(documents)
                    )
                )
            )

        sem = asyncio.Semaphore(max_concurrency)

        async def _one(
            doc: str,
            seg: SentenceSegmenter | None,
        ) -> tuple[list[Chunk], NDArray[np.float64]]:
            async with sem:
                return await chunk_document(
                    doc, embedder, max_size=max_size, segmenter=seg
                )

        return list(
            await asyncio.gather(
                *(
                    _one(doc, per_doc_segmenters[i])
                    for i, doc in enumerate(documents)
                )
            )
        )


async def _maybe_presegment(
    documents: list[str],
    segmenter: SentenceSegmenter | None,
    segmenter_batch_size: int | None,
) -> list[SentenceSegmenter | None]:
    """Pre-segment ``documents`` if a batch size is requested.

    Returns a list of per-document segmenter overrides (or ``None``
    for the default path). The pre-segmentation is wrapped in
    ``asyncio.to_thread`` so the event loop is not blocked while the
    SaT forward pass runs.
    """
    if segmenter_batch_size is None:
        return [segmenter] * len(documents)

    resolved = make_segmenter(segmenter)
    if not isinstance(resolved, BatchSentenceSegmenter):
        raise ValidationError(
            "segmenter_batch_size requires a segmenter that implements "
            "predict_proba_batch (BatchSentenceSegmenter); the bundled "
            "SaTSegmenter does. Got: " + type(resolved).__name__
        )

    batch_fn = resolved.predict_proba_batch
    tracer = get_tracer()
    per_doc: list[Vector | None] = [None] * len(documents)
    with tracer.start_as_current_span(
        "fancychunk.chunk_documents.presegment"
    ) as span:
        span.set_attribute("fancychunk.segmenter_batch_size", segmenter_batch_size)
        span.set_attribute("fancychunk.documents.count", len(documents))
        for start in range(0, len(documents), segmenter_batch_size):
            slice_docs = documents[start : start + segmenter_batch_size]
            vecs = await asyncio.to_thread(batch_fn, slice_docs)
            for offset, vec in enumerate(vecs):
                per_doc[start + offset] = vec

    return [
        precomputed_segmenter(vec) if vec is not None else None
        for vec in per_doc
    ]
