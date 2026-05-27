# Changelog

All notable changes to fancychunk are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-05-26

### Changed (breaking — pre-1.0)
- **Async-first public API.** ``split_chunks``,
  ``embed_with_late_chunking``, and ``chunk_document`` are now
  ``async def``. The ``Embedder`` / ``SegmentEmbedder`` /
  ``ChunkletEmbedder`` protocols are async-only — every method
  (``count_tokens``, ``embed_segment``, ``embed_chunklets``) is
  ``async``. Sync callers wrap with ``asyncio.run(...)``.
  ``split_sentences``, ``split_chunklets``, ``heading_paths``, and
  ``enrich_with_headings`` stay sync (no await points).
- ``embed_with_late_chunking`` embeds segments in parallel via
  ``asyncio.gather`` instead of serially. For remote / true-parallel
  embedders this overlaps network or GPU work; for the bundled
  embedders the internal lock still serializes to device throughput,
  so behavior is unchanged.
- Bundled embedders (``PooledSegmentEmbedder``, ``NoopSegmentEmbedder``)
  implement the new async protocol. ``PooledSegmentEmbedder`` wraps
  every forward pass in ``asyncio.to_thread`` so the event loop keeps
  spinning while the GPU works; the internal ``RLock`` continues to
  serialize concurrent worker-thread access to the underlying model.
- Reference adapters under ``examples/embedders/`` migrated:
  ``remote_http.py`` now uses ``httpx.AsyncClient`` and is an async
  context manager; ``huggingface_offsets.py`` and ``qwen3_mlx.py``
  expose async wrappers over their sync forward passes.

### Added
- ``chunk_documents(documents, embedder, max_size=2048,
  max_concurrency=None)`` — batched variant of ``chunk_document``
  that runs the pipeline over a list of documents via
  ``asyncio.gather``. Returns one ``(chunks, vectors)`` tuple per
  input document, in order. Pass ``max_concurrency=N`` to cap
  fan-in via a semaphore (useful when driving a remote embedder
  with finite capacity).
- ``PooledSegmentEmbedder`` is now thread-safe. An internal
  ``threading.RLock`` guards weight loading and every forward pass, so
  ``count_tokens`` / ``embed_segment`` / ``embed_chunklets`` can be
  called concurrently from multiple threads (``asyncio.to_thread`` or a
  ``ThreadPoolExecutor``) without racing on the lazy load or on the
  underlying torch / MLX model. Concurrent callers are serialized
  internally — single-instance throughput matches what the device can
  deliver. For more parallelism, create multiple embedder instances.
- ``SaTSegmenter`` and the module-level ``get_default_segmenter``
  singleton are now thread-safe. Double-checked locking around the
  408 MB SaT weight download means concurrent first callers no longer
  race on the lazy load (previously they could both initiate the
  download and both overwrite the cached attribute).
- ``tests/test_concurrency.py`` regresses the shared markdown-it-py
  parser's reentrancy assumption so a future markdown-it-py release
  that mutates parser state during ``.parse()`` would surface as a
  loud test failure instead of silent token-stream corruption.

## [0.1.1]

### Added
- Linux / NVIDIA CUDA benchmark results in the README Models table,
  measured on an NVIDIA GeForce RTX 3090 (24 GB VRAM, driver
  580.159.03) with Intel Core i9-10900KF and 32 GB system RAM, on
  Linux 6.17 with PyTorch 2.12.0 + bundled CUDA 13.0 wheels.
- **MLX backend** for ``PooledSegmentEmbedder``. On Apple Silicon
  with ``mlx_embeddings`` installed, every factory transparently
  loads the corresponding ``mlx-community/...-mxfp8`` (Qwen3) or
  ``mlx-community/bge-m3-mlx-fp16`` (BGE-M3) build instead of the
  torch + MPS path. 2-4× faster on the same hardware. ``[torch]`` /
  ``[mlx]`` / ``[all]`` install extras gate the heavy deps.
- ``chunk_document(document, embedder)`` one-call composed pipeline.
- ``fancychunk.embedders`` module with model-named factories
  ``bge_m3``, ``qwen3_600m``, ``qwen3_4b``, ``qwen3_8b``, and
  ``noop``. All return a ``PooledSegmentEmbedder`` (or
  ``NoopSegmentEmbedder``) implementing both halves of the embedder
  contract. Lazy model loading; pick a device automatically.
- ``split_chunks`` accepts an optional ``chunklet_embeddings``
  argument; the structural-only fallback uses uniform ``sim = 1.0`` +
  the SPEC-CHUNK-322 heading-aware modification so the pipeline runs
  end-to-end with no embedder.
- Reference Embedder adapters under ``examples/embedders/`` (MLX +
  Qwen3-Embedding, HuggingFace transformers, remote HTTP) implement
  the full protocol (both ``embed_segment`` and ``embed_chunklets``).

### Changed (breaking — pre-1.0)
- ``embed_with_late_chunking`` takes a ``SegmentEmbedder`` instead of
  the four-method ``TokenLevelEmbedder``. The new protocol is two
  methods + one attribute (``n_ctx``, ``count_tokens``,
  ``embed_segment``). Tokenization, special-token policy, and
  sentence-to-token alignment are now the embedder's concern.
- ``embed_with_late_chunking`` takes chunks (not sentences) and
  returns one vector per chunk.
- ``split_chunks`` returns just ``list[str]`` (dropped the matrix
  tuple — late chunking owns the storage embeddings now).

### Spec changes
- SPEC-CHUNK-420 rewritten: per-sentence alignment is the embedder's
  responsibility; the library's contract is that
  ``sum(per_sentence_counts) == matrix_row_count``.
- SPEC-CHUNK-421 removed (sentinel character requirements are no
  longer normative).
- SPEC-CHUNK-412 simplified to four steps; the largest-remainder
  safety net stays as the absorber for count drift.

## [0.1.0]

### Added
- Initial implementation of every pipeline stage in the spec:
  - Stage 1 ``split_sentences`` (SaT-backed default segmenter via
    ``wtpsplit-lite``, lazy-loaded; punctuation fallback; heading
    override; whitespace-trailing pass; vectorised boundary-score DP).
  - Stage 2 ``split_chunklets`` (markdown-it-driven boundary
    probabilities, statement-count function, vectorised
    minimum-cost DP).
  - Stage 3 ``split_chunks`` (unit-norm + discourse-vector projection,
    rescaled cosine similarity, heading-aware modification recognising
    both ATX and Setext, vectorised DP under covering constraint).
  - Stage 4 ``embed_with_late_chunking`` (greedy segment construction
    with backward preamble, sentinel-token alignment, largest-remainder
    safety net).
  - Stage 5 ``heading_paths`` (Markdown heading stack with reset
    semantics).
- Public exception hierarchy (``fancychunk.FancyChunkError`` and
  subclasses).
- OpenTelemetry tracing for every public entry point. Spans are
  zero-cost no-ops when no SDK is configured; span names are
  ``fancychunk.<function>`` and attributes use the
  ``fancychunk.<key>`` namespace.
- 88-test pytest suite covering every remaining test vector plus
  cross-stage invariants and the inspection-only SPEC-CHUNK IDs
  (101, 110, 113, 116). Tests use the punctuation segmenter by
  default; set ``FANCYCHUNK_TEST_USE_SAT=1`` to exercise the real
  model.
- GitHub Actions CI workflow runs pyright (strict mode) and pytest
  against Python 3.12 and 3.13.

[0.2.0]: https://github.com/emerose/fancychunk/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/emerose/fancychunk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/emerose/fancychunk/releases/tag/v0.1.0
