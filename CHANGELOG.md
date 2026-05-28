# Changelog

All notable changes to fancychunk are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-05-28

### Changed
- The default sentence segmenter is now **`sat-9l-sm`** (was
  `sat-3l-sm`), run with **`weighting="hat"`** inference
  (`stride=128, block_size=256`). The previous defaults
  (`weighting="uniform"`, larger blocks) averaged every overlapping
  window equally, producing *context-sensitive* boundary artifacts at
  low-context window edges; `hat` weighting plus a higher-capacity
  checkpoint segment scientific-prose constructs — abbreviation
  references (`Tab. TABREF21`, `Eq. EQREF9`) and years before a
  capitalised word (`SemEval-2014 Task`) — correctly with no
  post-processing. `sat-9l-sm` is artifact-free like `sat-12l-sm`,
  tracks its boundary placement closely (corpus F1 0.97), and is ~1.3×
  faster on the batched GPU path; see
  `benchmarks/sat-model-selection.md`. Trade-off vs the old default:
  more segmentation compute per character (a one-time per-document cost,
  amortized on the batched GPU path).

### Added
- `fancychunk.segmenters` — factories for the bundled SaT checkpoints
  (`sat_3l()`, `sat_9l()`, `sat_12l()`, `sat_default()`) and the
  rule-based `punctuation()` fallback, mirroring `fancychunk.embedders`.
  Each SaT factory forwards kwargs to `SaTSegmenter` (e.g.
  `segmenters.sat_12l(device="cuda")`).
- Chunk grouping prefers paragraph boundaries (SPEC-CHUNK-324): a
  partition point that would cut between two sentences of the *same*
  paragraph is penalized (`MID_PARAGRAPH_PENALTY`), so the optimizer
  routes the required cuts to paragraph (or stronger) breaks when one
  is available within budget. Self-cancelling when a paragraph exceeds
  `max_size` (all candidates are mid-paragraph), and never overrides a
  strong semantic boundary.

### Removed
- The numeral-before-capital boundary guard (SPEC-CHUNK-118, added in
  0.5.1). The default model now segments these correctly on its own
  (e.g. the `SemEval-2014` boundary probability drops from ~0.52 to
  ~0.06), so the probability post-processing is no longer needed. A
  caller who switches to the lighter `sat-3l-sm` may reintroduce the
  artifact.

## [0.5.1] - 2026-05-28

### Fixed
- Sentence splitting no longer breaks a sentence mid-phrase at a
  numeral followed by a capitalized word (SPEC-CHUNK-118). The SaT
  model assigns a spuriously high boundary probability to the last
  digit of a year/number directly followed by whitespace and a capital
  (e.g. ``"...SemEval-2014 Task 4..."``, ``"WMT 2016 Task"``,
  ``"ICLR 2020 Workshop"``); that prediction is now forced to ``0``
  before the heading override and merge. Genuine breaks where the
  boundary sits on terminating punctuation (``"...in 2014. Later,
  we..."``) are unaffected.
- Semantic chunking no longer emits a chunk with no standalone
  retrieval value (SPEC-CHUNK-323). The chunk-partition objective now
  includes a per-chunk *badness* term, graded by how far below a target
  the chunk falls, so the optimizer extends an undersized chunk forward
  rather than emit it (the ``max_size`` covering constraint is honored
  automatically since it is the same DP). It is the larger of two
  terms: a strong **front-matter** penalty over a wide size range
  (``FRONT_MATTER_CHUNK_PENALTY`` /
  ``FRONT_MATTER_CHUNK_TARGET_FRACTION``) for the leading preamble (a
  title with no body of its own, e.g. immediately followed by
  ``## Abstract``); and a gentle, short-range **general** penalty
  (``SMALL_CHUNK_PENALTY`` / ``SMALL_CHUNK_TARGET_FRACTION``) that bites
  only on a genuinely tiny fragment. Two terms are needed because chunk
  size alone cannot tell a preamble from a short-but-coherent section.
  The effective size cutoff is not fixed: a chunk is merged only when
  its badness exceeds the split-quality it would surrender, so a more
  distinct chunk is allowed to be smaller before it is kept. (A bare
  heading head needs no term: SPEC-CHUNK-322 already prevents the DP
  from isolating a heading.)

## [0.5.0] - 2026-05-27

### Added
- ``chunk_documents`` now defaults ``segmenter_batch_size="auto"``:
  the batched SaT path turns itself on when the resolved segmenter
  reports a GPU execution provider, off on CPU. The bundled
  ``SaTSegmenter`` exposes ``wants_batching()`` for this decision —
  ``True`` for ``device="cuda"/"gpu"``, ``False`` for
  ``device="cpu"``, and for ``device="auto"`` it peeks at
  ``onnxruntime.get_available_providers()``. Net effect: install
  ``fancychunk`` plus ``onnxruntime-gpu`` on a CUDA box and
  ``chunk_documents(docs, embedder)`` runs at GPU + batched speeds
  with no further configuration. Explicit ``segmenter_batch_size``
  (``None`` to force off, ``int`` to force a specific size) still
  overrides.
- ``SaTSegmenter(device=...)`` selects the onnxruntime execution
  provider. ``"auto"`` (the default) defers to wtpsplit-lite's own
  GPU-first auto-detect — installing ``onnxruntime-gpu`` and asking
  for ``device="cuda"`` is enough to run SaT on the GPU. ``"cpu"``
  forces the CPU EP. The power-user escape hatch
  ``ort_providers=[...]`` is also exposed for explicit provider
  lists (e.g. ROCm, OpenVINO), as is ``ort_kwargs=`` for
  ``InferenceSession`` options.
- ``SaTSegmenter.predict_proba_batch(documents)`` runs one batched
  SaT forward pass over a list of documents and returns a vector
  per input. Empty / whitespace-only documents are passed through
  as zero-length / zero-filled vectors (the downstream
  ``split_sentences`` short-circuits them anyway). Required by the
  new ``BatchSentenceSegmenter`` protocol.
- ``BatchSentenceSegmenter(Protocol)`` — runtime-checkable
  extension of ``SentenceSegmenter`` that adds
  ``predict_proba_batch``. Lets ``chunk_documents`` opt into the
  batched path; BYO segmenters can satisfy it to participate.
- ``chunk_documents(..., segmenter_batch_size=N)`` pre-segments
  documents in groups of N. SaT inference runs off the event loop
  on a worker thread via ``asyncio.to_thread``, and each wave's
  downstream chunking/embedding tasks fire immediately so the next
  wave's forward pass overlaps with the current wave's downstream
  work.

  Measured on a 1,000-doc / 1,500-char corpus (RTX 3090, sat-3l-sm,
  ``embedders.noop()``), ``chunk_documents`` is ~6.6× faster than
  the CPU-no-batch baseline; just turning on ``device="cuda"`` is
  already ~4.9×. The SaT-only batched-vs-serial ratio on the same
  GPU is ~2.2× (raw segmenter cost ~1.45 ms/doc serial, ~0.67
  ms/doc batched).

  CPU-only callers see no benefit (forward-pass FLOPs scale
  linearly with batch size under ``CPUExecutionProvider``); leave
  ``segmenter_batch_size`` unset — on CPU the streaming overlap
  serialises downstream work behind SaT waves with no payoff.
- ``SaTSegmenter`` installs a vectorised replacement for
  ``wtpsplit_lite._utils.token_to_char_probs`` on first load —
  upstream's per-document Python loop scattering per-token logits
  onto a per-character array was consuming ~45% of the batched SaT
  wall on CUDA. The replacement does the same projection in two
  numpy operations and round-trips bit-identically to upstream on
  realistic inputs; correctness covered by
  ``tests/test_segmenter_batching.py``. Set
  ``FANCYCHUNK_DISABLE_SAT_FAST_POSTPROCESS=1`` to keep upstream's
  binding (e.g. if a future ``wtpsplit-lite`` release ships its
  own fix). Effect: the SaT-only batched path drops from ~1.06
  ms/doc to ~0.67 ms/doc (1.58×); ``chunk_documents`` e2e CUDA
  +batched improves from ~5.5 ms/doc to ~5.0 ms/doc (~9%).
- ``chunk_document(..., segmenter=...)`` and
  ``chunk_documents(..., segmenter=...)`` accept a segmenter
  override so per-doc callers (e.g. ingestion pipelines that drive
  ``chunk_document`` one document at a time) can install a
  CUDA-configured ``SaTSegmenter`` once and reuse it.
- ``precomputed_segmenter(probas)`` — wraps a precomputed
  per-character probability vector as a ``SentenceSegmenter``,
  letting advanced callers cache / share boundary probabilities
  across re-ingests of the same document.
- ``bench_sat_batching.py`` — microbenchmark over a synthetic short-
  Markdown corpus that prints per-doc vs batched wall time and
  optionally fails (``--assert-speedup``) below a target ratio.
  Use ``--device cuda`` on a CUDA box to validate the GPU path;
  the API surface is also covered by mocked unit tests in
  ``tests/test_segmenter_batching.py``.

### Changed
- ``chunk_document`` and ``chunk_documents`` accept a keyword-only
  ``segmenter=`` argument; existing callers (``segmenter`` unset)
  see no behavior change.

## [0.4.0] - 2026-05-27

### Added
- ``Chunk.heading_path: tuple[str, ...] | None`` — the Markdown
  heading stack in scope at the chunk's start, as a tuple of full
  heading lines (e.g. ``("# Top", "## **Bold** Sub")``). Each entry
  preserves the ``#`` markers and inline markdown formatting but is
  stripped of trailing whitespace and newlines. The marker count
  encodes heading level, so documents with skipped levels are
  faithfully represented (``("# H1", "### H3")`` not the misleading
  ``("H1", "H3")``). Empty tuple means "no heading in scope";
  ``None`` means "not computed."
  
  ``split_chunks`` and ``chunk_document`` always populate it.
  Use cases: filter chunks by heading
  (``if any("Methods" in h for h in c.heading_path)``), render
  breadcrumbs, or attach as vector-store metadata.
- ``fancychunk.headings.render_heading_path(path) -> str`` —
  convert a tuple-form heading path to a single Markdown string
  (newline-joined with a trailing newline). Used internally by
  late chunking and ``enrich_with_headings``; exposed for callers
  who want the same rendering convention.
- ``fancychunk.headings.resolve_heading_paths(chunks)`` — return
  per-chunk heading paths, preferring ``chunk.heading_path`` when
  populated and falling back to a fresh ``heading_paths`` scan
  otherwise. Lets standalone consumers of ``embed_with_late_chunking``
  / ``enrich_with_headings`` work whether or not the chunks carry
  pre-computed metadata.

### Changed (breaking — pre-1.0)
- ``heading_paths(chunks: list[Chunk]) -> list[tuple[str, ...]]`` —
  return type changed from rendered Markdown strings
  (``"# Top\n## Sub\n"``) to tuples of stripped heading lines
  (``("# Top", "## Sub")``). For the old-style rendered output,
  apply ``render_heading_path`` to each entry. Trailing whitespace
  is no longer preserved (per the new tuple-form semantics).
- ``enrich_with_headings`` uses ``chunk.heading_path`` when
  populated; falls back to ``heading_paths()`` otherwise. Behavior
  unchanged for callers; just an internal optimization for chunks
  that already carry the metadata.
- ``embed_with_late_chunking`` uses ``chunk.heading_path`` when
  populated (same fallback). Eliminates an internal pass over the
  chunks when the metadata is already there.

## [0.3.0] - 2026-05-27

### Changed (breaking — pre-1.0)
- **Chunks are now a typed object, not raw strings.** New
  ``fancychunk.Chunk`` frozen dataclass with:
  - ``text: str`` — always present, the chunk content.
  - ``start: int | None`` — character offset (inclusive) into the
    source, when computed.
  - ``end: int | None`` — character offset (exclusive) into the
    source, when computed. ``source[chunk.start:chunk.end] == chunk.text``.

  The signature changes are: ``split_chunks(...) -> list[Chunk]``,
  ``chunk_document(...) -> tuple[list[Chunk], NDArray]``,
  ``chunk_documents(...) -> list[tuple[list[Chunk], NDArray]]``,
  ``embed_with_late_chunking(chunks: list[Chunk], ...)``,
  ``heading_paths(chunks: list[Chunk]) -> list[str]``, and
  ``enrich_with_headings(chunks: list[Chunk]) -> list[Chunk]``.

  Callers that did ``"".join(chunks)`` need
  ``"".join(c.text for c in chunks)``. Callers that passed raw
  string lists to ``embed_with_late_chunking`` /
  ``heading_paths`` / ``enrich_with_headings`` need
  ``[Chunk(text=s) for s in strings]``. The pipeline producers
  (``split_chunks`` and ``chunk_document``) always populate
  ``start`` / ``end``; ``Chunk(text=s)`` (no offsets) is fine for
  ad-hoc construction. ``str(chunk)`` returns ``chunk.text``.

  Future optional metadata fields can be added without further
  breakage — they'll default to ``None``.

- ``enrich_with_headings`` returns ``list[Chunk]`` with enriched
  ``text`` and the original ``start`` / ``end`` preserved. After
  enrichment, ``len(chunk.text) != chunk.end - chunk.start`` —
  metadata still references the original source range.

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

[0.4.0]: https://github.com/emerose/fancychunk/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/emerose/fancychunk/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/emerose/fancychunk/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/emerose/fancychunk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/emerose/fancychunk/releases/tag/v0.1.0
