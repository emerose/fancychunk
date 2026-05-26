# Changelog

All notable changes to fancychunk are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (breaking — pre-1.0)
- ``fancychunk.embedders`` factories renamed and the set extended.
  The new order is **`fastest` → `fast` → `medium` → `high`** by
  increasing quality / parameter count:
  - ``fastest()`` was ``fast()`` (BGE-M3 / CLS pooling).
  - ``fast()`` was ``default()`` (Qwen3-Embedding-0.6B).
  - ``medium(dim=1024)`` was ``high_quality(dim=1024)`` (Qwen3-4B + MRL).
  - ``high(dim=1024)`` is new (Qwen3-8B + MRL, ~70.58 MTEB-Multi).

### Added
- **MLX backend** for ``PooledSegmentEmbedder``. On Apple Silicon
  with ``mlx_embeddings`` installed, every factory transparently
  loads the corresponding ``mlx-community/...-mxfp8`` (Qwen3) or
  ``mlx-community/bge-m3-mlx-fp16`` (BGE-M3) build instead of the
  torch + MPS path. 2-4× faster on the same hardware. The
  ``[embedders]`` extra now installs ``mlx`` and ``mlx-embeddings``
  via PEP 508 platform markers so the deps land only on macOS arm64.
- ``high()`` factory wrapping Qwen3-Embedding-8B with MRL truncation
  to ``dim=1024`` (configurable up to native 4096). MTEB Multilingual
  70.58 / English 75.22 — top of the bundled options.


- ``split_chunks`` accepts an optional ``chunklet_embeddings`` argument
  (now ``None`` by default). The structural-only fallback uses uniform
  ``sim = 1.0`` + the SPEC-CHUNK-322 heading-aware modification, so
  the three-stage pipeline runs end-to-end with no embedder at all.
  SPEC-CHUNK-320 grew a "No-embeddings path" paragraph; the public
  API contract reflects the new default.
- ``fancychunk.embedders`` module with three opinionated defaults
  behind the ``[embedders]`` extra:
  - ``default()`` → Qwen3-Embedding-0.6B (last-token pooling).
  - ``fast()`` → BGE-M3 (CLS pooling), ~2.5× faster.
  - ``high_quality(dim=1024)`` → Qwen3-Embedding-4B with Matryoshka
    truncation to ``dim`` (default 1024 to match the others' width).
  All three return a ``PooledSegmentEmbedder`` implementing the
  ``SegmentEmbedder`` protocol (for late chunking) plus an
  ``embed_chunklets()`` convenience for pooled per-chunklet
  embeddings. Lazy model loading; pick a device automatically (mps /
  cuda / cpu).

### Changed (breaking — pre-1.0)
- `embed_with_late_chunking` now takes a `SegmentEmbedder` instead of
  `TokenLevelEmbedder`. The new protocol is two methods + one
  attribute (`n_ctx`, `count_tokens`, `embed_segment`) — replacing
  the four-method `tokenize` / `detokenize` / `embed` / `n_ctx`
  contract.
- Tokenization, special-token policy, and sentence-to-token
  alignment are now the embedder's concern, not the library's. The
  sentinel-token method (with `⊕` default), sentinel discovery,
  and the `sentinel` keyword argument are removed.
- New `examples/embedders/` directory with reference adapters for
  MLX (`qwen3_mlx.py`), HuggingFace transformers
  (`huggingface_offsets.py`), and a remote HTTP service
  (`remote_http.py`), each runnable.

### Spec changes
- SPEC-CHUNK-420 rewritten: per-sentence alignment is the embedder's
  responsibility; the library's contract is that
  `sum(per_sentence_counts) == matrix_row_count`.
- SPEC-CHUNK-421 removed (sentinel character requirements are no
  longer normative — implementations that adopt the sentinel method
  test it against their own tokenizer).
- SPEC-CHUNK-412 simplified to four steps; the largest-remainder
  safety net stays as the absorber for count drift between
  budget-planning and the actual joined-input tokenization.
- TV-407 (sentinel collision detection) removed; TV-408 rewritten
  for the new protocol.

### Added
- Initial implementation of every pipeline stage in the spec:
  - Stage 1 `split_sentences` (SaT-backed default segmenter via
    `wtpsplit-lite`, lazy-loaded; punctuation fallback; heading
    override; whitespace-trailing pass; vectorised boundary-score DP).
  - Stage 2 `split_chunklets` (markdown-it-driven boundary
    probabilities, statement-count function, vectorised
    minimum-cost DP).
  - Stage 3 `split_chunks` (unit-norm + discourse-vector projection,
    rescaled cosine similarity, heading-aware modification recognising
    both ATX and Setext, vectorised DP under covering constraint).
  - Stage 4 `embed_with_late_chunking` (greedy segment construction
    with backward preamble, sentinel-token alignment, largest-remainder
    safety net).
  - Stage 5 `heading_paths` (Markdown heading stack with reset
    semantics).
- Public exception hierarchy (`fancychunk.FancyChunkError` and
  subclasses).
- OpenTelemetry tracing for every public entry point. Spans are
  zero-cost no-ops when no SDK is configured; span names are
  `fancychunk.<function>` and attributes use the
  `fancychunk.<key>` namespace.
- 88-test pytest suite covering every remaining test vector plus
  cross-stage invariants and the inspection-only SPEC-CHUNK IDs
  (101, 110, 113, 116). Tests use the punctuation segmenter by
  default; set `FANCYCHUNK_TEST_USE_SAT=1` to exercise the real
  model.
- GitHub Actions CI workflow runs pyright (strict mode) and pytest
  against Python 3.12 and 3.13.

### Removed
- Four test vectors that contradicted the spec's own cost math
  (TV-113 as originally written, TV-205's downstream-partition
  property, TV-209, TV-306). TV-113 was rewritten to test genuine
  infeasibility; TV-205 was rewritten to test the upstream
  probability mapping.

[Unreleased]: https://github.com/anthropics/fancychunk/compare/HEAD
