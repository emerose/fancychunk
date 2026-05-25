# Changelog

All notable changes to fancychunk are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
