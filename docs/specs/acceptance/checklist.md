# Acceptance Checklist

An implementation conforms to fancychunk if it passes every check
below. The checklist mirrors the SPEC-CHUNK IDs in the specs.

For each numbered item:

1. Write a test that exercises the behavior.
2. Run it against the implementation.
3. Mark it `[x]` when passing.

Each check cites the test vectors that cover it (`TV-NNN`) or is
marked **"verify by inspection"** when the behavior is best checked
by reading the code rather than by a black-box test.

A conforming implementation has every box checked.

## Cross-cutting (SPEC-CHUNK-9xx)

- [ ] **SPEC-CHUNK-900** — Concatenating any stage's outputs reconstructs
  its input byte-for-byte. Covered by TV-103, TV-203, TV-303.
- [ ] **SPEC-CHUNK-901** — Output is deterministic for a fixed input,
  configuration, and embedder. Run each test vector twice and assert
  identical output (verify across all test vectors).
- [ ] **SPEC-CHUNK-902** — Size limits are upper bounds. Covered by
  TV-107, TV-204, TV-304.
- [ ] **SPEC-CHUNK-903** — Trivial-input short-circuits in stages 1
  and 3. Covered by TV-101 (stage 1), TV-301, TV-302, TV-305
  (stage 3).

## Sentence splitting (SPEC-CHUNK-1xx)

- [ ] **SPEC-CHUNK-100** — Concatenation reproduces document.
  Covered by TV-103 (general property), TV-109 (multi-byte UTF-8
  content).
- [ ] **SPEC-CHUNK-101** — Every sentence has at least one non-whitespace
  character. Verify by inspection (no test vector explicitly checks
  the property; every 1xx TV implicitly relies on it).
- [ ] **SPEC-CHUNK-102** — Only the first sentence may begin with
  whitespace. Covered by TV-106.
- [ ] **SPEC-CHUNK-103** — Sentence length `≥ min_len` (with
  SPEC-CHUNK-114 short-circuit exception). Covered by TV-102
  (boundary case) and TV-107 (length-bounded splitting).
- [ ] **SPEC-CHUNK-104** — Sentence length `≤ max_len` when set. Covered
  by TV-107 (length-bounded splitting), TV-108 (`max_len` larger
  than document is a no-op).
- [ ] **SPEC-CHUNK-105** — Operates on per-character boundary probability
  vector. Verify by inspection.
- [ ] **SPEC-CHUNK-106** — Uses a sentence-segmentation model. Verify by
  inspection (the implementation should expose a swappable model
  interface).
- [ ] **SPEC-CHUNK-107** — Known overrides take precedence over
  predicted values; `NaN` defers to predicted. Covered by TV-111.
- [ ] **SPEC-CHUNK-108** — Markdown headings are forced to be standalone
  sentences. Covered by TV-104, TV-105, TV-112.
- [ ] **SPEC-CHUNK-109** — Whitespace is trailing, not leading. Covered
  by TV-106.
- [ ] **SPEC-CHUNK-110** — Splitting maximizes
  `Σ (probability − BOUNDARY_SCORE_THRESHOLD)` over chosen boundaries.
  Verify by inspection (the property is the optimization objective;
  test it by constructing a small probability vector with known
  optimum).
- [ ] **SPEC-CHUNK-111** — Max-length handling produces sentences
  `≤ max_len`. Covered by TV-107. (One-pass vs. two-pass is
  implementation-defined.)
- [ ] **SPEC-CHUNK-112** — Deterministic. Run any TV twice and assert
  identical output.
- [ ] **SPEC-CHUNK-113** — Ties broken by smallest predecessor index.
  Verify by inspection (no TV depends on the tie-breaker; required
  for cross-run reproducibility).
- [ ] **SPEC-CHUNK-114** — Document shorter than `min_len` returns single
  sentence. Covered by TV-101.
- [ ] **SPEC-CHUNK-115** — Unsatisfiable length constraints raise.
  Covered by TV-113.
- [ ] **SPEC-CHUNK-116** — No boundary above threshold + no-boundary
  case valid → single sentence. Verify by inspection (construct an
  input where the model predicts all-zero probabilities).
- [ ] **SPEC-CHUNK-117** — Empty document returns `[]`. Covered by
  TV-110.

## Chunklet grouping (SPEC-CHUNK-2xx)

- [ ] **SPEC-CHUNK-200** — Concatenation reproduces input sentences.
  Covered by TV-203 (property held by every other 2xx TV).
- [ ] **SPEC-CHUNK-201** — Every chunklet `≤ max_size`. Covered by TV-204.
- [ ] **SPEC-CHUNK-202** — Number of chunklets in `[1, len(sentences)]`.
  Covered by TV-201, TV-202 (boundary cases).
- [ ] **SPEC-CHUNK-210** — Optimization minimizes
  `Σ (boundary_cost + statement_cost)`. Verify by inspection
  (the property is the optimization objective).
- [ ] **SPEC-CHUNK-220** — Default boundary cost is
  `(1 - p[0]) + sum(p[1:])`. Verify by inspection.
- [ ] **SPEC-CHUNK-221** — Default statement cost is
  `STATEMENT_COST_SCALE * (s - 3)² / sqrt(max(s, 1e-6))`. Covered by
  TV-206.
- [ ] **SPEC-CHUNK-230** — Statement count is the documented piecewise
  function over word count, anchored at document `q25`/`q75`.
  Covered by TV-206, TV-207.
- [ ] **SPEC-CHUNK-240** — Per-sentence boundary probabilities follow
  the Markdown token-type table. Covered by TV-205, TV-208, TV-209.
- [ ] **SPEC-CHUNK-241** — Consecutive non-zero boundaries: only the max
  survives. Covered by TV-208.
- [ ] **SPEC-CHUNK-250** — Deterministic. Run any TV twice and assert
  identical output.
- [ ] **SPEC-CHUNK-251** — Ties broken in favor of smallest predecessor
  index (forward-DP equivalent). Covered by TV-210.
- [ ] **SPEC-CHUNK-260** — Empty input returns `[]`. Covered by TV-201.
- [ ] **SPEC-CHUNK-261** — Single sentence returns single chunklet.
  Covered by TV-202.
- [ ] **SPEC-CHUNK-262** — Total length within `max_size` may still
  produce multi-chunklet partition. Covered by TV-206 (12 short
  sentences fitting in `max_size = 2048` partition into four
  chunklets driven by statement cost).
- [ ] **SPEC-CHUNK-263** — Sentence exceeding `max_size` raises an
  error. Covered by TV-211.

## Semantic chunking (SPEC-CHUNK-3xx)

- [ ] **SPEC-CHUNK-300** — Concatenation reproduces input chunklets.
  Covered by TV-303 (property held by every other 3xx TV).
- [ ] **SPEC-CHUNK-301** — Every chunk `≤ max_size`. Covered by TV-304.
- [ ] **SPEC-CHUNK-302** — Embedding rows preserved in order across the
  partition. Covered by TV-303 (verifies the property explicitly).
- [ ] **SPEC-CHUNK-310** — Optimization minimizes total partition
  similarity. Verify by inspection (this is the optimization
  objective).
- [ ] **SPEC-CHUNK-311** — Covering constraint enforced; no chunk
  exceeds `max_size`. Covered by TV-304, TV-306.
- [ ] **SPEC-CHUNK-320** — Partition similarity construction follows the
  four-step procedure. Verify by inspection (TV-306 exercises step
  3's rescaling; TV-307/TV-308 exercise step 4).
- [ ] **SPEC-CHUNK-321** — Discourse-vector correction with two-stage
  safeguard. Covered by TV-311 (degenerate case forces fallback).
- [ ] **SPEC-CHUNK-322** — Heading-aware similarity modification.
  Covered by TV-307, TV-308.
- [ ] **SPEC-CHUNK-330** — Deterministic given solver. Run any TV
  twice and assert identical output (with a deterministic solver).
- [ ] **SPEC-CHUNK-340** — Short-circuit on single chunklet, total-fits,
  or empty input. Covered by TV-301, TV-302, TV-305.
- [ ] **SPEC-CHUNK-341** — Oversized chunklet rejected. Covered by TV-310.
- [ ] **SPEC-CHUNK-342** — Zero-norm embedding rejected. Covered by
  TV-309.
- [ ] **SPEC-CHUNK-343** — Solver failure raises. Verify by inspection
  (construct or mock a solver that reports failure).

## Late chunking (SPEC-CHUNK-4xx) — optional

If late chunking is not implemented, mark the section N/A. Otherwise:

- [ ] **SPEC-CHUNK-400** — One row per input sentence. Covered by
  TV-401, TV-402.
- [ ] **SPEC-CHUNK-401** — Fixed-dimensional rows; dimension equals
  the embedder's hidden size. Covered by TV-401.
- [ ] **SPEC-CHUNK-402** — Normalization controlled by the function's
  `normalize` parameter. Covered by TV-410.
- [ ] **SPEC-CHUNK-410** — Every sentence appears in exactly one
  segment's content range. Covered by TV-405, TV-406.
- [ ] **SPEC-CHUNK-411** — Greedy segment construction with backward
  preamble; first segment edge case. Covered by TV-405 (general
  case), TV-406 (first-segment empty preamble).
- [ ] **SPEC-CHUNK-412** — Per-segment encoding, largest-remainder
  apportionment, mean-pool, discard preamble rows. Covered by
  TV-408 (token-count alignment), TV-405 (preamble discard).
- [ ] **SPEC-CHUNK-420** — Per-sentence token counts match the
  embedder's tokenization of the joined input. Covered by TV-408.
- [ ] **SPEC-CHUNK-421** — Sentinel character validation (if using
  sentinel method). Covered by TV-407.
- [ ] **SPEC-CHUNK-430** — Output normalization honored. Covered by
  TV-410.
- [ ] **SPEC-CHUNK-440** — Deterministic given a deterministic embedder.
  Run TV-401 twice and assert identical output.
- [ ] **SPEC-CHUNK-450** — Single sentence input handled. Covered by
  TV-403.
- [ ] **SPEC-CHUNK-451** — Over-context-size sentence rejected.
  Covered by TV-404.
- [ ] **SPEC-CHUNK-452** — Very short sentence handling documented.
  Covered by TV-409.

## Contextual headings (SPEC-CHUNK-5xx) — optional

If the heading-paths helper is not implemented, mark the section N/A.
Otherwise:

- [ ] **SPEC-CHUNK-500** — `len(output) == len(chunks)`. Covered by
  TV-501, TV-502, TV-503.
- [ ] **SPEC-CHUNK-501** — Path is either `""` or heading lines
  joined by `HEADING_PATH_SEPARATOR`. Covered by TV-503, TV-504.
- [ ] **SPEC-CHUNK-502** — First chunk's path is always `""`. Covered
  by TV-505.
- [ ] **SPEC-CHUNK-510** — Heading stack updates on heading
  encounter; deeper slots clear. Covered by TV-504.
- [ ] **SPEC-CHUNK-511** — Path snapshotted before scanning the
  chunk's own headings. Covered by TV-505.
- [ ] **SPEC-CHUNK-512** — ATX heading detection only; `^#+\s` at
  line start. Covered by TV-507.
- [ ] **SPEC-CHUNK-513** — Path-string formatting. Covered by TV-508,
  TV-509.
- [ ] **SPEC-CHUNK-520** — Stack reset semantics on level rise.
  Covered by TV-504.
- [ ] **SPEC-CHUNK-530** — Deterministic. Run any TV twice and assert
  identical output.
- [ ] **SPEC-CHUNK-540** — Empty input. Covered by TV-501.
- [ ] **SPEC-CHUNK-541** — Document without headings. Covered by
  TV-502.
- [ ] **SPEC-CHUNK-542** — First chunk introduces the first heading.
  Covered by TV-505.
- [ ] **SPEC-CHUNK-543** — Heading levels skipped. Covered by TV-506.
- [ ] **SPEC-CHUNK-544** — More than 6 `#` characters → not a
  heading. Covered by TV-507.

## Public API contract

- [ ] The three required functions (`split_sentences`,
  `split_chunklets`, `split_chunks`) and the two optional helpers
  (`embed_with_late_chunking`, `heading_paths`) exist and have the
  documented signatures (or language-equivalents).
- [ ] Defaults match the spec (`min_len=4`, `max_size=2048`,
  `preamble_fraction=0.382`).
- [ ] All error cases produce distinguishable exception types or
  language-native error signals.
- [ ] When the implementation renames or restructures the public
  surface, it publishes a mapping from concrete names to SPEC-CHUNK
  IDs.
