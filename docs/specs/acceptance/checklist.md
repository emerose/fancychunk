# Acceptance Checklist

An implementation conforms to fancychunk if it passes every check
below. The checklist mirrors the SPEC-CHUNK IDs in the specs.

For each numbered item:

1. Write a test that exercises the behavior.
2. Run it against the implementation.
3. Mark it `[x]` when passing.

A conforming implementation has every box checked.

## Cross-cutting (SPEC-CHUNK-9xx)

- [ ] **SPEC-CHUNK-900** — Concatenating any stage's outputs reconstructs
  its input byte-for-byte. Covered by TV-103, TV-203, TV-303.
- [ ] **SPEC-CHUNK-901** — Output is deterministic for a fixed input,
  configuration, and embedder. Run each test vector twice and assert
  identical output.
- [ ] **SPEC-CHUNK-902** — Size limits are upper bounds. Covered by
  TV-107, TV-204, TV-304.
- [ ] **SPEC-CHUNK-903** — Single-unit short-circuits trigger when input
  fits in one unit. Covered by TV-101, TV-202, TV-301, TV-302,
  TV-305.

## Sentence splitting (SPEC-CHUNK-1xx)

- [ ] **SPEC-CHUNK-100** — Concatenation reproduces document.
- [ ] **SPEC-CHUNK-101** — Every sentence has at least one non-whitespace
  character.
- [ ] **SPEC-CHUNK-102** — Only the first sentence may begin with
  whitespace. Covered by TV-106.
- [ ] **SPEC-CHUNK-105** — Sentence length `≥ min_len`.
- [ ] **SPEC-CHUNK-106** — Sentence length `≤ max_len` when set. Covered
  by TV-107.
- [ ] **SPEC-CHUNK-110** — Operates on per-character boundary probability
  vector.
- [ ] **SPEC-CHUNK-111** — Uses a sentence-segmentation model. (Verify
  the implementation exposes a swappable model interface.)
- [ ] **SPEC-CHUNK-112** — Known overrides take precedence over
  predicted values; `NaN` defers to predicted. Covered by TV-111.
- [ ] **SPEC-CHUNK-113** — Markdown headings are forced to be standalone
  sentences. Covered by TV-104, TV-105, TV-112.
- [ ] **SPEC-CHUNK-114** — Whitespace is trailing, not leading. Covered
  by TV-106.
- [ ] **SPEC-CHUNK-115** — Splitting maximizes
  `Σ (probability − BOUNDARY_SCORE_THRESHOLD)` over chosen boundaries.
- [ ] **SPEC-CHUNK-116** — Max-length handling produces sentences
  `≤ max_len`. (Two-pass vs one-pass is implementation-defined.)
- [ ] **SPEC-CHUNK-120** — Deterministic.
- [ ] **SPEC-CHUNK-121** — Ties broken consistently.
- [ ] **SPEC-CHUNK-130** — Document shorter than `min_len` returns single
  sentence. Covered by TV-101.
- [ ] **SPEC-CHUNK-131** — Unsatisfiable length constraints raise.
  Covered by TV-113.
- [ ] **SPEC-CHUNK-132** — No boundary above threshold + no-boundary
  case valid → single sentence.
- [ ] **SPEC-CHUNK-133** — Empty document behavior is documented (either
  `[]` or `[""]`). Covered by TV-110.

## Chunklet grouping (SPEC-CHUNK-2xx)

- [ ] **SPEC-CHUNK-200** — Concatenation reproduces input sentences.
  Covered by TV-203.
- [ ] **SPEC-CHUNK-201** — Every chunklet `≤ max_size`. Covered by TV-204.
- [ ] **SPEC-CHUNK-202** — Number of chunklets in `[1, len(sentences)]`.
- [ ] **SPEC-CHUNK-210** — Optimization minimizes
  `Σ (boundary_cost + statement_cost)`.
- [ ] **SPEC-CHUNK-220** — Default boundary cost is
  `(1 - p[0]) + sum(p[1:])`.
- [ ] **SPEC-CHUNK-221** — Default statement cost is
  `(s - 3)² / sqrt(max(s, 1e-6)) / 2`.
- [ ] **SPEC-CHUNK-230** — Statement count is the documented piecewise
  function over word count, anchored at document `q25`/`q75`.
  Covered by TV-207.
- [ ] **SPEC-CHUNK-240** — Per-sentence boundary probabilities follow
  the Markdown token-type table. Covered by TV-205, TV-208, TV-209.
- [ ] **SPEC-CHUNK-241** — Consecutive non-zero boundaries: only the max
  survives. Covered by TV-208.
- [ ] **SPEC-CHUNK-250** — Deterministic.
- [ ] **SPEC-CHUNK-251** — Ties broken in favor of earliest split point.
  Covered by TV-210.
- [ ] **SPEC-CHUNK-260** — Empty input returns `[]`. Covered by TV-201.
- [ ] **SPEC-CHUNK-261** — Single sentence returns single chunklet.
  Covered by TV-202.
- [ ] **SPEC-CHUNK-262** — Total length within `max_size` may still
  produce multi-chunklet partition.
- [ ] **SPEC-CHUNK-263** — Sentence exceeding `max_size`: behavior is
  documented.

## Semantic chunking (SPEC-CHUNK-3xx)

- [ ] **SPEC-CHUNK-300** — Concatenation reproduces input chunklets.
  Covered by TV-303.
- [ ] **SPEC-CHUNK-301** — Every chunk `≤ max_size`. Covered by TV-304.
- [ ] **SPEC-CHUNK-302** — Embedding rows preserved in order across the
  partition.
- [ ] **SPEC-CHUNK-310** — Optimization minimizes total partition
  similarity.
- [ ] **SPEC-CHUNK-311** — Covering constraint enforced; no chunk
  exceeds `max_size`. Covered by TV-304, TV-306.
- [ ] **SPEC-CHUNK-320** — Partition similarity construction follows the
  four-step procedure.
- [ ] **SPEC-CHUNK-321** — Discourse-vector correction with safeguard.
  Covered by TV-311.
- [ ] **SPEC-CHUNK-322** — Heading-aware similarity modification.
  Covered by TV-307, TV-308.
- [ ] **SPEC-CHUNK-330** — Deterministic given solver.
- [ ] **SPEC-CHUNK-340** — Short-circuit on single chunklet or
  total-fits. Covered by TV-301, TV-302, TV-305.
- [ ] **SPEC-CHUNK-341** — Oversized chunklet rejected. Covered by TV-310.
- [ ] **SPEC-CHUNK-342** — Zero-norm embedding rejected. Covered by
  TV-309.
- [ ] **SPEC-CHUNK-343** — Solver failure raises.

## Late chunking (SPEC-CHUNK-4xx) — optional

If late chunking is not implemented, mark the section N/A. Otherwise:

- [ ] **SPEC-CHUNK-400** — One row per input sentence.
- [ ] **SPEC-CHUNK-401** — Fixed-dimensional rows.
- [ ] **SPEC-CHUNK-402** — Optional L2 normalization honored.
- [ ] **SPEC-CHUNK-410** — Every sentence appears in exactly one
  segment's content range.
- [ ] **SPEC-CHUNK-411** — Greedy segment construction with backward
  preamble.
- [ ] **SPEC-CHUNK-412** — Per-segment encoding, largest-remainder
  apportionment, mean-pool, discard preamble rows.
- [ ] **SPEC-CHUNK-420** — Per-sentence token counts match the
  embedder's tokenization of the joined input.
- [ ] **SPEC-CHUNK-421** — Sentinel character validation (if using
  sentinel method).
- [ ] **SPEC-CHUNK-430** — Output normalization honored.
- [ ] **SPEC-CHUNK-440** — Deterministic given a deterministic embedder.
- [ ] **SPEC-CHUNK-450** — Single sentence input handled.
- [ ] **SPEC-CHUNK-451** — Over-context-size sentence rejected.
- [ ] **SPEC-CHUNK-452** — Very short sentence handling documented.

## Contextual headings (SPEC-CHUNK-5xx) — optional

If the heading-paths helper is not implemented, mark the section N/A.
Otherwise:

- [ ] **SPEC-CHUNK-500** — `len(output) == len(chunks)`. Covered by
  TV-501, TV-502, TV-503.
- [ ] **SPEC-CHUNK-501** — Path is either `""` or heading lines
  joined by `HEADING_PATH_SEPARATOR`. Covered by TV-503, TV-504.
- [ ] **SPEC-CHUNK-502** — First chunk's path is `""`. Covered by
  TV-505.
- [ ] **SPEC-CHUNK-510** — Heading stack updates on heading
  encounter; deeper slots clear. Covered by TV-504.
- [ ] **SPEC-CHUNK-511** — Path snapshotted before scanning the
  chunk's own headings. Covered by TV-505.
- [ ] **SPEC-CHUNK-512** — ATX heading detection only; `^#+\s` at
  line start. Covered by TV-507.
- [ ] **SPEC-CHUNK-513** — Path-string formatting. Covered by TV-508.
- [ ] **SPEC-CHUNK-520** — Stack reset semantics on level rise.
  Covered by TV-504.
- [ ] **SPEC-CHUNK-530** — Deterministic.
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
