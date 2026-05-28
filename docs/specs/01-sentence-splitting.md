# Spec 01 — Sentence Splitting

Partition a Markdown document into sentences such that each sentence
is a contiguous substring of the document, sentences respect a
configurable length range, and structurally meaningful boundaries
(notably Markdown headings) are honored.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `document` | string (UTF-8) | yes | — | The Markdown document to split. |
| `min_len` | non-negative integer | no | `4` | Minimum characters per sentence. See "About `min_len`" below. |
| `max_len` | positive integer or `None` | no | `None` | Maximum characters per sentence. `None` means no upper bound. |
| `known_boundary_probas` | per-character probability vector or callable producing one | no | the Markdown-heading boundary function (SPEC-CHUNK-108) | An override mechanism: positions where the caller already knows the boundary probability. See SPEC-CHUNK-107. |

> **About `min_len = 4`.** Sentence segmenters sometimes emit
> degenerate sentences when given noisy input: a stray punctuation
> mark, an isolated letter, or a typesetting artifact. `min_len = 4`
> excludes those cases without rejecting legitimately short sentences
> like `"OK."` (3 chars — would round-trip as a single document but
> rarely arises as a *result* of splitting a longer one) or `"Done."`
> (5 chars — safely above the floor). Tune downward to allow more
> aggressive splitting on terse content; tune upward if a particular
> segmenter is noisy.

## Outputs

A list of strings (the sentences), satisfying:

- **SPEC-CHUNK-100** — Concatenation reproduces the input exactly: the
  concatenation of all sentences, in order, equals `document`
  byte-for-byte. No normalization, escaping, or whitespace adjustment
  is applied.
- **SPEC-CHUNK-101** — Every sentence contains at least one
  non-whitespace character.
- **SPEC-CHUNK-102** — No sentence except the first begins with
  whitespace. The first sentence may begin with whitespace only if the
  document itself does.
- **SPEC-CHUNK-103** — Every sentence is at least `min_len` characters
  long. (Exception: the short-circuit in SPEC-CHUNK-114 returns a
  single sentence shorter than `min_len` when the entire document is.)
- **SPEC-CHUNK-104** — When `max_len` is set, every sentence is at most
  `max_len` characters long. (Exception: the short-circuit in
  SPEC-CHUNK-114 returns a single sentence longer than `max_len` when
  the document itself is `≤ min_len` characters; see SPEC-CHUNK-114
  for the precedence.)

## Behavior

### SPEC-CHUNK-105 — Boundary probabilities are the input

Sentence splitting operates on a per-character vector of *boundary
probabilities*. A boundary probability at index `k` represents the
probability that the character at index `k` is the *last* character of
a sentence (i.e. the next sentence begins at index `k+1`).

The vector has length `N` (the document's character count), indexed
`0` through `N - 1`. Any operation that would index outside this
range (e.g., position `-1` or position `N`) is a no-op.

### SPEC-CHUNK-106 — Predicted probabilities come from a model

Boundary probabilities are produced by a sentence-segmentation model
that, given a document, returns a per-character probability vector.
The choice of model is implementation-defined; any model whose output
is a length-N vector of values in `[0, 1]` (where N is the character
count) is conforming.

Any segmenter — rule-based, statistical, or learned — is acceptable.
Quality of sentence boundaries will track the quality of the
segmenter, but the contract here is purely the vector shape.

> **A natural choice** is a model from the
> [SaT (Segment any Text)](https://arxiv.org/abs/2406.16678) family
> (Frohmann et al., 2024), available via
> [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite).
> SaT is multilingual, punctuation-agnostic, and exposes per-character
> boundary probabilities directly — matching the contract above
> without any adaptation. Lighter alternatives (rule-based splitters
> from `nltk`, `spacy`, or `blingfire`) work too but will produce
> probability vectors with most values clamped to 0 or 1.

### SPEC-CHUNK-107 — Known overrides take precedence

If the caller (or a default function) supplies *known* boundary
probabilities for some positions, those values override the predicted
values at exactly those positions. A position is "known" when its
value in the override vector is a finite number (not `NaN`). `NaN`
means "no override; use the predicted value".

The override vector has the same length as the predicted vector.

### SPEC-CHUNK-108 — Markdown headings are forced to be standalone sentences

The default `known_boundary_probas` function inspects the document's
Markdown structure and constructs a per-character override vector
based on each heading's character span.

A heading's character span runs from the **first character of the
heading marker** (e.g., the `#` in `# Hello`) through the **final
non-whitespace character of the heading text** (e.g., the `o` in
`Hello`). Trailing whitespace within or after the heading line
(newlines, blank lines that the Markdown parser includes in the
heading token) is *not* part of the span; SPEC-CHUNK-109's
whitespace-trailing rule handles those uniformly.

For every heading's span at character positions `[first, last]`:

- Set probability `1` at position `first - 1` (the character before
  the heading marker). The heading starts a new sentence. **Edge
  case:** if `first == 0` (the heading is at the very start of the
  document), skip this write — there is no position `-1`.
- Set probability `0` at every position `[first, last - 1]` (inside
  the heading body). No sentence can split inside the heading.
- Set probability `1` at position `last` (the final non-whitespace
  character of the heading text). The heading ends a sentence.
  **Edge case:** if the heading line has no non-whitespace text
  after the marker (e.g., `# \n` — a `#` followed by whitespace and
  no title), the body span is degenerate. Skip the in-body and
  last-position writes; the predicted probabilities determine
  sentence boundaries around the empty heading.

All other positions are `NaN` (defer to predicted probabilities).

A "heading" is determined by Markdown parsing: any token that opens a
heading element (ATX-style `# Heading` or Setext-style underlined
heading) and its corresponding content span.

The net effect: a Markdown heading is always exactly one sentence,
neither split internally nor joined to adjacent text.

### SPEC-CHUNK-109 — Whitespace is trailing, not leading

After the predicted and known probabilities are merged, the final
probability vector is adjusted so that whitespace attaches to the
preceding sentence rather than starting the next one. For every
maximal run of whitespace characters at positions `[i, j)` flanked by
non-whitespace characters, the *extended run* is `[i', j)` where
`i' = i - 1` when `i > 0` (the immediately-preceding non-whitespace
position is included in the extended run) and `i' = i` otherwise.
Then:

- Let `M = max(p[i'], p[i'+1], ..., p[j-1])`.
- Let `m = min` over the same positions.
- For positions `[i', j-1)`, set the probability to `m`.
- For position `j-1` (the last whitespace position), set the
  probability to `M`.

This biases the splitter to break *after* whitespace, so the next
sentence starts at a non-whitespace character.

Three reasons:

1. **Reader expectation.** A printed sentence starts with a word, not
   with the space separating it from the previous sentence. Storing
   sentences with leading whitespace would surprise downstream
   consumers (display, search-result snippets, highlighting).

2. **Unambiguous re-concatenation.** The boundary between two
   sentences sits in a whitespace run. Without this rule, the
   splitter could place the boundary anywhere inside the run, so
   `"".join(sentences)` would round-trip but the *split points*
   would be ambiguous. Pinning every internal whitespace position to
   the run's minimum and the final position to the run's maximum
   means whichever boundary the model preferred ends up at the same
   place: just before the next non-whitespace character. Min/max
   specifically (rather than, say, zero/one) preserves the model's
   *relative* preferences across runs while making each run's
   boundary location deterministic.

3. **Transport of boundary signal across whitespace.** Including
   `i' = i - 1` in the extended run is what carries a structural
   boundary signal — most importantly the heading-end `1.0` placed
   by SPEC-CHUNK-108 — across the trailing whitespace to the last
   position in the run. Without the inclusion of `i - 1`, a heading
   like `# Title\n\n` would leave its boundary at the heading's
   last non-whitespace character (the `e`), and the trailing
   newlines would begin the next sentence, violating SPEC-CHUNK-102.
   The extended-run rule shifts the boundary one whitespace run
   forward so the heading sentence consumes its blank-line tail.

### SPEC-CHUNK-110 — Splitting maximizes total score above threshold

Given the final per-character probability vector, sentence boundaries
are chosen to **maximize the sum of
`(probability[k] − BOUNDARY_SCORE_THRESHOLD)`** over the selected
boundary positions `k`, subject to the length constraints
(SPEC-CHUNK-103, SPEC-CHUNK-104).

`BOUNDARY_SCORE_THRESHOLD` defaults to `0.25`, the recommended
operating point published for SaT's `-sm` model family (see
SPEC-CHUNK-106). Positions with probability above this value
contribute positive score (the splitter is rewarded for placing a
boundary there); positions below contribute negative score (the
splitter is penalized).

If a different sentence-segmentation model is used, the threshold
should be recalibrated to that model's recommended operating point.
The exact value is meaningful only relative to the model's
calibration — it is not a universal "goodness" cutoff.

This is an optimization problem with `O(N)` candidate boundary
positions and a length-range constraint coupling them. The standard
solution is dynamic programming, but any solver that finds the optimum
is conforming.

**Output structure.** Given the selected boundary positions
`{k₁ < k₂ < … < kₘ}` (possibly empty), the returned sentences are:

```
document[0 : k₁ + 1],
document[k₁ + 1 : k₂ + 1],
…,
document[kₘ + 1 : N]
```

Position `N - 1` (the document's final character) is always a
sentence end and is not represented in the boundary set — it is
implicit in the final slice's upper bound. The empty boundary set is
a valid solution: it yields the single-sentence partition
`[document]` whenever that satisfies SPEC-CHUNK-103/104.

### SPEC-CHUNK-111 — Two-pass max-length handling

When `max_len` is set, the implementation may first solve the
optimization with no max-length constraint, then for each resulting
sentence that exceeds `max_len`, re-run the optimization on that
sentence with the max-length constraint applied. This is a performance
optimization: the unconstrained problem admits a faster solution than
the constrained one.

The two-pass structure is implementation-defined behavior. A single
constrained solve is conforming, provided the final output satisfies
SPEC-CHUNK-103 and SPEC-CHUNK-104.

> **A note on segmenter quality.** Boundary quality tracks the
> segmenter (SPEC-CHUNK-106). Lighter SaT checkpoints mis-segment some
> scientific-prose constructs — an abbreviation reference like
> `"Tab. TABREF21"` (boundary on the abbreviation period) or a year
> before a capitalised word like `"SemEval-2014 Task"` (boundary on the
> digit). The default implementation addresses these by choosing a
> sufficiently high-capacity checkpoint (`sat-9l-sm`) and the
> `weighting="hat"` inference mode (which de-weights low-context window
> edges) rather than by post-processing the probability vector. A
> caller using a lighter segmenter (e.g. `sat-3l-sm`) may reintroduce
> these artifacts.

## Determinism and tie-breaking

### SPEC-CHUNK-112 — Deterministic given a deterministic segmenter

For a given document, configuration, and segmenter, the output is
deterministic across runs.

### SPEC-CHUNK-113 — Ties broken by smallest predecessor index

When two partitions yield the same total score, the DP picks the one
obtained by always choosing the **smallest** predecessor index `j`
whenever multiple `j`'s achieve the maximum of
`dp[j] + score(j..i)` during table construction, where `score(j..i)`
is the score contribution of placing a boundary at position `i - 1`
(equal to `probability[i-1] − BOUNDARY_SCORE_THRESHOLD` per
SPEC-CHUNK-110, or `0` for the case where no boundary is placed at
`i - 1`).

Behaviorally, this rule has two consequences worth knowing:

- Among equal-score partitions, the one with the **fewest boundaries**
  is preferred (the smallest-`j` choice at the final step is `j = 0`
  when all scores tie, yielding the single-sentence partition).
- Among equal-score partitions of the same size, the one whose **last
  boundary is at the smallest position** is preferred (applied
  recursively for earlier boundaries).

Stage 2 (SPEC-CHUNK-251) uses the same smallest-predecessor rule
under a minimization objective — the rule is the same; only the
objective's direction differs between the two stages.
Determinism is required so the output is reproducible across runs.

## Edge cases

### SPEC-CHUNK-114 — Document shorter than `min_len`

If `len(document) <= min_len`, return `[document]` as the single
sentence. No splitting occurs.

This short-circuit takes precedence over both length constraints: it
may return a sentence shorter than `min_len` (the document itself
is) and, in the unusual case where `max_len` is set very small
(`max_len < len(document) <= min_len`), longer than `max_len`. If
the configuration is genuinely unsatisfiable in this short-circuit
case, the caller should detect it before calling
(`min_len > max_len` is a configuration error).

### SPEC-CHUNK-115 — Document cannot be split to satisfy length constraints

If no boundary placement satisfies both `min_len` and `max_len` (e.g.,
the document is longer than `max_len` but has no internal split
positions that yield two sentences each `≥ min_len`), the
implementation raises an error. The exact error type is
implementation-defined; the error message should indicate that no
valid partition exists.

### SPEC-CHUNK-116 — Document with no valid boundaries

If the predicted probabilities yield no positions above the threshold,
but the no-boundary partition (single sentence equal to the full
document) is itself valid under the length constraints, the splitter
returns `[document]`.

### SPEC-CHUNK-117 — Empty document or whitespace-only document

For an empty document (`""`) or a document containing only whitespace
characters, return `[]`. This matches stage 2's empty-input convention
(SPEC-CHUNK-260) and stage 3's (SPEC-CHUNK-340), so a content-free
document flows through the whole pipeline as empty lists, and avoids
producing a single-sentence output that would violate SPEC-CHUNK-101
(every sentence contains at least one non-whitespace character).
The whitespace-only case is grouped with the empty case rather than
the SPEC-CHUNK-114 short-circuit because the SPEC-CHUNK-114 path
returns the document as a single sentence, which would itself violate
SPEC-CHUNK-101 when the document is all whitespace.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `BOUNDARY_SCORE_THRESHOLD` | `0.25` | SPEC-CHUNK-110 |

## Implementation-defined behavior

- Choice of sentence-segmentation model.
- Whether to use one solve or two for max-length handling.
- Memory representation of the probability vector.
- Whether to support per-call override of `BOUNDARY_SCORE_THRESHOLD`
  or hard-code it.

## Unspecified behavior

- Behavior on documents containing non-UTF-8 byte sequences (input is
  required to be a valid string).
- Behavior when the override callable raises (caller's responsibility).

## Dependencies

- A Markdown parser that exposes heading token positions (start and
  end line of every heading). Any CommonMark-conforming parser is
  sufficient.
- A sentence-segmentation model satisfying SPEC-CHUNK-106.
