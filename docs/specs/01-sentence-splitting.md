# Spec 01 — Sentence Splitting

Partition a Markdown document into sentences such that each sentence
is a contiguous substring of the document, sentences respect a
configurable length range, and structurally meaningful boundaries
(notably Markdown headings) are honored.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `document` | string (UTF-8) | yes | — | The Markdown document to split. |
| `min_len` | non-negative integer | no | `4` | Minimum characters per sentence. |
| `max_len` | positive integer or `None` | no | `None` | Maximum characters per sentence. `None` means no upper bound. |
| `known_boundary_probas` | per-character probability vector or callable producing one | no | the Markdown-heading boundary function (SPEC-CHUNK-113) | An override mechanism: positions where the caller already knows the boundary probability. See SPEC-CHUNK-112. |

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
- **SPEC-CHUNK-105** — Every sentence is at least `min_len` characters
  long.
- **SPEC-CHUNK-106** — When `max_len` is set, every sentence is at most
  `max_len` characters long.

## Behavior

### SPEC-CHUNK-110 — Boundary probabilities are the input

Sentence splitting operates on a per-character vector of *boundary
probabilities*. A boundary probability at index `k` represents the
probability that the character at index `k` is the *last* character of
a sentence (i.e. the next sentence begins at index `k+1`).

The vector has length equal to the number of characters in the
document.

### SPEC-CHUNK-111 — Predicted probabilities come from a model

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

### SPEC-CHUNK-112 — Known overrides take precedence

If the caller (or a default function) supplies *known* boundary
probabilities for some positions, those values override the predicted
values at exactly those positions. A position is "known" when its
value in the override vector is a finite number (not `NaN`). `NaN`
means "no override; use the predicted value".

The override vector has the same length as the predicted vector.

### SPEC-CHUNK-113 — Markdown headings are forced to be standalone sentences

The default `known_boundary_probas` function inspects the document's
Markdown structure and constructs a per-character override vector with:

- Probability `1` at the position **immediately before** the first
  character of every heading. (The heading starts a new sentence.)
- Probability `0` at every position **inside** the heading body
  (between first and last heading characters). (No sentence can split
  inside the heading.)
- Probability `1` at the position **of the last character** of every
  heading. (The heading ends a sentence.)
- `NaN` at all other positions. (Defer to predicted probabilities.)

A "heading" is determined by Markdown parsing: any token that opens a
heading element (ATX-style `# Heading` or Setext-style underlined
heading) and its corresponding content span.

The net effect: a Markdown heading is always exactly one sentence,
neither split internally nor joined to adjacent text.

### SPEC-CHUNK-114 — Whitespace is trailing, not leading

After the predicted and known probabilities are merged, the final
probability vector is adjusted so that whitespace attaches to the
preceding sentence rather than starting the next one. For every
maximal run of whitespace characters at positions `[i, j)` flanked by
non-whitespace characters:

- For positions `[i, j-1)` (every whitespace position except the last
  in the run), set the probability to the minimum probability in the
  run.
- For position `j-1` (the last whitespace position), set the
  probability to the maximum probability in the run.

This biases the splitter to break *after* whitespace, so the next
sentence starts at a non-whitespace character.

Two reasons:

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

### SPEC-CHUNK-115 — Splitting maximizes total score above threshold

Given the final per-character probability vector, sentence boundaries
are chosen to **maximize the sum of `(probability − BOUNDARY_SCORE_THRESHOLD)`**
over the selected boundary positions, subject to the length constraints
(SPEC-CHUNK-105, SPEC-CHUNK-106).

`BOUNDARY_SCORE_THRESHOLD` defaults to `0.25`, the recommended
operating point published for SaT's `-sm` model family (see
SPEC-CHUNK-111). Positions with probability above this value
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

### SPEC-CHUNK-116 — Two-pass max-length handling

When `max_len` is set, the implementation may first solve the
optimization with no max-length constraint, then for each resulting
sentence that exceeds `max_len`, re-run the optimization on that
sentence with the max-length constraint applied. This is a performance
optimization: the unconstrained problem admits a faster solution than
the constrained one.

The two-pass structure is implementation-defined behavior. A single
constrained solve is conforming, provided the final output satisfies
SPEC-CHUNK-105 and SPEC-CHUNK-106.

## Determinism and tie-breaking

### SPEC-CHUNK-120 — Deterministic given a deterministic segmenter

For a given document, configuration, and segmenter, the output is
deterministic across runs.

### SPEC-CHUNK-121 — Ties broken by first-found

When two partitions yield the same total score, the splitter selects
the one found first in its enumeration. Implementations using DP with
forward iteration produce the lexicographically-earliest set of
boundary indices; implementations using backward iteration may produce
the lexicographically-latest. Both are conforming. Test vectors must
not depend on this tie-breaker.

## Edge cases

### SPEC-CHUNK-130 — Document shorter than `min_len`

If `len(document) <= min_len`, return `[document]` as the single
sentence. No splitting occurs.

### SPEC-CHUNK-131 — Document cannot be split to satisfy length constraints

If no boundary placement satisfies both `min_len` and `max_len` (e.g.,
the document is longer than `max_len` but has no internal split
positions that yield two sentences each `≥ min_len`), the
implementation raises an error. The exact error type is
implementation-defined; the error message should indicate that no
valid partition exists.

### SPEC-CHUNK-132 — Document with no valid boundaries

If the predicted probabilities yield no positions above the threshold,
but the no-boundary partition (single sentence equal to the full
document) is itself valid under the length constraints, the splitter
returns `[document]`.

### SPEC-CHUNK-133 — Empty document

The behavior for an empty document (`""`) is implementation-defined.
Reasonable choices are returning `[""]` or `[]`. Document the choice.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `BOUNDARY_SCORE_THRESHOLD` | `0.25` | SPEC-CHUNK-115 |

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
- A sentence-segmentation model satisfying SPEC-CHUNK-111.
