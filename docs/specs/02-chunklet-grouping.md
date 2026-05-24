# Spec 02 — Chunklet Grouping

Partition an ordered sequence of sentences into *chunklets*, where each
chunklet is a contiguous group of sentences targeting roughly three
"statements" of information content and aligned to Markdown structure
(headings, paragraph starts, list openings).

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sentences` | list of strings | yes | — | An ordered sequence of sentences. Typically the output of stage 1. |
| `max_size` | positive integer | no | `DEFAULT_MAX_SIZE_CHARS` (`= 2048`) | Hard upper bound on chunklet length in characters. See U-CHUNK-203. |
| `boundary_cost` | callable | no | the default in SPEC-CHUNK-220 | Cost contributed by a chunklet's boundary probabilities. |
| `statement_cost` | callable | no | the default in SPEC-CHUNK-221 | Cost contributed by a chunklet's statement count. |

## Outputs

A list of strings (the chunklets). Each chunklet is the concatenation
of one or more contiguous input sentences.

- **SPEC-CHUNK-200** — `"".join(chunklets) == "".join(sentences)`.
- **SPEC-CHUNK-201** — Every chunklet is at most `max_size`
  characters. This is enforced as a hard constraint during
  optimization, not as a post-filter.
- **SPEC-CHUNK-202** — The number of chunklets is between 1 and
  `len(sentences)`.

## Behavior

### SPEC-CHUNK-210 — Optimization framing

Chunklet grouping is a partition optimization. Given the input
sentences `s[0..N-1]`, find the partition into chunklets that
minimizes the sum of per-chunklet costs:

```
total_cost = Σ ( boundary_cost(p[j..i]) + statement_cost(Σ statements[j..i]) )
```

where `p[j..i]` is the slice of per-sentence boundary probabilities
covering the chunklet, and `Σ statements[j..i]` is the chunklet's
total statement count. The sum runs over every chunklet `s[j..i]` in
the partition.

This is solvable in `O(N²)` time by dynamic programming (the standard
1-D segmentation DP). Any solver that finds the optimum is conforming.

### SPEC-CHUNK-220 — Default boundary cost

The default `boundary_cost` for a chunklet whose sentence boundary
probabilities are `p[0], p[1], ..., p[k-1]` (length `k`) is:

```
boundary_cost = (1 - p[0]) + sum(p[1:])
```

Interpretation:
- The chunklet is rewarded for *starting* at a strong boundary
  (`p[0]` high → `1 - p[0]` low).
- The chunklet is penalized for *swallowing* boundaries inside it
  (every internal `p[i]` adds to the cost).


The reimplementor may expose a way to override this function. If they
do, the default must match the formula above.

### SPEC-CHUNK-221 — Default statement cost

The default `statement_cost` for a chunklet containing `s` statements
(see SPEC-CHUNK-230 for what "statements" means) is:

```
statement_cost = (s - TARGET_STATEMENTS_PER_CHUNKLET)² / sqrt(max(s, STATEMENT_COST_FLOOR)) / 2
```

with `TARGET_STATEMENTS_PER_CHUNKLET = 3` and
`STATEMENT_COST_FLOOR = 1e-6`.

Interpretation:
- Minimum at `s = TARGET_STATEMENTS_PER_CHUNKLET` (target = 3
  statements per chunklet).
- Quadratic penalty for deviation from the target.
- `sqrt(s)` divisor flattens the cost slightly for large `s` (so a
  10-statement chunklet is not 49× worse than a 4-statement
  chunklet); for small `s` it would explode, hence the
  `STATEMENT_COST_FLOOR` clamp.

The target value `3` and the divisor structure are preserved as part
of the spec. See U-CHUNK-201 for the (heuristic) rationale behind
the target.

### SPEC-CHUNK-230 — "Statements" as a soft information-content measure

A sentence's *statement count* is a real-valued, document-relative
measure of its information content. It is computed from the sentence's
word count via a piecewise-linear function anchored at the document's
word-count quartiles:

Let `wc(s)` be the word count of sentence `s` (whitespace-separated
tokens). Let `q25` and `q75` be the 25th and 75th percentiles of
`wc(·)` across the document's sentences, with `q25` clamped to a
small positive value and `q75` clamped to be strictly greater than
`q25`.

The function is anchored at two design points:

| Named constant | Value | Meaning |
|---|---|---|
| `STATEMENTS_AT_Q25` | `0.75` | A sentence at the document's 25th-percentile word count contributes this many statements. |
| `QUARTILE_GAP_STATEMENTS` | `0.50` | Each quartile-gap in word count corresponds to this many additional statements. |

The statement count of a sentence with word count `n` is then:

```
if n ≤ q25:     STATEMENTS_AT_Q25 * n / q25
if n > q25:     STATEMENTS_AT_Q25 + QUARTILE_GAP_STATEMENTS * (n - q25) / (q75 - q25)
```

So:
- A sentence with `n = 0` words contributes `0` statements.
- A sentence with `n = q25` words contributes `0.75` statements.
- A sentence with `n = q75` words contributes `1.25` statements.
- A sentence with `n > q75` words contributes proportionally more,
  unbounded above.

> **Why these two constants?** They encode a clean design intent
> rather than arbitrary tuning: the *median* sentence contributes
> ≈ 1 statement (since the median sits halfway between q25 and q75,
> the formula gives `0.75 + 0.5 * 0.5 = 1.0`), and each quartile-gap
> step changes the contribution by `±0.25`. So the function is "one
> statement at the document's typical sentence, plus or minus a
> quarter-statement per quartile of word-count deviation." This is
> the resolution of U-CHUNK-202.

The shape: short sentences within the document's typical range
contribute less than one statement; long sentences contribute more;
the median sentence contributes ≈ 1 statement.

The reimplementor is not free to substitute a different
statement-counting function as the default. A custom function may be
exposed as a parameter, but the default must match.

### SPEC-CHUNK-240 — Per-sentence boundary probabilities from Markdown

Each sentence is assigned a single *boundary probability* indicating
how structurally strong its starting position is. The vector has
length equal to the number of sentences. Sentence `i`'s boundary
probability is determined by Markdown parsing of the concatenated
document according to the following ranked table:

| Markdown token opening on the same line as sentence `i`'s start | Named constant | Value |
|-----------------------------------------------------------------|----------------|-------|
| Heading (`heading_open`) | `BOUNDARY_STRENGTH_HEADING` | `1.00` |
| Blockquote (`blockquote_open`) | `BOUNDARY_STRENGTH_BLOCKQUOTE` | `0.75` |
| Paragraph (`paragraph_open`) | `BOUNDARY_STRENGTH_PARAGRAPH` | `0.50` |
| Bullet list (`bullet_list_open`) | `BOUNDARY_STRENGTH_LIST` | `0.25` |
| Ordered list (`ordered_list_open`) | `BOUNDARY_STRENGTH_LIST` | `0.25` |
| (none of the above) | — | `0.00` |

> **Why these specific values?** Only the *ordering* matters:
> `heading > blockquote > paragraph > list-item > nothing`. The
> chunklet-grouping DP uses these as relative weights in its boundary
> cost (SPEC-CHUNK-220), so any monotonically-ranked positive values
> with the same order would produce the same partitions. The
> particular choice of `1, 0.75, 0.5, 0.25` is the simplest
> evenly-spaced ranking that fits in `[0, 1]` and matches the
> probability convention used throughout the pipeline. Implementors
> may adjust the values to tune the *gaps* between ranks but should
> preserve the order.

When multiple token openings would assign a probability to the same
sentence, the *first* assignment wins (the iteration is in document
order, and reassignment to a sentence already assigned is skipped).

The token-type names above are the CommonMark / markdown-it token
type names. The reimplementor must use a parser that produces
equivalent token types (or map their parser's tokens to this table).

### SPEC-CHUNK-241 — Suppress consecutive non-zero boundaries

After the per-sentence boundary probabilities are assigned, the vector
is post-processed: within each maximal contiguous run of non-zero
probabilities, only the maximum value is kept; the others are set to
zero.

This encourages splitting at the *strongest* nearby structural
boundary, not at multiple weaker ones in a row.

Example:
- Before: `[0.0, 0.5, 0.75, 0.25, 0.0, 0.5, 0.0]`
- After:  `[0.0, 0.0, 0.75, 0.0,  0.0, 0.5, 0.0]`

(In the first run `[0.5, 0.75, 0.25]`, only `0.75` survives. In the
second run `[0.5]`, the lone value survives.)

## Determinism and tie-breaking

### SPEC-CHUNK-250 — Deterministic

Given a deterministic Markdown parser, chunklet grouping is fully
deterministic for the same inputs.

### SPEC-CHUNK-251 — Tie-breaking prefers smaller chunklets

When two partitions have equal total cost, prefer the one whose
*earlier* splits use the *earlier* possible split point. (Equivalently,
in DP terms: when extending the partition table, ties are broken in
favor of the smaller predecessor index.)

## Edge cases

### SPEC-CHUNK-260 — Empty input

For `sentences == []`, return `[]`.

### SPEC-CHUNK-261 — Single sentence

For `sentences == [s]`, return `[s]` regardless of `len(s)` relative
to `max_size`. (Stage 1 owns the `max_size`/`max_len` constraint at
the sentence level; stage 2 will not split a single sentence.)

### SPEC-CHUNK-262 — Total length within `max_size`

If the concatenated input fits in one chunklet (`sum(len(s) for s in
sentences) <= max_size`), the DP may still produce a multi-chunklet
partition if doing so reduces total cost. The size constraint is an
upper bound, not a forcing function.

### SPEC-CHUNK-263 — No valid partition (sentence exceeds max_size)

If any single input sentence exceeds `max_size`, no partition can
satisfy SPEC-CHUNK-201. The behavior is implementation-defined; the
reimplementor should either raise an explicit error or fall back to
placing the oversized sentence in its own chunklet (violating
SPEC-CHUNK-201). Stage 1 is responsible for ensuring this does not
happen by passing `max_len = max_size` when called upstream.

## Named constants

| Name | Value | Spec ref | Type |
|------|-------|----------|------|
| `DEFAULT_MAX_SIZE_CHARS` | `2048` | input table | tunable (heuristic; U-CHUNK-203) |
| `TARGET_STATEMENTS_PER_CHUNKLET` | `3` | SPEC-CHUNK-221 | tunable (heuristic; U-CHUNK-201) |
| `STATEMENT_COST_FLOOR` | `1e-6` | SPEC-CHUNK-221 | numerical safety |
| `STATEMENTS_AT_Q25` | `0.75` | SPEC-CHUNK-230 | design anchor (resolved; see SPEC-CHUNK-230) |
| `QUARTILE_GAP_STATEMENTS` | `0.50` | SPEC-CHUNK-230 | design anchor (resolved; see SPEC-CHUNK-230) |
| `BOUNDARY_STRENGTH_HEADING` | `1.00` | SPEC-CHUNK-240 | structural-cue ranking |
| `BOUNDARY_STRENGTH_BLOCKQUOTE` | `0.75` | SPEC-CHUNK-240 | structural-cue ranking |
| `BOUNDARY_STRENGTH_PARAGRAPH` | `0.50` | SPEC-CHUNK-240 | structural-cue ranking |
| `BOUNDARY_STRENGTH_LIST` | `0.25` | SPEC-CHUNK-240 | structural-cue ranking |

## Implementation-defined behavior

- Choice of DP implementation (forward vs. backward iteration; SciPy
  / Cython / pure Python).
- Whether to expose `boundary_cost` and `statement_cost` as
  user-overridable parameters or to hard-code the defaults.
- Whether to compute per-sentence boundary probabilities lazily or
  up-front.

## Unspecified behavior

- Behavior when `sentences` contains an empty string (`""`). The
  reimplementor should either filter empty strings out or treat them
  as zero-statement, zero-length contributions.
- Behavior when sentences contain trailing whitespace such that the
  Markdown parser sees a different structure than the caller
  expects. Stage 1's SPEC-CHUNK-114 should make this rare.

## Dependencies the implementor must satisfy

- A Markdown parser exposing per-token start lines (any CommonMark
  parser).
- A DP or equivalent optimization implementation over O(N²) candidate
  partitions.

## Uncertainties

### U-CHUNK-201 — Choice of target = 3 statements

`TARGET_STATEMENTS_PER_CHUNKLET = 3` in SPEC-CHUNK-221 is a
heuristic. The plausible intent: a chunklet of 3 statements roughly
corresponds to a paragraph of moderate-density prose — large enough
to carry a complete thought (so the embedding has enough content to
be discriminative) but small enough to remain topically coherent (so
the embedding's direction is unambiguous). This is a tuning choice,
not a derived constant; the implementor is free to expose it as a
configuration parameter and should default to `3` to match the
calibration of downstream stages.

### U-CHUNK-202 — Resolved

The two constants in SPEC-CHUNK-230 (`STATEMENTS_AT_Q25 = 0.75`,
`QUARTILE_GAP_STATEMENTS = 0.50`) encode a clean design intent: the
median sentence contributes ≈ 1 statement; each quartile-gap step
changes the contribution by ±0.25. See the "Why these two constants?"
explanation under SPEC-CHUNK-230.

### U-CHUNK-203 — Default `max_size = 2048` characters

The default chunklet (and chunk) size of `2048` characters is a
rule-of-thumb rather than a derived value. It produces chunklets of
roughly ~400-600 tokens for typical English prose, which fits
comfortably inside the context window of every commonly-used
embedding model (most allow ≥ 8k tokens). Implementors should tune
this for their corpus and embedder: shorter chunklets (≈ 1024 chars)
give finer-grained retrieval; longer chunklets (≈ 4096 chars) give
more context per retrieved unit.
