# Spec 02 — Chunklet Grouping

Partition an ordered sequence of sentences into *chunklets*, where each
chunklet is a contiguous group of sentences targeting roughly three
"statements" of information content and aligned to Markdown structure
(headings, paragraph starts, list openings).

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sentences` | list of strings | yes | — | An ordered sequence of sentences. Typically the output of stage 1. |
| `max_size` | positive integer | no | `DEFAULT_MAX_SIZE_CHARS` (`= 2048`) | Hard upper bound on chunklet length in characters. |
| `boundary_cost` | callable taking a length-`k` probability vector (one entry per sentence in the chunklet) and returning a non-negative float | no | the default in SPEC-CHUNK-220 | Cost contributed by a chunklet's boundary probabilities. |
| `statement_cost` | callable taking a non-negative float (the chunklet's total statement count) and returning a non-negative float | no | the default in SPEC-CHUNK-221 | Cost contributed by a chunklet's statement count. |

> **About `DEFAULT_MAX_SIZE_CHARS = 2048`.** This is a rule-of-thumb
> rather than a derived value. It produces chunklets of roughly
> ~400-600 tokens for typical English prose, which fits comfortably
> inside the context window of every commonly-used embedding model.
> The *order of magnitude* — a few hundred tokens per retrievable
> unit — is well-supported by retrieval-benchmark literature (MTEB,
> BEIR): retrieval quality typically peaks somewhere in the
> ~256–768 token range and degrades both for very short units (low
> recall, fragments) and very long units (low precision, mixed
> topics). Implementations should tune for their corpus and embedder:
> shorter chunklets (≈ 1024 chars) give finer-grained retrieval;
> longer chunklets (≈ 4096 chars) give more context per retrieved
> unit.

## Outputs

A list of strings (the chunklets). Each chunklet is the concatenation
of one or more contiguous input sentences.

- **SPEC-CHUNK-200** — `"".join(chunklets) == "".join(sentences)`.
- **SPEC-CHUNK-201** — Every chunklet is at most `max_size`
  characters. This is enforced as a hard constraint during
  optimization, not as a post-filter.
- **SPEC-CHUNK-202** — The number of chunklets is between `1` and
  `len(sentences)`, except for the empty-input case (SPEC-CHUNK-260)
  which returns `0` chunklets.

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

A custom `boundary_cost` function may be exposed as a parameter; the
default must match the formula above.

### SPEC-CHUNK-221 — Default statement cost

The default `statement_cost` for a chunklet containing `s` statements
(see SPEC-CHUNK-230 for what "statements" means) is:

```
statement_cost = STATEMENT_COST_SCALE * (s - TARGET_STATEMENTS_PER_CHUNKLET)² / sqrt(max(s, STATEMENT_COST_FLOOR))
```

with `TARGET_STATEMENTS_PER_CHUNKLET = 3`,
`STATEMENT_COST_FLOOR = 1e-6`, and `STATEMENT_COST_SCALE = 0.5`.

Interpretation:
- Minimum at `s = TARGET_STATEMENTS_PER_CHUNKLET` (target = 3
  statements per chunklet).
- Quadratic penalty for deviation from the target.
- `sqrt(s)` divisor flattens the cost slightly for large `s` (so a
  10-statement chunklet is not 49× worse than a 4-statement
  chunklet); for `s` near 0 the denominator approaches zero and the
  quotient is undefined, so `STATEMENT_COST_FLOOR` clamps `s` from
  below to keep the denominator strictly positive.
- `STATEMENT_COST_SCALE = 0.5` is an overall scale factor that sets
  the magnitude of the statement cost relative to the boundary cost
  (SPEC-CHUNK-220), which competes with it additively in the DP.
  Useful tuning range: roughly `[0.1, 2.0]`. Below ≈ `0.1` the
  boundary cost dominates and chunklets snap to structural
  boundaries regardless of statement balance; above ≈ `2.0` the
  statement cost dominates and the optimizer ignores structural
  cues. The factor doesn't change the minimizer for any single
  chunklet but shifts the relative weighting of the two cost
  components.

> **About `TARGET_STATEMENTS_PER_CHUNKLET = 3`.** This is a heuristic:
> a chunklet of 3 statements roughly corresponds to a paragraph of
> moderate-density prose — large enough to carry a complete thought
> (so the embedding has enough content to be discriminative), small
> enough to remain topically coherent (so the embedding's direction
> is unambiguous). It may be exposed as a configuration parameter;
> default to `3` because downstream stages are calibrated around it.

### SPEC-CHUNK-230 — "Statements" as a soft information-content measure

A sentence's *statement count* is a real-valued, document-relative
measure of its information content. It is computed from the sentence's
word count via a piecewise-linear function anchored at the document's
word-count quartiles.

Let `wc(s)` be the word count of sentence `s` (whitespace-separated
tokens). Let `q25` and `q75` be the 25th and 75th percentiles of
`wc(·)` across the document's sentences, computed by **linear
interpolation between the two nearest ranks** (NumPy's default
percentile method; equivalent to R's type 7). Clamp the results:

- `q25 = max(q25, MIN_Q25_WORDS)` with `MIN_Q25_WORDS = 1.0`. This
  prevents division by zero in the piecewise function when the
  document is dominated by zero- or one-word sentences.
- `q75 = max(q75, q25 + MIN_Q25_WORDS)`. The same `MIN_Q25_WORDS`
  value also serves as the minimum quartile gap, guaranteeing a
  strictly-positive denominator `(q75 - q25)` in the upper branch
  of the piecewise function. The two uses are independent — one
  floors q25, the other floors the q75-minus-q25 gap — but both
  happen to want the same magnitude, so a single constant suffices.

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

The two constants encode a clean design intent: the *median* sentence
contributes ≈ 1 statement (since the median sits halfway between
q25 and q75, the formula gives `0.75 + 0.5 * 0.5 = 1.0`), and each
quartile-gap step changes the contribution by `±0.25`. So the function
reads as "one statement at the document's typical sentence, plus or
minus a quarter-statement per quartile of word-count deviation."

A custom statement-counting function may be exposed as a parameter,
but the default must match the formula above.

### SPEC-CHUNK-240 — Per-sentence boundary probabilities from Markdown

Each sentence is assigned a single *boundary probability* indicating
how structurally strong its starting position is. The vector has
length equal to the number of sentences.

**Matching rule.** For each sentence `i`, gather every token (of
the table's listed types) that opens on the line containing sentence
`i`'s first non-whitespace character — but *only if sentence `i`
opens that block*, i.e. only whitespace precedes sentence `i`'s first
non-whitespace character on that line. A block opener begins at the
first non-whitespace character of its line, so a sentence whose start
is preceded by other text on the same line is *interior* to the block
and scores `0.00`. This case is common when a whole paragraph is a
single unwrapped line holding several sentences: only the first earns
`paragraph_open` strength; the rest are interior. (Without this guard
every sentence in a one-line paragraph would inherit `paragraph_open`,
leaving no `0.00` separators between blocks, so SPEC-CHUNK-241
suppression would collapse the document to a single surviving boundary
and discard every structural cue after the first.) Apply these rules
in order:

1. If any of `heading_open`, `blockquote_open` opens on the line,
   take the **strongest** of those that apply per the table below
   (heading > blockquote). Stop.
2. Otherwise, if any `bullet_list_open` or `ordered_list_open` opens
   on the line, sentence `i` gets `BOUNDARY_STRENGTH_LIST` —
   suppressing an accompanying `paragraph_open` that most parsers
   emit for the first item of a list. Stop.
3. Otherwise, if `paragraph_open` opens on the line, sentence `i`
   gets `BOUNDARY_STRENGTH_PARAGRAPH`.
4. Otherwise, the probability is `0.00`.

The two-stage structure ensures a blockquote that contains a nested
list (e.g., `> - item`) keeps its blockquote strength `0.75` rather
than being demoted to `0.25` by the list cue — consistent with the
ranking documented below.

The mapping from token type to probability:

| Markdown token opening on the same line as sentence `i`'s start | Named constant | Value |
|-----------------------------------------------------------------|----------------|-------|
| Heading (`heading_open`) | `BOUNDARY_STRENGTH_HEADING` | `1.00` |
| Blockquote (`blockquote_open`) | `BOUNDARY_STRENGTH_BLOCKQUOTE` | `0.75` |
| Paragraph (`paragraph_open`) | `BOUNDARY_STRENGTH_PARAGRAPH` | `0.50` |
| Bullet list (`bullet_list_open`) | `BOUNDARY_STRENGTH_LIST` | `0.25` |
| Ordered list (`ordered_list_open`) | `BOUNDARY_STRENGTH_LIST` | `0.25` |
| (none of the above) | — | `0.00` |

The magnitudes (not just the order) directly enter the boundary cost
in SPEC-CHUNK-220: a heading worth `1.00` and a paragraph worth
`0.50` produce noticeably different costs from, say, `1.00` and
`0.90`, so the gaps matter. The chosen values are evenly-spaced
rule-of-thumb weights on the `[0, 1]` probability scale, with the
ranking `heading > blockquote > paragraph > list-item > nothing`.
Implementations may tune the values; tuning changes optimization
outcomes.

Blockquote outranks paragraph because a blockquote shift almost
always marks a quotation boundary or an attribution change — a
topic-relevant break — whereas a paragraph break can also occur for
purely visual or rhythmic reasons within a single topic.

The token-type names above are markdown-it's token-stream type
names. AST-based parsers (which traverse a tree rather than emit a
token stream) must map their node types equivalently — e.g., a
"heading" AST node corresponds to `heading_open`.

### SPEC-CHUNK-241 — Suppress consecutive non-zero boundaries

After the per-sentence boundary probabilities are assigned, the vector
is post-processed: within each maximal contiguous run of non-zero
probabilities, only the maximum value is kept; the others are set to
zero. A run of length 1 (a singleton) is its own maximum and survives
unchanged. If multiple positions in a run share the maximum value,
the *earliest* such position survives (matching the deterministic
tie-breaking rule of SPEC-CHUNK-251).

This encourages splitting at the *strongest* nearby structural
boundary, not at multiple weaker ones in a row.

Example:
- Before: `[0.0, 0.5, 0.75, 0.25, 0.0, 0.5, 0.0]`
- After:  `[0.0, 0.0, 0.75, 0.0,  0.0, 0.5, 0.0]`

In the first run `[0.5, 0.75, 0.25]`, only `0.75` survives. In the
second run `[0.5]`, the lone value survives.

## Determinism and tie-breaking

### SPEC-CHUNK-250 — Deterministic

Given a deterministic Markdown parser, chunklet grouping is fully
deterministic for the same inputs.

### SPEC-CHUNK-251 — Tie-breaking is deterministic

When two partitions have equal total cost, the DP picks the one
obtained by always choosing the **smallest** predecessor index `j`
whenever multiple `j`'s achieve the minimum of `dp[j] + cost(j..i)`
during table construction.

Behaviorally, this rule has two consequences worth knowing:

- Among equal-cost partitions, the one with the **fewest chunklets**
  is preferred (the smallest-`j` choice at the final step is `j = 0`
  when all costs tie, yielding the single-chunklet partition).
- Among equal-cost partitions of the same size, the one whose **last
  split is at the smallest sentence index** is preferred (applied
  recursively for earlier splits).

Determinism is required so the output is reproducible across runs.

## Edge cases

### SPEC-CHUNK-260 — Empty input

For `sentences == []`, return `[]`.

### SPEC-CHUNK-261 — Single sentence

For `sentences == [s]`, return `[s]` regardless of `len(s)` relative
to `max_size`. Stage 1 owns the size constraint at the sentence
level; stage 2 will not split a single sentence.

### SPEC-CHUNK-262 — Total length within `max_size`

If the concatenated input fits in one chunklet
(`sum(len(s) for s in sentences) <= max_size`), the DP may still
produce a multi-chunklet partition if doing so reduces total cost.
The size constraint is an upper bound, not a forcing function.

### SPEC-CHUNK-263 — No valid partition (sentence exceeds max_size)

If any single input sentence exceeds `max_size` characters, no
partition can satisfy SPEC-CHUNK-201. The implementation raises an
error before optimization begins. The exact error type is
implementation-defined; the message should indicate that the input
contains a sentence longer than `max_size`.

The upstream stage is responsible for ensuring this never happens:
pass `max_len = max_size` to `split_sentences` when wiring the
stages end-to-end.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `DEFAULT_MAX_SIZE_CHARS` | `2048` | inputs table |
| `TARGET_STATEMENTS_PER_CHUNKLET` | `3` | SPEC-CHUNK-221 |
| `STATEMENT_COST_FLOOR` | `1e-6` | SPEC-CHUNK-221 |
| `STATEMENT_COST_SCALE` | `0.5` | SPEC-CHUNK-221 |
| `MIN_Q25_WORDS` | `1.0` | SPEC-CHUNK-230 |
| `STATEMENTS_AT_Q25` | `0.75` | SPEC-CHUNK-230 |
| `QUARTILE_GAP_STATEMENTS` | `0.50` | SPEC-CHUNK-230 |
| `BOUNDARY_STRENGTH_HEADING` | `1.00` | SPEC-CHUNK-240 |
| `BOUNDARY_STRENGTH_BLOCKQUOTE` | `0.75` | SPEC-CHUNK-240 |
| `BOUNDARY_STRENGTH_PARAGRAPH` | `0.50` | SPEC-CHUNK-240 |
| `BOUNDARY_STRENGTH_LIST` | `0.25` | SPEC-CHUNK-240 |

## Implementation-defined behavior

- Choice of DP implementation (forward vs. backward iteration; pure
  Python, NumPy, compiled).
- Whether to expose `boundary_cost` and `statement_cost` as
  user-overridable parameters or to hard-code the defaults.
- Whether to compute per-sentence boundary probabilities lazily or
  up-front.

## Unspecified behavior

- Behavior when `sentences` contains an empty string (`""`). Either
  filter empty strings out or treat them as zero-statement,
  zero-length contributions.
- Behavior when sentences contain trailing whitespace such that the
  Markdown parser sees a different structure than the caller expects.
  Stage 1's SPEC-CHUNK-109 should make this rare.

## Dependencies

- A Markdown parser exposing per-token start lines (any CommonMark
  parser).
- A DP or equivalent optimization implementation over O(N²) candidate
  partitions.
