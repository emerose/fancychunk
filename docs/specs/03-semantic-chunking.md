# Spec 03 — Semantic Chunking

Partition a sequence of chunklets (with their embeddings) into
*chunks*, where each chunk is a contiguous group of chunklets that
forms one semantic unit. Split points are chosen where adjacent
chunklets are *least* similar, subject to a hard upper bound on chunk
size and special handling of Markdown headings.

This stage corresponds to "level 4" in Greg Kamradt's
[5 Levels of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s)
taxonomy: split where adjacent units are *least* semantically similar,
rather than at fixed character/token counts (levels 1-3). The
integer-programming framing below is one specific implementation of
that idea.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `chunklets` | list of strings | yes | — | An ordered sequence of chunklets, typically from stage 2. |
| `chunklet_embeddings` | matrix `[N, D]` of floats | yes | — | One row per chunklet, in the same order. Each row must have nonzero L2 norm. |
| `max_size` | positive integer | no | `DEFAULT_MAX_SIZE_CHARS` (`= 2048`) | Hard upper bound on chunk length in characters. (Same default as stage 2; see Spec 02 for the rationale.) |

## Outputs

A tuple of two values:

1. `chunks` — list of strings. Each chunk is the concatenation of one
   or more contiguous input chunklets.
2. `chunk_embeddings` — list of matrices. The `i`-th matrix has one
   row per chunklet inside the `i`-th chunk, taken from
   `chunklet_embeddings` in the same row order.

Invariants:

- **SPEC-CHUNK-300** — `"".join(chunks) == "".join(chunklets)`.
- **SPEC-CHUNK-301** — Every chunk is at most `max_size` characters.
- **SPEC-CHUNK-302** — The rows of `chunklet_embeddings`, concatenated
  across `chunk_embeddings` in order, equal `chunklet_embeddings`.

## Behavior

### SPEC-CHUNK-310 — Optimization framing

Semantic chunking selects a subset of *partition points*. A partition
point sits between adjacent chunklets `i` and `i+1`; there are `N-1`
candidate partition points for `N` chunklets.

Given a partition similarity `sim[i]` for each candidate partition
point (defined in SPEC-CHUNK-320), the optimization finds the subset
of partition points `P ⊆ {0, 1, ..., N-2}` that **minimizes**
`Σ_{i ∈ P} sim[i]` (lower similarity = better split), subject to the
covering constraint in SPEC-CHUNK-311.

This is a *minimum-cost set cover* over candidate partition points
with linear cost. It is solvable as a binary integer program; it can
also be solved by other means (e.g., column generation, LP relaxation
plus rounding) provided the result is optimal.

### SPEC-CHUNK-311 — Covering constraint

For every contiguous window of chunklets `[a, b)` whose total
character length exceeds `max_size`, at least one partition point
inside `[a, b-1)` must be selected. Equivalently: no chunk in the
output may exceed `max_size` characters.

A practical encoding: for each chunklet `i`, find the smallest `j > i`
such that `chunklets[i..j]` exceeds `max_size`. The constraint is that
at least one partition point in `[i, j-1)` is selected. Generating one
constraint per chunklet (or per *covering window*) is sufficient. Any
constraint encoding that yields the same feasible region is
conforming.

### SPEC-CHUNK-320 — Partition similarity construction

The partition similarity `sim[i]` for candidate partition point `i`
(between chunklets `i` and `i+1`) is constructed from the chunklet
embeddings in four steps:

**Step 1 — Unit-normalize all embeddings.**
Each embedding row is divided by its L2 norm.

**Step 2 — Remove the discourse vector (SPEC-CHUNK-321).** A
document-wide "topic" direction is computed and projected out of
every embedding.

**Step 3 — Compute base partition similarity.**
For each partition point `i`, compute the dot product of the
discourse-corrected, re-normalized embeddings of chunklets `i` and
`i+1`. Then rescale and clamp:

```
sim[i] = max( (dot_product + 1) / 2,  MIN_PARTITION_SIMILARITY )
```

where `MIN_PARTITION_SIMILARITY = sqrt(epsilon)` and `epsilon` is the
machine epsilon of the embedding's float dtype. The rescaling maps
cosine similarity from `[-1, 1]` to `[0, 1]`; the clamp ensures every
partition point has strictly positive cost.

The positive floor matters: without it, an antipodal pair of
embeddings (`dot_product = -1`) would yield `sim[i] = 0`, and the
optimizer would be indifferent to adding extra splits at zero-cost
points — producing nondeterministic over-splitting where two
partitions of different sizes have the same total cost. Floor at
`sqrt(epsilon)` (rather than `epsilon` directly) keeps the floor
safely above floating-point noise without distorting any real
similarity value.

**Step 4 — Heading-aware modification (SPEC-CHUNK-322).** Adjust
`sim[i]` for partition points adjacent to heading chunklets.

### SPEC-CHUNK-321 — Discourse-vector removal

The *discourse vector* represents the document's overall topic.
Subtracting it makes the remaining cosine similarity reflect *local*
topic shifts rather than the document's central theme. This step is
on by default; an implementation may expose a flag to disable it for
benchmarking, but the default must include the correction.

The technique of subtracting a single dominant direction from
sentence embeddings to surface local semantic content is structurally
the same as the "common discourse vector" step in Arora, Liang & Ma,
[*A Simple but Tough-to-Beat Baseline for Sentence Embeddings*](https://openreview.net/forum?id=SyK00v5xx)
(ICLR 2017). That paper subtracts the top principal component across
a *corpus* to remove a shared frequency direction; this spec applies
the same idea per-*document* to remove the document's central topic.

Compute it as follows:

1. Determine the `TYPICAL_CHUNKLET_LOWER_QUANTILE`-th
   (`= 0.15`) and `TYPICAL_CHUNKLET_UPPER_QUANTILE`-th
   (`= 0.85`) percentiles of chunklet character length; call them
   `q15` and `q85`.
2. Identify *typical* chunklets: those whose length is in
   `[q15, q85]` — i.e., the middle 70% of the chunklet-length
   distribution.
3. If any typical chunklets exist, set `discourse` to the
   L2-normalized mean of those chunklets' unit-normalized embeddings.
4. Project each chunklet's embedding onto the hyperplane orthogonal
   to `discourse`:
   ```
   X_corrected = X - (X · discourse) * discourse
   ```
5. Re-normalize `X_corrected` to unit norm.
6. **Safeguard:** if step 4 would zero out any chunklet (i.e., a row
   becomes shorter than the machine epsilon after projection), abandon
   the correction and use the un-corrected, unit-normalized embeddings
   instead.

When fewer than two typical chunklets exist (a degenerate or
very-short document), no correction is applied.

The middle-70% trim exists because unusually-short chunklets (titles,
one-line code blocks, list-item fragments) and unusually-long
chunklets (large preformatted blocks, table dumps) are systematically
*off-topic* relative to the document's typical prose. Including them
in the topic estimate pulls the discourse vector toward those outliers
and weakens the correction for the bulk of the document. The exact
`0.15`/`0.85` choice is hand-tuned; reasonable values in
`[0.10, 0.25]` and `[0.75, 0.90]` produce qualitatively similar
discourse vectors.

### SPEC-CHUNK-322 — Heading-aware modification of partition similarity

After computing the base partition similarities, modify them based on
which chunklets are Markdown headings.

A chunklet is a *heading* if, after stripping newlines and surrounding
whitespace, it begins with `^#+\s` (one or more `#` characters
followed by whitespace — the Markdown heading syntax).

Walk the chunklets in order (treating the position before the first
chunklet as a virtual "previous was a heading"). For each chunklet `i`
from `0` to `N-2`:

- If chunklet `i` **is a heading**:
  - If chunklet `i-1` is **not** a heading and `i > 0`:
    `sim[i-1] = sim[i-1] / HEADING_SPLIT_BEFORE_DIVISOR` — encourage
    splitting *before* the heading.
  - `sim[i] = HEADING_SPLIT_AFTER_FORBID` — discourage splitting
    *immediately after* a heading (the heading is part of the next
    chunk's intro, not a standalone chunk).

- If chunklet `i` is **not** a heading: no modification.

Update "previous was a heading" for the next iteration based on
chunklet `i`.

The two constants play different roles:

| Named constant | Value | Role |
|---|---|---|
| `HEADING_SPLIT_BEFORE_DIVISOR` | `4` | Attractiveness boost for splitting before a heading. |
| `HEADING_SPLIT_AFTER_FORBID` | `1.0` | The maximum possible post-rescaling similarity (see SPEC-CHUNK-320 step 3), so a split immediately after a heading carries maximum cost — effectively forbidden. |

`HEADING_SPLIT_AFTER_FORBID = 1.0` is structural: it equals the
ceiling of the partition-similarity range, so the minimization will
never choose to split there unless absolutely required by the
covering constraint. This is not a tuning knob; any value `≥ 1.0`
produces the same effect.

`HEADING_SPLIT_BEFORE_DIVISOR = 4` is a bounded heuristic. The goal
is to make "split before a heading" at least as attractive as the
strongest non-heading boundary the document is likely to produce.
Cosine similarity between unrelated paragraphs typically maps
(post-rescaling) to `sim` values in the `0.5`-`0.7` range; dividing
by `2` already brings a heading boundary below that, and any divisor
`≥ 2` produces qualitatively similar partitions. The default value
`4` provides extra margin so the heuristic survives noisier
embeddings; values much greater than `~8` start to treat headings as
forced splits, which conflicts with cases where a heading and its
first paragraph genuinely belong together. Implementations may expose
this as a parameter; default to `4`, valid range roughly `[2, 8]`.

## Determinism and tie-breaking

### SPEC-CHUNK-330 — Deterministic given the solver

The output is deterministic for the same inputs, the same Markdown
parser, and a deterministic optimizer. Different integer-programming
solvers may produce different but equally-optimal partitions when
multiple partitions tie. Test vectors must not depend on tie-breaking
choices.

## Edge cases

### SPEC-CHUNK-340 — Short-circuit: zero or one chunklet

If `len(chunklets) <= 1` or `sum(len(c) for c in chunklets) <=
max_size`, return the input unchanged: one chunk that is the
concatenation of all chunklets, and one embedding matrix containing
all the input rows.

### SPEC-CHUNK-341 — Chunklet exceeds `max_size`

If any single input chunklet exceeds `max_size` characters, the
covering constraint is infeasible. The implementation raises a
validation error before optimization begins.

### SPEC-CHUNK-342 — Zero-norm embedding

If any chunklet embedding has L2 norm `0`, the implementation raises a
validation error before optimization begins.

### SPEC-CHUNK-343 — Optimization failure

If the underlying solver reports a failure (infeasible, numerical
issue, time limit), the implementation raises an error. The exact
error type is implementation-defined; the message should indicate
that partition optimization failed.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `DEFAULT_MAX_SIZE_CHARS` | `2048` | inputs table (same as Spec 02) |
| `MIN_PARTITION_SIMILARITY` | `sqrt(epsilon)` | SPEC-CHUNK-320 step 3 |
| `TYPICAL_CHUNKLET_LOWER_QUANTILE` | `0.15` | SPEC-CHUNK-321 |
| `TYPICAL_CHUNKLET_UPPER_QUANTILE` | `0.85` | SPEC-CHUNK-321 |
| `HEADING_SPLIT_BEFORE_DIVISOR` | `4` | SPEC-CHUNK-322 |
| `HEADING_SPLIT_AFTER_FORBID` | `1.0` | SPEC-CHUNK-322 |

## Implementation-defined behavior

- Choice of solver (binary integer programming, ILP, LP-with-rounding
  proven optimal, dynamic-programming reformulation if equivalent).
- Encoding of the covering constraint (one constraint per chunklet,
  one constraint per maximal infeasible window, etc.) provided the
  feasible region is the same.
- Precision (`float32` vs `float64`) of intermediate computations.
- Whether to materialize `X_corrected` separately or compute the
  modified dot products directly.

## Unspecified behavior

- Behavior when the embedding matrix has fewer or more rows than the
  chunklets list. The implementation should validate and raise.
- Behavior when `chunklets == []` (treat as the SPEC-CHUNK-340
  short-circuit, returning `([], [<empty matrix>])`).
