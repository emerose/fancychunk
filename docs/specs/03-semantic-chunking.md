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
| `chunklet_embeddings` | matrix `[N, D]` of floats, or `None` | no | `None` | One row per chunklet, in the same order. Each row must have nonzero L2 norm. When omitted, the cosine-similarity term is dropped and the partition similarity is uniformly `1.0` for every candidate split (see SPEC-CHUNK-320 §No-embeddings path). |
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
  The returned rows are the *original* input rows; the
  unit-normalized and discourse-corrected forms used internally
  during partition similarity construction are not exposed.

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

Structurally this is an *interval-cover minimization* with linear
partition cost: a one-dimensional segmentation problem solvable by
`O(N²)` dynamic programming. Other solvers (binary integer
programming, LP relaxation plus rounding) work too, as long as they
return the optimum.

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

**No-embeddings path.** When `chunklet_embeddings` is omitted, steps
1-3 are skipped: ``sim[i] = 1.0`` for every partition point, the
discourse-vector step (SPEC-CHUNK-321) does not run, and the
heading-aware modification of step 4 still applies. The resulting
DP minimizes the number of selected partition points subject to the
covering constraint (SPEC-CHUNK-311), with heading-before splits
preferred (their `sim` is divided by `HEADING_SPLIT_BEFORE_DIVISOR`)
and heading-after splits forbidden (`HEADING_SPLIT_AFTER_FORBID`).
This is the "structural-only" mode — useful as a no-dependency
default and as the fallback when the caller hasn't yet computed
embeddings.

### SPEC-CHUNK-321 — Discourse-vector removal

The *discourse vector* represents the document's overall topic.
Subtracting it makes the remaining cosine similarity reflect *local*
topic shifts rather than the document's central theme. This step is
on by default; an implementation may expose a flag to disable it for
benchmarking, but the default must include the correction.

The technique of subtracting a single dominant direction from
sentence embeddings to surface local semantic content is inspired by
Arora, Liang & Ma,
[*A Simple but Tough-to-Beat Baseline for Sentence Embeddings*](https://openreview.net/forum?id=SyK00v5xx)
(ICLR 2017). Two methodological differences are worth noting: that
paper subtracts the top **principal component** across a *corpus*;
this spec subtracts the **mean** of typical-chunklet embeddings
per-*document*. The mean is cheaper to compute and approximates the
top PC well when the typical-chunklet embeddings cluster around a
single dominant direction (which they usually do within one
document).

Compute the correction as a single ordered procedure. The "skip
correction" outcome falls back to the unit-normalized embeddings from
step 1 of SPEC-CHUNK-320 in every skip case.

1. Determine the 15th and 85th percentiles of chunklet character
   length; call them `q15` and `q85`. The percentile boundaries are
   the named constants `TYPICAL_CHUNKLET_LOWER_QUANTILE = 0.15` and
   `TYPICAL_CHUNKLET_UPPER_QUANTILE = 0.85`. Use the same percentile
   method as SPEC-CHUNK-230 (linear interpolation between the two
   nearest ranks; specified once there and inherited here).

2. Identify *typical* chunklets: those whose length is in
   `[q15, q85]` — the middle 70% of the chunklet-length
   distribution.

3. **First skip check.** If fewer than two typical chunklets exist
   (a degenerate or very-short document), skip the correction.

4. Otherwise, set `discourse` to the L2-normalized mean of the
   typical chunklets' unit-normalized embeddings.

5. Tentatively compute the projected embeddings:
   ```
   X_corrected = X - (X · discourse) * discourse
   ```

6. **Second skip check.** If any row of `X_corrected` has L2 norm
   below the machine epsilon of the embedding's float dtype (i.e., a
   chunklet was effectively zeroed by the projection), skip the
   correction. The threshold here is the bare machine epsilon, not
   `sqrt(epsilon)`: this check detects rows that *became* zero from
   floating-point cancellation, which happens at epsilon scale.
   SPEC-CHUNK-320's similarity floor, by contrast, is a *cost-design*
   choice keeping the optimization well-defined and sits at a larger
   scale.

7. Otherwise, re-normalize `X_corrected` to unit norm and use it as
   the corrected embeddings.

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

A chunklet is a *heading* if its full Markdown block-level structure
consists of exactly one heading element — equivalently, parsing the
chunklet yields exactly one `heading_open` token and no other
block-opening tokens. Both ATX-style (`^#{1,6}(\s|$)`, matching
SPEC-CHUNK-512) and Setext-style (a heading text followed by a line
of `=` or `-` characters) qualify, recognized through whatever
CommonMark-conforming parser the implementation uses elsewhere. A
chunklet that *begins* with a heading line but also contains body
text afterwards is not a heading for the purposes of this section
(its parse contains additional block tokens beyond the heading).
A line beginning with seven or more `#` characters is not an ATX
heading (no parser will emit `heading_open` for it).

The choice to use the parser rather than a hand-rolled regex matters
for cross-stage consistency: stage 1 (SPEC-CHUNK-108) and stage 2
(SPEC-CHUNK-240) already determine heading-ness through the same
parser, so a heading flagged by an earlier stage will always also be
flagged here. SPEC-CHUNK-512 in stage 5 deliberately recognizes ATX
only — it operates on chunk text without re-parsing the full
chunk — and is the only place where the two heading-detection
strategies diverge.

Apply the following procedure. The `previous_is_heading` flag tracks
whether the immediately preceding chunklet was itself a heading, so
that two adjacent headings don't trigger a redundant boost at the
boundary between them. The initial value is irrelevant — the loop's
first iteration overwrites it before any guard reads it — but `False`
matches the natural "no chunklet has been seen yet" reading.

```
# i is a CHUNKLET index in [0, N).
# sim is indexed by PARTITION POINT — sim[k] sits between
# chunklets k and k+1 — so sim has length N-1 and valid indices
# are [0, N-2]. The bounds guards below enforce that.
previous_is_heading = False
for i in range(N):           # i = 0, 1, ..., N-1 inclusive
    if is_heading(chunklets[i]):
        # Encourage splitting before this heading
        # (only if there is a partition point at i-1, and the
        # previous chunklet was not itself a heading).
        if i >= 1 and not previous_is_heading:
            sim[i - 1] = max(
                sim[i - 1] / HEADING_SPLIT_BEFORE_DIVISOR,
                MIN_PARTITION_SIMILARITY,
            )

        # Discourage splitting immediately after this heading
        # (only if there is a partition point at i; the heading
        # belongs with the next chunk's intro, not as a standalone
        # chunk). The last chunklet has no partition point after it.
        if i <= N - 2:
            sim[i] = HEADING_SPLIT_AFTER_FORBID

        previous_is_heading = True
    else:
        previous_is_heading = False
```

The iteration covers every chunklet (indices `0` through `N-1`
inclusive) so that a heading at the *end* of the document still
triggers the "encourage split before" boost on `sim[N-2]`. The
bounds guards prevent indexing into nonexistent partition points at
either end. The `max(..., MIN_PARTITION_SIMILARITY)` re-application
on the boosted similarity preserves SPEC-CHUNK-320's strictly-positive
cost invariant even when the divisor would push the value below the
floor.

The two constants play different roles:

| Named constant | Value | Role |
|---|---|---|
| `HEADING_SPLIT_BEFORE_DIVISOR` | `4` | Attractiveness boost for splitting before a heading. |
| `HEADING_SPLIT_AFTER_FORBID` | `1.0` | The maximum possible post-rescaling similarity (see SPEC-CHUNK-320 step 3), so a split immediately after a heading carries maximum cost — effectively forbidden. |

`HEADING_SPLIT_AFTER_FORBID = 1.0` is structural: it equals the
ceiling of the partition-similarity range, so the minimization will
never *prefer* to split there. This is a strong cost penalty, not a
hard exclusion: if the covering constraint requires a split at this
exact partition point (no other partition point lies in the
infeasible window), the optimizer chooses it despite the cost. The
constant is not a tuning knob; any value `≥ 1.0` produces the same
effect.

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

### SPEC-CHUNK-340 — Short-circuit: trivial input

- If `chunklets == []`, return `([], [])`.
- If `len(chunklets) == 1`, return
  `([chunklets[0]], [chunklet_embeddings])` — one chunk equal to the
  sole input chunklet, paired with the full input embedding matrix.
- Otherwise, if `sum(len(c) for c in chunklets) <= max_size`, return
  `(["".join(chunklets)], [chunklet_embeddings])` — one chunk that
  concatenates all `N` chunklets, paired with the full input
  embedding matrix as a single element of the outer list. (The inner
  matrix has all `N` rows in their original order; concatenating
  across the single-element outer list yields the original
  `chunklet_embeddings` unchanged, satisfying SPEC-CHUNK-302.)

In all three cases no optimization is performed and the
heading-aware modification of SPEC-CHUNK-322 is skipped, even if
the input contains heading chunklets. This is intentional: the
covering constraint is trivially satisfied, so there is no
similarity-driven split to bias.

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
