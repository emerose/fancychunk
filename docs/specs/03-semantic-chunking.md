# Spec 03 — Semantic Chunking

Partition a sequence of chunklets (with their embeddings) into
*chunks*, where each chunk is a contiguous group of chunklets that
forms one semantic unit. Split points are chosen where adjacent
chunklets are *least* similar, subject to a hard upper bound on chunk
size and special handling of Markdown headings.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `chunklets` | list of strings | yes | — | An ordered sequence of chunklets, typically from stage 2. |
| `chunklet_embeddings` | matrix `[N, D]` of floats | yes | — | One row per chunklet, in the same order. Each row must have nonzero L2 norm. |
| `max_size` | positive integer | no | `2048` | Hard upper bound on chunk length in characters. <!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L15, confidence=confirmed, agent=human --> |

## Outputs

A tuple of two values:

1. `chunks` — list of strings. Each chunk is the concatenation of one
   or more contiguous input chunklets.
2. `chunk_embeddings` — list of matrices. The `i`-th matrix has one
   row per chunklet inside the `i`-th chunk, taken from
   `chunklet_embeddings` in the same row order.

Invariants:

- **SPEC-CHUNK-300** — `"".join(chunks) == "".join(chunklets)`. <!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L113-L117, confidence=confirmed, agent=human -->
- **SPEC-CHUNK-301** — Every chunk is at most `max_size` characters. <!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L43-L46, confidence=confirmed, agent=human -->
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
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L96-L112, confidence=confirmed, agent=human -->

### SPEC-CHUNK-311 — Covering constraint

For every contiguous window of chunklets `[a, b)` whose total
character length exceeds `max_size`, at least one partition point
inside `[a, b-1)` must be selected. Equivalently: no chunk in the
output may exceed `max_size` characters.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L86-L95, confidence=confirmed, agent=human -->

A practical encoding: for each chunklet `i`, find the smallest `j > i`
such that `chunklets[i..j]` exceeds `max_size`. The constraint is that
at least one partition point in `[i, j-1)` is selected. Generating one
constraint per chunklet (or per *covering window*) is sufficient.

The reimplementor is free to choose any constraint encoding that
yields the same feasible region.

### SPEC-CHUNK-320 — Partition similarity construction

The partition similarity `sim[i]` for candidate partition point `i`
(between chunklets `i` and `i+1`) is constructed from the chunklet
embeddings in four steps:

**Step 1 — Unit-normalize all embeddings.**
Each embedding row is divided by its L2 norm.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L54-L56, confidence=confirmed, agent=human -->

**Step 2 — Remove the discourse vector (SPEC-CHUNK-321).** A
document-wide "topic" direction is computed and projected out of
every embedding.

**Step 3 — Compute base partition similarity.**
For each partition point `i`, compute the dot product of the
discourse-corrected, re-normalized embeddings of chunklets `i` and
`i+1`. Then rescale and clamp:

```
sim[i] = max( (dot_product + 1) / 2,  sqrt(epsilon) )
```

where `epsilon` is the machine epsilon of the embedding's float dtype.
This maps cosine similarity from `[-1, 1]` to `[0, 1]` (with a small
positive floor).
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L66-L72, confidence=confirmed, agent=human -->

**Step 4 — Heading-aware modification (SPEC-CHUNK-322).** Adjust
`sim[i]` for partition points adjacent to heading chunklets.

### SPEC-CHUNK-321 — Discourse-vector removal

The *discourse vector* represents the document's overall topic; the
intent is that subtracting it makes the remaining cosine similarity
reflect *local* topic shifts rather than the document's central theme.

Compute it as follows:

1. Determine the 15th and 85th percentiles of chunklet character
   length, call them `q15` and `q85`.
2. Identify *non-outlying* chunklets: those whose length is in
   `[q15, q85]`.
3. If any non-outlying chunklets exist, set `discourse` to the L2-normalized
   mean of those chunklets' unit-normalized embeddings.
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

<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L57-L65, confidence=confirmed, agent=human -->

When fewer than two non-outlying chunklets exist (a degenerate or
very-short document), no correction is applied.

### SPEC-CHUNK-322 — Heading-aware modification of partition similarity

After computing the base partition similarities, modify them based on
which chunklets are Markdown headings.

A chunklet is a *heading* if, after stripping newlines and surrounding
whitespace, it begins with `^#+\s` (one or more `#` characters
followed by whitespace — the Markdown heading syntax).
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L76, confidence=confirmed, agent=human -->

Walk the chunklets in order (treating the position before the first
chunklet as a virtual "previous was a heading"). For each chunklet `i`
from `0` to `N-2`:

- If chunklet `i` **is a heading**:
  - If chunklet `i-1` is **not** a heading and `i > 0`:
    `sim[i-1] = sim[i-1] / 4` — encourage splitting *before* the
    heading.
  - `sim[i] = 1.0` — discourage splitting *immediately after* a
    heading (the heading is part of the next chunk's intro, not a
    standalone chunk).

- If chunklet `i` is **not** a heading: no modification.

Update "previous was a heading" for the next iteration based on
chunklet `i`.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L74-L85, confidence=confirmed, agent=human -->

The factor `1/4` (the boost to "split before a heading") and the
value `1.0` (the penalty to "split immediately after a heading") are
preserved as part of the spec.

## Determinism and tie-breaking

### SPEC-CHUNK-330 — Deterministic given the solver

The output is deterministic for the same inputs, the same Markdown
parser, and a deterministic optimizer. Different integer-programming
solvers may produce different but equally-optimal partitions when
multiple partitions tie. Test vectors should not depend on
tie-breaking choices.

## Edge cases

### SPEC-CHUNK-340 — Short-circuit: zero or one chunklet

If `len(chunklets) <= 1` or `sum(len(c) for c in chunklets) <=
max_size`, return the input unchanged: one chunk that is the
concatenation of all chunklets, and one embedding matrix containing
all the input rows.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L50-L52, confidence=confirmed, agent=human -->

### SPEC-CHUNK-341 — Chunklet exceeds `max_size`

If any single input chunklet exceeds `max_size` characters, the
covering constraint is infeasible. The implementation raises a
validation error before optimization begins.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L43-L46, confidence=confirmed, agent=human -->

### SPEC-CHUNK-342 — Zero-norm embedding

If any chunklet embedding has L2 norm `0`, the implementation raises a
validation error before optimization begins.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L47-L49, confidence=confirmed, agent=human -->

### SPEC-CHUNK-343 — Optimization failure

If the underlying solver reports a failure (infeasible, numerical
issue, time limit), the implementation raises an error. The exact
error type is implementation-defined; the message should indicate
that partition optimization failed.
<!-- cite: source=source-code, ref=raglite@6a540e1:src/raglite/_split_chunks.py:L109-L111, confidence=confirmed, agent=human -->

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
  chunklets list. The implementor should validate and raise.
- Behavior when `chunklets == []` (treat as the SPEC-CHUNK-340
  short-circuit, returning `([], [<empty matrix>])`).

## Uncertainties

### U-CHUNK-301 — Discourse vector necessity

The discourse-vector correction is mathematically meaningful (it
removes a shared topic direction), but the *empirical benefit* is not
documented in the source. The reimplementor should preserve it; an
optional flag to disable it for benchmarking is acceptable, but the
default behavior must include the correction.

### U-CHUNK-302 — Percentile choice (`q15`, `q85`)

The choice of 15th and 85th percentiles in SPEC-CHUNK-321 is not
explained. These are preserved as defaults. Tweaking them is unlikely
to dramatically change behavior, but exact reproduction requires the
documented values.

### U-CHUNK-303 — Heading-penalty constants

The `1/4` and `1.0` constants in SPEC-CHUNK-322 are preserved as
defaults. They encode a strong preference; the reimplementor should
not change them silently.
