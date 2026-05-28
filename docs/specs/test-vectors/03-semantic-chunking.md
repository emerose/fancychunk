# Test Vectors — Semantic Chunking

Concrete input/output pairs for stage 3. Inputs are chunklets paired
with synthetic embeddings designed to make the expected partition
determinable by inspection.

## Notation

- `chunklets`: list of strings.
- `embeddings`: matrix `[N, D]` of floats. Synthetic embeddings are
  used so the expected output does not depend on a real model.
- `max_size`: integer.

## TV-301 — Short-circuit on single chunklet (model-independent)

Validates SPEC-CHUNK-340.

| Input | Value |
|-------|-------|
| `chunklets` | `["Single chunklet content."]` |
| `embeddings` | `[[1.0, 0.0]]` |
| `max_size` | `2048` |

**Expected chunks:** `["Single chunklet content."]`.
**Expected chunk_embeddings:** `[[[1.0, 0.0]]]` (one matrix
containing the single input row).

## TV-302 — Short-circuit when total fits in `max_size` (model-independent)

Validates SPEC-CHUNK-340.

| Input | Value |
|-------|-------|
| `chunklets` | `["one ", "two ", "three"]` (total 13 chars) |
| `embeddings` | three orthogonal unit vectors of any dimension |
| `max_size` | `2048` |

**Expected chunks:** `["one two three"]` (single chunk, no
optimization performed).
**Expected chunk_embeddings:** one matrix with all three rows.

This holds even when embeddings are maximally dissimilar — the size
short-circuit overrides similarity-based splitting.

## TV-303 — Round-trip property

For any input `(C, E)` and output `(K, KE)`:

```
"".join(K) == "".join(C)
sum(len(matrix) for matrix in KE) == len(E)
concatenate(KE) == E  (row-wise)
```

Must hold for every test case below; the properties are
SPEC-CHUNK-300 and SPEC-CHUNK-302.

## TV-304 — Hard size constraint forces split (model-independent)

Validates SPEC-CHUNK-301 and SPEC-CHUNK-311.

| Input | Value |
|-------|-------|
| `chunklets` | `["a" * 1000, "b" * 1000, "c" * 1000]` |
| `embeddings` | three identical unit vectors `[[1, 0], [1, 0], [1, 0]]` |
| `max_size` | `2048` |

**Expected output (property):** every chunk has length `≤ 2048`. With
identical embeddings the similarity-based optimization has no
preference; the covering constraint forces at least one split.
Conforming partitions:
- `[chk[0], chk[1] + chk[2]]`
- `[chk[0] + chk[1], chk[2]]`

Not conforming: `[chk[0] + chk[1] + chk[2]]` (3000 chars).

## TV-305 — Identical embeddings: no split beyond size requirement (model-independent)

Validates the partition-similarity ranking.

| Input | Value |
|-------|-------|
| `chunklets` | 10 chunklets of 100 chars each (1000 chars total) |
| `embeddings` | 10 identical unit vectors |
| `max_size` | `2048` |

**Expected output:** single chunk of 1000 chars. The total fits, the
SPEC-CHUNK-340 short-circuit applies, and no optimization is
performed.

## TV-307 — Heading-aware modification: no split immediately after heading (model-independent)

Validates SPEC-CHUNK-322.

| Input | Value |
|-------|-------|
| `chunklets` | `["# Heading\n\n", "Body para one.\n\n", "Body para two.\n\n", "Body para three.\n\n"]` |
| `embeddings` | the heading row equal to `[1, 0]`; body rows equal to `[0, 1], [0, 1], [0.7, 0.7]` |
| `max_size` | `2048` |

Heading detection per SPEC-CHUNK-322: chunklet 0 matches `^#+\s`;
chunklets 1, 2, 3 do not.

Per SPEC-CHUNK-322 modification (iteration runs over all four
chunklets, `i = 0, 1, 2, 3`):
- `i = 0` (chunklet 0 is heading): `i = 0` and previous-is-heading is
  True (the virtual prior), so the "encourage split before" branch
  is skipped. The "discourage split after" branch sets
  `sim[0] = HEADING_SPLIT_AFTER_FORBID`.
- `i = 1, 2, 3` (non-heading): no modification.

So `sim = [1.0, sim_base[1], sim_base[2]]`. Splitting between
chunklet 0 and chunklet 1 is heavily penalized (the heading should
stay with its body).

**Expected output (property):** the partition does NOT begin with a
single-chunklet heading `["# Heading\n\n"]` followed by another
chunk. The heading is grouped with at least the first body
chunklet.

## TV-308 — Heading-aware modification: encourage split *before* heading (model-independent)

Validates SPEC-CHUNK-322 (divide-by-4 case).

| Input | Value |
|-------|-------|
| `chunklets` | `["a"*900, "b"*900, "## Subhead\n\n", "c"*900]` (total ~2700 chars) |
| `embeddings` | all four with near-identical unit vectors (e.g., all `[1, 0]` with tiny perturbations to satisfy nonzero norm) |
| `max_size` | `2048` |

Heading detection: chunklet 2 is a heading; 0, 1, 3 are not.

Per SPEC-CHUNK-322 modification (walking the iteration):
- `i = 0` (non-heading): no modification.
- `i = 1` (non-heading): no modification.
- `i = 2` (heading):
  - Previous (chunklet 1) is non-heading:
    `sim[1] = sim_base[1] / HEADING_SPLIT_BEFORE_DIVISOR`
    (encourage split before the heading, between chunklets 1 and 2).
  - `sim[2] = HEADING_SPLIT_AFTER_FORBID` (no split immediately after
    heading).

With near-identical embeddings, all `sim_base[i] ≈ 1.0`. After
modification: `sim ≈ [1.0, 0.25, 1.0]`.

The covering constraint requires at least one split (total exceeds
`max_size`). With `sim ≈ [1.0, 0.25, 1.0]`, the minimum-cost single
split is between chunklets 1 and 2 (`sim = 0.25`).

**Expected output (property):** partition starts a new chunk at
chunklet 2. Conforming: `[chk[0] + chk[1], chk[2] + chk[3]]`.

## TV-309 — Zero-norm embedding rejected (model-independent)

Validates SPEC-CHUNK-342.

| Input | Value |
|-------|-------|
| `chunklets` | `["a", "b"]` (any content) |
| `embeddings` | `[[0.0, 0.0], [1.0, 0.0]]` |
| `max_size` | `2048` |

**Expected output:** implementation raises a validation error before
optimization begins.

## TV-310 — Oversized chunklet rejected (model-independent)

Validates SPEC-CHUNK-341.

| Input | Value |
|-------|-------|
| `chunklets` | `["a" * 3000, "b"]` |
| `embeddings` | two orthogonal unit vectors |
| `max_size` | `2048` |

**Expected output:** implementation raises a validation error before
optimization begins. (The chunklet of 3000 chars cannot fit in any
chunk of `max_size = 2048`.)

## TV-311 — Discourse vector skipped when degenerate (model-independent)

Validates the safeguard in SPEC-CHUNK-321.

Construct chunklets whose embeddings are all parallel to the same
vector (so projection out of that vector would zero them).

| Input | Value |
|-------|-------|
| `chunklets` | 5 chunklets of 1000 chars each |
| `embeddings` | 5 copies of `[1.0, 0.0]` |
| `max_size` | `2048` |

The discourse vector is `[1.0, 0.0]` (mean of identical rows). Step 4
of SPEC-CHUNK-321 would zero every row. Per the safeguard, the
implementation falls back to un-corrected unit-normalized embeddings.
Cosine similarity is then `1.0` between every adjacent pair;
`sim[i] = (1 + 1) / 2 = 1.0`.

The covering constraint requires splits; with all `sim = 1.0`, any
satisfying partition is optimal.

**Expected output (property):** every chunk `≤ 2048` chars;
round-trip holds. The implementation must NOT raise (which it would
if it tried to normalize zero vectors).

## TV-323a — Leading front-matter bundled forward (model-independent)

Validates SPEC-CHUNK-323. Deterministic, model-free analogue of the
qasper `1903.09588` repro.

| Input | Value |
|-------|-------|
| `chunklets` | `["# Title\n\n## Abstract\n\n" + "a"*579, "## Introduction\n\n" + "b"*583, "c"*600, "d"*600]` |
| `embeddings` | `[[1,0], [0,1], [0,1], [0,1]]` |
| `max_size` | `2048` |

The front matter (chunklet 0) is dissimilar from the body, which is
internally uniform, so the *bare* similarity DP (badness disabled)
isolates it as chunk 0 (`cuts == [0, 1, …]`). With the SPEC-CHUNK-323
badness term, the optimizer extends the front matter forward into the
first body section instead.

**Expected output (property):** the first chunk is **not**
front-matter-only — it contains `"## Introduction"` (the front matter
bundled forward). Round-trip holds; every chunk `≤ 2048`.

## TV-323b — No heading-only chunk emitted (model-independent)

Validates that a heading is not split off into a chunk of its own. This
is a consequence of SPEC-CHUNK-322 (the split-after-heading cost is the
maximum, so the DP never voluntarily isolates a heading) rather than a
dedicated SPEC-CHUNK-323 term.

| Input | Value |
|-------|-------|
| `chunklets` | `["x"*1000, "## Lonely Heading\n\n", "y"*1000, "z"*1000]` |
| `embeddings` | `[[1,0], [0,1], [1,0], [1,0]]` |
| `max_size` | `2048` |

**Expected output (property):** no emitted chunk consists solely of
heading lines (every chunk has body content). Round-trip holds. The
same holds for a heading at the very end of the document — it rides
along with the preceding body rather than forming its own chunk.

## TV-323c — Tiny chunk merges; short distinct section kept (model-independent)

Validates the general small-chunk term of SPEC-CHUNK-323 and that the
size cutoff scales with distinctness (it is not a fixed number). Three
chunklets where isolating the lead chunk (`[c0][c1+c2]`) and merging it
(`[c0+c1][c2]`) are *both* feasible single cuts, so the small-chunk
badness — not cut minimization — decides. `c0` is a distinct topic;
`c1` and `c2` are the same topic.

| Input | Value |
|-------|-------|
| `chunklets` | `[lead, "b"*700, "b"*700]` |
| `embeddings` | `[[1,0], [0,1], [0,1]]` (lead distinct; body uniform) |
| `max_size` | `1410` (general badness ceiling ≈ `0.2 × 1410 ≈ 282`) |

**Expected output (property):**

- `lead = "t"*20` (below the ceiling): the first chunk is **not** the
  bare 20-char fragment — it is merged forward, even though it is a
  distinct topic.
- `lead = "s"*400` (above the ceiling): the first chunk **is** the
  400-char section — a distinct short section is kept, the semantic
  split honoured.

Round-trip holds and every chunk is `≤ max_size` in both cases.
