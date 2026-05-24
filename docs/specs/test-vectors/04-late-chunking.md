# Test Vectors — Late Chunking

Concrete input/output pairs for the late-chunking embed strategy
(spec 04). These vectors test the function-level contract; embedder
behavior is mocked using a deterministic fake that satisfies the
contract in SPEC-CHUNK 04 §Embedder contract.

## Notation

- `sentences`: list of strings.
- `embedder`: a fake satisfying the embedder contract:
  - `n_ctx`: integer.
  - `tokenize(text)`: deterministic; returns a list of integer token
    IDs.
  - `detokenize(tokens)`: inverse of `tokenize` for sentinel-token
    lookup.
  - `embed(text)`: returns a `[T, D]` matrix where `T` is the number
    of tokens `tokenize` would return and each row is a deterministic
    function of the token (e.g., `row[i] = one-hot(token_id[i])` in
    dimension `D`).

Most vectors use `D = 8` for tractable output checking. Where a row's
exact value isn't important, the vector specifies the property (shape,
norm, ordering) rather than the literal vector.

## TV-401 — Shape conforms to input length (model-independent)

Validates SPEC-CHUNK-400, SPEC-CHUNK-401.

| Input | Value |
|-------|-------|
| `sentences` | `["First sentence.", "Second sentence.", "Third sentence."]` (3 items) |
| `embedder` | any conforming embedder with hidden size `D` |
| `max_tokens_per_segment` | `512` |

**Expected output (property):** a matrix of shape `[3, D]`. The
output row count equals the input sentence count; the output column
count equals the embedder's hidden size.

## TV-402 — Row order matches input order (model-independent)

Validates SPEC-CHUNK-400.

Using a fake embedder whose `embed(text)` returns one-hot rows keyed
by sentence index (so we can identify which output row came from
which input sentence):

| Input | Value |
|-------|-------|
| `sentences` | `["alpha", "bravo", "charlie", "delta"]` |
| `embedder` | a fake that produces a distinct, identifiable per-sentence direction in the output |

**Expected output (property):** row `i` is the (mean-pooled)
embedding of `sentences[i]`. Row 0 corresponds to "alpha"; row 3
corresponds to "delta". No reordering across segments.

## TV-403 — Single-sentence input handled (model-independent)

Validates SPEC-CHUNK-450.

| Input | Value |
|-------|-------|
| `sentences` | `["Only one sentence here."]` |
| `embedder` | any conforming embedder |
| `max_tokens_per_segment` | `512` |

**Expected output:** a `[1, D]` matrix. One segment with empty
preamble (`content_start == 0`). The single output row is the
mean-pooled embedding of the one sentence over all its token
embeddings.

## TV-404 — Sentence exceeds embedder context (model-independent)

Validates SPEC-CHUNK-451.

| Input | Value |
|-------|-------|
| `sentences` | `["A" + "a" * 10000]` (one sentence whose tokenization exceeds the embedder context) |
| `embedder` | a fake with `n_ctx = 512` |
| `max_tokens_per_segment` | `512` |

**Expected output:** the implementation raises an error before
encoding. The error message should indicate that the sentence does
not fit in the embedder's context.

## TV-405 — Segment boundaries with non-empty preamble (model-independent)

Validates SPEC-CHUNK-411.

| Input | Value |
|-------|-------|
| `sentences` | 20 sentences, each tokenizing to ~50 tokens |
| `embedder` | a fake with `n_ctx = 256` |
| `max_tokens_per_segment` | `256` |
| `preamble_fraction` | `0.382` |

Budget split: `max_tokens_preamble = round(0.382 * 256) = 98`,
`max_tokens_content = 256 - 98 = 158`.

**Expected behavior:**
- Segment 1: `content_start = 0`. Empty preamble (no sentences before
  index 0); unused preamble budget (98 tokens) is added to content
  budget → effective content budget is `256` tokens. Content walks
  forward from sentence 0, accumulating sentences until adding the
  next would exceed 256 tokens. With ~50 tokens each, this captures
  ~5 sentences (250 tokens). Suppose `segment_end = 5`.
- Segment 2: `content_start = 5`. Preamble walks backward from
  sentence 5, accumulating ~98 tokens (~2 sentences) → `segment_start
  = 3`. Content walks forward from sentence 5 with budget 158 tokens
  (~3 sentences) → `segment_end ≈ 8`. The first 5 output rows
  (sentences 0–4 from segment 1) are already produced; segment 2
  produces rows 5–7.
- Continue until `content_start >= 20`.

The conformance check: every sentence index `0..19` appears in
exactly one segment's content range. Sentence indices in *some*
segments' preamble ranges (3, 4 in the example above) also appeared
as content in earlier segments; their preamble-range embeddings are
discarded.

## TV-406 — First segment has empty preamble; unused budget rolls into content (model-independent)

Validates SPEC-CHUNK-411 step 2.

| Input | Value |
|-------|-------|
| `sentences` | 3 sentences totalling ~100 tokens |
| `embedder` | a fake with `n_ctx = 200` |
| `max_tokens_per_segment` | `200` |
| `preamble_fraction` | `0.382` |

Budget: preamble `= 76`, content `= 124`. Since `content_start = 0`,
preamble is empty and unused (76 tokens) → effective content budget
becomes `200`. All 3 sentences (~100 tokens) fit in one segment.

**Expected output:** a `[3, D]` matrix produced from a single
embedder call on the joined input.

## TV-407 — Sentinel character collision (model-independent; if sentinel method used)

Validates SPEC-CHUNK-421.

| Input | Value |
|-------|-------|
| `sentences` | `["First sentence.", "Has the symbol ⊕ inside.", "Third sentence."]` (the sentinel character appears in sentence 1) |
| `embedder` | any conforming embedder |
| `max_tokens_per_segment` | `512` |

**Expected behavior (if the implementation uses the sentinel-token
method):** the implementation detects the collision and either
chooses a different sentinel or falls back to a different counting
method (e.g., offset-based) and produces correct per-sentence token
counts.

**Expected behavior (if the implementation uses a different
counting method, e.g., offset-based):** the input is handled without
any special-case logic.

In both cases the final output is a `[3, D]` matrix with correctly
mean-pooled rows.

## TV-408 — Per-sentence token counts match embedder tokenization (model-independent)

Validates SPEC-CHUNK-420.

| Input | Value |
|-------|-------|
| `sentences` | `["AB", "CD"]` |
| `embedder` | a fake whose `tokenize("AB" + "CD")` returns `[a, b, c, d]` (4 tokens, no subword merging) AND whose `tokenize` of each sentence in isolation also returns 2 tokens each. The fake's `embed("ABCD")` returns 4 rows. |

**Expected output:** sentence 0's row is the mean of rows 0–1 of the
embedder output; sentence 1's row is the mean of rows 2–3. The sum
of per-sentence token counts (2 + 2 = 4) equals the embedder output
row count.

## TV-409 — Many short sentences exhaust largest-remainder allocation (model-independent)

Validates SPEC-CHUNK-452.

| Input | Value |
|-------|-------|
| `sentences` | 20 single-character sentences `["a", "b", ..., "t"]` |
| `embedder` | a fake where some single-character sentences tokenize to `0` tokens (e.g., the fake skips whitespace) |
| `max_tokens_per_segment` | `512` |

**Expected behavior:** the implementation either
(a) floors every sentence's allocated share at `1` token embedding
and proceeds with slight apportionment inaccuracy, or
(b) detects sentences with zero allocation and raises a clear
validation error.

Document which choice the implementation makes.

## TV-410 — Normalization control (model-independent)

Validates SPEC-CHUNK-402, SPEC-CHUNK-430.

| Input | Value |
|-------|-------|
| `sentences` | `["First.", "Second."]` |
| `embedder` | any conforming embedder |
| `normalize` (function parameter) | `True` in run A, `False` in run B |

**Expected output:**
- Run A: every output row has L2 norm `1.0` (within float precision).
- Run B: rows are raw mean-pooled embeddings; their L2 norms are
  whatever the embedder produced.
