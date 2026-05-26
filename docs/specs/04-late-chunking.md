# Spec 04 — Late Chunking

An embedding strategy that produces *one context-aware embedding per
chunk* by encoding wide windows of the document together and
mean-pooling the per-token outputs by chunk boundary. The
chunk-level vector preserves anaphoric references that an
encoded-in-isolation chunk would lose ("the algorithm" picks up its
real referent because the embedder's attention reaches the preceding
heading or text).

This spec describes a contract on the *embedding function*. The
caller's "embed each chunk in isolation" step is replaced by one
call to `embed_with_late_chunking(chunks, embedder, …)` that returns
a `[len(chunks), D]` matrix — the storage vector for each chunk,
ready to drop into a vector store.

The technique was introduced and named in Weaviate's blog post
[*Late Chunking in Long-Context Embedding Models*](https://weaviate.io/blog/late-chunking),
and quantified in the Jina AI paper
[*Late Chunking: Contextual Chunk Embeddings Using Long-Context
Embedding Models*](https://arxiv.org/abs/2409.04701) (Günther et al.,
2024).

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `chunks` | list of strings | yes | — | The chunks to embed, in document order. Typically the output of `split_chunks` *before* any `enrich_with_headings()` post-processing (that helper is for stored-text breadcrumbs, not embedding-time context). |
| `embedder` | object satisfying the embedder contract below | yes | — | A token-level embedding model. |
| `max_tokens_per_segment` | positive integer | no | `embedder.n_ctx` | The upper bound on tokens fed to the embedder in one call. Defaults to the embedder's reported context-window size; must not exceed it. |
| `preamble_fraction` | float in `[0, 1)` | no | `DEFAULT_PREAMBLE_FRACTION` (`= 0.382`) | Fraction of `max_tokens_per_segment` reserved for the segment's preamble (heading prepend + backward-walk context). `0.0` is permitted and degenerates to standard chunking (no contextualization); useful for benchmarking against late chunking. Values `≥ 1.0` are rejected. |
| `normalize` | boolean | no | `True` | Whether to L2-normalize each output row. |
| `include_headings` | boolean | no | `True` | Whether to prepend the in-scope Markdown heading stack to each segment as part of the preamble (SPEC-CHUNK-470). Pass `False` for non-markdown inputs, for ablation, or when the caller is managing heading context themselves. |

## Outputs

A matrix of shape `[len(chunks), embedding_dim]`. Row `i` is the
context-aware embedding of `chunks[i]`.

- **SPEC-CHUNK-400** — One row per input chunk, in the same order.
- **SPEC-CHUNK-401** — Each row is a fixed-dimensional vector of
  floats. The dimension `D` equals the per-token output dimension of
  the embedder's `embed` operation (the embedder's hidden size).
- **SPEC-CHUNK-402** — Rows are L2-normalized when the function's
  `normalize` parameter is `True` (the default); otherwise rows
  reflect the raw mean-pooled token embeddings. The embedder itself
  does not control normalization.

## Embedder contract

The embedder is supplied by the caller. The library owns the
late-chunking algorithm — segment planning, optional heading prepend,
mean-pool per chunk, preamble discard, normalization — and nothing
else. The caller owns tokenization, special-token policy, and the
choice of method for mapping joined-input tokens back to source
texts.

The contract is two methods plus one attribute:

| Operation | Kind | Behavior |
|-----------|------|----------|
| `n_ctx` | integer attribute | Maximum number of tokens the embedder accepts in one segment. |
| `count_tokens(texts: list[str])` | method, returns `list[int]` | Per-text token count for budget planning. May be approximate (subword merges across boundaries can shift counts by ±1); SPEC-CHUNK-412's largest-remainder safety net absorbs the drift. The library calls this both for chunks themselves and for heading-stack prepends. Used only for segment construction. |
| `embed_segment(texts: list[str])` | method, returns `(matrix[T, D], list[int])` | Embed the segment's texts as one contextualized sequence. Returns the per-token embedding matrix and the per-text token allocation. The allocation must conserve the matrix row count: `sum(per_text_counts) == T`. Any special tokens (`[CLS]`, `[SEP]`, BOS, EOS) the embedder injects are the implementation's concern. |

The protocol's `texts` parameter is intentionally generic: the
library passes whatever sequence of contiguous text units the
segment holds — chunks plus (optionally) one heading-stack prepend.
The embedder treats them all uniformly.

Any embedding model that exposes token-level outputs (a "no pooling"
or "per-token output" mode) is acceptable. Cloud embedding APIs that
return only one vector per input do not satisfy the contract — late
chunking needs the per-token outputs to pool within chunk boundaries
that the embedder doesn't know about.

**Why this shape.** The caller already knows which tokenizer is in
use, whether special tokens are injected, and what alignment method
suits their stack (sentinel-token, offset-based, or anything else).
Pushing the alignment work to the caller keeps the library
unentangled with tokenizer-specific edge cases (e.g., BERT-family
models subword-merging stable-looking sentinel characters) and lets
each adapter pick the simplest valid method for its embedder. See
`examples/embedders/` for reference adapters.

## Behavior

### SPEC-CHUNK-410 — Segments cover the document exactly once

The document is processed as a sequence of *segments*. Each segment
has two parts:

- **Preamble:** an optional heading-stack prepend plus the chunks
  immediately before the segment's content range, providing
  contextualization. Their token-level embeddings are discarded
  after pooling.
- **Content:** chunks whose final per-chunk embeddings are kept.

Every chunk appears in exactly one segment's *content* range
(though it may also appear as another segment's *preamble*).

**Why preambles exist.** Transformer embeddings are *contextual* —
every token's output vector is a function of the surrounding tokens
via attention. Consider:

> "The algorithm achieves O(n log n) by maintaining a balanced
> binary tree."

Encoded in isolation, "the algorithm" has no antecedent and the
embedder produces a generic "thing being discussed" direction for
it. Encoded with the preceding heading "## Quicksort with random
pivot selection" and the preceding paragraph as preamble, the
attention mechanism connects "the algorithm" to "Quicksort" and the
embedding for that token picks up a meaningful direction. The
preamble exists so that chunks near the start of each new content
range have prior context to attend to.

**The trade-off the preamble fraction controls.**
`DEFAULT_PREAMBLE_FRACTION` (defined in SPEC-CHUNK-411) splits each
segment's token budget between preamble and content:

```
┌──────────── max_tokens_per_segment ────────────┐
│  ░░░ preamble (fraction × budget) ░░░ │ content │
│  ←── encoded for context, then       │ ←── kept │
│     embeddings DISCARDED ──→         │   in     │
│                                      │ output → │
└────────────────────────────────────────────────┘
                                       ↑
                           content_start
```

- **Higher fraction →** more context behind each content chunk
  → context-aware embeddings even near segment boundaries → BUT
  the same chunks get encoded multiple times (once as content,
  then again as preamble for the next segment's pass) → slower
  wall-clock, more compute.
- **Lower fraction →** less redundant work → faster → BUT chunks
  near each segment's `content_start` have weaker context and their
  embeddings are noisier.

Defensible operating band: roughly `[0.25, 0.45]`. Below ≈ 25%, the
first chunk of each segment dominates with insufficient grounding;
above ≈ 50%, more than half the compute is redundant re-encoding of
text whose embeddings are already kept.

### SPEC-CHUNK-411 — Segment construction is greedy with backward preamble

Starting from `content_start = 0`, each segment is constructed as
follows:

1. **Heading prepend reservation (SPEC-CHUNK-470).** If
   `include_headings` is true and the heading stack in scope at
   `content_start` is non-empty, reserve its token count out of the
   preamble budget first. Call this `heading_tokens`; the remaining
   `backward_budget = max(0, preamble_budget - heading_tokens)` is
   what's available for the backward walk in step 2.

2. **Backward walk.** Reserve up to `max_tokens_preamble =
   floor(preamble_fraction * max_tokens_per_segment)` tokens of
   preamble *before* `content_start`. (Use `floor`, not `round`:
   this never overshoots the budget and is unambiguous across
   languages.) Walk backwards from `content_start`, accumulating
   chunks until the next chunk would push the backward-walk token
   count above `backward_budget`. This gives `segment_start`.

   **Token counting during segment construction.** The walk uses
   *isolated-text* token counts from `embedder.count_tokens(...)`.
   These may differ slightly from the joined-input token counts the
   embedder ultimately sees (due to subword merges across boundaries
   or sentinel insertion). Isolated counts are a budgeting heuristic;
   SPEC-CHUNK-420 derives the *authoritative* per-text counts from
   the joined-input tokenization, and SPEC-CHUNK-412's
   largest-remainder allocation absorbs any drift.

   **First segment edge case.** When `content_start == 0`, there are
   no chunks before it. Set `segment_start = 0` and skip the
   backward walk; the unused preamble budget rolls into the content
   budget per step 3. The heading prepend (if any) still applies.

3. **Forward walk.** The unused preamble budget (heading +
   backward-walk slack) is added to the content budget. Walk forward
   from `content_start`, accumulating chunks as content until the
   next chunk would push the *content* tokens above the augmented
   content budget (equivalently: the *segment* tokens above
   `max_tokens_per_segment`). This gives `segment_end`.

4. Append the segment `(segment_start, content_start, segment_end)`.
   Set the next iteration's `content_start = segment_end`.

5. Repeat until `content_start >= len(chunks)`.

**Progress guarantee.** Step 3's forward walk must include at least
one chunk in the content range — that is, `segment_end >
content_start` — so that each iteration consumes at least one chunk
and the loop terminates. If a single content chunk exceeds the
augmented content budget but `preamble_tokens + chunk_tokens` still
fits inside `embedder.n_ctx`, include that single chunk as the
segment's sole content chunk even though the segment exceeds
`max_tokens_per_segment`. If `preamble_tokens + chunk_tokens` would
exceed `embedder.n_ctx`, shrink the preamble in this order:
1. Drop the oldest (earliest-indexed) backward-walk preamble chunks
   one at a time until the sum fits, or until only the heading
   prepend remains.
2. As a last resort, drop the heading prepend.

If even the bare chunk exceeds `embedder.n_ctx`, raise per
SPEC-CHUNK-451.

The default `DEFAULT_PREAMBLE_FRACTION = 0.382` is the inverse golden
ratio (`1 - 1/φ`); it sits in the middle of the defensible operating
band (SPEC-CHUNK-410). Implementations may tune it as a
configuration parameter.

### SPEC-CHUNK-412 — Per-segment encoding and pooling

For each segment `(segment_start, content_start, segment_end)`:

1. Construct the embedder input as a list of texts:
   - If `include_headings` is true and the heading stack at
     `content_start` is non-empty, the heading-stack string is text
     0; the chunks from `chunks[segment_start:segment_end]` follow.
   - Otherwise, the chunks alone.

2. Call `embedder.embed_segment(texts)`. The embedder returns a
   tuple `(token_embeddings, per_text_counts)` where
   `token_embeddings.shape == (T, D)` and the counts sum to `T`.
   How the embedder joins, tokenizes, and aligns is its concern; the
   library treats those choices as opaque.

3. **Largest-remainder safety net.** If
   `sum(per_text_counts) != T`, the library apportions via the
   [largest-remainder method](https://en.wikipedia.org/wiki/Largest_remainder_method):
   each text gets `floor(T * counts[i] / total)` rows, and any
   leftover rows go one each to the texts with the largest fractional
   parts. This absorbs drift between `count_tokens` (approximate,
   used for budgeting) and the actual joined-input tokenization. If
   the embedder's `embed_segment` already conserves the row count
   (the usual case), this step is a no-op.

4. Allocate per-token embeddings to texts sequentially: text `i`
   receives `per_text_counts[i]` consecutive token-embedding rows.

5. Mean-pool the token embeddings within each text.

6. Discard the pooled embedding for the heading-stack prepend (if
   present) and for each chunk in the preamble range (indices
   `[segment_start, content_start)`). Keep the pooled embeddings
   for each chunk in the content range (indices
   `[content_start, segment_end)`).

### SPEC-CHUNK-420 — Per-text token alignment is the embedder's responsibility

The `per_text_counts` returned by `embed_segment` must conserve the
matrix row count and reflect the embedder's actual tokenization of
the joined segment. The *method* of alignment is up to the
embedder's implementation. Common choices:

- **Sentinel-token method.** Insert a stable, rare character between
  texts before tokenizing; locate the sentinel tokens in the output;
  counts are derived from the gaps between sentinel positions.
  Works for tokenizers that treat the sentinel as one atomic token
  across positions. (BERT-family tokenizers subword-merge many
  candidate sentinels — pick the sentinel after probing the specific
  tokenizer in use.)
- **Offset-based method.** Join with no separator; use the
  tokenizer's offset_mapping (HuggingFace `tokenizers`,
  `tiktoken`, etc.) to map each token back to its source text by
  character offset.
- **Custom.** Any approach that yields counts conserving the matrix
  row count is conforming.

**Special tokens.** Many transformer tokenizers inject special
tokens (`[CLS]`, `[SEP]`, BOS, EOS). These belong to no input text.
The embedder must either (a) use a per-token output that excludes
specials, or (b) absorb them into a neighbour text's allocation
(typically: leading specials → text 0; trailing specials → last
text). Either choice is conforming as long as the counts conserve
the matrix row count.

`examples/embedders/` demonstrates each method against a real
embedder.

## Output normalization

### SPEC-CHUNK-430 — Optional L2 normalization

If the function's `normalize` parameter is `True` (the default),
each output row is divided by its L2 norm before being returned.
Otherwise rows are returned as raw mean-pooled vectors.

## Determinism

### SPEC-CHUNK-440 — Deterministic given a deterministic embedder

For a fixed embedder (same weights, same precision, same hardware),
the output is deterministic. Hardware-dependent floating-point
nondeterminism (e.g., GPU reductions in different orders) is not
considered a violation.

## Edge cases

### SPEC-CHUNK-450 — Single chunk input

For `chunks == [c]`, the document has one segment with empty
preamble (the heading prepend is also empty, since there are no
prior chunks establishing a heading context). The output is a
single-row matrix with the mean-pooled embedding of `c` taken from
a single embedder call. (Subject to SPEC-CHUNK-451 if `c` exceeds
`embedder.n_ctx`.)

### SPEC-CHUNK-451 — Chunk longer than the embedder's context

If any single chunk exceeds the embedder's context size in tokens,
no valid segment can contain it. The implementation raises
`ChunkExceedsContextError` before encoding. Splitting the chunk
further is the caller's responsibility (lower `max_size` at stages
2 and 3).

The error type is also exported under the legacy alias
`SentenceExceedsContextError` for back-compat; new code should
prefer `ChunkExceedsContextError`.

### SPEC-CHUNK-452 — Many short texts

For corpora with many short texts (small `tokens_in(c)`), the
largest-remainder allocation may allocate `0` token embeddings to
some texts. The mean-pool of zero vectors is undefined. The
implementation should either floor every text's share at `1` (at
the cost of slight inaccuracy in apportionment) or detect and
raise.

## Heading prepend

### SPEC-CHUNK-470 — Per-segment heading prepend

When `include_headings` is `True` (the default), each segment is
preceded by the Markdown heading stack in scope at the segment's
`content_start`. The heading text is passed to the embedder as
text 0 of the segment input, joining the segment's token sequence
*before* both the backward-walk preamble and the content chunks.
The embedder's attention can reach the heading from any position
in the segment, so chunks anywhere in the content range receive
hierarchical context (the document outline) that they would
otherwise miss.

The heading stack is computed via the same logic as
`heading_paths(chunks)` (SPEC-CHUNK-510 / -511): the stack of
heading lines that were in scope as of the chunk's start position.
For the first content chunk of a segment, that's the relevant
hierarchical context.

**Cost model.** The heading prepend appears *once per segment*, not
once per content chunk. For a segment containing N content chunks
under the same heading, the heading-stack tokens appear once in
the embedder's input, and attention propagates the context to all
N chunks. This is much cheaper than prepending the heading to each
chunk text individually (which would spend the heading-stack token
cost N times for the same gain).

**Token budget.** The heading-stack tokens count against the
preamble budget — they are conceptually preamble (warm-up context),
not content. The budget consumption order is:

1. Heading-stack tokens (up to `preamble_budget`).
2. Backward-walk preamble text (up to whatever remains).
3. Content chunks (with any unused preamble budget rolled into
   content per SPEC-CHUNK-411 step 3).

In the typical case heading-stack tokens are small (~10-50 tokens)
relative to the preamble budget (~2000+ tokens at default
settings), so the backward-walk preamble is barely affected.

**When the heading stack is empty.** Chunks before any heading in
the document have an empty heading path → no prepend → segment
input is just the content (and backward-walk) chunks. The first
segment of a document that opens with a non-heading chunk hits this
case.

**Deeper headings inside the segment.** When the segment text
contains intermediate headings (e.g., the content range spans a
transition from `## Sub A` to `## Sub B`), the second heading's
text is naturally in the segment between the two content ranges,
and attention picks it up there. The single segment-start prepend
covers only the *outermost* heading-stack that would otherwise have
fallen out of the segment's text window.

**Opt-out.** Pass `include_headings=False` for non-markdown inputs,
for ablation against the no-prepend baseline, or when the caller is
managing heading context themselves (e.g., they've already
prepended headings to the chunk text directly).

**Implementation note.** The library calls
`embedder.count_tokens([heading_text])` once per segment whose
heading prepend is non-empty, batched across segments where
practical, to know how many tokens the prepend will consume. This
budgeting call is in addition to the per-chunk `count_tokens`
batch.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `DEFAULT_PREAMBLE_FRACTION` | `0.382` | SPEC-CHUNK-411 |

## Implementation-defined behavior

- Choice of token-level embedder (any model with no-pooling output).
- Choice of per-text token counting method (sentinel, offset-based,
  etc.) subject to SPEC-CHUNK-420.
- Whether to batch segments across multiple embedder calls.
- Storage precision (`float16` vs `float32`) of returned embeddings.
  `float16` is acceptable if downstream similarity computations cast
  to `float32` or higher before computing dot products.
- Whether to expose `DEFAULT_PREAMBLE_FRACTION` and
  `max_tokens_per_segment` as configuration or to derive both from
  the embedder.
- Whether to use a CommonMark parser or a regex for the heading
  detection that drives SPEC-CHUNK-470. The Python implementation
  delegates to `heading_paths()`, which uses a regex for ATX
  headings only — same compromise as SPEC-CHUNK-512.

## Unspecified behavior

- Behavior when the embedder's tokenizer cannot represent characters
  in the document (rare for modern tokenizers; would fall under
  validation errors).
- Behavior when `max_tokens_per_segment` is smaller than any single
  chunk's token count (covered by SPEC-CHUNK-451).
