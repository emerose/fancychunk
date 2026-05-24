# Spec 04 — Late Chunking

An optional embedding strategy that produces per-sentence embeddings
which incorporate the context of a longer surrounding document
segment. Late chunking replaces the standard "encode each chunk in
isolation" pattern; it is interchangeable with stage-2/stage-3 input
provided the embedder supports per-token output.

This spec describes a contract on the *embedding function*, not on
the pipeline directly. It replaces the caller's "embed each chunklet"
step that sits between stage 2 and stage 3: instead of embedding each
chunklet directly, the caller embeds each *sentence* with late
chunking, then aggregates per-chunklet (typically by mean-pool over
the sentences in each chunklet) to produce the
`chunklet_embeddings` matrix stage 3 requires. The aggregation step
is the caller's responsibility; this spec produces sentence
embeddings only.

The technique was introduced and named in Weaviate's blog post
[*Late Chunking in Long-Context Embedding Models*](https://weaviate.io/blog/late-chunking),
and quantified in the Jina AI paper
[*Late Chunking: Contextual Chunk Embeddings Using Long-Context
Embedding Models*](https://arxiv.org/abs/2409.04701) (Günther et al.,
2024). The core observation: when a chunk is embedded in isolation,
anaphoric references ("it", "this method", "the algorithm") lose
their referents because the surrounding context is gone. Encoding the
whole context first and pooling *after* preserves those references in
the per-chunk embedding.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sentences` | list of strings | yes | — | The sentences to embed, in document order. Typically the output of stage 1. |
| `embedder` | object satisfying the embedder contract below | yes | — | A token-level embedding model. |
| `max_tokens_per_segment` | positive integer | no | `embedder.n_ctx` | The upper bound on tokens fed to the embedder in one call. Defaults to the embedder's reported context-window size. |
| `preamble_fraction` | float in `[0, 1)` | no | `DEFAULT_PREAMBLE_FRACTION` (`= 0.382`) | Fraction of `max_tokens_per_segment` reserved for the segment's preamble. `0.0` is permitted and degenerates to standard chunking (no contextualization); useful for benchmarking against late chunking. Values `≥ 1.0` are rejected (the segment would have no content). |

## Outputs

A matrix of shape `[len(sentences), embedding_dim]`. Row `i` is the
embedding of `sentences[i]`, computed with context from surrounding
sentences.

- **SPEC-CHUNK-400** — One row per input sentence, in the same order.
- **SPEC-CHUNK-401** — Each row is a fixed-dimensional vector of
  floats. The dimension `D` equals the per-token output dimension of
  the embedder's `embed` operation (the embedder's hidden size).
- **SPEC-CHUNK-402** — Rows are L2-normalized when the function's
  `normalize` parameter is `True` (the default); otherwise rows
  reflect the raw mean-pooled token embeddings. The embedder itself
  does not control normalization.

## Embedder contract

The embedder is a black box that satisfies four operations:

| Operation | Kind | Behavior |
|-----------|------|----------|
| `tokenize(text)` | method, returns `list[int]` | Returns the token IDs the model would receive for this text. Required to be deterministic. |
| `detokenize(list[int])` | method, returns `str` | Inverse of `tokenize`, used only for sentinel-token discovery (SPEC-CHUNK-420). |
| `embed(text)` | method, returns matrix `[T, D]` | Returns one embedding vector per token in `text`, with `T` equal to the number of tokens. The embedder must NOT pool tokens internally for this call. |
| `n_ctx` | integer attribute (property; not a method) | The maximum number of input tokens per `embed` call. `n_ctx` is the canonical Python attribute name; in other languages or libraries, an idiomatic equivalent (`max_tokens`, `context_size`, `model_max_length`) is fine as long as the value's meaning matches. |

Any embedding model that exposes token-level outputs (a "no pooling"
or "per-token output" mode) is acceptable. Cloud embedding APIs that
return only one vector per input do not satisfy the contract — late
chunking needs the per-token outputs to pool within sentence
boundaries that the embedder doesn't know about.

## Behavior

### SPEC-CHUNK-410 — Segments cover the document exactly once

The document is processed as a sequence of *segments*. Each segment
has two parts:

- **Preamble:** sentences before the segment's content that provide
  contextualization. Their token-level embeddings are discarded after
  pooling.
- **Content:** sentences whose final per-sentence embeddings are kept.

Every sentence appears in exactly one segment's *content* range
(though it may also appear as another segment's *preamble*).

**Why preambles exist.** Transformer embeddings are *contextual* —
every token's output vector is a function of the surrounding tokens
via attention. Consider:

> "The algorithm achieves O(n log n) by maintaining a balanced
> binary tree."

Encoded in isolation, "the algorithm" has no antecedent and the
embedder produces a generic "thing being discussed" direction for it.
Encoded with the preceding heading "## Quicksort with random pivot
selection" as preamble, the attention mechanism connects "the
algorithm" to "Quicksort" and the embedding for that token picks up a
meaningful direction. The preamble exists so that sentences near the
start of each new content range have prior context to attend to, even
when those sentences happen to fall near a segment boundary. Without
it, late chunking degrades to standard chunking around segment edges.

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

- **Higher fraction →** more context behind each content sentence
  → context-aware embeddings even near segment boundaries → BUT
  the same sentences get encoded multiple times (once as content,
  then again as preamble for the next segment's pass) → slower
  wall-clock, more compute.
- **Lower fraction →** less redundant work → faster → BUT sentences
  near each segment's `content_start` have weaker context and their
  embeddings are noisier.

Defensible operating band: roughly `[0.25, 0.45]`. Below ≈ 25%, the
first paragraph of each segment dominates with insufficient grounding;
above ≈ 50%, more than half the compute is redundant re-encoding of
text whose embeddings are already kept.

### SPEC-CHUNK-411 — Segment construction is greedy with backward preamble

Starting from `content_start = 0`, each segment is constructed as
follows:

1. Reserve up to `max_tokens_preamble = floor(preamble_fraction *
   max_tokens_per_segment)` tokens of preamble *before*
   `content_start`. (Use `floor`, not `round`: this never overshoots
   the budget and is unambiguous across languages — `round`'s
   half-to-even vs. half-up behavior differs between standard
   libraries.) Walk backwards from `content_start`,
   accumulating sentences until the next sentence would push the
   preamble token count above the budget. This gives
   `segment_start`.

   **Token counting during segment construction.** The walk uses
   *isolated-sentence* token counts (e.g., `len(embedder.tokenize(s))`
   per sentence). These may differ slightly from the joined-input
   token counts the embedder ultimately sees (due to subword merges
   across sentence boundaries or sentinel insertion). Isolated counts
   are a budgeting heuristic; SPEC-CHUNK-420 derives the *authoritative*
   per-sentence token counts from the joined-input tokenization, and
   SPEC-CHUNK-412's largest-remainder allocation absorbs any drift.

   **First segment edge case:** when `content_start == 0`, there are
   no sentences before it. Set `segment_start = 0` and skip the
   backward walk; the full preamble budget is then unused and rolls
   into the content budget per step 2.

2. The unused preamble budget (if any) is added to the content
   budget. So if the preamble used fewer tokens than allowed (e.g.,
   the document just started), content gets the remainder.

3. Walk forward from `content_start`, accumulating sentences as
   content until the next sentence would push the *content* tokens
   above the augmented content budget (equivalently: the *segment*
   tokens above `max_tokens_per_segment`). Token counts here are
   isolated-sentence counts, same as in step 1. This gives
   `segment_end`.

4. Append the segment `(segment_start, content_start, segment_end)`.
   Set the next iteration's `content_start = segment_end`.

5. Repeat until `content_start >= len(sentences)`.

**Progress guarantee.** Step 3's forward walk must include at least
one sentence in the content range — that is, `segment_end >
content_start` — so that each iteration consumes at least one
sentence and the loop terminates. If a single content sentence
exceeds the augmented content budget but `preamble_tokens +
sentence_tokens` still fits inside `embedder.n_ctx`, include that
single sentence as the segment's sole content sentence even though
the segment exceeds `max_tokens_per_segment`. If even
`preamble_tokens + sentence_tokens` exceeds `embedder.n_ctx`, shrink
the preamble — drop the oldest (earliest-indexed) preamble sentences
one at a time until the sum fits, or until the preamble is empty.
If `sentence_tokens` alone exceeds `embedder.n_ctx`, raise per
SPEC-CHUNK-451.

The default `DEFAULT_PREAMBLE_FRACTION = 0.382` is the inverse golden
ratio (`1 - 1/φ`); it sits in the middle of the defensible operating
band (SPEC-CHUNK-410) and is aesthetically pleasing. Implementations
may tune it as a configuration parameter.

### SPEC-CHUNK-412 — Per-segment encoding and pooling

For each segment `(segment_start, content_start, segment_end)`:

1. Build the joined string per the counting method chosen for
   SPEC-CHUNK-420 (the method itself specifies which joiner to use).
   Call the embedder's `embed` operation on this same string. The
   result is a matrix of per-token embeddings; the per-sentence token
   counts derived in step 2 must add up to exactly this matrix's row
   count.

2. Compute the per-sentence token count *as the embedder would see
   it* for each sentence in the segment, using SPEC-CHUNK-420.

3. Allocate per-token embeddings to sentences using the
   per-sentence token counts from step 2: sentence `s` receives
   `tokens_in(s)` consecutive token-embedding rows from the
   beginning of the segment forward. When SPEC-CHUNK-420 holds,
   `sum(tokens_in(s)) == len(token_embeddings)` exactly and the
   allocation is unambiguous.

   If an implementation derives `tokens_in(s)` through any
   approximate method (e.g., re-tokenizing each sentence in
   isolation), the integer counts may not sum to the segment's
   actual token count. In that case use the
   [largest-remainder method](https://en.wikipedia.org/wiki/Largest_remainder_method)
   to apportion the leftover: each sentence gets
   `floor(len(token_embeddings) * tokens_in(s) / total_tokens)`
   rows, and any remaining rows go one each to the sentences with
   the largest fractional parts. This is a safety net; with
   SPEC-CHUNK-420 satisfied it reduces to the identity allocation.

4. Mean-pool the token embeddings within each sentence.

5. Discard the pooled embeddings for sentences in the preamble range
   (indices `[segment_start, content_start)`); keep those in the
   content range (indices `[content_start, segment_end)`).

### SPEC-CHUNK-420 — Per-sentence token counts must align with what the embedder saw

The per-sentence token count required by SPEC-CHUNK-412 step 2 must
equal the number of token embeddings that the embedder produced for
that sentence's substring within the concatenated segment input.

This is non-trivial because:
- Tokenizing each sentence in isolation can produce a different total
  than tokenizing the concatenation (e.g., subword merges across
  sentence boundaries).
- Many tokenizers drop or add tokens at the start/end of input.
- Many transformer tokenizers add special tokens (`[CLS]`, `[SEP]`,
  BOS, EOS) at the start and/or end of the sequence. These tokens
  belong to no input sentence. The implementation must either (a)
  use an `embed` operation that strips special tokens before
  returning per-token embeddings, or (b) detect them in the output
  and assign them to a neighbouring content sentence (typically: a
  leading special token joins sentence 0's allocation; a trailing
  special token joins the last content sentence's allocation). Either
  choice must be applied consistently so that
  `sum(tokens_in(s)) == len(token_embeddings)`.

The implementation must use a method that recovers per-sentence
counts *from the embedder's actual tokenization of the joined input*.

One valid implementation is the sentinel-token method:

1. Pick a character that, when inserted between sentences, tokenizes
   to a known sentinel token. A good default is `⊕` (CIRCLED PLUS,
   U+2295) — most tokenizers handle it as a stable single token and
   it is rare in natural language.
2. Tokenize `sentinel.join(sentences)`. Note that there is no
   leading or trailing sentinel — sentinels appear only *between*
   sentences.
3. Locate the sentinel token positions in the resulting sequence:
   call them `s_1, s_2, ..., s_{k-1}` for `k` sentences.
4. Sentence 0 spans token positions `[0, s_1]` (from the start
   through and including the first sentinel). Sentence `j` for
   `1 ≤ j ≤ k-2` spans `(s_j, s_{j+1}]`. The last sentence
   (`j = k-1`) spans `(s_{k-1}, end_of_sequence)`. In each case the
   per-sentence token count is the number of token positions in the
   span; sentinel tokens are counted as part of the preceding
   sentence.

Other valid implementations include the **offset-based method**:
join sentences with the empty string (i.e., `"".join(sentences)`),
tokenize the result, and use the tokenizer's offset metadata (if
exposed — e.g., HuggingFace `tokenizers` returns `(start, end)` byte
offsets per token) to map each token back to its source sentence by
character position. Per-sentence token counts are the number of
tokens whose offset range falls within each sentence's character
range. The joiner here is the empty string because no separator is
needed when offsets are available.

Any method satisfying this section and SPEC-CHUNK-421 is acceptable.
The sentinel method is not normative.

### SPEC-CHUNK-421 — Sentinel character requirements (if using the sentinel method)

If the sentinel-token method is used, the sentinel character must
satisfy:

- It does not appear anywhere in the input document. If it does, the
  implementation must detect this and either choose a different
  sentinel or fall back to a different counting method.
- It tokenizes to at least one stable token across all positions in
  the input it can occupy (start of string, mid-sequence, after
  whitespace, etc.). If the tokenizer produces token variants
  depending on context, all variants must be recognized as sentinels.

Sentinel discovery uses both `tokenize` and `detokenize`. A
conforming approach: tokenize a small probe string that contains the
candidate sentinel in several positions, collect the candidate
sentinel token IDs, then for each candidate ID call
`detokenize([id])` and check that the result contains the sentinel
character. This identifies the set of token IDs the implementation
must recognize as sentinels in step 3 of SPEC-CHUNK-420.

## Output normalization

### SPEC-CHUNK-430 — Optional L2 normalization

If the function's `normalize` parameter is `True` (the default), each
output row is divided by its L2 norm before being returned. Otherwise
rows are returned as raw mean-pooled vectors.

## Determinism

### SPEC-CHUNK-440 — Deterministic given a deterministic embedder

For a fixed embedder (same weights, same precision, same hardware),
the output is deterministic. Hardware-dependent floating-point
nondeterminism (e.g., GPU reductions in different orders) is not
considered a violation.

## Edge cases

### SPEC-CHUNK-450 — Single sentence input

For `sentences == [s]`, the document has one segment with empty
preamble. The output is a single-row matrix with the mean-pooled
embedding of `s` taken from a single embedder call. (Subject to
SPEC-CHUNK-451 if `s` exceeds `embedder.n_ctx`.)

### SPEC-CHUNK-451 — Sentence longer than the embedder's context

If any single sentence exceeds the embedder's context size in tokens,
no valid segment can contain it. The implementation should raise an
error before encoding. Splitting the sentence further is the caller's
responsibility (use stage 1's `max_len` parameter).

### SPEC-CHUNK-452 — Many short sentences

For corpora with many short sentences (small `tokens_in(s)`), the
largest-remainder allocation may allocate `0` token embeddings to
some sentences. The mean-pool of zero vectors is undefined. The
implementation should either floor every sentence's share at `1`
(at the cost of slight inaccuracy in apportionment) or detect and
raise.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `DEFAULT_PREAMBLE_FRACTION` | `0.382` | SPEC-CHUNK-411 |

## Implementation-defined behavior

- Choice of token-level embedder (any model with no-pooling output).
- Choice of per-sentence token counting method (sentinel,
  offset-based, etc.) subject to SPEC-CHUNK-420.
- Whether to batch segments across multiple embedder calls.
- Storage precision (`float16` vs `float32`) of returned embeddings.
  `float16` is acceptable if downstream similarity computations cast
  to `float32` or higher before computing dot products.
- Whether to expose `DEFAULT_PREAMBLE_FRACTION` and
  `max_tokens_per_segment` as configuration or to derive both from
  the embedder.

## Unspecified behavior

- Behavior when the embedder's tokenizer cannot represent characters
  in the document (rare for modern tokenizers; would fall under
  validation errors).
- Behavior when `max_tokens_per_segment` is smaller than any single
  sentence's token count (covered by SPEC-CHUNK-451).
