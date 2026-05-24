# Spec 04 — Late Chunking

An optional embedding strategy that produces per-sentence embeddings
which incorporate the context of a longer surrounding document
segment. Late chunking replaces the standard "encode each chunk in
isolation" pattern; it is interchangeable with stage-2/stage-3 input
provided the embedder supports per-token output.

This spec describes a contract on the *embedding function*, not on
fancychunk's pipeline directly. A late-chunking embed function can be
plugged into stages 2 and 3 to produce more context-aware embeddings.

> **Origin and rationale.** The technique was introduced and named in
> Weaviate's blog post
> [*Late Chunking in Long-Context Embedding Models*](https://weaviate.io/blog/late-chunking),
> and quantified in the Jina AI paper
> [*Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*](https://arxiv.org/abs/2409.04701)
> (Günther et al., 2024). The core observation: when a chunk is
> embedded in isolation, anaphoric references ("it", "this method",
> "the algorithm") lose their referents because the surrounding
> context is gone. Encoding the whole context first and pooling
> *after* preserves those references in the per-chunk embedding.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `sentences` | list of strings | yes | — | The sentences to embed, in document order. Typically the output of stage 1. |
| `embedder` | object satisfying the embedder contract below | yes | — | A token-level embedding model. |
| `max_tokens_per_segment` | positive integer | no | derived from the embedder's context window | The upper bound on tokens fed to the embedder in one call. |
| `preamble_fraction` | float in `(0, 1)` | no | `DEFAULT_PREAMBLE_FRACTION` (`= 0.382`) | Fraction of `max_tokens_per_segment` reserved for the segment's preamble. See U-CHUNK-401. |

## Outputs

A matrix of shape `[len(sentences), embedding_dim]`. Row `i` is the
embedding of `sentences[i]`, computed with context from surrounding
sentences.

- **SPEC-CHUNK-400** — One row per input sentence, in the same order.
- **SPEC-CHUNK-401** — Each row is a fixed-dimensional vector of
  floats.
- **SPEC-CHUNK-402** — Rows are L2-normalized when the embedder is
  configured to normalize; otherwise rows reflect the raw mean-pooled
  token embeddings.

## Embedder contract

The embedder is a black box that satisfies three operations:

| Operation | Behavior |
|-----------|----------|
| `tokenize(text) → list[int]` | Returns the token IDs the model would receive for this text. Required to be deterministic. |
| `detokenize(list[int]) → str` | Inverse of `tokenize`, used only for sentinel-token discovery (SPEC-CHUNK-420). |
| `embed(text) → matrix[T, D]` | Returns one embedding vector per token in `text`, with `T` equal to the number of tokens. The embedder must NOT pool tokens internally for this call. |
| `n_ctx → int` | The maximum number of input tokens per `embed` call. |


The reimplementor may use any embedding model that exposes
token-level outputs (a "no pooling" or "per-token output" mode).
Cloud embedding APIs that return only one vector per input do not
satisfy the contract.

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

> **Why preambles exist.** Transformer embeddings are *contextual* —
> every token's output vector is a function of the surrounding tokens
> via attention. Consider:
>
> > "The algorithm achieves O(n log n) by maintaining a balanced
> > binary tree."
>
> Encoded in isolation, "the algorithm" has no antecedent and the
> embedder produces a generic "thing being discussed" direction for
> it. Encoded with the preceding heading "## Quicksort with random
> pivot selection" as preamble, the attention mechanism connects "the
> algorithm" to "Quicksort" and the embedding for that token picks up
> a meaningful direction. The preamble exists so that sentences near
> the start of each new content range have prior context to attend to,
> even when those sentences happen to fall near a segment boundary.
> Without it, late chunking degrades to standard chunking around
> segment edges.
>
> **The trade-off the preamble fraction controls.**
> `DEFAULT_PREAMBLE_FRACTION` (defined in SPEC-CHUNK-411) splits each
> segment's token budget between preamble and content:
>
> ```
> ┌──────────── max_tokens_per_segment ────────────┐
> │  ░░░ preamble (fraction × budget) ░░░ │ content │
> │  ←── encoded for context, then       │ ←── kept │
> │     embeddings DISCARDED ──→         │   in     │
> │                                      │ output → │
> └────────────────────────────────────────────────┘
>                                        ↑
>                            content_start
> ```
>
> - **Higher fraction →** more context behind each content sentence
>   → context-aware embeddings even near segment boundaries → BUT
>   the same sentences get encoded multiple times (once as content,
>   then again as preamble for the next segment's pass) → slower
>   wall-clock, more compute.
> - **Lower fraction →** less redundant work → faster → BUT sentences
>   near each segment's `content_start` have weaker context and their
>   embeddings are noisier.
>
> Defensible operating band: roughly `[0.25, 0.45]`. Below ≈ 25%, the
> first paragraph of each segment dominates with insufficient
> grounding; above ≈ 50%, more than half the compute is redundant
> re-encoding of text whose embeddings are already kept.

### SPEC-CHUNK-411 — Segment construction is greedy with backward preamble

Starting from `content_start = 0`, each segment is constructed as
follows:

1. Reserve up to `max_tokens_preamble = round(preamble_fraction *
   max_tokens_per_segment)` tokens of preamble *before*
   `content_start`. Walk backwards from `content_start`,
   accumulating sentences until the next sentence would push the
   preamble token count above the budget. This gives
   `segment_start`.

2. The unused preamble budget (if any) is added to the content
   budget. So if the preamble used fewer tokens than allowed (e.g.,
   the document just started), content gets the remainder.

3. Walk forward from `content_start`, accumulating sentences as
   content until the next sentence would push the segment's total
   tokens above the (possibly augmented) content budget. This gives
   `segment_end`.

4. Append the segment `(segment_start, content_start, segment_end)`.
   Set the next iteration's `content_start = segment_end`.

5. Repeat until `content_start >= len(sentences)`.

The constant `DEFAULT_PREAMBLE_FRACTION = 0.382` (the inverse golden
ratio, `1 - 1/φ`) and the addition-of-unused-budget behavior are
preserved as defaults. The trade-off the fraction controls is
explained under SPEC-CHUNK-410.

### SPEC-CHUNK-412 — Per-segment encoding and pooling

For each segment `(segment_start, content_start, segment_end)`:

1. Concatenate the segment's sentences (preamble + content) into one
   string and call the embedder's `embed` operation. The result is a
   matrix of per-token embeddings.

2. Compute the per-sentence token count *as the embedder would see
   it* for each sentence in the segment (SPEC-CHUNK-420 gives one
   implementation).

3. Apportion the per-token embeddings to sentences using the
   [largest remainder method](https://en.wikipedia.org/wiki/Largest_remainder_method)
   (a standard apportionment algorithm from voting systems, applied
   here so the per-sentence token counts sum to exactly the segment's
   total):

   - For each sentence `s` in the segment, the fractional share is
     `len(token_embeddings) * (tokens_in(s) / total_tokens)`.
   - Each sentence gets `floor(fractional_share)` token embeddings
     to start.
   - Any leftover token embeddings (the integer remainder) are
     distributed one each to the sentences with the largest
     fractional parts.

4. Mean-pool the token embeddings within each sentence.

5. Discard the pooled embeddings for sentences in the preamble range
   (indices `[segment_start, content_start)`); keep those in the
   content range (indices `[content_start, segment_end)`).

The largest-remainder allocation is preserved as part of the spec —
it ensures the per-sentence token counts sum to exactly the segment's
token count without leaving any embedding unallocated.

### SPEC-CHUNK-420 — Per-sentence token counts must align with what the embedder saw

The per-sentence token count required by SPEC-CHUNK-412 step 2 must
equal the number of token embeddings that the embedder produced for
that sentence's substring within the concatenated segment input.

This is non-trivial because:
- Tokenizing each sentence in isolation can produce a different total
  than tokenizing the concatenation (e.g., subword merges across
  sentence boundaries).
- Many tokenizers drop or add tokens at the start/end of input.

The reimplementor must use a method that recovers per-sentence counts
*from the embedder's actual tokenization of the joined input*.

One valid implementation (used by the source under analysis) is the
sentinel-token method:

1. Pick a character that, when inserted between sentences, tokenizes
   to a known sentinel token. The source uses `⊕` (CIRCLED PLUS, U+2295).
2. Tokenize `sentinel.join(sentences)`.
3. Locate the sentinel tokens in the resulting sequence.
4. Per-sentence token counts are the differences between consecutive
   sentinel positions (with sentinel tokens themselves counted as
   part of the preceding sentence).

Other valid implementations include: tokenizing the joined input and
mapping byte offsets back to sentence boundaries (if the tokenizer
exposes offset metadata), or any other mechanism that produces
correct per-sentence counts. The sentinel method is not normative.

### SPEC-CHUNK-421 — Sentinel character requirements (if using SPEC-CHUNK-420's example method)

If the sentinel-token method is used, the sentinel character must
satisfy:

- It does not appear anywhere in the input document. If it does, the
  implementation must detect this and either choose a different
  sentinel or fall back to a different counting method.
- It tokenizes to at least one stable token across all positions in
  the input it can occupy (start of string, mid-sequence, after
  whitespace, etc.). If the tokenizer produces token variants
  depending on context, all variants must be recognized as sentinels.

## Output normalization

### SPEC-CHUNK-430 — Optional L2 normalization

If the embedder configuration requests normalization, each output row
is divided by its L2 norm before being returned. Otherwise rows are
returned as raw mean-pooled vectors.

### SPEC-CHUNK-431 — Precision

Output embeddings may be stored in `float16` precision to save space,
provided downstream similarity computations cast to `float32` or
higher before computing dot products.

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
embedding of `s` taken from a single embedder call.

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

| Name | Value | Spec ref | Type |
|------|-------|----------|------|
| `DEFAULT_PREAMBLE_FRACTION` | `0.382` | SPEC-CHUNK-411 | tunable (heuristic; U-CHUNK-401) |

## Implementation-defined behavior

- Choice of token-level embedder (any model with no-pooling output).
- Choice of per-sentence token counting method (sentinel,
  offset-based, etc.) subject to SPEC-CHUNK-420.
- Whether to batch segments across multiple embedder calls.
- Storage precision (`float16` vs `float32`) of returned embeddings.
- Whether to expose `DEFAULT_PREAMBLE_FRACTION` and
  `max_tokens_per_segment` as configuration or to derive both from
  the embedder.

## Unspecified behavior

- Behavior when the embedder's tokenizer cannot represent characters
  in the document (rare for modern tokenizers; would fall under
  validation errors).
- Behavior when `max_tokens_per_segment` is smaller than any single
  sentence's token count (covered by SPEC-CHUNK-451).

## Uncertainties

### U-CHUNK-401 — Choice of `DEFAULT_PREAMBLE_FRACTION = 0.382`

We use `0.382` because raglite uses it and the inverse golden ratio
is aesthetically pleasing. It sits in the middle of the defensible
operating band (see SPEC-CHUNK-410). Implementors may tune it but
should default to `0.382` for behavioral parity with the reference.

### U-CHUNK-402 — Sentinel choice

The character `⊕` (U+2295) is used as the sentinel in the source. It
is chosen because most tokenizers handle it as a stable single token
and it is rare in natural language. The reimplementor may use any
character or method satisfying SPEC-CHUNK-420 and SPEC-CHUNK-421.

### U-CHUNK-403 — Why per-token outputs, why not pool inside the embedder

The technique exists because most embedders pool tokens *before*
returning, but late chunking needs the per-token outputs to pool
within sentence boundaries that the embedder doesn't know about. If
the embedder exposed a "pool over these token ranges" API, late
chunking would not need the explicit per-token output and pooling. No
such standard API exists in 2026, so the spec requires per-token
output.
