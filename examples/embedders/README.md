# Reference SegmentEmbedder adapters

The fancychunk library doesn't host any embedding models — the
`embed_with_late_chunking` function takes a caller-supplied
`SegmentEmbedder` (see [spec 04 §Embedder contract](../../docs/specs/04-late-chunking.md#embedder-contract)).

This directory contains reference adapters covering the three common
deployment shapes. Each file is self-contained and runnable.

| File | Backend | Best for |
|---|---|---|
| [`qwen3_mlx.py`](qwen3_mlx.py) | MLX + Qwen3-Embedding | Apple Silicon; offline / batch indexing |
| [`huggingface_offsets.py`](huggingface_offsets.py) | HuggingFace transformers | Any platform, any model with a fast tokenizer; recommended default |
| [`remote_http.py`](remote_http.py) | HTTP client + tokenizer locally | When the GPU and the chunking logic live on different machines |

## Picking an alignment method

The library's protocol asks the embedder to return per-text token
counts that sum to the row count of the embedding matrix. (The
"texts" are whatever fancychunk passes — chunks, plus optionally one
heading-stack prepend.) *How* the embedder produces those counts is
its choice. Two methods cover ~99% of real embedders:

### Offset-based (preferred when available)

Join the input texts with the empty string, tokenize once, use the
tokenizer's `offset_mapping` to map each token back to its source
text by character offset. **No sentinel character needed, no risk of
subword-merging surprises.** Works with any HuggingFace fast
tokenizer (i.e., any modern tokenizer based on the Rust `tokenizers`
crate). This is what [`huggingface_offsets.py`](huggingface_offsets.py)
does.

### Sentinel-token

Pick a character (e.g., `§`, `¶`) that the tokenizer treats as a
stable single token across positions; join texts with it before
tokenizing; locate the sentinel positions in the output and derive
counts from the gaps. Use this when the tokenizer doesn't expose
offsets (some MLX builds, older tokenizers). [`qwen3_mlx.py`](qwen3_mlx.py)
demonstrates this against MLX's Qwen3 implementation.

**Important caveat for sentinel:** many tokenizers — notably the BPE
families used by BERT and some Llama variants — subword-merge
candidate sentinels in context. Probe your specific tokenizer before
committing. `⊕` (CIRCLED PLUS) is a poor choice for BERT-family
models; `§` works for both BERT and Qwen3-Embedding.

### Special-token handling (SPEC-CHUNK-420 option b)

Transformer tokenizers typically inject `[CLS]`, `[SEP]`, BOS, EOS
tokens. These belong to no input text. Both reference adapters
absorb leading specials into text 0's count and trailing specials
into the last text's count — the conforming "option (b)" from the
spec. Choosing option (a) (an `embed_segment` that excludes specials
from its output) is also conforming if your embedder supports it.

## Skeleton: anatomy of a SegmentEmbedder

```python
class MyEmbedder:
    n_ctx = 4096  # tokens per segment (your context window)

    def count_tokens(self, texts: list[str]) -> list[int]:
        # Approximate per-text count, used by fancychunk for
        # segment-budget planning. May differ from the actual
        # joined-input tokenization by small amounts — the
        # largest-remainder safety net in fancychunk absorbs drift.
        return [len(your_tokenizer.encode(s)) for s in texts]

    def embed_segment(self, texts: list[str]) -> tuple[NDArray, list[int]]:
        # 1. Tokenize the joined input however you like.
        # 2. Run the model, get last_hidden_state (per-token output).
        # 3. Compute per-text counts that conserve the row count.
        # 4. Return (matrix, counts).
        return matrix, counts
```

That's it. Twenty lines of glue per backend; the rest is the
algorithm, which lives in fancychunk.

## Performance notes

- **MLX on Apple Silicon** is 2–4× faster than PyTorch + MPS for the
  same model. If you're on a Mac, prefer the MLX adapter.
- **`bge-m3` and similar multilingual models** are excellent general
  defaults: 8k context, ~95% the quality of Qwen3-Embedding at a
  fraction of the size.
- **Remote embedding** adds network round-trip per `embed_segment`
  call. Co-locate the embedder with the chunking logic when possible;
  if you do go remote, use a binary protocol (msgpack, protobuf) —
  the matrix transport dominates JSON serialization cost.
- **`bench_qwen3.py`** at the repo root has a complete end-to-end
  benchmark you can adapt for other models.
