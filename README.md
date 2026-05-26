# fancychunk

> Markdown chunking for RAG that attempts to craft artisanal,
> meaningful chunks while remaining reasonably fast and efficient.

```bash
pip install fancychunk
```

## How it compares

Traditional chunkers split at character or token counts, possibly
including a recursive separator list to dodge the worst cuts. This
is fast and efficient, but can lead to awkward breaks and chunks
that don't capture a particular idea well.  Other chunkers use an 
LLM to find meaningful semantic boundaries, but this is slow and
expensive, and can be inconsistent.

fancychunk attempts to find a middle ground, producing meaningful
chunks reasonably quickly. It uses markdown structure alongside
multiple small, local models to produce meaningful, correctly-sized
chunks that capture the underlying text's semantic value well.

[insert benchmark results: MB/sec throughput and example 
NDCG@10/Recall@10/MRR@10 stats from ragkit.  compare: 
- simple token-count splitter from langchain
- chonkie's recursive splitter
- chonkie semantic splitter
- fancychunk]

## Quick start

```python
from fancychunk import chunk_document
from fancychunk.embedders import qwen3_600m

embedder = qwen3_600m()                          # probably the right pick for most uses
chunks, vectors = chunk_document(open("my-document.md").read(), embedder)
# chunks[i] ⇄ vectors[i] — drop straight into your vector store.
```

That's it. `chunk_document` runs the full pipeline — semantic
topic-shift chunking, then late chunking to produce one
context-aware vector per chunk with the document's heading stack
folded into the embedding context. The same embedder instance is
used for the partition decision and the late-chunking pass, so the
model loads exactly once.

`qwen3_600m()` is the recommended default for most workloads
(~596M params, native 1024-dim, MTEB Multilingual 64.33 — the
leader at sub-1B). For other choices see [Models](#models) below.

### Building blocks

`chunk_document` is sugar over the four primitives. Compose them
directly when you want more control — different embedders per
stage, different `max_size` per stage, a structural-only split, or
storage-time heading breadcrumbs:

```python
from fancychunk import (
    split_sentences,
    split_chunklets,
    split_chunks,
    embed_with_late_chunking,
    enrich_with_headings,
)
from fancychunk.embedders import qwen3_600m

embedder = qwen3_600m()
doc = open("my-document.md").read()

sentences = split_sentences(doc, max_len=2048)
chunklets = split_chunklets(sentences, max_size=2048)
chunks    = split_chunks(chunklets, embedder, max_size=2048)
vectors   = embed_with_late_chunking(chunks, embedder)

# Optional: decorate the *stored* chunk text with its heading path
# as a retrieval-time breadcrumb. `enrich_with_headings` does NOT
# affect `vectors` — late chunking already saw the in-document
# headings via its per-segment heading prepend.
chunks = enrich_with_headings(chunks)
```

For a no-model-download structural split, swap the embedder for
`embedders.noop()`:

```python
from fancychunk.embedders import noop
chunks = split_chunks(chunklets, noop(), max_size=2048)
```

`noop()` returns constant per-chunklet vectors, which collapses
the semantic-similarity term and leaves heading-aware boundaries
as the only signal — the same shape the old "no embeddings
supplied" path produced.

**Lifecycle.** Each call to `qwen3_600m()` (or any factory)
returns a fresh embedder instance. The model weights load lazily
on first use of `embed_chunklets` / `embed_segment`. Hold the
embedder reference while you need it; drop it to free its memory.
Two separate factory calls = two independent instances and two
model loads — pass one instance everywhere when you want to share
weights, which is what `chunk_document` does internally.

## What it does

fancychunk treats chunking as three separable problems, each solved 
by its own optimization against its own signal:

```
document  →  split_sentences  →  split_chunklets  →  split_chunks  →  chunks
              (punctuation +     (Markdown headings,    (cosine of
               SaT segmenter)     paragraphs, lists)     adjacent chunklets,
                                                         discourse-corrected)
```

**Stage 1 — `split_sentences`.** Punctuation alone misses too many
real-world cases (missing terminals, multilingual text, technical
abbreviations like "e.g."), so the default segmenter is
[SaT](https://arxiv.org/abs/2406.16678) (Frohmann et al., 2024) from
`wtpsplit-lite` — a learned model that produces per-character
boundary probabilities. A sliding-window dynamic-programming pass
(O(N) amortised) then picks boundary positions to maximise total
score subject to a configurable min/max sentence length.

**Stage 2 — `split_chunklets`.** Sentences are grouped into
*chunklets* — paragraph-sized units targeting roughly three
"statements" of information content each. The signal is Markdown
block-level structure and a document-relative *statement density* 
measure. A 1-D dynamic-programming pass picks chunklet boundaries 
big enough to embed meaningfully but small enough that each one 
stays topically coherent.

**Stage 3 — `split_chunks`.** Adjacent chunklets are compared by
cosine similarity, then *discourse-corrected* — the mean of typical
chunklets' embeddings is projected out so similarity reflects local
topic shifts rather than the document's overall theme
([Arora et al., 2017](https://openreview.net/forum?id=SyK00v5xx)).
A third dynamic-programming pass picks split points where adjacent
chunklets are *least* similar (this is "level 4" in Greg Kamradt's
[5 Levels of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s)
taxonomy), subject to a hard max-size covering constraint.

## Enrichment

The pipeline includes two enrichment steps that pull document
context into each chunk's output. **Both are baked into
`chunk_document`** with sensible defaults; the building-blocks form
exposes them as separate primitives.

### Late chunking (does the heavy lifting)

`embed_with_late_chunking(chunks, embedder)` produces one
context-aware vector per chunk. Instead of embedding each chunk in
isolation, the embedder sees windows of adjacent chunks together so
attention can resolve anaphora ("the algorithm" picks up its real
referent), and the in-scope Markdown heading stack is **prepended
once per segment** as additional preamble (controlled by
`include_headings=True`, on by default). Typical retrieval-quality
win is 4–8 MTEB points (Jina AI's paper has the numbers).

Because the heading stack is already in the embedder's input, **the
embedding already incorporates heading context** — there's no need
to also prepend headings to the chunk text before embedding.
`enrich_with_headings` is for the *stored* text only (see below).

The bundled embedders all satisfy the token-level contract late
chunking requires. Cloud APIs that only return one vector per input
do not.

### Heading-path enrichment (display only)

`enrich_with_headings(chunks)` returns each chunk with the Markdown
heading stack in scope at its start prepended (e.g.
`"# Top\n## Sub\n\n<chunk text>"`). The use case is **stored-text
breadcrumbs**: when a chunk gets surfaced back to the user from a
vector store, the prepended heading shows where in the document it
came from.

Apply it *after* embedding (the Quick start's building-blocks form
shows this) so the breadcrumb doesn't double up on the heading
context the vectors already carry. If you're not using late
chunking — e.g., you swapped in a BYO embedder that doesn't expose
per-token outputs — you can apply `enrich_with_headings` before
embedding instead, to push outline context into the vector that
way; that's the d-star.ai
[Out-of-Context Chunk Problem](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag)
trick. With late chunking that mode is redundant.

### Bring your own embedder

fancychunk ships four bundled embedders (see [Models](#models)).
If you need your own — different backend, custom model, remote
service — implement the protocol:

```python
class Embedder(Protocol):
    n_ctx: int
    def count_tokens(self, texts: list[str]) -> list[int]: ...
    def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray, list[int]]: ...
    def embed_chunklets(self, chunklets: list[str]) -> NDArray: ...
```

`embed_chunklets` is the pooled per-chunklet path for `split_chunks`;
`embed_segment` + `count_tokens` are the token-level path for
`embed_with_late_chunking`. A single class implements both — the
bundled embedders do.

Three runnable reference adapters in
[`examples/embedders/`](examples/embedders/): MLX + Qwen3-Embedding,
HuggingFace transformers, and a remote HTTP client. They currently
implement just the late-chunking half of the protocol; add a
`embed_chunklets` method (one batched forward pass over each
chunklet, pooled the same way the model was trained) to use them
with `split_chunks` or `chunk_document`.

## Models

fancychunk uses two kinds of model: a *sentence segmenter* (Stage 1)
and an *embedder* (Stage 3 + late chunking). Both are
**lazy-loaded on first use** — importing `fancychunk` itself is
cheap and triggers no network calls, and constructing an embedder
(e.g. `qwen3_600m()`) doesn't load the weights either. Weights
cache under `~/.cache/huggingface/` so the download happens once
per machine; subsequent process runs hit the cache.

**Sentence segmenter — SaT.** The default is `sat-3l-sm` from
[Segment Any Text](https://arxiv.org/abs/2406.16678) (Frohmann et
al., 2024) via `wtpsplit-lite`, shipped as ONNX. **408 MB** download
on first call, ~500 MB resident. Multilingual, punctuation-agnostic,
and exposes per-character boundary probabilities directly — exactly
the SPEC-CHUNK-106 contract Stage 1 wants. For zero-dependency
deployments where you can tolerate lower segmentation quality, pass
`segmenter=punctuation_segmenter` instead: a ~50-line rule-based
fallback bundled with the library.

**Embedders.** Four bundled models trade quality for latency. You
pick one explicitly — there's no hidden default — and pass it
through to `chunk_document` (or to the individual primitives). The
recommended choice for most workloads is **`qwen3_600m()`**: good
quality (MTEB Multilingual 64.33, the sub-1B leader), modest
memory (~0.5 GB on MLX-mxfp8, ~1 GB on torch), and fast enough to
keep interactive workflows responsive.

The MLX backend is auto-selected on Apple Silicon when
`mlx_embeddings` is installed (skipped via PEP 508 marker on
Linux/Windows); the factories transparently pick the
MLX-community build of each model. MTEB scores are from each
model's published tables; throughput is measured on this machine.

Apple Silicon, MLX path (M2 MacBook Air):

| Model factory | Backend default | Model | Params | Native dim | Resident | `embed_chunklets` mean | Tokens/s | MTEB-Multi | MTEB-Eng |
|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| `bge_m3()` | MLX¹ / torch | BGE-M3 (CLS pooling) | 568M | 1024 | ~1 GB | 139 ms | 890 | 59.50 | 63.50 |
| `qwen3_600m()` | MLX¹ / torch | Qwen3-Embedding-0.6B | 596M | 1024 | ~0.5 GB | 79 ms | 1,186 | **64.33** | **70.70** |
| `qwen3_4b()` | MLX¹ / torch | Qwen3-Embedding-4B | 3.6B | 2560 | ~4 GB | 516 ms | 182 | **69.45** | **74.60** |
| `qwen3_8b()` | MLX¹ / torch | Qwen3-Embedding-8B | 7.6B | 4096 | ~7 GB | 950 ms | 99 | **70.58** | **75.22** |

Linux, torch + CUDA path (RTX 3090)²:

| Model factory | Backend | `embed_chunklets` mean | Tokens/s |
|---|:---:|---:|---:|
| `bge_m3()` | torch | 18 ms | 6,843 |
| `qwen3_600m()` | torch | 32 ms | 2,974 |
| `qwen3_4b()` | torch | 39 ms | 2,426 |
| `qwen3_8b()` | torch | 44 ms | 2,162 |

`qwen3_4b` and `qwen3_8b` accept a `dim=N` argument to truncate via
Matryoshka Representation Learning and re-L2-normalize; the compute
cost is unchanged. Pass `dim=1024` to keep storage-pin-compatibility
with `qwen3_600m` and `bge_m3`.

¹ MLX builds: `mlx-community/bge-m3-mlx-fp16`,
`mlx-community/Qwen3-Embedding-{0.6B,4B,8B}-mxfp8`. The Qwen3
variants use 8-bit microscaling (mxfp8) — small enough to fit
comfortably on a 24 GB Mac at every tier and the highest-quality
MLX build the community publishes. On non-Apple-Silicon, each
factory transparently loads the canonical HuggingFace weights and
runs on torch + MPS / CUDA / CPU.

² CUDA numbers measured on an NVIDIA GeForce RTX 3090 (24 GB VRAM,
driver 580.159.03) with Intel Core i9-10900KF and 32 GB system RAM,
on Linux 6.17 with PyTorch 2.12.0 + bundled CUDA 13.0 wheels (Python
3.13). All factories load canonical HuggingFace weights in fp16;
weights live on VRAM. Same 3-chunklet `bench_factories.py` batch as
the Mac measurements.

A few things worth knowing:

- **MTEB-Multi deltas:** `qwen3_600m` beats `bge_m3` by ~5 points (a
  meaningful gap on multilingual retrieval); `qwen3_4b` beats
  `qwen3_600m` by another ~5 points at ~6× the latency; `qwen3_8b`
  adds another ~1 point on top.
- **Speed vs quality by backend:** on MLX, `qwen3_600m` at mxfp8
  actually outruns `bge_m3` at fp16 — Apple Silicon's the one place
  the decoder model wins on throughput too. On torch (MPS or CUDA)
  the encoder-vs-decoder architectural gap reasserts itself: BGE-M3
  is the throughput king, and on an RTX 3090 it's ~2.3× faster than
  Qwen3-0.6B. The spread across all four models compresses to ~3×
  on CUDA vs ~12× on MLX — a discrete GPU hides the per-pass
  overhead that dominates the small-batch MLX path.
- **`noop()`** is the fifth bundled embedder: zero downloads,
  returns constant per-chunklet vectors. Use it for a
  structural-only split via `split_chunks(chunklets, noop())`.
- **8B on a 24 GB Mac is tight.** `qwen3_8b()` runs at ~7 GB
  resident and shares unified memory with everything else; expect
  thermal throttling on sustained workloads on an Air. Stick with
  `qwen3_600m()` (the recommended default) unless you've measured
  the quality gain matters for your retrieval task.
- **None of the above:** implement the BYO protocol from
  [Bring your own embedder](#bring-your-own-embedder) above — three
  methods plus an attribute, ~50 lines of glue.

## Observability

Every public function emits an OpenTelemetry span — names like
`fancychunk.split_sentences`, attributes like
`fancychunk.sentences.count`. The library pulls only
`opentelemetry-api` so spans are no-ops until your app configures an
SDK. Useful for figuring out which stage just got slow in
production.

## Status

Alpha (`0.1.x`). Public API is documented in
[`docs/specs/contracts/public-api.md`](docs/specs/contracts/public-api.md)
and locked in by the test suite, but not yet SemVer-stable — that
lands at `1.0.0`. CI runs pyright strict + pytest on Python 3.12
and 3.13 on every push.

## Where the specs live

Behavioral specs in [`docs/specs/`](docs/specs/) describe *what*
each function does, not *how*. Every behavior has a SPEC-CHUNK-NNN
ID; every ID has a test. Implementations in other languages are
welcome to use the specs verbatim and ignore this Python code
entirely.

## TODO

- **Make heavy dependencies optional again.** Torch alone is ~1 GB
  and not everyone needs it — BYO-embedder users in particular pay
  the cost for nothing. Revisit splitting the bundled embedders
  back into an extra (probably one per backend: `[torch]`,
  `[mlx]`), with the base install carrying only the structural
  pipeline + `noop()`. Tracking issue TBD.
- **Reference adapters could implement `embed_chunklets`.** The
  three example adapters in `examples/embedders/` currently
  implement only the late-chunking half of the protocol (n_ctx +
  count_tokens + embed_segment). Adding a thin `embed_chunklets`
  pass would make them usable as `chunk_document` embedders.

## Acknowledgments

The three-stage pipeline (sentence → chunklet → chunk), the
late-chunking strategy, and the contextual-headings helper come from
[raglite](https://github.com/superlinear-ai/raglite). Specific
techniques: the [SaT](https://arxiv.org/abs/2406.16678) segmenter
(Frohmann et al., 2024), Greg Kamradt's
[5 Levels of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s),
Arora et al.'s
[discourse vector](https://openreview.net/forum?id=SyK00v5xx) (ICLR
2017), the Weaviate / Jina
[late-chunking work](https://arxiv.org/abs/2409.04701) (Günther et
al., 2024), and Dan Stites's
[contextual headings post](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag).
