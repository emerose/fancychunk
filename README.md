# fancychunk

> Markdown chunking for RAG that attempts to craft artisanal,
> meaningful chunks while remaining reasonably fast and efficient.

```bash
pip install 'fancychunk[torch]'     # most users: qwen3, bge_m3 via torch + transformers
pip install 'fancychunk[mlx]'       # macOS arm64: same models via Apple MLX (~2-4× faster)
pip install 'fancychunk[all]'       # both backends
pip install fancychunk              # no backend: structural-only chunking via noop()
```

The base install is ~180 MB; `[torch]` adds ~750 MB on CPU Linux,
~2.5 GB on CUDA Linux, ~80 MB on macOS; `[mlx]` adds ~40 MB on
Apple Silicon (no-op elsewhere). Pick the backend you need.

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

fancychunk is async-first — the entry points that touch an embedder
are `async def`. Sync callers wrap with `asyncio.run(...)`.

```python
import asyncio
from fancychunk import chunk_document
from fancychunk.embedders import qwen3_600m

async def main():
    embedder = qwen3_600m()                                 # probably the right pick
    chunks, vectors = await chunk_document(
        open("my-document.md").read(), embedder
    )
    # chunks[i].text is the chunk content; chunks[i].start / .end are
    # character offsets into the original document; vectors[i] is the
    # storage embedding for chunks[i]. Drop straight into your store.

asyncio.run(main())
```

Each chunk is a `Chunk` — a frozen dataclass with `text` (always
present) plus optional metadata:
- `start` / `end` — half-open character offsets, so
  `document[chunk.start:chunk.end] == chunk.text`.
- `heading_path` — tuple of full Markdown heading lines in scope at
  the chunk's start, e.g. `("# Top", "## **Bold** Sub")`. Marker
  count encodes level; inline formatting preserved. Useful for
  filter-by-section in your vector store, breadcrumb rendering, or
  attaching as metadata. `()` means no heading in scope.

More optional fields may be added over time without breaking
existing code.


### Building blocks

`chunk_document` is sugar over the four primitives. Compose them
directly when you want more control — different embedders per
stage, different `max_size` per stage, a structural-only split, or
storage-time heading breadcrumbs.

`split_sentences` and `split_chunklets` are sync (no await points);
`split_chunks` and `embed_with_late_chunking` are async (they call
the embedder):

```python
import asyncio
from fancychunk import (
    split_sentences,
    split_chunklets,
    split_chunks,
    embed_with_late_chunking,
    enrich_with_headings,
)
from fancychunk.embedders import qwen3_600m

async def main():
    embedder = qwen3_600m()
    doc = open("my-document.md").read()

    sentences = split_sentences(doc, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    chunks    = await split_chunks(chunklets, embedder, max_size=2048)
    vectors   = await embed_with_late_chunking(chunks, embedder)

asyncio.run(main())
```

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

### Late chunking

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


### Heading-path enrichment

`enrich_with_headings(chunks)` returns each chunk with the Markdown
heading stack in scope at its start prepended (e.g.
`"# Top\n## Sub\n\n<chunk text>"`). This is useful to add context
to chunks that might otherwise lack it; for more information, see
[Out-of-Context Chunk Problem](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag).
Note that the late chunking mode already includes this context;
only use this method if you're not using late chunking.


## Models

fancychunk uses two kinds of model: a *sentence segmenter* (Stage 1)
and an *embedder* (Stage 3 + late chunking). Both are
**lazy-loaded on first use** — importing `fancychunk` itself is
cheap and triggers no network calls, and constructing an embedder
(e.g. `qwen3_600m()`) doesn't load the weights either. Weights
cache under `~/.cache/huggingface/` so the download happens once
per machine; subsequent process runs hit the cache.

**Sentence segmenter — SaT.** The default is `sat-9l-sm` from
[Segment Any Text](https://arxiv.org/abs/2406.16678) (Frohmann et
al., 2024) via `wtpsplit-lite`, shipped as ONNX, run with
`weighting="hat"` inference (which de-weights low-context
sliding-window edges). Multilingual, punctuation-agnostic, and exposes
per-character boundary probabilities directly — exactly the
SPEC-CHUNK-106 contract Stage 1 wants. Three checkpoints are bundled as
factories in `fancychunk.segmenters`, trading speed for scientific-prose
quality (see `benchmarks/sat-model-selection.md`):

```python
from fancychunk import segmenters
split_sentences(doc, segmenter=segmenters.sat_3l())   # fastest; mis-splits "Tab. TABREF21", "SemEval-2014 Task"
split_sentences(doc, segmenter=segmenters.sat_9l())   # default: artifact-free, ~1.3× faster than 12l
split_sentences(doc, segmenter=segmenters.sat_12l())  # highest quality, slowest
split_sentences(doc, segmenter=segmenters.punctuation())  # ~50-line rule-based fallback, no download
```

For corpora of many short documents (think BeIR scifact: ~5K
abstracts at ~1.5K chars each), SaT can become the dominant cost
in the pipeline. **Install `onnxruntime-gpu` on a CUDA box and the
defaults do the right thing** — no code changes:

```python
from fancychunk import chunk_documents
from fancychunk.embedders import qwen3_600m

# Picks CUDAExecutionProvider automatically if onnxruntime-gpu is
# installed (else falls back to CPU). Batches the SaT forward
# passes when running on a GPU; skips batching on CPU (where it
# wouldn't help).
await chunk_documents(docs, qwen3_600m())
```

Under the hood:
* `SaTSegmenter()` defaults to `device="auto"`, which defers to
  wtpsplit-lite's GPU-first provider auto-detect.
* `chunk_documents(..., segmenter_batch_size="auto")` (the default)
  asks the resolved segmenter whether batching will help via
  `wants_batching()`. The bundled `SaTSegmenter` says yes on any
  GPU EP, no on CPU.
* Power-user overrides still available: pass an explicit
  `SaTSegmenter(device="cuda"/"cpu")`, set
  `segmenter_batch_size=None` to force per-doc segmentation, or
  pass an int to force a specific batch size.

Verify the win on your hardware with `python bench_sat_batching.py
--device cuda --n-docs 1000`. Measured on a 1,000-doc / 1,500-char
corpus (RTX 3090, sat-3l-sm, `embedders.noop()`):

| `chunk_documents` config | ms/doc | vs CPU baseline |
|---|---:|---:|
| CPU, no batch (baseline) | 33.27 | 1.00× |
| CUDA, no batch | 6.77 | **4.91×** |
| **CUDA, default (`"auto"` → batch=64)** | **5.06** | **6.57×** |
| CUDA, `segmenter_batch_size=128` | 5.05 | 6.58× |
| CPU, `segmenter_batch_size=64` | 58.55 | 0.57× (slower — auto skips this) |

The headline number is the e2e CUDA win. The SaT-only batched-vs-
serial ratio on the same GPU is ~2.2× (raw segmenter throughput
goes from ~1.45 ms/doc serial to ~0.67 ms/doc batched). About half
the post-`device="cuda"` wall is the actual ORT forward pass; the
rest sat in a per-document Python loop in `wtpsplit-lite`'s
`token_to_char_probs`, which `SaTSegmenter` now monkey-patches with
a vectorised replacement on first load. Set
`FANCYCHUNK_DISABLE_SAT_FAST_POSTPROCESS=1` to opt out.

**Embedders.** Four bundled models trade quality for latency. You
pick one explicitly — there's no hidden default — and pass it
through to `chunk_document` (or to the individual primitives). The
recommended choice for most workloads is **`qwen3_600m()`**: good
quality (MTEB Multilingual 64.33, the sub-1B leader), modest
memory (~0.5 GB on MLX-mxfp8, ~1 GB on torch), and fast enough to
keep interactive workflows responsive.

The factories live in `fancychunk.embedders` and require one of
the install extras above (`[torch]` or `[mlx]`). Calling
`qwen3_600m()` without the right backend installed raises an
`ImportError` with the install hint baked in. The MLX backend is
auto-selected on Apple Silicon when `mlx_embeddings` is importable;
elsewhere the factories fall back to torch (which requires
`[torch]`). MTEB scores are from each model's published tables;
throughput is measured on this machine.

> **Note on CPU-only torch (Linux).** `pip install 'fancychunk[torch]'`
> pulls the default torch wheel, which on Linux is the CUDA-bundled
> build (~2.5 GB) even if you don't have a GPU. If you only need CPU
> inference, install the CPU wheel first, then add fancychunk:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install 'fancychunk[torch]'  # picks up the already-installed torch
> ```
>
> PyPI extras can't express the `--index-url` redirect, so this
> two-step is the workaround until upstream torch ships size-tagged
> variants on standard PyPI. macOS torch wheels are already small
> (~80 MB, no CUDA bundle) — this is a Linux-only concern.

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


### Bring your own embedder

fancychunk ships four bundled embedders (see [Models](#models)).
If you need your own — different backend, custom model, remote
service — implement the protocol. All three methods are **async**:

```python
class Embedder(Protocol):
    n_ctx: int
    async def count_tokens(self, texts: list[str]) -> list[int]: ...
    async def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray, list[int]]: ...
    async def embed_chunklets(self, chunklets: list[str]) -> NDArray: ...
```

`embed_chunklets` is the pooled per-chunklet path for `split_chunks`;
`embed_segment` + `count_tokens` are the token-level path for
`embed_with_late_chunking`. A single class implements both — the
bundled embedders do.

For a CPU/GPU embedder (torch, MLX, etc.) wrap your sync forward
pass in `asyncio.to_thread` inside each async method so the call
yields control while the device works; for a remote embedder, await
your HTTP client directly. The bundled `PooledSegmentEmbedder`
shows the former; `examples/embedders/remote_http.py` shows the
latter against `httpx.AsyncClient`.

Three runnable reference adapters in
[`examples/embedders/`](examples/embedders/): MLX + Qwen3-Embedding,
HuggingFace transformers, and an async-HTTP remote client. All
three now implement both halves of the protocol so they're
drop-in for `split_chunks` and `chunk_document`.


## Concurrency

The public async API (`split_chunks`, `embed_with_late_chunking`,
`chunk_document`) is safe to drive from multiple coroutines
concurrently — `asyncio.gather(chunk_document(doc1, emb),
chunk_document(doc2, emb), ...)` works. Inside
`embed_with_late_chunking`, independent segments are themselves
embedded via `asyncio.gather`, so each document overlaps its own
segments' embedding calls.

For a batch of documents, `chunk_documents` wraps that gather with
an optional concurrency cap:

```python
import asyncio
from fancychunk import chunk_documents
from fancychunk.embedders import qwen3_600m

async def main():
    embedder = qwen3_600m()
    docs = [open(p).read() for p in paths]
    results = await chunk_documents(docs, embedder, max_concurrency=8)
    # results[i] is (chunks, vectors) for docs[i].

asyncio.run(main())
```

Pass `max_concurrency=N` to cap fan-in (sensible for remote
embedders so you don't hammer the server). Omit it to gather all
documents at once — fine for bundled embedders since the internal
lock serializes to device throughput anyway.

Bundled embedder instances are also safe to drive from multiple
threads — internal locking serializes worker-thread access to the
underlying model. This covers any code that uses the embedder via
`asyncio.to_thread` or a `ThreadPoolExecutor` directly. The lock
matches what the device can actually deliver — one forward pass at
a time — so callers don't need their own synchronization.

For higher throughput, create multiple embedder instances; each
loads its own copy of the weights. A remote / true-parallel embedder
(`examples/embedders/remote_http.py`) gets real concurrency from
`asyncio.gather` since it isn't bottlenecked on a single local
device.

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
