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

The end-to-end RAG pipeline — semantic topic-shift chunking, late
chunking for context-aware embeddings, heading-path prepending, and a
single pooled vector per chunk ready for your vector store:

```python
from fancychunk import (
    split_sentences,
    split_chunklets,
    split_chunks,
    embed_with_late_chunking,
    enrich_with_headings,
    embedders,
)

doc = open("my-document.md").read()

# Stages 1-3 — produce chunks. split_chunks defaults to
# `embedders.default()`, the hardware-appropriate recommended
# embedder; override with any other factory, or pass
# `embedder=embedders.noop()` for a no-model-download
# structural-only split.
sentences = split_sentences(doc, max_len=2048)
chunklets = split_chunklets(sentences, max_size=2048)
chunks, _ = split_chunks(chunklets, max_size=2048)

# Late chunking — one context-aware embedding per chunk. The
# embedder sees adjacent chunks together so attention can resolve
# anaphora ("the algorithm" → real referent); the in-scope heading
# stack is prepended once per segment as additional outline context
# (controlled by `include_headings=True`, on by default). Reuses
# split_chunks's cached embedder singleton — same weights, one load.
vectors = embed_with_late_chunking(chunks, embedders.default())

# Heading-path enrichment — prepend each chunk's Markdown heading
# stack onto the *stored* text, as a retrieval-time breadcrumb. The
# vectors above already incorporate the heading context via late
# chunking; this step is for what gets displayed back to the user.
chunks = enrich_with_headings(chunks)

# chunks[i] ⇄ vectors[i] — drop straight into your vector store.
```

Five calls to go from a Markdown document to indexable vectors:
three pipeline stages, late chunking for the storage embeddings, and
the optional heading-path enrichment. A higher-level
`chunk_document(doc, embedder=default())` is still on the roadmap
to collapse this into one call, but the building blocks here are
the documented, test-locked API. The rest of this README walks
through what each piece does and what you can swap out.

`fancychunk.embedders` ships two parallel sets of factories. The
**tier-named** ones pick the best model for the current hardware:
`default()` (recommended) → `fastest()` (throughput king) → `fast()`
→ `medium()` → `high()`. The **model-named** ones — `bge_m3()`,
`qwen3_600m()`, `qwen3_4b(dim=...)`, `qwen3_8b(dim=...)` — pin a
specific model regardless of backend, for when reproducibility across
machines matters. On Apple Silicon both sets automatically pick the
MLX-mxfp8 builds when `mlx_embeddings` is installed. The Qwen models
return their native dimension by default (`qwen3_4b` → 2560,
`qwen3_8b` → 4096); pass `dim=N` for Matryoshka truncation.

For a no-model-download structural split, pass `embedder=noop()` to
`split_chunks`. `noop()` returns constant per-chunklet vectors, which
collapses the semantic-similarity term and leaves heading-aware
boundaries as the only signal — the same behavior the old "no
embeddings supplied" path produced, now reachable through the
embedder interface.

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

Two optional steps that push context from the surrounding document
into each chunk's output. Either can be dropped if you don't want it,
and they partly overlap — pick what fits your retrieval setup.

### Late chunking (recommended)

`embed_with_late_chunking(sentences, embedder)` gives each sentence
an embedding computed in the *context* of its neighbours — anaphor
references like "it" and "the algorithm" pick up real referents
instead of generic directions. Mean-pooling those per-sentence
embeddings within each final chunk yields a per-chunk vector that
carries inter-chunk context an in-isolation embedding would lose.
Typical retrieval-quality win is 4–8 MTEB points (Jina AI's paper
has the numbers). Every bundled embedder satisfies the token-level
contract late chunking requires; cloud APIs that only return one
vector per input do not.

### Heading-path enrichment (optional)

`enrich_with_headings(chunks)` returns each chunk with the Markdown
heading stack in scope at its start prepended (e.g.
`"# Top\n## Sub\n\n<chunk text>"`). Two reasons to do this:

- **Outline-context boost for the embedding** — apply
  `enrich_with_headings` *before* embedding, and the heading stack
  goes into the vector. Useful when each chunk would otherwise be
  embedded in isolation, since the embedder gains document-outline
  context the chunk's own text doesn't carry. The trick from Dan
  Stites's
  [Out-of-Context Chunk Problem](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag).
  With late chunking the nearest heading is already in the
  preamble window, so this use is largely redundant.
- **Breadcrumb for display** — apply *after* embedding (as in the
  Quick start) so the prepended heading shows up when the chunk is
  surfaced back to the user, without changing the vector.

fancychunk doesn't host any embedding model. The bundled factories
are an opinionated default; if you want your own, write a thin
adapter against this protocol:

```python
class SegmentEmbedder(Protocol):
    n_ctx: int
    def count_tokens(self, sentences: list[str]) -> list[int]: ...
    def embed_segment(
        self, sentences: list[str]
    ) -> tuple[NDArray, list[int]]: ...
```

Three runnable reference adapters in
[`examples/embedders/`](examples/embedders/): MLX + Qwen3-Embedding,
HuggingFace transformers, and a remote HTTP client. Each is ~50 lines
of glue.

## Models

fancychunk uses two kinds of model: a *sentence segmenter* (Stage 1)
and an *embedder* (Stage 3 + optional late chunking). Both are
**lazy-loaded on first use** — importing `fancychunk` itself is
cheap and triggers no network calls. Weights cache under
`~/.cache/huggingface/` so the download happens once per machine.

**Sentence segmenter — SaT.** The default is `sat-3l-sm` from
[Segment Any Text](https://arxiv.org/abs/2406.16678) (Frohmann et
al., 2024) via `wtpsplit-lite`, shipped as ONNX. **408 MB** download
on first call, ~500 MB resident. Multilingual, punctuation-agnostic,
and exposes per-character boundary probabilities directly — exactly
the SPEC-CHUNK-106 contract Stage 1 wants. For zero-dependency
deployments where you can tolerate lower segmentation quality, pass
`segmenter=punctuation_segmenter` instead: a ~50-line rule-based
fallback bundled with the library.

**Embedders.** Four bundled models trade quality for latency. Each
has a **model-named factory** that pins the model regardless of
backend, plus five **tier-named convenience factories** that pick a
hardware-appropriate model:

| Tier factory | Picks | Rationale |
|---|---|---|
| `default()` | `qwen3_600m()` on MLX, `qwen3_8b(dim=1024)` on torch | Recommended default. Mac stays interactive; CUDA gets near-leaderboard quality at ~44 ms per pass. Output is 1024-dim either way. |
| `fastest()` | `qwen3_600m()` on MLX, `bge_m3()` on torch (CUDA/MPS/CPU) | Throughput winner depends on backend — Qwen3-0.6B-mxfp8 leads on Apple Silicon; BGE-M3 leads on discrete GPUs. |
| `fast()` | `qwen3_600m()` everywhere | Best MTEB at sub-1B; only ~1.3× slower than `fastest()` on the worst backend. |
| `medium(dim=...)` | `qwen3_4b(dim=...)` everywhere | Alias. Defaults to native 2560-dim. |
| `high(dim=...)` | `qwen3_8b(dim=...)` everywhere | Alias. Defaults to native 4096-dim. |

The MLX backend is auto-selected on Apple Silicon when
`mlx_embeddings` is installed (skipped via PEP 508 marker on
Linux/Windows). MTEB scores are from each model's published tables;
throughput is measured on this machine.

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
cost is unchanged. Pass `dim=1024` (what `default()` does on CUDA) to
keep storage-pin-compatibility with the smaller models.

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
- **Speed vs quality:** on MLX, `qwen3_600m` at mxfp8 actually
  outruns `bge_m3` at fp16 — which is why `fastest()` picks Qwen3 on
  Apple Silicon. On torch (MPS or CUDA) the encoder-vs-decoder
  architectural gap reasserts itself: BGE-M3 is the throughput
  king, and on an RTX 3090 it's ~2.3× faster than Qwen3-0.6B. The
  spread across all four models compresses to ~3× on CUDA vs ~12×
  on MLX — a discrete GPU hides the per-pass overhead that dominates
  the small-batch MLX path.
- **`noop()`** is the fifth bundled embedder: zero downloads, returns
  constant per-chunklet vectors. Use it for a structural-only split
  via `split_chunks(chunklets, embedder=noop())`.
- **8B on a 24 GB Mac is tight.** `qwen3_8b()` runs at ~7 GB
  resident and shares unified memory with everything else; expect
  thermal throttling on sustained workloads on an Air. `default()`
  picks `qwen3_600m()` on MLX precisely to avoid this.
- **None of the above:** the BYO protocol is two methods and one
  attribute — see [Late chunking (optional)](#late-chunking-optional)
  and [`examples/embedders/`](examples/embedders/) for templates
  covering MLX, HuggingFace, and remote HTTP backends.

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
- **`chunk_document()` convenience.** Wrap the Quick start's
  remaining glue into a single entry point that returns
  `(chunks, vectors)` ready for a vector store. With factory
  caching in place, `chunk_document` can reuse one embedder across
  stage 3 and late chunking without the caller doing anything.

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
