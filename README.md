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
chunks reasonbly quickly. It uses markdown structure alongside
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
from fancychunk import (
    split_sentences,
    split_chunklets,
    split_chunks,
    heading_paths,
)

doc = open("my-document.md").read()

sentences  = split_sentences(doc, max_len=2048)
chunklets  = split_chunklets(sentences, max_size=2048)
chunks, _  = split_chunks(chunklets, max_size=2048)   # structural-only
paths      = heading_paths(chunks)                    # ["# Top\n## Sub\n", ...]
```

For semantic topic-shift splitting, supply a chunklet-embedding
matrix. Either BYO embedder, or use one of the bundled defaults
(`pip install 'fancychunk[embedders]'`):

```python
from fancychunk.embedders import fast

embedder    = fast()                                  # Qwen3-Embedding-0.6B
embeddings  = embedder.embed_chunklets(chunklets)
chunks, _   = split_chunks(chunklets, embeddings, max_size=2048)
```

Four factories, in order of increasing quality and decreasing
throughput: `fastest()` (BGE-M3) → `fast()` (Qwen3-0.6B,
recommended) → `medium(dim=1024)` (Qwen3-4B + MRL) →
`high(dim=1024)` (Qwen3-8B + MRL). On Apple Silicon they
automatically pick MLX builds when `mlx_embeddings` is installed.

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

After producing the chunks, fancychunk then enhances those chunks with two optional steps designed to enrich the
resulting embeddings with context from the surrounding document:

## Heading-path context (optional)

After chunking, prepend each chunk with the Markdown heading stack
that was in scope at its start. Embedders gain document-outline
context the chunk itself doesn't carry — the trick from Dan Stites's
[Out-of-Context Chunk Problem](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag).

```python
chunks, _ = split_chunks(chunklets, embeddings, max_size=2048)
paths     = heading_paths(chunks)
indexable = [(p + "\n" + c if p else c) for p, c in zip(paths, chunks)]
```

## Late chunking (optional)

Late chunking gives each chunk an embedding computed in the *context*
of its neighbours — anaphor references like "it" and "the algorithm"
pick up real referents instead of generic directions. Typical
retrieval-quality win is 4–8 MTEB points over naive per-chunk
embedding (Jina AI's paper has the numbers).

fancychunk doesn't host any embedding model. You write a thin
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

**Embedders.** The four bundled choices (`pip install
'fancychunk[embedders]'`) trade quality for latency, in order of
increasing parameter count. The MLX backend is auto-selected on
Apple Silicon when `mlx_embeddings` is installed (skipped via PEP
508 marker on Linux/Windows). Numbers measured on an M2 MacBook Air
running the MLX path; MTEB scores are from each model's published
tables.

| Factory | Backend default | Model | Params | Output dim | Resident | `embed_chunklets` mean | Tokens/s | MTEB-Multi | MTEB-Eng |
|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| `fastest()` | MLX¹ / torch | BGE-M3 (CLS pooling) | 568M | 1024 | ~1 GB | 139 ms | 890 | 59.50 | 63.50 |
| `fast()` | MLX¹ / torch | Qwen3-Embedding-0.6B | 596M | 1024 | ~0.5 GB | 79 ms | 1,186 | **64.33** | **70.70** |
| `medium(dim=1024)` | MLX¹ / torch | Qwen3-Embedding-4B + MRL | 3.6B | 1024 *(native 2560)* | ~4 GB | 516 ms | 182 | **69.45** | **74.60** |
| `high(dim=1024)` | MLX¹ / torch | Qwen3-Embedding-8B + MRL | 7.6B | 1024 *(native 4096)* | ~7 GB | 950 ms | 99 | **70.58** | **75.22** |

¹ MLX builds: `mlx-community/bge-m3-mlx-fp16`,
`mlx-community/Qwen3-Embedding-{0.6B,4B,8B}-mxfp8`. The Qwen3
variants use 8-bit microscaling (mxfp8) — small enough to fit
comfortably on a 24 GB Mac at every tier and the highest-quality
MLX build the community publishes. On non-Apple-Silicon, the same
factory functions transparently load the canonical HuggingFace
weights and run on torch + MPS / CUDA / CPU.

A few things worth knowing:

- **MTEB-Multi deltas:** `fast` beats `fastest` by ~5 points (a
  meaningful gap on multilingual retrieval); `medium` beats `fast`
  by another ~5 points at ~6× the latency; `high` adds another ~1
  point on top of `medium`.
- **Speed vs quality:** on the MLX path `fast` is the actual
  throughput winner — Qwen3-0.6B at mxfp8 runs faster than BGE-M3
  at fp16 on Apple Silicon. On torch + MPS the order flips and
  `fastest` lives up to the name.
- **Matryoshka:** `medium` and `high` truncate their native
  2560-dim and 4096-dim outputs respectively to `dim=1024` by
  default, so all four factories produce storage-pin-compatible
  vectors. Pass a larger `dim` to either to get the full native
  width (2560 / 4096) at the same compute cost.
- **8B on a 24 GB Mac is tight.** `high()` runs at ~7 GB resident
  and shares unified memory with everything else; expect thermal
  throttling on sustained workloads on an Air.
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
