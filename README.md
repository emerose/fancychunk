# fancychunk

> Markdown chunking for RAG that respects what your document actually
> *is* — sentence boundaries, headings, lists, and topic shifts —
> instead of cutting every N characters and hoping the cuts don't
> land mid-thought.

```bash
pip install fancychunk
```

## How it compares

Most chunkers split at character or token counts and bolt on a
recursive separator list to dodge the worst cuts. fancychunk treats
chunking as three separable problems, each solved by its own
optimization against its own signal:

|                                | char/token splitter | recursive (`\n\n`, `\n`, …) | sentence-N grouping | semantic chunking | **fancychunk** |
|--------------------------------|:--:|:--:|:--:|:--:|:--:|
| Never cuts mid-sentence        | ✗ | ~  | ✓ | ✓ | **✓** |
| Honors Markdown structure      | ✗ | ~  | ✗ | ✗ | **✓** |
| Detects topic shifts           | ✗ | ✗  | ✗ | ✓ | **✓** |
| Bounded chunk size guarantee   | ✓ | ✓  | ~ | ~ | **✓** |
| Multi-stage (sentence→group→topic) | ✗ | ✗ | partial | ✗ | **✓** |
| Contextual heading paths       | ✗ | ✗  | ✗ | ✗ | **✓** |
| Late chunking (context-aware embeddings) | ✗ | ✗ | ✗ | ✗ | **opt-in** |

Most importantly: fancychunk **doesn't pick one signal** and ride it
into the ground. Sentences come from punctuation + a learned
segmenter; chunklet groupings come from Markdown structure and a
"statement density" measure; final chunks come from embedding
similarity between adjacent groups. Each stage's output is the next
stage's input, so a bad call at any one level is contained.

## What it does

```
document  →  split_sentences  →  split_chunklets  →  split_chunks  →  chunks
              (punctuation +     (Markdown headings,    (cosine of
               SaT segmenter)     paragraphs, lists)     adjacent chunklets,
                                                         discourse-corrected)
```

Three stages, three signals, three DPs that each maximize a defined
cost function rather than guessing with heuristics. Every stage is
deterministic, fully spec'd, and exhaustively tested
([93 tests](tests/) against [normative test vectors](docs/specs/test-vectors/)).

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
embeddings = my_embedder(chunklets)       # any embedder you already have
chunks, _  = split_chunks(chunklets, embeddings, max_size=2048)
paths      = heading_paths(chunks)        # ["# Top\n## Sub\n", ...]
```

The first call lazily downloads
[SaT](https://arxiv.org/abs/2406.16678) (408 MB) for sentence
segmentation. Pre-warm in your image build, or pass
`segmenter=punctuation_segmenter` for a zero-dependency fallback.

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

## Observability

Every public function emits an OpenTelemetry span — names like
`fancychunk.split_sentences`, attributes like
`fancychunk.sentences.count`. The library pulls only
`opentelemetry-api` so spans are no-ops until your app configures an
SDK. Useful for figuring out which stage just got slow in
production.

## Performance

End-to-end on an M2 MacBook Air (punctuation segmenter, no model
inference):

| Document | Size | Pipeline |
|---|---:|---:|
| Blog post | 8.6 KB | **13 ms** |
| Long article | 12 KB | **15 ms** |
| Book chapter | 104 KB | **127 ms** |

Stage 1 is sliding-window-DP (O(N) amortized); stages 2 and 3 are
vectorized 1-D DPs. Throughput is roughly flat ~0.7 MB/s across
two orders of magnitude in document size.

## Status

Alpha (`0.1.x`). Public API is documented in
[`docs/specs/contracts/public-api.md`](docs/specs/contracts/public-api.md)
and locked in by the test suite, but not yet SemVer-stable — that
lands at `1.0.0`. CI runs pyright strict + pytest on Python 3.12
and 3.13 on every push.

## What it doesn't do

- Doesn't parse PDFs, Word, or HTML. Input is Markdown.
- Doesn't embed text. You bring an embedder (or skip embedding and
  use the three-stage pipeline with synthetic vectors, which is
  fine — Stage 3's structural awareness still helps).
- Doesn't index, retrieve, or generate. Output is `list[str]`.

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
