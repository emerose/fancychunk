# fancychunk

A small, focused library for splitting text documents into semantically
coherent chunks suitable for retrieval-augmented generation.

> **Status:** initial implementation. The full specification lives in
> [`docs/specs/`](docs/specs/README.md); the public API in
> [`docs/specs/contracts/public-api.md`](docs/specs/contracts/public-api.md);
> the test vectors in
> [`docs/specs/test-vectors/`](docs/specs/test-vectors/). The
> implementation lives in [`src/fancychunk/`](src/fancychunk/) and
> covers the three required pipeline stages plus the two optional
> helpers (`embed_with_late_chunking`, `heading_paths`).

## Quick start

```python
import numpy as np
from fancychunk import (
    split_sentences,
    split_chunklets,
    split_chunks,
    heading_paths,
)

doc = open("README.md").read()
sentences = split_sentences(doc, max_len=2048)
chunklets = split_chunklets(sentences, max_size=2048)

# Caller supplies the embedding matrix; embedding is not part of
# fancychunk's core pipeline. Any deterministic embedder works.
embeddings = my_embedder(chunklets)
chunks, chunk_embeddings = split_chunks(chunklets, embeddings, max_size=2048)
paths = heading_paths(chunks)
```

## Observability

Every public stage emits an OpenTelemetry span with attributes that
describe input/output sizes and the option choices that affected the
outcome. The library depends only on `opentelemetry-api`; spans are
zero-cost no-ops until the host application configures an SDK and
exporter.

Span names are `fancychunk.<function>` (e.g.
`fancychunk.split_sentences`). Attribute keys use the
`fancychunk.<key>` namespace:

| Stage | Attribute keys |
|---|---|
| `split_sentences` | `document.length`, `min_len`, `max_len`, `segmenter`, `sentences.count`, `short_circuit` |
| `split_chunklets` | `sentences.count`, `max_size`, `custom_costs`, `chunklets.count`, `short_circuit` |
| `split_chunks` | `chunklets.count`, `max_size`, `embedding.dim`, `chunks.count`, `short_circuit` |
| `embed_with_late_chunking` | `sentences.count`, `embedder`, `embedder.n_ctx`, `budget`, `preamble_budget`, `preamble_fraction`, `normalize`, `sentinel_method`, `segments.count`, `embedding.dim` |
| `heading_paths` | `chunks.count`, `paths.non_empty` |

To see them locally, install `opentelemetry-sdk` and configure a
console exporter:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# subsequent fancychunk calls now emit spans to stdout
```

The library also exposes a standard `logging.Logger` at
`fancychunk` (currently silent by default; future versions may add
INFO-level breadcrumbs at stage transitions).

## What it does

Given a Markdown document, fancychunk partitions it into chunks where
each chunk:

- Respects sentence and paragraph boundaries.
- Targets a configurable maximum size.
- Begins at a structurally meaningful point (heading, paragraph start).
- Groups together semantically related material, splitting where the
  topic shifts.

Optionally:

- When paired with a token-level embedding model, fancychunk can
  produce *per-sentence* embeddings that incorporate surrounding-
  document context ("late chunking"). The caller aggregates them to
  per-chunklet level (typically by mean-pool over the sentences in
  each chunklet) before passing them to the semantic-chunking stage.
- For each chunk, fancychunk can compute the Markdown heading path
  that was in scope at the chunk's start, suitable for prepending as
  embedding context.

## What it does *not* do

- It does not parse PDFs, Word documents, or HTML. Input is Markdown.
- It does not embed text in the core three-stage pipeline. Embedding
  is the caller's responsibility; fancychunk consumes pre-computed
  chunklet embeddings for the semantic-chunking stage. (The optional
  `embed_with_late_chunking` helper does invoke an embedder, but it
  is opt-in and requires the caller to supply one.)
- It does not store, index, or retrieve. Output is a list of strings.
- It does not generate. There is no LLM in the loop.

## How to read the specs

The specs in [`docs/specs/`](docs/specs/) are behavioral, not
prescriptive about implementation. A spec line says *what* a function
must do, not *how* to do it. Implementations are free to choose
tools, algorithms, libraries, and internal architecture.

Specs are numbered. SPEC-CHUNK-NNN identifiers within each spec
correspond to a single testable behavior; the
[acceptance checklist](docs/specs/acceptance/checklist.md) tracks every
ID.

## Repo layout

```
fancychunk/
├── README.md                     # This file
├── LICENSE                       # MIT
├── pyproject.toml                # Package metadata + runtime deps
├── docs/specs/
│   ├── README.md                 # Glossary and reading order
│   ├── 00-pipeline-overview.md   # End-to-end data flow
│   ├── 01-sentence-splitting.md  # Stage 1
│   ├── 02-chunklet-grouping.md   # Stage 2
│   ├── 03-semantic-chunking.md   # Stage 3
│   ├── 04-late-chunking.md       # Optional embed strategy
│   ├── 05-contextual-headings.md # Optional helper
│   ├── contracts/                # Public API signatures
│   ├── test-vectors/             # Concrete input → expected output pairs
│   └── acceptance/               # Pass/fail criteria
├── src/fancychunk/               # Implementation
│   ├── sentences.py              # Stage 1 — sentence splitting
│   ├── chunklets.py              # Stage 2 — chunklet grouping
│   ├── chunks.py                 # Stage 3 — semantic chunking
│   ├── late_chunking.py          # Stage 4 — late chunking (optional)
│   ├── headings.py               # Stage 5 — heading paths (optional)
│   ├── _markdown.py              # Markdown-it heading + opener helpers
│   ├── _segmenter.py             # SaT default + punctuation fallback
│   ├── _constants.py             # Named constants from the specs
│   └── errors.py                 # Exception hierarchy
└── tests/                        # pytest suite covering every TV-*
```

## Production readiness

This is an alpha release (`0.1.x`). The behaviour the public API
documents is fully spec-conforming and locked in by the 88-test
suite; what's *not* yet promised:

- **API stability.** Names and defaults are unlikely to change but
  aren't yet contract-stable. SemVer applies once the version hits
  `1.0.0`.
- **SaT model on first run.** The default segmenter downloads ~408 MB
  of weights from Hugging Face on first call. For production
  deployment, either pre-warm the cache during image build or pass
  `segmenter=punctuation_segmenter` if you can tolerate its quality.
- **Thread safety.** The module-level SaT singleton and markdown-it
  parser are reentrant for *read*; the library doesn't synchronise.
  Concurrent calls from multiple threads work because every operation
  reads-only. Concurrent first-time SaT loading from multiple threads
  may load the model twice (harmless but wasteful) — pre-warm if
  this matters.
- **No global state writes.** No caches, no temp files, no logging
  side effects. The library does not call `logging.basicConfig` and
  attaches no handlers.
- **Determinism.** Cross-run reproducibility is guaranteed for every
  stage given a deterministic segmenter / embedder (see
  SPEC-CHUNK-901 in the specs).

CI runs `pyright` in strict mode and `pytest` against Python 3.12
and 3.13 on every push. Tests use the lightweight punctuation
segmenter so CI doesn't need the SaT weights; set
`FANCYCHUNK_TEST_USE_SAT=1` to exercise the real model.

## Acknowledgments

The three-stage pipeline (sentence → chunklet → chunk), the
late-chunking strategy, and the contextual-headings helper are
inspired by the chunking pipeline in
[raglite](https://github.com/superlinear-ai/raglite). Specific
techniques cite their originators inline in the specs: the SaT
segmenter, Greg Kamradt's "5 levels" taxonomy, Arora et al.'s
discourse-vector technique, the Weaviate / Jina late-chunking work,
and Dan Stites's contextual-headings post.
