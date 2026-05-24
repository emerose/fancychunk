# fancychunk

A small, focused library for splitting text documents into semantically
coherent chunks suitable for retrieval-augmented generation. Specs only
at this stage — no implementation has been written.

> **Status:** spec phase. The full specification lives in
> [`docs/specs/`](docs/specs/README.md). The implementation team builds
> against those specs.

## What it does

Given a Markdown document, fancychunk partitions it into chunks where
each chunk:

- Respects sentence and paragraph boundaries.
- Targets a configurable maximum size.
- Begins at a structurally meaningful point (heading, paragraph start).
- Groups together semantically related material, splitting where the
  topic shifts.
- Carries a heading-path "preamble" describing where it sits in the
  document outline.

Optionally, when paired with a token-level embedding model, fancychunk
can produce chunk embeddings that incorporate surrounding-document
context ("late chunking").

## What it does *not* do

- It does not parse PDFs, Word documents, or HTML. Input is Markdown.
- It does not embed text by default. Embedding is the caller's
  responsibility; fancychunk consumes pre-computed chunklet embeddings
  for the semantic-chunking stage.
- It does not store, index, or retrieve. Output is a list of strings.
- It does not generate. There is no LLM in the loop.

## How to read the specs

The specs in [`docs/specs/`](docs/specs/) are behavioral, not
prescriptive about implementation. A spec line says *what* the function
must do, not *how* to do it. The implementor is free to choose tools,
algorithms, libraries, and internal architecture.

Specs are numbered. SPEC-CHUNK-NNN identifiers within each spec
correspond to a single testable behavior; the
[acceptance checklist](docs/specs/acceptance/checklist.md) tracks every
ID.

The specs were extracted from an upstream codebase; see
[provenance/sources.md](docs/specs/provenance/sources.md) for what
was read.

## Spec layout

```
docs/specs/
├── README.md                     # Methodology, glossary
├── 00-pipeline-overview.md       # End-to-end data flow
├── 01-sentence-splitting.md      # Stage 1
├── 02-chunklet-grouping.md       # Stage 2
├── 03-semantic-chunking.md       # Stage 3
├── 04-late-chunking.md           # Optional embed strategy
├── contracts/public-api.md       # Function signatures
├── test-vectors/                 # Concrete input → expected output pairs
├── acceptance/checklist.md       # Pass/fail criteria for an implementation
└── provenance/sources.md         # Upstream sources and licensing note
```
