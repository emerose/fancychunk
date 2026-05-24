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

Optionally:

- When paired with a token-level embedding model, fancychunk can
  produce chunk embeddings that incorporate surrounding-document
  context ("late chunking").
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
├── pyproject.toml                # Package metadata (no deps yet)
├── docs/specs/
│   ├── README.md                 # Glossary and reading order
│   ├── 00-pipeline-overview.md   # End-to-end data flow
│   ├── 01-sentence-splitting.md  # Stage 1
│   ├── 02-chunklet-grouping.md   # Stage 2
│   ├── 03-semantic-chunking.md   # Stage 3
│   ├── 04-late-chunking.md       # Optional embed strategy
│   ├── 05-contextual-headings.md # Optional helper
│   ├── contracts/public-api.md   # Function signatures
│   ├── test-vectors/             # Concrete input → expected output pairs
│   └── acceptance/checklist.md   # Pass/fail criteria
├── src/fancychunk/               # (Empty stub — implementation TBD)
└── tests/                        # (Placeholder)
```

## Acknowledgments

fancychunk's three-stage pipeline (sentence → chunklet → chunk), the
late-chunking strategy, and the contextual-headings helper are
inspired by the chunking pipeline in
[raglite](https://github.com/superlinear-ai/raglite).

This repo is a **greenfield, clean-room rewrite**: the specs describe
externally observable behavior, name their own constants, and cite
the underlying ideas (the SaT segmenter, Greg Kamradt's "5 levels"
taxonomy, Arora et al.'s discourse-vector technique, the Weaviate /
Jina late-chunking work, Dan Stites's contextual-headings post) where
those ideas were first published. No code is copied from raglite;
the eventual implementation is released under the MIT license above.
