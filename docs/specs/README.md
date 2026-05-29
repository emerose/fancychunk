# fancychunk specs

Behavioral specifications for fancychunk's text-splitting pipeline.
These specs describe what each function must do, not how to do it.
The implementation is free to choose tools, algorithms, libraries,
and internal architecture.

## Glossary

| Term | Meaning |
|------|---------|
| **document** | Input Markdown string. UTF-8. |
| **sentence** | A contiguous substring of the document, the smallest unit fancychunk operates on. Sentences exhaustively partition the document; their concatenation reconstructs it byte-for-byte. |
| **chunklet** | A paragraph-sized contiguous group of sentences. Targets ≈ 3 statements of information content. The basic unit that gets embedded. |
| **chunk** | A contiguous group of chunklets representing one semantic unit. The unit a caller indexes for retrieval. |
| **statement** | A soft, document-relative measure of a sentence's information content. A sentence with the document's median word count is defined as containing one statement. |
| **boundary probability** | A real number in `[0, 1]` indicating how likely a position is to be a structural break. Comes from two sources: predicted (a sentence segmentation model) and known (Markdown structure). |
| **discourse vector** | The mean embedding of "typical" chunklets in a document, used to subtract a document's overall topic from chunklet embeddings so similarity reflects *local* topic shifts. |
| **partition similarity** | The cosine similarity between adjacent chunklets after discourse correction. Low values indicate a good place to split. |
| **late chunking** | An embedding strategy where each sentence's embedding is computed within the context of a longer surrounding document segment, then mean-pooled per sentence. Produces context-aware sentence embeddings. |
| **preamble** | In late chunking, the leading portion of an encoded segment that provides context but whose output embeddings are discarded. |

## Specification IDs

Each spec contains numbered behaviors of the form `SPEC-CHUNK-NNN`,
each describing a single testable property. The
[acceptance checklist](acceptance/checklist.md) tracks every ID.

| Range | Stage |
|-------|-------|
| SPEC-CHUNK-1xx | Sentence splitting |
| SPEC-CHUNK-2xx | Chunklet grouping |
| SPEC-CHUNK-3xx | Semantic chunking |
| SPEC-CHUNK-4xx | Late chunking |
| SPEC-CHUNK-5xx | Contextual chunk headings |
| SPEC-CHUNK-9xx | Cross-cutting (concatenation, determinism) |

Concrete input/output test vectors are referenced as `TV-NNN`.

## Reading order

1. [`00-pipeline-overview.md`](00-pipeline-overview.md) — what the
   three stages are and how they compose.
2. The four stage specs in order:
   [`01-sentence-splitting.md`](01-sentence-splitting.md),
   [`02-chunklet-grouping.md`](02-chunklet-grouping.md),
   [`03-semantic-chunking.md`](03-semantic-chunking.md),
   [`04-late-chunking.md`](04-late-chunking.md) (optional component),
   [`05-contextual-headings.md`](05-contextual-headings.md) (optional
   helper),
   [`06-structural-chunking.md`](06-structural-chunking.md) (the
   composition behind `chunk_document`).
3. [`contracts/public-api.md`](contracts/public-api.md) — the
   external surface of the library.
4. [`test-vectors/`](test-vectors/) — concrete input/expected-output
   pairs for each stage.
5. [`acceptance/checklist.md`](acceptance/checklist.md) — final
   conformance check.
