# Pipeline Overview

## Three stages

fancychunk partitions a Markdown document in three stages:

```
document (str)
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 1 — Sentence Splitting        │
│   Spec: 01-sentence-splitting.md    │
└─────────────────────────────────────┘
    │
    ▼
sentences (list[str])
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 2 — Chunklet Grouping         │
│   Spec: 02-chunklet-grouping.md     │
└─────────────────────────────────────┘
    │
    ▼
chunklets (list[str])
    │
    ▼
[ caller embeds each chunklet ]
    │
    ▼
chunklet embeddings (matrix)
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 3 — Semantic Chunking         │
│   Spec: 03-semantic-chunking.md     │
└─────────────────────────────────────┘
    │
    ▼
chunks (list[str]), chunk_embeddings (list[matrix])
```

The three stages exist because they answer different questions at
different scales: sentences answer "where could we split at all?",
chunklets answer "what are the paragraph-sized units?", and chunks
answer "what semantic units do we index?". Each stage's output is the
next stage's input, and the boundaries chosen at stage N constrain
stage N+1.

## Optional: Late chunking

Late chunking is an alternative embed strategy that replaces "caller
embeds each chunklet" with a token-level embedding pass over longer
document segments. It produces sentence-level embeddings that
incorporate surrounding-document context. Late-chunked sentence
embeddings can be aggregated to chunklet level for use in stage 3.

See [04-late-chunking.md](04-late-chunking.md).

## Cross-stage invariants

### SPEC-CHUNK-900 — Concatenation round-trip

For every stage, concatenating its outputs in order reproduces the
stage's input exactly, byte for byte. No characters are added,
removed, normalized, or re-ordered between stages.

Therefore: `"".join(sentences) == document`,
`"".join(chunklets) == "".join(sentences)`,
`"".join(chunks) == "".join(chunklets)`.

### SPEC-CHUNK-901 — Determinism

Given the same input and the same configuration, every stage produces
the same output across runs. Where a stage depends on a learned model
(stage 1's sentence segmenter; the caller's embedder for stage 3),
determinism is conditional on that model being deterministic.

### SPEC-CHUNK-902 — Size monotonicity

A unit produced by stage N is no larger than the size limit configured
for stage N. Specifically:
- Sentences respect their configured `max_len` (when supplied).
- Chunklets are no larger than `max_size` characters.
- Chunks are no larger than `max_size` characters.

A unit at stage N is the concatenation of one or more units from stage
N-1; the stage-N size limit is therefore an upper bound on the
combined size of the stage-(N-1) units inside it.

### SPEC-CHUNK-903 — Single-unit short-circuit

When a stage's input would produce only one unit (because the input
fits inside the stage's size limit, or because there are too few
upstream units to partition), the stage returns its input as a
single-element list with no optimization performed.

## Rationale for the three-stage structure

Each stage exploits a different signal:

- **Stage 1 (sentences)** uses *intra-sentence* signal: punctuation,
  capitalization, learned sentence-boundary cues from a model. It does
  not need to understand the document's topical structure.

- **Stage 2 (chunklets)** uses *document structure*: Markdown headings,
  paragraph breaks, list openings. It groups sentences into units big
  enough to embed meaningfully (≈ 3 statements) but small enough that
  each unit is topically homogeneous.

- **Stage 3 (chunks)** uses *semantic similarity* between adjacent
  chunklets. By this stage, every candidate split point is already
  structurally reasonable (it's between two chunklets), so the
  optimization can focus purely on topic shift.

Conflating the stages — e.g., doing semantic similarity at sentence
level — would be both more expensive (more embeddings, larger
optimization problem) and lower quality (single sentences have noisier
embeddings than paragraph-sized chunklets).

## Configuration surface

All three stages share a single notion of "maximum unit size" in
characters, defaulting to `2048`. This is the size limit applied at the
chunklet and chunk stage, and is also used as `max_len` for sentence
splitting in the upstream composition.

Implementations are free to expose stage-specific size limits if they
have a use case requiring different limits per stage. The default
behavior must be that all three stages share the same limit.
