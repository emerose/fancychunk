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

Late chunking is an alternative embed strategy that the caller plugs
in *between* stages 2 and 3 in place of a naive "embed each chunklet
in isolation" step. The late-chunking helper takes the sentences from
stage 1 and returns *per-sentence* embeddings computed in a token-
level pass over longer document segments. The caller then aggregates
the per-sentence embeddings to per-chunklet level (typically by
mean-pool over the sentences in each chunklet) before passing the
chunklet-embedding matrix into stage 3.

See [04-late-chunking.md](04-late-chunking.md).

> Status note: late chunking is experimental, not a recommended
> default. Downstream RAG benchmarking did not see it beat plain
> isolated-chunk embedding (and it regressed on long documents with
> causal, last-token-pooled models). The behavior specified here is
> unchanged; see the README "Late chunking (experimental)" section
> for the benchmark detail.

## Optional: Contextual chunk headings

A small helper that consumes stage 3's output and produces, for each
chunk, the Markdown heading path that was in scope at the chunk's
start. Prepending this path to the chunk's text before embedding
gives the embedder document-outline context that the chunk's own
content doesn't carry.

See [05-contextual-headings.md](05-contextual-headings.md).

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
the same output across runs. Stage 1's determinism is conditional on
its sentence-segmentation model being deterministic. Stage 3's
determinism is conditional on the precomputed chunklet embeddings
being deterministic — which in turn depends on the caller's embedder.
(Stage 3 itself takes a precomputed embedding matrix as input; it
does not invoke an embedder.)

### SPEC-CHUNK-902 — Size monotonicity

A unit produced by stage N is no larger than the size limit configured
for stage N, on the normal path:
- Sentences respect their configured `max_len` (when supplied),
  subject to the short-circuit exception in SPEC-CHUNK-114 (a
  document shorter than `min_len` is returned as a single sentence
  even if it exceeds `max_len`).
- Chunklets are no larger than `max_size` characters.
- Chunks are no larger than `max_size` characters (subject to the
  oversize-sentence carve-out in SPEC-CHUNK-411's progress
  guarantee, when late chunking is in use).

A unit at stage N is the concatenation of one or more units from stage
N-1; the stage-N size limit is therefore an upper bound on the
combined size of the stage-(N-1) units inside it.

### SPEC-CHUNK-903 — Trivial-input short-circuits

All three stages short-circuit on the trivial *size* cases (empty
input and single-item input). Beyond that they differ:

- **Stage 1:** empty document returns `[]` (SPEC-CHUNK-117); document
  no longer than `min_len` returns `[document]` (SPEC-CHUNK-114).
- **Stage 2:** empty input returns `[]` (SPEC-CHUNK-260); single
  sentence returns `[s]` (SPEC-CHUNK-261). Beyond those, stage 2
  does *not* short-circuit on the "fits in one chunklet" case — it
  always runs its optimization and may produce a multi-chunklet
  partition even when all sentences fit (SPEC-CHUNK-262). The size
  constraint is an upper bound, not a forcing function.
- **Stage 3:** empty input returns `([], [])`; at most one chunklet,
  or total length fits inside `max_size`, returns the input as a
  single chunk (SPEC-CHUNK-340).

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
characters, defaulting to `2048`. This is the `max_size` parameter
on the chunklet and chunk stage.

The pipeline has no top-level "do everything" function; callers wire
the three stages themselves. When doing so, the caller should pass
`max_len = max_size` to `split_sentences` so that no sentence exceeds
the downstream size limit, satisfying stage 2's precondition
(SPEC-CHUNK-263). `split_sentences`'s own default is `max_len = None`
because the function is also useful standalone.

Implementations are free to expose stage-specific size limits if they
have a use case requiring different limits per stage. The recommended
default is that all three stages share the same limit.
