# Public API Contract

External surface of fancychunk. These are the signatures and defaults
that callers depend on. Internals are unconstrained by this document;
only the contracts below cross the API boundary.

There are three required functions (`split_sentences`,
`split_chunklets`, `split_chunks`) and two optional helpers
(`embed_with_late_chunking`, `heading_paths`). The signatures are
written in Python type-hint syntax for clarity; any language is fine,
as the contract is the shape of the operation, not the syntax.

Implementations may rename, group, or wrap these operations (e.g., as
methods on a `Chunker` class). When they do, the implementation must
publish a mapping from its concrete names to the
[acceptance-checklist](../acceptance/checklist.md) SPEC-CHUNK IDs, so
that the test vectors and acceptance criteria can be applied without
guessing. The
[checklist](../acceptance/checklist.md) refers to operations by
behavior, not by function name.

## Function: split sentences

```python
def split_sentences(
    document: str,
    min_len: int = 4,
    max_len: int | None = None,
    known_boundary_probas: Vector | Callable[[str], Vector] | None = None,
) -> list[str]
```

Implements [spec 01](../01-sentence-splitting.md).

| Parameter | Default | Contract |
|-----------|---------|----------|
| `document` | — | A UTF-8 string. |
| `min_len` | `4` | Minimum sentence length in characters. |
| `max_len` | `None` | Optional maximum sentence length in characters. |
| `known_boundary_probas` | the Markdown-heading function (see SPEC-CHUNK-108) | Either a per-character probability vector or a callable that produces one. Finite values override the model; `NaN` defers to the model. Passing `None` selects the default. |

Returns a list of sentences satisfying SPEC-CHUNK-100 through
SPEC-CHUNK-104.

## Function: split chunklets

```python
def split_chunklets(
    sentences: list[str],
    max_size: int = 2048,
    boundary_cost: Callable[[Vector], float] | None = None,
    statement_cost: Callable[[float], float] | None = None,
) -> list[str]
```

Implements [spec 02](../02-chunklet-grouping.md).

| Parameter | Default | Contract |
|-----------|---------|----------|
| `sentences` | — | Ordered list of sentences. |
| `max_size` | `2048` | Hard upper bound on chunklet length in characters. |
| `boundary_cost` | the default of SPEC-CHUNK-220 | Optional override of the per-chunklet boundary cost. |
| `statement_cost` | the default of SPEC-CHUNK-221 | Optional override of the per-chunklet statement cost. |

Returns a list of chunklets satisfying SPEC-CHUNK-200 through
SPEC-CHUNK-202.

## Function: split chunks

```python
def split_chunks(
    chunklets: list[str],
    chunklet_embeddings: Matrix,
    max_size: int = 2048,
) -> tuple[list[str], list[Matrix]]
```

Implements [spec 03](../03-semantic-chunking.md).

| Parameter | Default | Contract |
|-----------|---------|----------|
| `chunklets` | — | Ordered list of chunklets. |
| `chunklet_embeddings` | — | Matrix `[N, D]`. One row per chunklet, in order. All rows must have nonzero L2 norm. |
| `max_size` | `2048` | Hard upper bound on chunk length in characters. |

Returns `(chunks, chunk_embeddings)` satisfying SPEC-CHUNK-300
through SPEC-CHUNK-302.

## Function: embed with late chunking (optional)

```python
def embed_with_late_chunking(
    sentences: list[str],
    embedder: TokenLevelEmbedder,
    max_tokens_per_segment: int | None = None,
    preamble_fraction: float = 0.382,
    normalize: bool = True,
) -> Matrix
```

Implements [spec 04](../04-late-chunking.md). Optional component;
implementations may omit it if they do not support late chunking.

| Parameter | Default | Contract |
|-----------|---------|----------|
| `sentences` | — | Ordered list of sentences. |
| `embedder` | — | An object satisfying the embedder contract in [spec 04 §Embedder contract](../04-late-chunking.md#embedder-contract). At minimum: methods `tokenize(text) → list[int]`, `detokenize(list[int]) → str`, `embed(text) → matrix[T, D]` (per-token, no internal pooling); and a property `n_ctx: int`. |
| `max_tokens_per_segment` | derived from embedder | Optional override of the per-segment token budget. |
| `preamble_fraction` | `0.382` | Fraction of the segment budget reserved for the preamble. |
| `normalize` | `True` | Whether to L2-normalize each output row. |

Returns a `[len(sentences), embedding_dim]` matrix satisfying
SPEC-CHUNK-400 through SPEC-CHUNK-402.

## Function: heading paths (optional)

```python
def heading_paths(chunks: list[str]) -> list[str]
```

Implements [spec 05](../05-contextual-headings.md). Optional helper;
implementations may omit it.

| Parameter | Default | Contract |
|-----------|---------|----------|
| `chunks` | — | Ordered list of chunks (typically the first element of `split_chunks`'s output). |

Returns a list of heading-path strings of length `len(chunks)`,
satisfying SPEC-CHUNK-500 through SPEC-CHUNK-502.

Common downstream use:

```python
chunks, _ = split_chunks(chunklets, embeddings)
paths = heading_paths(chunks)
texts_for_embedding = [
    (p + "\n" + c) if p else c
    for p, c in zip(paths, chunks)
]
```

## Wiring the stages

The pipeline has no top-level "do everything" function; callers
compose the three stages themselves. When doing so, pass
`max_len = max_size` to `split_sentences` so that no individual
sentence exceeds the downstream chunklet size limit (which would
trigger SPEC-CHUNK-263). `split_sentences`'s own default for
`max_len` is `None` because the function is also useful standalone.

Example:

```python
sentences = split_sentences(doc, max_len=2048)
chunklets = split_chunklets(sentences, max_size=2048)
embeddings = my_embedder(chunklets)
chunks, chunk_embeddings = split_chunks(chunklets, embeddings, max_size=2048)
```

## Error contract

All functions must signal errors via exceptions (in Python) or the
language-native error mechanism. Each error case in the specs
(SPEC-CHUNK-115, -263, -341, -342, -343, -451) must produce a
distinguishable signal, not a silent failure or a returned sentinel.

Validation errors (caller-fixable) and computation errors
(implementation-internal) should be distinguishable. A single shared
error base type is recommended.

## What this contract does NOT specify

- Module layout, file structure, class hierarchies.
- Whether the four operations are free functions, methods on a
  `Chunker` class, or a pipeline object.
- Async vs sync API style.
- Configuration loading (YAML, env vars, etc.).
- Logging, tracing, metrics.
- Caching of embedder calls, Markdown parses, etc.

Test vectors and the acceptance checklist verify behavior, not
architecture.
