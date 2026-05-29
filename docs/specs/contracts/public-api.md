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
    embedder: ChunkletEmbedder,
    max_size: int = 2048,
) -> list[str]
```

Implements [spec 03](../03-semantic-chunking.md).

| Parameter | Default | Contract |
|-----------|---------|----------|
| `chunklets` | — | Ordered list of chunklets. |
| `embedder` | — | Caller-supplied object exposing `embed_chunklets(chunklets) -> Matrix[N, D]`. Each returned row must have nonzero L2 norm. Pass a constant-output embedder (e.g. `fancychunk.embedders.noop()`) for a no-model-download structural-only split. The embedder is invoked only on the multi-chunk partition path; trivial-input short-circuits (SPEC-CHUNK-340) skip it but the argument remains required for signature consistency. |
| `max_size` | `2048` | Hard upper bound on chunk length in characters. |

Returns the list of chunks satisfying SPEC-CHUNK-300 and
SPEC-CHUNK-301. The embedder's output drives the partition decision
internally but is not returned; per-chunk storage vectors come from
`embed_with_late_chunking(chunks, embedder)` ([spec 04](../04-late-chunking.md)).

## Function: embed with late chunking (optional)

```python
def embed_with_late_chunking(
    chunks: list[str],
    embedder: SegmentEmbedder,
    max_tokens_per_segment: int | None = None,
    preamble_fraction: float = 0.382,
    normalize: bool = True,
    include_headings: bool = True,
) -> Matrix
```

Implements [spec 04](../04-late-chunking.md). Optional component;
implementations may omit it if they do not support late chunking.

| Parameter | Default | Contract |
|-----------|---------|----------|
| `chunks` | — | Ordered list of chunks (typically the first element of `split_chunks`'s output, before any `enrich_with_headings` post-processing). |
| `embedder` | — | An object satisfying the SegmentEmbedder contract in [spec 04 §Embedder contract](../04-late-chunking.md#embedder-contract): `n_ctx: int`, `count_tokens(texts) → list[int]`, and `embed_segment(texts) → (matrix[T, D], list[int])`. The `texts` parameter is generic — the library passes chunks plus (optionally) one heading-stack prepend. See `examples/embedders/` for reference adapters over MLX, HuggingFace transformers, and remote HTTP services. |
| `max_tokens_per_segment` | derived from embedder | Optional override of the per-segment token budget. |
| `preamble_fraction` | `0.382` | Fraction of the segment budget reserved for the preamble (heading prepend + backward-walk context). |
| `normalize` | `True` | Whether to L2-normalize each output row. |
| `include_headings` | `True` | Whether to prepend the in-scope Markdown heading stack to each segment's preamble (SPEC-CHUNK-470). |

Returns a `[len(chunks), embedding_dim]` matrix satisfying
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
chunks = split_chunks(chunklets)
paths = heading_paths(chunks)
texts_for_embedding = [
    (p + "\n" + c) if p else c
    for p, c in zip(paths, chunks)
]
```

## Function: chunk document (optional)

```python
def chunk_document(
    document: str,
    embedder: Embedder,
    max_size: int = 2048,
    min_size: int | None = None,
) -> tuple[list[str], Matrix]
```

Optional convenience that composes the required operations via
**structure-first chunking** ([spec 06](../06-structural-chunking.md))
plus late chunking. Implementations may omit it.

| Parameter | Default | Contract |
|-----------|---------|----------|
| `document` | — | UTF-8 string. |
| `embedder` | — | An object satisfying both the `ChunkletEmbedder` (`embed_chunklets`) and `SegmentEmbedder` (`n_ctx`, `count_tokens`, `embed_segment`) contracts. The same instance is used for the fallback partition decision and for late chunking, so model weights load exactly once. |
| `max_size` | `2048` | Hard upper bound on chunk length in characters (SPEC-CHUNK-601). On the fallback path it is passed as `max_len` to `split_sentences` and as `max_size` to `split_chunklets` / `split_chunks`. |
| `min_size` | `0.35 × max_size` | Chunk-size floor below which a structural unit is merged into a neighbor (SPEC-CHUNK-631). `0` disables merging. |

Returns `(chunks, vectors)` where:
- `chunks` is the structure-first partition (satisfying SPEC-CHUNK-600,
  -601). Sections that fit `max_size` are emitted whole with no model
  call; only overflowing sections invoke the embedder/segmenter.
- `vectors` is `embed_with_late_chunking(chunks, embedder)` — one
  L2-normalized context-aware embedding per chunk, with the heading
  stack prepended once per segment (SPEC-CHUNK-470, default on).

For storage-time heading-breadcrumb decoration, apply
`enrich_with_headings(chunks)` to the returned chunks. That step
does **not** affect `vectors` — late chunking already saw the
in-document headings via SPEC-CHUNK-470.

## Wiring the stages

`chunk_document` is the one-call entry point. For finer control —
different embedders per stage, different `max_size` per stage, or a
structural-only split via `embedders.noop()` — compose the
underlying stages yourself. When doing so, pass
`max_len = max_size` to `split_sentences` so that no individual
sentence exceeds the downstream chunklet size limit (which would
trigger SPEC-CHUNK-263). `split_sentences`'s own default for
`max_len` is `None` because the function is also useful standalone.

Example (manual composition):

```python
sentences = split_sentences(doc, max_len=2048)
chunklets = split_chunklets(sentences, max_size=2048)
chunks = split_chunks(chunklets, embedder, max_size=2048)
vectors = embed_with_late_chunking(chunks, embedder)
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
