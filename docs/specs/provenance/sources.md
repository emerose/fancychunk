# Sources

These specs were extracted from the
[`raglite`](https://github.com/superlinear-ai/raglite) codebase, pinned
at commit:

> **`6a540e1bd10f80093316deb44e049c1308e9f7e7`** (on `main`)

## Files read

| Path | What was extracted |
|------|--------------------|
| `src/raglite/_split_sentences.py` | Stage 1 algorithm: per-character boundary probabilities, dynamic-programming optimization under length constraints, Markdown-heading boundary overrides, whitespace-trailing normalization. |
| `src/raglite/_split_chunklets.py` | Stage 2 algorithm: dynamic-programming partition of sentences with combined boundary + statement cost; per-sentence "statement count" piecewise function; Markdown-token boundary probability table. |
| `src/raglite/_split_chunks.py` | Stage 3 algorithm: discourse-vector correction, partition-similarity construction, heading-aware modification, integer programming covering constraint. |
| `src/raglite/_embed.py` | Late-chunking embed strategy: segment + preamble structure (golden-ratio split), token-to-sentence mapping, mean-pooling within content range. |
| `src/raglite/_insert.py` | The composition order of the three stages. |

## License note

raglite is licensed under MPL-2.0 (file-level copyleft). These specs do
not copy source code or substantial structure from raglite; they
describe externally observable behavior derived from reading the
source. The resulting `fancychunk` implementation is therefore not a
derivative work of raglite under MPL-2.0 and may be released under any
license the implementor chooses.

Where a spec preserves a *behavioral constant* (e.g., a threshold of
`0.25`, a golden-ratio split of `0.382`, a heading-level probability of
`1.0`), this is preservation of a behavioral specification, not
copyright of code.
