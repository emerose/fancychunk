# Sources and Provenance

These specs were extracted from the [`raglite`](https://github.com/superlinear-ai/raglite)
codebase, pinned at commit:

> **`6a540e1bd10f80093316deb44e049c1308e9f7e7`** (on `main`)

All citations in the spec files (`<!-- cite: ... -->` comments) refer to
files at this SHA. To resolve any citation:

```
https://github.com/superlinear-ai/raglite/blob/6a540e1bd10f80093316deb44e049c1308e9f7e7/<path>
```

## Files analyzed

| Path | What the spec extracted from it |
|------|---------------------------------|
| `src/raglite/_split_sentences.py` | Stage 1 algorithm: per-character boundary probabilities, dynamic-programming optimization under length constraints, Markdown-heading boundary overrides, whitespace-trailing normalization. |
| `src/raglite/_split_chunklets.py` | Stage 2 algorithm: dynamic-programming partition of sentences with combined boundary + statement cost; per-sentence "statement count" piecewise function; Markdown-token boundary probability table. |
| `src/raglite/_split_chunks.py` | Stage 3 algorithm: discourse-vector correction, partition-similarity construction, heading-aware modification, integer programming covering constraint. |
| `src/raglite/_embed.py` | Late-chunking embed strategy: segment + preamble structure (golden-ratio split), token-to-sentence mapping, mean-pooling within content range. |
| `src/raglite/_insert.py` | The composition order of the three stages and the contextual heading carry-over. |

## Method

Each `.py` file above was read end-to-end. Behavior was extracted as
mathematical / behavioral statements; implementation choices (the
specific solver, array library, sentinel-token trick, etc.) were dropped.
Numeric constants, threshold values, Markdown token-type names, and
regex patterns are preserved verbatim because they constitute the
*behavior*, not the *implementation*.

The reimplementor must never see raglite's source. The test for every
sentence in these specs is the [reimplementor test](../README.md#the-reimplementor-test):
*"Can someone implement this from this spec alone, never having seen
the original code?"*

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
copyright of code. The constants are documented as part of the spec's
external contract; the reimplementor is free to adjust them with a
documented behavioral consequence.

## Citation format

Every behavioral claim in the specs ends with:

```
<!-- cite: source=source-code, ref=raglite@6a540e1:<file>:L<start>[-L<end>], confidence=<level>, agent=human -->
```

- `source`: always `source-code` (these specs draw exclusively from
  raglite source; no docs, runtime observation, or community sources
  were consulted).
- `ref`: file path + line range at the pinned SHA.
- `confidence`:
  - `confirmed` — explicit in source; obvious behavior.
  - `inferred` — clear from source but no comment / docstring confirms the *intent*.
  - `assumed` — reasonable interpretation; behavior could be a
    side effect of an unrelated choice. Always paired with an
    `[UNCERTAINTY: U-CHUNK-NNN]` tag elsewhere in the spec.
- `agent`: `human` for everything here (this was a manual extraction,
  not a Greenfield agent run).
