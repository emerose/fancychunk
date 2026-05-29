# Spec 06 — Structural Chunking

Partition a Markdown document into *chunks* by honoring its heading
structure first, falling back to the semantic pipeline (specs 01-03)
only where a section is too large to emit whole. This is the strategy
behind the `chunk_document` convenience entry point.

A heading-delimited section that already fits `max_size` becomes one
chunk directly — with **no** sentence-segmentation model and **no**
embedder call. Only a section that overflows `max_size` is handed to
`split_sentences → split_chunklets → split_chunks` on that span alone.
Because each candidate boundary is already a structural one (a heading
or a fallback split inside an oversized section), headings land at
chunk starts rather than mid-chunk.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `document` | string | yes | — | A UTF-8 Markdown document. |
| `embedder` | `ChunkletEmbedder` | yes | — | Used **only** on the fallback path (an overflowing section). A constant-output embedder (`fancychunk.embedders.noop()`) yields a structural-only split. |
| `max_size` | positive integer | no | `DEFAULT_MAX_SIZE_CHARS` (`= 2048`) | Hard upper bound on chunk length in characters. |
| `min_size` | non-negative integer or unset | no | `0.35 × max_size` | Chunk-size floor; units below it are merged into a neighbor (SPEC-CHUNK-630). `0` disables merging. |
| `segmenter` | sentence segmenter or unset | no | the process SaT singleton | Used **only** on the fallback path. |

## Outputs

A list of chunks. Each chunk is a contiguous span of the document.

Invariants (shared with the rest of the pipeline):

- **SPEC-CHUNK-600** — Round-trip: `"".join(chunks) == document`
  (consistent with SPEC-CHUNK-300/900).
- **SPEC-CHUNK-601** — Covering: every chunk is at most `max_size`
  characters, subject to the same oversize carve-out the fallback
  pipeline allows (SPEC-CHUNK-301/411).

## Behavior

### SPEC-CHUNK-610 — Heading level

The level of a heading line is `(count of leading "#") + (count of
" ::: " separators in the heading text)`. The `#` count is the
Markdown level; the `:::` count recovers hierarchy that has been
flattened into the heading text as a path
(`## Methodology ::: Sub ::: Step`). A line that is not a heading has
level 0.

Heading detection uses the same line-anchored scan as the rest of the
library (it is not fence-aware): a `#`-prefixed line inside a fenced
code block is treated as a heading.

### SPEC-CHUNK-620 — Segments and the subtree-fit rule

The document is split into heading-delimited *segments* (a heading line
plus the prose that follows it, up to the next heading). Any text
before the first heading is a *preamble* segment. The segments form an
implied forest by level.

Walking the forest top-down, for each node:

- If the node's **entire subtree** fits within `max_size`, emit it as
  one chunk directly — no models. The heading therefore leads the
  chunk.
- Otherwise emit the node's own heading+body (the prose before its
  first child) as a unit — directly if it fits, or via the fallback
  split if even that overflows — then recurse into the children.

The preamble is emitted directly when it fits, or via the fallback
split when it overflows.

### SPEC-CHUNK-621 — Fallback split

A unit that overflows `max_size` is split by running
`split_sentences → split_chunklets → split_chunks` on that span alone,
then rebasing the resulting offsets back into the full document. This
is the *only* place the slow models run, so structural chunking can
only ever introduce *fewer* internal split points than running the
semantic pipeline over the whole document — never a mid-section split
the whole-document pipeline would not also make.

### SPEC-CHUNK-630 — Bare-heading merge

A *bare* heading unit — heading line(s) plus whitespace, with no body
of its own (e.g. a `# Title` container before `## Abstract`) — is
merged **forward** into the following unit so a lone heading is never
stranded at a chunk tail. If gluing forward would exceed `max_size`,
the combined span is routed through the fallback split so covering
still holds; a trailing bare heading with no following unit is glued
**backward** when it fits.

### SPEC-CHUNK-631 — Minimum-size merge

A unit shorter than `min_size` absorbs the following unit(s) until it
clears the floor, as long as the combined span stays within
`max_size`. A thin unit that cannot grow forward (the next would
overflow) glues backward into its predecessor when that fits. The
merge fires *only* to clear the floor — it stops the moment a unit
reaches `min_size`, so distinct sections are never packed up to the
cap. Small chunks above the floor are kept as-is; the floor only
suppresses thin, fragmented stubs.

The merge preserves covering (units are a contiguous tiling and a merge
only joins adjacent spans). A genuine leftover tail (the last unit, or
a section wedged against an oversized neighbor it cannot legally merge
into) may remain below the floor.

### SPEC-CHUNK-632 — Parent-intro fold

When an overflowing parent section emits its own heading + lead-in prose
(the text before its first child subheading) as a unit before recursing
into its children (SPEC-CHUNK-620), and that intro is shorter than
`min_size`, it is folded **forward** into its first child: the first
child's unit is extended back to the intro's start so the two are
emitted — or semantically split — together.

This is strictly a parent → own-first-child fold. It never reaches
across into an unrelated sibling section (which would mix distinct
topics, e.g. gluing an abstract into the introduction). It exists
because the minimum-size merge (SPEC-CHUNK-631) cannot rescue this case:
when the first child is itself oversized, the combined intro+child span
exceeds `max_size`, so the floor merge cannot absorb the intro forward,
and gluing it backward into the previous sibling is topically wrong. The
intro is the natural lead-in to the first subsection and shares the
parent's heading context, so folding it forward dissolves the strand at
its source.

When the folded span overflows `max_size` it goes to the fallback split
(SPEC-CHUNK-621). The leading chunk's small-chunk badness
(SPEC-CHUNK-323) outweighs the heading-aware split-before discount at
the child heading, so the splitter carries the intro into the first
child's first chunk rather than re-severing it.

The fold runs after the bare-heading merge (SPEC-CHUNK-630) and before
the minimum-size merge (SPEC-CHUNK-631): a *bare* parent heading (no
lead-in prose) is already merged forward by SPEC-CHUNK-630 and so is not
handled here.

## Rationale

For a well-sectioned document, most text lives in sections that already
fit `max_size`, so the slow models run on only a fraction of the
corpus — the latency win. And because the section is the primary unit,
a heading lands at a chunk start instead of mid-chunk, which is the
boundary-quality win. Documents dominated by a few huge sections see
little latency benefit (the fallback still pays full segmentation
cost), but the boundary behavior is unchanged from the semantic
pipeline because the fallback *is* that pipeline.
