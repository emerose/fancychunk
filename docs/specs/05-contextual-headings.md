# Spec 05 — Contextual Chunk Headings

Given a sequence of chunks from stage 3, compute the **heading path**
that applies to each chunk — the cumulative Markdown heading stack
that was in scope at the chunk's starting position. The path is
intended for use as embedding context: prepending it to a chunk's
text before embedding gives the embedder the surrounding-document
context that the chunk itself doesn't carry.

A chunk like:

> "This approach handles the rare case where the pivot is at the
> boundary..."

loses most of its meaning when embedded in isolation. With its
heading path prepended:

> "# Quicksort with random pivot selection
> ## Edge cases
>
> This approach handles the rare case where the pivot is at the
> boundary..."

the embedder can place the chunk's content semantically near other
material about Quicksort, pivot selection, and edge cases — rather
than near generic "rare case" content.

The technique is described in Dan Stites,
[*Solving the Out-of-Context Chunk Problem for RAG*](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag).

This is an optional helper: chunks produced by stage 3 are
self-contained as text, and callers that don't need additional
embedding context may skip this stage entirely.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `chunks` | list of strings | yes | — | An ordered sequence of chunks. Typically the first element of stage 3's output. |

## Outputs

A list of strings of length `len(chunks)`. The `i`-th string is the
heading path that applies to `chunks[i]`.

- **SPEC-CHUNK-500** — `len(output) == len(chunks)`.
- **SPEC-CHUNK-501** — Each path is either the empty string `""` or a
  sequence of one to `MAX_HEADING_LEVELS` Markdown heading lines
  joined by `HEADING_PATH_SEPARATOR`, with the deepest level last.
- **SPEC-CHUNK-502** — The first chunk's path is always `""`; no
  headings precede the document's start.

## Behavior

### SPEC-CHUNK-510 — Heading stack model

Maintain a stack of `MAX_HEADING_LEVELS` slots, one per Markdown
heading level (h1 through h6). All slots are initially empty.

A *heading line* is a contiguous substring of a chunk that matches
the Markdown heading syntax `^#+\s` at the start of a line, up to and
including that line's terminating newline (or end of chunk). The
heading's *level* is the number of `#` characters before the
whitespace.

When a heading at level `N` is encountered:
- Set slot `N` to that heading's line.
- Clear slots `N+1` through `MAX_HEADING_LEVELS`.

The *heading path* at any point in the document is the concatenation
of the non-empty slots from level 1 upward, joined by
`HEADING_PATH_SEPARATOR`.

### SPEC-CHUNK-511 — Per-chunk computation

Walk the chunks in order, maintaining the heading stack across
chunks. For each chunk `c[i]`:

1. **Snapshot the current stack** as `c[i]`'s heading path. This is
   the state of the stack at the chunk's starting position — the
   chunk's own headings are not included in its path.

2. **Scan `c[i]`** for heading lines in document order; for each one,
   update the stack per SPEC-CHUNK-510.

3. The updated stack becomes the starting state for `c[i+1]`.

Snapshotting *before* scanning means a chunk that starts with a
heading does not include that heading in its own path (the heading
text is already in the chunk's content). The path provides context
that is *missing* from the chunk, not context already present.

### SPEC-CHUNK-512 — Heading detection

A heading line is detected by matching `^#+\s` at the start of a line
(after any leading newline). The match captures:

- The heading marker (`#`, `##`, `###`, `####`, `#####`, or `######`).
- A required whitespace character.
- The heading text up to the line's terminating newline.

Headings with more than 6 `#` characters are not standard Markdown
and are treated as non-headings (paragraph content).

Setext-style headings (a line of `=` or `-` characters underlining a
heading text) are *not* recognized by this spec. Stage 1's Markdown
parser handles Setext headings at the sentence-splitting level, but
by stage 3 those have been resolved into chunklet content; this
spec's heading detection is purely the ATX-style `#+\s` form.

### SPEC-CHUNK-513 — Path formatting

A heading path is the non-empty stack slots, in level order, joined
by `HEADING_PATH_SEPARATOR = "\n"`. Each slot already contains its
trailing newline (if the heading line had one); the separator
provides the join between slots.

The empty path is the empty string `""`, not `None` or a placeholder.

Example: with the stack `[h1="# Sorting\n", h2="## Quicksort\n",
h3="### Random pivots\n", h4=None, h5=None, h6=None]`, the heading
path is the literal string:

```
# Sorting

## Quicksort

### Random pivots

```

Each stored heading line already ends in `\n`; the `\n` separator
then introduces the visible blank line between successive headings.
The path ends with the trailing `\n` from the deepest heading line.

### SPEC-CHUNK-520 — Stack reset semantics

When a heading at level `N` is encountered, slots `N+1` through `6`
are cleared. This matches Markdown semantics: a new `## Section`
ends any subsections that were open under the previous `## Section`.

Example trace:

| Step | Input | Stack after |
|---|---|---|
| 1 | `# A` | `["# A\n", —, —, —, —, —]` |
| 2 | `## A.1` | `["# A\n", "## A.1\n", —, —, —, —]` |
| 3 | `### A.1.x` | `["# A\n", "## A.1\n", "### A.1.x\n", —, —, —]` |
| 4 | `## A.2` | `["# A\n", "## A.2\n", —, —, —, —]` |
| 5 | `# B` | `["# B\n", —, —, —, —, —]` |

## Determinism

### SPEC-CHUNK-530 — Deterministic

For a given list of chunks, the output is deterministic.

## Edge cases

### SPEC-CHUNK-540 — Empty input

For `chunks == []`, return `[]`.

### SPEC-CHUNK-541 — Document without any headings

If no chunk contains a heading line, every output path is `""`.

### SPEC-CHUNK-542 — First chunk contains the document's first heading

The first chunk's path is `""` (the document had no prior headings).
The chunk's content includes the heading; subsequent chunks' paths
will reflect it.

### SPEC-CHUNK-543 — Heading levels skipped

If a document jumps from `# H1` directly to `### H3` without an
intervening `## H2`, the stack tracks exactly what the document says:
slot 1 = `"# H1\n"`, slot 2 = empty, slot 3 = `"### H3\n"`. The
heading path is `"# H1\n### H3\n"` (with the separator). Skipping
levels is the document's choice; this spec does not normalize.

### SPEC-CHUNK-544 — Heading at depth > 6

A line beginning with 7 or more `#` characters is treated as
paragraph content. Standard Markdown defines heading levels 1 through
6 only.

## Named constants

| Name | Value | Defined in |
|------|-------|------------|
| `MAX_HEADING_LEVELS` | `6` | SPEC-CHUNK-510 |
| `HEADING_PATH_SEPARATOR` | `"\n"` | SPEC-CHUNK-513 |

## Implementation-defined behavior

- Whether to scan each chunk with a regex, a streaming parser, or a
  full Markdown AST. Any method that identifies ATX heading lines per
  SPEC-CHUNK-512 is conforming.
- Whether to return paths as strings, structured objects, or both.
  The contract above specifies strings; implementations may offer
  structured forms as additional output.

## Unspecified behavior

- Behavior when a chunk's text contains a `#` character at the start
  of a line *inside* a fenced code block (e.g., a Python comment).
  The naive regex would treat it as a heading. Implementations should
  document whether they parse code fences. The simplest conforming
  implementation does not.
- Behavior when heading text contains other Markdown syntax (links,
  emphasis). The heading line is preserved verbatim; rendering is the
  caller's concern.

## Dependencies

- None beyond standard string/regex operations. No Markdown parser is
  strictly required (ATX heading syntax is regular).
