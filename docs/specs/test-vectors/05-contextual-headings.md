# Test Vectors — Contextual Chunk Headings

Concrete input/output pairs for the contextual-headings helper
(spec 05). Each should pass as a test case in the implementation's
preferred test framework.

## Notation

- `chunks`: list of strings.
- `expected`: list of expected heading-path strings, same length as
  `chunks`.

## TV-501 — Empty input

Validates SPEC-CHUNK-540.

| Input | Value |
|-------|-------|
| `chunks` | `[]` |

**Expected output:** `[]`

## TV-502 — Document without headings

Validates SPEC-CHUNK-541.

| Input | Value |
|-------|-------|
| `chunks` | `["First paragraph.\n\n", "Second paragraph.\n", "Third paragraph.\n"]` |

**Expected output:** `["", "", ""]`

## TV-503 — Simple linear heading structure

Validates SPEC-CHUNK-510, SPEC-CHUNK-511.

| Input | Value |
|-------|-------|
| `chunks` | `["# Introduction\n\nOpening text.\n\n", "## Background\n\nMore detail.\n\n", "Continuing background.\n", "## Method\n\nDescription.\n"]` |

**Expected output:**

```
[
  "",
  "# Introduction\n",
  "# Introduction\n\n## Background\n",
  "# Introduction\n",
]
```

Notes:
- Chunk 0 starts before any heading is in scope → empty path.
- Chunk 1 starts after `# Introduction` was set → path is just that
  heading. (`# Introduction` is in the path because chunk 0
  *introduced* it; chunk 1's `## Background` is in chunk 1's content,
  not yet in its path.)
- Chunk 2 starts after `## Background` was set → path is both
  headings, joined by the separator.
- Chunk 3 starts after `## Method` was set, which cleared deeper
  levels but slot 2 just got overwritten by `## Method`... wait,
  chunk 3's *path* is what was in scope at chunk 3's start, which is
  after chunk 2 finished (no new heading in chunk 2). At that point
  the stack was `["# Introduction\n", "## Background\n"]`. Chunk 3's
  content then sets slot 2 to `## Method` and clears slot 3+ — but
  that update happens *after* chunk 3's path is snapshotted.

So the corrected expected output for chunk 3 is:

```
"# Introduction\n\n## Background\n"
```

(Same as chunk 2.) The path snapshot happens before scanning each
chunk's content.

## TV-504 — Stack reset when heading level rises

Validates SPEC-CHUNK-520.

| Input | Value |
|-------|-------|
| `chunks` | `["# A\n\n## A.1\n\n### A.1.x\n\nContent.\n", "Next chunk content.\n", "# B\n\nB content.\n", "More B content.\n"]` |

**Expected output:**

```
[
  "",
  "# A\n\n## A.1\n\n### A.1.x\n",
  "# A\n\n## A.1\n\n### A.1.x\n",
  "# B\n",
]
```

Notes:
- Chunk 0's path is empty (document just started).
- Chunk 1's path is the full stack after chunk 0 processed three
  headings.
- Chunk 2's path is *the same* as chunk 1's — chunk 1 contained no
  new headings, so the stack didn't change.
- Chunk 3's path reflects that chunk 2 set slot 1 to `# B` and
  cleared slots 2-6.

## TV-505 — First chunk starts with a heading

Validates SPEC-CHUNK-502, SPEC-CHUNK-542.

| Input | Value |
|-------|-------|
| `chunks` | `["# Title\n\nBody text.\n\n", "More body text.\n"]` |

**Expected output:**

```
[
  "",
  "# Title\n",
]
```

The first chunk introduces `# Title` but its own path is empty —
the heading is in the chunk's content.

## TV-506 — Heading levels skipped

Validates SPEC-CHUNK-543.

| Input | Value |
|-------|-------|
| `chunks` | `["# H1\n\n### H3\n\nContent under H3.\n", "More content.\n"]` |

**Expected output:**

```
[
  "",
  "# H1\n\n### H3\n",
]
```

The path includes `# H1` and `### H3` with slot 2 empty. The
separator joins only the non-empty slots; the resulting string
contains the two heading lines back-to-back with a separator between
them.

## TV-507 — Headings deeper than h6 are not headings

Validates SPEC-CHUNK-544.

| Input | Value |
|-------|-------|
| `chunks` | `["####### Not a heading\n\nBody text.\n", "More text.\n"]` |

**Expected output:** `["", ""]`

Seven `#` characters do not form a valid Markdown heading and do not
update the stack. The chunk is content, not a heading.

## TV-508 — Path format with multi-line heading text

Validates SPEC-CHUNK-512, SPEC-CHUNK-513.

Headings in standard Markdown occupy a single line. If a chunk
contains `# Title with trailing spaces  \n`, the entire line up to
and including the newline is captured in the stack slot.

| Input | Value |
|-------|-------|
| `chunks` | `["# Title with trailing spaces  \n\nBody.\n", "More body.\n"]` |

**Expected output:**

```
[
  "",
  "# Title with trailing spaces  \n",
]
```

(The trailing spaces are preserved verbatim; this spec does not
normalize whitespace within heading lines.)

## TV-509 — Round-trip composition with chunk text

Property test: for any chunk `c[i]` and its computed path `p[i]`, the
string `p[i] + "\n" + c[i]` is a valid Markdown fragment that, when
parsed, would have `c[i]`'s content nested under exactly the heading
hierarchy in `p[i]`. (This is a sanity check on the output format;
implementations can verify by parsing the composed string with a
Markdown parser.)
