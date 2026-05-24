# Test Vectors — Chunklet Grouping

Concrete input/output pairs for stage 2. Inputs are lists of
sentences (with implied Markdown structure where relevant) and the
expected output is a chunklet partition.

## Notation

- `sentences`: list of strings. The Markdown structure of
  `"".join(sentences)` determines per-sentence boundary
  probabilities (SPEC-CHUNK-240).
- `max_size`: integer.
- `expected_partition`: list of tuples `(start_idx, end_idx)` over
  the input sentences, OR a list of chunklet strings.

## TV-201 — Empty input (model-independent)

Validates SPEC-CHUNK-260.

| Input | Value |
|-------|-------|
| `sentences` | `[]` |
| `max_size` | `2048` |

**Expected output:** `[]`.

## TV-202 — Single sentence input (model-independent)

Validates SPEC-CHUNK-261.

| Input | Value |
|-------|-------|
| `sentences` | `["Just one sentence."]` |
| `max_size` | `2048` |

**Expected output:** `["Just one sentence."]`.

## TV-203 — Concatenation round-trip (property)

For any input `S` and output `C = split_chunklets(S, ...)`:

```
"".join(C) == "".join(S)
```

Must hold for every test case below; the property is SPEC-CHUNK-200.

## TV-204 — Hard size constraint forces split (model-independent)

Validates SPEC-CHUNK-201.

| Input | Value |
|-------|-------|
| `sentences` | `["a" * 1000, "b" * 1000, "c" * 1000]` (three sentences of 1000 `a`/`b`/`c`) |
| `max_size` | `2048` |

**Expected output (property):** every chunklet has length `≤ 2048`.

Since the three sentences sum to 3000 chars and pairs sum to 2000
chars (both fit), the conforming partitions are:
- `[sent[0], sent[1] + sent[2]]` (chunk sizes 1000, 2000)
- `[sent[0] + sent[1], sent[2]]` (chunk sizes 2000, 1000)
- `[sent[0], sent[1], sent[2]]` (chunk sizes 1000, 1000, 1000)

The implementation chooses the one minimizing total cost. None of:
- `[sent[0] + sent[1] + sent[2]]` (3000 chars, violates SPEC-CHUNK-201)

## TV-205 — Heading boundary is a strong split point (model-independent)

Validates SPEC-CHUNK-240 and SPEC-CHUNK-241.

| Input | Value |
|-------|-------|
| `sentences` | `["First paragraph sentence.\n\n", "## A new section\n\n", "Body sentence one.\n", "Body sentence two.\n"]` |
| `max_size` | `2048` |

**Expected output (property):** a chunklet boundary must occur such
that the heading sentence `"## A new section\n\n"` starts a new
chunklet. That is, the heading sentence's index (1) appears as a
chunklet start in the partition.

Conforming partition: `[sent[0], sent[1] + sent[2] + sent[3]]` or
`[sent[0], sent[1] + sent[2], sent[3]]` — both put the heading at a
chunklet boundary.

Non-conforming: `[sent[0] + sent[1] + sent[2] + sent[3]]` (heading
swallowed into the same chunklet as preceding sentence).

The boundary may not be *required* depending on the statement cost
balance, but the strength of the heading probability (`1.0`) versus
non-headings makes the split heavily preferred. This test is robust
when input sentence statement counts are similar.

## TV-206 — Three-statement target (model-dependent on statement counting)

Validates SPEC-CHUNK-221 and SPEC-CHUNK-230.

Construct a document where every sentence has the same word count
(say 10 words each), and there are no Markdown structural cues
beyond paragraph_open at the start.

| Input | Value |
|-------|-------|
| `sentences` | 12 sentences, each 10 words, all in the same paragraph |
| `max_size` | `2048` |

**Expected output (property):** since every sentence has the same
word count, each is `1.0` statements (the median is itself). The
chunklet partition should favor 3-sentence chunklets to minimize the
statement-cost penalty `(s - 3)² / sqrt(s) / 2`.

Conforming partition (preferred): `[3, 3, 3, 3]` (four chunklets of
3 sentences each).

Acceptable alternatives if boundary costs intervene: `[3, 3, 2, 4]`,
`[4, 4, 4]`, etc.

## TV-207 — Long sentences reach statement count of 3 quickly

Validates SPEC-CHUNK-230 with sentences far above the document
median.

| Input | Value |
|-------|-------|
| `sentences` | 6 sentences. Sentences 0-2 are 5 words each; sentences 3-5 are 30 words each. |
| `max_size` | `2048` |

Computed per-sentence statement counts (using `q25 = 5`, `q75 = ~30`):
- Sentences 0-2: `n=5 ≤ q25`, so `0.75 * 5 / 5 = 0.75` statements
  each.
- Sentences 3-5: `n=30 > q25`, so `0.75 + 0.5 * (30 - 5) / (30 - 5)
  = 1.25` statements each.

**Expected output (property):** chunklets including sentences 0-2
should target ~4 short sentences to reach 3 statements
(`4 * 0.75 = 3.0`). Chunklets including sentences 3-5 should target
~2-3 long sentences (`3 * 1.25 = 3.75`, `2 * 1.25 = 2.5`).

A conforming partition might be `[3, 3]` (three short + three long).
Slight variations are acceptable.

## TV-208 — Consecutive non-zero boundaries: only the strongest survives

Validates SPEC-CHUNK-241.

Construct an input where sentences `1, 2, 3` each start with a
non-zero structural cue.

| Input | Value |
|-------|-------|
| `sentences` | `["Intro.\n\n", "> Blockquote line.\n", "* Bullet one.\n", "Continued text.\n"]` |
| `max_size` | `2048` |

Per-sentence boundary probabilities *before* SPEC-CHUNK-241 cleanup:
- Sentence 0: `paragraph_open` → `0.5`
- Sentence 1: `blockquote_open` → `0.75`
- Sentence 2: `bullet_list_open` → `0.25`
- Sentence 3: `paragraph_open` → `0.5`

The run `[0.75, 0.25]` at sentences 1-2 has its `0.25` suppressed to
`0.0`. The run `[0.5]` at sentence 0 and the run `[0.5]` at sentence
3 are isolated singletons; both survive.

Final boundary probas: `[0.5, 0.75, 0.0, 0.5]`.

**Expected output (property):** sentence 2 ("bullet one") is *less*
likely to start a chunklet than sentence 1 ("blockquote") because the
suppression rule made sentence 2's probability zero. A partition that
makes sentence 1 a chunklet boundary while folding sentence 2 into
the same chunklet as sentence 1 (or the next one) is conforming and
preferred.

## TV-209 — Heading on first sentence (default state)

Validates SPEC-CHUNK-240.

| Input | Value |
|-------|-------|
| `sentences` | `["# Title\n\n", "Body sentence one.\n", "Body sentence two.\n"]` |
| `max_size` | `2048` |

**Expected output (property):** sentence 0 (heading) starts a
chunklet, which is trivially true (it's the first sentence). The body
sentences may or may not be in the same chunklet as the heading
depending on statement cost.

## TV-210 — Custom cost functions (model-independent)

If the implementation supports custom `boundary_cost` and
`statement_cost`, supplying constant-zero functions for both should
yield: every partition has equal cost, so any conforming partition is
optimal. The implementation should still return a deterministic
choice; tie-breaking per SPEC-CHUNK-251 picks the partition with
earliest split points.

| Input | Value |
|-------|-------|
| `sentences` | 6 identical sentences |
| `max_size` | `2048` |
| `boundary_cost` | `lambda p: 0.0` |
| `statement_cost` | `lambda s: 0.0` |

**Expected output:** a single chunklet containing all sentences,
because the empty-partition case has cost 0 and any further splits
add nothing. (With zero cost everywhere, the chunklet count is
unconstrained from below; the early-split-preference tie-breaker
prefers the partition with the smallest number of chunklets when cost
is constant. Implementations conforming to SPEC-CHUNK-251 produce
`["".join(sentences)]`.)
