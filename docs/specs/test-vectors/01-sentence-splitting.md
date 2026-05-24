# Test Vectors — Sentence Splitting

Concrete input/output pairs for stage 1. Each should pass as a test
case in the implementation's preferred test framework.

Where a test depends on a particular sentence-segmentation model, the
test name calls it out. Most tests are model-independent because they
exercise the override mechanism, length-constraint logic, or
short-circuit paths.

## Notation

- `document`: a string.
- `min_len`, `max_len`: integers.
- `expected`: list of expected sentences. Implementations conform if
  `"".join(actual) == document` AND the partition boundaries equal
  those implied by `expected`.

A test is **model-dependent** if the expected output relies on
specific probabilities from a particular segmenter; otherwise it is
**model-independent**.

## TV-101 — Document shorter than `min_len` (model-independent)

Validates SPEC-CHUNK-130.

| Input | Value |
|-------|-------|
| `document` | `"ab"` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output:** `["ab"]`

## TV-102 — Document exactly `min_len` (model-independent)

| Input | Value |
|-------|-------|
| `document` | `"abcd"` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output:** `["abcd"]` (no internal split possible without
violating `min_len`).

## TV-103 — Round-trip preservation (model-independent property)

For any document `D` and any output `S = split_sentences(D, ...)`:

```
"".join(S) == D
```

This must hold for every test case below; the property is
SPEC-CHUNK-100.

## TV-104 — Heading forced standalone with default override (model-independent)

Validates SPEC-CHUNK-113.

| Input | Value |
|-------|-------|
| `document` | `"# Hello\n\nFirst sentence here. Second sentence here.\n"` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output (partition):** the heading `"# Hello\n\n"` must be
its own sentence, separated from the body. The body may further split
into one or two sentences depending on the segmenter.

The conformance check: `expected[0]` ends at or after the position of
the final character of the heading (inclusive of the trailing
newline(s)).

## TV-105 — Heading with body on same paragraph (model-independent)

| Input | Value |
|-------|-------|
| `document` | `"## Title\nBody text follows immediately.\n"` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output (partition):** at least two sentences, with the
first sentence containing `"## Title"` (and possibly trailing
whitespace) and the second sentence containing `"Body text follows
immediately.\n"`.

## TV-106 — Whitespace is trailing, not leading (model-independent)

Validates SPEC-CHUNK-102 and SPEC-CHUNK-114.

| Input | Value |
|-------|-------|
| `document` | `"First sentence. Second sentence. Third sentence."` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output (property):** for every sentence except the first,
the first character must NOT be whitespace.

Acceptable partitions include:
- `["First sentence. ", "Second sentence. ", "Third sentence."]`
- `["First sentence. Second sentence. Third sentence."]`

Unacceptable:
- `["First sentence.", " Second sentence.", " Third sentence."]`
  (second and third sentences start with whitespace)

## TV-107 — `max_len` splits an overlong sentence (model-independent)

Validates SPEC-CHUNK-106 and SPEC-CHUNK-116.

| Input | Value |
|-------|-------|
| `document` | `"a" * 100` (100 characters of `"a"`) |
| `min_len` | `4` |
| `max_len` | `40` |

**Expected output (property):** every output sentence has length `≤
40`. Round-trip holds (`"".join(out) == document`). Exact split
points are implementation-defined.

## TV-108 — `max_len` larger than document is a no-op (model-independent)

| Input | Value |
|-------|-------|
| `document` | `"Short."` |
| `min_len` | `4` |
| `max_len` | `100` |

**Expected output:** `["Short."]`.

## TV-109 — Multi-byte UTF-8 round-trip (model-independent)

Validates SPEC-CHUNK-100 with non-ASCII content.

| Input | Value |
|-------|-------|
| `document` | `"Héllo, wörld. ⊕ symbol here. 日本語テスト。"` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output (property):** `"".join(out) == document` byte-for-byte.
No re-encoding, no normalization (NFC/NFD), no whitespace collapsing.

## TV-110 — Empty document (implementation-defined)

| Input | Value |
|-------|-------|
| `document` | `""` |
| `min_len` | `4` |
| `max_len` | `None` |

**Expected output:** implementation-defined per SPEC-CHUNK-133.
Acceptable: `[""]` or `[]`. The implementor must document the choice.

## TV-111 — Override forces split at specified position (model-independent)

Validates SPEC-CHUNK-112.

| Input | Value |
|-------|-------|
| `document` | `"abcde fghij klmno"` (17 chars) |
| `min_len` | `4` |
| `max_len` | `None` |
| `known_boundary_probas` | vector of length 17; all `NaN` except index `5` which is `1.0` and index `11` which is `1.0` |

**Expected output:** `["abcde ", "fghij ", "klmno"]`. Boundaries at
index 5 and 11 are forced by overrides; nothing else split.

Note: SPEC-CHUNK-114 may shift the exact boundary location within the
whitespace run. Acceptable variants include
`["abcde", " fghij", " klmno"]` only if SPEC-CHUNK-114 is *not*
applied to overrides — but since override values participate in the
final probability vector, the whitespace-trailing rule applies and
the first partition is the conforming one.

## TV-112 — Override prevents split inside heading (model-independent)

Validates SPEC-CHUNK-113.

Construct a document where the segmenter model would naturally split
inside a heading (e.g., a long heading with internal punctuation).
With the default override active, the heading must remain one
sentence.

| Input | Value |
|-------|-------|
| `document` | `"# Long heading: with punctuation, and more.\n\nBody text.\n"` |
| `min_len` | `4` |
| `max_len` | `None` |
| `known_boundary_probas` | (default — Markdown headings) |

**Expected output (property):** the substring `"# Long heading: with
punctuation, and more.\n"` is contained within a single sentence.

## TV-113 — Unsplittable document (model-independent)

Validates SPEC-CHUNK-131.

| Input | Value |
|-------|-------|
| `document` | `"abcdefghij"` (10 chars) |
| `min_len` | `4` |
| `max_len` | `3` (impossible: max < min) |

**Expected output:** the implementation raises an error indicating
no valid partition exists. The exception type is
implementation-defined.

## Model-dependent vectors (informational only)

The following tests depend on the specific sentence segmenter and
serve as sanity checks rather than strict conformance tests. The
acceptance checklist does not require these to pass on a specific
output.

### TV-150 — Period-separated English sentences

| Input | Value |
|-------|-------|
| `document` | `"First sentence here. Second sentence here. Third sentence here."` |
| `min_len` | `4` |
| `max_len` | `None` |

**Plausible output:** three sentences split at the periods. A
segmenter that fails to identify period+space as a strong boundary
may produce a different partition; this is a model quality issue, not
a spec violation, provided round-trip and length constraints hold.
