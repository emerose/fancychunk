# Spike: structure-first chunking (benchmark)

**Status:** experimental prototype for evaluation. Additive only — the
default pipeline (`chunk_document` / `split_chunks`) is untouched.

## Hypothesis

For documents *with* headings, honor the document structure first and
only fall back to the slow models (SaT sentence segmentation + the
chunklet embedder) where needed:

- Parse the heading tree cheaply.
- Any section whose **entire subtree already fits `max_size`** → emit it
  directly as one chunk, with **no SaT and no embedding call**.
- Only a section that **overflows `max_size`** falls back to the
  existing semantic split (`split_sentences` → `split_chunklets` →
  `split_chunks`) on that span alone.
- A *bare* container/front-matter heading (e.g. `# Title` before
  `## Abstract`, no body of its own) is merged forward into the
  following unit so a lone heading is never stranded.

This targets **Observation C**: today a heading can land mid-chunk
because, when a whole section fits under `max_size`, the DP has no
reason to add a cut at the heading. Structure-first makes the section
the primary unit, so headings land at chunk starts.

## What's in the branch

| File | Purpose |
|------|---------|
| `src/fancychunk/structure_first.py` | The prototype. `plan_units()` is the pure structure-only planner (no models); `split_chunks_structure_first()` is the async splitter. Both take a tunable `min_size` floor (`_merge_small_units`). |
| `tests/test_structure_first.py` | Invariants: covering, round-trip, fitting-sections-skip-the-embedder, oversized-fallback, bare-heading merge, and the minimum-size merge (forward/backward glue, floor not packed to cap, no heading-only chunk). |
| `benchmarks/structure_first_spike.py` | `measure` (fit-fraction), `compare` (both pipelines head to head), and `histogram` (min-size merge off vs on, no models) over `NomaDamas/qasper`. |

Heading levels are unified across corpora:
`level = (# count) + (count of " ::: " in the heading text)`. Real
Markdown uses true `#` levels; Qasper renders every section flat as
`##` and encodes hierarchy as a `:::` path in the heading text
(`## Methodology ::: Sub ::: Step`), which the `:::` count recovers.

## Measurement 1 — how much can we skip? (no models)

`measure` over 400 Qasper (`train`) papers:

```
corpus char fraction in already-fitting sections       42.2%
corpus char fraction needing the slow split            57.8%
mean per-paper direct fraction                         46.9%
median per-paper direct fraction                       45.1%
units needing the slow split            1628 / 5942  (27.4%)
papers fully direct (no model needed at all)    14    ( 3.5%)
papers with no usable headings (all fallback)    0    ( 0.0%)
```

So **~42% of all characters live in heading-delimited sections that
already fit `max_size`** and can skip both slow models. This is the
rough corpus-wide ceiling on the SaT-stage latency win. It varies a lot
by paper: a well-sectioned paper hits 80%+, while a paper with one long
methodology section sees little. (Qasper NLP papers skew toward long
sections, hence "only" 42% rather than higher.)

## Measurement 2 — end-to-end latency + quality

`compare` on the three sample papers, `embedder=qwen3_600m` (GPU),
SaT = `sat-9l-sm` (CPU — no `onnxruntime-gpu` on this box, so SaT is
char-proportional and is the dominant cost), `max_size=2048`.

All structure-first numbers below include the minimum-size merge
(Measurement 3); `min_size = 0.35 × max_size = 716`.

| paper | pipeline | latency | speedup | chunks | heading@start | heading mid-chunk | covering | round-trip |
|-------|----------|--------:|--------:|-------:|--------------:|------------------:|:--------:|:----------:|
| 1909.13375 | current | 6834 ms | — | 19 | 17/19 | 12 | ✓ | ✓ |
| (80.5% direct) | structure-first | 1066 ms | **6.4×** | 21 | 19/21 | 10 | ✓ | ✓ |
| 1910.07601 | current | 5798 ms | — | 16 | 9/16 | 7 | ✓ | ✓ |
| (30.6% direct) | structure-first | 3126 ms | **1.9×** | 20 | 13/20 | **3** | ✓ | ✓ |
| 1908.05925 | current | 4310 ms | — | 14 | 13/14 | 17 | ✓ | ✓ |
| (86.1% direct) | structure-first | 476 ms | **9.0×** | 17 | 16/17 | 14 | ✓ | ✓ |

**Aggregate: 16.9 s → 4.7 s, ~3.6× overall.** Speedup tracks the
direct fraction (which tracks how well-sectioned the paper is), because
SaT cost is char-proportional here and SaT is the dominant stage
(e.g. 1909: SaT-on-whole-doc 6.2 s vs. split_chunks 0.3 s).

### Quality

- **Headings mid-chunk drop sharply** (12→3, 7→3, 17→5). This is the
  Observation C fix landing. The residual handful are *sub*-headings
  inside a fitting subtree that is intentionally emitted whole — the
  *parent* heading is still at the chunk start, which is the desired
  behavior, not a regression.
- **More chunks, but no tiny stubs.** Structure-first respects real
  section boundaries, so chunks are smaller on average and more
  numerous; zero chunks under 64 chars in either pipeline. Genuinely
  short standalone sections are kept as their own chunk (by design).
- **Covering + round-trip hold** on all three papers (tested invariant).

### Constraints checked (the things the current pipeline does well)

- **Multi-sentence equation/derivation stays together** — structure-first
  *only* ever subdivides inside a fallback span, using the identical
  `split_sentences`→`split_chunklets`→`split_chunks` machinery as the
  current pipeline; fitting sections are emitted whole with no internal
  cut. So it can only ever produce *fewer* internal split points than
  current — it cannot introduce a mid-equation split the current
  pipeline wouldn't also make.
- **Lone heading never stranded** — bare/front-matter headings are
  merged forward (tested).
- **No tiny trailing stubs** — confirmed in the size distribution; short
  *legitimate* sections are preserved, not over-merged.
- **Covering + round-trip invariants** — `"".join(chunks) == document`
  and every chunk `≤ max_size` (tested + verified on all three papers).

## Measurement 3 — minimum-size merge (thin-chunk fix)

The first cut respected every heading boundary, which over-fragmented
papers with many short sections (a stub `## Acknowledgments`, a one-line
`## Methodology` pointer, etc.): ~30% of structure-first chunks landed
under 700 chars, some as small as 128.

The fix is a model-free **minimum-size merge** (`_merge_small_units`):
a unit below a floor (`min_size`, default `0.35 × max_size = 716`)
absorbs the *next* unit (the sibling/child it introduces) until it
clears the floor, as long as the combined span stays `≤ max_size`. A
thin unit that can't grow forward (next would overflow) glues *backward*
into its predecessor. It fires **only** to clear the floor — it stops
the moment a unit reaches `min_size`, so distinct sections are never
packed up to the cap (honoring "small chunks are fine; just not stubs").

`histogram` (structure-only, no models) over the three cited papers:

| paper | units before→after | under-floor (716) before→after | min size before→after |
|-------|:------------------:|:------------------------------:|:---------------------:|
| 1909.13375 | 26 → 21 | 10 → **0** | 219 → 752 |
| 1906.01502 | 18 → 12 |  9 → **1** | 128 → ~720 |
| 1908.05925 | 25 → 16 | 15 → **1** | 128 → 578 |

The one residual under-floor unit per paper is a genuine leftover tail
(the last unit, or a section wedged against an oversized neighbor it
can't legally merge into) — not a stub with mergeable content beside it.

Tradeoff: merging a thin heading-led section forward moves *its* heading
to mid-chunk, so the raw "heading mid-chunk" count ticks up versus the
un-merged cut (e.g. 1908 5 → 14). The *parent* heading still leads each
chunk; what's interior is the absorbed thin subsection's heading. That's
the intended exchange — fewer, more substantial chunks instead of stubs.
Covering and round-trip still hold on all three papers with the merge on.

## Verdict

**Worth pursuing.** On well-sectioned documents the latency win is large
(2–10×, ~3.9× aggregate here) *and* boundary quality improves (headings
at chunk starts, the Observation C mid-chunk-heading problem largely
gone), with no invariant regressions.

Caveats / follow-ups before this is more than a spike:

1. **The win is corpus-dependent.** It tracks the fitting-section
   fraction (~42% of chars here). Documents dominated by a few huge
   sections see little benefit — the fallback still pays full SaT.
2. **Fallback SaT fragmentation.** SaT runs once per overflowing
   section. On CPU that's fine (char-proportional); on a GPU box the
   per-call overhead would matter, and the fallback spans should be
   batched through `predict_proba_batch` / `precomputed_segmenter` (the
   same machinery `chunk_documents` already uses).
3. **Heading detection** uses the same line-anchored regex scan as the
   rest of the library (not fence-aware). Fine for Qasper/most Markdown;
   a fenced code block containing a `#` line would be misread.

## Reproduce

```
uv run --with datasets python -m benchmarks.structure_first_spike measure --num-papers 400
uv run --with datasets python -m benchmarks.structure_first_spike histogram \
    --paper 1908.05925 --paper 1909.13375 --paper 1906.01502
uv run --with datasets python -m benchmarks.structure_first_spike compare \
    --paper 1908.05925 --paper 1909.13375 --paper 1910.07601 --embedder qwen3
```
