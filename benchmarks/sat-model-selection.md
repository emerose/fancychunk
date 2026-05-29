# SaT model selection for the default segmenter

Which Segment Any Text (SaT) checkpoint should `fancychunk` default to?
The defect that motivated this — sentence-splitting artifacts in
scientific prose — turned out to be a model/inference-parameter
problem, not something to patch with probability post-processing
(see SPEC-CHUNK-118, removed). Two parameters dominate quality:

- **`weighting="hat"`** (vs wtpsplit-lite's `"uniform"` default).
  Uniform averaging weights low-context sliding-window edges as much as
  window centres, producing *context-sensitive* boundary artifacts.
  `hat` de-weights edges. We run all checkpoints with `hat`.
- **Checkpoint depth** (`sat-3l-sm` / `sat-9l-sm` / `sat-12l-sm`). The
  data below.

**Conclusion: default to `sat-9l-sm`.** It is artifact-free like 12l,
tracks 12l's boundary placement far better than 3l, and is ~1.3× faster
than 12l on a batched GPU path. 3l is fastest but
mis-segments scientific prose; 12l is highest quality but slowest. All
three remain selectable (`fancychunk.segmenters.sat_3l/9l/12l`).

## Quality — artifact probes

Boundary probability at the marked position (threshold `0.25`; low is
correct for an abbreviation/year mid-clause, high for a genuine end):

| probe | 3l | 9l | 12l |
|---|---|---|---|
| `Tab.` period (standalone) | 0.430 ✗ | 0.002 ✓ | 0.005 ✓ |
| `Eq.` period | 0.645 ✗ | 0.004 ✓ | 0.001 ✓ |
| year `4` before `Task` | 0.515 ✗ | 0.058 ✓ | 0.045 ✓ |
| real full-doc `Tab.` (1707, uniform/256/512) | 0.301 ✗ | 0.001 ✓ | 0.001 ✓ |
| genuine `datasets.` end | 0.444 ✓ | 0.997 ✓ | 0.995 ✓ |

9l is artifact-free, and equally robust across inference params (a
model-level fix). Only 3l breaks.

## Agreement with 12l

Boundary placement over the whole Qasper corpus (prob ≥ 0.25, ±1 char),
treating 12l as reference:

| candidate | precision | recall | F1 | false-positive boundaries |
|---|---|---|---|---|
| sat-3l-sm | 0.957 | 0.957 | 0.957 | 753 |
| sat-9l-sm | 0.979 | 0.968 | 0.973 | 367 |

9l tracks 12l noticeably better than 3l, with half the spurious
boundaries.

## Throughput — GPU (production path)

173 docs, mean 12.1K chars, RTX 3090, **batched** (a batched
SaT path), `hat`/128/256:

| model | Kchar/s (batched) | ms/doc | vs 12l |
|---|---|---|---|
| sat-3l-sm | 1481 | 8.2 | 3.2× faster |
| sat-9l-sm | 596 | 20.3 | 1.29× faster |
| sat-12l-sm | 461 | 26.2 | 1.0× |

Inference params barely move throughput: 9l `uniform/256/512` = 583,
9l `hat/256/512` = 580 — all within 3% of `hat/128/256`.

## End-to-end coherence (9l segmenter + qwen3_600m)

Passes all three Qasper coherence cases identically to 12l, same chunk
counts (10 / 16 / 13):

- `1707.06806` — no chunk starts with `TABREF21` (abbreviation not split).
- `1909.13375` — the `(1)`–`(4)` model-head items stay in one chunk.
- `1710.07695` — the section lead-in stays with its definitions.

## Notes — Apple Silicon (CPU / CoreML)

Measured on an M-series Mac (single-doc, thermally noisy absolute
numbers; ratios are reliable):

- **CoreML execution provider is *slower*** than CPU for these models
  (~0.7×) — the ONNX graph partially falls back and the round-trips
  cost more than they save. Use the CPU provider on a Mac.
- **`block_size` is not a dependable lever.** Larger blocks (512/1024)
  were ~1.5× faster in one run and slower in another — per-window
  overhead vs quadratic attention trade off unpredictably with thermal
  state. Quality is unaffected either way.
- **Model depth is the reliable lever.** 9l is ~1.3–2× faster than 12l
  on CPU and fixes the same artifacts, so `sat-9l-sm` is the
  Mac-friendly default too. `sat-6l-sm` is *not* sufficient — it fixes
  the abbreviation but still mis-segments the year (prob ~0.82).
