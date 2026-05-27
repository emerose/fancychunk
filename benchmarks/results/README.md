# Captured benchmark output

Snapshots of `benchmarks/*.py` runs from specific hardware, kept as
reference data. These are **not** auto-regenerated — they will go
stale relative to the live code. Treat them as historical reference,
not ground truth. Re-run the corresponding script to get current
numbers on your hardware.

| File | Script | Hardware (approx.) |
|------|--------|--------------------|
| `embedders.linux.txt` | `python -m benchmarks.embedders` | Linux / RTX 3090 |
| `factories.linux.txt` | `python -m benchmarks.factories` | Linux / RTX 3090 |
| `pipeline.linux.txt` | `python -m benchmarks.pipeline` | Linux / RTX 3090 |

To add a new snapshot, just run the script with output redirected
into this directory (e.g. `python -m benchmarks.sat_batching --device
cuda --n-docs 1000 > benchmarks/results/sat_batching.linux.txt`).
