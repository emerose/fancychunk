"""Benchmark harness for fancychunk vs comparator chunkers on Qasper.

Not part of the installed library. Entry points:

* ``python -m benchmarks.retrieval`` — retrieval quality (Recall@k,
  NDCG@k, Hit@k).
* ``python -m benchmarks.latency`` — throughput + per-stage timings.

See ``benchmarks/README.md`` for setup and notes.
"""
