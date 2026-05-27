# fancychunk benchmark harness

Two questions this harness tries to answer:

1. **Does fancychunk actually retrieve better chunks** than fixed-size
   or semantic chunkers, on realistic long scientific documents?
2. **What does that quality cost** in throughput and per-stage latency?

The corpus is [Qasper](https://allenai.org/data/qasper) — questions
about full NLP papers, with ground-truth evidence spans. Long
documents, real section structure, scientific prose: the regime
where chunking decisions actually matter. Short-document benchmarks
like SciFact won't show any signal for late chunking or heading-aware
splits because there's nothing to be clever about.

## Setup

```bash
uv pip install -r benchmarks/requirements.txt
# or
pip install -r benchmarks/requirements.txt
```

You also need fancychunk itself installed with at least one embedder
backend — `pip install -e '.[torch]'` or `'.[mlx]'` on Apple Silicon.

First run downloads the Qasper validation split (~50 MB) and any
embedder weights you haven't already cached.

## Retrieval quality

```bash
.venv/bin/python -m benchmarks.retrieval --num-papers 20  # quick check
.venv/bin/python -m benchmarks.retrieval                  # full validation split
.venv/bin/python -m benchmarks.retrieval --chunker fancychunk-late --chunker recursive-langchain
```

Output is a table:

```
chunker                        N_q     R@5    R@10   NDCG@10   Hit@5  Hit@10  chunks/doc
-----------------------------------------------------------------------------------------
recursive-langchain            420   0.412   0.561     0.483   0.687   0.812        38.4
recursive-chonkie              420   0.418   0.567     0.487   0.691   0.815        37.9
semantic-chonkie               420   0.435   0.582     0.501   0.703   0.823        35.2
fancychunk-noop                420   0.448   0.594     0.512   0.715   0.832        33.7
fancychunk-vanilla             420   0.461   0.605     0.524   0.728   0.841        31.8
fancychunk-late                420   0.487   0.638     0.553   0.752   0.864        31.8
```

(Numbers above are illustrative — your run will produce real ones.)

How to read it:
- `R@k` = Recall@k — of chunks containing gold evidence, what
  fraction is in the top-k by similarity to the question.
- `NDCG@10` = ranking quality, binary relevance.
- `Hit@k` = does any relevant chunk appear in the top-k. Closest to
  "would a RAG pipeline find the answer."
- `chunks/doc` = mean chunks per paper. Lower means bigger chunks
  (less embedder overhead at retrieval time but harder to pin a
  specific fact).

Differences to look for:
- `recursive-langchain` vs `recursive-chonkie`: should be nearly
  identical (both are character-recursive splitters).
- `semantic-chonkie` vs `fancychunk-noop`: chonkie uses embedding
  similarity for boundaries; fancychunk-noop uses heading structure.
  Which heuristic wins depends on the corpus.
- `fancychunk-vanilla` vs `fancychunk-late`: difference here is the
  late-chunking embedding contribution (chunks are identical
  between them; only the storage vectors change).

## Latency

```bash
.venv/bin/python -m benchmarks.latency --num-papers 50
.venv/bin/python -m benchmarks.latency --chunker fancychunk-late
```

Output:

```
chunker                       N   mean_ms median_ms    p95_ms   total_s     MB/s  chunks/doc
---------------------------------------------------------------------------------------------
recursive-langchain          50      3.2       2.8       8.1       0.2    23.45        38.4
fancychunk-late              50    485.1     452.3     780.2      24.3     0.19        31.8
```

For fancychunk-* chunkers, a per-stage breakdown follows that uses
the OpenTelemetry spans the library emits natively. Lets you see
where time actually goes (sentence segmentation, chunklet grouping,
chunk DP, late chunking).

## Notes / caveats

- **All chunkers use the same retrieval embedder** (`qwen3_600m`).
  Each chunker's only contribution is the chunks it produces.
  Exception: `fancychunk-late` uses its late-chunked vectors directly
  as the storage vectors (that's the *point* of late chunking).
- **Substring relevance**, not character-offset overlap. A chunk is
  scored relevant if any gold evidence span appears as a substring.
  Robust to chunkers that normalize whitespace.
- **Skipped questions**: a question is dropped from scoring if no
  chunk contains its evidence — this happens occasionally when a
  chunker strips content. The `N_q` column shows how many questions
  actually scored.
- **Qasper schema**: the HF dataset's qas structure is column-of-lists
  nested; we extract defensively. If a future Qasper revision changes
  the schema, `_qasper.py:_extract_questions` is where to look.
- **No LLM in the loop**. This is a retrieval benchmark, not an
  end-to-end RAG benchmark. Adding answer F1 / exact match on top of
  the retrieved chunks is a natural next step but requires an LLM
  budget you may not want during iteration.
