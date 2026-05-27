"""Retrieval-quality benchmark on Qasper.

For each (chunker, paper, question) we:

1. Chunk the paper.
2. Embed all chunks (with the chunker's native vectors when it
   produces them — fancychunk's late chunking does; everyone else
   gets re-embedded with the common ``qwen3_600m`` embedder).
3. Embed the question with the same common embedder.
4. Rank chunks by cosine similarity to the question.
5. Compute Recall@k, NDCG@k, Hit@k against the gold evidence
   (relevance = chunk contains an evidence span).

Aggregate per chunker and print a results table.

Usage:
    .venv/bin/python -m benchmarks.retrieval --num-papers 20
    .venv/bin/python -m benchmarks.retrieval                  # full validation split
    .venv/bin/python -m benchmarks.retrieval --chunker fancychunk-late --num-papers 5

This is the slow benchmark — expect ~minutes for the full validation
split on Apple Silicon. Cache the per-chunker embeddings between runs
if you iterate on metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._chunkers import Chunker, all_chunkers
from ._metrics import hit_at_k, ndcg_at_k, recall_at_k, relevance_flags
from ._qasper import QasperPaper, load_qasper


# Common embedder used for query embedding everywhere, and for
# re-embedding chunks from chunkers that don't produce native vectors.
def _common_embedder():
    from fancychunk.embedders import qwen3_600m

    return qwen3_600m()


@dataclass
class ChunkerScores:
    """Aggregated metric sums across all questions; divide by count to mean."""

    name: str
    n_questions: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    n_chunks_total: int = 0
    n_papers: int = 0


async def _embed_chunks_common(
    embedder, texts: list[str]
) -> NDArray[np.float64]:
    """Embed a list of chunk texts with the common embedder. Used for
    chunkers that don't produce native vectors."""
    if not texts:
        return np.zeros((0, embedder.embedding_dim), dtype=np.float64)
    return await embedder.embed_chunklets(texts)


async def _rank_chunks(
    chunk_vecs: NDArray[np.float64], query_vec: NDArray[np.float64]
) -> list[int]:
    """Return chunk indices sorted by descending cosine similarity to
    the query. Assumes both are L2-normalized (qwen3_600m's
    embed_chunklets always normalizes; late-chunked vectors are
    normalized by embed_with_late_chunking)."""
    # Dot product == cosine similarity when both sides are unit norm.
    sims = chunk_vecs @ query_vec
    return np.argsort(-sims).tolist()


async def _eval_chunker_on_paper(
    chunker: Chunker,
    paper: QasperPaper,
    common_embedder,
) -> tuple[ChunkerScores, int]:
    """Score one chunker on one paper. Returns (scores_for_this_paper,
    n_chunks)."""
    scores = ChunkerScores(name=chunker.name)

    chunks, native_vecs = await chunker.achunk(paper.markdown)
    if not chunks:
        return scores, 0

    chunk_vecs = native_vecs
    if chunk_vecs is None:
        chunk_vecs = await _embed_chunks_common(common_embedder, chunks)
    chunk_vecs = np.asarray(chunk_vecs)

    # Embed all questions in one batched call to amortize the embedder
    # overhead — qwen3_600m's embed_chunklets pads internally.
    question_texts = [q.question for q in paper.questions]
    query_vecs = await common_embedder.embed_chunklets(question_texts)

    for q, qvec in zip(paper.questions, query_vecs):
        relevant = relevance_flags(chunks, q.evidence)
        if not any(relevant):
            # Evidence didn't survive as a substring in any chunk —
            # this happens occasionally when the chunker normalizes
            # whitespace or drops content. Skip; no signal to score.
            continue
        ranking = await _rank_chunks(chunk_vecs, np.asarray(qvec))
        scores.n_questions += 1
        scores.recall_at_5 += recall_at_k(ranking, relevant, 5)
        scores.recall_at_10 += recall_at_k(ranking, relevant, 10)
        scores.ndcg_at_10 += ndcg_at_k(ranking, relevant, 10)
        scores.hit_at_5 += hit_at_k(ranking, relevant, 5)
        scores.hit_at_10 += hit_at_k(ranking, relevant, 10)

    return scores, len(chunks)


async def run_benchmark(
    chunkers: list[Chunker],
    papers: list[QasperPaper],
    progress: bool = True,
) -> dict[str, ChunkerScores]:
    """Drive the benchmark. One pass per chunker; per chunker, sequential
    over papers (avoids GPU contention from running multiple chunkers'
    embed_chunklets concurrently on the same device)."""
    from tqdm import tqdm  # type: ignore[import-untyped]

    common = _common_embedder()
    results: dict[str, ChunkerScores] = {}

    for chunker in chunkers:
        agg = ChunkerScores(name=chunker.name)
        iterator = tqdm(papers, desc=chunker.name, disable=not progress)
        for paper in iterator:
            paper_scores, n_chunks = await _eval_chunker_on_paper(
                chunker, paper, common
            )
            agg.n_questions += paper_scores.n_questions
            agg.recall_at_5 += paper_scores.recall_at_5
            agg.recall_at_10 += paper_scores.recall_at_10
            agg.ndcg_at_10 += paper_scores.ndcg_at_10
            agg.hit_at_5 += paper_scores.hit_at_5
            agg.hit_at_10 += paper_scores.hit_at_10
            agg.n_chunks_total += n_chunks
            agg.n_papers += 1
        results[chunker.name] = agg

    return results


def print_results(results: dict[str, ChunkerScores]) -> None:
    """Pretty-print the aggregate scores as a fixed-width table."""
    header = (
        f"{'chunker':<28} {'N_q':>5} {'R@5':>7} {'R@10':>7} {'NDCG@10':>9} "
        f"{'Hit@5':>7} {'Hit@10':>7} {'chunks/doc':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, agg in results.items():
        n = max(agg.n_questions, 1)
        chunks_per_doc = (
            agg.n_chunks_total / max(agg.n_papers, 1)
        )
        print(
            f"{name:<28} {agg.n_questions:>5} "
            f"{agg.recall_at_5 / n:>7.3f} {agg.recall_at_10 / n:>7.3f} "
            f"{agg.ndcg_at_10 / n:>9.3f} "
            f"{agg.hit_at_5 / n:>7.3f} {agg.hit_at_10 / n:>7.3f} "
            f"{chunks_per_doc:>11.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-papers",
        type=int,
        default=None,
        help="Limit to N papers (default: all of the validation split).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
    )
    parser.add_argument(
        "--chunker",
        action="append",
        help="Run only the named chunker(s). May be passed multiple times. "
        "Default: all 6.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm bars."
    )
    args = parser.parse_args()

    print(f"loading Qasper {args.split} split…")
    t0 = time.perf_counter()
    papers = load_qasper(split=args.split, limit=args.num_papers)
    print(
        f"  loaded {len(papers)} papers, "
        f"{sum(len(p.questions) for p in papers)} answerable questions, "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    chunkers = all_chunkers()
    if args.chunker:
        wanted = set(args.chunker)
        chunkers = [c for c in chunkers if c.name in wanted]
        if not chunkers:
            available = ", ".join(c.name for c in all_chunkers())
            raise SystemExit(
                f"no chunker matched {sorted(wanted)}; available: {available}"
            )

    print(f"running {len(chunkers)} chunker(s) over {len(papers)} papers…")
    t0 = time.perf_counter()
    results = asyncio.run(
        run_benchmark(chunkers, papers, progress=not args.no_progress)
    )
    print(f"done in {time.perf_counter() - t0:.1f}s\n")
    print_results(results)


if __name__ == "__main__":
    main()
