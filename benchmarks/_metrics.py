"""Retrieval metrics with chunk-level binary relevance.

A chunk is *relevant* for a question iff its text contains any of
the question's gold evidence spans as a substring. Substring rather
than offset overlap because some chunkers (LangChain) don't expose
reliable character offsets — and substring is what actually matters
for RAG (if the evidence text is in the chunk, the LLM has the
info it needs).
"""

from __future__ import annotations

import math


def relevance_flags(chunks: list[str], evidence: list[str]) -> list[bool]:
    """Per-chunk binary relevance (True = chunk contains at least one
    evidence span as a substring)."""
    return [any(span in chunk for span in evidence) for chunk in chunks]


def recall_at_k(ranked_indices: list[int], relevant: list[bool], k: int) -> float:
    """Recall@k. Fraction of relevant chunks that appear in the top-k
    of the ranking. Returns 0.0 when there are no relevant chunks
    (degenerate — caller should filter)."""
    total_relevant = sum(relevant)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_indices[:k]
    hits = sum(1 for i in top_k if relevant[i])
    return hits / total_relevant


def ndcg_at_k(ranked_indices: list[int], relevant: list[bool], k: int) -> float:
    """NDCG@k with binary relevance (gain = 1 for relevant, 0 else).

    Manual impl — sklearn's ndcg_score wants a different shape and
    we don't need its features here.
    """
    if not any(relevant):
        return 0.0
    dcg = 0.0
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if relevant[idx]:
            dcg += 1.0 / math.log2(rank + 1)
    # Ideal DCG: all relevant items packed into the top positions.
    n_rel = sum(relevant)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, n_rel) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked_indices: list[int], relevant: list[bool], k: int) -> float:
    """1.0 if any relevant chunk is in top-k, else 0.0. Useful for
    short answer-bearing queries where catching *any* evidence is
    the practical win."""
    if not any(relevant):
        return 0.0
    top_k = ranked_indices[:k]
    return 1.0 if any(relevant[i] for i in top_k) else 0.0
