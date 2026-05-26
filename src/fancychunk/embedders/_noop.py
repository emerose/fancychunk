"""NoopSegmentEmbedder — constant-output embedder, no model download.

Satisfies the :class:`fancychunk.SegmentEmbedder` protocol so it can
be passed to :func:`split_chunks` (via the embedder argument) and to
:func:`embed_with_late_chunking`. All outputs are identical
L2-normalized vectors of dimension :attr:`embedding_dim` (default
1024 for storage compatibility with the bundled embedders).

Intended use: structural-only chunking when you don't want a model
download. With every chunklet vector identical, every adjacent-
chunklet cosine is 1, the discourse correction projects everything
to zero (triggering the SPEC-CHUNK-321 skip), and partition
similarity falls through to a uniform constant. Stage 3's DP then
minimizes the number of splits subject to ``max_size``, with
heading-aware boundaries preferred — equivalent to the legacy "no
embeddings supplied" path.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


class NoopSegmentEmbedder:
    """Constant-output embedder. Loads no model, downloads nothing."""

    def __init__(self, embedding_dim: int = 1024, n_ctx: int = 4096) -> None:
        self.embedding_dim = embedding_dim
        self.n_ctx = n_ctx
        # Pre-compute the unit vector once. 1/sqrt(D) per element so the
        # L2 norm is exactly 1 — satisfies SPEC-CHUNK-342 (no zero-norm
        # embeddings).
        self._unit_value = 1.0 / math.sqrt(embedding_dim)

    # ----- SegmentEmbedder protocol -----

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Approximate token counts.

        Used only for late-chunking segment-budget planning, which
        noop won't realistically be used for. A character-quarter
        heuristic is sufficient.
        """
        return [max(1, len(s) // 4) for s in texts]

    def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        """Return constant per-token embeddings + per-text counts."""
        counts = self.count_tokens(texts)
        total = sum(counts)
        mat = np.full(
            (total, self.embedding_dim), self._unit_value, dtype=np.float64
        )
        return mat, counts

    # ----- pooled-chunklet convenience -----

    def embed_chunklets(self, chunklets: list[str]) -> NDArray[np.float64]:
        """One identical unit vector per chunklet."""
        return np.full(
            (len(chunklets), self.embedding_dim),
            self._unit_value,
            dtype=np.float64,
        )
