"""Deterministic fake embedders that satisfy the SegmentEmbedder contract.

These fakes own their own tokenization and per-text alignment —
mirroring how a real adapter is written. ``FakeEmbedder`` joins with
no separator and reports counts via isolated tokenization (the joined
tokenization has the same length because each character is one
token). ``WhitespaceDroppingFakeEmbedder`` skips ``b`` characters so
single-letter texts ``"b"`` produce zero tokens (SPEC-CHUNK-452).
``BertLikeFakeEmbedder`` exercises the special-token absorption case
(SPEC-CHUNK-420 option b). ``RecordingFakeEmbedder`` captures every
``embed_segment`` call's input list — used by late-chunking tests
that assert *what* the embedder was asked to embed (e.g., that the
heading-stack prepend showed up exactly once per segment).

All three methods are ``async def`` to satisfy the post-0.3.0
async-only protocol. The actual work is trivial (microseconds) so
no ``to_thread`` wrapping is needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FakeEmbedder:
    """One-character-per-token deterministic embedder.

    Each character ``ch`` maps to token id ``ord(ch)``; each token has
    a one-hot embedding indexed by ``token_id % dim``. No special
    tokens. ``embed_segment`` joins the input texts with the empty
    string and reports counts as per-text character counts (which
    equals the joined tokenization).
    """

    def __init__(self, dim: int = 8, n_ctx: int = 512) -> None:
        self.dim = dim
        self.n_ctx = n_ctx

    async def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(s) for s in texts]

    async def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        counts = [len(s) for s in texts]
        joined = "".join(texts)
        mat = np.zeros((len(joined), self.dim), dtype=np.float64)
        for i, ch in enumerate(joined):
            mat[i, ord(ch) % self.dim] = 1.0
        return mat, counts

    async def embed_chunklets(
        self, chunklets: list[str]
    ) -> NDArray[np.float64]:
        """Pooled per-chunklet embedding — sum of per-character one-hot
        vectors, L2-normalized. Satisfies the ChunkletEmbedder
        contract so the fake can be used as a full ``Embedder``
        (e.g., with ``chunk_document``)."""
        rows: list[NDArray[np.float64]] = []
        for c in chunklets:
            v = np.zeros(self.dim, dtype=np.float64)
            for ch in c:
                v[ord(ch) % self.dim] += 1.0
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            else:
                # Empty chunklet → satisfy SPEC-CHUNK-342 by using a
                # canonical unit vector. Real callers shouldn't have
                # empty chunklets, but tests sometimes do.
                v[0] = 1.0
            rows.append(v)
        return np.stack(rows) if rows else np.zeros((0, self.dim), dtype=np.float64)


class WhitespaceDroppingFakeEmbedder(FakeEmbedder):
    """Variant that drops every ``b`` character — single-letter texts
    ``"b"`` produce zero tokens. Exercises SPEC-CHUNK-452.
    """

    async def count_tokens(self, texts: list[str]) -> list[int]:  # type: ignore[override]
        return [sum(1 for ch in s if ch != "b") for s in texts]

    async def embed_segment(  # type: ignore[override]
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        # Strip 'b' per text; counts reflect post-strip lengths.
        kept = ["".join(ch for ch in s if ch != "b") for s in texts]
        counts = [len(s) for s in kept]
        joined = "".join(kept)
        mat = np.zeros((len(joined), self.dim), dtype=np.float64)
        for i, ch in enumerate(joined):
            mat[i, ord(ch) % self.dim] = 1.0
        return mat, counts


_BERT_CLS = 1000
_BERT_SEP = 1001


class BertLikeFakeEmbedder(FakeEmbedder):
    """Wraps every segment in ``[CLS] ... [SEP]`` — the BERT-family
    special-token convention. Exercises SPEC-CHUNK-420 option (b):
    leading/trailing specials must be absorbed into the first/last
    text's allocation by the embedder's ``embed_segment``.
    """

    async def count_tokens(self, texts: list[str]) -> list[int]:  # type: ignore[override]
        # Specials are budgeted into the first/last text's count so
        # the segment planner accounts for them too. +1 on the first
        # text for [CLS], +1 on the last for [SEP].
        counts = [len(s) for s in texts]
        if counts:
            counts[0] += 1
            counts[-1] += 1
        return counts

    async def embed_segment(  # type: ignore[override]
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        # Concatenate, prepend [CLS], append [SEP], emit one-hot rows.
        joined = "".join(texts)
        toks = [_BERT_CLS] + [ord(ch) for ch in joined] + [_BERT_SEP]
        mat = np.zeros((len(toks), self.dim), dtype=np.float64)
        for i, t in enumerate(toks):
            mat[i, t % self.dim] = 1.0

        # Per-text counts: first text absorbs [CLS], last absorbs
        # [SEP]. (SPEC-CHUNK-420 option b.)
        counts = [len(s) for s in texts]
        if counts:
            counts[0] += 1
            counts[-1] += 1
        return mat, counts


class RecordingFakeEmbedder(FakeEmbedder):
    """Records every ``embed_segment`` and ``count_tokens`` call.

    ``calls`` is a list of the ``texts`` arguments each
    ``embed_segment`` invocation received, in order. ``count_calls``
    is the same for ``count_tokens``. Tests use these to assert
    *what* the late-chunking algorithm asked the embedder to embed
    (e.g., that the heading-stack prepend appears as text 0 of the
    segment).
    """

    def __init__(self, dim: int = 8, n_ctx: int = 512) -> None:
        super().__init__(dim=dim, n_ctx=n_ctx)
        self.calls: list[list[str]] = []
        self.count_calls: list[list[str]] = []

    async def count_tokens(self, texts: list[str]) -> list[int]:  # type: ignore[override]
        self.count_calls.append(list(texts))
        return await super().count_tokens(texts)

    async def embed_segment(  # type: ignore[override]
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        self.calls.append(list(texts))
        return await super().embed_segment(texts)
