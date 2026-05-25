"""Deterministic fake embedders that satisfy the SegmentEmbedder contract.

These fakes own their own tokenization and per-sentence alignment —
mirroring how a real adapter is written. ``FakeEmbedder`` joins with
no separator and reports counts via isolated tokenization (the joined
tokenization has the same length because each character is one
token). ``WhitespaceDroppingFakeEmbedder`` skips ``b`` characters so
single-letter sentences ``"b"`` produce zero tokens (TV-409). The
fakes have no special tokens; tests in
``test_telemetry.py``/``test_late_chunking.py`` verify the
``BertLikeFakeEmbedder`` for the special-token absorption case.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FakeEmbedder:
    """One-character-per-token deterministic embedder.

    Each character ``ch`` maps to token id ``ord(ch)``; each token has
    a one-hot embedding indexed by ``token_id % dim``. No special
    tokens. ``embed_segment`` joins sentences with the empty string
    and reports counts as per-sentence character counts (which equals
    the joined tokenization).
    """

    def __init__(self, dim: int = 8, n_ctx: int = 512) -> None:
        self.dim = dim
        self.n_ctx = n_ctx

    def count_tokens(self, sentences: list[str]) -> list[int]:
        return [len(s) for s in sentences]

    def embed_segment(
        self, sentences: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        counts = [len(s) for s in sentences]
        joined = "".join(sentences)
        mat = np.zeros((len(joined), self.dim), dtype=np.float64)
        for i, ch in enumerate(joined):
            mat[i, ord(ch) % self.dim] = 1.0
        return mat, counts


class WhitespaceDroppingFakeEmbedder(FakeEmbedder):
    """Variant that drops every ``b`` character — single-letter sentences
    ``"b"`` produce zero tokens. Exercises SPEC-CHUNK-452.
    """

    def count_tokens(self, sentences: list[str]) -> list[int]:  # type: ignore[override]
        return [sum(1 for ch in s if ch != "b") for s in sentences]

    def embed_segment(  # type: ignore[override]
        self, sentences: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        # Strip 'b' per sentence; counts reflect post-strip lengths.
        kept = ["".join(ch for ch in s if ch != "b") for s in sentences]
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
    sentence's allocation by the embedder's ``embed_segment``.
    """

    def count_tokens(self, sentences: list[str]) -> list[int]:  # type: ignore[override]
        # Specials are budgeted into the first/last sentence's count
        # so the segment planner accounts for them too. +1 on the
        # first sentence for [CLS], +1 on the last for [SEP].
        counts = [len(s) for s in sentences]
        if counts:
            counts[0] += 1
            counts[-1] += 1
        return counts

    def embed_segment(  # type: ignore[override]
        self, sentences: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        # Concatenate, prepend [CLS], append [SEP], emit one-hot rows.
        joined = "".join(sentences)
        toks = [_BERT_CLS] + [ord(ch) for ch in joined] + [_BERT_SEP]
        mat = np.zeros((len(toks), self.dim), dtype=np.float64)
        for i, t in enumerate(toks):
            mat[i, t % self.dim] = 1.0

        # Per-sentence counts: first sentence absorbs [CLS], last
        # absorbs [SEP]. (SPEC-CHUNK-420 option b.)
        counts = [len(s) for s in sentences]
        if counts:
            counts[0] += 1
            counts[-1] += 1
        return mat, counts
