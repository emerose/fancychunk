"""Deterministic fake embedder for testing late chunking.

The fake satisfies SPEC-CHUNK-04 §Embedder contract:

* ``tokenize(text)`` returns a list of integer token IDs (deterministic).
* ``detokenize(tokens)`` is its inverse.
* ``embed(text)`` returns one row per token, with each row being a
  one-hot vector indexed by ``token_id mod dim``.
* ``n_ctx`` is the maximum input length.

A token corresponds to one ``[A-Za-z0-9]`` character or one
non-alphanumeric character (whitespace included). The token ID of any
ASCII character is its ``ord``; the sentinel ``⊕`` is given ID 9999.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_SENTINEL = "⊕"
_SENTINEL_TOKEN = 9999


class FakeEmbedder:
    def __init__(self, dim: int = 8, n_ctx: int = 512) -> None:
        self.dim = dim
        self.n_ctx = n_ctx

    def tokenize(self, text: str) -> list[int]:
        return [self._char_to_token(ch) for ch in text]

    def detokenize(self, tokens: list[int]) -> str:
        out: list[str] = []
        for tok in tokens:
            if tok == _SENTINEL_TOKEN:
                out.append(_SENTINEL)
            elif 0 <= tok < 0x110000:
                out.append(chr(tok))
            else:
                out.append("?")
        return "".join(out)

    def embed(self, text: str) -> NDArray[np.float64]:
        tokens = self.tokenize(text)
        mat = np.zeros((len(tokens), self.dim), dtype=np.float64)
        for i, tok in enumerate(tokens):
            mat[i, tok % self.dim] = 1.0
        return mat

    @staticmethod
    def _char_to_token(ch: str) -> int:
        if ch == _SENTINEL:
            return _SENTINEL_TOKEN
        return ord(ch)


class WhitespaceDroppingFakeEmbedder(FakeEmbedder):
    """Fake embedder that produces zero tokens for any single ``b`` character.

    Used by TV-409 to exercise the zero-allocation case in
    SPEC-CHUNK-452: when a single-character sentence tokenises to no
    tokens, the implementation must either floor at one token or raise.
    """

    def tokenize(self, text: str) -> list[int]:  # type: ignore[override]
        # Drop every ``b`` character entirely.
        if not text:
            return []
        return [self._char_to_token(ch) for ch in text if ch != "b"]

    def embed(self, text: str):  # type: ignore[override]
        return super().embed(text.replace("b", ""))


_BERT_CLS = 1000
_BERT_SEP = 1001


class BertLikeFakeEmbedder(FakeEmbedder):
    """Fake embedder that wraps every embed/tokenize call in ``[CLS] ... [SEP]``.

    Used to exercise SPEC-CHUNK-420 option (b): leading and trailing
    special tokens are absorbed into the first and last content
    sentences' allocations.
    """

    def tokenize(self, text: str) -> list[int]:  # type: ignore[override]
        return [_BERT_CLS] + super().tokenize(text) + [_BERT_SEP]

    def detokenize(self, tokens: list[int]) -> str:  # type: ignore[override]
        out: list[str] = []
        for t in tokens:
            if t in (_BERT_CLS, _BERT_SEP):
                continue
            out.append(super().detokenize([t]))
        return "".join(out)

    def embed(self, text: str):  # type: ignore[override]
        tokens = self.tokenize(text)
        mat = np.zeros((len(tokens), self.dim), dtype=np.float64)
        for i, tok in enumerate(tokens):
            mat[i, tok % self.dim] = 1.0
        return mat
