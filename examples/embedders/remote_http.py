"""Reference SegmentEmbedder over a remote HTTP service.

The library never embeds anything itself — when you need to share a
GPU across services, or run the embedder on a different machine from
where the chunking happens, the SegmentEmbedder contract is small
enough to wrap a thin HTTP call. The complementary server is sketched
at the bottom of this file (drop in your favourite framework — FastAPI,
LiteServe, Sanic, whatever).

Tokenization-alignment strategy: **service-owned**. The server
returns ``per_sentence_counts`` alongside the embedding matrix; the
client trusts the server. The client uses a local tokenizer only for
the cheap ``count_tokens`` budget-planning call so it doesn't have
to round-trip just to plan segments.

Install:
    pip install requests transformers   # tokenizer is local for budgeting

Usage:
    from examples.embedders.remote_http import RemoteEmbedder
    embedder = RemoteEmbedder(
        url="https://my-embed-service.example.com/embed_segment",
        local_tokenizer="bert-base-multilingual-cased",
        n_ctx=512,
    )
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import requests  # type: ignore[import-untyped]
from numpy.typing import NDArray
from transformers import AutoTokenizer  # type: ignore[import-untyped]


class RemoteEmbedder:
    """Thin HTTP client implementing the SegmentEmbedder contract."""

    def __init__(
        self,
        url: str,
        local_tokenizer: str,
        n_ctx: int,
        timeout_seconds: float = 30.0,
        session: Any = None,
    ) -> None:
        self.url = url
        self.n_ctx = n_ctx
        self._timeout = timeout_seconds
        self._tok: Any = AutoTokenizer.from_pretrained(local_tokenizer)
        self._session: Any = session or requests.Session()

    def count_tokens(self, sentences: list[str]) -> list[int]:
        return [
            len(self._tok.encode(s, add_special_tokens=False)) for s in sentences
        ]

    def embed_segment(
        self, sentences: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        resp = self._session.post(
            self.url,
            json={"sentences": sentences},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        matrix = np.asarray(body["embeddings"], dtype=np.float64)
        counts: list[int] = list(body["per_sentence_counts"])
        if matrix.ndim != 2 or sum(counts) != matrix.shape[0]:
            raise ValueError(
                f"server returned matrix shape {matrix.shape} and counts "
                f"summing to {sum(counts)} — sums don't match."
            )
        return cast(NDArray[np.float64], matrix), counts


# ---------------------------------------------------------------------------
# Server sketch (commented out — drop into a file like ``server.py`` and run).
# ---------------------------------------------------------------------------
#
# from fastapi import FastAPI
# import torch
# from transformers import AutoModel, AutoTokenizer
#
# app = FastAPI()
# MODEL = "BAAI/bge-m3"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModel.from_pretrained(MODEL).eval()
#
# @app.post("/embed_segment")
# def embed_segment(payload: dict) -> dict:
#     sentences = payload["sentences"]
#     joined = "".join(sentences)
#     enc = tokenizer(
#         joined,
#         return_offsets_mapping=True,
#         return_tensors="pt",
#         add_special_tokens=True,
#     )
#     with torch.no_grad():
#         h = model(
#             input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
#         ).last_hidden_state[0]
#     mat = h.float().cpu().numpy()
#
#     # Compute per-sentence counts via offset_mapping (same as
#     # huggingface_offsets.py). The server returns counts as a list[int]
#     # alongside the matrix so the client doesn't need a tokenizer.
#     counts = compute_counts_via_offsets(sentences, enc["offset_mapping"][0])
#     return {
#         "embeddings": mat.tolist(),
#         "per_sentence_counts": counts,
#     }
#
# # Note: ``mat.tolist()`` is ~10x slower than binary serialization.
# # For production, swap JSON for msgpack/protobuf — see numpy's
# # serialization guidance.


if __name__ == "__main__":
    print(
        "This is a client sketch. Stand up the server stub from this file's "
        "comments, then point RemoteEmbedder at its URL."
    )
