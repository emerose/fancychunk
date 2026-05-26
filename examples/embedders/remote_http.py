"""Reference Embedder over a remote HTTP service.

The library never embeds anything itself — when you need to share a
GPU across services, or run the embedder on a different machine from
where the chunking happens, the protocol is small enough to wrap a
thin HTTP call. The complementary server is sketched at the bottom
of this file (drop in your favourite framework — FastAPI, LiteServe,
Sanic, whatever).

The client implements both halves of the protocol — ``embed_segment``
for late chunking and ``embed_chunklets`` for the partition decision
— against two endpoints under a shared base URL. The server handles
tokenization and pooling; the client uses a local tokenizer only for
the cheap ``count_tokens`` budget-planning call so it doesn't have
to round-trip just to plan segments.

Install:
    pip install requests transformers   # tokenizer is local for budgeting

Usage:
    from examples.embedders.remote_http import RemoteEmbedder
    embedder = RemoteEmbedder(
        base_url="https://my-embed-service.example.com",
        local_tokenizer="bert-base-multilingual-cased",
        n_ctx=512,
    )
    # GET {base_url}/embed_segment   for late chunking
    # GET {base_url}/embed_chunklets for the partition decision
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import requests  # type: ignore[import-untyped]
from numpy.typing import NDArray
from transformers import AutoTokenizer  # type: ignore[import-untyped]


class RemoteEmbedder:
    """Thin HTTP client implementing the full Embedder protocol."""

    def __init__(
        self,
        base_url: str,
        local_tokenizer: str,
        n_ctx: int,
        timeout_seconds: float = 30.0,
        session: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._segment_url = f"{self.base_url}/embed_segment"
        self._chunklets_url = f"{self.base_url}/embed_chunklets"
        self.n_ctx = n_ctx
        self._timeout = timeout_seconds
        self._tok: Any = AutoTokenizer.from_pretrained(local_tokenizer)
        self._session: Any = session or requests.Session()

    # ----- SegmentEmbedder contract -----

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [
            len(self._tok.encode(s, add_special_tokens=False)) for s in texts
        ]

    def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        resp = self._session.post(
            self._segment_url,
            json={"texts": texts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        matrix = np.asarray(body["embeddings"], dtype=np.float64)
        counts: list[int] = list(body["per_text_counts"])
        if matrix.ndim != 2 or sum(counts) != matrix.shape[0]:
            raise ValueError(
                f"server returned matrix shape {matrix.shape} and counts "
                f"summing to {sum(counts)} — sums don't match."
            )
        return cast(NDArray[np.float64], matrix), counts

    # ----- ChunkletEmbedder contract -----

    def embed_chunklets(
        self, chunklets: list[str]
    ) -> NDArray[np.float64]:
        """One pooled embedding per chunklet — server-owned pooling
        strategy. Used by ``split_chunks`` and ``chunk_document``.

        The server is expected to L2-normalize each row; the client
        does not validate the norms (the library's SPEC-CHUNK-342
        check will catch zero rows downstream)."""
        if not chunklets:
            # Server may not handle empty input; short-circuit. Dim
            # is unknown without a round-trip, so return a 0×0
            # placeholder — split_chunks doesn't invoke the embedder
            # on empty input anyway (SPEC-CHUNK-340).
            return np.zeros((0, 0), dtype=np.float64)
        resp = self._session.post(
            self._chunklets_url,
            json={"chunklets": chunklets},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        matrix = np.asarray(body["embeddings"], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunklets):
            raise ValueError(
                f"server returned matrix of shape {matrix.shape}; expected "
                f"({len(chunklets)}, D)."
            )
        return cast(NDArray[np.float64], matrix)


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
# POOLING = "cls"  # match your model's training: cls / mean / last_token
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModel.from_pretrained(MODEL).eval()
#
# @app.post("/embed_segment")
# def embed_segment(payload: dict) -> dict:
#     texts = payload["texts"]
#     joined = "".join(texts)
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
#     # Compute per-text counts via offset_mapping (same as
#     # huggingface_offsets.py). The server returns counts as a list[int]
#     # alongside the matrix so the client doesn't need a tokenizer.
#     counts = compute_counts_via_offsets(texts, enc["offset_mapping"][0])
#     return {
#         "embeddings": mat.tolist(),
#         "per_text_counts": counts,
#     }
#
# @app.post("/embed_chunklets")
# def embed_chunklets(payload: dict) -> dict:
#     chunklets = payload["chunklets"]
#     enc = tokenizer(
#         chunklets,
#         padding=True, truncation=True, return_tensors="pt",
#     )
#     with torch.no_grad():
#         h = model(
#             input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
#         ).last_hidden_state
#     # Pool per the model's training. CLS for BERT/BGE; mean for
#     # MPNet/MiniLM; last_token for Qwen3-Embedding.
#     if POOLING == "cls":
#         pooled = h[:, 0]
#     elif POOLING == "mean":
#         m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
#         pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
#     elif POOLING == "last_token":
#         seq_lens = enc["attention_mask"].sum(dim=1) - 1
#         idx = torch.arange(h.shape[0], device=h.device)
#         pooled = h[idx, seq_lens.clamp(min=0)]
#     pooled = torch.nn.functional.normalize(pooled, p=2.0, dim=1)
#     return {"embeddings": pooled.float().cpu().numpy().tolist()}
#
# # Note: ``mat.tolist()`` is ~10x slower than binary serialization.
# # For production, swap JSON for msgpack/protobuf — see numpy's
# # serialization guidance.


if __name__ == "__main__":
    print(
        "This is a client sketch. Stand up the server stub from this file's "
        "comments, then point RemoteEmbedder at its base URL."
    )
