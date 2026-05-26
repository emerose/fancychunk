"""Reference SegmentEmbedder over a HuggingFace transformer.

Uses the tokenizer's ``offset_mapping`` feature to map each output
token back to its source text by character offset — the most robust
alignment method when it's available. Recommended for any
HuggingFace ``AutoTokenizer`` based on a fast (Rust) tokenizer.

Tested with ``bert-base-multilingual-cased`` and ``BAAI/bge-m3``. For
larger models (Qwen3-Embedding-* via transformers) the same code
works; if you're on Apple Silicon prefer ``qwen3_mlx.py`` for speed.

Tokenization-alignment strategy: **offset-based**. The input texts
(chunks plus any heading-stack prepend the library passes us) are
joined with no separator. The tokenizer's offset_mapping reports
each token's character span; tokens whose offset falls inside a
text's character range count toward that text.

Specials: detected by their offset of ``(0, 0)`` and absorbed into
the first / last text's allocation (SPEC-CHUNK-420 option b).

Install:
    pip install torch transformers

Usage:
    from examples.embedders.huggingface_offsets import HFOffsetEmbedder
    from fancychunk import (
        embed_with_late_chunking,
        split_chunklets,
        split_chunks,
        split_sentences,
    )

    embedder = HFOffsetEmbedder("BAAI/bge-m3")
    sentences = split_sentences(my_document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    chunks = split_chunks(chunklets, max_size=2048)
    matrix = embed_with_late_chunking(chunks, embedder)
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import torch  # type: ignore[import-untyped]
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]


PoolingMethod = Literal["cls", "mean", "last_token"]


class HFOffsetEmbedder:
    """Late-chunking + pooled-chunklet adapter for HuggingFace
    transformers with offset mapping.

    Pooling for ``embed_chunklets`` is configurable since each
    HuggingFace model has its own intended strategy:

    * BERT-family (BGE-M3, mBERT, etc.) → ``"cls"``
    * Sentence-Transformers MPNet/MiniLM → ``"mean"``
    * Decoder embedders (Qwen3-Embedding via transformers) →
      ``"last_token"``

    Pick the right one for your model — pooling other than what
    the model was trained with gives random-looking embeddings.
    """

    def __init__(
        self,
        model_name: str,
        n_ctx: int | None = None,
        device: str = "cpu",
        pooling: PoolingMethod = "cls",
    ) -> None:
        self._tok: Any = AutoTokenizer.from_pretrained(model_name)
        self._model: Any = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._device = device
        self.pooling: PoolingMethod = pooling
        if device != "cpu":
            self._model = self._model.to(device)
        # n_ctx defaults to the tokenizer's model_max_length, clipped
        # to 4096 (HuggingFace sometimes reports astronomical values
        # like 1e30 for models without an explicit limit).
        if n_ctx is None:
            mml = int(getattr(self._tok, "model_max_length", 4096))
            n_ctx = min(mml, 4096) if mml > 0 else 4096
        self.n_ctx = n_ctx

    @property
    def embedding_dim(self) -> int:
        """Native hidden size of the loaded model."""
        return int(self._model.config.hidden_size)

    # ----- SegmentEmbedder contract -----

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [
            len(self._tok.encode(s, add_special_tokens=False)) for s in texts
        ]

    def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        joined = "".join(texts)
        enc = self._tok(
            joined,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"][0].tolist()

        if self._device != "cpu":
            input_ids = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            h = self._model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[0]
        mat = cast(NDArray[np.float64], h.float().cpu().numpy()).astype(np.float64)

        # Build the per-text character spans.
        spans: list[tuple[int, int]] = []
        pos = 0
        for s in texts:
            spans.append((pos, pos + len(s)))
            pos += len(s)

        counts = [0] * len(texts)
        for a, b in offsets:
            if a == 0 and b == 0:
                # Special token — defer; absorbed below.
                continue
            mid = (a + b) // 2
            for s_idx, (sa, sb) in enumerate(spans):
                if sa <= mid < sb:
                    counts[s_idx] += 1
                    break

        # Absorb leading specials into text 0; trailing into the last.
        for a, b in offsets:
            if a == 0 and b == 0:
                counts[0] += 1
            else:
                break
        for k in range(len(offsets) - 1, -1, -1):
            a, b = offsets[k]
            if a == 0 and b == 0:
                counts[-1] += 1
            else:
                break

        return mat, counts

    # ----- ChunkletEmbedder contract -----

    def embed_chunklets(
        self, chunklets: list[str]
    ) -> NDArray[np.float64]:
        """Pooled per-chunklet embeddings — used by ``split_chunks``
        and ``chunk_document`` to drive the partition decision.

        Batches all chunklets into a single padded forward pass, then
        applies the pooling strategy this adapter was constructed
        with (``cls`` / ``mean`` / ``last_token``). Output is
        L2-normalized; SPEC-CHUNK-342 requires nonzero rows.
        """
        if not chunklets:
            return np.zeros((0, self.embedding_dim), dtype=np.float64)

        enc = self._tok(
            chunklets,
            padding=True,
            truncation=True,
            max_length=self.n_ctx,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if self._device != "cpu":
            input_ids = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            h = self._model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        if self.pooling == "cls":
            pooled = h[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "last_token":
            # Index of the last non-padding token per row.
            seq_lens = attention_mask.sum(dim=1) - 1
            seq_lens = seq_lens.clamp(min=0)
            idx = torch.arange(h.shape[0], device=h.device)
            pooled = h[idx, seq_lens]
        else:  # pragma: no cover - validated by Literal type
            raise ValueError(f"unknown pooling: {self.pooling!r}")

        pooled = torch.nn.functional.normalize(pooled, p=2.0, dim=1)
        return cast(
            NDArray[np.float64], pooled.float().cpu().numpy()
        ).astype(np.float64)


if __name__ == "__main__":
    # End-to-end smoke test via chunk_document. BERT-family models
    # use CLS pooling; for BGE-M3 also pass `pooling="cls"`. Use
    # `pooling="mean"` for Sentence-Transformers MPNet/MiniLM
    # families, `pooling="last_token"` for Qwen3-Embedding-* via
    # transformers.
    from fancychunk import chunk_document

    doc = (
        "# Sorting\n\nQuicksort uses a pivot. It partitions around the pivot.\n\n"
        "## Random pivots\n\nThey give expected O(n log n) time.\n"
    )
    emb = HFOffsetEmbedder("bert-base-multilingual-cased", pooling="cls")
    chunks, vectors = chunk_document(doc, emb)
    print(f"chunks: {len(chunks)}")
    print(f"output shape: {vectors.shape}")
    print(f"norms: {np.linalg.norm(vectors, axis=1).round(4)}")
