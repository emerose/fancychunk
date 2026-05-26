"""Reference SegmentEmbedder over MLX-format Qwen3-Embedding.

Tested with ``mlx-community/Qwen3-Embedding-8B-mxfp8`` on an M2 Mac
(24 GB unified memory). Quantized builds keep the resident size around
9-10 GB, comfortable inside 24 GB.

Tokenization-alignment strategy: **sentinel-token**. We join the input
texts (chunks plus any heading-stack prepend the library passes us)
with ``§`` (SECTION SIGN, U+00A7) — picked because mBERT and Qwen3's
tokenizer both treat it as a single atomic token across positions. If
you switch base models, probe your tokenizer for a sentinel that
doesn't subword-merge; ``⊕`` / ``¶`` / ``§`` are good candidates.

Specials: Qwen3-Embedding wraps each input with ``<bos>``/``<eos>``
roughly. The leading special goes to text 0; the trailing to the last
text (SPEC-CHUNK-420 option b).

Install:
    pip install mlx mlx-embeddings

Usage:
    from examples.embedders.qwen3_mlx import Qwen3MLXEmbedder
    from fancychunk import (
        embed_with_late_chunking,
        split_chunklets,
        split_chunks,
        split_sentences,
    )

    embedder = Qwen3MLXEmbedder("mlx-community/Qwen3-Embedding-8B-mxfp8")
    sentences = split_sentences(my_document, max_len=2048)
    chunklets = split_chunklets(sentences, max_size=2048)
    chunks = split_chunks(chunklets, max_size=2048)
    matrix = embed_with_late_chunking(chunks, embedder)
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx  # type: ignore[import-untyped]
import numpy as np
from mlx_embeddings.utils import load  # type: ignore[import-untyped]
from numpy.typing import NDArray

_SENTINEL = "§"


class Qwen3MLXEmbedder:
    """Late-chunking adapter for MLX-format Qwen3-Embedding models."""

    n_ctx = 4096  # Qwen3-Embedding's context window per the HF config.

    def __init__(self, model_id: str = "mlx-community/Qwen3-Embedding-8B-mxfp8") -> None:
        model, tokenizer = load(model_id)
        model.eval()
        self._model: Any = model
        self._tok: Any = tokenizer
        # Cache the sentinel token id(s). One id is the common case;
        # we collect a set to tolerate context-dependent variants.
        self._sentinel_ids: set[int] = self._discover_sentinel_ids()

    @property
    def embedding_dim(self) -> int:
        """Native hidden size of the loaded model."""
        return int(self._model.config.hidden_size)

    # ----- SegmentEmbedder contract -----

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Per-text isolated token count — approximate, used by
        fancychunk only for segment budget planning. Subword merges
        across boundaries may shift actual counts by ±1; the
        largest-remainder safety net absorbs that."""
        return [
            len(self._tok.encode(s, add_special_tokens=False)) for s in texts
        ]

    def embed_segment(
        self, texts: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        joined = _SENTINEL.join(texts)
        ids = self._tok.encode(joined, add_special_tokens=True)
        x = mx.array([ids], dtype=mx.int32)
        out = self._model(x)
        h = out.last_hidden_state
        mx.eval(h)
        mat = cast(NDArray[np.float64], np.asarray(h.astype(mx.float32))[0]).astype(
            np.float64
        )

        # Locate sentinel-token positions and derive per-text counts.
        positions = [i for i, t in enumerate(ids) if t in self._sentinel_ids]
        expected_separators = len(texts) - 1
        if len(positions) != expected_separators:
            # Tokenizer didn't preserve every sentinel — pick a
            # different character (or use the offset-based method).
            raise RuntimeError(
                f"sentinel {_SENTINEL!r} produced {len(positions)} positions; "
                f"expected {expected_separators}. Pick a tokenizer-stable "
                f"sentinel for this model."
            )

        counts: list[int] = []
        last = -1
        for pos in positions:
            counts.append(pos - last)
            last = pos
        # Last text: from after the last sentinel to end-of-sequence.
        counts.append(len(ids) - 1 - last)
        return mat, counts

    # ----- ChunkletEmbedder contract -----

    def embed_chunklets(
        self, chunklets: list[str]
    ) -> NDArray[np.float64]:
        """Pooled per-chunklet embeddings — used by ``split_chunks``
        and ``chunk_document`` to drive the partition decision.

        Qwen3-Embedding uses **last-token pooling**: the hidden state
        at the final position is the sentence embedding. This
        reference adapter does one forward pass per chunklet for
        clarity; production code would group by length and pad-batch
        for throughput.
        """
        if not chunklets:
            return np.zeros((0, self.embedding_dim), dtype=np.float64)
        rows: list[NDArray[np.float64]] = []
        for chunklet in chunklets:
            ids = self._tok.encode(chunklet, add_special_tokens=True)
            x = mx.array([ids], dtype=mx.int32)
            out = self._model(x)
            h = out.last_hidden_state
            mx.eval(h)
            # Last-token pooling: take the final position's hidden
            # state. (Qwen3-Embedding's training objective puts the
            # pooled vector here.)
            vec = cast(
                NDArray[np.float64],
                np.asarray(h[0, -1].astype(mx.float32)),
            ).astype(np.float64)
            rows.append(vec)
        arr = np.stack(rows)
        # L2-normalize; SPEC-CHUNK-342 requires nonzero rows.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    # ----- internals -----

    def _discover_sentinel_ids(self) -> set[int]:
        """Probe the tokenizer for the id(s) corresponding to the sentinel.

        Tokenizes a small probe string with the sentinel in several
        positions and collects every id that decodes back to a string
        containing the sentinel character.
        """
        probe = f"a{_SENTINEL}b{_SENTINEL}c"
        probe_ids = self._tok.encode(probe, add_special_tokens=True)
        ids: set[int] = set()
        for tok in probe_ids:
            try:
                text = self._tok.decode([tok], skip_special_tokens=False)
            except Exception:
                continue
            if _SENTINEL in text:
                ids.add(tok)
        if not ids:
            raise RuntimeError(
                f"tokenizer did not preserve sentinel {_SENTINEL!r} as a "
                f"single decodable token; pick a different sentinel."
            )
        return ids


if __name__ == "__main__":
    # End-to-end smoke test via chunk_document. The adapter now
    # implements both halves of the protocol (embed_chunklets for
    # the split decision, embed_segment + count_tokens for late
    # chunking), so it works as a full Embedder.
    from fancychunk import chunk_document

    doc = (
        "# Sorting\n\nQuicksort uses a pivot. It partitions around the pivot.\n\n"
        "## Random pivots\n\nThey give expected O(n log n) time.\n"
    )
    emb = Qwen3MLXEmbedder()
    chunks, vectors = chunk_document(doc, emb)
    print(f"chunks: {len(chunks)}")
    print(f"output shape: {vectors.shape}")
    print(f"norms: {np.linalg.norm(vectors, axis=1).round(4)}")
