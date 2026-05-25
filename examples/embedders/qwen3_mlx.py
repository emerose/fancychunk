"""Reference SegmentEmbedder over MLX-format Qwen3-Embedding.

Tested with ``mlx-community/Qwen3-Embedding-8B-mxfp8`` on an M2 Mac
(24 GB unified memory). Quantized builds keep the resident size around
9-10 GB, comfortable inside 24 GB.

Tokenization-alignment strategy: **sentinel-token**. We join sentences
with ``§`` (SECTION SIGN, U+00A7) — picked because mBERT and Qwen3's
tokenizer both treat it as a single atomic token across positions. If
you switch base models, probe your tokenizer for a sentinel that
doesn't subword-merge; ``⊕`` / ``¶`` / ``§`` are good candidates.

Specials: Qwen3-Embedding wraps each input with ``<bos>``/``<eos>``
roughly. The leading special goes to sentence 0; the trailing to the
last sentence (SPEC-CHUNK-420 option b).

Install:
    pip install mlx mlx-embeddings

Usage:
    from examples.embedders.qwen3_mlx import Qwen3MLXEmbedder
    from fancychunk import embed_with_late_chunking, split_sentences

    embedder = Qwen3MLXEmbedder("mlx-community/Qwen3-Embedding-8B-mxfp8")
    sentences = split_sentences(my_document, max_len=2048)
    matrix = embed_with_late_chunking(sentences, embedder)
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

    # ----- SegmentEmbedder contract -----

    def count_tokens(self, sentences: list[str]) -> list[int]:
        """Per-sentence isolated token count — approximate, used by
        fancychunk only for segment budget planning. Subword merges
        across boundaries may shift actual counts by ±1; the
        largest-remainder safety net absorbs that."""
        return [
            len(self._tok.encode(s, add_special_tokens=False)) for s in sentences
        ]

    def embed_segment(
        self, sentences: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        joined = _SENTINEL.join(sentences)
        ids = self._tok.encode(joined, add_special_tokens=True)
        x = mx.array([ids], dtype=mx.int32)
        out = self._model(x)
        h = out.last_hidden_state
        mx.eval(h)
        mat = cast(NDArray[np.float64], np.asarray(h.astype(mx.float32))[0]).astype(
            np.float64
        )

        # Locate sentinel-token positions and derive per-sentence counts.
        positions = [i for i, t in enumerate(ids) if t in self._sentinel_ids]
        expected_separators = len(sentences) - 1
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
        # Last sentence: from after the last sentinel to end-of-sequence.
        counts.append(len(ids) - 1 - last)
        return mat, counts

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
    # End-to-end smoke test.
    from fancychunk import embed_with_late_chunking, split_sentences

    doc = (
        "# Sorting\n\nQuicksort uses a pivot. It partitions around the pivot.\n\n"
        "## Random pivots\n\nThey give expected O(n log n) time.\n"
    )
    sents = split_sentences(doc, max_len=2048)
    print(f"sentences: {len(sents)}")
    emb = Qwen3MLXEmbedder()
    out = embed_with_late_chunking(sents, emb)
    print(f"output shape: {out.shape}")
    print(f"norms: {np.linalg.norm(out, axis=1).round(4)}")
