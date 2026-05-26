"""PooledSegmentEmbedder — the workhorse class behind
``fancychunk.embedders.default()`` / ``fast()`` / ``high_quality()``.

Loads a HuggingFace transformer + tokenizer once, then serves three
use cases:

* :meth:`count_tokens` and :meth:`embed_segment` — the
  :class:`fancychunk.SegmentEmbedder` protocol, used by
  ``embed_with_late_chunking``.
* :meth:`embed_chunklets` — pooled per-chunklet embeddings using the
  model's intended pooling strategy, suitable for direct use as
  ``chunklet_embeddings`` argument to :func:`split_chunks`.

Pooling strategy is per-model. Three are bundled:

* ``last_token`` (Qwen3-Embedding family, decoder-only causal models)
* ``cls`` (BGE-M3 and most BERT-derived encoder models)
* ``mean`` (multilingual-e5-large and most "instructor"-style models)

Matryoshka Representation Learning (MRL) is supported for any model
trained with it: pass ``output_dim`` to truncate pooled embeddings to
that prefix and re-L2-normalize. Truncation applies only to
``embed_chunklets``; ``embed_segment`` returns the model's native
per-token width so late chunking can pool over the full
representation.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray


PoolingMethod = Literal["last_token", "cls", "mean"]


class PooledSegmentEmbedder:
    """Concrete SegmentEmbedder backed by a HuggingFace transformer.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier (e.g. ``"Qwen/Qwen3-Embedding-0.6B"``).
    pooling:
        Which pooling strategy to use for ``embed_chunklets``. The
        factory functions in :mod:`fancychunk.embedders` pre-select
        the right value per model.
    output_dim:
        Truncate pooled embeddings to this many leading dimensions
        and re-L2-normalize (Matryoshka representation learning). Only
        valid for models trained with MRL (the Qwen3-Embedding family
        and a handful of others); silently produces lower-quality
        embeddings if applied to a model not trained for it.
        Default ``None`` keeps the model's native output width.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"`` (default).
    batch_size:
        Batch size for ``embed_chunklets``. Default 32.
    """

    def __init__(
        self,
        model_id: str,
        pooling: PoolingMethod,
        output_dim: int | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.pooling: PoolingMethod = pooling
        self.output_dim = output_dim
        self._device_pref = device
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    # ----- internals -----

    def _pick_device(self) -> str:
        import torch  # type: ignore[import-untyped]

        if self._device_pref != "auto":
            return self._device_pref
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                import torch  # type: ignore[import-untyped]
                from transformers import (  # type: ignore[import-untyped]
                    AutoModel,
                    AutoTokenizer,
                )
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "fancychunk.embedders requires the [embedders] extra. "
                    "Install with: pip install 'fancychunk[embedders]'"
                ) from e

            self._device = self._pick_device()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(
                self.model_id, dtype=torch.float16
            )
            self._model.eval()
            if self._device != "cpu":
                self._model = self._model.to(self._device)
        return self._model, self._tokenizer

    # ----- public attribute -----

    @property
    def n_ctx(self) -> int:
        _, tokenizer = self._ensure_loaded()
        # ``model_max_length`` can be astronomical (1e30) when the
        # tokenizer config doesn't pin it; clamp to a sane default.
        mml = int(getattr(tokenizer, "model_max_length", 4096))
        return min(mml, 32768) if mml > 0 else 4096

    @property
    def embedding_dim(self) -> int:
        """Output dimension of ``embed_chunklets`` (after any MRL truncation)."""
        model, _ = self._ensure_loaded()
        native = int(model.config.hidden_size)
        return self.output_dim if self.output_dim else native

    # ----- SegmentEmbedder protocol -----

    def count_tokens(self, sentences: list[str]) -> list[int]:
        _, tokenizer = self._ensure_loaded()
        return [
            len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences
        ]

    def embed_segment(
        self, sentences: list[str]
    ) -> tuple[NDArray[np.float64], list[int]]:
        """Per-token output + per-sentence counts for late chunking.

        Uses the tokenizer's offset_mapping to align tokens to
        sentences by character offset; special tokens (offset
        ``(0, 0)``) are absorbed into the first/last sentence
        (SPEC-CHUNK-420 option b).
        """
        import torch  # type: ignore[import-untyped]

        model, tokenizer = self._ensure_loaded()
        joined = "".join(sentences)
        enc = tokenizer(
            joined,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.n_ctx,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        if self._device and self._device != "cpu":
            enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
        mat = cast(
            NDArray[np.float64], out.last_hidden_state[0].float().cpu().numpy()
        ).astype(np.float64)

        spans: list[tuple[int, int]] = []
        pos = 0
        for s in sentences:
            spans.append((pos, pos + len(s)))
            pos += len(s)

        counts = [0] * len(sentences)
        for a, b in offsets:
            if a == 0 and b == 0:
                continue
            mid = (a + b) // 2
            for s_idx, (sa, sb) in enumerate(spans):
                if sa <= mid < sb:
                    counts[s_idx] += 1
                    break

        # Absorb leading specials into sentence 0, trailing into the last.
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

    # ----- pooled-chunklet convenience -----

    def embed_chunklets(self, chunklets: list[str]) -> NDArray[np.float64]:
        """Pooled embeddings (one row per chunklet), suitable for
        passing as the ``chunklet_embeddings`` argument to
        :func:`fancychunk.split_chunks`.

        Uses the model's intended pooling strategy (configured at
        construction) and applies MRL truncation when ``output_dim``
        was supplied. Output rows are L2-normalized.
        """
        import torch  # type: ignore[import-untyped]

        if not chunklets:
            return np.zeros((0, self.embedding_dim), dtype=np.float64)

        model, tokenizer = self._ensure_loaded()
        rows: list[NDArray[np.float64]] = []
        for start in range(0, len(chunklets), self.batch_size):
            batch = chunklets[start : start + self.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.n_ctx,
                return_tensors="pt",
            )
            if self._device and self._device != "cpu":
                enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            pooled = _pool(out.last_hidden_state, enc["attention_mask"], self.pooling)
            pooled = torch.nn.functional.normalize(pooled, p=2.0, dim=1)
            rows.append(cast(NDArray[np.float64], pooled.float().cpu().numpy()))

        arr = np.vstack(rows).astype(np.float64)
        if self.output_dim is not None and self.output_dim < arr.shape[1]:
            arr = arr[:, : self.output_dim]
            # Re-normalize after MRL truncation.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1.0, norms)
        return arr


def _pool(hidden_states: Any, attention_mask: Any, method: PoolingMethod) -> Any:
    """Apply the configured pooling. Operates on torch tensors."""
    import torch  # type: ignore[import-untyped]

    if method == "cls":
        return hidden_states[:, 0]
    if method == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    if method == "last_token":
        # Qwen3-Embedding convention: take the last non-padding token
        # in each row. Handles both left- and right-padded inputs.
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        batch_size = hidden_states.shape[0]
        idx = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[idx, seq_lens]
    raise ValueError(f"unknown pooling method: {method}")
