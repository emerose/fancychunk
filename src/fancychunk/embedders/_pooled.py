"""PooledSegmentEmbedder — backs ``fancychunk.embedders.{fastest,
fast, medium, high}``.

Loads a HuggingFace transformer (or an MLX-format build via
``mlx_embeddings``) once, then serves three use cases:

* :meth:`count_tokens` and :meth:`embed_segment` — the
  :class:`fancychunk.SegmentEmbedder` protocol, used by
  ``embed_with_late_chunking``.
* :meth:`embed_chunklets` — pooled per-chunklet embeddings using the
  model's intended pooling strategy, suitable for direct use as
  ``chunklet_embeddings`` argument to :func:`split_chunks`.

Two backends:

* ``"torch"`` — HuggingFace ``transformers`` + PyTorch. Works
  everywhere; picks CUDA / MPS / CPU automatically.
* ``"mlx"`` — Apple's MLX via ``mlx_embeddings``. Apple Silicon only,
  typically 2-4× faster than torch + MPS on the same hardware. Used
  automatically when the platform supports it and an MLX build of the
  model is available.

Pooling strategy is per-model (``"last_token"`` for Qwen3-Embedding,
``"cls"`` for BGE-M3, ``"mean"`` for some encoder models). The
factory functions in :mod:`fancychunk.embedders` pre-select the
right pooling per model.

Matryoshka Representation Learning (MRL) truncation applies only to
``embed_chunklets`` (the pooled output) — ``embed_segment`` always
returns the model's native per-token width so late chunking pools
over the full representation.
"""

from __future__ import annotations

import sys
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray


PoolingMethod = Literal["last_token", "cls", "mean"]
Backend = Literal["torch", "mlx", "auto"]


class PooledSegmentEmbedder:
    """Concrete SegmentEmbedder backed by a HuggingFace transformer
    (torch) or an MLX-format build (mlx_embeddings).

    Parameters
    ----------
    model_id:
        HuggingFace model identifier. For MLX, use the ``mlx-community/``
        flavour of the desired model.
    pooling:
        Which pooling strategy to use for ``embed_chunklets``. The
        factory functions in :mod:`fancychunk.embedders` pre-select
        the right value per model.
    output_dim:
        Truncate pooled embeddings to this many leading dimensions
        and re-L2-normalize (Matryoshka Representation Learning). Only
        valid for models trained with MRL; default ``None`` keeps the
        model's native output width.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"`` (torch backend
        only; MLX always runs on the GPU).
    batch_size:
        Batch size for ``embed_chunklets``.
    backend:
        ``"torch"``, ``"mlx"``, or ``"auto"`` (the default). ``"auto"``
        picks ``"mlx"`` on Apple Silicon when ``mlx_embeddings`` is
        importable AND ``model_id`` is recognized as an MLX build
        (i.e. the namespace starts with ``mlx-community/``).
    """

    def __init__(
        self,
        model_id: str,
        pooling: PoolingMethod,
        output_dim: int | None = None,
        device: str = "auto",
        batch_size: int = 32,
        backend: Backend = "auto",
    ) -> None:
        self.model_id = model_id
        self.pooling: PoolingMethod = pooling
        self.output_dim = output_dim
        self._device_pref = device
        self.batch_size = batch_size
        self._backend_pref: Backend = backend
        self._backend: Literal["torch", "mlx"] | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    # ----- backend selection -----

    def _resolve_backend(self) -> Literal["torch", "mlx"]:
        if self._backend_pref == "mlx":
            return "mlx"
        if self._backend_pref == "torch":
            return "torch"
        # "auto": prefer MLX on Apple Silicon when (a) mlx_embeddings
        # is importable and (b) model_id is an MLX-community build.
        if sys.platform != "darwin":
            return "torch"
        if not self.model_id.startswith("mlx-community/"):
            return "torch"
        try:
            import mlx_embeddings  # noqa: F401
        except ImportError:
            return "torch"
        return "mlx"

    def _pick_torch_device(self) -> str:
        import torch  # type: ignore[import-untyped]

        if self._device_pref != "auto":
            return self._device_pref
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ----- loading -----

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            self._backend = self._resolve_backend()
            if self._backend == "mlx":
                self._load_mlx()
            else:
                self._load_torch()
        return self._model, self._tokenizer

    def _load_torch(self) -> None:
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

        self._device = self._pick_torch_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, dtype=torch.float16)
        self._model.eval()
        if self._device != "cpu":
            self._model = self._model.to(self._device)

    def _load_mlx(self) -> None:
        try:
            from mlx_embeddings.utils import load  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "MLX backend requires mlx_embeddings; install with "
                "pip install 'fancychunk[embedders]' on macOS."
            ) from e

        self._model, tokenizer_wrapper = load(self.model_id)
        self._model.eval()
        # Unwrap the HF tokenizer for offset_mapping support.
        self._tokenizer = tokenizer_wrapper._tokenizer  # noqa: SLF001
        self._device = "mlx"

    # ----- public attributes -----

    @property
    def n_ctx(self) -> int:
        _, tokenizer = self._ensure_loaded()
        mml = int(getattr(tokenizer, "model_max_length", 4096))
        return min(mml, 32768) if mml > 0 else 4096

    @property
    def embedding_dim(self) -> int:
        """Output dimension of ``embed_chunklets`` (after any MRL truncation)."""
        model, _ = self._ensure_loaded()
        # Both torch and mlx_embeddings models expose ``.config.hidden_size``.
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
        model, tokenizer = self._ensure_loaded()
        joined = "".join(sentences)
        enc = tokenizer(
            joined,
            return_offsets_mapping=True,
            return_tensors="np",
            add_special_tokens=True,
            truncation=True,
            max_length=self.n_ctx,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        if self._backend == "mlx":
            mat = self._forward_mlx_per_token(ids, attention_mask)
        else:
            mat = self._forward_torch_per_token(ids, attention_mask)

        counts = _align_counts(sentences, offsets)
        return mat, counts

    def _forward_torch_per_token(
        self, ids: Any, attention_mask: Any
    ) -> NDArray[np.float64]:
        import torch  # type: ignore[import-untyped]

        ids_t = torch.tensor(ids)
        am_t = torch.tensor(attention_mask) if attention_mask is not None else None
        if self._device and self._device != "cpu":
            ids_t = ids_t.to(self._device)
            if am_t is not None:
                am_t = am_t.to(self._device)
        with torch.no_grad():
            out = self._model(input_ids=ids_t, attention_mask=am_t)
        return cast(
            NDArray[np.float64],
            out.last_hidden_state[0].float().cpu().numpy(),
        ).astype(np.float64)

    def _forward_mlx_per_token(
        self, ids: Any, attention_mask: Any
    ) -> NDArray[np.float64]:
        import mlx.core as mx  # type: ignore[import-untyped]

        ids_mx = mx.array(ids, dtype=mx.int32)
        am_mx = (
            mx.array(attention_mask, dtype=mx.int32)
            if attention_mask is not None
            else None
        )
        out = self._model(ids_mx, attention_mask=am_mx)
        h = out.last_hidden_state
        mx.eval(h)
        return np.asarray(h.astype(mx.float32))[0].astype(np.float64)

    # ----- pooled-chunklet convenience -----

    def embed_chunklets(self, chunklets: list[str]) -> NDArray[np.float64]:
        """Pooled embeddings (one row per chunklet), suitable for
        passing as the ``chunklet_embeddings`` argument to
        :func:`fancychunk.split_chunks`."""
        if not chunklets:
            return np.zeros((0, self.embedding_dim), dtype=np.float64)

        _, tokenizer = self._ensure_loaded()
        rows: list[NDArray[np.float64]] = []
        for start in range(0, len(chunklets), self.batch_size):
            batch = chunklets[start : start + self.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.n_ctx,
                return_tensors="np",
            )
            if self._backend == "mlx":
                pooled_np = self._forward_mlx_pooled(
                    enc["input_ids"], enc["attention_mask"]
                )
            else:
                pooled_np = self._forward_torch_pooled(
                    enc["input_ids"], enc["attention_mask"]
                )
            rows.append(pooled_np.astype(np.float64))

        arr = np.vstack(rows)
        if self.output_dim is not None and self.output_dim < arr.shape[1]:
            arr = arr[:, : self.output_dim]
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1.0, norms)
        return arr

    def _forward_torch_pooled(
        self, ids: Any, attention_mask: Any
    ) -> NDArray[np.float64]:
        import torch  # type: ignore[import-untyped]

        ids_t = torch.tensor(ids)
        am_t = torch.tensor(attention_mask)
        if self._device and self._device != "cpu":
            ids_t = ids_t.to(self._device)
            am_t = am_t.to(self._device)
        with torch.no_grad():
            out = self._model(input_ids=ids_t, attention_mask=am_t)
        pooled = _pool_torch(out.last_hidden_state, am_t, self.pooling)
        pooled = torch.nn.functional.normalize(pooled, p=2.0, dim=1)
        return cast(NDArray[np.float64], pooled.float().cpu().numpy())

    def _forward_mlx_pooled(
        self, ids: Any, attention_mask: Any
    ) -> NDArray[np.float64]:
        import mlx.core as mx  # type: ignore[import-untyped]

        ids_mx = mx.array(ids, dtype=mx.int32)
        am_mx = mx.array(attention_mask, dtype=mx.int32)
        out = self._model(ids_mx, attention_mask=am_mx)
        # mlx_embeddings returns ``text_embeds`` already pooled +
        # L2-normalized using the model's intended strategy. Cast to
        # numpy and we're done.
        mx.eval(out.text_embeds)
        return np.asarray(out.text_embeds.astype(mx.float32))


# ---------------------------------------------------------------------------
# Helpers shared between backends.
# ---------------------------------------------------------------------------


def _align_counts(sentences: list[str], offsets: list[tuple[int, int]]) -> list[int]:
    """Map per-token offsets back to per-sentence counts. Special
    tokens (offset (0,0)) are absorbed into the first/last sentence
    (SPEC-CHUNK-420 option b).
    """
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
    return counts


def _pool_torch(hidden_states: Any, attention_mask: Any, method: PoolingMethod) -> Any:
    """Torch pooling. (MLX models pool internally via ``out.text_embeds``.)"""
    import torch  # type: ignore[import-untyped]

    if method == "cls":
        return hidden_states[:, 0]
    if method == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    if method == "last_token":
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        batch_size = hidden_states.shape[0]
        idx = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[idx, seq_lens]
    raise ValueError(f"unknown pooling method: {method}")
