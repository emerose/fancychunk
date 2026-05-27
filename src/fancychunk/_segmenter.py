"""Sentence-segmentation model interface and default implementations.

Two segmenters are bundled:

* :class:`SaTSegmenter` (the default) wraps a Segment Any Text (SaT)
  model from `wtpsplit-lite` and returns per-character boundary
  probabilities exactly as SPEC-CHUNK-106 prescribes. The 408 MB
  ``sat-3l-sm`` weights download lazily on first call so importing
  ``fancychunk`` stays cheap.
* :func:`punctuation_segmenter` is a no-dependencies fallback that
  marks ``.``/``!``/``?`` followed by whitespace or end-of-document.

Either is a valid SentenceSegmenter; callers may pass their own
through the keyword-only ``segmenter`` parameter on
``split_sentences``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np

from ._typing import Vector
from .errors import SegmenterError, ValidationError

if TYPE_CHECKING:
    from wtpsplit_lite import SaT


class SentenceSegmenter(Protocol):
    """Callable mapping a document to a per-character boundary probability.

    The returned array has length ``len(document)`` and dtype
    convertible to ``float64``.
    """

    def __call__(self, document: str) -> Vector: ...


@runtime_checkable
class BatchSentenceSegmenter(SentenceSegmenter, Protocol):
    """A :class:`SentenceSegmenter` that also exposes a batched call.

    ``predict_proba_batch(documents)`` returns one boundary-probability
    vector per input, in input order. Implementations may exploit
    cross-document batching (shared model forward passes, GPU
    utilisation) for substantially better throughput than calling the
    single-document ``__call__`` N times.

    ``chunk_documents`` opts into the batched path when its
    ``segmenter_batch_size`` argument is set and the resolved segmenter
    satisfies this protocol.
    """

    def predict_proba_batch(self, documents: list[str]) -> list[Vector]: ...


_TERMINATORS = frozenset({".", "!", "?"})
_DEFAULT_BOUNDARY_PROB = 0.9


def punctuation_segmenter(document: str) -> Vector:
    """Rule-based fallback segmenter.

    Assigns ``_DEFAULT_BOUNDARY_PROB`` at every character that is a
    sentence-final terminator (``.!?``) followed by whitespace or end
    of document; ``0.0`` elsewhere. Crude but adequate for tests that
    don't need real model output.
    """
    n = len(document)
    probs: Vector = np.zeros(n, dtype=np.float64)
    for i, ch in enumerate(document):
        if ch in _TERMINATORS:
            if i == n - 1 or document[i + 1].isspace():
                probs[i] = _DEFAULT_BOUNDARY_PROB
    return probs


_DEFAULT_SAT_MODEL = "sat-3l-sm"


def _providers_for_device(device: str) -> list[str]:
    """Resolve a fancychunk ``device`` string to onnxruntime providers.

    ``"cpu"`` → CPU-only; ``"cuda"``/``"gpu"`` → CUDA with CPU fallback
    (so a misconfigured GPU box still runs); ``"auto"`` → defer to
    wtpsplit-lite's default (GPU if available, else CPU). Anything
    else raises :class:`ValidationError`.
    """
    key = device.lower()
    if key == "cpu":
        return ["CPUExecutionProvider"]
    if key in ("cuda", "gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if key == "auto":
        return []  # sentinel — caller treats as "let wtpsplit auto-detect"
    raise ValidationError(
        f"unknown device {device!r}; expected one of "
        "'auto', 'cpu', 'cuda', 'gpu'"
    )


class SaTSegmenter:
    """SPEC-CHUNK-106 segmenter backed by wtpsplit-lite's SaT model.

    The 408 MB ``sat-3l-sm`` weights are downloaded by Hugging Face on
    first use (subsequent calls use the cache); the import itself is
    cheap because the model is only loaded on the first
    ``__call__``. Instances are thread-safe — lazy loading is
    serialized so concurrent first callers don't double-download; the
    ONNX ``predict_proba`` itself is reentrant after load and runs
    unlocked.

    Args:
        model_name: SaT checkpoint to load (defaults to ``sat-3l-sm``).
        device: ``"auto"`` (default) lets wtpsplit-lite pick the best
            available ONNX execution provider — typically
            ``CUDAExecutionProvider`` when ``onnxruntime-gpu`` is
            installed and a GPU is visible, else CPU. ``"cpu"`` /
            ``"cuda"`` force a specific path; ``"cuda"`` keeps CPU as a
            fallback in the provider list so a misconfigured GPU box
            still runs.
        ort_providers: Power-user escape hatch — pass an explicit
            onnxruntime provider list. Mutually exclusive with
            ``device``.
        ort_kwargs: Forwarded to ``onnxruntime.InferenceSession``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_SAT_MODEL,
        *,
        device: str = "auto",
        ort_providers: list[str] | None = None,
        ort_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if ort_providers is not None and device != "auto":
            raise ValidationError(
                "pass either device= or ort_providers=, not both"
            )
        self.model_name: str = model_name
        self.device: str = device
        # Resolve the providers eagerly so misspellings fail at
        # construction, not on first call.
        if ort_providers is not None:
            self._ort_providers: list[str] | None = list(ort_providers)
        else:
            providers = _providers_for_device(device)
            self._ort_providers = providers or None
        self._ort_kwargs: dict[str, Any] | None = ort_kwargs
        self._sat: SaT | None = None
        self._load_lock: threading.Lock = threading.Lock()

    @property
    def ort_providers(self) -> list[str] | None:
        """Return the resolved provider list (``None`` = auto-detect)."""
        return None if self._ort_providers is None else list(self._ort_providers)

    def _ensure_loaded(self) -> SaT:
        # Double-checked locking: the unlocked read is safe under
        # CPython's GIL (attribute reads are atomic), and the lock only
        # matters for the not-yet-loaded race.
        if self._sat is None:
            with self._load_lock:
                if self._sat is None:
                    # Local import keeps ``import fancychunk``
                    # lightweight when the SaT weights aren't cached.
                    from wtpsplit_lite import SaT as _SaT

                    kwargs: dict[str, Any] = {}
                    if self._ort_providers is not None:
                        kwargs["ort_providers"] = list(self._ort_providers)
                    if self._ort_kwargs is not None:
                        kwargs["ort_kwargs"] = dict(self._ort_kwargs)
                    self._sat = _SaT(self.model_name, **kwargs)
        assert self._sat is not None
        return self._sat

    def __call__(self, document: str) -> Vector:
        sat = self._ensure_loaded()
        raw = sat.predict_proba(document)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != len(document):
            raise SegmenterError(
                f"SaT returned shape {arr.shape}; expected ({len(document)},)"
            )
        return arr

    def predict_proba_batch(self, documents: list[str]) -> list[Vector]:
        """Compute boundary probabilities for ``documents`` in one batch.

        Equivalent in output to ``[seg(d) for d in documents]`` but
        exploits wtpsplit-lite's native cross-document batching: all
        non-empty documents share one ONNX forward pass per inner
        batch, which is several times faster on both CPU and GPU than
        the equivalent loop.

        Empty / whitespace-only inputs are passed through as
        zero-length (``len(d) == 0``) or zero-filled (whitespace)
        vectors of the right shape — :func:`split_sentences` short-
        circuits both cases before reaching the segmenter output.
        """
        if not documents:
            return []
        out: list[Vector] = [
            np.zeros(len(d), dtype=np.float64) for d in documents
        ]
        nonempty_idx = [i for i, d in enumerate(documents) if d.strip()]
        if not nonempty_idx:
            return out
        nonempty_docs = [documents[i] for i in nonempty_idx]

        sat = self._ensure_loaded()
        for idx, raw in zip(nonempty_idx, sat.predict_proba(nonempty_docs)):
            arr = np.asarray(raw, dtype=np.float64)
            expected_len = len(documents[idx])
            if arr.ndim != 1 or arr.shape[0] != expected_len:
                raise SegmenterError(
                    f"SaT returned shape {arr.shape}; expected ({expected_len},)"
                )
            out[idx] = arr
        return out


# Module-level default singleton (lazy weight load on first call).
_default_segmenter: SaTSegmenter | None = None
_default_segmenter_lock = threading.Lock()


def get_default_segmenter() -> SaTSegmenter:
    """Return the process-wide default segmenter (the SaT singleton).

    Thread-safe: concurrent callers on a cold process see exactly one
    ``SaTSegmenter`` construction.
    """
    global _default_segmenter
    if _default_segmenter is None:
        with _default_segmenter_lock:
            if _default_segmenter is None:
                _default_segmenter = SaTSegmenter()
    return _default_segmenter


def make_segmenter(
    segmenter: SentenceSegmenter | None,
) -> Callable[[str], Vector]:
    """Resolve ``segmenter`` to a callable, defaulting to SaT."""
    if segmenter is None:
        return get_default_segmenter()
    return segmenter


def precomputed_segmenter(probas: Vector) -> SentenceSegmenter:
    """Wrap a precomputed probability vector as a SentenceSegmenter.

    Used by :func:`chunk_documents` when the segmenter has been run
    ahead of the per-document pipeline — each document's slot in the
    batched output becomes a single-shot segmenter that just hands
    the precomputed vector back to :func:`split_sentences`.

    Advanced callers can use this directly to inject cached or
    externally-computed boundary probabilities into the pipeline:

    ::

        probas = my_cache.get(doc_hash)
        sentences = split_sentences(doc, segmenter=precomputed_segmenter(probas))
    """

    def _seg(document: str) -> Vector:
        del document
        return probas

    return _seg
