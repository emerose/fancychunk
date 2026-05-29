"""Sentence-segmentation model interface and default implementations.

Two segmenters are bundled:

* :class:`SaTSegmenter` (the default) wraps a Segment Any Text (SaT)
  model from `wtpsplit-lite` and returns per-character boundary
  probabilities exactly as SPEC-CHUNK-106 prescribes. The default
  checkpoint is ``sat-9l-sm`` (weights download lazily on first call so
  importing ``fancychunk`` stays cheap). The bundled checkpoints trade
  speed for quality — ``sat-3l-sm`` is fastest but mis-segments some
  scientific-prose constructs (abbreviation references like
  ``Tab. TABREF21``, years before a capitalised word like
  ``SemEval-2014 Task``); ``sat-9l-sm`` (default) fixes those and
  tracks ``sat-12l-sm`` closely at ~1.3× the throughput; ``sat-12l-sm``
  is highest quality. See ``fancychunk.segmenters`` for the factories
  and ``benchmarks/sat-model-selection.md`` for the data.
* :func:`punctuation_segmenter` is a no-dependencies fallback that
  marks ``.``/``!``/``?`` followed by whitespace or end-of-document.

Either is a valid SentenceSegmenter; callers may pass their own
through the keyword-only ``segmenter`` parameter on
``split_sentences``.
"""

from __future__ import annotations

import os
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    cast,
    runtime_checkable,
)

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
    single-document ``__call__`` N times. Callers segmenting many
    documents at once can invoke it directly; ``wants_batching()`` is a
    hint for whether batching is expected to pay off on the resolved
    execution provider.
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


_DEFAULT_SAT_MODEL = "sat-9l-sm"

# ``predict_proba`` inference parameters. wtpsplit-lite's defaults
# (``stride=256, block_size=512, weighting="uniform"``) average every
# overlapping window equally, so a character predicted near a window
# edge — with truncated left/right context — contributes as much as one
# predicted at a window's centre. That produces *context-sensitive*
# boundary artifacts: the same clause scores differently depending on
# where the sliding window happens to fall (e.g. the period of an
# abbreviation, or a year before a capitalised word, can spuriously
# cross the threshold in one document but not another).
#
# ``weighting="hat"`` weights each window's predictions by a triangular
# function so centre-of-window (full-context) predictions dominate, and
# the smaller ``block_size``/``stride`` give every character more
# windows to be near-centre in. These are the values wtpsplit-lite's
# own maintainers ship in raglite. Empirically this removes the
# window-edge artifacts (e.g. a full-document ``Tab.`` period drops from
# 0.345 to 0.039) without disturbing genuine high-confidence boundaries.
_SAT_PREDICT_KWARGS: dict[str, Any] = {
    "stride": 128,
    "block_size": 256,
    "weighting": "hat",
}

_FAST_POSTPROCESS_DISABLE_ENV = "FANCYCHUNK_DISABLE_SAT_FAST_POSTPROCESS"


def _fast_token_to_char_probs(
    text: str,
    tokens: list[str],
    token_logits: np.ndarray,
    tokenizer: Any,
    offsets_mapping: list[tuple[int, int]],
) -> np.ndarray:
    """Vectorised replacement for ``wtpsplit_lite._utils.token_to_char_probs``.

    The upstream implementation is a Python ``for`` loop over every
    non-special token (~400 iterations per 1,500-char doc), which on
    a CUDA box dominates the SaT post-process and ends up taking
    nearly as much wall-clock as the actual ORT forward pass. This
    version performs the same projection — last-character of each
    non-special token's span gets the token's logit row, everything
    else stays at ``-inf`` — but in two numpy operations.

    Bounds-safe: token spans whose ``end`` falls outside
    ``[0, len(text))`` are dropped. The upstream version silently
    wraps negatively (``char_probs[-1]``) on ``end == 0``; for the
    SaT models that exists is benign on realistic text but worth
    being explicit about.
    """
    n = len(text)
    cols = token_logits.shape[1]
    out = np.full((n, cols), -np.inf)
    specials = (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token)
    # offsets_mapping is a list of (start, end) tuples; convert in one shot.
    om = np.asarray(offsets_mapping, dtype=np.int64)
    keep_pylist = [t not in specials for t in tokens]
    keep = np.fromiter(keep_pylist, dtype=bool, count=len(keep_pylist))
    # Upstream guards with ``idx < len(offsets_mapping)``; mirror it.
    m = min(len(keep), om.shape[0])
    if m == 0:
        return out
    keep = keep[:m]
    om = om[:m]
    valid_idx = np.flatnonzero(keep)
    if valid_idx.size == 0:
        return out
    ends = om[valid_idx, 1] - 1
    in_bounds = (ends >= 0) & (ends < n)
    if not in_bounds.any():
        return out
    out[ends[in_bounds]] = token_logits[valid_idx[in_bounds]]
    return out


_fast_postprocess_installed: bool = False
_fast_postprocess_lock = threading.Lock()


def _install_fast_postprocess() -> None:
    """Replace ``wtpsplit_lite._sat.token_to_char_probs`` with the
    vectorised variant.

    Idempotent: only the first caller actually patches. The patch
    targets the binding inside ``_sat`` (where ``predict_proba``
    looks it up), not ``_utils`` (where it's defined) — anything
    that imported the helper before this runs would otherwise keep
    the slow version.

    Set ``FANCYCHUNK_DISABLE_SAT_FAST_POSTPROCESS=1`` to opt out;
    useful as an escape hatch if a future wtpsplit-lite release
    changes the function and our shim becomes stale before we can
    update the pin.
    """
    global _fast_postprocess_installed
    if _fast_postprocess_installed:
        return
    if os.environ.get(_FAST_POSTPROCESS_DISABLE_ENV):
        # Still flip the flag so we don't keep re-checking.
        with _fast_postprocess_lock:
            _fast_postprocess_installed = True
        return
    with _fast_postprocess_lock:
        if _fast_postprocess_installed:
            return
        try:
            from wtpsplit_lite import _sat as _wtpsplit_sat
        except ImportError:
            # Test suites stub ``wtpsplit_lite`` in ``sys.modules`` to
            # avoid loading 400 MB of weights; the stub doesn't expose
            # the private ``_sat`` submodule. Treat as installed so we
            # don't keep retrying.
            _fast_postprocess_installed = True
            return
        # setattr instead of attribute assignment so pyright's strict
        # ``reportPrivateImportUsage`` doesn't flag the rebinding of an
        # un-``__all__``-exported name on a private submodule.
        setattr(  # noqa: B010
            _wtpsplit_sat, "token_to_char_probs", _fast_token_to_char_probs
        )
        _fast_postprocess_installed = True


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

    The ``sat-9l-sm`` weights are downloaded by Hugging Face on first
    use (subsequent calls use the cache); the import itself is cheap
    because the model is only loaded on the first ``__call__``.
    Instances are thread-safe — lazy loading is serialized so concurrent
    first callers don't double-download; the ONNX ``predict_proba``
    itself is reentrant after load and runs unlocked.

    Args:
        model_name: SaT checkpoint to load (defaults to ``sat-9l-sm``;
            see ``fancychunk.segmenters`` for ``sat-3l-sm`` /
            ``sat-12l-sm``).
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

    def wants_batching(self) -> bool:
        """Whether batched inference is expected to be faster than
        per-document calls for this segmenter's resolved device.

        Returns ``True`` iff a GPU execution provider is what will (or
        is most likely to) run the forward pass. CPU EPs see no batch
        win — forward FLOPs scale linearly with batch size — so the
        answer is ``False`` for ``device="cpu"`` and for boxes without
        ``onnxruntime-gpu`` installed.

        Decision rule, in order of precedence:

        1. If ``ort_providers`` was explicitly configured (via
           ``device="cuda"/"cpu"`` or ``ort_providers=[...]``), look
           at the list — any GPU EP wins.
        2. Otherwise (``device="auto"``), peek at
           ``onnxruntime.get_available_providers()`` and check for a
           GPU EP.

        This is a heuristic, not a guarantee — if the actual session
        creation falls back from CUDA to CPU (e.g. cuDNN missing),
        the ``True`` answer can lie. The check is cheap (no model
        load), so callers are free to consult it on every dispatch.
        """
        providers: list[str] | None = self._ort_providers
        if providers is None:
            try:
                import onnxruntime as ort  # type: ignore[import-untyped]
            except ImportError:
                return False
            providers = cast(
                list[str],
                ort.get_available_providers(),  # pyright: ignore[reportUnknownMemberType]
            )
        gpu_eps = {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "MIGraphXExecutionProvider",
        }
        return any(p in gpu_eps for p in providers)

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

                    _install_fast_postprocess()
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
        raw = sat.predict_proba(document, **_SAT_PREDICT_KWARGS)
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
        for idx, raw in zip(
            nonempty_idx, sat.predict_proba(nonempty_docs, **_SAT_PREDICT_KWARGS)
        ):
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
