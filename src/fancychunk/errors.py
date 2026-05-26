"""Exception hierarchy for fancychunk."""

from __future__ import annotations


class FancyChunkError(Exception):
    """Base class for all fancychunk errors."""


class ValidationError(FancyChunkError):
    """Caller-fixable input violates a documented precondition."""


class UnsplittableDocumentError(ValidationError):
    """No partition of the document satisfies the configured length constraints.

    Raised by stage 1 (SPEC-CHUNK-115).
    """


class OversizedSentenceError(ValidationError):
    """A single sentence exceeds the chunklet ``max_size``.

    Raised by stage 2 (SPEC-CHUNK-263).
    """


class OversizedChunkletError(ValidationError):
    """A single chunklet exceeds the chunk ``max_size``.

    Raised by stage 3 (SPEC-CHUNK-341).
    """


class ZeroNormEmbeddingError(ValidationError):
    """An embedding row has L2 norm zero.

    Raised by stage 3 (SPEC-CHUNK-342).
    """


class ChunkExceedsContextError(ValidationError):
    """A single chunk exceeds the embedder's context size.

    Raised by stage 4 (SPEC-CHUNK-451) when the unit fed to the
    embedder (a chunk in the current API, historically a sentence)
    tokenizes to more tokens than ``embedder.n_ctx`` allows.
    """


# Back-compat alias. Older stage-4 callers used the per-sentence API
# and referenced this name; the unit is now a chunk, but external
# imports of the old name keep working.
SentenceExceedsContextError = ChunkExceedsContextError


class OptimizationFailedError(FancyChunkError):
    """Underlying optimization solver reported failure.

    Raised by stage 3 (SPEC-CHUNK-343).
    """


class SegmenterError(FancyChunkError):
    """A sentence-segmentation model returned output that violated
    SPEC-CHUNK-106 (wrong shape, NaN/Inf, etc.).
    """
