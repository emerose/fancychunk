"""fancychunk — text chunking for retrieval-augmented generation.

Behavioral specs live in ``docs/specs/``; this package implements the
three required pipeline stages and the two optional helpers documented
in ``docs/specs/contracts/public-api.md``.
"""

from __future__ import annotations

from . import _constants as constants
from .chunklets import split_chunklets
from .chunks import split_chunks
from .errors import (
    FancyChunkError,
    OptimizationFailedError,
    OversizedChunkletError,
    OversizedSentenceError,
    SentenceExceedsContextError,
    UnsplittableDocumentError,
    ValidationError,
    ZeroNormEmbeddingError,
)
from .headings import heading_paths
from .late_chunking import TokenLevelEmbedder, embed_with_late_chunking
from ._segmenter import SaTSegmenter, SentenceSegmenter, punctuation_segmenter
from .sentences import split_sentences

__all__ = [
    "split_sentences",
    "split_chunklets",
    "split_chunks",
    "embed_with_late_chunking",
    "heading_paths",
    "SaTSegmenter",
    "SentenceSegmenter",
    "punctuation_segmenter",
    "TokenLevelEmbedder",
    "FancyChunkError",
    "ValidationError",
    "UnsplittableDocumentError",
    "OversizedSentenceError",
    "OversizedChunkletError",
    "ZeroNormEmbeddingError",
    "SentenceExceedsContextError",
    "OptimizationFailedError",
    "constants",
]

__version__ = "0.1.0"
