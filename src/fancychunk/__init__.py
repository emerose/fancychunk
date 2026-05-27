"""fancychunk — text chunking for retrieval-augmented generation.

Behavioral specs live in ``docs/specs/``; this package implements the
three required pipeline stages and the two optional helpers documented
in ``docs/specs/contracts/public-api.md``.
"""

from __future__ import annotations

from . import _constants as constants
from . import embedders
from .chunklets import split_chunklets
from .chunks import Chunk, ChunkletEmbedder, split_chunks
from .document import Embedder, chunk_document, chunk_documents
from .errors import (
    ChunkExceedsContextError,
    FancyChunkError,
    OptimizationFailedError,
    OversizedChunkletError,
    OversizedSentenceError,
    SegmenterError,
    SentenceExceedsContextError,
    UnsplittableDocumentError,
    ValidationError,
    ZeroNormEmbeddingError,
)
from .headings import enrich_with_headings, heading_paths
from .late_chunking import SegmentEmbedder, embed_with_late_chunking
from ._segmenter import SaTSegmenter, SentenceSegmenter, punctuation_segmenter
from .sentences import split_sentences

__all__ = [
    "split_sentences",
    "split_chunklets",
    "split_chunks",
    "Chunk",
    "ChunkletEmbedder",
    "embed_with_late_chunking",
    "heading_paths",
    "enrich_with_headings",
    "chunk_document",
    "chunk_documents",
    "Embedder",
    "SaTSegmenter",
    "SentenceSegmenter",
    "punctuation_segmenter",
    "SegmentEmbedder",
    "FancyChunkError",
    "ValidationError",
    "UnsplittableDocumentError",
    "OversizedSentenceError",
    "OversizedChunkletError",
    "ZeroNormEmbeddingError",
    "ChunkExceedsContextError",
    "SentenceExceedsContextError",
    "OptimizationFailedError",
    "SegmenterError",
    "constants",
    "embedders",
]

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("fancychunk")
except Exception:
    # Source checkout or build-time call before metadata exists.
    __version__ = "0.0.0+unknown"
