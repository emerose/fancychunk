"""Chunker adapters with a uniform interface.

All adapters expose:

* ``name`` — short label for reporting.
* ``chunk(doc: str) -> list[str]`` — synchronous, returns chunk texts.
* ``async achunk(doc: str) -> tuple[list[str], NDArray | None]`` —
  async variant that may return native vectors (fancychunk's late
  chunking does; others return ``None`` and the harness re-embeds
  with the common embedder).

The six configurations the harness sweeps:

1. ``recursive-langchain``  — LangChain's RecursiveCharacterTextSplitter
2. ``recursive-chonkie``    — Chonkie's RecursiveChunker
3. ``semantic-chonkie``     — Chonkie's SemanticChunker
4. ``fancychunk-noop``      — fancychunk with the noop() embedder
                              (heading-aware structural splits only;
                              no semantic signal)
5. ``fancychunk-vanilla``   — fancychunk with qwen3_600m() for split
                              decisions; chunks re-embedded with the
                              common embedder (no late chunking)
6. ``fancychunk-late``      — fancychunk full pipeline including
                              late-chunked vectors, with heading
                              prepend
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


CHUNK_SIZE_CHARS = 2048


class Chunker(Protocol):
    name: str

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]: ...


# ---------------------------------------------------------------------------
# LangChain — RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------


@dataclass
class LangChainRecursive:
    name: str = "recursive-langchain"
    chunk_size: int = CHUNK_SIZE_CHARS
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]:
        # LangChain is sync; offload so we don't block.
        chunks = await asyncio.to_thread(self._splitter.split_text, doc)
        return chunks, None


# ---------------------------------------------------------------------------
# Chonkie — RecursiveChunker
# ---------------------------------------------------------------------------


@dataclass
class ChonkieRecursive:
    name: str = "recursive-chonkie"
    chunk_size: int = CHUNK_SIZE_CHARS

    def __post_init__(self) -> None:
        from chonkie import RecursiveChunker

        # Chonkie defaults to token-based sizing; we want char-based
        # to match the other configurations. Use the "character"
        # tokenizer if available (chonkie >= 0.5).
        self._chunker = RecursiveChunker(
            tokenizer_or_token_counter="character",
            chunk_size=self.chunk_size,
        )

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]:
        chonkie_chunks = await asyncio.to_thread(self._chunker.chunk, doc)
        return [c.text for c in chonkie_chunks], None


# ---------------------------------------------------------------------------
# Chonkie — SemanticChunker
# ---------------------------------------------------------------------------


@dataclass
class ChonkieSemantic:
    name: str = "semantic-chonkie"
    chunk_size: int = CHUNK_SIZE_CHARS
    # ``embedding_model`` here is chonkie's choice for boundary
    # detection — independent of the common-embedder used for
    # retrieval scoring. We pick BGE-M3 because it's the same family
    # fancychunk's bge_m3() uses.
    embedding_model: str = "BAAI/bge-m3"

    def __post_init__(self) -> None:
        from chonkie import SemanticChunker

        self._chunker = SemanticChunker(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            mode="window",
        )

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]:
        chonkie_chunks = await asyncio.to_thread(self._chunker.chunk, doc)
        return [c.text for c in chonkie_chunks], None


# ---------------------------------------------------------------------------
# fancychunk configurations
# ---------------------------------------------------------------------------


@dataclass
class FancyChunkNoop:
    """Structural-only — heading-aware splits, no embedder signal."""

    name: str = "fancychunk-noop"
    max_size: int = CHUNK_SIZE_CHARS

    def __post_init__(self) -> None:
        from fancychunk.embedders import noop

        self._embedder = noop()

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]:
        from fancychunk import split_chunklets, split_chunks, split_sentences

        sentences = split_sentences(doc, max_len=self.max_size)
        chunklets = split_chunklets(sentences, max_size=self.max_size)
        chunks = await split_chunks(chunklets, self._embedder, max_size=self.max_size)
        return [c.text for c in chunks], None


@dataclass
class FancyChunkVanilla:
    """Full fancychunk split decisions but no late chunking — chunks
    re-embedded with the common retrieval embedder. Isolates the
    contribution of fancychunk's split *decisions* (independent of
    late chunking's context-aware embeddings)."""

    name: str = "fancychunk-vanilla"
    max_size: int = CHUNK_SIZE_CHARS

    def __post_init__(self) -> None:
        from fancychunk.embedders import qwen3_600m

        self._embedder = qwen3_600m()

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]:
        from fancychunk import split_chunklets, split_chunks, split_sentences

        sentences = split_sentences(doc, max_len=self.max_size)
        chunklets = split_chunklets(sentences, max_size=self.max_size)
        chunks = await split_chunks(chunklets, self._embedder, max_size=self.max_size)
        return [c.text for c in chunks], None


@dataclass
class FancyChunkLate:
    """Full fancychunk pipeline including late-chunked vectors."""

    name: str = "fancychunk-late"
    max_size: int = CHUNK_SIZE_CHARS

    def __post_init__(self) -> None:
        from fancychunk.embedders import qwen3_600m

        self._embedder = qwen3_600m()

    async def achunk(
        self, doc: str
    ) -> tuple[list[str], NDArray[np.float64] | None]:
        from fancychunk import (
            embed_with_late_chunking,
            split_chunklets,
            split_chunks,
            split_sentences,
        )

        sentences = split_sentences(doc, max_len=self.max_size)
        chunklets = split_chunklets(sentences, max_size=self.max_size)
        chunks = await split_chunks(
            chunklets, self._embedder, max_size=self.max_size
        )
        vectors = await embed_with_late_chunking(chunks, self._embedder)
        return [c.text for c in chunks], vectors


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def all_chunkers() -> list[Chunker]:
    """The full 6-way sweep."""
    return [
        LangChainRecursive(),
        ChonkieRecursive(),
        ChonkieSemantic(),
        FancyChunkNoop(),
        FancyChunkVanilla(),
        FancyChunkLate(),
    ]
