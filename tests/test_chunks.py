"""Stage 3 tests — semantic chunking."""

from __future__ import annotations

import numpy as np
import pytest

from fancychunk import (
    OversizedChunkletError,
    ZeroNormEmbeddingError,
    split_chunks,
)


# TV-301 / SPEC-CHUNK-340 — single chunklet short-circuits.
def test_tv_301_single_chunklet() -> None:
    chunks, emb = split_chunks(["Single chunklet content."], np.array([[1.0, 0.0]]))
    assert chunks == ["Single chunklet content."]
    assert len(emb) == 1
    assert np.array_equal(emb[0], np.array([[1.0, 0.0]]))


# TV-302 / SPEC-CHUNK-340 — total fits in max_size short-circuits.
def test_tv_302_total_fits_short_circuit() -> None:
    chunklets = ["one ", "two ", "three"]
    emb = np.eye(3)
    chunks, e = split_chunks(chunklets, emb)
    assert chunks == ["one two three"]
    assert len(e) == 1
    assert np.array_equal(e[0], emb)


# TV-303 / SPEC-CHUNK-300, -302 — round-trip property.
def test_tv_303_round_trip() -> None:
    chunklets = ["a" * 1000, "b" * 1000, "c" * 1000, "d" * 1000]
    emb = np.eye(4)
    chunks, e = split_chunks(chunklets, emb)
    assert "".join(chunks) == "".join(chunklets)
    rows = np.vstack(e)
    assert np.array_equal(rows, emb)


# TV-304 / SPEC-CHUNK-301, -311 — hard size constraint.
def test_tv_304_size_constraint_forces_split() -> None:
    chunklets = ["a" * 1000, "b" * 1000, "c" * 1000]
    emb = np.tile([[1.0, 0.0]], (3, 1))
    chunks, _ = split_chunks(chunklets, emb)
    for c in chunks:
        assert len(c) <= 2048
    assert "".join(chunks) == "".join(chunklets)


# TV-305 — identical embeddings, total fits: single chunk via short-circuit.
def test_tv_305_identical_short_circuit() -> None:
    chunklets = ["x" * 100] * 10
    emb = np.tile([[1.0, 0.0]], (10, 1))
    chunks, _ = split_chunks(chunklets, emb)
    assert chunks == ["x" * 1000]


# TV-307 / SPEC-CHUNK-322 — no split immediately after a heading.
def test_tv_307_no_split_after_heading() -> None:
    # Use a pure heading chunklet plus body chunklets that together
    # exceed max_size so the short-circuit does not apply and the
    # heading-aware modification runs.
    chunklets = ["# Heading\n\n", "x" * 800, "y" * 800, "z" * 800]
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
    )
    chunks, _ = split_chunks(chunklets, emb, max_size=2048)
    # The heading must not be its own standalone chunk.
    assert chunks[0] != "# Heading\n\n"


# TV-308 / SPEC-CHUNK-322 — encourage split before heading.
def test_tv_308_split_before_heading() -> None:
    chunklets = ["a" * 900, "b" * 900, "## Subhead\n\n", "c" * 900]
    emb = np.array(
        [
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
        ]
    )
    chunks, _ = split_chunks(chunklets, emb, max_size=2048)
    assert len(chunks) == 2
    assert chunks[1].startswith("## Subhead")


# TV-309 / SPEC-CHUNK-342 — zero-norm embedding rejected.
def test_tv_309_zero_norm_rejected() -> None:
    with pytest.raises(ZeroNormEmbeddingError):
        split_chunks(["a", "b"], np.array([[0.0, 0.0], [1.0, 0.0]]))


# TV-310 / SPEC-CHUNK-341 — oversized chunklet rejected.
def test_tv_310_oversized_chunklet_rejected() -> None:
    with pytest.raises(OversizedChunkletError):
        split_chunks(["a" * 3000, "b"], np.eye(2), max_size=2048)


# TV-311 / SPEC-CHUNK-321 — discourse-vector fallback when projection would zero rows.
def test_tv_311_discourse_fallback() -> None:
    emb = np.tile([[1.0, 0.0]], (5, 1))
    chunks, _ = split_chunks(["x" * 1000] * 5, emb, max_size=2048)
    for c in chunks:
        assert len(c) <= 2048
    assert "".join(chunks) == "".join(["x" * 1000] * 5)


# SPEC-CHUNK-330 — determinism.
def test_determinism() -> None:
    chunklets = ["a" * 900, "b" * 900, "## Heading\n", "c" * 900]
    emb = np.array([[1.0, 0.1], [1.0, 0.2], [1.0, 0.05], [1.0, 0.1]])
    out_a = split_chunks(chunklets, emb, max_size=2048)
    out_b = split_chunks(chunklets, emb, max_size=2048)
    assert out_a[0] == out_b[0]
    for ea, eb in zip(out_a[1], out_b[1]):
        assert np.array_equal(ea, eb)


# SPEC-CHUNK-340 — empty input.
def test_empty_input() -> None:
    chunks, emb = split_chunks([], np.zeros((0, 4)))
    assert chunks == []
    assert emb == []


# SPEC-CHUNK-322 — heading detection accepts ATX and Setext forms,
# rejects heading-plus-body and non-heading content.
@pytest.mark.parametrize(
    "chunklet,expected",
    [
        ("# ATX H1\n", True),
        ("## ATX H2\n\n", True),
        ("###### ATX H6\n", True),
        ("Heading text\n============\n\n", True),  # Setext H1
        ("Heading text\n============", True),  # Setext H1, no trailing newline
        ("H2 form\n--------\n", True),  # Setext H2
        ("Title 1\nTitle 2\n========", True),  # multi-line Setext text
        ("Heading\n=====\n\nBody.\n", False),  # heading + body
        ("Just a paragraph.\n", False),
        ("Text\n# Heading\n", False),  # body before heading
        ("", False),
        ("####### Seven hashes\n", False),  # > 6 hashes is not a heading
    ],
)
def test_is_heading_atx_and_setext(chunklet: str, expected: bool) -> None:
    from fancychunk.chunks import _is_heading

    assert _is_heading(chunklet) is expected


# SPEC-CHUNK-322 — heading-aware modification fires for Setext exactly
# as it does for ATX. Same shape as TV-308: a Setext heading buried
# between large body chunklets pulls the cheap split to its left edge
# and forbids the split to its right.
def test_setext_heading_pulls_split_before() -> None:
    chunklets = [
        "a" * 900,
        "b" * 900,
        "Subhead\n=======\n\n",
        "c" * 900,
    ]
    emb = np.array(
        [
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
            [1.0, 1e-3],
        ]
    )
    chunks, _ = split_chunks(chunklets, emb, max_size=2048)
    assert len(chunks) == 2
    assert chunks[1].startswith("Subhead\n=======")
