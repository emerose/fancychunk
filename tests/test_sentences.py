"""Stage 1 tests — sentence splitting.

Each ``test_tv_XYZ`` corresponds to a test vector in
``docs/specs/test-vectors/01-sentence-splitting.md``. SPEC-CHUNK IDs
covered in passing are noted in the docstrings.
"""

from __future__ import annotations

import numpy as np
import pytest

from fancychunk import (
    UnsplittableDocumentError,
    split_sentences,
)


# ---------------------------------------------------------------------------
# TV-101 / SPEC-CHUNK-114 — document shorter than min_len short-circuits.
def test_tv_101_short_document_short_circuit() -> None:
    assert split_sentences("ab") == ["ab"]


# TV-102 / SPEC-CHUNK-103 — document exactly at min_len returns single sentence.
def test_tv_102_document_at_min_len() -> None:
    assert split_sentences("abcd") == ["abcd"]


# TV-104 / SPEC-CHUNK-108 — heading is forced standalone.
def test_tv_104_heading_standalone() -> None:
    doc = "# Hello\n\nFirst sentence here. Second sentence here.\n"
    out = split_sentences(doc)
    assert "".join(out) == doc
    assert out[0] == "# Hello\n\n"


# TV-105 — heading with body on next line.
def test_tv_105_heading_then_body() -> None:
    doc = "## Title\nBody text follows immediately.\n"
    out = split_sentences(doc)
    assert "".join(out) == doc
    assert any("## Title" in s for s in out)
    # Body text should be in its own sentence (per spec property).
    assert any("Body text follows immediately." in s for s in out)


# TV-106 / SPEC-CHUNK-102, -109 — whitespace is trailing, not leading.
def test_tv_106_whitespace_trailing() -> None:
    doc = "First sentence. Second sentence. Third sentence."
    out = split_sentences(doc)
    assert "".join(out) == doc
    for s in out[1:]:
        assert not s[0].isspace(), f"sentence begins with whitespace: {s!r}"


# TV-107 / SPEC-CHUNK-104, -111 — max_len splits an overlong sentence.
def test_tv_107_max_len_splits() -> None:
    doc = "a" * 100
    out = split_sentences(doc, max_len=40)
    assert "".join(out) == doc
    for s in out:
        assert len(s) <= 40


# TV-108 — max_len larger than document is a no-op.
def test_tv_108_max_len_no_op() -> None:
    assert split_sentences("Short.", max_len=100) == ["Short."]


# TV-109 / SPEC-CHUNK-100 — multi-byte UTF-8 round-trip.
def test_tv_109_multibyte_round_trip() -> None:
    doc = "Héllo, wörld. ⊕ symbol here. 日本語テスト。"
    out = split_sentences(doc)
    assert "".join(out) == doc


# TV-110 / SPEC-CHUNK-117 — empty document.
def test_tv_110_empty_document() -> None:
    assert split_sentences("") == []


# TV-111 / SPEC-CHUNK-107 — overrides force split positions.
def test_tv_111_overrides_force_splits() -> None:
    doc = "abcde fghij klmno"
    known = np.full(len(doc), np.nan)
    known[5] = 1.0
    known[11] = 1.0
    out = split_sentences(doc, known_boundary_probas=known)
    assert out == ["abcde ", "fghij ", "klmno"]


# TV-112 / SPEC-CHUNK-108 — override prevents split inside heading.
def test_tv_112_no_split_inside_heading() -> None:
    doc = "# Long heading: with punctuation, and more.\n\nBody text.\n"
    out = split_sentences(doc)
    assert "".join(out) == doc
    # The heading line must sit within a single sentence.
    joined = "".join(s for s in out if "# Long heading" in s)
    assert "# Long heading: with punctuation, and more." in joined
    # And specifically: exactly one sentence contains the heading text.
    matches = [s for s in out if "# Long heading" in s]
    assert len(matches) == 1


# TV-113 / SPEC-CHUNK-115 — genuinely unsatisfiable length constraints raise.
def test_tv_113_unsplittable_raises() -> None:
    # ``"abcdefghi"`` (length 9): every internal split produces a
    # sentence shorter than ``min_len = 5``, and the single-sentence
    # partition is longer than ``max_len = 3``. No partition is feasible.
    with pytest.raises(UnsplittableDocumentError):
        split_sentences("abcdefghi", min_len=5, max_len=3)


# SPEC-CHUNK-901 / -112 — determinism.
def test_determinism() -> None:
    doc = "First sentence. Second sentence. Third sentence."
    a = split_sentences(doc)
    b = split_sentences(doc)
    assert a == b


# SPEC-CHUNK-101 — every sentence has at least one non-whitespace char.
def test_every_sentence_has_nonwhitespace() -> None:
    doc = "First sentence. Second sentence. Third sentence."
    for s in split_sentences(doc):
        assert any(not ch.isspace() for ch in s)
