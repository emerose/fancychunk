"""Stage 1 tests — sentence splitting.

Each ``test_tv_XYZ`` corresponds to a test vector in
``docs/specs/test-vectors/01-sentence-splitting.md``. SPEC-CHUNK IDs
covered in passing are noted in the docstrings.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from fancychunk import (
    UnsplittableDocumentError,
    split_sentences,
)
from fancychunk._segmenter import SentenceSegmenter


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


# TV-107 / SPEC-CHUNK-103, -104, -111 — max_len splits an overlong
# sentence and every output sentence sits in [min_len, max_len].
def test_tv_107_max_len_splits() -> None:
    doc = "a" * 100
    out = split_sentences(doc, min_len=4, max_len=40)
    assert "".join(out) == doc
    for s in out:
        assert 4 <= len(s) <= 40


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


# SPEC-CHUNK-101 / SPEC-CHUNK-117 — whitespace-only documents return [].
# (Returning a single all-whitespace "sentence" would violate
# SPEC-CHUNK-101 on the no-non-whitespace-character clause.)
@pytest.mark.parametrize(
    "doc", ["     ", "\n\n\n", "\t\t", "   \n   ", "          "]
)
def test_whitespace_only_document_returns_empty(doc: str) -> None:
    assert split_sentences(doc) == []


# SPEC-CHUNK-110 — DP returns the score-maximizing boundary set.
# Constructed input: a 12-character document with one obvious boundary
# signal at position 5 (probability 1.0, well above threshold). With
# min_len=4 the only valid placements are k ∈ {3, 4, 5, 6, 7}; among
# these, k=5 has the highest score and is the unique optimum.
def test_spec_chunk_110_dp_finds_score_optimum() -> None:
    doc = "abcdef ghijkl"  # 13 chars
    known = np.full(len(doc), np.nan)
    known[5] = 1.0  # space after 'abcdef'
    out = split_sentences(doc, min_len=4, known_boundary_probas=known)
    # SPEC-CHUNK-109 shifts the boundary to the end of the whitespace
    # run; here that's position 6 (single-space run pinned to max via
    # the extended-run rule), giving the partition ['abcdef ', 'ghijkl'].
    assert out == ["abcdef ", "ghijkl"]


# SPEC-CHUNK-113 — tie-break by smallest predecessor index. With all
# probabilities equal (zero), every partition has the same total
# score; the spec selects the partition with the fewest boundaries.
def test_spec_chunk_113_tie_break_fewest_sentences() -> None:
    # 12 'a' characters, no internal boundary signals, no length
    # constraint pressure — the all-tied DP must pick the
    # single-sentence partition (smallest-predecessor at the final
    # step is j=0).
    doc = "a" * 12
    out = split_sentences(doc, min_len=4)
    assert out == [doc]


# SPEC-CHUNK-116 — when no position has probability above threshold
# but the single-sentence partition is feasible, return it.
def test_spec_chunk_116_no_boundaries_above_threshold() -> None:
    doc = "abcdefghij"  # 10 chars, no whitespace, no segmenter signal
    known = np.full(len(doc), np.nan)  # no overrides either
    # No max_len, so the single-sentence partition is valid.
    out = split_sentences(doc, min_len=4, known_boundary_probas=known)
    assert out == [doc]


# ---------------------------------------------------------------------------
# Numeral-boundary artifact suppression (TV-118).
#
# SaT assigns a spuriously high boundary probability to a numeral
# (typically a year) directly followed by whitespace and a capitalized
# word — e.g. the final "4" of "SemEval-2014" in "...SemEval-2014 Task
# 4..." — which makes the DP break the sentence mid-phrase. These tests
# inject that artifact directly via a synthetic segmenter so they are
# model-independent; the real-model versions live in test_sat.py.
# ---------------------------------------------------------------------------


def _fixed_segmenter(probs: NDArray[np.float64]) -> SentenceSegmenter:
    def _seg(document: str) -> NDArray[np.float64]:
        assert len(document) == len(probs)
        return probs

    return _seg


# TV-118 — a numeral followed by whitespace and a Capitalized word does
# not trigger a boundary; the genuine break at the period is kept.
def test_tv_118_numeral_capital_artifact_suppressed() -> None:
    doc = (
        "We achieve results on SemEval-2014 Task 4 datasets. "
        "The next sentence is here."
    )
    probs = np.zeros(len(doc))
    probs[doc.index("2014") + 3] = 0.52  # artifact spike on the final '4'
    probs[doc.index("datasets.") + len("datasets")] = 0.9  # the period
    out = split_sentences(doc, segmenter=_fixed_segmenter(probs))
    assert "".join(out) == doc
    assert out == [
        "We achieve results on SemEval-2014 Task 4 datasets. ",
        "The next sentence is here.",
    ]


# TV-118 — the same artifact with a space instead of a hyphen in the
# token preceding the year ("SemEval 2014 Task") is suppressed too.
def test_tv_118_numeral_capital_artifact_space_variant() -> None:
    doc = "We report on SemEval 2014 Task 4 here. Next one follows."
    probs = np.zeros(len(doc))
    probs[doc.index("2014") + 3] = 0.5
    probs[doc.index("here.") + len("here")] = 0.9
    out = split_sentences(doc, segmenter=_fixed_segmenter(probs))
    assert out == [
        "We report on SemEval 2014 Task 4 here. ",
        "Next one follows.",
    ]


# TV-118 — no regression: a year followed by a period and a Capitalized
# word still splits, because the boundary sits on the period (not the
# digit), which the suppression rule never touches.
def test_tv_118_year_period_capital_still_splits() -> None:
    doc = "We finished in 2014. Later, we extended the work."
    probs = np.zeros(len(doc))
    probs[doc.index("2014.") + 4] = 0.9  # the period
    out = split_sentences(doc, segmenter=_fixed_segmenter(probs))
    assert out == ["We finished in 2014. ", "Later, we extended the work."]


# TV-118 — the suppression helper is scoped precisely: it zeroes the
# digit only when the following word is capitalized, leaving lowercase
# followers and period-terminated numbers untouched.
def test_tv_118_suppression_helper_scope() -> None:
    from fancychunk.sentences import _suppress_numeral_boundary_artifacts

    doc = "a 2014 Task and 2014 task and 2014. Then"
    p = np.full(len(doc), 0.5)
    out = _suppress_numeral_boundary_artifacts(doc, p)
    assert out[doc.index("2014 Task") + 3] == 0.0  # capital -> suppressed
    assert out[doc.index("2014 task") + 3] == 0.5  # lowercase -> kept
    assert out[doc.index("2014. Then") + 3] == 0.5  # period -> kept
    # Input vector is not mutated in place.
    assert p[doc.index("2014 Task") + 3] == 0.5
