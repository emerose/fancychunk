"""Optional SaT model-dependent tests (TV-150 etc.).

These exercise the real SaT model and are skipped by default. Run
them with ``FANCYCHUNK_TEST_USE_SAT=1 pytest tests/test_sat.py``.
The model weights (~408 MB) download lazily on first use.
"""

from __future__ import annotations

import os

import pytest

from fancychunk import split_sentences

pytestmark = pytest.mark.skipif(
    os.environ.get("FANCYCHUNK_TEST_USE_SAT") != "1",
    reason="set FANCYCHUNK_TEST_USE_SAT=1 to run real-model tests",
)


# TV-150 — period-separated English sentences should split at periods.
def test_tv_150_period_separated_sentences() -> None:
    doc = "First sentence here. Second sentence here. Third sentence here."
    out = split_sentences(doc)
    assert "".join(out) == doc
    # SaT is expected to find the two internal sentence boundaries.
    assert len(out) >= 2
    # Whitespace must remain trailing (SPEC-CHUNK-102, -109).
    for s in out[1:]:
        assert not s[0].isspace()


# TV-118 — a numeral/year followed by a Capitalized word must not split
# the sentence mid-phrase. The real-model counterpart of the
# model-independent TV-118 vectors in test_sentences.py; this is the
# exact repro from the v0.4.0 bug report.
def test_tv_118_numeral_capital_no_midphrase_split() -> None:
    doc = (
        "We achieve new state-of-the-art results on SentiHood and "
        "SemEval-2014 Task 4 datasets. The next sentence is here."
    )
    assert split_sentences(doc) == [
        "We achieve new state-of-the-art results on SentiHood and "
        "SemEval-2014 Task 4 datasets. ",
        "The next sentence is here.",
    ]


# TV-118 — the space-instead-of-hyphen variant ("SemEval 2014 Task")
# triggers the same artifact and must also be suppressed.
def test_tv_118_numeral_capital_space_variant() -> None:
    doc = (
        "We achieve new state-of-the-art results on SentiHood and "
        "SemEval 2014 Task 4 datasets. The next sentence is here."
    )
    out = split_sentences(doc)
    assert "".join(out) == doc
    assert len(out) == 2
    assert out[0].endswith("datasets. ")


# TV-118 — no regression: a year ending a sentence ("...in 2014.")
# still splits, because the boundary sits on the period.
def test_tv_118_year_terminated_sentence_still_splits() -> None:
    doc = "We finished the work in 2014. Later, we extended the analysis."
    out = split_sentences(doc)
    assert "".join(out) == doc
    assert len(out) == 2
    assert out[0].endswith("2014. ")
