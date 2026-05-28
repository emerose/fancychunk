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


# Scientific-prose segmentation: the default model + inference params
# (sat-9l-sm, weighting="hat") segment these constructs correctly with
# no post-processing. Guards the model/params choice against regressions
# (the lighter sat-3l-sm mis-splits all of these).
def test_sat_segments_scientific_prose_correctly() -> None:
    # Abbreviation reference: no split after "Tab." / "Eq.".
    assert split_sentences(
        "The results of our experiments can be seen in Tab. TABREF21 and TABREF22 ."
    ) == ["The results of our experiments can be seen in Tab. TABREF21 and TABREF22 ."]
    assert split_sentences("We optimize the loss in Eq. EQREF9 .") == [
        "We optimize the loss in Eq. EQREF9 ."
    ]
    # Year before a capitalized word: split only at the period, not the year.
    out = split_sentences(
        "We achieve new state-of-the-art results on SentiHood and "
        "SemEval-2014 Task 4 datasets. The next sentence is here."
    )
    assert len(out) == 2 and out[0].rstrip().endswith("datasets.")
    # No regression: a year genuinely ending a sentence still splits.
    out = split_sentences(
        "We finished the work in 2014. Later, we extended the analysis."
    )
    assert len(out) == 2 and out[0].rstrip().endswith("2014.")
