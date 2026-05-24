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
