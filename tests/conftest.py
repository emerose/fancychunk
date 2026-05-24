"""Pytest configuration.

The default ``SaTSegmenter`` lazily downloads 408 MB of model weights
on first use; that is the right behavior for production but burdens
test runs (and CI). For tests we swap in the lightweight
``punctuation_segmenter`` so the suite is self-contained.

If you want to exercise the SaT path end-to-end, run a single test
with the environment variable ``FANCYCHUNK_TEST_USE_SAT=1`` set.
"""

from __future__ import annotations

import os

import pytest

from fancychunk import _segmenter
from fancychunk._segmenter import punctuation_segmenter


@pytest.fixture(autouse=True)
def _use_punctuation_segmenter(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("FANCYCHUNK_TEST_USE_SAT") == "1":
        return
    monkeypatch.setattr(
        _segmenter, "get_default_segmenter", lambda: punctuation_segmenter
    )
