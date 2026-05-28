"""Tests for the bundled segmenter factories (fancychunk.segmenters).

``SaTSegmenter`` loads weights lazily, so these construct instances and
inspect configuration without downloading any model.
"""

from __future__ import annotations

from fancychunk import segmenters
from fancychunk._segmenter import SaTSegmenter, punctuation_segmenter


def test_sat_factories_select_the_right_checkpoint() -> None:
    assert segmenters.sat_3l().model_name == "sat-3l-sm"
    assert segmenters.sat_9l().model_name == "sat-9l-sm"
    assert segmenters.sat_12l().model_name == "sat-12l-sm"


def test_sat_default_is_9l() -> None:
    # The default factory and a bare SaTSegmenter() agree on the model.
    assert segmenters.sat_default().model_name == "sat-9l-sm"
    assert SaTSegmenter().model_name == "sat-9l-sm"


def test_sat_factory_forwards_kwargs() -> None:
    seg = segmenters.sat_12l(device="cpu")
    assert isinstance(seg, SaTSegmenter)
    assert seg.ort_providers == ["CPUExecutionProvider"]


def test_punctuation_factory_returns_the_fallback() -> None:
    assert segmenters.punctuation() is punctuation_segmenter
