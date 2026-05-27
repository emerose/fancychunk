"""Thread-safety tests for the lazy-loaded singletons and shared parsers.

These cover the three bug classes the PooledSegmentEmbedder fix
established: check-then-act lazy loads, module-level singleton
getters, and shared parser instances. Each test runs without
downloading real model weights — stub loaders simulate the slow
init path, and the markdown-it test uses the real (cheap) parser.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

from fancychunk import _segmenter
from fancychunk._markdown import _PARSER, heading_spans, openers_by_line
from fancychunk._segmenter import SaTSegmenter, get_default_segmenter


# ---------------------------------------------------------------------------
# SaTSegmenter — lazy load is guarded.
# ---------------------------------------------------------------------------


class _StubSaT:
    def predict_proba(self, document: str):  # pragma: no cover - not used here
        raise AssertionError("not invoked in lazy-load test")


def test_sat_segmenter_lazy_load_does_not_double_load(
    monkeypatch,
) -> None:
    """N threads racing on a fresh SaTSegmenter see exactly one load."""
    seg = SaTSegmenter()
    load_count = 0
    load_lock = threading.Lock()

    class _FakeModule:
        @staticmethod
        def SaT(name: str) -> _StubSaT:
            nonlocal load_count
            time.sleep(0.05)  # widen the race window
            with load_lock:
                load_count += 1
            return _StubSaT()

    # The class does ``from wtpsplit_lite import SaT as _SaT`` inside
    # _ensure_loaded — patch sys.modules so the local import resolves
    # to our fake.
    import sys

    monkeypatch.setitem(sys.modules, "wtpsplit_lite", _FakeModule)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: seg._ensure_loaded(), range(8)))

    assert load_count == 1


# ---------------------------------------------------------------------------
# get_default_segmenter — module-level singleton is guarded.
# ---------------------------------------------------------------------------


def test_get_default_segmenter_singleton_is_thread_safe(monkeypatch) -> None:
    """N threads racing on a cold ``get_default_segmenter`` construct
    exactly one SaTSegmenter instance."""
    # Reset the module-level cache, and re-bind the SaTSegmenter
    # constructor to a slow version so the race window is observable.
    monkeypatch.setattr(_segmenter, "_default_segmenter", None)

    construct_count = 0
    construct_lock = threading.Lock()
    real_init = SaTSegmenter.__init__

    def slow_init(self, *args, **kwargs):
        nonlocal construct_count
        time.sleep(0.02)
        with construct_lock:
            construct_count += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(SaTSegmenter, "__init__", slow_init)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: get_default_segmenter(), range(8)))

    assert construct_count == 1
    # All callers must observe the same instance.
    assert len({id(inst) for inst in instances}) == 1


# ---------------------------------------------------------------------------
# markdown-it shared parser — reentrancy guard.
# ---------------------------------------------------------------------------


def test_shared_markdown_parser_is_reentrant() -> None:
    """The module-level ``_PARSER`` is assumed reentrant across threads.

    If a future markdown-it-py release breaks that assumption (e.g.
    starts mutating state on the parser instance during parse), this
    test will surface it as inconsistent token counts.
    """
    text = "# Heading\n\nPara one with some text.\n\n## Sub\n\nPara two.\n" * 20
    expected = len([t for t in _PARSER.parse(text) if t.type == "heading_open"])

    def parse_count(_: int) -> int:
        return len([t for t in _PARSER.parse(text) if t.type == "heading_open"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(parse_count, range(200)))

    assert set(results) == {expected}


def test_concurrent_heading_spans_and_openers_consistent() -> None:
    """The two markdown-it consumers in _markdown.py — ``heading_spans``
    and ``openers_by_line`` — must produce stable output under
    concurrent invocation."""
    text = "# H1\n\nParagraph one.\n\n## H2\n\n- list item\n\n> quote\n" * 10
    expected_spans = heading_spans(text)
    expected_openers = openers_by_line(text)

    def both(_: int) -> tuple[int, int]:
        return (
            len(heading_spans(text)),
            len(openers_by_line(text)),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(both, range(200)))

    assert set(results) == {(len(expected_spans), len(expected_openers))}
