"""Placeholder test so the suite is non-empty.

Real tests will live alongside each module of the implementation and
will be cross-referenced from docs/specs/acceptance/checklist.md.
"""

from fancychunk import __version__


def test_version_exposed() -> None:
    assert __version__ == "0.0.0"
