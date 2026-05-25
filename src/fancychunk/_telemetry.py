"""Module-shared tracer and logger.

The library uses ``opentelemetry-api`` for tracing. With no SDK
configured the spans created here are zero-cost no-ops; once the
caller's application installs an SDK and exporter the same spans
appear in their trace backend.

Naming conventions for span attributes follow OpenTelemetry's
recommendation: lowercase dotted strings, scoped under
``fancychunk.<stage>.<attribute>``. Counts and lengths are integers;
durations are not set explicitly (the SDK measures them).
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.trace import Tracer

_INSTRUMENTATION_NAME = "fancychunk"


def _instrumentation_version() -> str:
    """Best-effort package version, used as the tracer's library version."""
    try:
        from importlib.metadata import version

        return version("fancychunk")
    except Exception:
        return "0.0.0+unknown"


def get_tracer() -> Tracer:
    """Return the module-shared tracer.

    Re-resolved each call so that a caller installing an SDK *after*
    importing fancychunk still sees their spans (OpenTelemetry's API
    is designed for this — the underlying ``ProxyTracer`` delegates
    dynamically).
    """
    return trace.get_tracer(_INSTRUMENTATION_NAME, _instrumentation_version())


def get_logger() -> logging.Logger:
    """Return the library logger.

    By default Python's logging machinery silences messages from this
    logger; callers opt in with e.g.
    ``logging.getLogger('fancychunk').setLevel(logging.INFO)``.
    """
    return logging.getLogger("fancychunk")
