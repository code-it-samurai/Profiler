"""Shared progress callback for agent nodes.

Nodes call emit() to report what they're doing in real time.
The CLI (or API) sets the callback function at startup.

Supports two calling conventions:
  - New structured: emit(phase, event, detail, pct=None, meta=None)
  - Old compat:     emit(step, detail, pct=None)

The callback receives all 5 arguments. Old-style calls are auto-detected
and wrapped into the new format with event="info".
"""

from typing import Callable, Optional

# Valid event types for new-style calls
_EVENT_TYPES = frozenset(
    {
        "start",
        "task_start",
        "task_done",
        "task_fail",
        "phase_done",
        "info",
    }
)

# Callback signature: (phase, event, detail, pct, meta) -> None
_callback: Optional[Callable] = None


def set_progress_callback(cb: Callable) -> None:
    """Set the global progress callback. Call once at CLI/API startup."""
    global _callback
    _callback = cb


def emit(
    phase: str,
    event_or_detail: str,
    detail_or_pct: str | int | None = None,
    pct: int | None = None,
    meta: dict | None = None,
) -> None:
    """Emit a progress update. No-op if no callback is set.

    Supports two calling conventions:

    New structured (preferred):
        emit("discovery", "task_done", "DDG Search", meta={"count": 26})
        emit("extract", "task_start", "Scraping", pct=10)

    Old compat (auto-detected):
        emit("broad_search", "Executing 5 queries...", 40)
        emit("broad_search", "Starting search...")
    """
    if _callback is None:
        return

    # Detect calling convention: if event_or_detail is a known event type,
    # it's the new format. Otherwise it's the old format.
    if event_or_detail in _EVENT_TYPES:
        # New structured format
        _callback(
            phase,
            event_or_detail,
            detail_or_pct if isinstance(detail_or_pct, str) else "",
            pct,
            meta,
        )
    else:
        # Old compat format: emit(step, detail, pct)
        old_detail = event_or_detail
        old_pct = detail_or_pct if isinstance(detail_or_pct, int) else None
        _callback(phase, "info", old_detail, old_pct, None)
