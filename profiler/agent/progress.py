"""Shared progress callback for agent nodes.

Nodes call emit() to report what they're doing in real time.
The CLI (or API) sets the callback function at startup.
"""

from typing import Callable, Optional

# Global callback — set by the CLI/API before running the graph.
# Signature: (step: str, detail: str, pct: int | None) -> None
#   step:   short phase label like "broad_search", "scraping", "extracting"
#   detail: human-readable description of what's happening right now
#   pct:    optional 0-100 progress percentage within the current phase
_callback: Optional[Callable[[str, str, Optional[int]], None]] = None


def set_progress_callback(cb: Callable[[str, str, Optional[int]], None]) -> None:
    """Set the global progress callback. Call once at CLI/API startup."""
    global _callback
    _callback = cb


def emit(step: str, detail: str, pct: int | None = None) -> None:
    """Emit a progress update. No-op if no callback is set."""
    if _callback is not None:
        _callback(step, detail, pct)
