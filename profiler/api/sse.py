import json
from typing import Any


def sse_event(event_type: str, data: Any) -> dict:
    """Format an SSE event for sse-starlette."""
    return {
        "event": event_type,
        "data": json.dumps(data) if not isinstance(data, str) else data,
    }


def status_update(status: str, message: str, candidates_count: int = 0) -> dict:
    return sse_event(
        "status_update",
        {
            "status": status,
            "message": message,
            "candidates_count": candidates_count,
        },
    )


def question_event(
    question: str, field: str, options: list | None, round_num: int
) -> dict:
    return sse_event(
        "question",
        {
            "question": question,
            "field": field,
            "options": options,
            "round": round_num,
        },
    )


def profile_ready_event(session_id: str) -> dict:
    return sse_event("profile_ready", {"profile_id": session_id})


def error_event(message: str, recoverable: bool = False) -> dict:
    return sse_event("error", {"message": message, "recoverable": recoverable})
