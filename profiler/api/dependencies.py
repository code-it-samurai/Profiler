from collections import defaultdict
from fastapi import HTTPException, Request
from profiler.config import settings

# Simple in-memory rate limiter
_active_sessions: dict[str, int] = defaultdict(int)


async def check_rate_limit(request: Request):
    """Limit concurrent sessions per IP."""
    client_ip = request.client.host if request.client else "unknown"
    if _active_sessions[client_ip] >= settings.max_concurrent_sessions:
        raise HTTPException(429, "Too many concurrent sessions. Please wait.")


def track_session(client_ip: str):
    _active_sessions[client_ip] += 1


def release_session(client_ip: str):
    _active_sessions[client_ip] = max(0, _active_sessions[client_ip] - 1)
