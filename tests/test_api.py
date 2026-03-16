"""Tests for API layer."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from profiler.api.sse import (
    sse_event,
    status_update,
    question_event,
    profile_ready_event,
    error_event,
)
from profiler.api.dependencies import (
    check_rate_limit,
    track_session,
    release_session,
    _active_sessions,
)


class TestSSEHelpers:
    """Tests for SSE event formatting."""

    def test_sse_event_dict_data(self):
        result = sse_event("test_event", {"key": "value"})
        assert result["event"] == "test_event"
        assert json.loads(result["data"]) == {"key": "value"}

    def test_sse_event_string_data(self):
        result = sse_event("test_event", "raw string")
        assert result["event"] == "test_event"
        assert result["data"] == "raw string"

    def test_status_update(self):
        result = status_update("searching", "Looking...", 5)
        data = json.loads(result["data"])
        assert result["event"] == "status_update"
        assert data["status"] == "searching"
        assert data["message"] == "Looking..."
        assert data["candidates_count"] == 5

    def test_question_event(self):
        result = question_event(
            "Where do they live?", "location", ["Portland", "Seattle"], 1
        )
        data = json.loads(result["data"])
        assert result["event"] == "question"
        assert data["question"] == "Where do they live?"
        assert data["field"] == "location"
        assert data["options"] == ["Portland", "Seattle"]
        assert data["round"] == 1

    def test_question_event_no_options(self):
        result = question_event("What school?", "school", None, 2)
        data = json.loads(result["data"])
        assert data["options"] is None

    def test_profile_ready_event(self):
        result = profile_ready_event("session-123")
        data = json.loads(result["data"])
        assert result["event"] == "profile_ready"
        assert data["profile_id"] == "session-123"

    def test_error_event(self):
        result = error_event("Something broke", recoverable=True)
        data = json.loads(result["data"])
        assert result["event"] == "error"
        assert data["message"] == "Something broke"
        assert data["recoverable"] is True

    def test_error_event_not_recoverable(self):
        result = error_event("Fatal error")
        data = json.loads(result["data"])
        assert data["recoverable"] is False


class TestDependencies:
    """Tests for API dependencies."""

    @pytest.fixture(autouse=True)
    def clear_sessions(self):
        """Clear active sessions before each test."""
        _active_sessions.clear()

    async def test_rate_limit_allows_under_threshold(self):
        """Should not raise when under the limit."""
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        # No active sessions, should pass
        await check_rate_limit(mock_request)

    async def test_rate_limit_blocks_over_threshold(self):
        """Should raise 429 when over the limit."""
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        # Set active sessions to max
        _active_sessions["127.0.0.1"] = 100

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(mock_request)
        assert exc_info.value.status_code == 429

    def test_track_and_release(self):
        track_session("1.2.3.4")
        assert _active_sessions["1.2.3.4"] == 1
        track_session("1.2.3.4")
        assert _active_sessions["1.2.3.4"] == 2
        release_session("1.2.3.4")
        assert _active_sessions["1.2.3.4"] == 1
        release_session("1.2.3.4")
        assert _active_sessions["1.2.3.4"] == 0
        release_session("1.2.3.4")
        assert _active_sessions["1.2.3.4"] == 0  # doesn't go negative
