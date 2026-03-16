"""Tests for database layer."""

import os
import pytest
from unittest.mock import patch

from profiler.db.database import init_db, DB_PATH
from profiler.db import repository
from profiler.models.session import SearchSession
from profiler.models.enums import TargetType, SessionStatus
from profiler.models.profile import Profile


# Use a test-specific DB path
TEST_DB = "test_profiler.db"


@pytest.fixture(autouse=True)
async def setup_test_db():
    """Set up a clean test database for each test."""
    # Patch the DB_PATH in both modules
    with (
        patch.object(repository, "DB_PATH", TEST_DB),
        patch("profiler.db.database.DB_PATH", TEST_DB),
    ):
        import aiosqlite

        async with aiosqlite.connect(TEST_DB) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    target_name TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    context TEXT,
                    status TEXT NOT NULL DEFAULT 'searching',
                    narrowing_round INTEGER DEFAULT 0,
                    candidates_count INTEGER DEFAULT 0,
                    known_facts TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            await db.execute(
                """CREATE TABLE IF NOT EXISTS profiles (
                    session_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )"""
            )
            await db.commit()

        yield

    # Cleanup
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestDatabase:
    """Tests for database initialization."""

    async def test_init_db(self):
        """Test that init_db creates tables."""
        with patch("profiler.db.database.DB_PATH", TEST_DB):
            # Tables already exist from fixture, but init_db should not fail
            await init_db()


class TestRepository:
    """Tests for CRUD operations."""

    async def test_create_and_get_session(self):
        """Test creating and retrieving a session."""
        session = SearchSession(
            target_name="John Smith",
            target_type=TargetType.PERSON,
            context="works at Nike",
        )

        with patch.object(repository, "DB_PATH", TEST_DB):
            await repository.create_session(session)
            result = await repository.get_session(str(session.id))

        assert result is not None
        assert result["target_name"] == "John Smith"
        assert result["target_type"] == "person"
        assert result["context"] == "works at Nike"
        assert result["status"] == "searching"

    async def test_update_session(self):
        """Test updating session fields."""
        session = SearchSession(
            target_name="Jane Doe",
            target_type=TargetType.PERSON,
        )

        with patch.object(repository, "DB_PATH", TEST_DB):
            await repository.create_session(session)
            await repository.update_session(
                str(session.id),
                status="narrowing",
                narrowing_round=2,
                candidates_count=15,
            )
            result = await repository.get_session(str(session.id))

        assert result["status"] == "narrowing"
        assert result["narrowing_round"] == 2
        assert result["candidates_count"] == 15

    async def test_get_nonexistent_session(self):
        """Test getting a session that doesn't exist."""
        with patch.object(repository, "DB_PATH", TEST_DB):
            result = await repository.get_session("nonexistent-id")
        assert result is None

    async def test_save_and_get_profile(self):
        """Test saving and retrieving a profile."""
        session = SearchSession(
            target_name="John Smith",
            target_type=TargetType.PERSON,
        )
        profile = Profile(
            target_name="John Smith",
            target_type=TargetType.PERSON,
            summary="A marketing manager at Nike.",
            locations=["Portland, OR"],
            confidence_score=0.85,
        )

        with patch.object(repository, "DB_PATH", TEST_DB):
            await repository.create_session(session)
            await repository.save_profile(str(session.id), profile)
            result = await repository.get_profile(str(session.id))

        assert result is not None
        assert result.target_name == "John Smith"
        assert result.summary == "A marketing manager at Nike."
        assert result.confidence_score == 0.85

    async def test_get_nonexistent_profile(self):
        """Test getting a profile that doesn't exist."""
        with patch.object(repository, "DB_PATH", TEST_DB):
            result = await repository.get_profile("nonexistent-id")
        assert result is None

    async def test_delete_session(self):
        """Test deleting a session and its profile."""
        session = SearchSession(
            target_name="John Smith",
            target_type=TargetType.PERSON,
        )
        profile = Profile(
            target_name="John Smith",
            target_type=TargetType.PERSON,
            summary="Test.",
        )

        with patch.object(repository, "DB_PATH", TEST_DB):
            await repository.create_session(session)
            await repository.save_profile(str(session.id), profile)
            await repository.delete_session(str(session.id))
            session_result = await repository.get_session(str(session.id))
            profile_result = await repository.get_profile(str(session.id))

        assert session_result is None
        assert profile_result is None
