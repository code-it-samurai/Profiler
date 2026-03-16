"""Tests for agent state and nodes."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from profiler.agent.state import AgentState, merge_candidates
from profiler.agent.nodes import (
    broad_search,
    extract_and_normalize,
    analyze_candidates,
    filter_candidates,
    deep_scrape,
    compile_profile,
    SearchQueries,
    NarrowingDecision,
    CompilationResult,
)
from profiler.models.candidate import CandidateProfile
from profiler.models.enums import TargetType, SessionStatus, Platform
from profiler.models.profile import Profile


def make_state(**overrides) -> dict:
    """Create a minimal valid AgentState dict for testing."""
    base = {
        "target_name": "John Smith",
        "target_type": TargetType.PERSON,
        "initial_context": "",
        "session_id": "test-session-123",
        "known_facts": {},
        "candidates": [],
        "eliminated": [],
        "search_history": [],
        "narrowing_round": 0,
        "current_question": None,
        "user_answer": None,
        "_raw_search_results": [],
        "direct_urls": [],
        "final_profile": None,
        "status": SessionStatus.SEARCHING,
        "error": None,
    }
    base.update(overrides)
    return base


class TestMergeCandidates:
    """Tests for the merge_candidates helper."""

    def test_dedup_by_url(self):
        c1 = CandidateProfile(name="A", profile_url="http://a.com")
        c2 = CandidateProfile(name="B", profile_url="http://b.com")
        c3 = CandidateProfile(name="A dup", profile_url="http://a.com")
        merged = merge_candidates([c1], [c2, c3])
        assert len(merged) == 2
        urls = {c.profile_url for c in merged}
        assert urls == {"http://a.com", "http://b.com"}

    def test_keeps_none_urls(self):
        c1 = CandidateProfile(name="A", profile_url=None)
        c2 = CandidateProfile(name="B", profile_url=None)
        merged = merge_candidates([c1], [c2])
        assert len(merged) == 2

    def test_empty_lists(self):
        merged = merge_candidates([], [])
        assert merged == []


class TestBroadSearch:
    """Tests for the broad_search node."""

    async def test_broad_search_success(self):
        """Test broad search with mocked LLM and search."""
        state = make_state()

        mock_search_plan = SearchQueries(
            queries=[
                {"query": "John Smith", "site_filter": None, "purpose": "general"},
            ]
        )

        mock_search_results = [
            {"title": "John Smith", "url": "http://example.com", "snippet": "A person"},
        ]

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_search_plan
            with patch(
                "profiler.agent.nodes.google_search", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = mock_search_results
                result = await broad_search(state)

        assert result["status"] == SessionStatus.SEARCHING
        assert len(result["_raw_search_results"]) == 1
        assert "John Smith" in result["search_history"]

    async def test_broad_search_no_results_no_urls(self):
        """Test broad search fails only when no results AND no direct URLs."""
        state = make_state()

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = SearchQueries(
                queries=[
                    {
                        "query": "Nonexistent Person",
                        "site_filter": None,
                        "purpose": "general",
                    }
                ]
            )
            with patch(
                "profiler.agent.nodes.google_search", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = []
                result = await broad_search(state)

        assert result["status"] == SessionStatus.FAILED
        assert "No search results" in result["error"]

    async def test_broad_search_no_results_but_has_direct_urls(self):
        """Test broad search succeeds when search empty but direct URLs present."""
        state = make_state(direct_urls=["https://facebook.com/jsmith"])

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = SearchQueries(
                queries=[
                    {"query": "John Smith", "site_filter": None, "purpose": "general"}
                ]
            )
            with patch(
                "profiler.agent.nodes.google_search", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = []
                result = await broad_search(state)

        # Should NOT fail — direct URLs will be scraped in extract_and_normalize
        assert result["status"] == SessionStatus.SEARCHING
        assert "https://facebook.com/jsmith" in result["direct_urls"]

    async def test_broad_search_llm_fallback(self):
        """Test broad search falls back to manual queries when LLM fails."""
        state = make_state()

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = ValueError("LLM failed")
            with patch(
                "profiler.agent.nodes.google_search", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = [
                    {"title": "R", "url": "http://x.com", "snippet": "S"},
                ]
                result = await broad_search(state)

        assert result["status"] == SessionStatus.SEARCHING
        # Should have generated fallback queries
        assert len(result["search_history"]) > 0

    async def test_broad_search_enriched_queries(self):
        """Test that structured fields generate enriched search queries."""
        state = make_state(
            known_facts={"location": "Portland", "employer": "Nike"},
        )

        captured_queries = []

        async def mock_search(query, num_results=10, site_filter=None):
            captured_queries.append(query)
            return [
                {
                    "title": "R",
                    "url": f"http://ex.com/{len(captured_queries)}",
                    "snippet": "S",
                }
            ]

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = ValueError("LLM failed")
            with patch("profiler.agent.nodes.google_search", side_effect=mock_search):
                result = await broad_search(state)

        assert result["status"] == SessionStatus.SEARCHING
        # Should include enriched queries like "John Smith Portland" and "John Smith Nike"
        query_text = " ".join(result["search_history"])
        assert "Portland" in query_text
        assert "Nike" in query_text


class TestFilterCandidates:
    """Tests for the filter_candidates node."""

    async def test_filter_keeps_matching(self):
        candidates = [
            CandidateProfile(name="John Smith", location="Portland, OR"),
            CandidateProfile(name="John Smith", location="Seattle, WA"),
            CandidateProfile(name="John Smith", location=None),  # unknown = keep
        ]
        state = make_state(
            candidates=candidates,
            current_question={
                "field": "location",
                "question": "Where?",
                "options": None,
                "reasoning": "test",
            },
            user_answer="Portland",
        )

        result = await filter_candidates(state)

        assert len(result["candidates"]) == 2  # Portland + None (kept)
        assert len(result["eliminated"]) == 1  # Seattle
        assert result["known_facts"]["location"] == "Portland"
        assert result["narrowing_round"] == 1

    async def test_filter_eliminates_all_non_matching(self):
        candidates = [
            CandidateProfile(name="John Smith", employer="Google"),
            CandidateProfile(name="John Smith", employer="Apple"),
        ]
        state = make_state(
            candidates=candidates,
            current_question={
                "field": "employer",
                "question": "Where?",
                "options": None,
                "reasoning": "test",
            },
            user_answer="Nike",
        )

        result = await filter_candidates(state)

        assert len(result["candidates"]) == 0
        assert len(result["eliminated"]) == 2


class TestAnalyzeCandidates:
    """Tests for the analyze_candidates node."""

    async def test_no_fields_to_narrow(self):
        """When all fields are already known, go to compiling."""
        candidates = [CandidateProfile(name="John Smith")]
        state = make_state(
            candidates=candidates,
            known_facts={"location": "Portland", "school": "UO", "employer": "Nike"},
        )

        result = await analyze_candidates(state)
        assert result["status"] == SessionStatus.COMPILING

    async def test_analyze_picks_question(self):
        """Test with mocked LLM decision."""
        candidates = [
            CandidateProfile(name="John Smith", location="Portland"),
            CandidateProfile(name="John Smith", location="Seattle"),
            CandidateProfile(name="John Smith", location="Austin"),
        ]
        state = make_state(candidates=candidates)

        mock_decision = NarrowingDecision(
            field="location",
            question="Where does this person live?",
            options=["Portland", "Seattle", "Austin"],
            reasoning="Location has 3 unique values",
            expected_elimination_pct=0.66,
        )

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_decision
            result = await analyze_candidates(state)

        assert result["status"] == SessionStatus.ASKING_USER
        assert result["current_question"]["field"] == "location"
        assert "Portland" in result["current_question"]["options"]


class TestCompileProfile:
    """Tests for the compile_profile node."""

    async def test_compile_success(self):
        candidates = [
            CandidateProfile(
                name="John Smith",
                platform=Platform.LINKEDIN,
                profile_url="https://linkedin.com/in/jsmith",
                location="Portland",
                employer="Nike",
                bio="Marketing Manager",
            ),
        ]
        state = make_state(
            candidates=candidates,
            known_facts={"location": "Portland"},
        )

        mock_result = CompilationResult(
            summary="John Smith is a marketing manager at Nike in Portland.",
            locations=["Portland, OR"],
            education=[],
            employment=["Nike - Marketing Manager"],
            associated_entities=["Nike Inc."],
            confidence_score=0.85,
        )

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_result
            result = await compile_profile(state)

        assert result["status"] == SessionStatus.DONE
        profile = result["final_profile"]
        assert isinstance(profile, Profile)
        assert profile.target_name == "John Smith"
        assert profile.confidence_score == 0.85
        assert len(profile.social_profiles) == 1

    async def test_compile_failure(self):
        state = make_state(candidates=[])

        with patch(
            "profiler.agent.nodes.validated_llm_call", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = Exception("LLM crashed")
            result = await compile_profile(state)

        assert result["status"] == SessionStatus.FAILED
        assert "Failed to compile" in result["error"]
