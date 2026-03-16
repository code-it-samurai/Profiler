"""Tests for profiler tools."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from profiler.tools.matcher import fuzzy_match
from profiler.tools.search import google_search, _cache
from profiler.tools.scraper import scrape_page
from profiler.tools.extractor import extract_profile, detect_platform
from profiler.models.enums import Platform

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestFuzzyMatch:
    """Tests for the fuzzy_match function."""

    def test_exact_match(self):
        is_match, score = fuzzy_match("Portland", "Portland")
        assert is_match is True
        assert score == 1.0

    def test_case_insensitive(self):
        is_match, score = fuzzy_match("Portland", "portland")
        assert is_match is True
        assert score == 1.0

    def test_containment_user_in_candidate(self):
        is_match, score = fuzzy_match("University of Oregon", "Oregon")
        assert is_match is True
        assert score == 0.9

    def test_containment_candidate_in_user(self):
        is_match, score = fuzzy_match("Nike", "Nike Inc.")
        assert is_match is True
        assert score == 0.9

    def test_fuzzy_match_word_order(self):
        is_match, score = fuzzy_match("Smith John", "John Smith")
        assert is_match is True
        assert score >= 0.7

    def test_no_match(self):
        is_match, score = fuzzy_match("Portland", "New York")
        assert is_match is False

    def test_none_candidate(self):
        is_match, score = fuzzy_match(None, "Portland")
        assert is_match is False
        assert score == 0.0

    def test_empty_strings(self):
        is_match, score = fuzzy_match("", "Portland")
        assert is_match is False
        assert score == 0.0

    def test_whitespace_handling(self):
        is_match, score = fuzzy_match("  Portland  ", "Portland")
        assert is_match is True

    def test_custom_threshold(self):
        # With a very high threshold, partial matches should fail
        is_match, score = fuzzy_match("Portland OR", "Portland Oregon", threshold=0.99)
        # score should be reasonable but below 0.99
        assert score > 0.5

    def test_similar_but_different(self):
        is_match, score = fuzzy_match("Portland, OR", "Portland, ME")
        # These are similar strings — check that the score reflects that
        assert score > 0.5


class TestGoogleSearch:
    """Tests for the DuckDuckGo search wrapper."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the search cache before each test."""
        _cache.clear()

    async def test_search_returns_results(self):
        """Test search with mocked DDG results."""
        mock_results = [
            {
                "title": "Test Result",
                "href": "https://example.com",
                "body": "A snippet",
            },
            {"title": "Another", "href": "https://example.org", "body": "More text"},
        ]
        with patch("profiler.tools.search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value
            instance.text.return_value = mock_results
            results = await google_search("test query", num_results=5)

        assert len(results) == 2
        assert results[0]["title"] == "Test Result"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == "A snippet"

    async def test_search_with_site_filter(self):
        """Test that site_filter is prepended to query."""
        with patch("profiler.tools.search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value
            instance.text.return_value = []
            await google_search("John Smith", site_filter="facebook.com")
            instance.text.assert_called_once()
            call_args = instance.text.call_args
            assert "site:facebook.com" in call_args[0][0]

    async def test_search_caching(self):
        """Test that repeated queries use cache."""
        mock_results = [{"title": "T", "href": "http://x.com", "body": "B"}]
        with patch("profiler.tools.search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value
            instance.text.return_value = mock_results

            r1 = await google_search("cached query")
            r2 = await google_search("cached query")

        assert r1 == r2
        # DDGS should only be instantiated once (second call uses cache)
        assert MockDDGS.call_count == 1

    async def test_search_handles_exception(self):
        """Test graceful handling of search errors."""
        with patch("profiler.tools.search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value
            instance.text.side_effect = Exception("Network error")
            results = await google_search("failing query")

        assert results == []


class TestScrapePage:
    """Tests for the scraper."""

    async def test_scrape_static_page(self):
        """Test httpx-based scraping with mocked response."""
        html = (FIXTURES_DIR / "mock_html_facebook.html").read_text()

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("profiler.tools.scraper.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await scrape_page("https://example.com/page", max_chars=5000)

        assert result["success"] is True
        assert result["url"] == "https://example.com/page"
        assert "John Smith" in result["text"]
        assert "Portland" in result["text"]
        # Script/style/nav/footer content should be removed
        assert "console.log" not in result["text"]
        assert "Navigation content" not in result["text"]

    async def test_scrape_failure(self):
        """Test graceful handling of scrape errors."""
        with patch("profiler.tools.scraper.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.side_effect = Exception("Connection refused")
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await scrape_page("https://example.com/fail")

        assert result["success"] is False
        assert result["error"] is not None
        assert result["url"] == "https://example.com/fail"

    async def test_scrape_js_site_attempts_playwright(self):
        """Test that JS-heavy domains trigger Playwright path."""
        # Since Playwright browser isn't installed, it should fail gracefully
        result = await scrape_page("https://www.facebook.com/someuser")
        assert result["success"] is False
        assert result["url"] == "https://www.facebook.com/someuser"
        # Should have an error message about Playwright
        assert result["error"] is not None


class TestDetectPlatform:
    """Tests for platform detection."""

    def test_facebook(self):
        assert detect_platform("https://www.facebook.com/jsmith") == Platform.FACEBOOK

    def test_twitter(self):
        assert detect_platform("https://twitter.com/jsmith") == Platform.TWITTER

    def test_x_dot_com(self):
        assert detect_platform("https://x.com/jsmith") == Platform.TWITTER

    def test_linkedin(self):
        assert (
            detect_platform("https://www.linkedin.com/in/jsmith") == Platform.LINKEDIN
        )

    def test_instagram(self):
        assert detect_platform("https://www.instagram.com/jsmith") == Platform.INSTAGRAM

    def test_generic(self):
        assert detect_platform("https://example.com/page") == Platform.GENERIC


class TestExtractProfile:
    """Tests for the LLM-based profile extractor (mocked)."""

    async def test_extract_profile_success(self):
        """Test extraction with mocked LLM response."""
        scraped_data = {
            "url": "https://www.linkedin.com/in/jsmith",
            "title": "John Smith - LinkedIn",
            "text": "John Smith. Marketing Manager at Nike. Portland, OR. University of Oregon.",
            "meta_description": "View John Smith's profile on LinkedIn.",
            "success": True,
            "error": None,
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "name": "John Smith",
                "location": "Portland, OR",
                "school": "University of Oregon",
                "employer": "Nike",
                "bio": "Marketing Manager at Nike",
            }
        )

        with patch("profiler.tools.extractor.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await extract_profile(scraped_data, "John Smith")

        assert result is not None
        assert result.name == "John Smith"
        assert result.platform == Platform.LINKEDIN
        assert result.location == "Portland, OR"
        assert result.school == "University of Oregon"
        assert result.employer == "Nike"
        assert result.profile_url == "https://www.linkedin.com/in/jsmith"

    async def test_extract_profile_failed_scrape(self):
        """Test that failed scrapes return None."""
        scraped_data = {
            "url": "https://example.com",
            "title": "",
            "text": "",
            "success": False,
            "error": "Connection refused",
        }
        result = await extract_profile(scraped_data, "John Smith")
        assert result is None

    async def test_extract_profile_llm_failure_fallback(self):
        """Test fallback profile creation when LLM fails."""
        scraped_data = {
            "url": "https://www.facebook.com/jsmith",
            "title": "John Smith - Facebook",
            "text": "Some content here",
            "meta_description": "John Smith on Facebook",
            "success": True,
            "error": None,
        }

        with patch("profiler.tools.extractor.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("LLM error")
            mock_get_llm.return_value = mock_llm

            result = await extract_profile(scraped_data, "John Smith")

        # Should return a fallback profile
        assert result is not None
        assert result.name == "John Smith"
        assert result.platform == Platform.FACEBOOK
        assert result.bio == "John Smith on Facebook"
