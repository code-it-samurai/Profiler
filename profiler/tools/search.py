import hashlib
import logging
from typing import Optional
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# In-memory cache: query_hash -> results
_cache: dict[str, list[dict]] = {}


async def google_search(
    query: str,
    num_results: int = 10,
    site_filter: Optional[str] = None,
) -> list[dict]:
    """Search the web via DuckDuckGo.

    Uses the duckduckgo-search package. No API key needed.
    Function is named google_search for interface compatibility --
    swap in any search provider later without changing callers.

    Args:
        query: Search query string.
        num_results: Number of results (max 30 for DDG).
        site_filter: Restrict to a domain (e.g., "facebook.com").

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    if site_filter:
        query = f"site:{site_filter} {query}"

    cache_key = hashlib.md5(f"{query}:{num_results}".encode()).hexdigest()
    if cache_key in _cache:
        logger.debug(f"Cache hit for query: {query}")
        return _cache[cache_key]

    results = []

    try:
        # DDGS is sync -- run in thread to not block the event loop
        import asyncio

        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=min(num_results, 30))),
        )

        for item in raw_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
        # Return empty rather than crashing -- the agent handles empty results
        return []

    _cache[cache_key] = results
    return results
