"""robots.txt pre-check for scraping.

Checks whether a URL is allowed by the site's robots.txt before scraping.
Caches results per domain for the duration of a session.
"""

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

# Module-level cache: domain → RobotFileParser
_robots_cache: dict[str, RobotFileParser | None] = {}


async def is_scrapable(url: str) -> bool:
    """Check if a URL is allowed by robots.txt.

    Returns True (optimistic) on any error — if we can't fetch or parse
    robots.txt, we assume scraping is allowed.

    Args:
        url: Full URL to check.

    Returns:
        True if scraping is allowed or robots.txt can't be checked.
    """
    try:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain in _robots_cache:
            rp = _robots_cache[domain]
            if rp is None:
                return True  # previously failed to fetch — optimistic
            return rp.can_fetch("*", url)

        robots_url = f"{domain}/robots.txt"
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(robots_url, follow_redirects=True)

        if resp.status_code != 200:
            # No robots.txt or error — assume allowed
            _robots_cache[domain] = None
            return True

        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        _robots_cache[domain] = rp
        return rp.can_fetch("*", url)

    except Exception as e:
        logger.debug(f"robots.txt check failed for {url}: {e}")
        return True  # optimistic fallback
