"""Holehe integration: check which platforms an email is registered on.

Holehe is an optional dependency. If not installed, this module returns
empty results with a warning — the profiler continues without it.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def check_email_platforms(email: str) -> list[dict]:
    """Check which platforms an email address is registered on.

    Uses holehe to probe common platforms for account existence.
    Each result: {"platform": "instagram", "exists": True, "url": "..."}

    Args:
        email: Email address to check.

    Returns:
        List of dicts for platforms where the email exists.
        Empty list if holehe is not installed or on error.
    """
    try:
        import holehe.core as holehe_core
    except ImportError:
        logger.warning(
            "holehe not installed — skipping email platform check. "
            "Install with: uv pip install 'profiler[osint]'"
        )
        return []

    try:
        # holehe uses trio/httpx internally — run its checks
        out: list[dict] = []
        modules = holehe_core.import_submodules(holehe_core)
        websites = holehe_core.get_functions(modules)

        # holehe expects a trio event loop; run via its own mechanism
        # Each website function is async and appends to `out`
        loop = asyncio.get_event_loop()

        async def _run_checks():
            tasks = []
            for website in websites:
                tasks.append(website(email, out=out))
            await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.wait_for(_run_checks(), timeout=30)

        # Filter to only existing accounts
        results = []
        for entry in out:
            if entry.get("exists") or entry.get("rateLimit"):
                platform = entry.get("name", "unknown").lower()
                url = entry.get("emailrecovery") or entry.get("url") or ""
                results.append(
                    {
                        "platform": platform,
                        "exists": bool(entry.get("exists")),
                        "url": url,
                    }
                )

        logger.info(f"Holehe: found {len(results)} platforms for {email}")
        return results

    except asyncio.TimeoutError:
        logger.warning(f"Holehe timed out checking {email}")
        return []
    except Exception as e:
        logger.warning(f"Holehe check failed for {email}: {e}")
        return []
