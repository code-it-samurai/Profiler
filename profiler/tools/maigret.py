"""Maigret integration: search for a username across platforms.

Maigret is an optional dependency, run as a subprocess for reliability.
If not installed, this module returns empty results with a warning.
"""

import asyncio
import json
import logging
import shutil

from profiler.models.candidate import CandidateProfile
from profiler.models.enums import Platform

logger = logging.getLogger(__name__)

# Map common Maigret site names to Platform enum values
_SITE_TO_PLATFORM: dict[str, Platform] = {
    "GitHub": Platform.GITHUB,
    "Twitter": Platform.TWITTER,
    "X": Platform.TWITTER,
    "Facebook": Platform.FACEBOOK,
    "Instagram": Platform.INSTAGRAM,
    "LinkedIn": Platform.LINKEDIN,
    "Reddit": Platform.REDDIT,
    "TikTok": Platform.TIKTOK,
    "YouTube": Platform.YOUTUBE,
    "Medium": Platform.MEDIUM,
    "StackOverflow": Platform.STACKOVERFLOW,
    "Stack Overflow": Platform.STACKOVERFLOW,
    "Pinterest": Platform.PINTEREST,
    "Telegram": Platform.TELEGRAM,
}


async def search_username(username: str) -> list[CandidateProfile]:
    """Search for a username across platforms using Maigret.

    Runs maigret as a subprocess with JSON output, parses results,
    and returns CandidateProfile objects for each found account.

    Args:
        username: Username to search for.

    Returns:
        List of CandidateProfile objects for found accounts.
        Empty list if maigret is not installed or on error.
    """
    maigret_bin = shutil.which("maigret")
    if not maigret_bin:
        logger.warning(
            "maigret not installed — skipping username search. "
            "Install with: uv pip install 'profiler[osint]'"
        )
        return []

    try:
        proc = await asyncio.create_subprocess_exec(
            maigret_bin,
            username,
            "--json",
            "simple",
            "--timeout",
            "10",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(f"Maigret timed out searching for '{username}'")
            return []

        if proc.returncode != 0:
            logger.warning(
                f"Maigret exited with code {proc.returncode} for '{username}': "
                f"{stderr.decode(errors='replace')[:200]}"
            )
            # Still try to parse stdout — maigret sometimes writes results before erroring
            if not stdout:
                return []

        # Parse JSON output
        try:
            data = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError:
            logger.warning(f"Maigret returned invalid JSON for '{username}'")
            return []

        # Convert results to CandidateProfile objects
        candidates = []
        results = data if isinstance(data, list) else data.get("results", [])

        for entry in results:
            if not isinstance(entry, dict):
                continue

            site_name = entry.get("sitename") or entry.get("site_name") or ""
            url = entry.get("url") or entry.get("link") or ""
            status = entry.get("status") or ""

            # Only include confirmed/claimed results
            if "Claimed" not in str(status) and "Found" not in str(status):
                continue

            if not url:
                continue

            platform = _SITE_TO_PLATFORM.get(site_name, Platform.GENERIC)

            candidates.append(
                CandidateProfile(
                    name=username,
                    platform=platform,
                    profile_url=url,
                    usernames=[username],
                    source_tool="maigret",
                    source_urls=[url],
                    confidence=0.5,
                )
            )

        logger.info(f"Maigret: found {len(candidates)} profiles for '{username}'")
        return candidates

    except Exception as e:
        logger.warning(f"Maigret search failed for '{username}': {e}")
        return []
