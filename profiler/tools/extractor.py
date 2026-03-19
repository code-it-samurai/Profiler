import logging
from urllib.parse import urlparse as _urlparse

from profiler.models.candidate import CandidateProfile
from profiler.models.enums import Platform
from profiler.agent.llm import get_llm
from profiler.tools.matcher import fuzzy_match as _name_match
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# URLs that never contain useful profile data — skip LLM extraction entirely
_URL_BLOCKLIST_DOMAINS = {
    "profile.google.com",
    "accounts.google.com",
    "rocketreach.co",
    "zoominfo.com",
    "signalhire.com",
    "lusha.com",
    "apollo.io",
    "clearbit.com",
    "peopledatalabs.com",
    "amazon.com",
    "amazon.in",
    "amazon.co.uk",
    "amazon.de",
    "amazon.fr",
    "flipkart.com",
    "ebay.com",
    "yandex.by",
    "yandex.ru",
    "yandex.com",
}

_URL_BLOCKLIST_PATH_PATTERNS = [
    "/search",
    "/login",
    "/signup",
    "/register",
    "/404",
    "/about",
    "/privacy",
    "/terms",
    "/our-team",
    "/cart",
    "/product",
    "/dp/",
    "/buy",
]

# Platform detection by URL domain
PLATFORM_MAP = {
    "facebook.com": Platform.FACEBOOK,
    "twitter.com": Platform.TWITTER,
    "x.com": Platform.TWITTER,
    "linkedin.com": Platform.LINKEDIN,
    "instagram.com": Platform.INSTAGRAM,
    "github.com": Platform.GITHUB,
    "reddit.com": Platform.REDDIT,
    "tiktok.com": Platform.TIKTOK,
    "youtube.com": Platform.YOUTUBE,
    "medium.com": Platform.MEDIUM,
    "stackoverflow.com": Platform.STACKOVERFLOW,
    "pinterest.com": Platform.PINTEREST,
    "t.me": Platform.TELEGRAM,
}


def detect_platform(url: str) -> Platform:
    """Detect the social media platform from a URL."""
    for domain, platform in PLATFORM_MAP.items():
        if domain in url:
            return platform
    return Platform.GENERIC


async def extract_profile(
    scraped_data: dict,
    target_name: str,
) -> CandidateProfile | None:
    """Extract a CandidateProfile from scraped page data.

    Uses LLM-based extraction -- sends the page text to the model
    and asks it to extract structured profile fields.

    Args:
        scraped_data: Output from scrape_page().
        target_name: The name we're searching for.

    Returns:
        CandidateProfile or None if extraction fails.
    """
    if not scraped_data.get("success") or not scraped_data.get("text"):
        return None

    # Bug 5: Skip known non-profile domains and paths
    url = scraped_data.get("url", "")
    parsed_url = _urlparse(url)
    domain = parsed_url.netloc.replace("www.", "")
    if any(blocked in domain for blocked in _URL_BLOCKLIST_DOMAINS):
        logger.info(f"Skipping blocked domain: {url}")
        return None
    path = parsed_url.path.lower()
    if any(pattern in path for pattern in _URL_BLOCKLIST_PATH_PATTERNS):
        logger.info(f"Skipping blocked path pattern: {url}")
        return None

    platform = detect_platform(scraped_data["url"])

    # Truncate text for the LLM
    page_text = scraped_data["text"][:3000]

    system_prompt = (
        "You are a data extraction agent. Given web page text, extract profile "
        "information for the target person. Return ONLY a JSON object with these fields "
        "(use null for unknown): name, location, school, employer, bio. "
        "Only extract info that clearly relates to the target person."
    )

    user_prompt = (
        f"Target name: {target_name}\n"
        f"Page URL: {scraped_data['url']}\n"
        f"Page title: {scraped_data['title']}\n\n"
        f"Page text:\n{page_text}"
    )

    try:
        from profiler.agent.llm import _gemini_rate_limit, _extract_text_content
        from profiler.config import settings as _settings

        # Rate limit for Gemini free tier
        if _settings.llm_provider == "gemini":
            await _gemini_rate_limit()

        llm = get_llm(json_mode=True)
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        import json

        raw_content = _extract_text_content(response.content)
        data = json.loads(raw_content)

        # Bug 1: Verify the target name actually appears in the page text.
        # Guards against the LLM fabricating the target name on irrelevant pages.
        page_text_lower = page_text.lower()
        target_parts = target_name.lower().split()
        name_in_page = any(
            part in page_text_lower for part in target_parts if len(part) > 2
        )
        if not name_in_page:
            logger.info(
                f"Target name '{target_name}' not found in page text for "
                f"{scraped_data['url']}"
            )
            return None

        # Normalize values: empty strings to None, dicts/lists to strings
        for key in ["name", "location", "school", "employer", "bio"]:
            val = data.get(key)
            if val is None or val == "":
                data[key] = None
            elif isinstance(val, dict):
                # LLM sometimes returns {"city": "Portland", "country": "India"}
                data[key] = ", ".join(str(v) for v in val.values() if v)
            elif isinstance(val, list):
                data[key] = ", ".join(str(v) for v in val if v)
            elif not isinstance(val, str):
                data[key] = str(val)

        profile = CandidateProfile(
            name=data.get("name", target_name),
            platform=platform,
            profile_url=scraped_data["url"],
            location=data.get("location"),
            school=data.get("school"),
            employer=data.get("employer"),
            bio=data.get("bio"),
            source_urls=[scraped_data["url"]],
        )

        # Bug 1: Reject candidates whose name doesn't match the target
        is_match, score = _name_match(profile.name, target_name, threshold=0.5)
        if not is_match:
            logger.info(
                f"Rejecting candidate '{profile.name}' — doesn't match "
                f"target '{target_name}' (score={score:.2f})"
            )
            return None
        profile.confidence = score
        return profile
    except Exception as e:
        logger.warning(f"Profile extraction failed for {scraped_data['url']}: {e}")
        # Fallback: create minimal profile from metadata — uses target_name
        # directly so name match is guaranteed (score=1.0)
        return CandidateProfile(
            name=target_name,
            platform=platform,
            profile_url=scraped_data["url"],
            bio=scraped_data.get("meta_description"),
            source_urls=[scraped_data["url"]],
            confidence=0.3,
        )
