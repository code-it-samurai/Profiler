import logging
from profiler.models.candidate import CandidateProfile
from profiler.models.enums import Platform
from profiler.agent.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Platform detection by URL domain
PLATFORM_MAP = {
    "facebook.com": Platform.FACEBOOK,
    "twitter.com": Platform.TWITTER,
    "x.com": Platform.TWITTER,
    "linkedin.com": Platform.LINKEDIN,
    "instagram.com": Platform.INSTAGRAM,
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
        llm = get_llm(num_ctx=8192, json_mode=True)
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        import json

        raw_content = response.content
        if not isinstance(raw_content, str):
            raw_content = json.dumps(raw_content)
        data = json.loads(raw_content)

        return CandidateProfile(
            name=data.get("name", target_name),
            platform=platform,
            profile_url=scraped_data["url"],
            location=data.get("location"),
            school=data.get("school"),
            employer=data.get("employer"),
            bio=data.get("bio"),
            source_urls=[scraped_data["url"]],
        )
    except Exception as e:
        logger.warning(f"Profile extraction failed for {scraped_data['url']}: {e}")
        # Fallback: create minimal profile from metadata
        return CandidateProfile(
            name=target_name,
            platform=platform,
            profile_url=scraped_data["url"],
            bio=scraped_data.get("meta_description"),
            source_urls=[scraped_data["url"]],
        )
