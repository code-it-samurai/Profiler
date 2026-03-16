from typing import TypedDict, Optional
from profiler.models.candidate import CandidateProfile
from profiler.models.profile import Profile
from profiler.models.enums import TargetType, SessionStatus


def merge_candidates(left: list, right: list) -> list:
    """Merge candidate lists, deduplicating by profile_url."""
    seen = {c.profile_url for c in left if c.profile_url}
    merged = list(left)
    for c in right:
        if c.profile_url and c.profile_url not in seen:
            merged.append(c)
            seen.add(c.profile_url)
        elif not c.profile_url:
            merged.append(c)
    return merged


class AgentState(TypedDict):
    """Full state for the LangGraph agent."""

    # Input
    target_name: str
    target_type: TargetType
    initial_context: str  # extra context from user
    session_id: str

    # Working state
    known_facts: dict  # confirmed facts from user answers
    candidates: list[CandidateProfile]
    eliminated: list[CandidateProfile]
    search_history: list[str]  # queries already executed
    narrowing_round: int

    # Direct URLs provided by user (skip search, scrape directly)
    direct_urls: list[str]  # facebook_url, linkedin_url, website, etc.

    # Inter-node communication
    current_question: Optional[dict]  # {field, question, options, reasoning}
    user_answer: Optional[str]  # answer from user (set after interrupt)
    _raw_search_results: list[dict]  # raw results passed from broad_search to extract

    # Output
    final_profile: Optional[Profile]
    status: SessionStatus
    error: Optional[str]
