import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from typing import Optional

from profiler.config import settings
from profiler.agent.llm import validated_llm_call, get_llm
from profiler.agent.state import AgentState
from profiler.models.candidate import CandidateProfile
from profiler.models.enums import SessionStatus, Platform
from profiler.tools.search import google_search
from profiler.tools.scraper import scrape_page
from profiler.tools.extractor import extract_profile
from profiler.tools.matcher import fuzzy_match
from profiler.agent.progress import emit

logger = logging.getLogger(__name__)

# Load Jinja2 templates using absolute path relative to this file
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_template_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


# --- Pydantic models for LLM responses ---


class SearchQueries(BaseModel):
    queries: list[dict]  # each has: query, site_filter, purpose


class NarrowingDecision(BaseModel):
    field: str
    question: str
    options: Optional[list[str]] = None
    reasoning: str
    expected_elimination_pct: float


class CompilationResult(BaseModel):
    summary: str
    locations: list[str] = []
    education: list[str] = []
    employment: list[str] = []
    associated_entities: list[str] = []
    confidence_score: float = 0.0


# --- Node Functions ---


async def broad_search(state: AgentState) -> dict:
    """Node 1: Generate search queries, execute them, and collect direct URLs.

    Also pre-populates known_facts from any structured user input (location,
    school, employer) so the narrowing loop never re-asks those fields.
    """
    logger.info(f"BROAD_SEARCH: Searching for '{state['target_name']}'")
    emit("broad_search", f'Starting search for "{state["target_name"]}"')

    # --- A) Pre-populate known_facts from structured user input ---
    known_facts = dict(state.get("known_facts", {}))
    initial_context = state.get("initial_context", "")

    # These may already be set by the CLI/API layer, but merge defensively
    for field in ["location", "school", "employer"]:
        value = known_facts.get(field)
        if value:
            known_facts[field] = value

    # --- B) Collect direct URLs from user input ---
    direct_urls = list(state.get("direct_urls", []))

    # --- C) Build enriched search queries using structured fields ---
    name = state["target_name"]
    enriched_queries: list[dict] = []

    if known_facts.get("location"):
        enriched_queries.append(
            {
                "query": f"{name} {known_facts['location']}",
                "site_filter": None,
                "purpose": "name+location",
            }
        )
    if known_facts.get("employer"):
        enriched_queries.append(
            {
                "query": f"{name} {known_facts['employer']}",
                "site_filter": None,
                "purpose": "name+employer",
            }
        )
    if known_facts.get("school"):
        enriched_queries.append(
            {
                "query": f"{name} {known_facts['school']}",
                "site_filter": None,
                "purpose": "name+school",
            }
        )
    # Email exact-match search
    email_hint = None
    if initial_context:
        # Extract email from context string if present (set by CLI)
        for part in initial_context.split(";"):
            part = part.strip()
            if part.startswith("email:"):
                email_hint = part.split(":", 1)[1].strip()
    if email_hint:
        enriched_queries.append(
            {
                "query": f'"{email_hint}"',
                "site_filter": None,
                "purpose": "email exact match",
            }
        )

    if enriched_queries:
        emit(
            "broad_search",
            f"Built {len(enriched_queries)} enriched queries from known facts",
            10,
        )
    if direct_urls:
        emit(
            "broad_search",
            f"Will scrape {len(direct_urls)} direct URLs: {', '.join(direct_urls)}",
            10,
        )

    # --- D) Ask LLM for additional queries ---
    emit("broad_search", "Asking LLM to generate search queries...", 15)
    system_tpl = _template_env.get_template("system.jinja2")
    system_prompt = system_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=initial_context,
        known_facts=known_facts,
    )

    query_tpl = _template_env.get_template("search_query.jinja2")
    query_prompt = query_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=initial_context,
        known_facts=known_facts,
        search_history=state.get("search_history", []),
    )

    try:
        search_plan = await validated_llm_call(
            system_prompt=system_prompt,
            user_prompt=query_prompt,
            response_model=SearchQueries,
            num_ctx=4096,
        )
    except ValueError:
        # Fallback: manual query generation
        emit("broad_search", "LLM query generation failed, using fallback queries", 30)
        search_plan = SearchQueries(
            queries=[
                {"query": name, "site_filter": None, "purpose": "general"},
                {"query": name, "site_filter": "facebook.com", "purpose": "facebook"},
                {"query": name, "site_filter": "twitter.com", "purpose": "twitter"},
                {"query": name, "site_filter": "linkedin.com", "purpose": "linkedin"},
            ]
        )

    # Merge enriched queries with LLM-generated queries (enriched first)
    all_queries = enriched_queries + search_plan.queries
    emit("broad_search", f"Generated {len(all_queries)} search queries total", 35)

    # --- E) Execute all searches concurrently ---
    all_results = []
    new_search_history = list(state.get("search_history", []))

    tasks = []
    for q in all_queries:
        query_str = q["query"] if isinstance(q, dict) else q.query
        site = (
            q.get("site_filter")
            if isinstance(q, dict)
            else getattr(q, "site_filter", None)
        )
        if query_str not in new_search_history:
            tasks.append(google_search(query_str, num_results=10, site_filter=site))
            new_search_history.append(query_str)

    emit("broad_search", f"Executing {len(tasks)} search queries concurrently...", 40)
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results_lists:
        if isinstance(result, list):
            all_results.extend(result)
        elif isinstance(result, Exception):
            logger.warning(f"Search query failed: {result}")
            # Continue with other results — don't fail

    emit(
        "broad_search",
        f"Search complete: {len(all_results)} results from {len(tasks)} queries",
        90,
    )

    # --- F) Only fail if no results AND no direct URLs ---
    has_direct_urls = len(direct_urls) > 0
    if not all_results and not has_direct_urls:
        return {
            "status": SessionStatus.FAILED,
            "error": "No search results found for this name.",
            "known_facts": known_facts,
        }

    return {
        "search_history": new_search_history,
        "known_facts": known_facts,
        "direct_urls": direct_urls,
        "status": SessionStatus.SEARCHING,
        "_raw_search_results": all_results,
    }


async def extract_and_normalize(state: AgentState) -> dict:
    """Node 2: Scrape search result URLs and extract candidate profiles.

    Merges direct_urls (user-provided, high priority) with search results.
    """
    raw_results = list(state.get("_raw_search_results", []))
    direct_urls = state.get("direct_urls", [])

    # Prepend direct URLs as high-priority scrape targets
    for url in direct_urls:
        raw_results.insert(0, {"title": "", "url": url, "snippet": ""})

    logger.info(
        f"EXTRACT: Processing {len(raw_results)} results "
        f"({len(direct_urls)} direct URLs + {len(raw_results) - len(direct_urls)} search results)"
    )
    emit(
        "extract",
        f"Processing {len(raw_results)} URLs ({len(direct_urls)} direct, {len(raw_results) - len(direct_urls)} from search)",
        0,
    )

    # Deduplicate URLs
    seen_urls = set()
    unique_results = []
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    # Scrape pages concurrently (limit concurrency to avoid rate limits)
    semaphore = asyncio.Semaphore(5)

    async def scrape_with_limit(url):
        async with semaphore:
            return await scrape_page(url, max_chars=6000)

    # Only scrape top 20 results to keep things manageable
    to_scrape = unique_results[:20]
    emit(
        "scraping", f"Scraping {len(to_scrape)} unique URLs (max concurrency: 5)...", 10
    )
    scrape_tasks = [scrape_with_limit(r["url"]) for r in to_scrape]
    scraped_pages = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    success_count = sum(
        1 for p in scraped_pages if isinstance(p, dict) and p.get("success")
    )
    fail_count = len(scraped_pages) - success_count
    emit(
        "scraping",
        f"Scraped {success_count} pages successfully, {fail_count} failed",
        50,
    )

    # Extract profiles from scraped pages (LLM call per page — this is slow)
    candidates = list(state.get("candidates", []))
    total_to_extract = sum(
        1 for p in scraped_pages if isinstance(p, dict) and p.get("success")
    )
    extracted_count = 0
    for page_data in scraped_pages:
        if isinstance(page_data, dict) and page_data.get("success"):
            extracted_count += 1
            url = page_data.get("url", "?")
            pct = 50 + int((extracted_count / max(total_to_extract, 1)) * 45)
            emit(
                "extracting",
                f"[{extracted_count}/{total_to_extract}] Extracting profile from {url[:60]}...",
                pct,
            )
            profile = await extract_profile(page_data, state["target_name"])
            if profile:
                candidates.append(profile)

    # Deduplicate candidates by profile_url
    seen = set()
    deduped = []
    for c in candidates:
        key = c.profile_url or str(c.id)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    logger.info(f"EXTRACT: Found {len(deduped)} unique candidates")
    emit(
        "extract",
        f"Extraction complete: {len(deduped)} unique candidate profiles found",
        100,
    )

    return {
        "candidates": deduped,
        "status": SessionStatus.NARROWING,
    }


async def analyze_candidates(state: AgentState) -> dict:
    """Node 3: Analyze candidates and decide the best narrowing question."""
    candidates = state.get("candidates", [])
    known_facts = state.get("known_facts", {})

    logger.info(
        f"ANALYZE: {len(candidates)} candidates, round {state.get('narrowing_round', 0)}"
    )
    emit(
        "analyze",
        f"Analyzing {len(candidates)} candidates (round {state.get('narrowing_round', 0)})",
        0,
    )

    # Build field statistics
    fields = ["location", "school", "employer"]
    field_stats = {}
    for field in fields:
        if field in known_facts:
            continue  # skip already-known fields
        values = [getattr(c, field) for c in candidates if getattr(c, field)]
        if values:
            counter = Counter(values)
            field_stats[field] = {
                "unique_count": len(counter),
                "top_values": [v for v, _ in counter.most_common(5)],
                "coverage": len(values) / len(candidates),
            }

    if not field_stats:
        # No more fields to narrow on -- go to deep scrape
        emit("analyze", "No more fields to narrow on, moving to deep scrape", 100)
        return {"status": SessionStatus.COMPILING}

    for field_name, stats in field_stats.items():
        emit(
            "analyze",
            f"  Field '{field_name}': {stats['unique_count']} unique values, top: {', '.join(stats['top_values'][:3])}",
            30,
        )

    emit("analyze", "Asking LLM to pick the best narrowing question...", 50)
    system_tpl = _template_env.get_template("system.jinja2")
    system_prompt = system_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=state.get("initial_context", ""),
        known_facts=known_facts,
    )

    narrowing_tpl = _template_env.get_template("narrowing.jinja2")
    narrowing_prompt = narrowing_tpl.render(
        candidates=candidates[:50],  # cap to prevent context overflow
        field_stats=field_stats,
        known_facts=known_facts,
    )

    try:
        decision = await validated_llm_call(
            system_prompt=system_prompt,
            user_prompt=narrowing_prompt,
            response_model=NarrowingDecision,
            num_ctx=16384,
        )
        emit(
            "analyze",
            f"Best question: '{decision.question}' (field: {decision.field})",
            100,
        )
        return {
            "current_question": {
                "field": decision.field,
                "question": decision.question,
                "options": decision.options,
                "reasoning": decision.reasoning,
            },
            "status": SessionStatus.ASKING_USER,
        }
    except ValueError as e:
        # Fallback: ask about the field with most unique values
        best_field = max(field_stats, key=lambda f: field_stats[f]["unique_count"])
        top_vals = field_stats[best_field]["top_values"][:4]
        return {
            "current_question": {
                "field": best_field,
                "question": f"Can you tell me the {best_field} of the person you're looking for?",
                "options": top_vals if top_vals else None,
                "reasoning": f"Fallback: {best_field} has the most unique values.",
            },
            "status": SessionStatus.ASKING_USER,
        }


async def filter_candidates(state: AgentState) -> dict:
    """Node 5: Filter candidates based on user's answer."""
    answer = state.get("user_answer") or ""
    current_question = state.get("current_question") or {}
    field = current_question.get("field", "")
    candidates = state.get("candidates", [])
    eliminated = list(state.get("eliminated", []))
    known_facts = dict(state.get("known_facts", {}))

    logger.info(f"FILTER: Applying answer '{answer}' to field '{field}'")
    emit("filter", f'Filtering {len(candidates)} candidates by {field}="{answer}"', 0)

    # Add to known facts
    known_facts[field] = answer

    # Filter candidates using fuzzy matching
    kept = []
    for c in candidates:
        candidate_value = getattr(c, field, None)
        if candidate_value is None:
            # Unknown field -- keep the candidate (benefit of the doubt)
            kept.append(c)
        else:
            is_match, score = fuzzy_match(
                candidate_value, answer, settings.fuzzy_match_threshold
            )
            if is_match:
                c.confidence = max(c.confidence, score)
                kept.append(c)
            else:
                eliminated.append(c)

    round_num = state.get("narrowing_round", 0) + 1

    logger.info(f"FILTER: {len(kept)} kept, {len(eliminated)} total eliminated")
    emit(
        "filter",
        f"Result: {len(kept)} candidates kept, {len(candidates) - len(kept)} eliminated this round",
        100,
    )

    return {
        "candidates": kept,
        "eliminated": eliminated,
        "known_facts": known_facts,
        "narrowing_round": round_num,
        "user_answer": None,  # clear for next round
        "current_question": None,
        "status": SessionStatus.NARROWING,
    }


async def deep_scrape(state: AgentState) -> dict:
    """Node 6: Deep scrape the final shortlisted candidates."""
    candidates = state.get("candidates", [])
    to_scrape = candidates[:3]
    logger.info(f"DEEP_SCRAPE: Enriching {len(to_scrape)} candidates")
    emit(
        "deep_scrape",
        f"Deep scraping {len(to_scrape)} final candidates for full profiles",
        0,
    )

    enriched = []
    for i, candidate in enumerate(to_scrape):
        pct = int(((i + 1) / max(len(to_scrape), 1)) * 100)
        if candidate.profile_url:
            emit(
                "deep_scrape",
                f"[{i + 1}/{len(to_scrape)}] Scraping {candidate.profile_url[:60]}...",
                pct,
            )
            scraped = await scrape_page(candidate.profile_url, max_chars=12000)
            if scraped.get("success"):
                emit(
                    "deep_scrape",
                    f"[{i + 1}/{len(to_scrape)}] Extracting enriched data via LLM...",
                    pct,
                )
                # Re-extract with more data
                enriched_profile = await extract_profile(scraped, state["target_name"])
                if enriched_profile:
                    # Merge: keep original data, add new fields
                    for field in ["location", "school", "employer", "bio"]:
                        new_val = getattr(enriched_profile, field)
                        if new_val and not getattr(candidate, field):
                            setattr(candidate, field, new_val)
                    candidate.raw_data.update(enriched_profile.raw_data)
            else:
                emit(
                    "deep_scrape",
                    f"[{i + 1}/{len(to_scrape)}] Scrape failed for {candidate.profile_url[:40]}",
                    pct,
                )
        enriched.append(candidate)
    emit("deep_scrape", f"Deep scrape complete for {len(enriched)} candidates", 100)

    return {
        "candidates": enriched,
        "status": SessionStatus.COMPILING,
    }


async def compile_profile(state: AgentState) -> dict:
    """Node 7: Compile the final profile dossier using the LLM."""
    candidates = state.get("candidates", [])
    known_facts = state.get("known_facts", {})

    logger.info(f"COMPILE: Building profile from {len(candidates)} candidates")
    emit(
        "compile",
        f"Compiling final profile from {len(candidates)} candidates and {len(known_facts)} known facts",
        0,
    )
    emit("compile", "Asking LLM to synthesize all data into a dossier...", 20)

    system_tpl = _template_env.get_template("system.jinja2")
    system_prompt = system_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=state.get("initial_context", ""),
        known_facts=known_facts,
    )

    compile_tpl = _template_env.get_template("compilation.jinja2")
    compile_prompt = compile_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        candidates=candidates,
        known_facts=known_facts,
    )

    try:
        result = await validated_llm_call(
            system_prompt=system_prompt,
            user_prompt=compile_prompt,
            response_model=CompilationResult,
            num_ctx=16384,
        )

        from profiler.models.profile import Profile, SocialProfile, Source

        social_profiles = []
        for c in candidates:
            if c.profile_url:
                social_profiles.append(
                    SocialProfile(
                        platform=c.platform.value,
                        url=c.profile_url,
                        bio=c.bio,
                    )
                )

        sources = []
        for c in candidates:
            for url in c.source_urls:
                sources.append(Source(url=url))

        emit(
            "compile",
            f"LLM response received, building profile object (confidence: {result.confidence_score:.0%})",
            80,
        )

        profile = Profile(
            target_name=state["target_name"],
            target_type=state["target_type"],
            summary=result.summary,
            social_profiles=social_profiles,
            locations=result.locations,
            education=result.education,
            employment=result.employment,
            associated_entities=result.associated_entities,
            sources=sources,
            confidence_score=result.confidence_score,
        )

        emit(
            "compile",
            f"Profile complete: {len(social_profiles)} social profiles, {len(sources)} sources",
            100,
        )
        return {
            "final_profile": profile,
            "status": SessionStatus.DONE,
        }
    except Exception as e:
        logger.error(f"Profile compilation failed: {e}")
        emit("compile", f"Compilation failed: {e}", 100)
        return {
            "status": SessionStatus.FAILED,
            "error": f"Failed to compile profile: {e}",
        }
