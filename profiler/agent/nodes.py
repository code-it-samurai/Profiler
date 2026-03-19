import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, field_validator
from typing import Optional

from profiler.config import settings
from profiler.agent.llm import validated_llm_call, get_llm
from profiler.agent.state import AgentState
from profiler.models.candidate import CandidateProfile
from profiler.models.enums import SessionStatus, Platform, TargetType
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
    options: Optional[list] = None  # accept any list, we'll clean it
    reasoning: str = ""  # LLM sometimes omits this
    expected_elimination_pct: float = 0.5  # LLM sometimes omits this

    @field_validator("options", mode="before")
    @classmethod
    def clean_options(cls, v):
        if v is None:
            return None
        # Filter out None values and non-strings, keep only non-empty strings
        cleaned = [str(x) for x in v if x is not None and str(x).strip()]
        return cleaned if cleaned else None


class CompilationResult(BaseModel):
    summary: str
    locations: list[str] = []
    education: list[str] = []
    employment: list[str] = []
    associated_entities: list[str] = []
    confidence_score: float = 0.0
    candidate_profiles: list[dict] = []


# --- Node Functions ---


async def broad_search(state: AgentState) -> dict:
    """Node 1: Generate search queries, execute them, and collect direct URLs.

    Also pre-populates known_facts from any structured user input (location,
    school, employer) so the narrowing loop never re-asks those fields.
    """
    logger.info(f"BROAD_SEARCH: Searching for '{state['target_name']}'")
    emit("discovery", "start", f'Searching for "{state["target_name"]}"')

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
            "discovery",
            "info",
            f"Built {len(enriched_queries)} enriched queries from known facts",
        )
    if direct_urls:
        emit("discovery", "info", f"Will scrape {len(direct_urls)} direct URLs")

    # --- D) Ask LLM for additional queries ---
    emit("discovery", "task_start", "LLM Query Generation")
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
        emit(
            "discovery",
            "task_fail",
            "LLM Query Generation",
            meta={"error": "using fallback queries"},
        )
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
    emit(
        "discovery",
        "task_done",
        "LLM Query Generation",
        meta={"count": len(all_queries)},
    )

    # --- E) Execute all searches + external tools concurrently ---
    all_results = []
    new_search_history = list(state.get("search_history", []))
    external_candidates = []  # structured results from tools that bypass LLM

    # DDG tasks (existing logic)
    ddg_tasks = []
    for q in all_queries:
        # Extract query string defensively — LLM may use varying key names
        if isinstance(q, dict):
            query_str = q.get("query") or q.get("search_query") or q.get("q") or ""
            site = q.get("site_filter")
        else:
            query_str = getattr(q, "query", "") or ""
            site = getattr(q, "site_filter", None)
        if not query_str:
            logger.warning(f"Skipping malformed search query: {q}")
            continue
        if query_str not in new_search_history:
            ddg_tasks.append(google_search(query_str, num_results=10, site_filter=site))
            new_search_history.append(query_str)

    # External tool tasks (all return structured data, no LLM needed)
    external_tasks = []
    external_task_names = []

    if email_hint:
        from profiler.tools.holehe import check_email_platforms

        external_tasks.append(check_email_platforms(email_hint))
        external_task_names.append("holehe")

    employer_domain = known_facts.get("employer", "")
    if employer_domain and "." in employer_domain:
        from profiler.tools.harvester import harvest

        external_tasks.append(harvest(employer_domain))
        external_task_names.append("harvester")

    if state["target_type"] == TargetType.COMPANY:
        from profiler.tools.opencorporates import search_company

        external_tasks.append(search_company(state["target_name"]))
        external_task_names.append("opencorporates")

    emit("discovery", "task_start", "DDG Search")

    # Run ALL tasks concurrently
    all_task_results = await asyncio.gather(
        asyncio.gather(*ddg_tasks, return_exceptions=True),
        asyncio.gather(*external_tasks, return_exceptions=True),
    )

    ddg_results_lists = all_task_results[0]
    external_results_list = all_task_results[1] if external_tasks else []

    # Process DDG results (existing)
    for result in ddg_results_lists:
        if isinstance(result, list):
            all_results.extend(result)
        elif isinstance(result, Exception):
            logger.warning(f"Search query failed: {result}")

    # Process external tool results
    data_sources_used = ["ddg"]
    for i, result in enumerate(external_results_list):
        tool_name = (
            external_task_names[i] if i < len(external_task_names) else "unknown"
        )
        if isinstance(result, Exception):
            logger.warning(f"{tool_name} failed: {result}")
            continue
        data_sources_used.append(tool_name)

        if tool_name == "holehe" and isinstance(result, list):
            # Holehe returns [{platform, exists, url}] — convert to search results
            for entry in result:
                if entry.get("exists") and entry.get("url"):
                    all_results.append(
                        {
                            "title": f"{state['target_name']} on {entry['platform']}",
                            "url": entry["url"],
                            "snippet": f"Account confirmed by holehe on {entry['platform']}",
                        }
                    )

        elif tool_name == "harvester" and isinstance(result, dict):
            # theHarvester returns {emails, urls, hosts}
            for url in result.get("urls", []):
                all_results.append(
                    {"title": "", "url": url, "snippet": "Found by theHarvester"}
                )
            discovered_emails = result.get("emails", [])
            if discovered_emails:
                emit(
                    "discovery",
                    "info",
                    f"theHarvester found {len(discovered_emails)} emails",
                )

        elif tool_name == "opencorporates" and isinstance(result, list):
            # OpenCorporates returns structured company data — create candidates directly
            for company in result:
                external_candidates.append(
                    CandidateProfile(
                        name=company.get("name", state["target_name"]),
                        platform=Platform.GENERIC,
                        profile_url=company.get("url"),
                        location=company.get("registered_address"),
                        employer=company.get("name"),
                        source_tool="opencorporates",
                        source_urls=[company.get("url", "")],
                        confidence=0.6,
                    )
                )

    emit(
        "discovery",
        "task_done",
        "DDG Search",
        meta={"count": len(all_results), "queries": len(ddg_tasks)},
    )
    # Emit results for each external tool
    for i, result in enumerate(external_results_list):
        tool_name = (
            external_task_names[i] if i < len(external_task_names) else "unknown"
        )
        if not isinstance(result, Exception):
            if tool_name == "holehe" and isinstance(result, list):
                emit(
                    "discovery",
                    "task_done",
                    "Holehe",
                    meta={"count": len([e for e in result if e.get("exists")])},
                )
            elif tool_name == "harvester" and isinstance(result, dict):
                emit(
                    "discovery",
                    "task_done",
                    "theHarvester",
                    meta={
                        "emails": len(result.get("emails", [])),
                        "urls": len(result.get("urls", [])),
                    },
                )
            elif tool_name == "opencorporates" and isinstance(result, list):
                emit(
                    "discovery",
                    "task_done",
                    "OpenCorporates",
                    meta={"count": len(result)},
                )
        else:
            emit("discovery", "task_fail", tool_name, meta={"error": str(result)[:80]})
    emit(
        "discovery",
        "phase_done",
        "Discovery complete",
        meta={"results": len(all_results), "tools": ", ".join(data_sources_used)},
    )

    # --- F) Only fail if no results AND no direct URLs AND no external candidates ---
    has_direct_urls = len(direct_urls) > 0
    if not all_results and not has_direct_urls and not external_candidates:
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
        "_external_candidates": external_candidates,
        "data_sources_used": data_sources_used,
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
    emit("extract", "start", f"Processing {len(raw_results)} URLs")

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
    emit("extract", "task_start", "Scraping")
    scrape_tasks = [scrape_with_limit(r["url"]) for r in to_scrape]
    scraped_pages = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    success_count = sum(
        1 for p in scraped_pages if isinstance(p, dict) and p.get("success")
    )
    fail_count = len(scraped_pages) - success_count
    # Count robots.txt blocks
    robots_blocked = sum(
        1
        for p in scraped_pages
        if isinstance(p, dict) and p.get("error") == "blocked_by_robots_txt"
    )
    emit(
        "extract",
        "task_done",
        "Scraping",
        meta={"success": success_count, "failed": fail_count, "total": len(to_scrape)},
    )
    if robots_blocked > 0:
        emit(
            "extract",
            "task_done",
            "robots.txt",
            meta={"allowed": success_count, "blocked": robots_blocked},
        )

    # Extract profiles from scraped pages concurrently (LLM call per page)
    candidates = list(state.get("candidates", []))
    successful_pages = [
        p for p in scraped_pages if isinstance(p, dict) and p.get("success")
    ]
    emit("extract", "task_start", "LLM Extract")

    extract_semaphore = asyncio.Semaphore(3)

    async def extract_with_limit(page_data):
        async with extract_semaphore:
            return await extract_profile(page_data, state["target_name"])

    extract_tasks = [extract_with_limit(p) for p in successful_pages]
    profiles = await asyncio.gather(*extract_tasks, return_exceptions=True)

    for p in profiles:
        if isinstance(p, CandidateProfile):
            candidates.append(p)
        elif isinstance(p, Exception):
            logger.warning(f"Profile extraction failed: {p}")

    extracted_ok = sum(1 for p in profiles if isinstance(p, CandidateProfile))
    emit(
        "extract",
        "task_done",
        "LLM Extract",
        meta={"count": extracted_ok, "total": len(successful_pages)},
    )

    # --- Merge external candidates from broad_search ---
    external_cands = state.get("_external_candidates", [])
    if external_cands:
        emit(
            "extract", "info", f"Merging {len(external_cands)} external tool candidates"
        )
        candidates.extend(external_cands)

    # --- Maigret enrichment: extract usernames from known social URL patterns ---
    from urllib.parse import urlparse as _urlparse

    _USERNAME_URL_PATTERNS = {
        "twitter.com": lambda p: p[0] if len(p) == 1 else None,
        "x.com": lambda p: p[0] if len(p) == 1 else None,
        "github.com": lambda p: p[0] if len(p) == 1 else None,
        "instagram.com": lambda p: p[0] if len(p) == 1 else None,
        "medium.com": lambda p: p[0].lstrip("@") if len(p) == 1 else None,
        "reddit.com": lambda p: p[1] if len(p) >= 2 and p[0] in ("user", "u") else None,
    }

    discovered_usernames = set()
    for c in candidates:
        if c.profile_url:
            parsed = _urlparse(c.profile_url)
            domain = parsed.netloc.replace("www.", "")
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]

            for pattern_domain, extractor in _USERNAME_URL_PATTERNS.items():
                if pattern_domain in domain:
                    username = extractor(path_parts)
                    if (
                        username
                        and 2 < len(username) < 30
                        and not username.startswith("search")
                    ):
                        discovered_usernames.add(username)
                    break

        # Also use any usernames already on the candidate
        for u in getattr(c, "usernames", []):
            if u and 2 < len(u) < 30:
                discovered_usernames.add(u)

    if discovered_usernames:
        from profiler.tools.maigret import search_username

        emit("extract", "task_start", "Maigret")
        # Limit to 3 usernames to avoid long waits
        usernames_to_check = list(discovered_usernames)[:3]
        maigret_tasks = [search_username(u) for u in usernames_to_check]
        maigret_results = await asyncio.gather(*maigret_tasks, return_exceptions=True)
        maigret_count = 0
        for result in maigret_results:
            if isinstance(result, list):
                candidates.extend(result)
                maigret_count += len(result)
            elif isinstance(result, Exception):
                logger.warning(f"Maigret search failed: {result}")
        emit(
            "extract",
            "task_done",
            "Maigret",
            meta={"count": maigret_count, "total": len(usernames_to_check)},
        )

    # Deduplicate candidates by profile_url
    seen = set()
    deduped = []
    for c in candidates:
        key = c.profile_url or str(c.id)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    logger.info(f"EXTRACT: Found {len(deduped)} unique candidates")

    # Log field coverage for diagnostics
    field_coverage = {}
    for field in ["location", "school", "employer"]:
        count = sum(1 for c in deduped if getattr(c, field))
        field_coverage[field] = count
        logger.info(
            f"EXTRACT: Field '{field}' populated in {count}/{len(deduped)} candidates"
        )

    emit(
        "extract",
        "info",
        "Field coverage",
        meta={**field_coverage, "total": len(deduped)},
    )
    emit(
        "extract",
        "phase_done",
        f"{len(deduped)} unique candidates",
        meta={"candidates": len(deduped)},
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
    emit("narrowing", "start", f"Analyzing {len(candidates)} candidates")

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
        emit("narrowing", "info", "No more fields to narrow on, moving to deep scrape")
        return {"status": SessionStatus.COMPILING}

    for field_name, stats in field_stats.items():
        emit(
            "narrowing",
            "info",
            f"Field '{field_name}': {stats['unique_count']} unique values, top: {', '.join(stats['top_values'][:3])}",
        )

    emit("narrowing", "task_start", "LLM Narrowing Analysis")
    system_tpl = _template_env.get_template("system.jinja2")
    system_prompt = system_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=state.get("initial_context", ""),
        known_facts=known_facts,
    )

    narrowing_tpl = _template_env.get_template("narrowing.jinja2")
    narrowing_prompt = narrowing_tpl.render(
        candidates=candidates[:100],  # cap to prevent context overflow
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
            "narrowing",
            "task_done",
            "LLM Narrowing Analysis",
            meta={"field": decision.field},
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
    emit("narrowing", "task_start", "Filter")

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

    # Build narrowing history entry
    history = list(state.get("narrowing_history", []))
    history.append(
        {
            "round": round_num,
            "before": len(candidates),
            "after": len(kept),
            "field": field,
            "answer": answer,
        }
    )

    logger.info(f"FILTER: {len(kept)} kept, {len(eliminated)} total eliminated")
    emit(
        "narrowing",
        "task_done",
        "Filter",
        meta={
            "before": len(candidates),
            "after": len(kept),
            "field": field,
            "answer": answer,
        },
    )

    return {
        "candidates": kept,
        "eliminated": eliminated,
        "known_facts": known_facts,
        "narrowing_round": round_num,
        "narrowing_history": history,
        "user_answer": None,  # clear for next round
        "current_question": None,
        "status": SessionStatus.NARROWING,
    }


async def deep_scrape(state: AgentState) -> dict:
    """Node 6: Deep scrape candidates that have missing key fields."""
    candidates = state.get("candidates", [])
    key_fields = ["location", "school", "employer", "bio"]

    # Partition: candidates missing key fields vs already complete
    needs_scrape = []
    already_complete = []
    for c in candidates:
        missing = [f for f in key_fields if not getattr(c, f)]
        if missing:
            needs_scrape.append(c)
        else:
            already_complete.append(c)

    # Sort by confidence descending, take top N
    needs_scrape.sort(key=lambda c: c.confidence, reverse=True)
    to_scrape = needs_scrape[: settings.deep_scrape_limit]
    skipped = needs_scrape[settings.deep_scrape_limit :]

    logger.info(
        f"DEEP_SCRAPE: {len(to_scrape)} to scrape, "
        f"{len(already_complete)} already complete, {len(skipped)} skipped"
    )
    emit("deep_enrich", "start", "Deep enrichment")
    emit(
        "deep_enrich",
        "info",
        f"{len(to_scrape)} need enrichment, {len(already_complete)} already complete",
    )

    scrape_semaphore = asyncio.Semaphore(5)

    async def enrich_candidate(candidate):
        async with scrape_semaphore:
            if not candidate.profile_url:
                return candidate
            scraped = await scrape_page(candidate.profile_url, max_chars=12000)
            if scraped.get("success"):
                enriched_profile = await extract_profile(scraped, state["target_name"])
                if enriched_profile:
                    for field in key_fields:
                        new_val = getattr(enriched_profile, field)
                        if new_val and not getattr(candidate, field):
                            setattr(candidate, field, new_val)
                    candidate.raw_data.update(enriched_profile.raw_data)
            return candidate

    emit("deep_enrich", "task_start", "Deep Scrape")
    results = await asyncio.gather(
        *[enrich_candidate(c) for c in to_scrape], return_exceptions=True
    )

    enriched = []
    for r in results:
        if isinstance(r, CandidateProfile):
            enriched.append(r)
        elif isinstance(r, Exception):
            logger.warning(f"Deep scrape failed for candidate: {r}")
    emit(
        "deep_enrich",
        "task_done",
        "Deep Scrape",
        meta={"count": len(enriched), "total": len(to_scrape)},
    )

    # --- Photon crawl: discover additional links from candidate websites ---
    social_domains = [
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "linkedin.com",
    ]
    website_candidates = [
        c
        for c in enriched
        if c.profile_url and not any(d in c.profile_url for d in social_domains)
    ][:3]  # max 3 to avoid long waits

    if website_candidates:
        from profiler.tools.photon import crawl_url

        emit("deep_enrich", "task_start", "Photon")
        photon_tasks = [crawl_url(c.profile_url, depth=1) for c in website_candidates]
        photon_results = await asyncio.gather(*photon_tasks, return_exceptions=True)

        for i, result in enumerate(photon_results):
            if isinstance(result, dict):
                candidate = website_candidates[i]
                # Add discovered emails
                for email in result.get("emails", []):
                    if email not in candidate.emails:
                        candidate.emails.append(email)
                # Add discovered social URLs
                for url in result.get("social_urls", []):
                    if url not in candidate.discovered_urls:
                        candidate.discovered_urls.append(url)
            elif isinstance(result, Exception):
                logger.warning(f"Photon crawl failed: {result}")
        photon_emails = sum(len(c.emails) for c in website_candidates)
        photon_urls = sum(len(c.discovered_urls) for c in website_candidates)
        emit(
            "deep_enrich",
            "task_done",
            "Photon",
            meta={"emails": photon_emails, "urls": photon_urls},
        )

    # Combine ALL candidates: enriched + already_complete + skipped
    all_candidates = enriched + already_complete + skipped

    emit(
        "deep_enrich",
        "phase_done",
        f"{len(all_candidates)} candidates forwarded",
        meta={"enriched": len(enriched), "total": len(all_candidates)},
    )

    return {
        "candidates": all_candidates,
        "status": SessionStatus.COMPILING,
    }


async def compile_profile(state: AgentState) -> dict:
    """Node 7: Compile the final profile dossier using the LLM."""
    candidates = state.get("candidates", [])
    known_facts = state.get("known_facts", {})

    # Build narrowing summary from history
    history = state.get("narrowing_history", [])
    if history:
        first_entry = history[0]
        last_entry = history[-1]
        narrowing_summary = (
            f"{first_entry['before']} found \u2192 narrowed to {last_entry['after']} "
            f"across {len(history)} round(s)"
        )
    else:
        narrowing_summary = ""

    logger.info(f"COMPILE: Building profile from {len(candidates)} candidates")
    emit("compile", "start", "Compiling dossier")
    emit("compile", "task_start", "LLM Compilation")

    system_tpl = _template_env.get_template("system.jinja2")
    system_prompt = system_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=state.get("initial_context", ""),
        known_facts=known_facts,
    )

    # Track total before slicing so Profile numbers reflect reality
    total_candidates = len(candidates)

    # Sort by confidence so best candidates make the cut
    sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)

    compile_tpl = _template_env.get_template("compilation.jinja2")
    compile_prompt = compile_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        candidates=sorted_candidates[:30],  # cap for context window
        known_facts=known_facts,
        narrowing_summary=narrowing_summary,
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
            "task_done",
            "LLM Compilation",
            meta={"count": len(sorted_candidates[:30])},
        )

        # Collect emails from all candidates
        all_emails = []
        for c in candidates:
            all_emails.extend(getattr(c, "emails", []))

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
            candidates_found=history[0]["before"] if history else total_candidates,
            candidates_remaining=total_candidates,
            narrowing_summary=narrowing_summary,
            candidate_profiles=result.candidate_profiles,
            emails=list(set(all_emails)),
            data_sources_used=state.get("data_sources_used", ["ddg"]),
        )

        emit(
            "compile",
            "phase_done",
            "Dossier complete",
            meta={
                "confidence": f"{result.confidence_score:.0%}",
                "profiles": len(social_profiles),
                "sources": len(sources),
            },
        )
        return {
            "final_profile": profile,
            "status": SessionStatus.DONE,
        }
    except Exception as e:
        logger.error(f"Profile compilation failed: {e}")
        emit("compile", "task_fail", "LLM Compilation", meta={"error": str(e)[:100]})
        return {
            "status": SessionStatus.FAILED,
            "error": f"Failed to compile profile: {e}",
        }
