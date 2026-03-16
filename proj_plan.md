# Profiler — Implementation Specification

> **Purpose**: This document is the single source of truth for implementing the Profiler application. It is designed to be passed directly to an AI coding assistant (Cursor, Copilot, Claude Code, etc.) to scaffold and build the project step by step.
>
> **Reference PRD**: `Profiler_PRD.docx` — contains full product context, evaluation criteria, and ethical guidelines. This spec focuses only on what to build and how.

---

## Project Overview

Profiler is an agentic OSINT (Open Source Intelligence) app. Given a person or company name, it:

1. Searches the public internet (Google, social media)
2. Collects candidate profiles
3. Uses an LLM to pick the best disambiguating question
4. Asks the user that question
5. Filters candidates based on the answer
6. Repeats steps 3–5 until ≤3 candidates remain
7. Deep-scrapes the finalists
8. Compiles a structured dossier

**Stack**: Python 3.11+ · FastAPI · LangGraph · Ollama (Qwen 3.5 9B) · Playwright · DuckDuckGo Search · SQLite

---

## Project Structure

Create this exact directory layout. Every file described below must be implemented.

```
profiler/
├── main.py                     # FastAPI app entry point
├── config.py                   # Pydantic BaseSettings — all env vars
├── models/
│   ├── __init__.py
│   ├── enums.py                # TargetType, Platform, Status, SearchStatus
│   ├── candidate.py            # CandidateProfile model
│   ├── profile.py              # Final Profile / Dossier model
│   └── session.py              # SearchSession model
├── agent/
│   ├── __init__.py
│   ├── state.py                # AgentState TypedDict for LangGraph
│   ├── graph.py                # LangGraph graph definition + compilation
│   ├── nodes.py                # All node functions
│   ├── llm.py                  # validated_llm_call() helper with retry
│   └── prompts/
│       ├── system.jinja2
│       ├── narrowing.jinja2
│       ├── compilation.jinja2
│       └── search_query.jinja2
├── tools/
│   ├── __init__.py
│   ├── search.py               # google_search() — DuckDuckGo wrapper
│   ├── scraper.py              # scrape_page() — Playwright + BS4
│   ├── extractor.py            # extract_profile() — platform-specific
│   └── matcher.py              # fuzzy_match() — rapidfuzz wrapper
├── api/
│   ├── __init__.py
│   ├── routes.py               # All FastAPI route handlers
│   ├── sse.py                  # SSE event helpers
│   └── dependencies.py         # Session lookup, rate limiter
├── db/
│   ├── __init__.py
│   ├── database.py             # SQLite async setup + table creation
│   └── repository.py           # CRUD: sessions, profiles
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_agent.py
│   ├── test_api.py
│   └── fixtures/
│       ├── mock_search_results.json
│       └── mock_html_facebook.html
├── pyproject.toml
├── cli.py                      # CLI entry point — interactive terminal client
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Step 0: Project Setup

### pyproject.toml

```toml
[project]
name = "profiler"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "langgraph>=0.2",
    "langchain-community>=0.3",
    "langchain-core>=0.3",
    "ollama>=0.4",
    "playwright>=1.48",
    "beautifulsoup4>=4.12",
    "httpx>=0.27",
    "duckduckgo-search>=7.0",
    "rapidfuzz>=3.9",
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "jinja2>=3.1",
    "sse-starlette>=2.0",
    "aiosqlite>=0.20",
    "python-dotenv>=1.0",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.30",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project.scripts]
profiler = "cli:main"
```

### .env.example

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5
OLLAMA_TEMPERATURE=0.1
OLLAMA_DEFAULT_NUM_CTX=16384
OLLAMA_TIMEOUT_SECONDS=60
OLLAMA_MAX_RETRIES=2

# App
DATABASE_URL=sqlite+aiosqlite:///./profiler.db
MAX_CONCURRENT_SESSIONS=5
SESSION_TTL_MINUTES=60
```

---

## Step 1: Config & Models

### `profiler/config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5"
    ollama_temperature: float = 0.1
    ollama_default_num_ctx: int = 16384
    ollama_timeout_seconds: int = 60
    ollama_max_retries: int = 2

    # App
    database_url: str = "sqlite+aiosqlite:///./profiler.db"
    max_concurrent_sessions: int = 5
    session_ttl_minutes: int = 60

    # Agent
    max_narrowing_rounds: int = 5
    candidate_threshold: int = 3  # switch to deep scrape when <= this many
    fuzzy_match_threshold: float = 0.7

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

### `profiler/models/enums.py`

```python
from enum import Enum

class TargetType(str, Enum):
    PERSON = "person"
    COMPANY = "company"

class Platform(str, Enum):
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    GENERIC = "generic"

class SessionStatus(str, Enum):
    SEARCHING = "searching"
    NARROWING = "narrowing"
    ASKING_USER = "asking_user"
    COMPILING = "compiling"
    DONE = "done"
    FAILED = "failed"
```

### `profiler/models/candidate.py`

```python
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from .enums import Platform

class CandidateProfile(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    platform: Platform = Platform.GENERIC
    profile_url: Optional[str] = None
    location: Optional[str] = None
    school: Optional[str] = None
    employer: Optional[str] = None
    bio: Optional[str] = None
    profile_photo_url: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
```

### `profiler/models/profile.py`

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from .enums import TargetType

class SocialProfile(BaseModel):
    platform: str
    url: str
    username: Optional[str] = None
    bio: Optional[str] = None
    followers: Optional[int] = None

class Source(BaseModel):
    url: str
    title: Optional[str] = None
    accessed_at: datetime = Field(default_factory=datetime.utcnow)

class NewsMention(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published_date: Optional[str] = None

class Profile(BaseModel):
    target_name: str
    target_type: TargetType
    summary: str
    social_profiles: list[SocialProfile] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    employment: list[str] = Field(default_factory=list)
    associated_entities: list[str] = Field(default_factory=list)
    news_mentions: list[NewsMention] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence_score: float = 0.0
    compiled_at: datetime = Field(default_factory=datetime.utcnow)
```

### `profiler/models/session.py`

```python
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from .enums import TargetType, SessionStatus

class SearchRequest(BaseModel):
    name: str
    target_type: TargetType = TargetType.PERSON
    context: Optional[str] = None  # any extra info the user provides upfront

class SearchSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    target_name: str
    target_type: TargetType
    context: Optional[str] = None
    status: SessionStatus = SessionStatus.SEARCHING
    narrowing_round: int = 0
    candidates_count: int = 0
    known_facts: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AnswerRequest(BaseModel):
    answer: str
```

---

## Step 2: Ollama LLM Helper

### `profiler/agent/llm.py`

This is the most critical infrastructure file. Every agent node calls the LLM through this helper.

```python
import json
import logging
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from profiler.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def get_llm(num_ctx: int | None = None, json_mode: bool = True) -> ChatOllama:
    """Create a ChatOllama instance with appropriate settings.
    
    Args:
        num_ctx: Context window size. Use smaller values for simple tasks.
                 Defaults to settings.ollama_default_num_ctx.
        json_mode: If True, forces Ollama to output valid JSON.
    """
    kwargs = {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.ollama_temperature,
        "num_ctx": num_ctx or settings.ollama_default_num_ctx,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


async def validated_llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    num_ctx: int | None = None,
    max_retries: int | None = None,
) -> T:
    """Call the LLM and validate the response against a Pydantic model.
    
    Retries up to max_retries times if JSON parsing or validation fails,
    appending the error to the prompt each retry.
    
    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        response_model: Pydantic model class to validate against.
        num_ctx: Context window override.
        max_retries: Number of retries on failure.
    
    Returns:
        Validated Pydantic model instance.
    
    Raises:
        ValueError: If all retries exhausted.
    """
    retries = max_retries if max_retries is not None else settings.ollama_max_retries
    llm = get_llm(num_ctx=num_ctx, json_mode=True)
    
    current_user_prompt = user_prompt
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_user_prompt),
            ]
            response = await llm.ainvoke(messages)
            raw_text = response.content
            
            # Strip markdown fences if present (shouldn't happen with format=json, but defensive)
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            parsed = json.loads(cleaned)
            result = response_model.model_validate(parsed)
            return result
            
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}. Raw response: {raw_text[:200]}"
            logger.warning(f"LLM JSON parse failed (attempt {attempt + 1}): {last_error}")
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"
            logger.warning(f"LLM validation failed (attempt {attempt + 1}): {last_error}")
        
        # Append error context for retry
        current_user_prompt = (
            f"{user_prompt}\n\n"
            f"YOUR PREVIOUS RESPONSE WAS INVALID. Error: {last_error}\n"
            f"Please try again. Return ONLY valid JSON matching the schema."
        )
    
    raise ValueError(
        f"LLM call failed after {retries + 1} attempts. Last error: {last_error}"
    )
```

**Context window guide for callers:**

| Node | num_ctx | Why |
|------|---------|-----|
| search query generation | `4096` | Tiny input, short output |
| candidate analysis / narrowing | `16384` | Large candidate list + stats |
| profile extraction from HTML | `8192` | Truncated page content |
| profile compilation | `16384` | Enriched candidate data |

---

## Step 3: Tools

### `profiler/tools/search.py`

```python
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
    Function is named google_search for interface compatibility —
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
        # DDGS is sync — run in thread to not block the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().text(query, max_results=min(num_results, 30)))
        )
        
        for item in raw_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            })
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
        # Return empty rather than crashing — the agent handles empty results
        return []
    
    _cache[cache_key] = results
    return results
```

### `profiler/tools/scraper.py`

```python
import logging
from typing import Optional
from bs4 import BeautifulSoup
import httpx

logger = logging.getLogger(__name__)

# Domains that require JS rendering (Playwright)
JS_REQUIRED_DOMAINS = {"facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com"}


async def scrape_page(
    url: str,
    max_chars: int = 12000,
    wait_for_selector: Optional[str] = None,
) -> dict:
    """Scrape a web page and return structured content.
    
    Uses Playwright for JS-heavy sites, httpx+BS4 for static pages.
    
    Args:
        url: Full URL to scrape.
        max_chars: Maximum characters of text to return.
        wait_for_selector: CSS selector to wait for (Playwright only).
    
    Returns:
        Dict with keys: url, title, text, meta_description, success, error.
    """
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace("www.", "")
    
    needs_js = any(js_domain in domain for js_domain in JS_REQUIRED_DOMAINS)
    
    if needs_js:
        return await _scrape_with_playwright(url, max_chars, wait_for_selector)
    else:
        return await _scrape_with_httpx(url, max_chars)


async def _scrape_with_httpx(url: str, max_chars: int) -> dict:
    """Static page scraping with httpx + BeautifulSoup."""
    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Profiler/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]
        
        text = soup.get_text(separator="\n", strip=True)[:max_chars]
        
        return {
            "url": url,
            "title": title,
            "text": text,
            "meta_description": meta_desc,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return {"url": url, "title": "", "text": "", "meta_description": "", "success": False, "error": str(e)}


async def _scrape_with_playwright(url: str, max_chars: int, wait_for: Optional[str] = None) -> dict:
    """JS-rendered page scraping with Playwright."""
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=5000)
                except Exception:
                    pass  # proceed anyway
            
            title = await page.title()
            content = await page.content()
            await browser.close()
        
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n", strip=True)[:max_chars]
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]
        
        return {
            "url": url,
            "title": title,
            "text": text,
            "meta_description": meta_desc,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Playwright scrape failed for {url}: {e}")
        return {"url": url, "title": "", "text": "", "meta_description": "", "success": False, "error": str(e)}
```

### `profiler/tools/extractor.py`

```python
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
    
    Uses LLM-based extraction — sends the page text to the model
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
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        import json
        data = json.loads(response.content)
        
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
```

### `profiler/tools/matcher.py`

```python
from rapidfuzz import fuzz


def fuzzy_match(
    candidate_value: str | None,
    user_value: str,
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """Check if a candidate field value matches the user's answer.
    
    Uses token_sort_ratio for order-independent matching.
    Handles common abbreviations and variations.
    
    Args:
        candidate_value: Value from the candidate profile (may be None).
        user_value: Value provided by the user.
        threshold: Similarity threshold 0.0–1.0.
    
    Returns:
        Tuple of (is_match: bool, similarity_score: float).
    """
    if candidate_value is None:
        return False, 0.0
    
    # Normalize
    cv = candidate_value.strip().lower()
    uv = user_value.strip().lower()
    
    if not cv or not uv:
        return False, 0.0
    
    # Exact match shortcut
    if cv == uv:
        return True, 1.0
    
    # Containment check (e.g., "University of Oregon" contains "Oregon")
    if uv in cv or cv in uv:
        return True, 0.9
    
    # Fuzzy match with token sort (handles word order differences)
    score = fuzz.token_sort_ratio(cv, uv) / 100.0
    
    return score >= threshold, score
```

---

## Step 4: Agent State & Prompts

### `profiler/agent/state.py`

```python
from typing import TypedDict, Optional, Annotated
from profiler.models.candidate import CandidateProfile
from profiler.models.profile import Profile
from profiler.models.enums import TargetType, SessionStatus
import operator


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
    
    # Inter-node communication
    current_question: Optional[dict]  # {field, question, options, reasoning}
    user_answer: Optional[str]  # answer from user (set after interrupt)
    
    # Output
    final_profile: Optional[Profile]
    status: SessionStatus
    error: Optional[str]
```

### `profiler/agent/prompts/system.jinja2`

```
You are Profiler, an OSINT research agent. Your job is to find a specific person or company using only publicly available internet data.

Rules:
- You are methodical, precise, and privacy-conscious.
- You only use data that is freely accessible on the public internet.
- You NEVER fabricate or infer information. If uncertain, say so.
- You cite sources for every claim.
- All your responses must be valid JSON matching the requested schema.

Target: {{ target_name }}
Type: {{ target_type }}
{% if initial_context %}Additional context: {{ initial_context }}{% endif %}

Known facts so far:
{% for key, value in known_facts.items() %}
- {{ key }}: {{ value }}
{% endfor %}
```

### `profiler/agent/prompts/narrowing.jinja2`

```
You are analyzing search candidates to determine the BEST question to ask the user to narrow down results.

## Candidates ({{ candidates | length }} total):
{% for c in candidates %}
- Name: {{ c.name }} | Platform: {{ c.platform }} | Location: {{ c.location or "unknown" }} | School: {{ c.school or "unknown" }} | Employer: {{ c.employer or "unknown" }}
{% endfor %}

## Field Statistics:
{% for field, stats in field_stats.items() %}
- {{ field }}: {{ stats.unique_count }} unique values. Top values: {{ stats.top_values | join(", ") }}
{% endfor %}

## Already Known:
{% for key, value in known_facts.items() %}
- {{ key }}: {{ value }}
{% endfor %}

## Your Task:
Pick the ONE field that will eliminate the most candidates when the user answers.

Rules:
1. Pick fields with HIGH variance (many unique values = good discriminator).
2. Prefer fields the user can easily answer: location > employer > school > bio details.
3. NEVER ask about a field that is already in "Already Known".
4. If options are available for the field, include the top 3-4 most common values as suggested options.
5. The question should be natural and conversational.

Respond with ONLY this JSON:
{
    "field": "the field name (location, school, employer, etc.)",
    "question": "natural language question to ask the user",
    "options": ["option1", "option2", "option3"] or null,
    "reasoning": "why this field is the best discriminator",
    "expected_elimination_pct": 0.75
}
```

### `profiler/agent/prompts/compilation.jinja2`

```
You are compiling a final profile dossier from verified search data.

## Target: {{ target_name }} ({{ target_type }})

## Verified Data:
{% for c in candidates %}
### Source: {{ c.platform }} — {{ c.profile_url }}
- Name: {{ c.name }}
- Location: {{ c.location or "N/A" }}
- School: {{ c.school or "N/A" }}
- Employer: {{ c.employer or "N/A" }}
- Bio: {{ c.bio or "N/A" }}
- Additional data: {{ c.raw_data | tojson }}
{% endfor %}

## Known Facts (confirmed by user):
{% for key, value in known_facts.items() %}
- {{ key }}: {{ value }}
{% endfor %}

## Your Task:
Synthesize all data into a comprehensive profile. 

Rules:
1. Only include facts that appear in the source data above. DO NOT infer or fabricate.
2. If two sources conflict, note the discrepancy.
3. Flag any low-confidence fields.
4. Write a 2-3 paragraph narrative summary.

Respond with ONLY this JSON:
{
    "summary": "narrative summary paragraph(s)",
    "locations": ["list of known locations"],
    "education": ["list of schools/universities"],
    "employment": ["list of employers/roles"],
    "associated_entities": ["related people, companies, orgs"],
    "confidence_score": 0.0 to 1.0
}
```

### `profiler/agent/prompts/search_query.jinja2`

```
Generate search queries to find a {{ target_type }} named "{{ target_name }}" on the internet.

{% if initial_context %}Additional context: {{ initial_context }}{% endif %}

{% if known_facts %}
Known facts:
{% for key, value in known_facts.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

{% if search_history %}
Already searched (DO NOT repeat these):
{% for q in search_history %}
- {{ q }}
{% endfor %}
{% endif %}

Generate 3-5 search queries that will find this person's profiles. Include:
1. A general Google search with their name + any known facts
2. A Facebook-targeted search
3. A Twitter/X-targeted search  
4. A LinkedIn-targeted search (if person)
5. A news search if relevant

Respond with ONLY this JSON:
{
    "queries": [
        {"query": "search string", "site_filter": "facebook.com or null", "purpose": "why this query"}
    ]
}
```

---

## Step 5: Agent Graph (LangGraph)

### `profiler/agent/nodes.py`

Implement each node as an async function that takes `AgentState` and returns a partial state update dict.

```python
import asyncio
import json
import logging
from collections import Counter
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

logger = logging.getLogger(__name__)

# Load Jinja2 templates
_template_env = Environment(
    loader=FileSystemLoader("profiler/agent/prompts"),
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
    """Node 1: Generate search queries and execute them."""
    logger.info(f"BROAD_SEARCH: Searching for '{state['target_name']}'")
    
    system_tpl = _template_env.get_template("system.jinja2")
    system_prompt = system_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=state.get("initial_context", ""),
        known_facts=state.get("known_facts", {}),
    )
    
    query_tpl = _template_env.get_template("search_query.jinja2")
    query_prompt = query_tpl.render(
        target_name=state["target_name"],
        target_type=state["target_type"],
        initial_context=state.get("initial_context", ""),
        known_facts=state.get("known_facts", {}),
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
        name = state["target_name"]
        search_plan = SearchQueries(queries=[
            {"query": name, "site_filter": None, "purpose": "general"},
            {"query": name, "site_filter": "facebook.com", "purpose": "facebook"},
            {"query": name, "site_filter": "twitter.com", "purpose": "twitter"},
            {"query": name, "site_filter": "linkedin.com", "purpose": "linkedin"},
        ])
    
    # Execute all searches concurrently
    all_results = []
    new_search_history = list(state.get("search_history", []))
    
    tasks = []
    for q in search_plan.queries:
        query_str = q["query"] if isinstance(q, dict) else q.query
        site = q.get("site_filter") if isinstance(q, dict) else getattr(q, "site_filter", None)
        if query_str not in new_search_history:
            tasks.append(google_search(query_str, num_results=10, site_filter=site))
            new_search_history.append(query_str)
    
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results_lists:
        if isinstance(result, list):
            all_results.extend(result)
    
    if not all_results:
        return {
            "status": SessionStatus.FAILED,
            "error": "No search results found for this name.",
        }
    
    return {
        "search_history": new_search_history,
        "status": SessionStatus.SEARCHING,
        # Pass raw results forward — next node will scrape and extract
        "_raw_search_results": all_results,
    }


async def extract_and_normalize(state: AgentState) -> dict:
    """Node 2: Scrape search result URLs and extract candidate profiles."""
    raw_results = state.get("_raw_search_results", [])
    logger.info(f"EXTRACT: Processing {len(raw_results)} search results")
    
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
    scrape_tasks = [scrape_with_limit(r["url"]) for r in unique_results[:20]]
    scraped_pages = await asyncio.gather(*scrape_tasks, return_exceptions=True)
    
    # Extract profiles from scraped pages
    candidates = list(state.get("candidates", []))
    for page_data in scraped_pages:
        if isinstance(page_data, dict) and page_data.get("success"):
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
    
    return {
        "candidates": deduped,
        "status": SessionStatus.NARROWING,
    }


async def analyze_candidates(state: AgentState) -> dict:
    """Node 3: Analyze candidates and decide the best narrowing question."""
    candidates = state.get("candidates", [])
    known_facts = state.get("known_facts", {})
    
    logger.info(f"ANALYZE: {len(candidates)} candidates, round {state.get('narrowing_round', 0)}")
    
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
                "coverage": len(values) / len(candidates),  # % of candidates with this field
            }
    
    if not field_stats:
        # No more fields to narrow on — go to deep scrape
        return {"status": SessionStatus.COMPILING}
    
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
    answer = state.get("user_answer", "")
    field = state["current_question"]["field"]
    candidates = state.get("candidates", [])
    eliminated = list(state.get("eliminated", []))
    known_facts = dict(state.get("known_facts", {}))
    
    logger.info(f"FILTER: Applying answer '{answer}' to field '{field}'")
    
    # Add to known facts
    known_facts[field] = answer
    
    # Filter candidates using fuzzy matching
    kept = []
    for c in candidates:
        candidate_value = getattr(c, field, None)
        if candidate_value is None:
            # Unknown field — keep the candidate (benefit of the doubt)
            kept.append(c)
        else:
            is_match, score = fuzzy_match(candidate_value, answer, settings.fuzzy_match_threshold)
            if is_match:
                c.confidence = max(c.confidence, score)
                kept.append(c)
            else:
                eliminated.append(c)
    
    round_num = state.get("narrowing_round", 0) + 1
    
    logger.info(f"FILTER: {len(kept)} kept, {len(eliminated)} total eliminated")
    
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
    logger.info(f"DEEP_SCRAPE: Enriching {len(candidates)} candidates")
    
    enriched = []
    for candidate in candidates[:3]:  # max 3 deep scrapes
        if candidate.profile_url:
            scraped = await scrape_page(candidate.profile_url, max_chars=12000)
            if scraped.get("success"):
                # Re-extract with more data
                enriched_profile = await extract_profile(scraped, state["target_name"])
                if enriched_profile:
                    # Merge: keep original data, add new fields
                    for field in ["location", "school", "employer", "bio"]:
                        new_val = getattr(enriched_profile, field)
                        if new_val and not getattr(candidate, field):
                            setattr(candidate, field, new_val)
                    candidate.raw_data.update(enriched_profile.raw_data)
        enriched.append(candidate)
    
    return {
        "candidates": enriched,
        "status": SessionStatus.COMPILING,
    }


async def compile_profile(state: AgentState) -> dict:
    """Node 7: Compile the final profile dossier using the LLM."""
    candidates = state.get("candidates", [])
    known_facts = state.get("known_facts", {})
    
    logger.info(f"COMPILE: Building profile from {len(candidates)} candidates")
    
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
                social_profiles.append(SocialProfile(
                    platform=c.platform.value,
                    url=c.profile_url,
                    bio=c.bio,
                ))
        
        sources = []
        for c in candidates:
            for url in c.source_urls:
                sources.append(Source(url=url))
        
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
        
        return {
            "final_profile": profile,
            "status": SessionStatus.DONE,
        }
    except Exception as e:
        logger.error(f"Profile compilation failed: {e}")
        return {
            "status": SessionStatus.FAILED,
            "error": f"Failed to compile profile: {e}",
        }
```

### `profiler/agent/graph.py`

This is the LangGraph state machine definition. The key complexity is the `ASK_USER` interrupt.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from profiler.agent.state import AgentState
from profiler.agent.nodes import (
    broad_search,
    extract_and_normalize,
    analyze_candidates,
    filter_candidates,
    deep_scrape,
    compile_profile,
)
from profiler.config import settings
from profiler.models.enums import SessionStatus


def should_continue_narrowing(state: AgentState) -> str:
    """Router: decide whether to keep narrowing or move to deep scrape."""
    candidates = state.get("candidates", [])
    narrowing_round = state.get("narrowing_round", 0)
    status = state.get("status")
    
    # If analyze_candidates decided to compile directly (no more fields)
    if status == SessionStatus.COMPILING:
        return "deep_scrape"
    
    # Threshold reached
    if len(candidates) <= settings.candidate_threshold:
        return "deep_scrape"
    
    # Max rounds reached
    if narrowing_round >= settings.max_narrowing_rounds:
        return "deep_scrape"
    
    return "ask_user"


def after_filter(state: AgentState) -> str:
    """Router: after filtering, decide next step."""
    candidates = state.get("candidates", [])
    narrowing_round = state.get("narrowing_round", 0)
    
    if len(candidates) <= settings.candidate_threshold:
        return "deep_scrape"
    if narrowing_round >= settings.max_narrowing_rounds:
        return "deep_scrape"
    
    return "analyze_candidates"


def after_broad_search(state: AgentState) -> str:
    """Router: check if broad search failed."""
    if state.get("status") == SessionStatus.FAILED:
        return END
    return "extract_and_normalize"


def build_graph(checkpointer=None):
    """Build and compile the LangGraph agent graph.
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Required for the ASK_USER interrupt to work.
    
    Returns:
        Compiled graph.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("broad_search", broad_search)
    graph.add_node("extract_and_normalize", extract_and_normalize)
    graph.add_node("analyze_candidates", analyze_candidates)
    graph.add_node("ask_user", lambda state: state)  # no-op; interrupt happens here
    graph.add_node("filter_candidates", filter_candidates)
    graph.add_node("deep_scrape", deep_scrape)
    graph.add_node("compile_profile", compile_profile)
    
    # Set entry point
    graph.set_entry_point("broad_search")
    
    # Edges
    graph.add_conditional_edges("broad_search", after_broad_search)
    graph.add_edge("extract_and_normalize", "analyze_candidates")
    graph.add_conditional_edges("analyze_candidates", should_continue_narrowing)
    graph.add_edge("ask_user", "filter_candidates")
    graph.add_conditional_edges("filter_candidates", after_filter)
    graph.add_edge("deep_scrape", "compile_profile")
    graph.add_edge("compile_profile", END)
    
    # Compile with interrupt_before on ask_user
    # This pauses the graph before executing ask_user, 
    # allowing us to send the question to the user via SSE
    # and resume when they respond.
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["ask_user"],
    )
    
    return compiled


async def get_checkpointer():
    """Create an async SQLite checkpointer for LangGraph state persistence."""
    checkpointer = AsyncSqliteSaver.from_conn_string("profiler_checkpoints.db")
    return checkpointer
```

---

## Step 6: Database

### `profiler/db/database.py`

```python
import aiosqlite
import json
from pathlib import Path

DB_PATH = "profiler.db"


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                target_name TEXT NOT NULL,
                target_type TEXT NOT NULL,
                context TEXT,
                status TEXT NOT NULL DEFAULT 'searching',
                narrowing_round INTEGER DEFAULT 0,
                candidates_count INTEGER DEFAULT 0,
                known_facts TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                session_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        await db.commit()
```

### `profiler/db/repository.py`

```python
import aiosqlite
import json
from datetime import datetime
from profiler.db.database import DB_PATH
from profiler.models.session import SearchSession
from profiler.models.profile import Profile


async def create_session(session: SearchSession) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO sessions (id, target_name, target_type, context, status,
               narrowing_round, candidates_count, known_facts, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(session.id), session.target_name, session.target_type.value,
                session.context, session.status.value, session.narrowing_round,
                session.candidates_count, json.dumps(session.known_facts),
                session.created_at.isoformat(), session.updated_at.isoformat(),
            ),
        )
        await db.commit()


async def update_session(session_id: str, **kwargs) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        sets = []
        vals = []
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(json.dumps(v) if isinstance(v, dict) else v)
        sets.append("updated_at = ?")
        vals.append(datetime.utcnow().isoformat())
        vals.append(session_id)
        await db.execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", vals
        )
        await db.commit()


async def get_session(session_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
    return None


async def save_profile(session_id: str, profile: Profile) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO profiles (session_id, profile_json, created_at) VALUES (?, ?, ?)",
            (session_id, profile.model_dump_json(), datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_profile(session_id: str) -> Profile | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT profile_json FROM profiles WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row:
            return Profile.model_validate_json(row[0])
    return None


async def delete_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM profiles WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
```

---

## Step 7: API Layer

### `profiler/api/sse.py`

```python
import json
from typing import Any


def sse_event(event_type: str, data: Any) -> dict:
    """Format an SSE event for sse-starlette."""
    return {
        "event": event_type,
        "data": json.dumps(data) if not isinstance(data, str) else data,
    }


def status_update(status: str, message: str, candidates_count: int = 0) -> dict:
    return sse_event("status_update", {
        "status": status,
        "message": message,
        "candidates_count": candidates_count,
    })


def question_event(question: str, field: str, options: list | None, round_num: int) -> dict:
    return sse_event("question", {
        "question": question,
        "field": field,
        "options": options,
        "round": round_num,
    })


def profile_ready_event(session_id: str) -> dict:
    return sse_event("profile_ready", {"profile_id": session_id})


def error_event(message: str, recoverable: bool = False) -> dict:
    return sse_event("error", {"message": message, "recoverable": recoverable})
```

### `profiler/api/dependencies.py`

```python
import asyncio
from collections import defaultdict
from fastapi import HTTPException, Request
from profiler.config import settings

# Simple in-memory rate limiter
_active_sessions: dict[str, int] = defaultdict(int)


async def check_rate_limit(request: Request):
    """Limit concurrent sessions per IP."""
    client_ip = request.client.host if request.client else "unknown"
    if _active_sessions[client_ip] >= settings.max_concurrent_sessions:
        raise HTTPException(429, "Too many concurrent sessions. Please wait.")


def track_session(client_ip: str):
    _active_sessions[client_ip] += 1


def release_session(client_ip: str):
    _active_sessions[client_ip] = max(0, _active_sessions[client_ip] - 1)
```

### `profiler/api/routes.py`

This is the most complex file. It orchestrates the LangGraph agent via SSE streaming.

```python
import asyncio
import logging
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends, Request
from sse_starlette.sse import EventSourceResponse

from profiler.config import settings
from profiler.models.session import SearchRequest, AnswerRequest, SearchSession
from profiler.models.enums import SessionStatus, TargetType
from profiler.agent.graph import build_graph, get_checkpointer
from profiler.api.sse import status_update, question_event, profile_ready_event, error_event
from profiler.api.dependencies import check_rate_limit, track_session, release_session
from profiler.db.repository import (
    create_session, update_session, get_session, 
    save_profile, get_profile, delete_session,
)
import profiler.db.database as database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# In-memory store for pending user answers (session_id -> asyncio.Event + answer)
_pending_answers: dict[str, dict] = {}


@router.on_event("startup")
async def startup():
    await database.init_db()


@router.post("/search")
async def start_search(req: SearchRequest, request: Request):
    """Start a new profiling session."""
    await check_rate_limit(request)
    
    session = SearchSession(
        target_name=req.name,
        target_type=req.target_type,
        context=req.context,
    )
    await create_session(session)
    
    client_ip = request.client.host if request.client else "unknown"
    track_session(client_ip)
    
    return {"session_id": str(session.id), "status": "created"}


@router.get("/search/{session_id}/stream")
async def stream_search(session_id: str, request: Request):
    """SSE stream that runs the agent and yields events."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    async def event_generator():
        try:
            checkpointer = await get_checkpointer()
            graph = build_graph(checkpointer=checkpointer)
            
            thread_config = {"configurable": {"thread_id": session_id}}
            
            # Initial state
            initial_state = {
                "target_name": session["target_name"],
                "target_type": session["target_type"],
                "initial_context": session["context"] or "",
                "session_id": session_id,
                "known_facts": {},
                "candidates": [],
                "eliminated": [],
                "search_history": [],
                "narrowing_round": 0,
                "current_question": None,
                "user_answer": None,
                "final_profile": None,
                "status": SessionStatus.SEARCHING,
                "error": None,
            }
            
            yield status_update("searching", "Starting broad search...", 0)
            
            # Run graph — it will pause at interrupt_before=["ask_user"]
            state = None
            async for event in graph.astream(initial_state, config=thread_config):
                # event is a dict of {node_name: state_update}
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        state = {**initial_state, **(state or {}), **node_output}
                
                current_status = state.get("status") if state else None
                
                if current_status == SessionStatus.FAILED:
                    yield error_event(state.get("error", "Unknown error"))
                    return
                
                if current_status == SessionStatus.SEARCHING:
                    yield status_update("searching", "Scraping and extracting profiles...",
                                        len(state.get("candidates", [])))
                
                if current_status == SessionStatus.NARROWING:
                    yield status_update("narrowing", "Analyzing candidates...",
                                        len(state.get("candidates", [])))
            
            # Graph paused at ask_user interrupt — enter narrowing loop
            while True:
                # Get current state from checkpointer
                snapshot = await graph.aget_state(thread_config)
                current_state = snapshot.values
                
                if current_state.get("status") == SessionStatus.DONE:
                    break
                
                if current_state.get("status") == SessionStatus.FAILED:
                    yield error_event(current_state.get("error", "Unknown error"))
                    return
                
                question = current_state.get("current_question")
                if not question:
                    break
                
                # Send question to user
                yield question_event(
                    question=question["question"],
                    field=question["field"],
                    options=question.get("options"),
                    round_num=current_state.get("narrowing_round", 0),
                )
                
                await update_session(session_id,
                    status="asking_user",
                    narrowing_round=current_state.get("narrowing_round", 0),
                    candidates_count=len(current_state.get("candidates", [])),
                )
                
                # Wait for user answer
                answer_event = asyncio.Event()
                _pending_answers[session_id] = {"event": answer_event, "answer": None}
                
                try:
                    await asyncio.wait_for(answer_event.wait(), timeout=300)  # 5 min timeout
                except asyncio.TimeoutError:
                    yield error_event("Timed out waiting for answer", recoverable=True)
                    del _pending_answers[session_id]
                    return
                
                user_answer = _pending_answers[session_id]["answer"]
                del _pending_answers[session_id]
                
                yield status_update("narrowing",
                    f"Filtering candidates by {question['field']}...",
                    len(current_state.get("candidates", [])))
                
                # Resume graph with the user's answer
                await graph.aupdate_state(
                    thread_config,
                    {"user_answer": user_answer, "status": SessionStatus.NARROWING},
                )
                
                # Continue running from the interrupt
                async for event in graph.astream(None, config=thread_config):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict):
                            current_state = {**current_state, **node_output}
                    
                    cs = current_state.get("status")
                    if cs == SessionStatus.COMPILING:
                        yield status_update("compiling", "Compiling final profile...",
                                            len(current_state.get("candidates", [])))
            
            # Done — save profile
            final_state = (await graph.aget_state(thread_config)).values
            profile = final_state.get("final_profile")
            
            if profile:
                await save_profile(session_id, profile)
                await update_session(session_id, status="done")
                yield profile_ready_event(session_id)
            else:
                yield error_event("Failed to compile profile")
                await update_session(session_id, status="failed")
        
        except Exception as e:
            logger.exception(f"Stream error for session {session_id}")
            yield error_event(str(e))
            await update_session(session_id, status="failed")
        finally:
            client_ip = request.client.host if request.client else "unknown"
            release_session(client_ip)
    
    return EventSourceResponse(event_generator())


@router.post("/search/{session_id}/answer")
async def submit_answer(session_id: str, req: AnswerRequest):
    """Submit the user's answer to a narrowing question."""
    if session_id not in _pending_answers:
        raise HTTPException(400, "No pending question for this session")
    
    _pending_answers[session_id]["answer"] = req.answer
    _pending_answers[session_id]["event"].set()
    
    return {"status": "received"}


@router.get("/search/{session_id}/status")
async def get_status(session_id: str):
    """Get current session status."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.get("/search/{session_id}/profile")
async def get_session_profile(session_id: str):
    """Get the final compiled profile."""
    profile = await get_profile(session_id)
    if not profile:
        raise HTTPException(404, "Profile not ready")
    return profile.model_dump()


@router.delete("/search/{session_id}")
async def cancel_session(session_id: str):
    """Cancel and clean up a session."""
    await delete_session(session_id)
    if session_id in _pending_answers:
        _pending_answers[session_id]["event"].set()
        del _pending_answers[session_id]
    return {"status": "deleted"}


@router.get("/health")
async def health_check():
    """Check Ollama connectivity and model availability."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            has_model = any(settings.ollama_model in name for name in model_names)
            
            return {
                "status": "healthy" if has_model else "degraded",
                "ollama": "connected",
                "model_loaded": has_model,
                "available_models": model_names,
                "search_engine": "duckduckgo (no API key needed)",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama": f"error: {e}",
            "model_loaded": False,
            "fix": "Run: ollama serve (in a separate terminal)",
        }
```

---

## Step 8: App Entry Point

### `profiler/main.py`

```python
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from profiler.api.routes import router
from profiler.db.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="Profiler",
    description="Agentic OSINT person & company search",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup():
    await init_db()
```

---

## Step 9: Docker

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for Playwright
RUN apt-get update && apt-get install -y \
    wget gnupg libnss3 libatk-bridge2.0-0 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
    libgbm1 libpango-1.0-0 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
RUN playwright install chromium

COPY . .

EXPOSE 8000
CMD ["uvicorn", "profiler.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

```yaml
version: "3.8"

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Pull model on first run
    entrypoint: >
      sh -c "ollama serve &
             sleep 5 &&
             ollama pull qwen3.5 &&
             wait"

  profiler:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
    volumes:
      - ./profiler.db:/app/profiler.db

volumes:
  ollama_data:
```

---

## Implementation Order

Build in this exact order. Each step is testable independently before moving on.

| Order | What | Verify With |
|-------|------|-------------|
| 1 | `config.py`, all `models/`, `.env` | `python -c "from profiler.config import settings; print(settings)"` |
| 2 | `agent/llm.py` | Write a test that calls `validated_llm_call()` with a simple schema |
| 3 | `tools/search.py` | `pytest tests/test_tools.py::test_google_search` |
| 4 | `tools/scraper.py` | `pytest tests/test_tools.py::test_scrape_page` |
| 5 | `tools/extractor.py` | Test with a mock HTML fixture |
| 6 | `tools/matcher.py` | `pytest tests/test_tools.py::test_fuzzy_match` |
| 7 | `agent/prompts/` (all 4 templates) | Render each template and manually inspect |
| 8 | `agent/nodes.py` | Test each node in isolation with mock state |
| 9 | `agent/graph.py` | Run the full graph with a test name and debug logging |
| 10 | `db/database.py` + `db/repository.py` | Create, read, update, delete a test session |
| 11 | `api/sse.py` + `api/dependencies.py` | Unit tests |
| 12 | `api/routes.py` | `curl` against each endpoint |
| 13 | `main.py` | `uvicorn profiler.main:app --reload` and hit `/docs` |
| 14 | `cli.py` | `python cli.py "Elon Musk"` end-to-end test |
| 15 | Docker files | `docker-compose up` and run a full search |

---

## Testing Checklist

After implementation, manually test these scenarios:

- [ ] `python cli.py "Elon Musk"` — unique name, should compile without narrowing
- [ ] `python cli.py "John Smith"` — common name, should trigger 2–3 narrowing questions
- [ ] `python cli.py "John Smith" --context "lives in Portland, works at Nike"` — should narrow faster
- [ ] Ctrl+C during a search — should clean up gracefully
- [ ] Kill Ollama during a search — should show error, not crash
- [ ] Check output JSON file is valid: `python -m json.tool output/profile_*.json`
- [ ] `python cli.py --help` — should show usage
- [ ] Run with `--output custom_dir/` — should save there

---

## Step 10: CLI Client

### How it works

The CLI is a **direct client** — it does NOT go through the FastAPI server. It imports the agent graph directly and runs it in-process. This means you can develop and test the agent without running `uvicorn` at all. The only external dependency is Ollama running locally.

The flow:

1. User runs `python cli.py "John Smith"`
2. CLI builds the LangGraph agent and runs it
3. When the graph hits the `ask_user` interrupt, the CLI prints the question and waits for terminal input
4. User types their answer, graph resumes
5. When done, the profile is printed to terminal and saved as JSON

### `cli.py`

```python
#!/usr/bin/env python3
"""
Profiler CLI — Interactive OSINT search from the terminal.

Usage:
    python cli.py "John Smith"
    python cli.py "Acme Corp" --type company
    python cli.py "Jane Doe" --context "lives in Austin, went to UT"
    python cli.py "John Smith" --output ./results/
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from profiler.config import settings
from profiler.models.enums import TargetType, SessionStatus
from profiler.agent.graph import build_graph, get_checkpointer
from profiler.agent.state import AgentState

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]PROFILER[/bold cyan]\n"
        "[dim]Agentic OSINT Person & Company Search[/dim]",
        border_style="cyan",
    ))


def print_profile(profile_data: dict, target_name: str):
    """Pretty-print the final profile to the terminal."""
    console.print()
    console.print(Panel(
        f"[bold green]Profile Complete: {target_name}[/bold green]",
        border_style="green",
    ))

    # Summary
    if profile_data.get("summary"):
        console.print(f"\n[bold]Summary[/bold]")
        console.print(profile_data["summary"])

    # Social profiles table
    socials = profile_data.get("social_profiles", [])
    if socials:
        table = Table(title="Social Profiles", show_lines=True)
        table.add_column("Platform", style="cyan")
        table.add_column("URL", style="blue")
        table.add_column("Bio", max_width=40)
        for sp in socials:
            table.add_row(
                sp.get("platform", ""),
                sp.get("url", ""),
                (sp.get("bio") or "")[:40],
            )
        console.print(table)

    # Key facts
    for label, key in [
        ("Locations", "locations"),
        ("Education", "education"),
        ("Employment", "employment"),
        ("Associated Entities", "associated_entities"),
    ]:
        items = profile_data.get(key, [])
        if items:
            console.print(f"\n[bold]{label}:[/bold] {', '.join(items)}")

    # Sources
    sources = profile_data.get("sources", [])
    if sources:
        console.print(f"\n[bold]Sources ({len(sources)}):[/bold]")
        for s in sources[:10]:
            console.print(f"  [dim]• {s.get('url', '')}[/dim]")
        if len(sources) > 10:
            console.print(f"  [dim]... and {len(sources) - 10} more[/dim]")

    # Confidence
    confidence = profile_data.get("confidence_score", 0)
    color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
    console.print(f"\n[bold]Confidence:[/bold] [{color}]{confidence:.0%}[/{color}]")


def save_profile_json(profile_data: dict, target_name: str, output_dir: str) -> Path:
    """Save profile as a JSON file. Returns the file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in target_name)
    safe_name = safe_name.strip().replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"profile_{safe_name}_{timestamp}.json"

    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(profile_data, f, indent=2, default=str)

    return filepath


async def check_ollama():
    """Verify Ollama is running and model is available."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if not any(settings.ollama_model in name for name in model_names):
                console.print(f"[red]Model '{settings.ollama_model}' not found in Ollama.[/red]")
                console.print(f"[yellow]Run: ollama pull {settings.ollama_model}[/yellow]")
                return False
            return True
    except Exception:
        console.print("[red]Cannot connect to Ollama.[/red]")
        console.print("[yellow]Run: ollama serve[/yellow]")
        return False


async def run_search(name: str, target_type: TargetType, context: str, output_dir: str):
    """Run the full agent pipeline interactively."""

    # Pre-flight check
    if not await check_ollama():
        sys.exit(1)

    session_id = str(uuid4())
    checkpointer = await get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    thread_config = {"configurable": {"thread_id": session_id}}

    initial_state: AgentState = {
        "target_name": name,
        "target_type": target_type,
        "initial_context": context,
        "session_id": session_id,
        "known_facts": {},
        "candidates": [],
        "eliminated": [],
        "search_history": [],
        "narrowing_round": 0,
        "current_question": None,
        "user_answer": None,
        "final_profile": None,
        "status": SessionStatus.SEARCHING,
        "error": None,
    }

    console.print(f"\n[bold]Searching for:[/bold] {name}")
    if context:
        console.print(f"[bold]Context:[/bold] {context}")
    console.print()

    # --- Run the graph until first interrupt ---
    current_state = dict(initial_state)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Broad search — querying Google and social media...", total=None)

        async for event in graph.astream(initial_state, config=thread_config):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    current_state = {**current_state, **node_output}

                if node_name == "broad_search":
                    progress.update(task, description="Scraping and extracting profiles...")
                elif node_name == "extract_and_normalize":
                    n = len(current_state.get("candidates", []))
                    progress.update(task, description=f"Found {n} candidates. Analyzing...")
                elif node_name == "analyze_candidates":
                    progress.update(task, description="Preparing narrowing question...")

    # Check for early failure
    if current_state.get("status") == SessionStatus.FAILED:
        console.print(f"\n[red]Search failed:[/red] {current_state.get('error', 'Unknown error')}")
        sys.exit(1)

    # Check if it finished without needing narrowing
    if current_state.get("status") == SessionStatus.DONE:
        profile = current_state["final_profile"]
        profile_data = profile.model_dump()
        print_profile(profile_data, name)
        filepath = save_profile_json(profile_data, name, output_dir)
        console.print(f"\n[green]Saved to:[/green] {filepath}")
        return

    # --- Narrowing loop ---
    while True:
        snapshot = await graph.aget_state(thread_config)
        state = snapshot.values

        if state.get("status") == SessionStatus.DONE:
            break
        if state.get("status") == SessionStatus.FAILED:
            console.print(f"\n[red]Error:[/red] {state.get('error')}")
            sys.exit(1)

        question = state.get("current_question")
        if not question:
            break

        # Display the narrowing question
        n_candidates = len(state.get("candidates", []))
        round_num = state.get("narrowing_round", 0) + 1

        console.print(Panel(
            f"[bold yellow]Narrowing Round {round_num}[/bold yellow] — "
            f"{n_candidates} candidates remaining",
            border_style="yellow",
        ))

        console.print(f"\n[bold]{question['question']}[/bold]")

        options = question.get("options")
        if options:
            console.print("[dim]Suggestions:[/dim]")
            for i, opt in enumerate(options, 1):
                console.print(f"  [cyan]{i}.[/cyan] {opt}")
            console.print(f"  [dim]Type a number to select, or type your own answer[/dim]")

        # Get user input
        answer = Prompt.ask("\n[bold green]Your answer[/bold green]")

        # Handle numbered selection
        if options and answer.strip().isdigit():
            idx = int(answer.strip()) - 1
            if 0 <= idx < len(options):
                answer = options[idx]
                console.print(f"[dim]Selected: {answer}[/dim]")

        # Resume the graph
        await graph.aupdate_state(
            thread_config,
            {"user_answer": answer, "status": SessionStatus.NARROWING},
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Filtering candidates...", total=None)

            async for event in graph.astream(None, config=thread_config):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        state = {**state, **node_output}

                    if node_name == "filter_candidates":
                        n = len(state.get("candidates", []))
                        progress.update(task, description=f"{n} candidates remaining...")
                    elif node_name == "analyze_candidates":
                        progress.update(task, description="Analyzing for next question...")
                    elif node_name == "deep_scrape":
                        progress.update(task, description="Deep scraping final candidates...")
                    elif node_name == "compile_profile":
                        progress.update(task, description="Compiling final profile...")

    # --- Output ---
    final_snapshot = await graph.aget_state(thread_config)
    final_state = final_snapshot.values
    profile = final_state.get("final_profile")

    if profile:
        profile_data = profile.model_dump()
        print_profile(profile_data, name)
        filepath = save_profile_json(profile_data, name, output_dir)
        console.print(f"\n[green]Saved to:[/green] {filepath}")
    else:
        console.print("\n[red]Failed to compile a profile.[/red]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Profiler — Agentic OSINT Person & Company Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "Elon Musk"
  python cli.py "Jane Doe" --context "lives in Austin, works at Dell"
  python cli.py "Acme Corp" --type company
  python cli.py "John Smith" --output ./results/
        """,
    )
    parser.add_argument("name", help="Person or company name to search")
    parser.add_argument(
        "--type", "-t",
        choices=["person", "company"],
        default="person",
        help="Target type (default: person)",
    )
    parser.add_argument(
        "--context", "-c",
        default="",
        help="Additional context to narrow initial search (e.g., 'lives in Portland')",
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Directory to save the JSON profile (default: ./output)",
    )

    args = parser.parse_args()

    print_banner()

    target_type = TargetType.PERSON if args.type == "person" else TargetType.COMPANY

    try:
        asyncio.run(run_search(args.name, target_type, args.context, args.output))
    except KeyboardInterrupt:
        console.print("\n[yellow]Search cancelled.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### How the CLI interacts with the agent

The key thing to understand: the CLI **does not use the FastAPI server at all**. It imports and runs the LangGraph graph directly. Here's the flow mapped to what happens:

```
$ python cli.py "John Smith" --context "works at Nike"

1. cli.py builds the LangGraph graph (same graph.py the API uses)
2. Calls graph.astream(initial_state) — starts BROAD_SEARCH node
   └─ Terminal shows: spinner "Broad search — querying Google..."
3. EXTRACT_AND_NORMALIZE runs
   └─ Terminal shows: spinner "Found 47 candidates. Analyzing..."
4. ANALYZE_CANDIDATES runs, picks "location" as best narrowing field
5. Graph hits interrupt_before=["ask_user"] — astream() ends
6. CLI reads state, finds current_question:
   ┌──────────────────────────────────────────┐
   │ Narrowing Round 1 — 47 candidates        │
   └──────────────────────────────────────────┘
   Does this person live in Portland, Seattle, or Austin?
   Suggestions:
     1. Portland, OR
     2. Seattle, WA
     3. Austin, TX
   Type a number or your own answer

   Your answer: 1
   Selected: Portland, OR

7. CLI calls graph.aupdate_state() with user_answer="Portland, OR"
8. CLI calls graph.astream(None) to resume — runs FILTER_CANDIDATES
   └─ Terminal shows: spinner "12 candidates remaining..."
9. ANALYZE_CANDIDATES picks "employer" next
10. Graph interrupts again — CLI shows next question
    ...repeats until ≤3 candidates...
11. DEEP_SCRAPE + COMPILE_PROFILE run
12. CLI prints the profile table and saves JSON:

   ┌──────────────────────────────────────────┐
   │ Profile Complete: John Smith              │
   └──────────────────────────────────────────┘
   Summary: John Smith is a marketing manager at Nike based in...

   Social Profiles:
   ┌──────────┬─────────────────────────┬──────────┐
   │ Platform │ URL                     │ Bio      │
   ├──────────┼─────────────────────────┼──────────┤
   │ facebook │ facebook.com/jsmith...  │ Nike...  │
   │ twitter  │ x.com/johnsmith_pdx     │ PDX...   │
   └──────────┴─────────────────────────┴──────────┘

   Saved to: ./output/profile_john_smith_20260317_142355.json
```

### JSON Output Format

The saved file at `./output/profile_john_smith_20260317_142355.json`:

```json
{
  "target_name": "John Smith",
  "target_type": "person",
  "summary": "John Smith is a marketing manager at Nike based in Portland, OR...",
  "social_profiles": [
    {
      "platform": "facebook",
      "url": "https://facebook.com/jsmith.portland",
      "username": "jsmith.portland",
      "bio": "Nike | Portland | UO alum",
      "followers": null
    },
    {
      "platform": "twitter",
      "url": "https://x.com/johnsmith_pdx",
      "username": "johnsmith_pdx",
      "bio": "Marketing @ Nike. Go Ducks.",
      "followers": 342
    }
  ],
  "locations": ["Portland, OR"],
  "education": ["University of Oregon"],
  "employment": ["Nike — Marketing Manager"],
  "associated_entities": ["Nike Inc.", "University of Oregon"],
  "news_mentions": [],
  "sources": [
    {"url": "https://facebook.com/jsmith.portland", "title": null, "accessed_at": "2026-03-17T14:23:51"},
    {"url": "https://x.com/johnsmith_pdx", "title": null, "accessed_at": "2026-03-17T14:23:52"}
  ],
  "confidence_score": 0.82,
  "compiled_at": "2026-03-17T14:23:55"
}
```

---

## Key Gotchas

1. **LangGraph `interrupt_before`** requires a checkpointer. Without it, the graph can't pause and resume. The `AsyncSqliteSaver` is the simplest option.

2. **Ollama `format: "json"`** only constrains to valid JSON, not your schema. The `validated_llm_call()` retry loop is essential.

3. **Playwright needs `chromium` installed** — run `playwright install chromium` after pip install, or it will fail silently on the first JS-heavy scrape.

4. **DuckDuckGo rate limits** — DDG doesn't publish rate limits, but aggressive querying (50+ queries in a minute) may get you temporarily blocked. The in-memory cache in `search.py` prevents duplicate queries, and the `asyncio.Semaphore(5)` in the scraper limits concurrency. If you hit blocks, add a `time.sleep(1)` between queries in the search tool.

5. **`astream` vs `ainvoke`** — use `astream` for the SSE endpoint so you can yield events as nodes complete. `ainvoke` blocks until the entire graph finishes (or hits an interrupt).

6. **Context window overflow** — if you pass too many candidates to the narrowing prompt, Ollama will silently truncate. The `[:50]` cap in `analyze_candidates` prevents this, but watch your token counts.

7. **Facebook/LinkedIn scraping** — these platforms actively block scrapers. Playwright helps but isn't bulletproof. Expect ~50% success rate on social media pages. The system is designed to work with partial data.

8. **CLI vs API use the same graph** — the `cli.py` imports `build_graph()` directly and never touches FastAPI. This means you can develop and test the entire agent without running `uvicorn`. The API layer (`routes.py`) is only needed when you build a web frontend later.
