"""LLM wrapper with dual provider support (Gemini / Ollama).

get_llm() returns a LangChain chat model based on settings.llm_provider.
validated_llm_call() is provider-agnostic — works with any LangChain chat model.
Includes global rate limiter for Gemini free tier and 429-aware retry with backoff.
"""

import asyncio
import json
import logging
import re
import time
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from profiler.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Global rate limiter for Gemini free tier
# Free tier: 15 RPM / 20 RPD for preview models, 1500 RPM for stable models.
# We enforce a minimum interval between calls to stay under RPM limits.
# ---------------------------------------------------------------------------
_gemini_rate_lock = asyncio.Lock()
_gemini_last_call: float = 0.0
_GEMINI_MIN_INTERVAL = 4.0  # seconds between calls (safe for 15 RPM)


async def _gemini_rate_limit():
    """Enforce minimum interval between Gemini API calls."""
    global _gemini_last_call
    async with _gemini_rate_lock:
        now = time.monotonic()
        elapsed = now - _gemini_last_call
        if elapsed < _GEMINI_MIN_INTERVAL:
            wait = _GEMINI_MIN_INTERVAL - elapsed
            logger.debug(f"Rate limiter: waiting {wait:.1f}s before next Gemini call")
            await asyncio.sleep(wait)
        _gemini_last_call = time.monotonic()


def _extract_text_content(content) -> str:
    """Extract plain text from LLM response content.

    Handles multiple formats:
    - str: returned as-is
    - list of content blocks: [{"type": "text", "text": "..."}, ...] — extracts text
    - other: json.dumps fallback
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Gemini 3 returns list of content blocks
        for block in content:
            if isinstance(block, dict):
                # {"type": "text", "text": "..."} format
                if block.get("type") == "text" and "text" in block:
                    return block["text"]
                # {"text": "..."} format (no type field)
                if "text" in block and len(block) <= 2:
                    return block["text"]
            elif isinstance(block, str):
                return block
        # Fall through — dump the whole thing
        return json.dumps(content)

    return json.dumps(content)


def _parse_retry_delay(error_msg: str) -> float | None:
    """Extract retryDelay from a Gemini 429 error message.

    Returns seconds to wait, or None if not found.
    """
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+\.?\d*)s?['\"]", str(error_msg))
    if match:
        return float(match.group(1))
    # Also try "Please retry in Xs" pattern
    match = re.search(r"retry in (\d+\.?\d*)s", str(error_msg))
    if match:
        return float(match.group(1))
    return None


def get_llm(json_mode: bool = True, num_ctx: int | None = None):
    """Create a LangChain chat model instance.

    Uses Gemini by default, falls back to Ollama if configured.

    Args:
        json_mode: If True, forces the model to output valid JSON.
        num_ctx: Context window size (Ollama only, ignored for Gemini).
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": settings.gemini_model,
            "google_api_key": settings.gemini_api_key,
            "temperature": settings.gemini_temperature,
            "convert_system_message_to_human": False,
        }
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return ChatGoogleGenerativeAI(**kwargs)

    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": settings.ollama_model,
            "base_url": settings.ollama_base_url,
            "temperature": settings.ollama_temperature,
            "num_ctx": num_ctx or settings.ollama_default_num_ctx,
            "reasoning": False,
            "client_kwargs": {"timeout": float(settings.ollama_timeout_seconds)},
        }
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


async def validated_llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    num_ctx: int | None = None,
    max_retries: int | None = None,
) -> T:
    """Call the LLM and validate the response against a Pydantic model.

    Provider-agnostic — works with any LangChain chat model.
    Retries up to max_retries times if JSON parsing or validation fails.
    For 429 rate limit errors, waits the server-suggested retryDelay before retrying.

    Args:
        system_prompt: System message content.
        user_prompt: User message content.
        response_model: Pydantic model class to validate against.
        num_ctx: Context window override (Ollama only).
        max_retries: Number of retries on failure.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValueError: If all retries exhausted.
    """
    if settings.llm_provider == "gemini":
        retries = (
            max_retries if max_retries is not None else settings.gemini_max_retries
        )
    else:
        retries = (
            max_retries if max_retries is not None else settings.ollama_max_retries
        )

    llm = get_llm(json_mode=True, num_ctx=num_ctx)

    current_user_prompt = user_prompt
    last_error = None
    raw_text = ""

    for attempt in range(retries + 1):
        try:
            # Rate limit for Gemini free tier
            if settings.llm_provider == "gemini":
                await _gemini_rate_limit()

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_user_prompt),
            ]
            response = await llm.ainvoke(messages)
            raw_text = _extract_text_content(response.content)

            # Strip markdown fences if present (defensive)
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
            logger.warning(
                f"LLM JSON parse failed (attempt {attempt + 1}): {last_error}"
            )
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"
            logger.warning(
                f"LLM validation failed (attempt {attempt + 1}): {last_error}"
            )
        except Exception as e:
            error_str = str(e)
            last_error = f"LLM call error: {e}"
            logger.warning(f"LLM call failed (attempt {attempt + 1}): {last_error}")

            # For 429 rate limit errors, wait the suggested delay before retrying
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                delay = _parse_retry_delay(error_str) or 30.0
                delay = min(delay + 5, 65)  # add buffer, cap at 65s
                if attempt < retries:
                    logger.info(
                        f"Rate limited — waiting {delay:.0f}s before retry "
                        f"(attempt {attempt + 2}/{retries + 1})"
                    )
                    await asyncio.sleep(delay)
                continue  # skip appending error context for rate limits

        # Append error context for retry (JSON/validation errors)
        current_user_prompt = (
            f"{user_prompt}\n\n"
            f"YOUR PREVIOUS RESPONSE WAS INVALID. Error: {last_error}\n"
            f"Please try again. Return ONLY valid JSON matching the schema."
        )

    raise ValueError(
        f"LLM call failed after {retries + 1} attempts. Last error: {last_error}"
    )
