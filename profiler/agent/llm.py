import json
import logging
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
from langchain_ollama import ChatOllama
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
        # Disable "thinking/reasoning" mode for models that support it (e.g. qwen3.5).
        # When reasoning is enabled, the model puts output in a thinking field
        # instead of the content field, which breaks JSON extraction.
        "reasoning": False,
        # Pass timeout to the underlying ollama httpx client.
        # Needed for cold model loading (6.6GB qwen3.5:9b can take 30-60s).
        "client_kwargs": {"timeout": float(settings.ollama_timeout_seconds)},
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
    raw_text = ""

    for attempt in range(retries + 1):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_user_prompt),
            ]
            response = await llm.ainvoke(messages)
            raw_text = response.content
            # Ensure raw_text is a string (langchain types it as str | list)
            if not isinstance(raw_text, str):
                raw_text = json.dumps(raw_text)

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
            # Catch connection/transport errors (httpx.RemoteProtocolError,
            # ConnectionError, TimeoutError, etc.) and treat as retryable.
            last_error = f"LLM connection error: {e}"
            logger.warning(f"LLM call failed (attempt {attempt + 1}): {last_error}")

        # Append error context for retry
        current_user_prompt = (
            f"{user_prompt}\n\n"
            f"YOUR PREVIOUS RESPONSE WAS INVALID. Error: {last_error}\n"
            f"Please try again. Return ONLY valid JSON matching the schema."
        )

    raise ValueError(
        f"LLM call failed after {retries + 1} attempts. Last error: {last_error}"
    )
