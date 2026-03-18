from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = "gemini"  # "gemini" or "ollama"

    # Gemini
    gemini_api_key: str = ""  # set via GEMINI_API_KEY env var
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1
    gemini_max_retries: int = 2

    # Ollama (only used if llm_provider="ollama")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3"
    ollama_temperature: float = 0.1
    ollama_default_num_ctx: int = 16384
    ollama_timeout_seconds: int = 120
    ollama_max_retries: int = 2

    # App
    database_url: str = "sqlite+aiosqlite:///./profiler.db"
    max_concurrent_sessions: int = 5
    session_ttl_minutes: int = 60

    # Agent
    max_narrowing_rounds: int = 5
    candidate_threshold: int = 3  # switch to deep scrape when <= this many
    fuzzy_match_threshold: float = 0.7
    deep_scrape_limit: int = 10  # max candidates to deep-scrape for enrichment

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
