from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
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
