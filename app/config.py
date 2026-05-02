"""
Application configuration — loaded once at startup from environment variables.
All settings have sensible defaults so the app runs in development without
a complete .env file (except DATABASE_URL and API_KEY which are required in
production).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"

    # ── Security ─────────────────────────────────────────────────────────────
    api_key: str = "dev-insecure-key-change-in-production"

    # ── Embedding model ───────────────────────────────────────────────────────
    # all-MiniLM-L6-v2 produces 384-dimensional vectors and runs on CPU.
    # Swap this for any sentence-transformers-compatible model.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384          # Must match the chosen model
    embedding_batch_size: int = 64           # Tune for your RAM

    # ── Server ────────────────────────────────────────────────────────────────
    port: int = 8000
    environment: str = "development"

    # ── Search defaults ───────────────────────────────────────────────────────
    default_top_k: int = 10
    similarity_threshold: float = 0.3       # Cosine similarity floor (0-1)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def async_database_url(self) -> str:
        """
        Ensure the SQLAlchemy driver prefix is asyncpg.

        Handles three real-world forms:
          postgresql://...          -> Supabase / most providers
          postgres://...            -> Railway Postgres plugin, some PaaS
          postgresql+asyncpg://...  -> already correct, returned unchanged
        """
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of constructing Settings()."""
    return Settings()
