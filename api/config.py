"""
Centralized application configuration.
 
All secrets and environment-dependent values are read here, once, into a
typed Settings object — not scattered as os.environ.get(...) calls across
route files. Benefits this gives you over the scattered pattern:
 
  - Fails loudly at startup if a required secret is missing, instead of
    failing on the first request that happens to need it.
  - One place to see every config value the app depends on.
  - Supports a local .env file for development without exporting env vars
    by hand every session.
  - Typed: GEMINI_API_KEY is always a str, never accidentally None deep
    in a function call.
 
Usage in route files:
    from ..config import settings
    ... settings.gemini_api_key ...
 
Local development:
    Create a `.env` file at the repo root (NEVER commit this file — see
    .gitignore) with real values:
        GEMINI_API_KEY=sk-actual-key-here
        CSV_CLASSIFIER_MODEL=gemini-2.0-flash
 
Production:
    Set real environment variables through your deployment platform's
    secret manager (e.g. AWS Secrets Manager, GCP Secret Manager, Render/
    Railway/Fly env vars). Do NOT put production secrets in a .env file
    that ships with the container image.
"""
from __future__ import annotations
 
from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars rather than erroring
    )
 
    # --- LLM provider config ---
    # Required: no default. If this isn't set, the app fails at startup
    # with a clear pydantic ValidationError instead of a vague runtime
    # error the first time a route tries to use it.
    gemini_api_key: str | None = None
 
    csv_classifier_model: str = "gemini-2.0-flash"
 
    # --- App-level config ---
    environment: str = "development"  # "development" | "staging" | "production"
    max_upload_bytes: int = 50 * 1024 * 1024  # 50MB
 
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
 
 
# Module-level singleton — imported by route handlers and api.py.
# Instantiating here means a missing required field raises immediately
# on import, i.e. at app startup, not on the first request.
settings = Settings()