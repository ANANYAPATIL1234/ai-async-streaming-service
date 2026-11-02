import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_keys: frozenset[str]
    rate_limit_per_minute: int
    db_pool_size: int
    allowed_origins: list[str]
    token_delay_seconds: float


def load_settings() -> Settings:
    """Read configuration from environment variables (see .env.example)."""
    keys = os.getenv("API_KEYS", "dev-key-123")
    origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return Settings(
        api_keys=frozenset(k.strip() for k in keys.split(",") if k.strip()),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        db_pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
        token_delay_seconds=float(os.getenv("TOKEN_DELAY_SECONDS", "0.02")),
    )
