import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.db import DBPool, DBSession

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> str:
    settings: Settings = request.app.state.settings
    if api_key is None or api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


class RateLimiter:
    """In-memory sliding-window limiter, one window per API key."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = int(hits[0] + self.window - now) + 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


async def rate_limit(
    request: Request,
    api_key: Annotated[str, Depends(verify_api_key)],
) -> str:
    limiter: RateLimiter = request.app.state.rate_limiter
    limiter.check(api_key)
    return api_key


async def get_db_session(request: Request) -> AsyncIterator[DBSession]:
    pool: DBPool = request.app.state.db_pool
    async with pool.acquire() as session:
        yield session


ApiKey = Annotated[str, Depends(rate_limit)]
DBDep = Annotated[DBSession, Depends(get_db_session)]
