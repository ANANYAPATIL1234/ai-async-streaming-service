import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class DBSession:
    """Stand-in for an SQLAlchemy/AsyncPG session."""

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self.queries: list[str] = []

    async def log_request(self, note: str) -> None:
        await asyncio.sleep(0)
        self.queries.append(note)


class DBPool:
    """Tiny connection pool: at most `size` sessions are used at the same time."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._semaphore = asyncio.Semaphore(size)
        self._in_use = 0
        self._counter = 0

    @property
    def in_use(self) -> int:
        return self._in_use

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DBSession]:
        async with self._semaphore:
            self._in_use += 1
            self._counter += 1
            try:
                yield DBSession(self._counter)
            finally:
                self._in_use -= 1  # automatic teardown / cleanup
