"""Fake LLM + embedding backend so the service runs without any API keys.

Swap these functions for real calls (e.g. an HTTPX call to an LLM API).
"""

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator

EMBED_DIM = 8


async def stream_tokens(prompt: str, max_tokens: int, delay: float = 0.02) -> AsyncIterator[str]:
    """Yield tokens one by one, like an LLM streaming API."""
    reply = ["This", "is", "a", "simulated", "reply", "to:", *prompt.split()]
    for token in reply[:max_tokens]:
        await asyncio.sleep(delay)  # non-blocking wait, the event loop stays free
        yield token + " "


async def generate_completion(prompt: str, max_tokens: int, delay: float = 0.02) -> list[str]:
    return [token async for token in stream_tokens(prompt, max_tokens, delay)]


def _embed_one(text: str) -> list[float]:
    """CPU-bound work (hashing + math). Must NOT run directly on the event loop."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [b / 255 for b in digest[:EMBED_DIM]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [round(x / norm, 6) for x in raw]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    return [_embed_one(t) for t in texts]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Offload blocking/CPU work to a worker thread so other requests keep flowing."""
    return await asyncio.to_thread(_embed_batch, texts)
