"""Async concurrency limiter for LLM calls.

Caps concurrent ``ainvoke`` / ``extract`` calls using an ``asyncio.Semaphore``
to prevent rate-limit explosions without users needing to think about
semaphores.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pipeline.interfaces import LLMProvider


class RateLimitedLLMProvider(LLMProvider):
    """``LLMProvider`` wrapper that limits concurrent async calls.

    Sync ``invoke()`` passes through unchanged (it's already blocking).
    """

    def __init__(self, inner: LLMProvider, max_concurrent: int = 16) -> None:
        self._inner = inner
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        return self._inner.invoke(prompt, **kwargs)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        async with self._semaphore:
            return await self._inner.ainvoke(prompt, **kwargs)

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        async with self._semaphore:
            return await self._inner.extract(prompt, text)
