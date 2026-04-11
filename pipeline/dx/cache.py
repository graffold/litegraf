"""LLM response caching via hash-based KV store.

Same prompt → cached response.  Saves tokens during development iteration.
Provides ``LLMCache`` (the raw KV store) and ``CachedLLMProvider`` (a
transparent ``LLMProvider`` wrapper).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from pipeline.interfaces import LLMProvider

logger = logging.getLogger(__name__)


def _cache_key(prompt: str, **kwargs: Any) -> str:
    """Compute an MD5 cache key from prompt text and kwargs."""
    raw = prompt + str(sorted(kwargs.items()))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class LLMCache:
    """Hash-based KV cache for LLM responses stored as JSON files on disk."""

    def __init__(self, cache_dir: str = ".litegraf_cache") -> None:
        self._cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.json")

    def get(self, prompt: str, **kwargs: Any) -> str | None:
        """Return cached response for *prompt*, or ``None`` on miss."""
        key = _cache_key(prompt, **kwargs)
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("Corrupted cache entry %s — treating as miss", path)
            return None

    def put(self, prompt: str, response: str, **kwargs: Any) -> None:
        """Store *response* keyed by prompt hash."""
        key = _cache_key(prompt, **kwargs)
        path = self._path(key)
        with open(path, "w") as f:
            json.dump({"prompt": prompt, "response": response}, f)

    def clear(self) -> None:
        """Wipe all cached responses."""
        if os.path.isdir(self._cache_dir):
            shutil.rmtree(self._cache_dir)
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

    def wrap(self, llm: LLMProvider) -> CachedLLMProvider:
        """Return a ``CachedLLMProvider`` that checks cache before calling *llm*."""
        return CachedLLMProvider(inner=llm, cache=self)


class CachedLLMProvider(LLMProvider):
    """Transparent ``LLMProvider`` wrapper that caches responses on disk."""

    def __init__(self, inner: LLMProvider, cache: LLMCache) -> None:
        self._inner = inner
        self._cache = cache

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        cached = self._cache.get(prompt, **kwargs)
        if cached is not None:
            return cached
        response = self._inner.invoke(prompt, **kwargs)
        self._cache.put(prompt, response, **kwargs)
        return response

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        cached = self._cache.get(prompt, **kwargs)
        if cached is not None:
            return cached
        response = await self._inner.ainvoke(prompt, **kwargs)
        self._cache.put(prompt, response, **kwargs)
        return response

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        # Cache keyed on prompt+text combined
        combined = f"{prompt}\n---\n{text}"
        cached = self._cache.get(combined)
        if cached is not None:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                pass  # treat as miss
        result = await self._inner.extract(prompt, text)
        self._cache.put(combined, json.dumps(result))
        return result
