"""Ollama LLM provider using the ``ollama`` Python client.

Wraps the :pypi:`ollama` client library to satisfy the
:class:`~pipeline.interfaces.LLMProvider` ABC.  The default model is
``llama3`` and the default server URL is ``http://localhost:11434``.

A 10-second connection timeout is enforced; if the Ollama server is
unreachable a :class:`ConnectionError` is raised with a descriptive message.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import ollama as _ollama

from pipeline.interfaces import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3"
_DEFAULT_BASE_URL = "http://localhost:11434"
_CONNECT_TIMEOUT_SECONDS = 10.0


class OllamaLLMProvider(LLMProvider):
    """LLMProvider backed by a local Ollama server.

    Parameters
    ----------
    model:
        Ollama model name.  Defaults to ``"llama3"``.
    base_url:
        Ollama server URL.  Defaults to ``"http://localhost:11434"``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client = _ollama.Client(
            host=base_url,
            timeout=httpx.Timeout(
                _CONNECT_TIMEOUT_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS
            ),
        )

    # -- LLMProvider interface -----------------------------------------------

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous LLM call. Returns response text."""
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Connection to Ollama server at {self._base_url} timed out after "
                f"{_CONNECT_TIMEOUT_SECONDS}s: {exc}"
            ) from exc
        return response.message.content or ""

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronous LLM call. Returns response text."""
        async_client = _ollama.AsyncClient(
            host=self._base_url,
            timeout=httpx.Timeout(
                _CONNECT_TIMEOUT_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS
            ),
        )
        try:
            response = await async_client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Connection to Ollama server at {self._base_url} timed out after "
                f"{_CONNECT_TIMEOUT_SECONDS}s: {exc}"
            ) from exc
        return response.message.content or ""

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        """Extract entities and relationships from *text* using the given *prompt*.

        Sends a combined prompt requesting JSON output and parses the response.
        Returns a dict with ``"entities"`` and ``"relationships"`` keys.
        """
        combined_prompt = (
            f"{prompt}\n\n"
            f"Text: {text}\n\n"
            "Respond with valid JSON containing 'entities' and 'relationships' keys."
        )
        raw = await self.ainvoke(combined_prompt)
        return self._parse_json(raw)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output."""
        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find a JSON block inside markdown fences
        for start_marker in ("```json", "```"):
            if start_marker in text:
                start = text.index(start_marker) + len(start_marker)
                end = text.index("```", start)
                try:
                    result = json.loads(text[start:end].strip())
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass

        # Fallback: return empty structure
        logger.warning("Could not parse JSON from LLM response, returning empty result")
        return {"entities": [], "relationships": []}
