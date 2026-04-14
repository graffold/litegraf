"""Cloudflare Workers AI LLM provider for the LiteGraf pipeline.

Wraps the Cloudflare Workers AI REST API to satisfy the
:class:`~pipeline.interfaces.LLMProvider` ABC.

Required environment variables:
- ``CF_ACCOUNT_ID`` – Cloudflare account identifier.
- ``CF_API_TOKEN`` – API token with Workers AI read permission.

The default model is ``@cf/meta/llama-3.1-8b-instruct``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from pipeline.interfaces import LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"
_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class CloudflareLLMProvider(LLMProvider):
    """LLMProvider backed by Cloudflare Workers AI REST API.

    Parameters
    ----------
    model:
        Workers AI model name (e.g. ``"@cf/meta/llama-3.1-8b-instruct"``).
    account_id:
        Cloudflare account ID.  Falls back to ``CF_ACCOUNT_ID`` env var.
    api_token:
        Cloudflare API token.  Falls back to ``CF_API_TOKEN`` env var.
    max_tokens:
        Maximum tokens to generate.  Defaults to ``2048``.
    temperature:
        Sampling temperature.  Defaults to ``0.1``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        account_id: str | None = None,
        api_token: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **_: Any,
    ) -> None:
        self._model = model
        self._account_id = account_id or os.environ.get("CF_ACCOUNT_ID", "")
        self._api_token = api_token or os.environ.get("CF_API_TOKEN", "")
        self._max_tokens = max_tokens
        self._temperature = temperature

        if not self._account_id:
            raise ValueError(
                "Cloudflare account ID is required. "
                "Pass account_id= or set CF_ACCOUNT_ID env var."
            )
        if not self._api_token:
            raise ValueError(
                "Cloudflare API token is required. "
                "Pass api_token= or set CF_API_TOKEN env var."
            )

        self._url = f"{_BASE_URL}/{self._account_id}/ai/run/{self._model}"
        self._headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }

    # -- LLMProvider interface -----------------------------------------------

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        payload = self._build_payload(prompt, **kwargs)
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(self._url, headers=self._headers, json=payload)
        return self._handle_response(resp)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        payload = self._build_payload(prompt, **kwargs)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(self._url, headers=self._headers, json=payload)
        return self._handle_response(resp)

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        combined = (
            f"{prompt}\n\n"
            f"Text: {text}\n\n"
            "Respond with valid JSON containing 'entities' and 'relationships' keys."
        )
        raw = await self.ainvoke(combined)
        return self._parse_json(raw)

    # -- internals -----------------------------------------------------------

    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }

    @staticmethod
    def _handle_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        body = resp.json()
        if not body.get("success"):
            errors = body.get("errors", [])
            raise RuntimeError(f"Cloudflare Workers AI error: {errors}")
        result = body.get("result", {})
        return result.get("response", "")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    result = json.loads(text[start:end].strip())
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass
        logger.warning("Could not parse JSON from Cloudflare response")
        return {"entities": [], "relationships": []}
