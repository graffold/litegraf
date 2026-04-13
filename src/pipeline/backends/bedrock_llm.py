"""Bedrock LLM provider for the LiteGraf pipeline.

Wraps AWS Bedrock's Converse API to satisfy the
:class:`~pipeline.interfaces.LLMProvider` ABC.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pipeline.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class BedrockLLMProvider(LLMProvider):
    """LLMProvider backed by AWS Bedrock Converse API.

    Parameters
    ----------
    model:
        Bedrock model ID (e.g. ``"eu.amazon.nova-lite-v1:0"``).
    region_name:
        AWS region. Defaults to ``"eu-north-1"``.
    """

    def __init__(self, model: str = "eu.amazon.nova-lite-v1:0", region_name: str = "eu-north-1", **_: Any) -> None:
        import boto3
        from botocore.config import Config

        self._model = model
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            config=Config(read_timeout=120, connect_timeout=10, retries={"max_attempts": 1}),
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        resp = self._client.converse(
            modelId=self._model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
        )
        content = resp["output"]["message"]["content"]
        if isinstance(content, list) and content:
            return content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        return str(content)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        combined = f"{prompt}\n\nText: {text}\n\nRespond with valid JSON containing 'entities' and 'relationships' keys."
        raw = self.invoke(combined)
        return self._parse_json(raw)

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
        logger.warning("Could not parse JSON from Bedrock response")
        return {"entities": [], "relationships": []}
