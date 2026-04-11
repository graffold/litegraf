"""Unit tests for pipeline.backends.ollama_llm.OllamaLLMProvider.

Validates Requirements 8.1, 8.2, 8.3, 8.4:
- Implements LLMProvider interface using a local Ollama server
- Accepts configurable model name and server URL
- Defaults to http://localhost:11434
- Raises ConnectionError within 10 seconds when server is unreachable
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pipeline.backends.ollama_llm import OllamaLLMProvider
from pipeline.interfaces import LLMProvider

# -- ABC conformance --------------------------------------------------------


class TestABCConformance:
    """Requirement 8.1: OllamaLLMProvider SHALL implement the LLMProvider interface."""

    def test_is_subclass_of_llm_provider(self) -> None:
        assert issubclass(OllamaLLMProvider, LLMProvider)

    def test_instance_is_llm_provider(self) -> None:
        provider = OllamaLLMProvider()
        assert isinstance(provider, LLMProvider)


# -- Constructor defaults ---------------------------------------------------


class TestConstructorDefaults:
    """Requirements 8.2, 8.3: Configurable model/URL with sensible defaults."""

    def test_default_model(self) -> None:
        provider = OllamaLLMProvider()
        assert provider._model == "llama3"

    def test_default_base_url(self) -> None:
        provider = OllamaLLMProvider()
        assert provider._base_url == "http://localhost:11434"

    def test_custom_model(self) -> None:
        provider = OllamaLLMProvider(model="mistral")
        assert provider._model == "mistral"

    def test_custom_base_url(self) -> None:
        provider = OllamaLLMProvider(base_url="http://myhost:9999")
        assert provider._base_url == "http://myhost:9999"


# -- JSON parsing -----------------------------------------------------------


class TestParseJSON:
    """Test the _parse_json helper for various LLM output formats."""

    def test_direct_json(self) -> None:
        text = '{"entities": ["A"], "relationships": ["B"]}'
        result = OllamaLLMProvider._parse_json(text)
        assert result == {"entities": ["A"], "relationships": ["B"]}

    def test_markdown_fenced_json(self) -> None:
        text = 'Here is the result:\n```json\n{"entities": [1], "relationships": [2]}\n```\nDone.'
        result = OllamaLLMProvider._parse_json(text)
        assert result == {"entities": [1], "relationships": [2]}

    def test_markdown_fenced_no_lang(self) -> None:
        text = '```\n{"entities": [], "relationships": []}\n```'
        result = OllamaLLMProvider._parse_json(text)
        assert result == {"entities": [], "relationships": []}

    def test_unparseable_returns_empty(self) -> None:
        result = OllamaLLMProvider._parse_json("This is not JSON at all")
        assert result == {"entities": [], "relationships": []}

    def test_json_with_extra_text(self) -> None:
        text = 'Sure! Here you go:\n```json\n{"entities": ["x"], "relationships": []}\n```\nLet me know if you need more.'
        result = OllamaLLMProvider._parse_json(text)
        assert result == {"entities": ["x"], "relationships": []}


# -- invoke with mock -------------------------------------------------------


class TestInvoke:
    """Test synchronous invoke method."""

    def test_invoke_returns_content(self) -> None:
        provider = OllamaLLMProvider()
        mock_response = MagicMock()
        mock_response.message.content = "Hello world"

        with patch.object(
            provider._client, "chat", return_value=mock_response
        ) as mock_chat:
            result = provider.invoke("Say hello")

        assert result == "Hello world"
        mock_chat.assert_called_once_with(
            model="llama3",
            messages=[{"role": "user", "content": "Say hello"}],
        )

    def test_invoke_returns_empty_on_none_content(self) -> None:
        provider = OllamaLLMProvider()
        mock_response = MagicMock()
        mock_response.message.content = None

        with patch.object(provider._client, "chat", return_value=mock_response):
            result = provider.invoke("test")

        assert result == ""

    def test_invoke_raises_connection_error_on_connect_error(self) -> None:
        provider = OllamaLLMProvider()

        with patch.object(
            provider._client, "chat", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                provider.invoke("test")

    def test_invoke_raises_connection_error_on_timeout(self) -> None:
        provider = OllamaLLMProvider()

        with patch.object(
            provider._client, "chat", side_effect=httpx.TimeoutException("timed out")
        ):
            with pytest.raises(ConnectionError, match="timed out"):
                provider.invoke("test")


# -- ainvoke with mock ------------------------------------------------------


class TestAinvoke:
    """Test asynchronous ainvoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_content(self) -> None:
        provider = OllamaLLMProvider()
        mock_response = MagicMock()
        mock_response.message.content = "Async hello"

        with patch(
            "pipeline.backends.ollama_llm._ollama.AsyncClient"
        ) as MockAsyncClient:
            instance = MockAsyncClient.return_value
            instance.chat = AsyncMock(return_value=mock_response)
            result = await provider.ainvoke("Say hello async")

        assert result == "Async hello"

    @pytest.mark.asyncio
    async def test_ainvoke_raises_connection_error_on_connect_error(self) -> None:
        provider = OllamaLLMProvider()

        with patch(
            "pipeline.backends.ollama_llm._ollama.AsyncClient"
        ) as MockAsyncClient:
            instance = MockAsyncClient.return_value
            instance.chat = AsyncMock(side_effect=httpx.ConnectError("refused"))
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await provider.ainvoke("test")

    @pytest.mark.asyncio
    async def test_ainvoke_raises_connection_error_on_timeout(self) -> None:
        provider = OllamaLLMProvider()

        with patch(
            "pipeline.backends.ollama_llm._ollama.AsyncClient"
        ) as MockAsyncClient:
            instance = MockAsyncClient.return_value
            instance.chat = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            with pytest.raises(ConnectionError, match="timed out"):
                await provider.ainvoke("test")


# -- extract with mock -----------------------------------------------------


class TestExtract:
    """Test extract method with JSON parsing."""

    @pytest.mark.asyncio
    async def test_extract_parses_json_response(self) -> None:
        provider = OllamaLLMProvider()
        mock_response = MagicMock()
        mock_response.message.content = json.dumps(
            {"entities": [{"name": "TP53"}], "relationships": [{"type": "REGULATES"}]}
        )

        with patch(
            "pipeline.backends.ollama_llm._ollama.AsyncClient"
        ) as MockAsyncClient:
            instance = MockAsyncClient.return_value
            instance.chat = AsyncMock(return_value=mock_response)
            result = await provider.extract("Extract entities", "TP53 regulates BRCA1")

        assert result["entities"] == [{"name": "TP53"}]
        assert result["relationships"] == [{"type": "REGULATES"}]

    @pytest.mark.asyncio
    async def test_extract_returns_empty_on_bad_json(self) -> None:
        provider = OllamaLLMProvider()
        mock_response = MagicMock()
        mock_response.message.content = "I cannot parse this"

        with patch(
            "pipeline.backends.ollama_llm._ollama.AsyncClient"
        ) as MockAsyncClient:
            instance = MockAsyncClient.return_value
            instance.chat = AsyncMock(return_value=mock_response)
            result = await provider.extract("Extract entities", "some text")

        assert result == {"entities": [], "relationships": []}
