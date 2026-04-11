"""Abstract interfaces for pipeline backends.

Defines the four core abstractions that pipeline modules depend on:
- GraphStore: graph database operations
- EmbeddingProvider: text embedding
- LLMProvider: LLM invocation and entity extraction
- JobStore: job state persistence
"""

from abc import ABC, abstractmethod
from typing import Any


class GraphStore(ABC):
    """Abstract graph database backend."""

    @abstractmethod
    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return result records."""

    @abstractmethod
    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        """Create or update a node. Returns the node identifier."""

    @abstractmethod
    def upsert_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a relationship between two nodes."""

    @abstractmethod
    def close(self) -> None:
        """Release database connections."""

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()


class EmbeddingProvider(ABC):
    """Abstract embedding model backend."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings. Returns a list of float vectors."""


class LLMProvider(ABC):
    """Abstract LLM backend for entity/relationship extraction."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous LLM call. Returns response text."""

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronous LLM call. Returns response text."""

    @abstractmethod
    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        """Extract entities and relationships from text using the given prompt.

        Returns a dict with 'entities' and 'relationships' keys.
        """


class JobStore(ABC):
    """Abstract job state persistence backend."""

    @abstractmethod
    async def save(self, job_id: str, metadata: dict[str, Any]) -> None:
        """Persist job metadata. Upserts if job_id already exists."""

    @abstractmethod
    async def load(self, job_id: str) -> dict[str, Any] | None:
        """Load job metadata by ID. Returns None if not found."""

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Remove persisted job state."""

    @abstractmethod
    async def list_jobs(self) -> list[dict[str, Any]]:
        """Return all persisted job metadata dictionaries."""
