"""Source adapter protocol for pluggable data source connectors.

Defines the SourceAdapter runtime-checkable Protocol that all data source
connectors must satisfy. Uses generic dict[str, Any] types to remain
domain-agnostic — no domain-specific references.
"""

from typing import Any, Protocol, runtime_checkable

from pipeline.interfaces import GraphStore


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol for pluggable data source connectors."""

    @property
    def source_type(self) -> str:
        """Unique identifier for this source type (e.g. 'pubmed')."""
        ...

    @property
    def supports_batch(self) -> bool:
        """Whether this adapter supports batch fetching."""
        ...

    async def fetch(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch raw records from the data source.

        Args:
            params: Source-specific query parameters.

        Returns:
            List of raw record dicts from the source.
        """
        ...

    async def extract(self, record: dict[str, Any]) -> str:
        """Extract text content from a single raw record.

        Args:
            record: A single raw record from fetch().

        Returns:
            Extracted text content as a string.
        """
        ...

    async def deduplicate(
        self, records: list[dict[str, Any]], graph_store: GraphStore
    ) -> list[dict[str, Any]]:
        """Remove duplicate records using the graph store for lookups.

        Args:
            records: List of raw records to deduplicate.
            graph_store: Graph database backend for existing-record checks.

        Returns:
            Filtered list with duplicates removed.
        """
        ...
