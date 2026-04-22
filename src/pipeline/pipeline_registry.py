"""Pipeline registry mapping source types to adapters and schemas.

Provides the PipelineRegistry class that maps source_type strings to their
corresponding SourceAdapter and EntitySchema pairs. Coexists with
BackendRegistry (which resolves infrastructure backends like neo4j →
Neo4jGraphStore). PipelineRegistry resolves domain-level source types
like "pubmed" → PubMedAdapter + BiomedicalSchema.
"""

from pipeline.entity_schema import EntitySchema
from pipeline.source_adapter import SourceAdapter


class PipelineRegistry:
    """Maps source_type strings to SourceAdapter + EntitySchema pairs."""

    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter | type] = {}
        self._schemas: dict[str, EntitySchema] = {}

    def register(
        self,
        source_type: str,
        adapter: SourceAdapter | type,
        schema: EntitySchema,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a source type with its adapter and schema.

        Args:
            source_type: Unique identifier for the source (e.g. "pubmed").
            adapter: A SourceAdapter instance or class.
            schema: The EntitySchema for this source type.
            overwrite: If True, replace an existing registration.

        Raises:
            ValueError: If source_type is already registered and
                overwrite is False.
        """
        if source_type in self._adapters and not overwrite:
            raise ValueError(
                f"Source type '{source_type}' is already registered."
                f" Use overwrite=True to replace it."
            )
        self._adapters[source_type] = adapter
        self._schemas[source_type] = schema

    def get_adapter(self, source_type: str) -> SourceAdapter | type:
        """Return the registered adapter for a source type.

        Args:
            source_type: The source type identifier.

        Returns:
            The registered SourceAdapter instance or class.

        Raises:
            KeyError: If source_type is not registered, with a message
                listing all available source types.
        """
        if source_type not in self._adapters:
            available = list(self._adapters.keys())
            raise KeyError(
                f"Source type '{source_type}' is not registered."
                f" Available types: {available}"
            )
        return self._adapters[source_type]

    def get_schema(self, source_type: str) -> EntitySchema:
        """Return the registered schema for a source type.

        Args:
            source_type: The source type identifier.

        Returns:
            The registered EntitySchema instance.

        Raises:
            KeyError: If source_type is not registered, with a message
                listing all available source types.
        """
        if source_type not in self._schemas:
            available = list(self._schemas.keys())
            raise KeyError(
                f"Source type '{source_type}' is not registered."
                f" Available types: {available}"
            )
        return self._schemas[source_type]

    def list_sources(self) -> list[str]:
        """Return all registered source type identifiers."""
        return list(self._adapters.keys())
