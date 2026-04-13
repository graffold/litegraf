"""
Base classes and interfaces for enrichment pipeline components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ColumnStrategy(Enum):
    """Enumeration of different enrichment strategies for columns."""

    SKIP = "skip"
    PROPERTY = "property"
    KG_EXTRACTION = "kg_extraction"
    MEANINGFUL_NODE_CLASS = "meaningful_node_class"  # For columns with meaningful subclasses (e.g., subcellular localization)


class ManualColumnHandler(Enum):
    """Enumeration of manual column handling strategies."""

    SKIP = "skip"
    PROTEIN_ID = "protein_id"  # Match to existing protein nodes
    PROTEIN_PROPERTY = "protein_property"  # Add as property to protein nodes
    KG_EXTRACTION = "kg_extraction"  # Extract knowledge graph from text
    NODE_CLASS = "node_class"  # Create meaningful node classes
    RELATIONSHIP = "relationship"  # Create relationships (future use)


class ColumnType(Enum):
    """Enumeration of column data types."""

    TEXT_RICH = "text_rich"
    CATEGORICAL_SMALL = "categorical_small"
    CATEGORICAL_LARGE = "categorical_large"
    MULTI_VALUE_CATEGORICAL_SMALL = "multi_value_categorical_small"
    MULTI_VALUE_CATEGORICAL_LARGE = "multi_value_categorical_large"
    NUMERIC_CONTINUOUS = "continuous_numeric"
    NUMERIC_DISCRETE = "discrete_numeric"
    BINARY = "binary"
    UNIPROT_ID = "uniprot_id"
    ENSEMBL_ID = "ensembl_id"
    ONE_TO_ONE_MAPPING_NAME = "1:1_mapping_name"
    ONE_TO_ONE_MAPPING_ID = "1:1_mapping_id"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ColumnAnalysis:
    """Analysis results for a single column."""

    column_name: str
    column_type: ColumnType
    strategy: ColumnStrategy
    unique_count: int
    total_values: int
    empty_percentage: float
    na_percentage: float
    total_missing_percentage: float
    has_multiple_values: bool = False
    expanded_unique_count: int | None = None
    avg_text_length: float | None = None
    sample_values: list[str] | None = None
    partner_column: str | None = None
    reason: str | None = None


@dataclass
class EnrichmentStats:
    """Statistics tracking for enrichment operations."""

    proteins_processed: int = 0
    properties_added: int = 0
    node_classes_created: int = 0
    individual_nodes_created: int = 0
    relationships_created: int = 0
    kg_extractions_performed: int = 0
    errors: list[str] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BaseEnrichmentProcessor(ABC):
    """Base class for all enrichment processors."""

    def __init__(self, query_executor, **kwargs):
        self.query_executor = query_executor
        self.stats = EnrichmentStats()

    @abstractmethod
    async def process(
        self, data: list[dict[str, Any]], analysis: dict[str, ColumnAnalysis]
    ) -> dict[str, Any]:
        """Process data according to the processor's specific logic."""

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using the configured query executor."""
        return self.query_executor.execute_query(query, parameters)

    def get_stats(self) -> EnrichmentStats:
        """Get current processing statistics."""
        return self.stats
