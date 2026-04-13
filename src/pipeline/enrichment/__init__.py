"""
Enrichment Pipeline - Modular CSV data enrichment for knowledge graphs

This pipeline processes CSV data in a hierarchical manner:
1. Column Analysis - Detect sparsity, types, and text-rich columns
2. Text-Rich Processing - LLM-based KG extraction from biological text
3. Node Annotation - Add simple column data as properties to existing nodes

The pipeline is designed to be flexible and extensible, allowing for different
enrichment strategies based on column characteristics and data quality.
"""

from .base import (
    BaseEnrichmentProcessor,
    ColumnAnalysis,
    ColumnStrategy,
    ColumnType,
    EnrichmentStats,
)
from .column_analyzer import ColumnAnalyzer
from .enrichment_orchestrator import EnrichmentOrchestrator
from .node_annotator import NodeAnnotator
from .text_rich_processor import TextRichProcessor

__all__ = [
    "BaseEnrichmentProcessor",
    "ColumnAnalysis",
    "ColumnAnalyzer",
    "ColumnStrategy",
    "ColumnType",
    "EnrichmentOrchestrator",
    "EnrichmentStats",
    "NodeAnnotator",
    "TextRichProcessor",
]
