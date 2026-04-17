"""Data models for the PDF-first extraction pipeline.

This module defines the core data structures used throughout the extraction pipeline,
including source types, extraction results, quality metrics, and configuration options.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SourceType(Enum):
    """Enumeration of supported source types for content extraction."""

    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class ExtractionSource:
    """Source content for extraction.

    Attributes:
        source_type: Type of source (PDF or HTML)
        content: File path for PDF, HTML string for HTML
        metadata: Additional metadata (DOI, URL, authors, title, publication date, etc.)
    """

    source_type: SourceType
    content: str
    metadata: dict[str, Any]


@dataclass
class ExtractionResult:
    """Result of an extraction attempt.

    Attributes:
        success: Whether the extraction succeeded
        text: Extracted text content (empty string on failure)
        method: Name of the extraction method used
        execution_time: Time taken for extraction in seconds
        error: Error message if extraction failed, None otherwise
        metadata: Preserved metadata from the extraction source
    """

    success: bool
    text: str
    method: str
    execution_time: float
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class QualityMetrics:
    """Quality metrics for extracted text.

    Attributes:
        valid: Whether the text meets quality thresholds
        length: Length of the extracted text in characters
        nonalpha_ratio: Ratio of non-alphanumeric characters to total characters
    """

    valid: bool
    length: int
    nonalpha_ratio: float


@dataclass
class ExtractionConfig:
    """Configuration for content extraction pipeline.

    This configuration controls which extraction strategies are enabled,
    timeout values, section removal options, quality thresholds, and PDF caching.

    The default extraction chain order is:
        1. MarkItDown – markdown-native PDF extraction (PDF)
        2. MarkItDown – markdown-native PDF extraction via Microsoft markitdown (PDF)
        3. langextract – robust HTML content extraction (HTML)
        4. regex – always-available HTML fallback (HTML)

    Strategies are attempted in this priority order. Disabled strategies
    are skipped, and strategies whose dependencies are unavailable are
    filtered out during initialization.

    Attributes:
        enable_markitdown: Enable MarkItDown for markdown-native PDF extraction
        enable_markitdown: Enable Microsoft MarkItDown for markdown-native PDF extraction
        enable_langextract: Enable langextract for robust HTML parsing
        pdf_timeout: Timeout in seconds for PDF extraction strategies
        remove_references: Remove References section from extracted text
        remove_acknowledgements: Remove Acknowledgements section from extracted text
        quality_min_length: Minimum text length in characters for quality validation
        quality_max_nonalpha_ratio: Maximum ratio of non-alphanumeric characters
        pdf_cache_dir: Directory for caching downloaded PDFs (None to disable caching)
        pdf_download_timeout: Timeout in seconds for PDF downloads
        max_concurrent_extractions: Maximum number of concurrent extraction tasks
            when using batch extraction. Default is 5.
    """

    # Strategy enablement
    enable_markitdown: bool = True
    enable_pymupdf: bool = True
    enable_langextract: bool = True

    # Timeouts (seconds)
    pdf_timeout: int = 30

    # Section removal
    remove_references: bool = True
    remove_acknowledgements: bool = True

    # Quality thresholds
    quality_min_length: int = 100
    quality_max_nonalpha_ratio: float = 0.5

    # PDF download and caching
    pdf_cache_dir: str | None = None
    pdf_download_timeout: int = 30

    # Concurrency
    max_concurrent_extractions: int = 5  # Maximum concurrent extraction tasks
