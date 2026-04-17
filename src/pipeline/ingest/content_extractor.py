"""Content extractor for converting bioRxiv paper HTML/PDF into clean text.

Uses langextract for HTML content extraction with a regex-based fallback,
and Microsoft MarkItDown for PDF extraction.
"""

import logging
import asyncio
import re
from typing import Any

import aiohttp

from pipeline.ingest.extraction_models import (
    ExtractionConfig,
    ExtractionResult,
    ExtractionSource,
    QualityMetrics,
    SourceType,
)
from pipeline.ingest.extraction_strategies import (
    ExtractionStrategy,
    LangextractStrategy,
    MarkItDownStrategy,
    PyMuPDFStrategy,
    RegexStrategy,
)
logger = logging.getLogger(__name__)
class ContentExtractor:
    """Extracts clean text from bioRxiv paper HTML or PDF.

    Supports HTML extraction via langextract (with regex fallback) and
    PDF extraction via Microsoft MarkItDown. Optionally removes References and
    Acknowledgements sections from extracted text.
    """

    def __init__(
        self,
        remove_references: bool = True,
        remove_acknowledgements: bool = True,
        config: ExtractionConfig | None = None,
    ) -> None:
        self.remove_references = remove_references
        self.remove_acknowledgements = remove_acknowledgements
        self.config = config or ExtractionConfig(
            remove_references=remove_references,
            remove_acknowledgements=remove_acknowledgements,
        )
        self.strategies = self._initialize_strategies()
        self._temp_files: list[str] = []  # Track temporary files for cleanup
        self.last_metadata: dict[str, Any] | None = None  # Last extraction metadata
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_extractions)

    def _initialize_strategies(self) -> list[ExtractionStrategy]:
        """Initialize extraction strategies based on configuration.

        Creates a list of extraction strategies in priority order based on the
        configuration settings. Only includes strategies that are enabled in the
        config and have their dependencies available (checked via is_available()).

        The priority order is:
        1. MarkItDown (if enabled) - Markdown-native PDF extraction
        2. PyMuPDF (if enabled) - Fast plain-text fallback for PDFs
        3. Langextract (if enabled) - Robust HTML parsing
        4. Regex (always enabled) - Final fallback for HTML

        Returns:
            List of available extraction strategies in priority order.
        """
        strategies: list[ExtractionStrategy] = []

        if self.config.enable_markitdown:
            strategies.append(MarkItDownStrategy(timeout=self.config.pdf_timeout))

        if self.config.enable_pymupdf:
            strategies.append(PyMuPDFStrategy(timeout=self.config.pdf_timeout))

        if self.config.enable_langextract:
            strategies.append(LangextractStrategy())

        # Regex strategy is always available as final fallback
        strategies.append(RegexStrategy())

        # Filter to only include strategies with available dependencies
        return [s for s in strategies if s.is_available()]

    def get_available_strategies(self) -> list[str]:
        """Return list of available strategy names.

        Checks each strategy in the initialized list and returns the names
        of strategies whose dependencies are currently available.

        Returns:
            List of strategy name strings for available strategies.
        """
        return [s.name for s in self.strategies if s.is_available()]

    def _cleanup_temp_files(self) -> None:
        """Clean up all tracked temporary files.

        Deletes all temporary files that were created during extraction and
        clears the tracking list. Logs warnings for files that cannot be deleted
        but does not raise exceptions.

        This method is safe to call multiple times and will skip files that
        have already been deleted.
        """
        import os

        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

        # Clear the list after cleanup attempt
        self._temp_files.clear()

    def _construct_pdf_url(self, html_url: str) -> str | None:
        """Convert bioRxiv HTML URL to PDF URL.

        BioRxiv URLs follow the pattern:
        - HTML: https://www.biorxiv.org/content/10.1101/YYYY.MM.DD.NNNNNN[vN]
        - PDF: https://www.biorxiv.org/content/10.1101/YYYY.MM.DD.NNNNNN[vN].full.pdf

        Args:
            html_url: BioRxiv HTML URL.

        Returns:
            PDF URL if conversion successful, None if URL format is invalid.

        Example:
            >>> url = "https://www.biorxiv.org/content/10.1101/2024.01.15.123456v1"
            >>> pdf_url = extractor._construct_pdf_url(url)
            >>> print(pdf_url)
            https://www.biorxiv.org/content/10.1101/2024.01.15.123456v1.full.pdf
        """
        if not html_url or "biorxiv.org" not in html_url:
            return None

        # Remove any existing .full.pdf suffix if present
        base_url = html_url.rstrip("/").replace(".full.pdf", "")

        # Add .full.pdf suffix
        return f"{base_url}.full.pdf"

    async def _download_pdf(self, url: str) -> str | None:
        """Download PDF from bioRxiv URL to temporary file, with optional caching.

        Downloads the PDF asynchronously using aiohttp with a configured timeout.
        If ``config.pdf_cache_dir`` is set, the PDF is cached on disk using a
        SHA-256 hash of the URL as the cache key.  On subsequent calls with the
        same URL the cached file is returned directly without downloading.

        Cached files are **not** added to ``self._temp_files`` so they persist
        across extractions.  Only freshly-downloaded temporary files are tracked
        for cleanup.

        Args:
            url: BioRxiv paper URL (HTML or PDF format).

        Returns:
            Path to PDF file on success (cached or temporary), None on failure.
        """
        import hashlib
        import os
        import shutil
        import tempfile

        # Convert HTML URL to PDF URL if needed
        pdf_url = self._construct_pdf_url(url)
        if pdf_url is None:
            logger.error(f"Failed to construct PDF URL from: {url}")
            return None

        # --- Cache lookup ---
        cache_path: str | None = None
        if self.config.pdf_cache_dir:
            cache_key = hashlib.sha256(url.encode()).hexdigest()
            cache_path = os.path.join(self.config.pdf_cache_dir, f"{cache_key}.pdf")
            if os.path.exists(cache_path):
                logger.info(f"PDF cache hit for {url} at {cache_path}")
                # Do NOT add to _temp_files – cached files should persist
                return cache_path

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    pdf_url,
                    timeout=aiohttp.ClientTimeout(
                        total=self.config.pdf_download_timeout
                    ),
                ) as response:
                    # Check for HTTP errors
                    if response.status == 404:
                        logger.warning(f"PDF not found (404) for URL: {url}")
                        return None

                    response.raise_for_status()

                    # Read PDF content
                    pdf_content = await response.read()

                    # Write to temporary file
                    # Use delete=False so we can return the path and manage cleanup ourselves
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=".pdf", delete=False
                    ) as tmp_file:
                        tmp_file.write(pdf_content)
                        tmp_path = tmp_file.name

                    # Track the temporary file for cleanup
                    self._temp_files.append(tmp_path)

                    logger.info(f"Downloaded PDF from {pdf_url} to {tmp_path}")

                    # --- Save to cache ---
                    if cache_path is not None:
                        try:
                            os.makedirs(self.config.pdf_cache_dir, exist_ok=True)  # type: ignore[arg-type]
                            shutil.copy2(tmp_path, cache_path)
                            logger.info(f"Cached PDF for {url} at {cache_path}")
                        except Exception as e:
                            logger.warning(f"Failed to cache PDF for {url}: {e}")

                    return tmp_path

        except aiohttp.ClientError as e:
            logger.error(f"Network error downloading PDF from {url}: {e}")
            return None
        except TimeoutError:
            logger.error(
                f"Timeout downloading PDF from {url} after {self.config.pdf_download_timeout}s"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF from {url}: {e}")
            return None

    def _execute_extraction_chain(
        self,
        source: ExtractionSource,
    ) -> ExtractionResult:
        """Execute extraction strategies in priority order until success.

        Iterates through the configured extraction strategies in priority order,
        attempting each strategy that supports the source type. The chain
        short-circuits on the first successful extraction, implementing a
        graceful fallback mechanism.

        For each strategy:
        1. Check if it supports the source type (PDF vs HTML)
        2. Attempt extraction and measure execution time
        3. Log success (INFO) or failure (WARNING) with method name and timing
        4. Return immediately on success (short-circuit)
        5. Continue to next strategy on failure

        If all strategies fail, returns an empty ExtractionResult with success=False.

        Args:
            source: ExtractionSource containing source_type, content, and metadata.

        Returns:
            ExtractionResult with success=True and extracted text on success,
            or success=False with empty text if all strategies fail.

        Example:
            >>> source = ExtractionSource(
            ...     source_type=SourceType.PDF,
            ...     content="/path/to/paper.pdf",
            ...     metadata={"doi": "10.1101/2024.01.001"}
            ... )
            >>> result = extractor._execute_extraction_chain(source)
            >>> if result.success:
            ...     print(f"Extracted with {result.method} in {result.execution_time:.2f}s")
        """
        # Extract paper identifier for traceability
        paper_id = (
            source.metadata.get("doi")
            or source.metadata.get("url")
            or source.metadata.get("pdf_path")
            or "unknown"
        )

        attempted_methods: list[str] = []

        for strategy in self.strategies:
            # Skip strategies that don't support this source type
            if not strategy.supports_source_type(source.source_type):
                continue

            # Log fallback warning if a prior method already failed
            if attempted_methods:
                logger.warning(
                    f"Falling back to {strategy.name} after "
                    f"{attempted_methods[-1]} failed for paper={paper_id}"
                )

            logger.info(
                f"Attempting extraction with {strategy.name} for paper={paper_id}"
            )

            try:
                result = strategy.extract(source)
                attempted_methods.append(strategy.name)

                if result.success:
                    # Preserve metadata from source in the result
                    result.metadata = source.metadata
                    # Normalize whitespace for section parser compatibility
                    result.text = ExtractionStrategy.normalize_whitespace(result.text)
                    logger.info(
                        f"Extraction succeeded with {strategy.name} "
                        f"in {result.execution_time:.2f}s "
                        f"for paper={paper_id}"
                    )
                    return result
                logger.warning(
                    f"Extraction failed with {strategy.name}: "
                    f"{result.error} for paper={paper_id}"
                )
            except Exception as e:
                attempted_methods.append(strategy.name)
                logger.warning(
                    f"Extraction failed with {strategy.name}: {e} for paper={paper_id}"
                )
                continue

        # All strategies failed
        logger.error(
            f"All extraction methods failed for paper={paper_id}. "
            f"Attempted methods: {attempted_methods}"
        )
        return ExtractionResult(
            success=False,
            text="",
            method="none",
            execution_time=0.0,
            error="All extraction strategies failed",
        )

    async def extract_from_url(
        self, url: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Download content from a URL and extract clean text using PDF-first strategy.

        This method implements the PDF-first extraction strategy:
        1. Attempt to download and extract from PDF
        2. If PDF download fails, fall back to HTML extraction

        Args:
            url: URL to the paper HTML page.
            metadata: Optional metadata dict (DOI, authors, publication_date, title, etc.)
                to preserve through extraction. Merged with internal metadata.

        Returns:
            Extracted clean text, or empty string on failure.
        """
        # Merge caller metadata with internal metadata
        merged_metadata = {"url": url}
        if metadata:
            merged_metadata.update(metadata)

        try:
            # Step 1: Try PDF-first extraction
            pdf_path = await self._download_pdf(url)

            if pdf_path is not None:
                # PDF download succeeded, create PDF extraction source
                source = ExtractionSource(
                    source_type=SourceType.PDF,
                    content=pdf_path,
                    metadata=merged_metadata,
                )
                result = self._execute_extraction_chain(source)

                if result.success:
                    # Store metadata for retrieval
                    self.last_metadata = result.metadata

                    # Apply section removal only when enabled
                    text = result.text
                    if self.remove_references or self.remove_acknowledgements:
                        text = self._remove_sections(text)

                    # Validate quality and log warning if needed
                    quality = self._validate_quality(text)
                    if not quality.valid:
                        logger.warning(
                            f"Extracted text quality below threshold for {url}: "
                            f"length={quality.length}, nonalpha_ratio={quality.nonalpha_ratio:.2f}"
                        )

                    return text

            # Step 2: PDF download failed or PDF extraction failed, fall back to HTML
            logger.info(f"Falling back to HTML extraction for {url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()

            # Create HTML extraction source
            source = ExtractionSource(
                source_type=SourceType.HTML,
                content=html,
                metadata=merged_metadata,
            )
            result = self._execute_extraction_chain(source)

            if result.success:
                # Store metadata for retrieval
                self.last_metadata = result.metadata

                # Apply section removal only when enabled
                text = result.text
                if self.remove_references or self.remove_acknowledgements:
                    text = self._remove_sections(text)

                # Validate quality and log warning if needed
                quality = self._validate_quality(text)
                if not quality.valid:
                    logger.warning(
                        f"Extracted text quality below threshold for {url}: "
                        f"length={quality.length}, nonalpha_ratio={quality.nonalpha_ratio:.2f}"
                    )

                return text

            # All extraction methods failed
            return ""

        except Exception as e:
            logger.error(f"Failed to extract content from URL {url}: {e}")
            return ""
        finally:
            # Clean up any temporary files created during extraction
            self._cleanup_temp_files()

    async def extract_batch(
        self,
        urls: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Extract text from multiple URLs concurrently with rate limiting.

        Uses asyncio.Semaphore to limit concurrent extractions to
        config.max_concurrent_extractions (default 5) to manage memory usage,
        especially when Nougat is enabled.

        Args:
            urls: List of paper URLs to extract from.
            metadata_list: Optional list of metadata dicts, one per URL.
                If provided, must have same length as urls.

        Returns:
            List of extracted text strings, one per URL. Failed extractions
            return empty strings.

        Raises:
            ValueError: If metadata_list is provided but has different length than urls.
        """
        if metadata_list and len(metadata_list) != len(urls):
            raise ValueError("metadata_list must have same length as urls")

        async def _extract_with_semaphore(
            url: str, metadata: dict[str, Any] | None
        ) -> str:
            async with self._semaphore:
                return await self.extract_from_url(url, metadata=metadata)

        tasks = []
        for i, url in enumerate(urls):
            meta = metadata_list[i] if metadata_list else None
            tasks.append(_extract_with_semaphore(url, meta))

        return list(await asyncio.gather(*tasks))

    def extract_from_html(
        self, html: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Extract clean text from an HTML string.

        This method uses the HTML extraction chain (langextract → regex)
        to extract text from the HTML content.

        Args:
            html: Raw HTML string.
            metadata: Optional metadata dict (DOI, authors, publication_date, title, etc.)
                to preserve through extraction. Merged with internal metadata.

        Returns:
            Extracted clean text, or empty string on failure.
        """
        if not html:
            return ""

        # Merge caller metadata with internal metadata
        merged_metadata: dict[str, Any] = {"source": "html"}
        if metadata:
            merged_metadata.update(metadata)

        try:
            # Create HTML extraction source
            source = ExtractionSource(
                source_type=SourceType.HTML,
                content=html,
                metadata=merged_metadata,
            )

            # Execute extraction chain
            result = self._execute_extraction_chain(source)

            if result.success:
                # Store metadata for retrieval
                self.last_metadata = result.metadata

                # Apply section removal only when enabled
                text = result.text
                if self.remove_references or self.remove_acknowledgements:
                    text = self._remove_sections(text)

                # Validate quality and log warning if needed
                quality = self._validate_quality(text)
                if not quality.valid:
                    logger.warning(
                        f"Extracted text quality below threshold for HTML: "
                        f"length={quality.length}, nonalpha_ratio={quality.nonalpha_ratio:.2f}"
                    )

                return text

            # All extraction methods failed
            return ""

        except Exception as e:
            logger.error(f"Failed to extract content from HTML: {e}")
            return ""

    def extract_from_pdf(
        self, pdf_path: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Extract text from a PDF file using the extraction chain.

        This method uses the PDF extraction chain (Nougat → MarkItDown → pdftotext)
        to extract text from the PDF file.

        Args:
            pdf_path: Path to the PDF file.
            metadata: Optional metadata dict (DOI, authors, publication_date, title, etc.)
                to preserve through extraction. Merged with internal metadata.

        Returns:
            Extracted text, or empty string on failure.
        """
        # Merge caller metadata with internal metadata
        merged_metadata: dict[str, Any] = {"pdf_path": pdf_path}
        if metadata:
            merged_metadata.update(metadata)

        try:
            # Create PDF extraction source
            source = ExtractionSource(
                source_type=SourceType.PDF,
                content=pdf_path,
                metadata=merged_metadata,
            )

            # Execute extraction chain
            result = self._execute_extraction_chain(source)

            if result.success:
                # Store metadata for retrieval
                self.last_metadata = result.metadata

                # Apply section removal only when enabled
                text = result.text
                if self.remove_references or self.remove_acknowledgements:
                    text = self._remove_sections(text)

                # Validate quality and log warning if needed
                quality = self._validate_quality(text)
                if not quality.valid:
                    logger.warning(
                        f"Extracted text quality below threshold for {pdf_path}: "
                        f"length={quality.length}, nonalpha_ratio={quality.nonalpha_ratio:.2f}"
                    )

                return text

            # All extraction methods failed
            return ""

        except Exception as e:
            logger.error(f"Failed to extract content from PDF {pdf_path}: {e}")
            return ""
        finally:
            # Clean up any temporary files created during extraction
            self._cleanup_temp_files()

    def _remove_sections(self, text: str) -> str:
        """Remove References and Acknowledgements sections from text.

        Matches section headings (case-insensitive) and removes everything
        from the heading to the next major section heading or end of text.

        A major section heading is detected as a line that starts with an
        optional number/punctuation prefix followed by a capitalised word
        (at least 4 chars) and nothing else on the line, similar to how
        scientific papers format their top-level headings.

        Args:
            text: Input text potentially containing sections to remove.

        Returns:
            Text with configured sections removed.
        """
        if not text:
            return text

        # Build pattern for section headings to remove
        sections_to_remove: list[str] = []
        if self.config.remove_references:
            sections_to_remove.append(r"References")
        if self.config.remove_acknowledgements:
            sections_to_remove.append(r"Acknowledgements?")

        if not sections_to_remove:
            return text

        # Pattern for a generic major section heading (used to detect where
        # a removed section ends and the next section begins).
        # Matches lines like: "Introduction", "1. Methods", "## Results", "DISCUSSION"
        major_heading_pattern = (
            r"^(?:\#{1,3}\s+|\d+[\.\)]\s*)?[A-Z][A-Za-z]{3,}(?:\s+[A-Za-z]+){0,4}\s*$"
        )

        for section_pattern in sections_to_remove:
            # Match the section heading line
            heading_re = re.compile(
                r"(?:^|\n)\s*(?:\#{1,3}\s+|\d+[\.\)]\s*)?" + section_pattern + r"\s*\n",
                re.IGNORECASE,
            )

            match = heading_re.search(text)
            if match is None:
                continue

            section_start = match.start()
            remaining = text[match.end() :]

            # Find the next major section heading after the removed section
            next_heading = re.search(major_heading_pattern, remaining, re.MULTILINE)

            if next_heading is not None:
                section_end = match.end() + next_heading.start()
            else:
                section_end = len(text)

            text = text[:section_start].rstrip() + "\n\n" + text[section_end:].lstrip()

        return text.strip()

    def _validate_quality(self, text: str) -> QualityMetrics:
        """Validate extracted text quality against configured thresholds.

        Calculates quality metrics for the extracted text including length and
        non-alphanumeric character ratio, then compares against configured
        thresholds to determine if the text meets minimum quality standards.

        The non-alphanumeric ratio is calculated as the count of characters that
        are neither alphanumeric nor whitespace, divided by the total text length.
        This helps identify text with excessive special characters, which may
        indicate extraction errors.

        Args:
            text: Extracted text to validate.

        Returns:
            QualityMetrics object containing:
                - valid: True if text meets both length and character ratio thresholds
                - length: Total character count
                - nonalpha_ratio: Ratio of non-alphanumeric, non-whitespace characters

        Example:
            >>> metrics = extractor._validate_quality("This is good text.")
            >>> if not metrics.valid:
            ...     logger.warning(f"Low quality: length={metrics.length}, "
            ...                    f"nonalpha_ratio={metrics.nonalpha_ratio:.2f}")
        """
        length = len(text)

        # Handle empty text case
        if length == 0:
            return QualityMetrics(valid=False, length=0, nonalpha_ratio=1.0)

        # Count non-alphanumeric, non-whitespace characters
        nonalpha_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        nonalpha_ratio = nonalpha_count / length

        # Check against configured thresholds
        valid = (
            length >= self.config.quality_min_length
            and nonalpha_ratio <= self.config.quality_max_nonalpha_ratio
        )

        return QualityMetrics(
            valid=valid,
            length=length,
            nonalpha_ratio=nonalpha_ratio,
        )

    @staticmethod
    def _extract_html_fallback(html: str) -> str:
        """Simple regex-based HTML tag stripping as a fallback.

        Removes script/style blocks, HTML tags, and collapses whitespace
        while preserving paragraph boundaries.

        Args:
            html: Raw HTML string.

        Returns:
            Plain text extracted from HTML.
        """
        # Remove script and style blocks
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Replace block-level elements with newlines to preserve paragraph boundaries
        block_tags = r"</?(p|div|br|h[1-6]|li|tr|blockquote|section|article|header|footer|nav|aside|main|figure|figcaption|pre|table|ul|ol|dl|dd|dt)\b[^>]*>"
        text = re.sub(block_tags, "\n", text, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode common HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        # Collapse multiple blank lines into double newlines (paragraph boundaries)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Collapse multiple spaces within lines
        text = re.sub(r"[ \t]+", " ", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Final cleanup of excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
