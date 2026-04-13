"""Extraction strategy interface and base class for the PDF-first extraction pipeline.

This module defines the abstract base class for extraction strategies, which implement
the Strategy pattern to encapsulate different content extraction methods (Nougat, PyMuPDF,
pdftotext, langextract, regex). Each strategy is responsible for:

1. Checking if its dependencies are available (is_available)
2. Determining if it can handle a given source type (supports_source_type)
3. Executing the extraction and returning a result (extract)
4. Providing a timeout value for the extraction operation (get_timeout)

The Strategy pattern allows the ContentExtractor to orchestrate a fallback chain of
extraction methods without tight coupling to specific implementations. New extraction
strategies can be added by subclassing ExtractionStrategy and implementing the required
abstract methods.

Example:
    >>> class MyCustomStrategy(ExtractionStrategy):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_custom_strategy"
    ...
    ...     def is_available(self) -> bool:
    ...         try:
    ...             import my_custom_lib
    ...             return True
    ...         except ImportError:
    ...             return False
    ...
    ...     def supports_source_type(self, source_type: SourceType) -> bool:
    ...         return source_type == SourceType.PDF
    ...
    ...     def extract(self, source: ExtractionSource) -> ExtractionResult:
    ...         # Implementation here
    ...         pass
"""

from abc import ABC, abstractmethod

from pipeline.ingest.extraction_models import (
    ExtractionResult,
    ExtractionSource,
    SourceType,
)


class ExtractionStrategy(ABC):
    """Abstract base class for content extraction strategies.

    This class defines the contract that all extraction strategies must implement.
    Strategies are used by the ContentExtractor to attempt extraction from various
    source types (PDF, HTML) using different methods (Nougat, PyMuPDF, pdftotext,
    langextract, regex).

    The extraction chain executes strategies in priority order, attempting each
    strategy until one succeeds or all strategies are exhausted. Each strategy
    is responsible for:

    - Checking if its dependencies are available (libraries, command-line tools)
    - Determining if it can handle a given source type (PDF vs HTML)
    - Executing the extraction and returning a structured result
    - Providing a timeout value for the extraction operation

    Strategies should handle errors gracefully and return an ExtractionResult
    with success=False rather than raising exceptions, allowing the extraction
    chain to proceed to the next strategy.

    Attributes:
        name: Human-readable name for the strategy (used in logging)

    Example:
        >>> strategy = MyStrategy()
        >>> if strategy.is_available():
        ...     source = ExtractionSource(
        ...         source_type=SourceType.PDF,
        ...         content="/path/to/paper.pdf",
        ...         metadata={"doi": "10.1101/2024.01.001"}
        ...     )
        ...     result = strategy.extract(source)
        ...     if result.success:
        ...         print(f"Extracted {len(result.text)} characters")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this extraction strategy.

        The name is used for logging and debugging purposes to identify which
        strategy was used for extraction. It should be a short, descriptive
        identifier (e.g., "nougat", "pymupdf", "langextract").

        Returns:
            A string identifier for this strategy.

        Example:
            >>> strategy = NougatStrategy()
            >>> strategy.name
            'nougat'
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy's dependencies are available.

        This method should verify that all required dependencies (libraries,
        command-line tools, models) are installed and accessible. It should
        not raise exceptions, but return False if dependencies are missing.

        The ContentExtractor uses this method during initialization to filter
        out unavailable strategies from the extraction chain.

        Returns:
            True if the strategy can be used, False if dependencies are missing.

        Example:
            >>> strategy = NougatStrategy()
            >>> strategy.is_available()
            True  # If nougat library is installed

            >>> strategy = PdftotextStrategy()
            >>> strategy.is_available()
            False  # If pdftotext command is not in PATH
        """

    @abstractmethod
    def supports_source_type(self, source_type: SourceType) -> bool:
        """Check if this strategy can handle the given source type.

        Different strategies support different source types:
        - PDF strategies (Nougat, PyMuPDF, pdftotext) support SourceType.PDF
        - HTML strategies (langextract, regex) support SourceType.HTML

        The ContentExtractor uses this method to skip strategies that cannot
        handle the current source type.

        Args:
            source_type: The type of source to check (PDF or HTML).

        Returns:
            True if this strategy can handle the source type, False otherwise.

        Example:
            >>> strategy = NougatStrategy()
            >>> strategy.supports_source_type(SourceType.PDF)
            True
            >>> strategy.supports_source_type(SourceType.HTML)
            False
        """

    @abstractmethod
    def extract(self, source: ExtractionSource) -> ExtractionResult:
        """Execute extraction on the given source and return the result.

        This method performs the actual content extraction using the strategy's
        specific implementation (Nougat model, PyMuPDF library, pdftotext command,
        etc.). It should:

        1. Validate that the source type is supported
        2. Execute the extraction with appropriate timeout handling
        3. Return an ExtractionResult with success=True and extracted text on success
        4. Return an ExtractionResult with success=False and error message on failure
        5. Handle all exceptions internally and return a failure result

        The method should NOT raise exceptions, as this would break the fallback
        chain. Instead, catch exceptions and return a failure result with the
        error message.

        Args:
            source: The source content to extract from, including source type,
                   content (file path for PDF, HTML string for HTML), and metadata.

        Returns:
            An ExtractionResult containing:
            - success: True if extraction succeeded, False otherwise
            - text: Extracted text content (empty string on failure)
            - method: Name of the extraction method (same as self.name)
            - execution_time: Time taken for extraction in seconds
            - error: Error message if extraction failed, None otherwise

        Example:
            >>> strategy = PyMuPDFStrategy()
            >>> source = ExtractionSource(
            ...     source_type=SourceType.PDF,
            ...     content="/path/to/paper.pdf",
            ...     metadata={"doi": "10.1101/2024.01.001"}
            ... )
            >>> result = strategy.extract(source)
            >>> if result.success:
            ...     print(f"Extracted {len(result.text)} characters in {result.execution_time:.2f}s")
            ... else:
            ...     print(f"Extraction failed: {result.error}")
        """

    def get_timeout(self) -> int:
        """Return the timeout in seconds for this extraction strategy.

        This method provides the maximum time allowed for the extraction operation.
        If the extraction exceeds this timeout, it should be terminated and the
        next strategy in the chain should be attempted.

        The default implementation returns 30 seconds, which is appropriate for
        most extraction methods. Strategies that require longer processing times
        (e.g., Nougat with 120 seconds) should override this method.

        Returns:
            Timeout in seconds as an integer.

        Example:
            >>> strategy = PyMuPDFStrategy()
            >>> strategy.get_timeout()
            30  # Default timeout

            >>> strategy = NougatStrategy(timeout=120)
            >>> strategy.get_timeout()
            120  # Custom timeout for Nougat
        """
        return 30

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace for consistent section parser compatibility.

        Applies two normalizations to ensure extracted text has consistent
        whitespace regardless of which extraction strategy produced it:

        1. Multiple consecutive spaces/tabs → single space
        2. Three or more consecutive newlines → double newline (paragraph boundary)

        This ensures paragraph boundaries (double newlines) are preserved while
        removing excessive whitespace that could interfere with section heading
        detection in the SectionParser.

        Args:
            text: The extracted text to normalize.

        Returns:
            Text with normalized whitespace.

        Example:
            >>> ExtractionStrategy.normalize_whitespace("hello   world")
            'hello world'
            >>> ExtractionStrategy.normalize_whitespace("para1\\n\\n\\n\\npara2")
            'para1\\n\\npara2'
        """
        import re

        # Collapse multiple spaces/tabs to single space (within lines)
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines to double newline (paragraph boundary)
        return re.sub(r"\n{3,}", "\n\n", text)


class RegexStrategy(ExtractionStrategy):
    """Regex-based HTML tag stripping as final fallback.

    This strategy provides a simple, always-available fallback for HTML content
    extraction when more sophisticated methods (like langextract) fail or are
    unavailable. It uses regular expressions to:

    1. Remove script and style blocks
    2. Replace block-level HTML elements with newlines
    3. Remove all remaining HTML tags
    4. Decode common HTML entities
    5. Normalize whitespace

    This strategy is always available (no external dependencies) and serves as
    the last resort in the extraction chain to ensure that some text can always
    be extracted from HTML sources.

    Example:
        >>> strategy = RegexStrategy()
        >>> source = ExtractionSource(
        ...     source_type=SourceType.HTML,
        ...     content="<p>Hello <b>world</b></p>",
        ...     metadata={}
        ... )
        >>> result = strategy.extract(source)
        >>> result.text
        'Hello world'
    """

    @property
    def name(self) -> str:
        """Return the name of this strategy.

        Returns:
            The string "regex" identifying this strategy.
        """
        return "regex"

    def is_available(self) -> bool:
        """Check if this strategy is available.

        The regex strategy is always available as it only uses Python's
        built-in re module with no external dependencies.

        Returns:
            Always returns True.
        """
        return True

    def supports_source_type(self, source_type: SourceType) -> bool:
        """Check if this strategy supports the given source type.

        The regex strategy only supports HTML sources, as it is designed
        to strip HTML tags and extract text content.

        Args:
            source_type: The source type to check.

        Returns:
            True if source_type is HTML, False otherwise.
        """
        return source_type == SourceType.HTML

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        """Extract text from HTML using regex-based tag stripping.

        This method applies a series of regex transformations to remove HTML
        markup and extract plain text content. It handles script/style removal,
        block element conversion, tag stripping, entity decoding, and whitespace
        normalization.

        Args:
            source: The extraction source containing HTML content.

        Returns:
            An ExtractionResult with the extracted text or an error message.
        """
        import time

        start_time = time.time()

        try:
            text = self._strip_html_tags(source.content)
            execution_time = time.time() - start_time

            return ExtractionResult(
                success=True,
                text=text,
                method="regex",
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="regex",
                execution_time=execution_time,
                error=str(e),
            )

    @staticmethod
    def _strip_html_tags(html: str) -> str:
        """Strip HTML tags and extract plain text using regex.

        This method performs the following transformations:
        1. Remove script and style blocks (including their content)
        2. Replace block-level HTML elements with newlines
        3. Remove all remaining HTML tags
        4. Decode common HTML entities (&amp;, &lt;, &gt;, etc.)
        5. Normalize whitespace (collapse multiple spaces/newlines)

        Args:
            html: The HTML string to process.

        Returns:
            Plain text with HTML markup removed and whitespace normalized.

        Example:
            >>> RegexStrategy._strip_html_tags("<p>Hello</p><p>World</p>")
            'Hello\\n\\nWorld'
        """
        import re

        # Remove script and style blocks
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Replace block-level elements with newlines
        block_tags = r"</?(p|div|br|h[1-6]|li|tr|blockquote|section|article|header|footer|nav|aside|main|figure|figcaption|pre|table|ul|ol|dl|dd|dt)\b[^>]*>"
        text = re.sub(block_tags, "\n", text, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        # Normalize whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


class LangextractStrategy(ExtractionStrategy):
    """HTML extraction using Google's langextract library.

    This strategy uses the langextract library to extract main content from HTML
    documents. Langextract is more robust than regex-based extraction as it uses
    content analysis to identify and extract the primary text content while
    filtering out navigation, ads, and other non-content elements.

    This strategy is used as the primary HTML extraction method, falling back to
    RegexStrategy if langextract is unavailable or fails.

    Example:
        >>> strategy = LangextractStrategy()
        >>> if strategy.is_available():
        ...     source = ExtractionSource(
        ...         source_type=SourceType.HTML,
        ...         content="<html><body><article>Main content</article></body></html>",
        ...         metadata={"doi": "10.1101/2024.01.001"}
        ...     )
        ...     result = strategy.extract(source)
        ...     if result.success:
        ...         print(f"Extracted: {result.text}")
    """

    @property
    def name(self) -> str:
        """Return the name of this strategy.

        Returns:
            The string "langextract" identifying this strategy.
        """
        return "langextract"

    def is_available(self) -> bool:
        """Check if langextract library is available.

        Attempts to import the langextract library to verify it is installed.
        Returns False if the import fails, allowing the extraction chain to
        skip this strategy and proceed to the regex fallback.

        Returns:
            True if langextract is installed, False otherwise.
        """
        try:
            import langextract  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_source_type(self, source_type: SourceType) -> bool:
        """Check if this strategy supports the given source type.

        The langextract strategy only supports HTML sources, as it is designed
        to extract content from HTML documents.

        Args:
            source_type: The source type to check.

        Returns:
            True if source_type is HTML, False otherwise.
        """
        return source_type == SourceType.HTML

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        """Extract text from HTML using langextract.

        This method uses the langextract.extract() function to extract the main
        content from HTML. It handles timing, error handling, and returns a
        structured result.

        Args:
            source: The extraction source containing HTML content.

        Returns:
            An ExtractionResult with success=True and extracted text on success,
            or success=False with an error message on failure.

        Example:
            >>> strategy = LangextractStrategy()
            >>> source = ExtractionSource(
            ...     source_type=SourceType.HTML,
            ...     content="<html><body><p>Content</p></body></html>",
            ...     metadata={}
            ... )
            >>> result = strategy.extract(source)
            >>> result.success
            True
        """
        import time

        start_time = time.time()

        try:
            from langextract import extract

            # Extract main content from HTML
            text = extract(source.content)
            execution_time = time.time() - start_time

            return ExtractionResult(
                success=True,
                text=text,
                method="langextract",
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="langextract",
                execution_time=execution_time,
                error=str(e),
            )


class PdftotextStrategy(ExtractionStrategy):
    """PDF extraction using pdftotext command-line tool.

    This strategy uses the pdftotext command-line utility to extract text from
    PDF files. It provides a fast, reliable alternative to PyMuPDF for simple
    PDF documents. The pdftotext tool is part of the poppler-utils package and
    is commonly available on Linux systems.

    This strategy is used as a tertiary PDF extraction method, after Nougat and
    PyMuPDF, but before falling back to HTML extraction. It supports configurable
    timeouts to prevent hanging on problematic PDFs.

    Attributes:
        timeout: Maximum time in seconds to wait for pdftotext to complete.

    Example:
        >>> strategy = PdftotextStrategy(timeout=10)
        >>> if strategy.is_available():
        ...     source = ExtractionSource(
        ...         source_type=SourceType.PDF,
        ...         content="/path/to/paper.pdf",
        ...         metadata={"doi": "10.1101/2024.01.001"}
        ...     )
        ...     result = strategy.extract(source)
        ...     if result.success:
        ...         print(f"Extracted {len(result.text)} characters")
    """

    def __init__(self, timeout: int = 10):
        """Initialize the PdftotextStrategy.

        Args:
            timeout: Maximum time in seconds to wait for pdftotext to complete.
                    Defaults to 10 seconds as specified in the design document.
        """
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the name of this strategy.

        Returns:
            The string "pdftotext" identifying this strategy.
        """
        return "pdftotext"

    def is_available(self) -> bool:
        """Check if pdftotext command is available.

        Uses shutil.which() to check if the pdftotext command-line tool is
        installed and accessible in the system PATH. Returns False if the
        command is not found, allowing the extraction chain to skip this
        strategy.

        Returns:
            True if pdftotext command is available, False otherwise.

        Example:
            >>> strategy = PdftotextStrategy()
            >>> strategy.is_available()
            True  # If pdftotext is installed
        """
        import shutil

        return shutil.which("pdftotext") is not None

    def supports_source_type(self, source_type: SourceType) -> bool:
        """Check if this strategy supports the given source type.

        The pdftotext strategy only supports PDF sources, as it is designed
        to extract text from PDF files using the pdftotext command-line tool.

        Args:
            source_type: The source type to check.

        Returns:
            True if source_type is PDF, False otherwise.
        """
        return source_type == SourceType.PDF

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        """Extract text from PDF using pdftotext command-line tool.

        This method executes the pdftotext command as a subprocess with the
        following behavior:

        1. Runs pdftotext with the PDF file path and "-" to output to stdout
        2. Captures both stdout (extracted text) and stderr (error messages)
        3. Enforces the configured timeout using subprocess.run(timeout=...)
        4. Handles TimeoutExpired exception and returns failure result
        5. Checks return code and stderr for errors
        6. Returns success result with extracted text on success

        Args:
            source: The extraction source containing PDF file path in content field.

        Returns:
            An ExtractionResult with:
            - success=True and extracted text if pdftotext succeeds
            - success=False with "Timeout exceeded" error if timeout occurs
            - success=False with error message if pdftotext fails

        Example:
            >>> strategy = PdftotextStrategy(timeout=10)
            >>> source = ExtractionSource(
            ...     source_type=SourceType.PDF,
            ...     content="/tmp/paper.pdf",
            ...     metadata={}
            ... )
            >>> result = strategy.extract(source)
            >>> if result.success:
            ...     print(f"Extracted in {result.execution_time:.2f}s")
            ... else:
            ...     print(f"Failed: {result.error}")
        """
        import subprocess
        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                ["pdftotext", source.content, "-"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                execution_time = time.time() - start_time
                return ExtractionResult(
                    success=False,
                    text="",
                    method="pdftotext",
                    execution_time=execution_time,
                    error=f"pdftotext failed: {result.stderr}",
                )

            text = result.stdout
            execution_time = time.time() - start_time

            return ExtractionResult(
                success=True,
                text=text,
                method="pdftotext",
                execution_time=execution_time,
            )
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="pdftotext",
                execution_time=execution_time,
                error="Timeout exceeded",
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="pdftotext",
                execution_time=execution_time,
                error=str(e),
            )

    def get_timeout(self) -> int:
        """Return the configured timeout for this strategy.

        Returns:
            The timeout value in seconds configured during initialization.

        Example:
            >>> strategy = PdftotextStrategy(timeout=15)
            >>> strategy.get_timeout()
            15
        """
        return self.timeout


class PyMuPDFStrategy(ExtractionStrategy):
    """Fast PDF extraction using PyMuPDF (fitz).

    This strategy uses the PyMuPDF library (imported as 'fitz') to extract text
    from PDF files. PyMuPDF provides fast, reliable text extraction for simple
    PDF documents and serves as the secondary PDF extraction method after Nougat.

    This strategy is used when Nougat fails or is unavailable, providing a fast
    fallback for PDF extraction before attempting pdftotext or HTML extraction.
    It supports configurable timeouts, though PyMuPDF typically completes quickly
    for most documents.

    Attributes:
        timeout: Maximum time in seconds to wait for extraction to complete.

    Example:
        >>> strategy = PyMuPDFStrategy(timeout=10)
        >>> if strategy.is_available():
        ...     source = ExtractionSource(
        ...         source_type=SourceType.PDF,
        ...         content="/path/to/paper.pdf",
        ...         metadata={"doi": "10.1101/2024.01.001"}
        ...     )
        ...     result = strategy.extract(source)
        ...     if result.success:
        ...         print(f"Extracted {len(result.text)} characters")
    """

    def __init__(self, timeout: int = 10):
        """Initialize the PyMuPDFStrategy.

        Args:
            timeout: Maximum time in seconds to wait for extraction to complete.
                    Defaults to 10 seconds as specified in the design document.
        """
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Return the name of this strategy.

        Returns:
            The string "pymupdf" identifying this strategy.
        """
        return "pymupdf"

    def is_available(self) -> bool:
        """Check if PyMuPDF (fitz) library is available.

        Attempts to import the fitz module (PyMuPDF) to verify it is installed.
        Returns False if the import fails, allowing the extraction chain to
        skip this strategy and proceed to the next PDF extraction method.

        Returns:
            True if PyMuPDF is installed, False otherwise.

        Example:
            >>> strategy = PyMuPDFStrategy()
            >>> strategy.is_available()
            True  # If PyMuPDF is installed
        """
        try:
            import fitz  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_source_type(self, source_type: SourceType) -> bool:
        """Check if this strategy supports the given source type.

        The PyMuPDF strategy only supports PDF sources, as it is designed
        to extract text from PDF files using the PyMuPDF library.

        Args:
            source_type: The source type to check.

        Returns:
            True if source_type is PDF, False otherwise.
        """
        return source_type == SourceType.PDF

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        """Extract text from PDF using PyMuPDF.

        This method uses PyMuPDF (fitz) to open the PDF file and extract text
        from all pages. The extraction process:

        1. Opens the PDF file using fitz.open()
        2. Iterates through all pages in the document
        3. Extracts text from each page using page.get_text()
        4. Joins all pages with newlines to preserve page boundaries
        5. Ensures the document is closed in a finally block
        6. Returns the extracted text with timing information

        Args:
            source: The extraction source containing PDF file path in content field.

        Returns:
            An ExtractionResult with:
            - success=True and extracted text if extraction succeeds
            - success=False with error message if extraction fails
            - execution_time tracking the time taken for extraction

        Example:
            >>> strategy = PyMuPDFStrategy()
            >>> source = ExtractionSource(
            ...     source_type=SourceType.PDF,
            ...     content="/tmp/paper.pdf",
            ...     metadata={}
            ... )
            >>> result = strategy.extract(source)
            >>> if result.success:
            ...     print(f"Extracted {len(result.text)} characters in {result.execution_time:.2f}s")
            ... else:
            ...     print(f"Failed: {result.error}")
        """
        import time

        start_time = time.time()
        doc = None

        try:
            import fitz

            # Open the PDF document
            doc = fitz.open(source.content)

            # Extract text from all pages
            pages = []
            for page in doc:
                pages.append(page.get_text())

            # Join pages with newlines
            text = "\n".join(pages)
            execution_time = time.time() - start_time

            return ExtractionResult(
                success=True,
                text=text,
                method="pymupdf",
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="pymupdf",
                execution_time=execution_time,
                error=str(e),
            )
        finally:
            # Ensure document is closed even if an error occurs
            if doc is not None:
                doc.close()

    def get_timeout(self) -> int:
        """Return the configured timeout for this strategy.

        Returns:
            The timeout value in seconds configured during initialization.

        Example:
            >>> strategy = PyMuPDFStrategy(timeout=15)
            >>> strategy.get_timeout()
            15
        """
        return self.timeout


class NougatStrategy(ExtractionStrategy):
    """Nougat-based PDF extraction for complex scientific papers.

    This strategy uses Facebook Research's Nougat model, a learning-based PDF
    parser trained specifically on scientific documents. Nougat outputs structured
    markdown that preserves mathematical notation, tables, and document structure,
    making it ideal for complex scientific papers.

    This strategy is used as the primary PDF extraction method, before falling
    back to simpler methods like PyMuPDF or pdftotext. It supports configurable
    timeouts (default 120 seconds) to prevent hanging on problematic PDFs, and
    uses lazy model loading to avoid loading the model until needed.

    The strategy converts Nougat's markdown output to plain text while preserving
    section boundaries, making it compatible with downstream section parsing.

    Attributes:
        timeout: Maximum time in seconds to wait for Nougat inference to complete.
        model: Lazily-loaded Nougat model instance (None until first extraction).

    Example:
        >>> strategy = NougatStrategy(timeout=120)
        >>> if strategy.is_available():
        ...     source = ExtractionSource(
        ...         source_type=SourceType.PDF,
        ...         content="/path/to/paper.pdf",
        ...         metadata={"doi": "10.1101/2024.01.001"}
        ...     )
        ...     result = strategy.extract(source)
        ...     if result.success:
        ...         print(f"Extracted {len(result.text)} characters")
    """

    def __init__(self, timeout: int = 120):
        """Initialize the NougatStrategy.

        Args:
            timeout: Maximum time in seconds to wait for Nougat inference to complete.
                    Defaults to 120 seconds as specified in the design document.
        """
        self.timeout = timeout
        self.model = None  # type: ignore

    @property
    def name(self) -> str:
        """Return the name of this strategy.

        Returns:
            The string "nougat" identifying this strategy.
        """
        return "nougat"

    def is_available(self) -> bool:
        """Check if Nougat library is available.

        Attempts to import the nougat module to verify it is installed.
        Returns False if the import fails, allowing the extraction chain to
        skip this strategy and proceed to the next PDF extraction method.

        Returns:
            True if Nougat is installed, False otherwise.

        Example:
            >>> strategy = NougatStrategy()
            >>> strategy.is_available()
            True  # If nougat library is installed
        """
        try:
            import nougat  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_source_type(self, source_type: SourceType) -> bool:
        """Check if this strategy supports the given source type.

        The Nougat strategy only supports PDF sources, as it is designed
        to extract text from PDF files using the Nougat model.

        Args:
            source_type: The source type to check.

        Returns:
            True if source_type is PDF, False otherwise.
        """
        return source_type == SourceType.PDF

    def get_timeout(self) -> int:
        """Return the configured timeout for this strategy.

        Returns:
            The timeout value in seconds configured during initialization.

        Example:
            >>> strategy = NougatStrategy(timeout=180)
            >>> strategy.get_timeout()
            180
        """
        return self.timeout

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        """Extract text from PDF using Nougat model.

        This method uses the Nougat model to extract structured markdown from
        PDF files, then converts the markdown to plain text while preserving
        section boundaries. The extraction process:

        1. Lazily loads the Nougat model on first use
        2. Runs inference with timeout enforcement using threading
        3. Converts markdown output to plain text
        4. Returns the extracted text with timing information
        5. Handles TimeoutError and returns failure result
        6. Handles all other exceptions and returns failure result

        Args:
            source: The extraction source containing PDF file path in content field.

        Returns:
            An ExtractionResult with:
            - success=True and extracted text if extraction succeeds
            - success=False with "Timeout exceeded" error if timeout occurs
            - success=False with error message if extraction fails
            - execution_time tracking the time taken for extraction

        Example:
            >>> strategy = NougatStrategy(timeout=120)
            >>> source = ExtractionSource(
            ...     source_type=SourceType.PDF,
            ...     content="/tmp/paper.pdf",
            ...     metadata={}
            ... )
            >>> result = strategy.extract(source)
            >>> if result.success:
            ...     print(f"Extracted in {result.execution_time:.2f}s")
            ... else:
            ...     print(f"Failed: {result.error}")
        """
        import time

        start_time = time.time()

        try:
            from nougat import NougatModel

            # Lazy load model
            if self.model is None:
                self.model = NougatModel.from_pretrained()  # type: ignore

            # Run inference with timeout
            markdown_output = self._run_with_timeout(
                self.model.predict,  # type: ignore
                source.content,
                timeout=self.timeout,
            )

            # Convert markdown to plain text while preserving structure
            text = self._markdown_to_text(markdown_output)

            execution_time = time.time() - start_time
            return ExtractionResult(
                success=True,
                text=text,
                method="nougat",
                execution_time=execution_time,
            )
        except TimeoutError:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="nougat",
                execution_time=execution_time,
                error="Timeout exceeded",
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                text="",
                method="nougat",
                execution_time=execution_time,
                error=str(e),
            )

    def _run_with_timeout(self, func, *args, timeout: int):  # type: ignore
        """Execute function with timeout using threading.

        This method runs the given function in a separate thread and enforces
        a timeout. If the function completes within the timeout, its result is
        returned. If the timeout is exceeded, a TimeoutError is raised.

        The thread is not forcibly terminated if it exceeds the timeout (Python
        doesn't support thread termination), but the main thread will proceed
        and the result will be discarded.

        Args:
            func: The function to execute.
            *args: Positional arguments to pass to the function.
            timeout: Maximum time in seconds to wait for the function to complete.

        Returns:
            The return value of the function if it completes within the timeout.

        Raises:
            TimeoutError: If the function does not complete within the timeout.
            Exception: Any exception raised by the function is re-raised.

        Example:
            >>> def slow_function(x):
            ...     time.sleep(5)
            ...     return x * 2
            >>> strategy = NougatStrategy()
            >>> result = strategy._run_with_timeout(slow_function, 10, timeout=10)
            >>> result
            20
        """
        import threading

        result = [None]  # type: ignore
        exception = [None]  # type: ignore

        def target():  # type: ignore
            try:
                result[0] = func(*args)
            except Exception as e:
                exception[0] = e  # type: ignore

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Execution exceeded {timeout}s timeout")

        if exception[0]:
            raise exception[0]

        return result[0]

    def _markdown_to_text(self, markdown: str) -> str:
        """Convert Nougat markdown output to plain text.

        This method removes markdown formatting while preserving document structure:
        1. Removes markdown heading markers (# ## ###) but keeps heading text
        2. Removes bold formatting (**text**)
        3. Removes italic formatting (*text*)
        4. Removes link formatting ([text](url)) but keeps link text
        5. Preserves paragraph boundaries (double newlines)
        6. Preserves section boundaries

        The goal is to produce plain text that is compatible with the existing
        SectionParser while maintaining the structural information from Nougat's
        output.

        Args:
            markdown: The markdown text output from Nougat.

        Returns:
            Plain text with markdown formatting removed and structure preserved.

        Example:
            >>> strategy = NougatStrategy()
            >>> markdown = "# Introduction\\n\\nThis is **bold** text.\\n\\n## Methods\\n\\nMore text."
            >>> strategy._markdown_to_text(markdown)
            'Introduction\\n\\nThis is bold text.\\n\\nMethods\\n\\nMore text.'
        """
        import re

        # Remove markdown heading markers but keep the heading text
        text = re.sub(r"^#{1,6}\s+", "", markdown, flags=re.MULTILINE)

        # Remove bold formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

        # Remove italic formatting
        text = re.sub(r"\*(.+?)\*", r"\1", text)

        # Remove link formatting but keep link text
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

        return text.strip()
