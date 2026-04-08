"""
PDF Processor for vision-based content extraction.

Uses a vision-capable LLM (e.g., Llama-3.2-11B-Vision-Instruct) to extract
text, figures, tables, and metadata from PDF documents by converting pages
to images and sending them to the vision model.
"""

import base64
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from src.utils.logging_utils import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class CorruptedPDFError(Exception):
    """Raised when a PDF file is corrupted or unreadable."""


class VisionModelTimeoutError(Exception):
    """Raised when the vision model times out during extraction."""


# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract the following information from this PDF page:

1. All text content (paragraphs, headings, captions)
2. Figures: Describe each figure and its caption
3. Tables: Extract table data in structured format
4. Metadata (if first page): Title, authors, abstract

Return as JSON:
{
  "text": "...",
  "figures": [{"caption": "...", "description": "..."}],
  "tables": [{"caption": "...", "data": [[...]]}],
  "metadata": {"title": "...", "authors": [...], "abstract": "..."}
}
"""


# ---------------------------------------------------------------------------
# Vision Capability Inference
# ---------------------------------------------------------------------------


def infer_vision_capability(model_name: str) -> bool:
    """Infer whether a model is vision-capable from its name.

    A model is considered vision-capable if its name contains "vision"
    (case-insensitive).

    Args:
        model_name: The model identifier string.

    Returns:
        True if the model is recognized as vision-capable.
    """
    return "vision" in model_name.lower()


# ---------------------------------------------------------------------------
# PDF Page to Image Conversion
# ---------------------------------------------------------------------------


def convert_pdf_page_to_base64_image(page: Any) -> str:
    """Convert a single PyMuPDF page to a base64-encoded PNG string.

    Args:
        page: A ``fitz.Page`` object.

    Returns:
        Base64-encoded PNG image string.
    """
    # Render at 2x resolution for better OCR quality
    pix = page.get_pixmap(dpi=150)
    image_bytes = pix.tobytes("png")
    return base64.b64encode(image_bytes).decode("utf-8")


def build_extraction_prompt(page_number: int, total_pages: int) -> str:
    """Build the extraction prompt for a given page.

    Args:
        page_number: 1-based page number.
        total_pages: Total number of pages in the document.

    Returns:
        Formatted prompt string.
    """
    return f"Page {page_number} of {total_pages}.\n\n{EXTRACTION_PROMPT}"


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------


def parse_vision_response(response_text: str) -> dict[str, Any]:
    """Parse the vision model response into a structured dict.

    Handles JSON wrapped in markdown code blocks as well as raw JSON.

    Args:
        response_text: Raw text response from the vision model.

    Returns:
        Dict with keys: text, figures, tables, metadata.
    """
    json_text = response_text

    # Strip markdown code fences if present
    if "```json" in json_text:
        start = json_text.find("```json") + 7
        end = json_text.find("```", start)
        json_text = json_text[start:end].strip()
    elif "```" in json_text:
        start = json_text.find("```") + 3
        end = json_text.find("```", start)
        json_text = json_text[start:end].strip()

    try:
        parsed = json.loads(json_text)
        return {
            "text": parsed.get("text", ""),
            "figures": parsed.get("figures", []),
            "tables": parsed.get("tables", []),
            "metadata": parsed.get("metadata", {}),
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse vision model response as JSON, using raw text")
        return {
            "text": response_text,
            "figures": [],
            "tables": [],
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# PDFProcessor
# ---------------------------------------------------------------------------


class PDFProcessor:
    """Processes PDF files using vision models.

    Converts PDF pages to images, sends them to a vision-capable LLM,
    and parses the structured responses into text, figures, tables,
    and metadata.
    """

    def __init__(self, model: str = "bedrock") -> None:
        """Initialize with the vision processor for actual image understanding.

        Args:
            model: Ignored (kept for API compatibility). Always uses VisionProcessor.
        """
        from pipeline.processors.vision_processor import VisionProcessor

        self.model_name = model
        self.vision = VisionProcessor()
        logger.info(
            f"PDFProcessor initialized with VisionProcessor "
            f"(model={self.vision.vision_model_id})"
        )

    async def extract_content(self, pdf_path: str) -> dict[str, Any]:
        """Extract text, figures, tables, and metadata from a single PDF.

        Args:
            pdf_path: Filesystem path to the PDF file.

        Returns:
            Dict with keys ``text``, ``figures``, ``tables``, ``metadata``.

        Raises:
            CorruptedPDFError: If the PDF cannot be opened or read.
            VisionModelTimeoutError: If the vision model times out.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install pymupdf"
            )

        path = Path(pdf_path)
        if not path.exists():
            raise CorruptedPDFError(f"PDF file not found: {pdf_path}")

        # Open PDF
        try:
            pdf_doc = fitz.open(pdf_path)
        except Exception as exc:
            raise CorruptedPDFError(f"Failed to open PDF '{path.name}': {exc}") from exc

        try:
            total_pages = len(pdf_doc)
            all_text: list[str] = []
            all_figures: list[dict[str, Any]] = []
            all_tables: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}

            # Strategy 1: Extract selectable text directly (fast, no model needed)
            for page_idx in range(total_pages):
                page = pdf_doc[page_idx]
                page_text = page.get_text("text").strip()
                if page_text:
                    all_text.append(page_text)

            # If we got meaningful text (>100 chars), use it directly
            combined_text = "\n\n".join(all_text)
            if len(combined_text) > 100:
                logger.info(
                    f"Extracted {len(combined_text)} chars of text from {total_pages} pages "
                    f"of '{path.name}' using direct text extraction"
                )

                # Extract metadata from first page text
                lines = combined_text.split("\n")
                title_candidate = next(
                    (l.strip() for l in lines if len(l.strip()) > 10), ""
                )
                metadata = {"title": title_candidate[:200]}

                return {
                    "text": combined_text,
                    "figures": [],
                    "tables": [],
                    "metadata": metadata,
                }

            # Strategy 2: Fall back to vision model for scanned/image PDFs
            logger.info(
                f"No selectable text in '{path.name}', falling back to vision model"
            )
            all_text = []
            for page_idx in range(total_pages):
                page = pdf_doc[page_idx]
                page_number = page_idx + 1

                image_b64 = convert_pdf_page_to_base64_image(page)
                prompt = build_extraction_prompt(page_number, total_pages)

                try:
                    response = await self._call_vision_model(prompt, image_b64)
                except TimeoutError as exc:
                    raise VisionModelTimeoutError(
                        f"Vision model timed out on page {page_number} "
                        f"of '{path.name}': {exc}"
                    ) from exc
                except Exception as exc:
                    logger.warning(
                        f"Vision model failed on page {page_number} of '{path.name}': {exc}. "
                        f"Skipping page."
                    )
                    continue

                parsed = parse_vision_response(response)
                if parsed.get("text"):
                    all_text.append(parsed["text"])
                all_figures.extend(parsed.get("figures", []))
                all_tables.extend(parsed.get("tables", []))
                if page_number == 1 and parsed.get("metadata"):
                    metadata = parsed["metadata"]

            return {
                "text": "\n\n".join(all_text),
                "figures": all_figures,
                "tables": all_tables,
                "metadata": metadata,
            }
        finally:
            pdf_doc.close()

    async def extract_batch(
        self,
        pdf_paths: list[str],
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract content from multiple PDFs with progress tracking.

        Args:
            pdf_paths: List of filesystem paths to PDF files.
            progress_callback: Optional async callback ``(processed, total) -> None``.

        Returns:
            List of extraction results (one per PDF). Failed extractions
            include an ``error`` key instead of content.
        """
        total = len(pdf_paths)
        results: list[dict[str, Any]] = []

        for idx, pdf_path in enumerate(pdf_paths):
            try:
                content = await self.extract_content(pdf_path)
                results.append(content)
            except (CorruptedPDFError, VisionModelTimeoutError) as exc:
                logger.error(f"Extraction failed for '{pdf_path}': {exc}")
                results.append({"error": str(exc), "path": pdf_path})
            except Exception as exc:
                logger.error(
                    f"Unexpected error extracting '{pdf_path}': {exc}",
                    exc_info=True,
                )
                results.append({"error": str(exc), "path": pdf_path})

            if progress_callback is not None:
                await progress_callback(idx + 1, total)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_vision_model(self, prompt: str, image_b64: str) -> str:
        """Invoke the vision model with a prompt and base64 image.

        Uses VisionProcessor which sends the actual image to Bedrock's
        vision-capable model (Llama 3.2 Vision).

        Args:
            prompt: Text prompt for extraction.
            image_b64: Base64-encoded page image.

        Returns:
            Raw text response from the model.
        """
        try:
            # VisionProcessor._invoke_vision_model is sync, run in thread
            import asyncio

            response = await asyncio.to_thread(
                self.vision._invoke_vision_model, prompt, image_b64
            )
            return response  # noqa: RET504
        except TimeoutError:
            raise
        except Exception as exc:
            logger.error(f"Vision model invocation failed: {exc}", exc_info=True)
            raise
