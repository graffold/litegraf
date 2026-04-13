"""
Multimodal Document Processor
Handles extraction of text, images, tables, and other content from PDF documents.
Integrates with existing KGPipeline for biomedical knowledge graph construction.
"""

import logging
import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.ingest.ingestor import Chunk, ProcessedDocument
logger = logging.getLogger(__name__)
@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""

    image_id: str
    image_data: bytes  # Raw image bytes
    page_number: int
    position: dict[str, float]  # {x, y, width, height}
    image_type: str = "figure"  # figure, diagram, chart, pathway
    caption: str | None = None


@dataclass
class ExtractedTable:
    """Represents a table extracted from a document."""

    table_id: str
    page_number: int
    headers: list[str]
    rows: list[list[str]]
    position: dict[str, float]  # {x, y, width, height}
    caption: str | None = None


@dataclass
class MultimodalDocument:
    """Enhanced document with multimodal content."""

    doc_id: str
    source_path: str
    text_content: str
    metadata: dict[str, Any]
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    page_count: int = 0


class MultimodalProcessor:
    """
    Base processor for extracting multimodal content from documents.
    Converts to ProcessedDocument format for KGPipeline integration.
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize multimodal processor.

        Args:
            chunk_size: Target size for text chunks (in words)
            chunk_overlap: Overlap between chunks (in words)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized MultimodalProcessor "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def process_pdf(self, pdf_path: str) -> MultimodalDocument:
        """
        Extract all content from a PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            MultimodalDocument with extracted content
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install pymupdf"
            )

        logger.info(f"Processing PDF: {pdf_path}")
        doc_path = Path(pdf_path)
        doc_id = f"pdf_{doc_path.stem}_{hash(pdf_path) % 10000}"

        pdf_doc = fitz.open(pdf_path)

        # Extract text content
        full_text = ""
        images = []
        tables = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]

            # Extract text
            page_text = page.get_text()
            full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"

            # Extract images
            image_list = page.get_images()
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Get image position on page
                    img_rect = page.get_image_rects(xref)
                    position = {}
                    if img_rect:
                        rect = img_rect[0]
                        position = {
                            "x": rect.x0,
                            "y": rect.y0,
                            "width": rect.width,
                            "height": rect.height,
                        }

                    extracted_img = ExtractedImage(
                        image_id=f"{doc_id}_page{page_num + 1}_img{img_index}",
                        image_data=image_bytes,
                        page_number=page_num + 1,
                        position=position,
                        image_type="figure",
                    )
                    images.append(extracted_img)
                    logger.debug(
                        f"Extracted image {img_index} from page {page_num + 1}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_index} "
                        f"from page {page_num + 1}: {e}"
                    )

            # Extract tables (basic detection via text positioning)
            # For production, consider using pdfplumber or camelot-py for better table extraction
            tables_on_page = self._detect_tables_pymupdf(page, page_num + 1, doc_id)
            tables.extend(tables_on_page)

        pdf_doc.close()

        multimodal_doc = MultimodalDocument(
            doc_id=doc_id,
            source_path=pdf_path,
            text_content=full_text,
            metadata={
                "filename": doc_path.name,
                "page_count": len(pdf_doc),
                "image_count": len(images),
                "table_count": len(tables),
            },
            images=images,
            tables=tables,
            page_count=len(pdf_doc),
        )

        logger.info(
            f"Extracted {len(images)} images and {len(tables)} tables "
            f"from {len(pdf_doc)} pages"
        )

        return multimodal_doc

    def _detect_tables_pymupdf(
        self, page, page_num: int, doc_id: str
    ) -> list[ExtractedTable]:
        """
        Basic table detection using PyMuPDF.
        For production, use specialized libraries like pdfplumber or camelot-py.

        Args:
            page: PyMuPDF page object
            page_num: Page number
            doc_id: Document ID

        Returns:
            List of extracted tables
        """
        tables = []

        try:
            # PyMuPDF has basic table detection via find_tables()
            table_finder = page.find_tables()

            for table_index, table in enumerate(table_finder.tables):
                # Extract table data
                table_data = table.extract()

                if table_data and len(table_data) > 0:
                    headers = table_data[0] if table_data else []
                    rows = table_data[1:] if len(table_data) > 1 else []

                    # Get table position
                    bbox = table.bbox
                    position = {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1],
                    }

                    extracted_table = ExtractedTable(
                        table_id=f"{doc_id}_page{page_num}_table{table_index}",
                        page_number=page_num,
                        headers=headers,
                        rows=rows,
                        position=position,
                    )
                    tables.append(extracted_table)
                    logger.debug(
                        f"Extracted table {table_index} from page {page_num} "
                        f"({len(rows)} rows)"
                    )
        except Exception as e:
            logger.warning(f"Table detection failed for page {page_num}: {e}")

        return tables

    def to_processed_document(
        self,
        multimodal_doc: MultimodalDocument,
        include_image_refs: bool = True,
        include_table_refs: bool = True,
    ) -> ProcessedDocument:
        """
        Convert MultimodalDocument to ProcessedDocument for KGPipeline.

        Args:
            multimodal_doc: Source multimodal document
            include_image_refs: Add image references to chunks
            include_table_refs: Add table references to chunks

        Returns:
            ProcessedDocument compatible with existing pipeline
        """
        # Create chunks from text content
        chunks = self._create_chunks(multimodal_doc)

        # Add image references to appropriate chunks
        if include_image_refs:
            self._link_images_to_chunks(multimodal_doc.images, chunks)

        # Add table data to appropriate chunks
        if include_table_refs:
            self._link_tables_to_chunks(multimodal_doc.tables, chunks)

        processed_doc = ProcessedDocument(
            doc_id=multimodal_doc.doc_id,
            source=multimodal_doc.text_content,
            metadata={
                **multimodal_doc.metadata,
                "source_path": multimodal_doc.source_path,
                "has_images": len(multimodal_doc.images) > 0,
                "has_tables": len(multimodal_doc.tables) > 0,
            },
            chunks=chunks,
        )

        logger.info(
            f"Created ProcessedDocument with {len(chunks)} chunks "
            f"from multimodal content"
        )

        return processed_doc

    def _create_chunks(self, multimodal_doc: MultimodalDocument) -> list[Chunk]:
        """
        Create text chunks from document content.
        Uses word-based chunking similar to existing pipeline.

        Args:
            multimodal_doc: Source document

        Returns:
            List of text chunks
        """
        text = multimodal_doc.text_content
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{multimodal_doc.doc_id}_chunk{len(chunks)}"

            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                pmid=multimodal_doc.metadata.get("pmid"),
                title=multimodal_doc.metadata.get("title"),
                publication_year=multimodal_doc.metadata.get("publication_year"),
            )
            chunks.append(chunk)

        return chunks

    def _link_images_to_chunks(self, images: list[ExtractedImage], chunks: list[Chunk]):
        """
        Add image metadata to chunks based on page proximity.

        Args:
            images: Extracted images
            chunks: Text chunks
        """
        for image in images:
            # Store image reference in chunk metadata
            # In a real implementation, you'd determine which chunk(s)
            # are closest to the image based on page numbers and positions
            for chunk in chunks:
                # Simple heuristic: add to first chunk (improve this logic)
                if not hasattr(chunk, "image_refs"):
                    chunk.image_refs = []
                chunk.image_refs.append(
                    {
                        "image_id": image.image_id,
                        "page": image.page_number,
                        "caption": image.caption,
                        "type": image.image_type,
                    }
                )
                break  # Only add to first chunk for now

    def _link_tables_to_chunks(self, tables: list[ExtractedTable], chunks: list[Chunk]):
        """
        Add table data to chunks based on page proximity.

        Args:
            tables: Extracted tables
            chunks: Text chunks
        """
        for table in tables:
            # Store table reference in chunk metadata
            for chunk in chunks:
                if not hasattr(chunk, "table_refs"):
                    chunk.table_refs = []
                chunk.table_refs.append(
                    {
                        "table_id": table.table_id,
                        "page": table.page_number,
                        "caption": table.caption,
                        "row_count": len(table.rows),
                        "headers": table.headers,
                    }
                )
                break  # Only add to first chunk for now

    @staticmethod
    def image_to_base64(image_data: bytes) -> str:
        """
        Convert image bytes to base64 string for LLM consumption.

        Args:
            image_data: Raw image bytes

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(image_data).decode("utf-8")

    @staticmethod
    def resize_image(
        image_data: bytes, max_size: tuple[int, int] = (1024, 1024)
    ) -> bytes:
        """
        Resize image to reduce token consumption in vision models.

        Args:
            image_data: Raw image bytes
            max_size: Maximum (width, height)

        Returns:
            Resized image bytes
        """
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format=img.format or "PNG")
        return output.getvalue()
