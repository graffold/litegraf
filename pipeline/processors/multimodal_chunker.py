"""Multimodal chunking strategy for PMC full-text articles.

Preserves text-figure-table relationships while respecting section boundaries
and optimizing chunk size for LLM processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pipeline.ingest.ingestor import Chunk, ProcessedDocument
from pipeline.ingest.token_chunker import TokenChunker
logger = logging.getLogger(__name__)
@dataclass
class MultimodalChunk:
    """A chunk with linked figures and tables."""

    chunk_id: str
    text: str
    token_count: int
    section_title: str | None = None
    figure_refs: list[dict] = field(default_factory=list)
    table_refs: list[dict] = field(default_factory=list)


class MultimodalChunker(TokenChunker):
    """Token-based chunker that preserves text-figure-table relationships.

    Extends TokenChunker to:
    1. Respect section boundaries (avoid splitting mid-section when possible)
    2. Link figures/tables to chunks based on proximity
    3. Maintain context across modalities
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        model_name: str = "default",
        respect_sections: bool = True,
    ):
        super().__init__(max_tokens, overlap_tokens, model_name)
        self.respect_sections = respect_sections

    def chunk_fulltext(
        self,
        doc_id: str,
        sections: list[dict],
        figures: list[dict],
        tables: list[dict],
    ) -> list[MultimodalChunk]:
        """Chunk full-text article with multimodal content.

        Args:
            doc_id: Document identifier
            sections: List of {"title": str, "content": str}
            figures: List of figure metadata from FullTextProcessor
            tables: List of table metadata from FullTextProcessor

        Returns:
            List of multimodal chunks with linked figures/tables
        """
        if not sections:
            return []

        chunks: list[MultimodalChunk] = []

        if self.respect_sections:
            # Chunk each section separately to preserve boundaries
            for section in sections:
                section_chunks = self._chunk_section(doc_id, section, len(chunks))
                chunks.extend(section_chunks)
        else:
            # Chunk entire document as single text
            full_text = "\n\n".join(
                f"## {s['title']}\n\n{s['content']}" for s in sections
            )
            token_chunks = self.chunk_text(full_text, doc_id)
            chunks = [
                MultimodalChunk(
                    chunk_id=tc.chunk_id,
                    text=tc.text,
                    token_count=tc.token_count,
                )
                for tc in token_chunks
            ]

        # Link figures and tables to chunks
        self._link_figures(chunks, figures)
        self._link_tables(chunks, tables)

        logger.info(
            f"Created {len(chunks)} multimodal chunks "
            f"({len(figures)} figures, {len(tables)} tables)"
        )

        return chunks

    def _chunk_section(
        self, doc_id: str, section: dict, chunk_offset: int
    ) -> list[MultimodalChunk]:
        """Chunk a single section, preserving section title."""
        section_title = section["title"]
        section_content = section["content"]

        # Add section title as context
        text_with_title = f"## {section_title}\n\n{section_content}"

        token_chunks = self.chunk_text(text_with_title, doc_id)

        return [
            MultimodalChunk(
                chunk_id=f"{doc_id}_{chunk_offset + i + 1}",
                text=tc.text,
                token_count=tc.token_count,
                section_title=section_title,
            )
            for i, tc in enumerate(token_chunks)
        ]

    def _link_figures(self, chunks: list[MultimodalChunk], figures: list[dict]) -> None:
        """Link figures to chunks based on text proximity.

        Strategy: Search for figure references (e.g., "Figure 1", "Fig. 2")
        in chunk text and link accordingly. If no reference found, link to
        first chunk.
        """
        for figure in figures:
            label = figure.get("label", "")
            if not label:
                continue

            figure_ref = {
                "id": figure.get("id", ""),
                "label": label,
                "title": figure.get("title", ""),
                "caption": figure.get("caption", ""),
                "graphics": figure.get("graphics", []),
            }

            # Find chunks that mention this figure
            linked = False
            for chunk in chunks:
                if label.lower() in chunk.text.lower():
                    chunk.figure_refs.append(figure_ref)
                    linked = True

            if not linked and chunks:
                # No explicit reference found - link to first chunk
                # (figures often appear at document start)
                chunks[0].figure_refs.append(figure_ref)

    def _link_tables(self, chunks: list[MultimodalChunk], tables: list[dict]) -> None:
        """Link tables to chunks based on text proximity.

        Strategy: Search for table references (e.g., "Table 1", "Table 2")
        in chunk text and link accordingly. If no reference found, link to
        first chunk.
        """
        for table in tables:
            label = table.get("label", "")
            if not label:
                continue

            table_ref = {
                "id": table.get("id", ""),
                "label": label,
                "title": table.get("title", ""),
                "caption": table.get("caption", ""),
                "structured_content": table.get("structured_content", []),
                "graphic": table.get("graphic"),
            }

            # Find chunks that mention this table
            linked = False
            for chunk in chunks:
                if label.lower() in chunk.text.lower():
                    chunk.table_refs.append(table_ref)
                    linked = True

            if not linked and chunks:
                # No explicit reference found - link to first chunk
                chunks[0].table_refs.append(table_ref)

    def to_processed_document(
        self,
        doc_id: str,
        fulltext_data: dict,
        multimodal_chunks: list[MultimodalChunk],
    ) -> ProcessedDocument:
        """Convert multimodal chunks to ProcessedDocument for pipeline.

        Args:
            doc_id: Document identifier
            fulltext_data: Full-text data from FullTextProcessor
            multimodal_chunks: Chunked content with linked figures/tables

        Returns:
            ProcessedDocument compatible with KGPipeline
        """
        # Convert multimodal chunks to standard Chunk objects
        chunks = [
            Chunk(
                chunk_id=mc.chunk_id,
                text=mc.text,
                pmid=fulltext_data.get("pmid"),
                title=fulltext_data.get("title"),
                publication_year=None,  # Extract from publication_date if needed
            )
            for mc in multimodal_chunks
        ]

        # Store multimodal metadata in document metadata
        metadata = {
            "pmc_id": fulltext_data.get("pmc_id"),
            "pmid": fulltext_data.get("pmid"),
            "title": fulltext_data.get("title"),
            "authors": fulltext_data.get("authors", []),
            "journal": fulltext_data.get("journal"),
            "publication_date": fulltext_data.get("publication_date"),
            "doi": fulltext_data.get("doi"),
            "keywords": fulltext_data.get("keywords", []),
            "figure_count": len(fulltext_data.get("figures", [])),
            "table_count": len(fulltext_data.get("tables", [])),
            "section_count": len(fulltext_data.get("sections", [])),
            "has_multimodal_content": (
                len(fulltext_data.get("figures", [])) > 0
                or len(fulltext_data.get("tables", [])) > 0
            ),
        }

        # Reconstruct source text from sections
        source_text = "\n\n".join(
            f"## {s['title']}\n\n{s['content']}"
            for s in fulltext_data.get("sections", [])
        )

        return ProcessedDocument(
            doc_id=doc_id,
            source=source_text,
            metadata=metadata,
            chunks=chunks,
        )
