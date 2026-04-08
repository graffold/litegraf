"""Orchestrator for bioRxiv full-paper ingestion into the knowledge graph.

Coordinates fetching, content extraction, section parsing, deduplication,
and handoff to the existing KG pipeline, embedding pipeline, relationship
consolidation, and UniProt mapping.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pipeline.ingest.biorxiv_deduplicator import BioRxivDeduplicator
from pipeline.ingest.biorxiv_fetcher import BioRxivFetcher, BioRxivPaperMetadata
from pipeline.ingest.content_extractor import ContentExtractor
from pipeline.ingest.embedding_pipeline import EmbeddingPipeline
from pipeline.ingest.ingestor import ProcessedDocument
from pipeline.ingest.kg_pipeline import KGPipeline
from pipeline.ingest.section_parser import PaperSection, SectionParser
from pipeline.processors.relationship_counter import RelationshipCounter
from src.core.database import Neo4jDatabase
from src.utils.logging_utils import setup_logging
from src.utils.map_proteins_to_uniprot import ProteinUniProtMapper

logger = setup_logging(name=__name__)

# Stage weights for progress calculation
STAGE_WEIGHTS = {
    "fetching": 0.10,  # 10% - Fetch papers from API
    "deduplication": 0.05,  # 5%  - Filter duplicates
    "extraction": 0.20,  # 20% - Extract content from PDFs
    "kg_pipeline": 0.40,  # 40% - Entity/relationship extraction
    "embedding": 0.15,  # 15% - Generate embeddings
    "consolidation": 0.05,  # 5%  - Consolidate relationships
    "uniprot": 0.05,  # 5%  - Map to UniProt
}


def _sanitize_doi(doi: str) -> str:
    """Replace ``/`` and ``.`` in a DOI with ``_`` for use in identifiers."""
    return doi.replace("/", "_").replace(".", "_")


class BioRxivIngestor:
    """Orchestrates bioRxiv paper ingestion into the knowledge graph."""

    def __init__(
        self,
        service: str = "local",
        database: str = "olink3",
        backend: str = "neo4j",
        enable_consolidation: bool = False,
        rate_limit_delay: float = 1.0,
        sections_filter: list[str] | None = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        skip_node_labeling: bool = False,
    ) -> None:
        self.service = service
        self.database = database
        self.backend = backend
        self.enable_consolidation = enable_consolidation
        self.sections_filter = sections_filter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # --- sub-components ---
        self.fetcher = BioRxivFetcher(rate_limit_delay=rate_limit_delay)
        self.extractor = ContentExtractor()
        self.parser = SectionParser()

        # Database connection for deduplicator (needs _execute_cypher)
        use_opencypher = backend == "neptune"
        if backend == "neo4j":
            self.db = Neo4jDatabase(database=database)
        else:
            self.db = None  # Neptune handled internally by KGPipeline

        self.deduplicator = BioRxivDeduplicator(db=self.db, backend=backend)

        self.kg_pipeline = KGPipeline(
            service=service,
            database=database,
            use_opencypher=use_opencypher,
            backend=backend,
            enable_consolidation=enable_consolidation,
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
            skip_node_labeling=skip_node_labeling,
        )

        self.embedding_pipeline = EmbeddingPipeline(
            service=service,
            database=database,
            use_opencypher=use_opencypher,
            backend=backend,
        )

        self.relationship_counter = RelationshipCounter(
            database=database,
        )

        self.uniprot_mapper = ProteinUniProtMapper(database=database)

        logger.info(
            f"Initialized BioRxivIngestor (service={service}, database={database}, "
            f"backend={backend}, consolidation={enable_consolidation})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str | None = None,
        dois: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        subject: str | None = None,
        max_results: int = 100,
        force: bool = False,
        progress_callback: Callable[[int, int, str], Awaitable[None]] | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> list[ProcessedDocument]:
        """Run the full bioRxiv ingestion pipeline.

        Args:
            query: Search query for bioRxiv papers
            dois: List of specific DOIs to fetch
            date_from: Start date for paper search (YYYY-MM-DD)
            date_to: End date for paper search (YYYY-MM-DD)
            subject: Subject category filter
            max_results: Maximum number of papers to fetch
            force: Force reprocessing of existing papers
            progress_callback: Optional callback for progress reporting.
                Called with (processed_count, total_count, stage_name).
            cancellation_event: Optional event that signals job should stop.
                Checked before processing each paper.

        Returns:
            List of successfully processed documents.
        """
        stats: dict[str, int] = {
            "fetched": 0,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
        }

        completed_stages: set[str] = set()

        # Helper to calculate and report progress
        async def report_progress(
            stage: str, stage_progress: float, processed: int, total: int
        ) -> None:
            """Calculate overall progress and call the callback if provided."""
            if progress_callback:
                await progress_callback(processed, total, stage)

        # Helper to check cancellation
        def check_cancellation() -> bool:
            """Check if cancellation has been requested."""
            if cancellation_event and cancellation_event.is_set():
                logger.info("Cancellation requested, stopping pipeline")
                return True
            return False

        # ---- 1. Fetch papers ----
        logger.info("Stage 1/10: Fetching papers from bioRxiv")
        await report_progress("fetching", 0.0, 0, max_results)

        if check_cancellation():
            return []

        papers = await self._fetch_papers(
            query, dois, date_from, date_to, subject, max_results
        )
        stats["fetched"] = len(papers)
        logger.info(f"Fetched {stats['fetched']} papers from bioRxiv")

        await report_progress("fetching", 1.0, stats["fetched"], stats["fetched"])
        completed_stages.add("fetching")

        if not papers:
            self._log_summary(stats)
            return []

        if check_cancellation():
            return []

        # ---- 2. Deduplicate ----
        logger.info("Stage 2/10: Filtering duplicate papers")
        await report_progress("deduplication", 0.0, 0, stats["fetched"])

        new_papers = self.deduplicator.filter_new_papers(papers, force=force)
        stats["skipped"] = stats["fetched"] - len(new_papers)
        logger.info(
            f"{len(new_papers)} new papers after deduplication "
            f"({stats['skipped']} skipped)"
        )

        await report_progress("deduplication", 1.0, len(new_papers), stats["fetched"])
        completed_stages.add("deduplication")

        if not new_papers:
            self._log_summary(stats)
            return []

        if check_cancellation():
            return []

        # ---- 3. Extract, parse, build documents ----
        logger.info("Stage 3/10: Extracting content and parsing sections")
        documents: list[ProcessedDocument] = []
        processed_dois: list[str] = []
        total_papers = len(new_papers)

        for idx, paper in enumerate(new_papers):
            # Check cancellation before processing each paper
            if check_cancellation():
                logger.info(
                    f"Cancelled during extraction after processing {idx} papers"
                )
                break

            try:
                await report_progress(
                    "extraction", idx / total_papers, idx, total_papers
                )

                text = await self.extractor.extract_from_url(paper.content_url)
                if not text:
                    logger.warning(f"Empty extraction for DOI {paper.doi}, skipping")
                    stats["failed"] += 1
                    continue

                sections = self.parser.parse(text)

                # Apply section filter if configured
                if self.sections_filter:
                    sections = SectionParser.filter_sections(
                        sections, self.sections_filter
                    )
                    if not sections:
                        logger.warning(
                            f"No sections matched filter for DOI {paper.doi}, skipping"
                        )
                        stats["failed"] += 1
                        continue

                doc = self._build_processed_document(paper, sections)
                documents.append(doc)
                processed_dois.append(paper.doi)
                stats["processed"] += 1
                logger.debug(f"Built ProcessedDocument for DOI {paper.doi}")
            except Exception as e:
                logger.error(f"Failed to process paper {paper.doi}: {e}")
                stats["failed"] += 1
                continue

        await report_progress("extraction", 1.0, len(documents), total_papers)
        completed_stages.add("extraction")

        logger.info(
            f"Built {len(documents)} ProcessedDocuments "
            f"({stats['failed']} failed during extraction/parsing)"
        )

        if not documents:
            self._log_summary(stats)
            return []

        if check_cancellation():
            return []

        # ---- 4. KG Pipeline (chunking + entity/relationship extraction) ----
        logger.info("Stage 4/10: Processing documents through KG pipeline")
        await report_progress("kg_pipeline", 0.0, 0, len(documents))

        try:
            documents = await self.kg_pipeline.process_documents_enhanced(
                documents, cleanup_existing=False
            )
            await report_progress("kg_pipeline", 1.0, len(documents), len(documents))
            completed_stages.add("kg_pipeline")
        except Exception as e:
            logger.error(f"KG pipeline processing failed: {e}")
            self._log_summary(stats)
            return []

        # Count chunks and warn for large papers
        entity_counts: dict[str, int] = {}
        total_relationships = 0
        for doc in documents:
            chunk_count = len(doc.chunks)
            stats["total_chunks"] += chunk_count
            if chunk_count > 50:
                logger.warning(
                    f"Paper {doc.doc_id} produced {chunk_count} chunks (>50)"
                )
            # Count entities by type across all chunks
            for chunk in doc.chunks:
                for node in chunk.nodes:
                    node_type = node.get("type", node.get("label", "unknown")).lower()
                    entity_counts[node_type] = entity_counts.get(node_type, 0) + 1
                total_relationships += len(chunk.relationships)

        logger.info(f"KG pipeline produced {stats['total_chunks']} total chunks")
        logger.info(
            f"Entity counts: {entity_counts}, Relationships: {total_relationships}"
        )

        # Report entity counts via progress callback
        if progress_callback:
            # We can't pass details through the standard callback signature,
            # but we report the counts in the log for now
            pass

        if check_cancellation():
            return []

        # ---- 5. Generate embeddings ----
        logger.info("Stage 5/10: Generating embeddings")
        await report_progress("embedding", 0.0, 0, len(documents))

        try:
            documents = await self.embedding_pipeline.process_documents_chunks_only(
                documents
            )
            await report_progress("embedding", 1.0, len(documents), len(documents))
            completed_stages.add("embedding")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")

        if check_cancellation():
            return []

        # ---- 6. Relationship consolidation ----
        logger.info("Stage 6/10: Consolidating relationships")
        await report_progress("consolidation", 0.0, 0, 1)

        try:
            self.relationship_counter.consolidate_duplicate_relationships()
            await report_progress("consolidation", 1.0, 1, 1)
            completed_stages.add("consolidation")
        except Exception as e:
            logger.error(f"Relationship consolidation failed: {e}")

        if check_cancellation():
            return []

        # ---- 7. UniProt mapping ----
        logger.info("Stage 7/10: Mapping proteins to UniProt IDs")
        await report_progress("uniprot", 0.0, 0, 1)

        try:
            if self.kg_pipeline.skip_node_labeling:
                logger.info("Skipping UniProt mapping (skip_node_labeling)")
            else:
                self.uniprot_mapper.run_mapping()
            await report_progress("uniprot", 1.0, 1, 1)
            completed_stages.add("uniprot")
        except Exception as e:
            logger.error(f"UniProt mapping failed: {e}")

        if check_cancellation():
            return []

        # ---- 8. Link published counterparts ----
        logger.info("Stage 8/10: Linking published counterparts")
        try:
            links = self.deduplicator.link_published_counterparts(processed_dois)
            logger.info(f"Created {links} PUBLISHED_AS links")
        except Exception as e:
            logger.error(f"Published counterpart linking failed: {e}")

        # ---- 9-10. Summary ----
        self._log_summary(stats)

        return documents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_processed_document(
        self,
        metadata: BioRxivPaperMetadata,
        sections: list[PaperSection],
    ) -> ProcessedDocument:
        """Convert fetched paper metadata and parsed sections into a ProcessedDocument."""
        doi_sanitized = _sanitize_doi(metadata.doi)
        doc_id = f"biorxiv_{doi_sanitized}"

        # Concatenate section texts as the document source
        source_text = "\n\n".join(section.text for section in sections)

        sections_processed = [s.label for s in sections]

        doc_metadata: dict[str, Any] = {
            "doi": metadata.doi,
            "title": metadata.title,
            "authors": metadata.authors,
            "posting_date": metadata.posting_date,
            "content_url": metadata.content_url,
            "source_type": "biorxiv",
            "sections_processed": sections_processed,
            "doc_id_prefix": doc_id,
        }

        return ProcessedDocument(
            doc_id=doc_id,
            source=source_text,
            metadata=doc_metadata,
        )

    async def _fetch_papers(
        self,
        query: str | None,
        dois: list[str] | None,
        date_from: str | None,
        date_to: str | None,
        subject: str | None,
        max_results: int,
    ) -> list[BioRxivPaperMetadata]:
        """Fetch papers using the appropriate BioRxivFetcher method."""
        if dois:
            return await self.fetcher.fetch_by_dois(dois)
        if date_from and date_to:
            return await self.fetcher.fetch_by_date_range(
                date_from, date_to, subject=subject, max_results=max_results
            )
        if query:
            return await self.fetcher.search(query, max_results=max_results)
        logger.warning("No search parameters provided for bioRxiv fetch")
        return []

    def _log_summary(self, stats: dict[str, int]) -> None:
        """Log pipeline summary statistics and verify the counting invariant."""
        fetched = stats["fetched"]
        processed = stats["processed"]
        failed = stats["failed"]
        skipped = stats["skipped"]
        total_chunks = stats["total_chunks"]

        logger.info("=" * 60)
        logger.info("bioRxiv Ingestion Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"  Total fetched:    {fetched}")
        logger.info(f"  Processed:        {processed}")
        logger.info(f"  Failed:           {failed}")
        logger.info(f"  Skipped (dupes):  {skipped}")
        logger.info(f"  Total chunks:     {total_chunks}")
        logger.info("=" * 60)

        # Verify invariant: fetched == processed + failed + skipped
        if fetched != processed + failed + skipped:
            logger.warning(
                f"Summary invariant violated: fetched ({fetched}) != "
                f"processed ({processed}) + failed ({failed}) + skipped ({skipped})"
            )

    def close(self) -> None:
        """Clean up resources."""
        try:
            self.kg_pipeline.close()
        except Exception:
            pass
        try:
            self.embedding_pipeline.close()
        except Exception:
            pass
        try:
            self.relationship_counter.close()
        except Exception:
            pass
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
        logger.info("Closed BioRxivIngestor resources")
