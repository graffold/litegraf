"""PubMed Central (PMC) full-text ingestor.

Orchestrates fetching open-access full-text articles from PMC and
processing them through the KG pipeline, following the same pattern
as BioRxivIngestor.
"""

from __future__ import annotations

import logging
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pipeline.interfaces import GraphStore
from pipeline.ingest.embedding_pipeline import EmbeddingPipeline
from pipeline.ingest.ingestor import ProcessedDocument
from pipeline.ingest.kg_pipeline import KGPipeline
from pipeline.ingest.pmc_fetcher import PMCArticle, PMCFetcher
from pipeline.processors.relationship_counter import RelationshipCounter
try:
    from pipeline.utils.uniprot_mapper import ProteinUniProtMapper
except ImportError:
    ProteinUniProtMapper = None

logger = logging.getLogger(__name__)
class PMCIngestor:
    """Orchestrates PMC full-text ingestion into the knowledge graph.

    Uses a batch-streaming pipeline: articles are fetched and processed
    in batches of ``pipeline_batch_size`` so that KG extraction starts
    as soon as the first batch is ready instead of waiting for all
    articles to download.
    """

    def __init__(
        self,
        service: str = "local",
        database: str = "neo4j",
        backend: str = "neo4j",
        enable_consolidation: bool = False,
        rate_limit_delay: float = 0.4,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        pipeline_batch_size: int = 50,
        skip_node_labeling: bool = False,
    ) -> None:
        self.service = service
        self.database = database
        self.backend = backend
        self.enable_consolidation = enable_consolidation
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pipeline_batch_size = pipeline_batch_size

        self.fetcher = PMCFetcher(rate_limit_delay=rate_limit_delay)

        use_opencypher = backend == "neptune"
        self.db: GraphStore | None = None
        if backend == "neo4j":
            from pipeline.backends.neo4j_store import Neo4jGraphStore
            self.db = Neo4jGraphStore(database=database)

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

        self.relationship_counter = RelationshipCounter(database=database)
        self.uniprot_mapper = ProteinUniProtMapper(database=database) if ProteinUniProtMapper is not None else None

        logger.info(
            f"Initialized PMCIngestor (service={service}, database={database}, "
            f"backend={backend}, consolidation={enable_consolidation})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str | None = None,
        pmcids: list[str] | None = None,
        max_results: int = 50,
        force: bool = False,
        progress_callback: Callable[[int, int, str], Awaitable[None]] | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> list[ProcessedDocument]:
        """Run the PMC ingestion pipeline with batch-streaming.

        Instead of fetching all articles then processing all at once,
        this fetches and processes in batches of ``pipeline_batch_size``
        so KG extraction starts as soon as the first batch is ready.

        Args:
            query: Search query for PMC articles.
            pmcids: Explicit list of PMC IDs to fetch.
            max_results: Maximum articles to fetch when using query search.
            force: Force reprocessing of existing articles.
            progress_callback: Called with (processed, total, stage).
            cancellation_event: Signals the job should stop.

        Returns:
            List of all successfully processed documents across all batches.
        """
        stats: dict[str, int] = {
            "fetched": 0,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
        }
        all_docs: list[ProcessedDocument] = []

        async def report(stage: str, processed: int, total: int) -> None:
            if progress_callback:
                await progress_callback(processed, total, stage)

        def cancelled() -> bool:
            if cancellation_event and cancellation_event.is_set():
                logger.info("Cancellation requested, stopping PMC pipeline")
                return True
            return False

        # ---- 1. Search for PMC IDs ----
        logger.info("Stage 1: Searching PMC for article IDs")
        await report("fetching", 0, max_results)

        if cancelled():
            return []

        if pmcids:
            # Strip PMC prefix for efetch compatibility
            numeric_ids = [p.replace("PMC", "") for p in pmcids]
        elif query:
            numeric_ids = await self.fetcher.search(query, max_results)
        else:
            logger.error("Either query or pmcids must be provided")
            return []

        if not numeric_ids:
            logger.info("No PMC articles found")
            self._log_summary(stats)
            return []

        logger.info(f"Found {len(numeric_ids)} PMC IDs")

        # ---- 2. Deduplicate against existing PMCIDs ----
        if not force and self.db:
            existing = self._get_existing_pmcids()
            # Compare with PMC prefix since that's what we store
            before = len(numeric_ids)
            numeric_ids = [
                nid
                for nid in numeric_ids
                if f"PMC{nid}" not in existing and nid not in existing
            ]
            stats["skipped"] = before - len(numeric_ids)
            logger.info(
                f"{len(numeric_ids)} new articles after dedup ({stats['skipped']} skipped)"
            )

        if not numeric_ids:
            self._log_summary(stats)
            return []

        total_ids = len(numeric_ids)
        await report("fetching", 0, total_ids)

        # ---- 3. Batch-streaming: fetch → build docs → KG → embeddings per batch ----
        batch_size = self.pipeline_batch_size
        total_batches = (total_ids + batch_size - 1) // batch_size

        for batch_idx in range(0, total_ids, batch_size):
            if cancelled():
                break

            batch_ids = numeric_ids[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            logger.info(
                f"\n{'=' * 60}\n"
                f"Batch {batch_num}/{total_batches}: "
                f"fetching & processing {len(batch_ids)} articles "
                f"(total progress: {len(all_docs)}/{total_ids})\n"
                f"{'=' * 60}"
            )

            # 3a. Fetch this batch
            await report("fetching", batch_idx, total_ids)
            articles = await self.fetcher.fetch_batch(batch_ids)
            stats["fetched"] += len(articles)
            logger.info(
                f"Batch {batch_num}: fetched {len(articles)}/{len(batch_ids)} articles"
            )

            if not articles:
                continue

            # 3b. Build ProcessedDocuments
            documents: list[ProcessedDocument] = []
            for article in articles:
                try:
                    doc = self._build_processed_document(article)
                    if doc.source.strip():
                        documents.append(doc)
                        stats["processed"] += 1
                    else:
                        logger.warning(f"Empty text for {article.pmcid}, skipping")
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Failed to build doc for {article.pmcid}: {e}")
                    stats["failed"] += 1

            if not documents:
                continue

            if cancelled():
                break

            # 3c. KG Pipeline
            logger.info(
                f"Batch {batch_num}: processing {len(documents)} docs through KG pipeline"
            )
            await report("kg_pipeline", len(all_docs), total_ids)

            try:
                documents = await self.kg_pipeline.process_documents_enhanced(
                    documents, cleanup_existing=False
                )
            except Exception as e:
                logger.error(f"Batch {batch_num} KG pipeline failed: {e}")
                continue

            for doc in documents:
                stats["total_chunks"] += len(doc.chunks)

            if cancelled():
                break

            # 3d. Embeddings
            logger.info(f"Batch {batch_num}: generating embeddings")
            await report("embedding", len(all_docs), total_ids)

            try:
                documents = await self.embedding_pipeline.process_documents_chunks_only(
                    documents
                )
            except Exception as e:
                logger.error(f"Batch {batch_num} embedding failed: {e}")

            all_docs.extend(documents)
            logger.info(
                f"Batch {batch_num} complete: {len(documents)} docs, "
                f"running total: {len(all_docs)}/{total_ids}"
            )

        # ---- 4. Post-processing (once, after all batches) ----
        if all_docs:
            logger.info("Post-processing: consolidation & UniProt mapping")
            await report("consolidation", 0, 1)

            try:
                self.relationship_counter.consolidate_duplicate_relationships()
            except Exception as e:
                logger.error(f"Relationship consolidation failed: {e}")

            try:
                if self.kg_pipeline.skip_node_labeling:
                    logger.info("Skipping UniProt mapping (skip_node_labeling)")
                elif self.uniprot_mapper is not None:
                    self.uniprot_mapper.run_mapping()
                else:
                    logger.info("Skipping UniProt mapping (mapper not available)")
            except Exception as e:
                logger.error(f"UniProt mapping failed: {e}")

            await report("consolidation", 1, 1)

        self._log_summary(stats)
        return all_docs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Section headings (lowercased) that rarely contain extractable
    # protein-disease relationships.  Skipping them saves LLM calls.
    _SKIP_HEADINGS: set[str] = {
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
        "funding",
        "competing interests",
        "conflict of interest",
        "conflicts of interest",
        "author contributions",
        "authors' contributions",
        "data availability",
        "data availability statement",
        "supplementary material",
        "supplementary information",
        "supplementary data",
        "supporting information",
        "abbreviations",
        "ethics",
        "ethics statement",
        "ethics approval",
        "consent",
        "informed consent",
        "declarations",
        "additional information",
        "appendix",
    }

    def _build_processed_document(self, article: PMCArticle) -> ProcessedDocument:
        doc_id = f"pmc_{article.pmcid}"

        # Filter out low-value sections before assembling text
        kept_sections = [
            s
            for s in article.sections
            if s.get("heading", "").strip().lower() not in self._SKIP_HEADINGS
        ]
        skipped = len(article.sections) - len(kept_sections)
        if skipped:
            logger.debug(
                f"{article.pmcid}: skipped {skipped}/{len(article.sections)} "
                f"low-value sections"
            )

        source_text = "\n\n".join(s["text"] for s in kept_sections if s.get("text"))
        if article.abstract and article.abstract not in source_text:
            source_text = article.abstract + "\n\n" + source_text

        metadata: dict[str, Any] = {
            "pmcid": article.pmcid,
            "pmid": article.pmid,
            "title": article.title,
            "authors": article.authors,
            "journal": article.journal,
            "year": article.year,
            "source_type": "pmc",
            "sections_processed": [s.get("heading", "") for s in kept_sections],
            "doc_id_prefix": doc_id,
        }

        return ProcessedDocument(doc_id=doc_id, source=source_text, metadata=metadata)

    def _get_existing_pmcids(self) -> set[str]:
        """Query Neo4j for existing PMC IDs to avoid re-ingestion."""
        if not self.db:
            return set()
        try:
            result = self.db.execute_query(
                "MATCH (d) WHERE d.pmcid IS NOT NULL RETURN COLLECT(d.pmcid) AS ids"
            )
            if result and result[0]:
                return set(result[0].get("ids", []))
        except Exception as e:
            logger.warning(f"Could not fetch existing PMCIDs: {e}")
        return set()

    def _log_summary(self, stats: dict[str, int]) -> None:
        logger.info(
            "PMC ingestion summary: "
            f"fetched={stats['fetched']}, processed={stats['processed']}, "
            f"failed={stats['failed']}, skipped={stats['skipped']}, "
            f"chunks={stats['total_chunks']}"
        )

    def close(self) -> None:
        if self.db:
            self.db.close()
