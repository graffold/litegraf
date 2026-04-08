#!/usr/bin/env python3
"""
Incremental consolidation strategy for running entity resolution during database population.
"""

import time
from typing import Any

from pipeline.processors.entity_resolver import EntityResolver
from src.core.database import Neo4jDatabase
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class IncrementalConsolidator:
    """
    Manages incremental consolidation during database population.
    Balances performance with data quality.
    """

    def __init__(self, db: Neo4jDatabase | None = None, chunk_threshold: int = 50, skip_full_resolution: bool = False):
        """
        Initialize incremental consolidator.

        Args:
            db: Neo4j database connection
            chunk_threshold: Run consolidation every N chunks
            skip_full_resolution: Skip full entity resolution (UniProt mapping, etc.) in final consolidation
        """
        self.db = db or Neo4jDatabase(database="cvd1")
        self.chunk_threshold = chunk_threshold
        self.skip_full_resolution = skip_full_resolution
        self.chunks_processed = 0
        self.last_consolidation = time.time()
        self.consolidation_interval = 300  # 5 minutes minimum between runs

    def should_consolidate(self) -> bool:
        """Determine if consolidation should run now."""
        current_time = time.time()

        # Check chunk threshold
        chunk_ready = self.chunks_processed >= self.chunk_threshold

        # Check time threshold
        time_ready = (
            current_time - self.last_consolidation
        ) >= self.consolidation_interval

        return chunk_ready and time_ready

    def on_chunk_processed(self, chunk_id: str) -> dict[str, Any]:
        """
        Called after each chunk is processed.

        Args:
            chunk_id: ID of the processed chunk

        Returns:
            Consolidation statistics if run, empty dict otherwise
        """
        self.chunks_processed += 1

        if self.should_consolidate():
            logger.info(
                f"Running incremental consolidation after {self.chunks_processed} chunks"
            )
            return self._run_consolidation()

        return {}

    def _run_consolidation(self) -> dict[str, Any]:
        """Run lightweight consolidation."""
        try:
            resolver = EntityResolver(db=self.db)

            # Run only name-based consolidation (faster)
            stats = resolver.consolidate_entities_by_name(dry_run=False)

            # Reset counters
            self.chunks_processed = 0
            self.last_consolidation = time.time()

            logger.info(f"Incremental consolidation completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Incremental consolidation failed: {e}")
            return {"error": str(e)}

    def final_consolidation(self) -> dict[str, Any]:
        """Run complete consolidation at the end."""
        logger.info("Running final comprehensive consolidation")

        try:
            resolver = EntityResolver(db=self.db)

            if self.skip_full_resolution:
                # Only run name-based consolidation (skip UniProt mapping + full resolution)
                logger.info("Skipping full entity resolution (skip_node_labeling mode)")
                stats = resolver.consolidate_entities_by_name(dry_run=False)
            else:
                # Run full consolidation including UniProt mapping and hierarchy
                stats = resolver.run_full_entity_resolution(dry_run=False)

            logger.info(f"Final consolidation completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Final consolidation failed: {e}")
            return {"error": str(e)}


def integrate_with_ingestion_pipeline():
    """
    Example of how to integrate with the ingestion pipeline.
    """
    db = Neo4jDatabase(database="cvd1")
    consolidator = IncrementalConsolidator(db=db, chunk_threshold=25)

    # In your ingestion loop:
    for chunk_id in ["chunk_1", "chunk_2", "..."]:
        # ... process chunk ...

        # After chunk processing:
        consolidation_stats = consolidator.on_chunk_processed(chunk_id)
        if consolidation_stats:
            print(f"Incremental consolidation: {consolidation_stats}")

    # At the end of ingestion:
    final_stats = consolidator.final_consolidation()
    print(f"Final consolidation: {final_stats}")


if __name__ == "__main__":
    integrate_with_ingestion_pipeline()
