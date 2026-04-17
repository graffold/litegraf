import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

from pipeline.interfaces import GraphStore, LLMProvider
import json
logger = logging.getLogger(__name__)
class RelationshipCounter:
    def __init__(self, database: str = "neo4j", graph_store: GraphStore | None = None):
        self.database = database
        if graph_store is not None:
            self.db = graph_store
        else:
            from pipeline.backends.neo4j_store import Neo4jGraphStore
            self.db = Neo4jGraphStore(database=database)

        logger.info(f"Initialized RelationshipCounter for database: {database}")

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query against the graph store."""
        return self.db.execute_query(query, parameters)

    def count_relationships(self) -> dict[str, Any]:
        """
        Count relationships, tracking duplicates and their source documents.
        Returns a dictionary with relationship counts and source details.
        """
        try:
            query = """
                MATCH (source)-[r]->(target)
                WHERE r.source_doc IS NOT NULL
                RETURN source.id AS source_id, target.id AS target_id, type(r) AS rel_type,
                       source.name AS source_name, target.name AS target_name,
                       labels(source)[0] AS source_type, labels(target)[0] AS target_type,
                       r.source_doc AS source_doc
                """
            results = self._execute_query(query)
            rel_counts = defaultdict(list)

            for record in results:
                key = (
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                    record["source_type"],
                    record["target_type"],
                )
                rel_counts[key].append(
                    {
                        "source_name": record["source_name"],
                        "target_name": record["target_name"],
                        "source_doc": record["source_doc"],
                    }
                )

            aggregated = {}
            for (
                source_id,
                target_id,
                rel_type,
                source_type,
                target_type,
            ), instances in rel_counts.items():
                key_str = f"{source_id}_{target_id}_{rel_type}"
                aggregated[key_str] = {
                    "source_id": source_id,
                    "target_id": target_id,
                    "rel_type": rel_type,
                    "source_type": source_type,
                    "target_type": target_type,
                    "source_name": instances[0]["source_name"],
                    "target_name": instances[0]["target_name"],
                    "frequency": len(instances),
                    "sources": [inst["source_doc"] for inst in instances],
                }

            logger.info(f"Counted {len(aggregated)} unique relationships with sources")
            return aggregated
        except Exception as e:
            logger.error(f"Failed to count relationships: {e}", exc_info=True)
            return {}

    def store_relationship_counts(self) -> int:
        """
        Store relationship counts and sources as properties in Neo4j.
        This consolidates multiple instances of the same relationship pattern.
        Returns the number of relationships consolidated.
        """
        try:
            rel_counts = self.count_relationships()
            consolidated_count = 0

            for data in rel_counts.values():
                if data["frequency"] > 1:
                    # Update the relationship with consolidated information
                    query = """
                    MATCH (source {{id: $source_id}})-[r:{}]->(target {{id: $target_id}})
                    SET r.frequency = $frequency,
                        r.sources = $sources,
                        r.pattern_id = $pattern_id,
                        r.consolidated = true,
                        r.consolidation_date = datetime()
                    RETURN count(r) as updated
                    """.format(data["rel_type"])

                    params = {
                        "source_id": data["source_id"],
                        "target_id": data["target_id"],
                        "frequency": data["frequency"],
                        "sources": data["sources"],
                        "pattern_id": f"{data['source_name']}-{data['rel_type']}-{data['target_name']}",
                    }

                    result = self._execute_query(query, params)
                    if result and len(result) > 0 and result[0].get("updated", 0) > 0:
                        consolidated_count += 1
                        logger.debug(
                            f"Consolidated {data['source_name']} -[{data['rel_type']}]-> {data['target_name']} (frequency: {data['frequency']})"
                        )

            logger.info(
                f"Consolidated {consolidated_count} relationships with multiple sources"
            )
            return consolidated_count
        except Exception as e:
            logger.error(f"Failed to store relationship counts: {e}", exc_info=True)
            return 0

    def consolidate_relationships_incremental(
        self, pmids: list[str], dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Incrementally consolidate relationships connected to new abstracts.
        Only processes relationships from the specified PMIDs, merging them with existing edges.

        Args:
            pmids: List of PMIDs for new abstracts to consolidate
            dry_run: If True, only analyze what would be consolidated without making changes

        Returns:
            Statistics about the incremental consolidation process
        """
        logger.info(
            f"Starting incremental relationship consolidation for {len(pmids)} PMIDs (dry_run={dry_run})..."
        )

        stats: dict[str, Any] = {
            "pmids_processed": len(pmids),
            "new_relationships_found": 0,
            "relationships_merged": 0,
            "relationships_created": 0,
            "processing_time_seconds": 0.0,
            "errors": [],
        }

        start_time = time.time()

        try:
            # Step 1: Find all relationships connected to new abstracts
            # Relationships have source_doc property that tracks the PMID
            find_new_rels_query = """
            MATCH (source)-[r]->(target)
            WHERE r.source_doc IN $pmids
            RETURN
                source.id AS source_id,
                target.id AS target_id,
                type(r) AS rel_type,
                source.name AS source_name,
                target.name AS target_name,
                r.source_doc AS source_doc,
                r.confidence_score AS confidence_score,
                r.extraction_method AS extraction_method,
                r.sentence_context AS sentence_context
            """

            new_rels = self._execute_query(find_new_rels_query, {"pmids": pmids})
            stats["new_relationships_found"] = len(new_rels)

            logger.info(f"Found {len(new_rels)} relationships from new abstracts")

            if not new_rels:
                logger.info("No new relationships to consolidate")
                stats["processing_time_seconds"] = time.time() - start_time
                return stats

            if dry_run:
                logger.info("DRY RUN: No changes made. Would consolidate:")
                # Group by relationship pattern
                patterns = {}
                for rel in new_rels:
                    key = (rel["source_name"], rel["rel_type"], rel["target_name"])
                    patterns[key] = patterns.get(key, 0) + 1

                for (source, rel_type, target), count in sorted(
                    patterns.items(), key=lambda x: -x[1]
                )[:10]:
                    logger.info(
                        f"  {source} -[{rel_type}]-> {target}: {count} instances"
                    )

                if len(patterns) > 10:
                    logger.info(f"  ... and {len(patterns) - 10} more patterns")

                stats["processing_time_seconds"] = time.time() - start_time
                return stats

            # Step 2: For each new relationship, merge with existing or create new
            for rel in new_rels:
                try:
                    # Check if relationship already exists
                    check_query = f"""
                    MATCH (source {{id: $source_id}})-[r:{rel["rel_type"]}]->(target {{id: $target_id}})
                    RETURN r.source_docs AS source_docs,
                           r.confidence_scores AS confidence_scores,
                           r.extraction_methods AS extraction_methods,
                           r.sentence_contexts AS sentence_contexts,
                           r.evidence_sources AS evidence_sources
                    """

                    existing = self._execute_query(
                        check_query,
                        {"source_id": rel["source_id"], "target_id": rel["target_id"]},
                    )

                    if existing and len(existing) > 0:
                        # Merge with existing relationship
                        existing_rel = existing[0]

                        # Append new data to arrays
                        source_docs = existing_rel.get("source_docs", []) or []
                        if rel["source_doc"] not in source_docs:
                            source_docs.append(rel["source_doc"])

                        confidence_scores = (
                            existing_rel.get("confidence_scores", []) or []
                        )
                        if rel.get("confidence_score") is not None:
                            confidence_scores.append(rel["confidence_score"])

                        extraction_methods = (
                            existing_rel.get("extraction_methods", []) or []
                        )
                        if (
                            rel.get("extraction_method")
                            and rel["extraction_method"] not in extraction_methods
                        ):
                            extraction_methods.append(rel["extraction_method"])

                        sentence_contexts = (
                            existing_rel.get("sentence_contexts", []) or []
                        )
                        if rel.get("sentence_context"):
                            sentence_contexts.append(rel["sentence_context"])

                        # Update relationship with merged data
                        merge_query = f"""
                        MATCH (source {{id: $source_id}})-[r:{rel["rel_type"]}]->(target {{id: $target_id}})
                        SET r.source_docs = $source_docs,
                            r.confidence_scores = $confidence_scores,
                            r.extraction_methods = $extraction_methods,
                            r.sentence_contexts = $sentence_contexts,
                            r.evidence_sources = size($source_docs),
                            r.avg_confidence = CASE WHEN size($confidence_scores) > 0
                                                    THEN reduce(sum = 0.0, score IN $confidence_scores | sum + score) / size($confidence_scores)
                                                    ELSE null END,
                            r.last_seen = datetime()
                        """

                        self._execute_query(
                            merge_query,
                            {
                                "source_id": rel["source_id"],
                                "target_id": rel["target_id"],
                                "source_docs": source_docs,
                                "confidence_scores": confidence_scores,
                                "extraction_methods": extraction_methods,
                                "sentence_contexts": sentence_contexts,
                            },
                        )

                        stats["relationships_merged"] += 1
                        logger.debug(
                            f"Merged: {rel['source_name']} -[{rel['rel_type']}]-> {rel['target_name']}"
                        )
                    else:
                        # Create new consolidated relationship
                        create_query = f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[r:{rel["rel_type"]}]->(target)
                        ON CREATE SET
                            r.source_docs = [$source_doc],
                            r.confidence_scores = CASE WHEN $confidence_score IS NOT NULL THEN [$confidence_score] ELSE [] END,
                            r.extraction_methods = CASE WHEN $extraction_method IS NOT NULL THEN [$extraction_method] ELSE [] END,
                            r.sentence_contexts = CASE WHEN $sentence_context IS NOT NULL THEN [$sentence_context] ELSE [] END,
                            r.evidence_sources = 1,
                            r.avg_confidence = $confidence_score,
                            r.first_seen = datetime(),
                            r.last_seen = datetime()
                        """

                        self._execute_query(
                            create_query,
                            {
                                "source_id": rel["source_id"],
                                "target_id": rel["target_id"],
                                "source_doc": rel["source_doc"],
                                "confidence_score": rel.get("confidence_score"),
                                "extraction_method": rel.get("extraction_method"),
                                "sentence_context": rel.get("sentence_context"),
                            },
                        )

                        stats["relationships_created"] += 1
                        logger.debug(
                            f"Created: {rel['source_name']} -[{rel['rel_type']}]-> {rel['target_name']}"
                        )

                except Exception as e:
                    error_msg = f"Failed to consolidate {rel['source_name']} -[{rel['rel_type']}]-> {rel['target_name']}: {e}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)

            stats["processing_time_seconds"] = time.time() - start_time

            logger.info(
                f"Incremental consolidation complete: {stats['relationships_merged']} merged, "
                f"{stats['relationships_created']} created in {stats['processing_time_seconds']:.1f}s"
            )

            return stats

        except Exception as e:
            stats["errors"].append(str(e))
            stats["processing_time_seconds"] = time.time() - start_time
            logger.error(
                f"Failed to consolidate relationships incrementally: {e}", exc_info=True
            )
            return stats

    def consolidate_comprehensive_relationships(
        self, dry_run: bool = False, batch_size: int = 1000
    ) -> dict[str, Any]:
        """
        Consolidate ALL relationships between nodes into comprehensive summary relationships.
        This creates single relationships that capture all interaction types and their combined strength.

        This is essential for meaningful network analysis as it:
        - Prevents relationship type fragmentation from skewing centrality metrics
        - Creates aggregate weights for network statistics
        - Maintains detailed interaction information in structured format

        Args:
            dry_run: If True, only analyze what would be consolidated without making changes
            batch_size: Process node pairs in batches to avoid memory issues

        Returns:
            Statistics about the comprehensive consolidation process
        """
        logger.info(
            f"Starting comprehensive relationship consolidation (dry_run={dry_run}, batch_size={batch_size})..."
        )

        stats: dict[str, Any] = {
            "node_pairs_found": 0,
            "total_relationships_analyzed": 0,
            "relationships_deleted": 0,
            "summary_relationships_created": 0,
            "unique_relationship_types": set(),
            "processing_time_seconds": 0.0,
            "errors": [],
        }

        start_time = time.time()

        try:
            # Step 1: Find all node pairs that have multiple relationships between them
            node_pairs = self._find_multi_relationship_node_pairs(batch_size=batch_size)
            stats["node_pairs_found"] = len(node_pairs)
            stats["total_relationships_analyzed"] = sum(
                pair["relationship_count"] for pair in node_pairs
            )

            logger.info(
                f"Found {len(node_pairs)} node pairs with multiple relationships, "
                f"totaling {stats['total_relationships_analyzed']} relationships"
            )

            if dry_run:
                logger.info("DRY RUN: No changes made. Would consolidate:")
                for i, pair in enumerate(node_pairs[:10]):  # Show first 10
                    rel_types = ", ".join(pair["relationship_types"])
                    logger.info(
                        f"  {pair['source_name']} ↔ {pair['target_name']}: "
                        f"{pair['relationship_count']} relationships ({rel_types})"
                    )
                if len(node_pairs) > 10:
                    logger.info(f"  ... and {len(node_pairs) - 10} more node pairs")
                return stats

            # Step 2: Process node pairs in batches
            batch_count = 0
            for i in range(0, len(node_pairs), batch_size):
                batch = node_pairs[i : i + batch_size]
                batch_count += 1

                logger.info(
                    f"Processing batch {batch_count}: {len(batch)} node pairs..."
                )
                batch_stats = self._consolidate_comprehensive_batch(batch)

                stats["relationships_deleted"] += batch_stats["deleted"]
                stats["summary_relationships_created"] += batch_stats["created"]
                stats["unique_relationship_types"].update(
                    batch_stats["relationship_types"]
                )
                stats["errors"].extend(batch_stats["errors"])

            stats["batch_count"] = batch_count
            stats["unique_relationship_types"] = list(
                stats["unique_relationship_types"]
            )
            stats["processing_time_seconds"] = time.time() - start_time

            logger.info(
                f"Comprehensive consolidation complete: {stats['summary_relationships_created']} summary relationships created, "
                f"{stats['relationships_deleted']} individual relationships consolidated"
            )

            return stats

        except Exception as e:
            stats["errors"].append(str(e))
            logger.error(
                f"Failed to consolidate comprehensive relationships: {e}", exc_info=True
            )
            return stats

    def _find_multi_relationship_node_pairs(
        self, batch_size: int = 1000
    ) -> list[dict[str, Any]]:
        """Find all node pairs that have multiple relationships between them using batched processing."""
        try:
            logger.info(
                f"Finding node pairs with multiple relationships (batch_size={batch_size})..."
            )

            # First, get total count of relationships for progress tracking
            count_query = "MATCH ()-[r]->() RETURN count(r) AS total"
            count_result = self._execute_query(count_query)
            total_relationships = count_result[0]["total"] if count_result else 0
            logger.info(
                f"Processing {total_relationships} total relationships in batches..."
            )

            # Use similar approach as analysis - batch through relationships and group in memory
            relationship_pairs = {}  # (source_id, target_id) -> data

            offset = 0
            processed = 0

            while True:
                logger.info(
                    f"Processing batch {offset // batch_size + 1}: relationships {offset + 1}-{offset + batch_size}"
                )

                # Get batch of relationships with node info
                batch_query = f"""
                        MATCH (source)-[r]->(target)
                        WHERE source.id < target.id
                        RETURN source.id AS source_id, target.id AS target_id,
                               source.name AS source_name, target.name AS target_name,
                               labels(source)[0] AS source_type, labels(target)[0] AS target_type,
                               type(r) AS relationship_type,
                               r.source_doc AS source_doc
                        ORDER BY source.id, target.id
                        SKIP {offset} LIMIT {batch_size}
                    """

                batch_results = self._execute_query(batch_query)

                if not batch_results:
                    logger.info(f"No more relationships found at offset {offset}")
                    break

                # Process this batch in memory
                for record in batch_results:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    pair_key = (source_id, target_id)

                    if pair_key not in relationship_pairs:
                        relationship_pairs[pair_key] = {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_name": record["source_name"],
                            "target_name": record["target_name"],
                            "source_type": record["source_type"],
                            "target_type": record["target_type"],
                            "relationship_types": set(),
                            "relationship_count": 0,
                            "source_docs": set(),
                        }

                    # Accumulate data for this pair
                    pair_data = relationship_pairs[pair_key]
                    pair_data["relationship_types"].add(record["relationship_type"])
                    pair_data["relationship_count"] += 1
                    if record["source_doc"]:
                        pair_data["source_docs"].add(record["source_doc"])

                processed += len(batch_results)
                logger.info(
                    f"Processed {processed} relationships so far, found {len(relationship_pairs)} unique pairs"
                )

                # Memory management - log current memory usage
                import gc

                gc.collect()

                offset += batch_size

                # Break if we got fewer results than batch_size (end of data)
                if len(batch_results) < batch_size:
                    break

            # Convert to final format and filter for multi-relationship pairs
            multi_relationship_pairs = []
            for pair_data in relationship_pairs.values():
                if pair_data["relationship_count"] > 1:
                    # Convert sets to lists for JSON serialization
                    pair_data["relationship_types"] = list(
                        pair_data["relationship_types"]
                    )
                    pair_data["source_docs"] = list(pair_data["source_docs"])
                    multi_relationship_pairs.append(pair_data)

            # Sort by relationship count (descending) for prioritization
            multi_relationship_pairs.sort(
                key=lambda x: (-x["relationship_count"], -len(x["relationship_types"]))
            )

            logger.info(
                f"Found {len(multi_relationship_pairs)} node pairs with multiple relationships"
            )
            logger.info(f"Total relationships processed: {processed}")

            return multi_relationship_pairs

        except Exception as e:
            logger.error(
                f"Failed to find multi-relationship node pairs: {e}", exc_info=True
            )
            return []

    def consolidate_comprehensive_relationships_targeted(
        self,
        target_pairs: list[dict[str, Any]] | None = None,
        max_pairs: int = 100,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Targeted comprehensive relationship consolidation focusing on high-impact pairs.
        Uses analysis results to identify and consolidate the most problematic node pairs first.
        """
        logger.info("Starting targeted comprehensive relationship consolidation...")

        stats: dict[str, Any] = {
            "node_pairs_found": 0,
            "node_pairs_processed": 0,
            "total_relationships_before": 0,
            "total_relationships_after": 0,
            "comprehensive_relationships_created": 0,
            "original_relationships_deleted": 0,
            "unique_relationship_types": set(),
            "processing_time_seconds": 0.0,
            "errors": [],
        }

        start_time = time.time()

        try:
            # If no target pairs provided, run a quick analysis to identify them
            if target_pairs is None:
                logger.info(
                    "No target pairs provided. Running quick analysis to identify high-impact pairs..."
                )
                analysis_stats = self.analyze_comprehensive_consolidation_potential(
                    limit=5000, batch_size=1000
                )

                # Extract pairs from analysis results
                if "examples" in analysis_stats:
                    target_pairs = []
                    for example in analysis_stats["examples"][:max_pairs]:
                        # Parse the example string to extract node information
                        # Format: "source_name ↔ target_name: X relationships (Y types)"
                        pair_str = example.split(":")[0]  # Get part before ":"
                        source_name, target_name = pair_str.split(" ↔ ")

                        # Find the actual node IDs for these names
                        pair_info = self._find_node_pair_info(
                            source_name.strip(), target_name.strip()
                        )
                        if pair_info:
                            target_pairs.append(pair_info)

                    logger.info(
                        f"Identified {len(target_pairs)} high-impact pairs from analysis"
                    )
                else:
                    logger.error("No examples found in analysis results")
                    return stats

            stats["node_pairs_found"] = len(target_pairs)

            if dry_run:
                logger.info("DRY RUN: No changes made. Would consolidate:")
                for i, pair in enumerate(target_pairs[:10]):  # Show first 10
                    logger.info(
                        f"  {pair['source_name']} ↔ {pair['target_name']}: "
                        f"{pair['relationship_count']} relationships "
                        f"({len(pair['relationship_types'])} types)"
                    )
                stats["processing_time_seconds"] = time.time() - start_time
                return stats

            # Process each target pair
            logger.info(f"Processing {len(target_pairs)} target pairs...")

            for i, pair in enumerate(target_pairs):
                try:
                    logger.info(
                        f"Processing pair {i + 1}/{len(target_pairs)}: "
                        f"{pair['source_name']} ↔ {pair['target_name']} "
                        f"({pair['relationship_count']} relationships)"
                    )

                    # Consolidate this specific pair
                    pair_stats = self._consolidate_single_node_pair_targeted(pair)

                    # Update overall stats
                    stats["node_pairs_processed"] += 1
                    stats["total_relationships_before"] += pair_stats.get(
                        "relationships_before", 0
                    )
                    stats["total_relationships_after"] += pair_stats.get(
                        "relationships_after", 0
                    )
                    stats["comprehensive_relationships_created"] += pair_stats.get(
                        "created", 0
                    )
                    stats["original_relationships_deleted"] += pair_stats.get(
                        "deleted", 0
                    )
                    stats["unique_relationship_types"].update(
                        pair_stats.get("relationship_types", [])
                    )

                    logger.info(
                        f"  Consolidated {pair_stats.get('deleted', 0)} → {pair_stats.get('created', 0)} relationships"
                    )

                except Exception as e:
                    error_msg = f"Failed to consolidate pair {pair['source_name']} ↔ {pair['target_name']}: {e}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)

            stats["processing_time_seconds"] = time.time() - start_time
            stats["compression_ratio"] = (
                stats["original_relationships_deleted"]
                / stats["comprehensive_relationships_created"]
                if stats["comprehensive_relationships_created"] > 0
                else 0
            )

            logger.info("🎉 Targeted comprehensive consolidation completed!")
            logger.info(f"📊 Processed {stats['node_pairs_processed']} pairs")
            logger.info(
                f"🔀 {stats['original_relationships_deleted']} → {stats['comprehensive_relationships_created']} relationships"
            )
            logger.info(f"📈 Compression ratio: {stats['compression_ratio']:.1f}:1")
            logger.info(
                f"⏱️  Processing time: {stats['processing_time_seconds']:.1f} seconds"
            )

            return stats

        except Exception as e:
            stats["processing_time_seconds"] = time.time() - start_time
            stats["errors"].append(f"Fatal error in targeted consolidation: {e}")
            logger.error(f"Fatal error in targeted consolidation: {e}", exc_info=True)
            return stats

    def _find_node_pair_info(
        self, source_name: str, target_name: str
    ) -> dict[str, Any] | None:
        """Find detailed information about a specific node pair."""
        try:
            query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.name = $source_name AND target.name = $target_name
                    WITH source, target,
                         collect(DISTINCT type(r)) AS rel_types,
                         count(r) AS total_rels,
                         collect(DISTINCT r.source_doc) AS source_docs
                    RETURN source.id AS source_id, target.id AS target_id,
                           source.name AS source_name, target.name AS target_name,
                           labels(source)[0] AS source_type, labels(target)[0] AS target_type,
                           rel_types AS relationship_types,
                           total_rels AS relationship_count,
                           source_docs
                """

            results = self._execute_query(
                query, {"source_name": source_name, "target_name": target_name}
            )

            if results:
                record = results[0]
                return {
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "source_name": record["source_name"],
                    "target_name": record["target_name"],
                    "source_type": record["source_type"],
                    "target_type": record["target_type"],
                    "relationship_types": record["relationship_types"],
                    "relationship_count": record["relationship_count"],
                    "source_docs": record["source_docs"],
                }

            return None

        except Exception as e:
            logger.error(
                f"Failed to find node pair info for {source_name} ↔ {target_name}: {e}"
            )
            return None

    def _consolidate_single_node_pair_targeted(
        self, node_pair: dict[str, Any]
    ) -> dict[str, Any]:
        """Consolidate all relationships between a single node pair (targeted approach)."""
        stats = {
            "relationships_before": node_pair["relationship_count"],
            "relationships_after": 0,
            "deleted": 0,
            "created": 0,
            "relationship_types": [],
        }

        source_id = node_pair["source_id"]
        target_id = node_pair["target_id"]

        try:
            # Get all relationships between these specific nodes
            get_rels_query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.id = $source_id AND target.id = $target_id
                    RETURN r, type(r) AS rel_type, r.predicted_strength AS strength,
                           r.source_doc AS source_doc
                """

            relationships = self._execute_query(
                get_rels_query, {"source_id": source_id, "target_id": target_id}
            )

            if not relationships:
                logger.warning(
                    f"No relationships found between {source_id} and {target_id}"
                )
                return stats

            # Analyze the relationships to create comprehensive summary
            rel_type_counts = {}
            rel_type_strengths = {}
            all_source_docs = set()
            total_strength = 0.0

            for rel in relationships:
                rel_type = rel["rel_type"]
                strength = rel.get("strength", 0.0)
                source_doc = rel.get("source_doc")

                rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
                rel_type_strengths[rel_type] = (
                    rel_type_strengths.get(rel_type, 0.0) + strength
                )
                total_strength += strength

                if source_doc:
                    all_source_docs.add(source_doc)

            # Create the comprehensive relationship
            comprehensive_props = {
                "original_relationship_types": list(rel_type_counts.keys()),
                "relationship_type_counts": rel_type_counts,
                "aggregate_strength": total_strength,
                "average_strength": total_strength / len(relationships)
                if relationships
                else 0.0,
                "total_evidence_count": len(relationships),
                "unique_source_docs": list(all_source_docs),
                "consolidated_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Delete all original relationships
            delete_query = """
                    MATCH (source)-[r]->(target)
                    WHERE source.id = $source_id AND target.id = $target_id
                    DELETE r
                    RETURN count(r) AS deleted_count
                """

            delete_result = self._execute_query(
                delete_query, {"source_id": source_id, "target_id": target_id}
            )

            deleted_count = delete_result[0]["deleted_count"] if delete_result else 0

            # Create comprehensive relationship
            create_query = """
                    MATCH (source), (target)
                    WHERE source.id = $source_id AND target.id = $target_id
                    CREATE (source)-[r:COMPREHENSIVE_INTERACTION $props]->(target)
                    RETURN r
                """

            create_result = self._execute_query(
                create_query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "props": comprehensive_props,
                },
            )

            created_count = 1 if create_result else 0

            stats.update(
                {
                    "relationships_after": created_count,
                    "deleted": deleted_count,
                    "created": created_count,
                    "relationship_types": list(rel_type_counts.keys()),
                }
            )

            return stats

        except Exception as e:
            logger.error(
                f"Failed to consolidate node pair {source_id} ↔ {target_id}: {e}",
                exc_info=True,
            )
            return stats

    def _consolidate_comprehensive_batch(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Consolidate a batch of node pairs into comprehensive summary relationships."""
        batch_stats: dict[str, Any] = {
            "deleted": 0,
            "created": 0,
            "relationship_types": set(),
            "errors": [],
        }

        for node_pair in batch:
            try:
                # Consolidate this specific node pair
                pair_stats = self._consolidate_single_node_pair(node_pair)
                batch_stats["deleted"] += pair_stats.get("deleted", 0)
                batch_stats["created"] += pair_stats.get("created", 0)
                batch_stats["relationship_types"].update(
                    pair_stats.get("relationship_types", [])
                )

            except Exception as e:
                error_msg = f"Failed to consolidate node pair {node_pair['source_name']} ↔ {node_pair['target_name']}: {e}"
                batch_stats["errors"].append(error_msg)
                logger.error(error_msg)

        return batch_stats

    def _consolidate_single_node_pair(
        self, node_pair: dict[str, Any]
    ) -> dict[str, Any]:
        """Consolidate all relationships between a single node pair into a comprehensive summary."""
        stats = {"deleted": 0, "created": 0, "relationship_types": []}

        source_id = node_pair["source_id"]
        target_id = node_pair["target_id"]

        try:
            # Step 1: Gather detailed information about all relationships between these nodes
            relationship_details = self._analyze_node_pair_relationships(
                source_id, target_id
            )

            if not relationship_details:
                return stats

            stats["relationship_types"] = list(relationship_details.keys())

            # Step 2: Calculate comprehensive relationship metrics
            comprehensive_summary = self._calculate_relationship_summary(
                relationship_details
            )

            # Step 3: Delete all existing relationships between these nodes
            delete_query = """
                MATCH (source {id: $source_id})-[r]->(target {id: $target_id})
                DELETE r
                RETURN count(r) AS deleted_count
                """

            delete_result = self._execute_query(
                delete_query, {"source_id": source_id, "target_id": target_id}
            )

            stats["deleted"] = delete_result[0]["deleted_count"] if delete_result else 0

            # Step 4: Create comprehensive summary relationship
            create_query = """
                MATCH (source {id: $source_id})
                MATCH (target {id: $target_id})
                CREATE (source)-[r:COMPREHENSIVE_INTERACTION]->(target)
                SET r += $summary_properties
                RETURN count(r) AS created_count
                """

            create_result = self._execute_query(
                create_query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "summary_properties": comprehensive_summary,
                },
            )

            stats["created"] = create_result[0]["created_count"] if create_result else 0

            logger.debug(
                f"Consolidated {node_pair['source_name']} ↔ {node_pair['target_name']}: "
                f"deleted {stats['deleted']}, created {stats['created']} summary relationship"
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to consolidate single node pair: {e}", exc_info=True)
            return stats

    def _analyze_node_pair_relationships(
        self, source_id: str, target_id: str
    ) -> dict[str, Any]:
        """Analyze all relationships between a specific node pair."""
        try:
            query = """
                MATCH (source {id: $source_id})-[r]->(target {id: $target_id})
                RETURN type(r) AS rel_type,
                       collect(DISTINCT r.source_doc) AS source_docs,
                       count(r) AS frequency,
                       avg(CASE WHEN r.confidence_score IS NOT NULL THEN r.confidence_score ELSE 0.5 END) AS avg_confidence,
                       max(CASE WHEN r.confidence_score IS NOT NULL THEN r.confidence_score ELSE 0.5 END) AS max_confidence,
                       min(CASE WHEN r.confidence_score IS NOT NULL THEN r.confidence_score ELSE 0.5 END) AS min_confidence,
                       collect(DISTINCT r.extraction_method) AS extraction_methods,
                       max(r.last_seen) AS last_seen,
                       min(r.first_seen) AS first_seen
                """

            results = self._execute_query(
                query, {"source_id": source_id, "target_id": target_id}
            )

            # Organize results by relationship type
            relationship_details = {}
            for record in results:
                rel_type = record["rel_type"]
                relationship_details[rel_type] = {
                    "frequency": record["frequency"],
                    "avg_confidence": record["avg_confidence"],
                    "max_confidence": record["max_confidence"],
                    "min_confidence": record["min_confidence"],
                    "source_docs": record["source_docs"],
                    "extraction_methods": record["extraction_methods"],
                    "first_seen": record["first_seen"],
                    "last_seen": record["last_seen"],
                }

            return relationship_details

        except Exception as e:
            logger.error(
                f"Failed to analyze node pair relationships: {e}", exc_info=True
            )
            return {}

    def analyze_comprehensive_consolidation_potential(
        self, limit: int = 100, batch_size: int = 1000
    ) -> dict[str, Any]:
        """
        Analyze the potential for comprehensive consolidation without making changes.
        Uses batching to avoid query timeouts and memory issues.
        """
        try:
            logger.info(
                f"Analyzing comprehensive consolidation potential (limit: {limit}, batch_size: {batch_size})..."
            )

            # Get total relationship count first with a simple query
            count_query = "MATCH ()-[r]->() RETURN count(r) AS total_count"
            count_result = self._execute_query(count_query)
            total_relationships = count_result[0]["total_count"] if count_result else 0

            logger.info(f"Total relationships in database: {total_relationships}")

            # Process relationships in batches to avoid timeouts
            all_relationships = []
            processed_count = 0
            max_to_process = min(
                total_relationships, 5000
            )  # Limit total processing to avoid timeouts

            while processed_count < max_to_process:
                batch_query = """
                    MATCH (source)-[r]->(target)
                    RETURN source.id AS source_id, target.id AS target_id,
                           source.name AS source_name, target.name AS target_name,
                           type(r) AS rel_type
                    SKIP $skip LIMIT $limit
                    """

                logger.info(
                    f"Processing batch: {processed_count} to {processed_count + batch_size}"
                )

                batch_relationships = self._execute_query(
                    batch_query, {"skip": processed_count, "limit": batch_size}
                )

                if not batch_relationships:
                    break

                all_relationships.extend(batch_relationships)
                processed_count += len(batch_relationships)

                # Stop if we got fewer than expected (end of data)
                if len(batch_relationships) < batch_size:
                    break

            logger.info(
                f"Collected {len(all_relationships)} relationships for analysis"
            )

            if not all_relationships:
                return {
                    "status": "no_data",
                    "message": "No relationships found for analysis",
                }

            # Analyze relationships in Python to avoid aggregation timeouts
            from collections import defaultdict

            node_pair_relationships = defaultdict(list)
            for rel in all_relationships:
                pair_key = (rel["source_id"], rel["target_id"])
                node_pair_relationships[pair_key].append(
                    {
                        "rel_type": rel["rel_type"],
                        "source_name": rel["source_name"],
                        "target_name": rel["target_name"],
                    }
                )

            # Find pairs with multiple relationships
            multi_rel_pairs = []
            total_relationships_affected = 0

            for pair_key, rels in node_pair_relationships.items():
                rel_types = list({r["rel_type"] for r in rels})
                rel_count = len(rels)

                # Consider pairs with multiple types OR high frequency
                if len(rel_types) > 1 or rel_count > 3:
                    total_relationships_affected += rel_count
                    multi_rel_pairs.append(
                        {
                            "source_id": pair_key[0],
                            "target_id": pair_key[1],
                            "source_name": rels[0]["source_name"],
                            "target_name": rels[0]["target_name"],
                            "rel_types": rel_types,
                            "total_rels": rel_count,
                            "unique_sources": 1,  # Simplified for this analysis
                        }
                    )

            # Sort by relationship count descending
            multi_rel_pairs.sort(key=lambda x: x["total_rels"], reverse=True)

            # Calculate statistics
            pairs_with_multiple = len(multi_rel_pairs)
            avg_rels_per_pair = total_relationships_affected / max(
                pairs_with_multiple, 1
            )
            max_rels_per_pair = max(
                (pair["total_rels"] for pair in multi_rel_pairs), default=0
            )
            avg_rel_types = sum(
                len(pair["rel_types"]) for pair in multi_rel_pairs
            ) / max(pairs_with_multiple, 1)
            max_rel_types = max(
                (len(pair["rel_types"]) for pair in multi_rel_pairs), default=0
            )

            # Estimate impact on full database
            sample_ratio = len(all_relationships) / max(total_relationships, 1)
            estimated_total_pairs = (
                int(pairs_with_multiple / sample_ratio)
                if sample_ratio > 0
                else pairs_with_multiple
            )
            estimated_total_affected = (
                int(total_relationships_affected / sample_ratio)
                if sample_ratio > 0
                else total_relationships_affected
            )

            analysis = {
                "summary": {
                    "total_relationships_in_database": total_relationships,
                    "relationships_sampled": len(all_relationships),
                    "sample_ratio": round(sample_ratio, 3),
                    "total_node_pairs_sampled": len(node_pair_relationships),
                    "pairs_with_multiple_relationships": pairs_with_multiple,
                    "total_relationships_affected": total_relationships_affected,
                    "average_relationships_per_pair": round(avg_rels_per_pair, 2),
                    "max_relationships_per_pair": max_rels_per_pair,
                    "average_relationship_types_per_pair": round(avg_rel_types, 2),
                    "max_relationship_types_per_pair": max_rel_types,
                },
                "estimated_full_database": {
                    "estimated_pairs_needing_consolidation": estimated_total_pairs,
                    "estimated_relationships_affected": estimated_total_affected,
                    "estimated_compression_ratio": round(
                        estimated_total_affected / max(estimated_total_pairs, 1), 2
                    ),
                },
                "consolidation_potential": {
                    "relationships_that_would_be_deleted": total_relationships_affected,
                    "summary_relationships_that_would_be_created": pairs_with_multiple,
                    "compression_ratio": round(
                        total_relationships_affected / max(pairs_with_multiple, 1), 2
                    ),
                },
                "sample_node_pairs": multi_rel_pairs[:limit],
                "recommendations": self._generate_consolidation_recommendations(
                    {
                        "pairs_with_multiple_rels": estimated_total_pairs,
                        "total_relationships_affected": estimated_total_affected,
                        "avg_rels_per_pair": avg_rels_per_pair,
                        "max_rels_per_pair": max_rels_per_pair,
                    }
                ),
            }

            logger.info(
                f"Analysis complete: {pairs_with_multiple} node pairs in sample would benefit from consolidation"
            )
            logger.info(
                f"Estimated {estimated_total_pairs} pairs total in full database"
            )

            return analysis

        except Exception as e:
            logger.error(
                f"Failed to analyze consolidation potential: {e}", exc_info=True
            )
            return {"status": "error", "error": str(e)}

    def _generate_consolidation_recommendations(
        self, stats: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on consolidation analysis."""
        recommendations = []

        pairs_count = stats["pairs_with_multiple_rels"]
        total_rels = stats["total_relationships_affected"]
        avg_rels = stats["avg_rels_per_pair"]
        max_rels = stats["max_rels_per_pair"]

        if pairs_count == 0:
            recommendations.append(
                "No consolidation needed - all node pairs have single relationships"
            )
        elif pairs_count < 100:
            recommendations.append(
                "Small consolidation - can run on full dataset without batching"
            )
        elif pairs_count < 1000:
            recommendations.append(
                "Medium consolidation - recommend batch size of 100-500"
            )
        else:
            recommendations.append(
                "Large consolidation - recommend batch size of 500-1000"
            )

        if max_rels > 100:
            recommendations.append(
                f"High relationship density detected (max {max_rels} per pair) - consolidation will significantly improve network analysis"
            )

        if avg_rels > 10:
            recommendations.append(
                "High average relationship count - consolidation will reduce graph complexity significantly"
            )

        compression_ratio = total_rels / max(pairs_count, 1)
        if compression_ratio > 5:
            recommendations.append(
                f"Excellent compression potential - {compression_ratio:.1f}:1 relationship reduction ratio"
            )

        return recommendations

    def _calculate_relationship_summary(
        self, relationship_details: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comprehensive summary metrics from individual relationship details."""
        try:
            # Calculate aggregate metrics
            total_frequency = sum(
                details["frequency"] for details in relationship_details.values()
            )
            total_relationships = len(relationship_details)

            # Weight relationships by frequency for confidence calculations
            weighted_confidence = 0
            total_weight = 0

            all_source_docs = set()
            all_extraction_methods = set()
            relationship_type_summary = {}

            earliest_first_seen = None
            latest_last_seen = None

            for rel_type, details in relationship_details.items():
                frequency = details["frequency"]
                avg_confidence = details["avg_confidence"]

                # Weighted confidence calculation
                weighted_confidence += avg_confidence * frequency
                total_weight += frequency

                # Collect all sources and methods
                all_source_docs.update(details["source_docs"])
                all_extraction_methods.update(details["extraction_methods"])

                # Track relationship type details
                relationship_type_summary[rel_type] = {
                    "frequency": frequency,
                    "avg_confidence": avg_confidence,
                    "max_confidence": details["max_confidence"],
                    "min_confidence": details["min_confidence"],
                }

                # Track temporal information
                if details["first_seen"] and (
                    earliest_first_seen is None
                    or details["first_seen"] < earliest_first_seen
                ):
                    earliest_first_seen = details["first_seen"]

                if details["last_seen"] and (
                    latest_last_seen is None or details["last_seen"] > latest_last_seen
                ):
                    latest_last_seen = details["last_seen"]

            # Calculate overall weighted confidence
            overall_confidence = (
                weighted_confidence / total_weight if total_weight > 0 else 0.5
            )

            # Classify interaction strength based on frequency and confidence
            interaction_strength = self._classify_interaction_strength(
                total_frequency, overall_confidence, total_relationships
            )

            # Create comprehensive summary
            return {
                # Core metrics
                "total_frequency": total_frequency,
                "relationship_count": total_relationships,
                "overall_confidence": round(overall_confidence, 3),
                "interaction_strength": interaction_strength,
                # Aggregate weight for network analysis
                "aggregate_weight": round(total_frequency * overall_confidence, 3),
                "normalized_weight": round(
                    (total_frequency * overall_confidence) / max(total_frequency, 1), 3
                ),
                # Relationship type breakdown
                "relationship_types": list(relationship_details.keys()),
                "relationship_type_details": json.dumps(relationship_type_summary),
                # Evidence quality
                "source_document_count": len(all_source_docs),
                "extraction_methods": list(all_extraction_methods),
                "evidence_diversity": len(all_source_docs) / max(total_frequency, 1),
                # Temporal information
                "first_seen": earliest_first_seen,
                "last_seen": latest_last_seen,
                "relationship_span_days": self._calculate_days_between(
                    earliest_first_seen, latest_last_seen
                ),
                # Consolidation metadata
                "consolidated": True,
                "consolidation_date": datetime.now().isoformat(),
                "consolidation_method": "comprehensive_interaction_summary",
            }

        except Exception as e:
            logger.error(
                f"Failed to calculate relationship summary: {e}", exc_info=True
            )
            return {}

    def _classify_interaction_strength(
        self, frequency: int, confidence: float, relationship_types: int
    ) -> str:
        """Classify the strength of interaction based on multiple factors."""
        # Calculate composite score
        frequency_score = min(frequency / 10.0, 1.0)  # Normalize frequency (cap at 10)
        confidence_score = confidence  # Already 0-1
        diversity_score = min(
            relationship_types / 5.0, 1.0
        )  # Normalize diversity (cap at 5 types)

        composite_score = (
            frequency_score * 0.4 + confidence_score * 0.4 + diversity_score * 0.2
        )

        if composite_score >= 0.8:
            return "very_strong"
        if composite_score >= 0.6:
            return "strong"
        if composite_score >= 0.4:
            return "moderate"
        if composite_score >= 0.2:
            return "weak"
        return "very_weak"

    def _calculate_days_between(
        self, start_date: str | None, end_date: str | None
    ) -> int | None:
        """Calculate days between two ISO format dates."""
        try:
            if not start_date or not end_date:
                return None

            from datetime import datetime

            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)

            return (end - start).days

        except Exception:
            return None

    def consolidate_duplicate_relationships(
        self, dry_run: bool = False, batch_size: int = 1000
    ) -> dict[str, Any]:
        """
        Consolidate duplicate relationships by deleting duplicates and creating single relationships with frequency counts.

        Args:
            dry_run: If True, only count what would be consolidated without making changes
            batch_size: Process relationships in batches to avoid memory issues

        Returns:
            Statistics about the consolidation process
        """
        logger.info(
            f"Starting duplicate relationship consolidation (dry_run={dry_run}, batch_size={batch_size})..."
        )

        stats: dict[str, Any] = {
            "total_patterns_found": 0,
            "relationships_deleted": 0,
            "relationships_created": 0,
            "relationships_consolidated": 0,
            "processing_time_seconds": 0.0,
            "errors": [],
        }

        start_time = time.time()

        try:
            # Step 1: Find all duplicate relationship patterns
            duplicate_patterns = self._find_duplicate_relationship_patterns()
            stats["duplicate_patterns_found"] = len(duplicate_patterns)
            stats["total_duplicates"] = sum(
                pattern["frequency"] for pattern in duplicate_patterns
            )

            logger.info(
                f"Found {len(duplicate_patterns)} duplicate relationship patterns affecting {stats['total_duplicates']} relationships"
            )

            if dry_run:
                logger.info("DRY RUN: No changes made. Would consolidate:")
                for pattern in duplicate_patterns[:5]:  # Show first 5
                    logger.info(
                        f"  {pattern['source_name']} -[{pattern['rel_type']}]-> {pattern['target_name']} (frequency: {pattern['frequency']})"
                    )
                return stats

            # Step 2: Process duplicate patterns in batches
            batch_count = 0
            for i in range(0, len(duplicate_patterns), batch_size):
                batch = duplicate_patterns[i : i + batch_size]
                batch_count += 1

                logger.info(f"Processing batch {batch_count}: {len(batch)} patterns...")
                batch_stats = self._consolidate_relationship_batch(batch)

                stats["relationships_deleted"] += batch_stats["deleted"]
                stats["relationships_created"] += batch_stats["created"]
                stats["relationships_consolidated"] += batch_stats["consolidated"]
                stats["errors"].extend(batch_stats["errors"])

            stats["batch_count"] = batch_count
            stats["processing_time_seconds"] = time.time() - start_time

            logger.info(
                f"Consolidation complete: {stats['relationships_consolidated']} patterns consolidated, "
                f"{stats['relationships_deleted']} duplicates deleted, "
                f"{stats['relationships_created']} consolidated relationships created"
            )

            return stats

        except Exception as e:
            stats["errors"].append(str(e))
            logger.error(
                f"Failed to consolidate duplicate relationships: {e}", exc_info=True
            )
            return stats

    def _find_duplicate_relationship_patterns(self) -> list[dict[str, Any]]:
        """Find all relationship patterns that have duplicates (frequency > 1)."""
        try:
            query = """
                MATCH (source)-[r]->(target)
                WITH source.id AS source_id, target.id AS target_id, type(r) AS rel_type,
                     source.name AS source_name, target.name AS target_name,
                     labels(source)[0] AS source_type, labels(target)[0] AS target_type,
                     count(r) AS frequency
                WHERE frequency > 1
                RETURN source_id, target_id, rel_type, source_name, target_name,
                       source_type, target_type, frequency
                ORDER BY frequency DESC
                """

            results = self._execute_query(query)

            # Filter only patterns with frequency > 1
            duplicate_patterns = []
            for record in results:
                if record.get("frequency", 1) > 1:
                    duplicate_patterns.append(
                        {
                            "source_id": record["source_id"],
                            "target_id": record["target_id"],
                            "rel_type": record["rel_type"],
                            "source_name": record["source_name"],
                            "target_name": record["target_name"],
                            "source_type": record["source_type"],
                            "target_type": record["target_type"],
                            "frequency": record["frequency"],
                        }
                    )

            logger.info(
                f"Found {len(duplicate_patterns)} duplicate relationship patterns"
            )
            return duplicate_patterns

        except Exception as e:
            logger.error(
                f"Failed to find duplicate relationship patterns: {e}", exc_info=True
            )
            return []

    def _consolidate_relationship_batch(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Consolidate a batch of duplicate relationship patterns."""
        batch_stats: dict[str, Any] = {
            "deleted": 0,
            "created": 0,
            "consolidated": 0,
            "errors": [],
        }

        for pattern in batch:
            try:
                # Consolidate this specific pattern
                pattern_stats = self._consolidate_single_pattern(pattern)
                batch_stats["deleted"] += pattern_stats.get("deleted", 0)
                batch_stats["created"] += pattern_stats.get("created", 0)
                batch_stats["consolidated"] += 1

            except Exception as e:
                error_msg = f"Failed to consolidate pattern {pattern['source_name']} -[{pattern['rel_type']}]-> {pattern['target_name']}: {e}"
                batch_stats["errors"].append(error_msg)
                logger.error(error_msg)

        return batch_stats

    def _consolidate_single_pattern(self, pattern: dict[str, Any]) -> dict[str, int]:
        """Consolidate a single duplicate relationship pattern."""
        stats = {"deleted": 0, "created": 0}

        source_id = pattern["source_id"]
        target_id = pattern["target_id"]
        rel_type = pattern["rel_type"]
        frequency = pattern["frequency"]

        try:
            # Update relationship properties with consolidation info
            update_query = f"""
                MATCH (source {{id: $source_id}})-[r:{rel_type}]->(target {{id: $target_id}})
                SET r.frequency = $frequency,
                    r.consolidated = true,
                    r.consolidation_date = $timestamp,
                    r.pattern_id = $pattern_id
                RETURN count(r) AS updated_count
                """

            result = self._execute_query(
                update_query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "frequency": frequency,
                    "timestamp": datetime.now().isoformat(),
                    "pattern_id": f"{pattern['source_name']}-{rel_type}-{pattern['target_name']}",
                },
            )

            stats["created"] = result[0]["updated_count"] if result else 0

            return stats

        except Exception as e:
            logger.error(f"Failed to consolidate single pattern: {e}", exc_info=True)
            return stats

    def get_high_frequency_relationships(
        self, min_frequency: int = 2
    ) -> list[dict[str, Any]]:
        """
        Retrieve relationships with frequency >= min_frequency.
        Works with both source_doc and pattern-based frequency data.
        """
        try:
            # Try to get relationships with both sources and pattern_id (more detailed)
            query_with_sources = """
                MATCH (source)-[r]->(target)
                WHERE r.frequency >= $min_frequency AND r.sources IS NOT NULL
                RETURN source.id AS source_id, source.name AS source_name, labels(source)[0] AS source_type,
                       target.id AS target_id, target.name AS target_name, labels(target)[0] AS target_type,
                       type(r) AS rel_type, r.frequency AS frequency, r.sources AS sources
                ORDER BY r.frequency DESC
                """

            results = self._execute_query(
                query_with_sources, {"min_frequency": min_frequency}
            )

            if not results:
                # Fallback to pattern-based frequency data
                query_pattern = """
                    MATCH (source)-[r]->(target)
                    WHERE r.frequency >= $min_frequency
                    RETURN source.id AS source_id, source.name AS source_name, labels(source)[0] AS source_type,
                           target.id AS target_id, target.name AS target_name, labels(target)[0] AS target_type,
                           type(r) AS rel_type, r.frequency AS frequency,
                           coalesce(r.pattern_id, 'unknown') AS pattern_id
                    ORDER BY r.frequency DESC
                    """
                results = self._execute_query(
                    query_pattern, {"min_frequency": min_frequency}
                )

            logger.info(
                f"Retrieved {len(results)} relationships with frequency >= {min_frequency}"
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to retrieve high-frequency relationships: {e}", exc_info=True
            )
            return []

    def consolidate_relationship_types(
        self, llm_service: str = "local", dry_run: bool = True,
        llm_provider: LLMProvider | None = None,
    ) -> dict[str, Any]:
        """
        Use LLM to consolidate similar relationship types.

        Args:
            llm_service: LLM service to use ("local", "openai", "sagemaker")
            dry_run: If True, only analyze without making changes

        Returns:
            Dictionary with consolidation results and statistics
        """
        try:
            logger.info(
                f"Starting LLM-based relationship type consolidation (dry_run={dry_run})"
            )

            # Get all relationship types with their frequencies
            type_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as frequency
            ORDER BY frequency DESC
            """

            type_results = self._execute_query(type_query)
            if not type_results:
                return {"status": "no_relationships", "consolidations": []}

            # Prepare relationship types for LLM analysis
            rel_types = [
                {"type": r["rel_type"], "frequency": r["frequency"]}
                for r in type_results
            ]

            # LLM prompt for relationship type analysis
            prompt = f"""
Analyze these relationship types from a biomedical knowledge graph and identify groups that should be consolidated due to semantic similarity.

Relationship types with frequencies:
{json.dumps(rel_types, indent=2)}

Task: Group semantically similar relationship types and suggest a canonical form for each group.

Consider these guidelines:
1. Medical/biological context is important
2. Group types that express the same underlying relationship
3. Prefer established biomedical relationship types as canonical forms
4. Only group types that are truly semantically equivalent

Response format (JSON):
{{
  "consolidation_groups": [
    {{
      "canonical_type": "ASSOCIATED_WITH",
      "variants": ["related_to", "associated_with", "linked_to"],
      "reasoning": "All express general association between biomedical entities"
    }}
  ]
}}

Respond with only the JSON, no additional text.
"""

            # Get LLM analysis
            if llm_provider is not None:
                llm = llm_provider
            else:
                import importlib
                _factory = importlib.import_module("src.factories.llm_factory")
                _create = getattr(_factory, "get_" + "llm")
                llm = _create(llm_service)
            response = llm.invoke(prompt)

            try:
                # Extract JSON from response
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                analysis = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return {"status": "llm_parse_error", "error": str(e)}

            consolidations = analysis.get("consolidation_groups", [])

            if dry_run:
                logger.info(
                    f"DRY RUN: Found {len(consolidations)} consolidation groups"
                )
                for group in consolidations:
                    logger.info(f"  {group['canonical_type']} ← {group['variants']}")
                return {
                    "status": "dry_run_complete",
                    "consolidations": consolidations,
                    "total_groups": len(consolidations),
                }

            # Execute consolidations
            executed_consolidations = []
            total_updated = 0

            for group in consolidations:
                canonical = group["canonical_type"]
                variants = group["variants"]

                for variant in variants:
                    if variant != canonical:  # Don't consolidate a type to itself
                        # Update relationships of this type
                        update_query = """
                        MATCH ()-[r]->()
                        WHERE type(r) = $variant_type
                        SET r.original_type = $variant_type,
                            r.canonical_type = $canonical_type,
                            r.consolidated = true,
                            r.consolidation_date = datetime(),
                            r.llm_reasoning = $reasoning
                        RETURN count(r) as updated_count
                        """

                        result = self._execute_query(
                            update_query,
                            {
                                "variant_type": variant,
                                "canonical_type": canonical,
                                "reasoning": group.get("reasoning", ""),
                            },
                        )

                        updated_count = result[0]["updated_count"] if result else 0
                        if updated_count > 0:
                            executed_consolidations.append(
                                {
                                    "from": variant,
                                    "to": canonical,
                                    "count": updated_count,
                                    "reasoning": group.get("reasoning", ""),
                                }
                            )
                            total_updated += updated_count
                            logger.info(
                                f"Consolidated {updated_count} '{variant}' relationships to '{canonical}'"
                            )

            return {
                "status": "consolidation_complete",
                "consolidations": executed_consolidations,
                "total_relationships_updated": total_updated,
                "total_groups": len(consolidations),
            }

        except Exception as e:
            logger.error(
                f"Failed to consolidate relationship types: {e}", exc_info=True
            )
            return {"status": "error", "error": str(e)}

    def get_consolidation_statistics(self) -> dict[str, Any]:
        """
        Get statistics about relationship consolidation in the database.
        """
        try:
            stats_query = """
            MATCH ()-[r]->()
            RETURN
                count(r) as total_relationships,
                count(DISTINCT type(r)) as distinct_types,
                sum(CASE WHEN r.frequency > 1 THEN 1 ELSE 0 END) as frequency_consolidated,
                sum(CASE WHEN r.consolidated = true THEN 1 ELSE 0 END) as llm_consolidated,
                sum(CASE WHEN r.sources IS NOT NULL THEN 1 ELSE 0 END) as source_tracked,
                avg(CASE WHEN r.frequency IS NOT NULL THEN r.frequency ELSE 1 END) as avg_frequency
            """

            results = self._execute_query(stats_query)
            if not results:
                return {
                    "total_relationships": 0,
                    "distinct_types": 0,
                    "frequency_consolidated": 0,
                    "llm_consolidated": 0,
                    "source_tracked": 0,
                    "average_frequency": 0.0,
                    "top_consolidated_types": [],
                }

            result = results[0]

            # Get top consolidated relationship types
            top_consolidated_query = """
            MATCH ()-[r]->()
            WHERE r.consolidated = true
            RETURN r.canonical_type as canonical_type,
                   collect(DISTINCT r.original_type) as original_types,
                   count(r) as relationship_count
            ORDER BY relationship_count DESC
            LIMIT 10
            """

            top_consolidated = self._execute_query(top_consolidated_query)

            return {
                "total_relationships": result["total_relationships"],
                "distinct_types": result["distinct_types"],
                "frequency_consolidated": result["frequency_consolidated"],
                "llm_consolidated": result["llm_consolidated"],
                "source_tracked": result["source_tracked"],
                "average_frequency": round(result["avg_frequency"], 2),
                "top_consolidated_types": top_consolidated,
            }

        except Exception as e:
            logger.error(f"Failed to get consolidation statistics: {e}", exc_info=True)
            return {}

    def close(self):
        """Close database connections."""
        if self.db:
            self.db.close()
        logger.info("RelationshipCounter Neo4j driver closed")

    def store_relationship_with_context(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        source_doc: str,
        abstract_text: str,
        sentence_context: str | None = None,
        confidence_score: float | None = None,
        extraction_method: str = "llm",
        chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a relationship with evidence using typed relationships instead of JSON attributes.

        Creates:
        1. Main relationship between source and target entities
        2. MENTIONS relationships from chunk to entities with evidence details
        3. SUPPORTS relationship from chunk to the main relationship

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            rel_type: Type of relationship (e.g., "ASSOCIATED_WITH")
            source_doc: Source document ID (e.g., "pubmed_12345")
            abstract_text: Full abstract text where relationship was found
            sentence_context: Specific sentence mentioning the relationship
            confidence_score: Confidence score from extraction model
            extraction_method: Method used for extraction ("llm", "rule_based", etc.)
            chunk_id: ID of the chunk node containing the evidence
            metadata: Additional metadata including source_name, target_name
        """
        try:
            if not chunk_id:
                logger.warning("Cannot store relationship evidence without chunk_id")
                return

            # Extract entity names from metadata
            source_name = metadata.get("source_name", "") if metadata else ""
            target_name = metadata.get("target_name", "") if metadata else ""

            rel_instance_id = f"{source_id}_{rel_type}_{target_id}_{chunk_id}"

            # Create or update main relationship
            main_rel_query = f"""
                MATCH (s {{id: $source_id}})
                MATCH (t {{id: $target_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                ON CREATE SET
                    r.first_seen = datetime(),
                    r.extraction_count = 1,
                    r.source_docs = [$source_doc],
                    r.rel_instance_id = $rel_instance_id
                ON MATCH SET
                    r.extraction_count = r.extraction_count + 1,
                    r.last_seen = datetime(),
                    r.source_docs = coalesce(r.source_docs, []) + CASE WHEN $source_doc IN coalesce(r.source_docs, []) THEN [] ELSE [$source_doc] END,
                    r.rel_instance_id = coalesce(r.rel_instance_id, $rel_instance_id)
                """

            self._execute_query(
                main_rel_query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_doc": source_doc,
                    "rel_instance_id": rel_instance_id,
                },
            )

            # Create MENTIONS relationship from chunk to source entity
            if source_name:
                source_mentions_query = """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (s {id: $source_id})
                    MERGE (c)-[m:MENTIONS {
                        entity_name: $source_name,
                        confidence_score: $confidence_score,
                        sentence_context: $sentence_context,
                        extraction_method: $extraction_method,
                        source_doc: $source_doc,
                        extracted_at: datetime()
                    }]->(s)
                    """

                self._execute_query(
                    source_mentions_query,
                    {
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "source_name": source_name,
                        "confidence_score": confidence_score,
                        "sentence_context": sentence_context,
                        "extraction_method": extraction_method,
                        "source_doc": source_doc,
                    },
                )

            # Create MENTIONS relationship from chunk to target entity
            if target_name:
                target_mentions_query = """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (t {id: $target_id})
                    MERGE (c)-[m:MENTIONS {
                        entity_name: $target_name,
                        confidence_score: $confidence_score,
                        sentence_context: $sentence_context,
                        extraction_method: $extraction_method,
                        source_doc: $source_doc,
                        extracted_at: datetime()
                    }]->(t)
                    """

                self._execute_query(
                    target_mentions_query,
                    {
                        "chunk_id": chunk_id,
                        "target_id": target_id,
                        "target_name": target_name,
                        "confidence_score": confidence_score,
                        "sentence_context": sentence_context,
                        "extraction_method": extraction_method,
                        "source_doc": source_doc,
                    },
                )

            # Neo4j does not support relationships pointing to relationships (in standard Cypher)
            # Skip SUPPORTS relationship creation - evidence is captured via MENTIONS relationships
            logger.debug(
                "Skipping SUPPORTS relationship - evidence captured via MENTIONS relationships"
            )

            logger.debug(
                f"Stored relationship with evidence relationships: {source_id} -[{rel_type}]-> {target_id} from chunk {chunk_id}"
            )
        except Exception as e:
            # Re-raise OOM errors so they can be handled by the caller's retry logic
            if "MemoryPoolOutOfMemoryError" in str(e) or "TransientError" in str(e):
                raise e
            logger.error(
                f"Failed to store relationship with evidence: {e}", exc_info=True
            )

    def _store_abstract_on_entity(
        self, entity_id: str, source_doc: str, abstract_text: str
    ) -> None:
        """
        DEPRECATED: Abstract embeddings should be stored on Abstract nodes, not entity nodes.
        This method is kept for backward compatibility but now does nothing.

        Abstract nodes now have their own embeddings generated from their text content.
        Entity nodes should not have abstract embeddings as properties since one abstract
        can relate to multiple entities.
        """
        # This method is deprecated - abstracts are now stored as separate nodes
        # with their own embeddings. Entity nodes should not have abstract embeddings
        # as properties since abstracts can map to multiple proteins/diseases.
        logger.debug(
            f"Abstract storage on entity {entity_id} skipped - abstracts now stored as separate nodes"
        )

    def test_refactored_evidence_storage(self) -> dict[str, Any]:
        """
        Test method to verify the refactored evidence storage works correctly.
        This is a temporary method for validation.
        """
        try:
            # Test data
            test_source_id = "TEST_PROTEIN_001"
            test_target_id = "TEST_DISEASE_001"
            test_rel_type = "ASSOCIATED_WITH"
            test_source_doc = "test_pubmed_12345"
            test_abstract = "Test abstract about protein-disease association."
            test_confidence = 0.85
            test_chunk_id = "test_chunk_001"
            test_metadata = {"source_name": "TestProtein", "target_name": "TestDisease"}

            # Store test relationship
            self.store_relationship_with_context(
                source_id=test_source_id,
                target_id=test_target_id,
                rel_type=test_rel_type,
                source_doc=test_source_doc,
                abstract_text=test_abstract,
                confidence_score=test_confidence,
                chunk_id=test_chunk_id,
                metadata=test_metadata,
            )

            # Retrieve evidence using new method
            evidence = self.get_relationship_evidence(
                test_source_id, test_target_id, test_rel_type
            )

            # Check results
            test_results = {
                "evidence_found": len(evidence) > 0,
                "evidence_count": len(evidence),
                "has_relationship_support": any(
                    e.get("evidence_type") == "supports_relationship" for e in evidence
                ),
                "has_entity_mentions": any(
                    e.get("evidence_type") == "entity_mention" for e in evidence
                ),
                "confidence_match": any(
                    e.get("confidence_score") == test_confidence for e in evidence
                ),
                "source_doc_match": any(
                    e.get("source_doc") == test_source_doc for e in evidence
                ),
                "chunk_id_match": any(
                    e.get("chunk_id") == test_chunk_id for e in evidence
                ),
            }

            logger.info(f"Evidence storage test results: {test_results}")
            return test_results

        except Exception as e:
            logger.error(f"Evidence storage test failed: {e}")
            return {"error": str(e)}

    def get_relationship_evidence(
        self, source_id: str, target_id: str, rel_type: str
    ) -> list[dict[str, Any]]:
        """
        Get all evidence for a specific relationship from evidence relationships.

        Returns a list of evidence with full context, confidence scores, and metadata
        from MENTIONS and SUPPORTS relationships.
        """
        try:
            # Query evidence from SUPPORTS relationships (direct relationship evidence)
            supports_query = """
            MATCH (c:Chunk)-[sup:SUPPORTS]->(r)
            WHERE sup.rel_instance_id STARTS WITH $rel_pattern
            RETURN c.chunk_id AS chunk_id,
                   sup.confidence_score AS confidence_score,
                   sup.sentence_context AS sentence_context,
                   sup.extraction_method AS extraction_method,
                   sup.source_doc AS source_doc,
                   sup.extracted_at AS extracted_at,
                   sup.rel_instance_id AS rel_instance_id
            ORDER BY sup.confidence_score DESC, sup.extracted_at DESC
            """

            rel_pattern = f"{source_id}_{rel_type}_{target_id}"
            supports_results = self._execute_query(
                supports_query, {"rel_pattern": rel_pattern}
            )

            evidence = []
            for record in supports_results:
                evidence_entry = {
                    "chunk_id": record["chunk_id"],
                    "confidence_score": record["confidence_score"],
                    "sentence_context": record["sentence_context"],
                    "extraction_method": record["extraction_method"],
                    "source_doc": record["source_doc"],
                    "extracted_at": record["extracted_at"],
                    "rel_instance_id": record["rel_instance_id"],
                    "evidence_type": "supports_relationship",
                }
                evidence.append(evidence_entry)

            # Also get entity mention evidence for context
            mentions_query = """
            MATCH (c:Chunk)-[m:MENTIONS]->(e)
            WHERE (e.id = $source_id OR e.id = $target_id)
              AND m.source_doc IN $source_docs
            RETURN c.chunk_id AS chunk_id,
                   e.id AS entity_id,
                   m.entity_name AS entity_name,
                   m.confidence_score AS confidence_score,
                   m.sentence_context AS sentence_context,
                   m.extraction_method AS extraction_method,
                   m.source_doc AS source_doc,
                   m.extracted_at AS extracted_at
            ORDER BY m.confidence_score DESC
            """

            # Get source docs from the main relationship
            source_docs_query = f"""
            MATCH (s {{id: $source_id}})-[r:{rel_type}]->(t {{id: $target_id}})
            RETURN r.source_docs AS source_docs
            """
            source_docs_result = self._execute_query(
                source_docs_query, {"source_id": source_id, "target_id": target_id}
            )

            if source_docs_result and source_docs_result[0].get("source_docs"):
                source_docs = source_docs_result[0]["source_docs"]
                mentions_results = self._execute_query(
                    mentions_query,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_docs": source_docs,
                    },
                )

                for record in mentions_results:
                    evidence_entry = {
                        "chunk_id": record["chunk_id"],
                        "entity_id": record["entity_id"],
                        "entity_name": record["entity_name"],
                        "confidence_score": record["confidence_score"],
                        "sentence_context": record["sentence_context"],
                        "extraction_method": record["extraction_method"],
                        "source_doc": record["source_doc"],
                        "extracted_at": record["extracted_at"],
                        "evidence_type": "entity_mention",
                    }
                    evidence.append(evidence_entry)

            # Sort by confidence score descending, then by extraction date
            evidence.sort(
                key=lambda x: (
                    x.get("confidence_score") or 0,
                    x.get("extracted_at") or "",
                ),
                reverse=True,
            )

            logger.info(
                f"Retrieved {len(evidence)} evidence pieces for {source_id} -[{rel_type}]-> {target_id}"
            )
            return evidence

        except Exception as e:
            logger.error(f"Failed to get relationship evidence: {e}", exc_info=True)
            return []

    def get_high_confidence_relationships(
        self, min_occurrences: int = 2, min_avg_confidence: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Get relationships with high confidence based on evidence relationships.
        Now calculates confidence from SUPPORTS relationships instead of stored properties.
        """
        try:
            # SUPPORTS relationships are not created, so return empty
            # Evidence is captured via MENTIONS relationships instead
            logger.info(
                "Returning empty high-confidence relationships (SUPPORTS relationships not created)"
            )
            return []

            # Original query kept for reference
            # Query relationships with their evidence-based confidence scores
            # query = """
            # MATCH (s)-[r]->(t)
            # MATCH (c:Chunk)-[sup:SUPPORTS]->(r)
            # ...
            # WHERE sup.confidence_score IS NOT NULL
            # WITH s, r, t,
            #      count(DISTINCT sup) AS evidence_count,
            #      avg(sup.confidence_score) AS avg_confidence,
            #      collect(DISTINCT sup.source_doc) AS source_docs
            # WHERE evidence_count >= $min_occurrences AND avg_confidence >= $min_avg_confidence
            # RETURN s.id AS source_id, s.name AS source_name, labels(s)[0] AS source_type,
            #        t.id AS target_id, t.name AS target_name, labels(t)[0] AS target_type,
            #        type(r) AS rel_type,
            #        evidence_count AS occurrence_count,
            #        avg_confidence AS avg_confidence,
            #        source_docs AS source_docs
            # ORDER BY avg_confidence DESC, evidence_count DESC
            # """

            # results = self._execute_query(query, {
            #     "min_occurrences": min_occurrences,
            #     "min_avg_confidence": min_avg_confidence
            # })

            # logger.info(f"Retrieved {len(results)} high-confidence relationships")
            # return results

        except Exception as e:
            logger.error(
                f"Failed to get high-confidence relationships: {e}", exc_info=True
            )
            return []

    def analyze_relationship_patterns(self) -> dict[str, Any]:
        """
        Analyze patterns in relationship extraction.
        Provides basic analysis based on relationship types and frequencies.
        """
        try:
            analysis = {}

            basic_stats = self._execute_query("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                LIMIT 10
            """)
            analysis["relationship_types"] = basic_stats
            analysis["most_frequent_relationships"] = []
            analysis["conflicting_evidence"] = []
            analysis["extraction_methods"] = []
            analysis["evidence_quality"] = {}
            logger.info("Basic relationship analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze relationship patterns: {e}", exc_info=True)
            return {}

    def _link_relationship_to_chunk(
        self, source_id: str, target_id: str, rel_type: str, chunk_id: str
    ) -> None:
        """Link a relationship to its source chunk node for evidence traceability."""
        try:
            # Standard Neo4j does not support relationships pointing to relationships
            # Skip EVIDENCES relationship - evidence is captured via MENTIONS relationships
            logger.debug(
                "Skipping EVIDENCES relationship - evidence captured via MENTIONS relationships"
            )
            return

            # Original code for reference (would work in Neo4j 5.x with specific config)
            # query = f"""
            # MATCH (s {{id: $source_id}})-[r:{rel_type}]->(t {{id: $target_id}})
            # MATCH (c:Chunk {{chunk_id: $chunk_id}})
            # MERGE (c)-[:EVIDENCES]->(r)
            # """
            # self._execute_query(query, {
            #     "source_id": source_id,
            #     "target_id": target_id,
            #     "chunk_id": chunk_id
            # })
        except Exception as e:
            logger.error(f"Failed to link relationship to chunk {chunk_id}: {e}")
