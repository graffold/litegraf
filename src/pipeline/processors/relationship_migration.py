#!/usr/bin/env python3
"""
Relationship Migration Utility

Migrates from RelationshipOccurrence nodes to direct relationship edge annotation
and AbstractDocument nodes.
"""

import logging
import json
from typing import Any

from pipeline.interfaces import GraphStore
logger = logging.getLogger(__name__)
class RelationshipMigration:
    """
    Utility to migrate RelationshipOccurrence nodes to the new architecture:
    - Direct evidence storage on relationship edges
    - AbstractDocument nodes for full text and embeddings
    """

    def __init__(self, database: str = "neo4j", dry_run: bool = True, graph_store: GraphStore | None = None):
        self.database = database
        self.dry_run = dry_run
        if graph_store is not None:
            self.db = graph_store
        else:
            from pipeline.backends.neo4j_store import Neo4jGraphStore
            self.db = Neo4jGraphStore(database=database)

        logger.info(
            f"Initialized RelationshipMigration for database: {database} (dry_run: {dry_run})"
        )

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query against the graph store."""
        if self.dry_run:
            logger.info(f"DRY RUN - Would execute query: {query[:100]}...")
            return []

        return self.db.execute_query(query, parameters)

    def analyze_current_structure(self) -> dict[str, Any]:
        """
        Analyze the current RelationshipOccurrence structure to understand migration scope.
        """
        try:
            analysis = {}

            # Count RelationshipOccurrence nodes
            occ_count_query = """
                MATCH (occ:RelationshipOccurrence)
                RETURN count(occ) AS total_occurrences
                """
            occ_results = self._execute_query(occ_count_query)
            analysis["total_occurrence_nodes"] = (
                occ_results[0]["total_occurrences"] if occ_results else 0
            )

            # Count relationships with occurrences
            rel_with_occ_query = """
                MATCH ()-[r]-(:RelationshipOccurrence)
                WHERE type(r) = 'HAS_OCCURRENCE'
                RETURN count(DISTINCT r) AS relationships_with_occurrences
                """
            rel_results = self._execute_query(rel_with_occ_query)
            analysis["relationships_with_occurrences"] = (
                rel_results[0]["relationships_with_occurrences"] if rel_results else 0
            )

            # Get unique source documents
            source_docs_query = """
                MATCH (occ:RelationshipOccurrence)
                WHERE occ.source_doc IS NOT NULL
                RETURN count(DISTINCT occ.source_doc) AS unique_source_docs
                """
            docs_results = self._execute_query(source_docs_query)
            analysis["unique_source_documents"] = (
                docs_results[0]["unique_source_docs"] if docs_results else 0
            )

            # Get relationship types with occurrences
            rel_types_query = """
                MATCH ()-[r:HAS_OCCURRENCE]->(occ:RelationshipOccurrence)
                MATCH (s)-[main_rel]->(t)
                WHERE (main_rel)-[:HAS_OCCURRENCE]->(occ)
                RETURN type(main_rel) AS rel_type, count(*) AS occurrence_count
                ORDER BY occurrence_count DESC
                """
            types_results = self._execute_query(rel_types_query)
            analysis["relationship_types"] = types_results

            logger.info(f"Current structure analysis: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze current structure: {e}", exc_info=True)
            return {}

    def migrate_to_new_structure(self) -> dict[str, Any]:
        """
        Migrate RelationshipOccurrence nodes to the new architecture.
        """
        try:
            migration_stats = {
                "abstract_documents_created": 0,
                "relationships_updated": 0,
                "occurrence_nodes_processed": 0,
                "errors": [],
            }

            # Step 1: Create AbstractDocument nodes
            logger.info("Step 1: Creating AbstractDocument nodes...")
            self._create_abstract_documents(migration_stats)

            # Step 2: Migrate relationship evidence
            logger.info("Step 2: Migrating relationship evidence...")
            self._migrate_relationship_evidence(migration_stats)

            # Step 3: Clean up old structure (only if not dry run)
            if not self.dry_run:
                logger.info("Step 3: Cleaning up old RelationshipOccurrence nodes...")
                self._cleanup_old_structure(migration_stats)
            else:
                logger.info("Step 3: Skipping cleanup in dry run mode")

            logger.info(f"Migration completed: {migration_stats}")
            return migration_stats

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _create_abstract_documents(self, stats: dict[str, Any]) -> None:
        """Create AbstractDocument nodes from RelationshipOccurrence abstract texts."""
        try:
            # Get unique abstract texts from occurrence nodes
            abstracts_query = """
                MATCH (occ:RelationshipOccurrence)
                WHERE occ.source_doc IS NOT NULL AND occ.abstract_text IS NOT NULL
                WITH DISTINCT occ.source_doc AS doc_id, occ.abstract_text AS text
                RETURN doc_id, text
                """

            abstracts = self._execute_query(abstracts_query)

            for abstract in abstracts:
                doc_id = abstract["doc_id"]
                text = abstract["text"]

                if not self.dry_run:
                    create_doc_query = """
                        MERGE (doc:AbstractDocument {id: $doc_id})
                        ON CREATE SET
                            doc.text = $text,
                            doc.created_at = datetime(),
                            doc.text_length = size($text),
                            doc.migrated_from_occurrence = true
                        """

                    self._execute_query(
                        create_doc_query, {"doc_id": doc_id, "text": text}
                    )

                stats["abstract_documents_created"] += 1

            logger.info(
                f"Created {stats['abstract_documents_created']} AbstractDocument nodes"
            )

        except Exception as e:
            error_msg = f"Failed to create AbstractDocument nodes: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    def _migrate_relationship_evidence(self, stats: dict[str, Any]) -> None:
        """Migrate evidence from RelationshipOccurrence nodes to relationship edges."""
        try:
            # Get all relationships with their occurrences
            relationships_query = """
                MATCH (s)-[main_rel]->(t)-[has_occ:HAS_OCCURRENCE]->(occ:RelationshipOccurrence)
                RETURN
                    s.id AS source_id,
                    t.id AS target_id,
                    type(main_rel) AS rel_type,
                    collect({
                        source_doc: occ.source_doc,
                        sentence_context: occ.sentence_context,
                        confidence_score: coalesce(occ.confidence_score, 0.5),
                        extraction_method: coalesce(occ.extraction_method, 'unknown'),
                        metadata: coalesce(occ.metadata, '{}'),
                        extracted_at: coalesce(occ.extracted_at, datetime())
                    }) AS evidence_list
                """

            relationships = self._execute_query(relationships_query)

            for rel_data in relationships:
                source_id = rel_data["source_id"]
                target_id = rel_data["target_id"]
                rel_type = rel_data["rel_type"]
                evidence_list = rel_data["evidence_list"]

                if not evidence_list:
                    continue

                # Prepare evidence arrays
                evidence_sources = []
                evidence_details = []
                confidence_scores = []
                extraction_methods = []

                for evidence in evidence_list:
                    evidence_sources.append(evidence["source_doc"])
                    confidence_scores.append(evidence["confidence_score"])
                    extraction_methods.append(evidence["extraction_method"])

                    # Create evidence detail JSON
                    detail = {
                        "source_doc": evidence["source_doc"],
                        "sentence_context": evidence["sentence_context"],
                        "confidence_score": evidence["confidence_score"],
                        "extraction_method": evidence["extraction_method"],
                        "metadata": json.loads(evidence["metadata"])
                        if evidence["metadata"] != "{}"
                        else {},
                        "extracted_at": str(evidence["extracted_at"]),
                    }
                    evidence_details.append(json.dumps(detail))

                # Calculate average confidence
                avg_confidence = (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0.5
                )

                if not self.dry_run:
                    # Update relationship with evidence
                    update_rel_query = f"""
                        MATCH (s {{id: $source_id}})-[r:{rel_type}]->(t {{id: $target_id}})
                        SET r.evidence_sources = $evidence_sources,
                            r.evidence_details = $evidence_details,
                            r.confidence_scores = $confidence_scores,
                            r.extraction_methods = $extraction_methods,
                            r.avg_confidence = $avg_confidence,
                            r.evidence_count = $evidence_count,
                            r.migrated_from_occurrence = true,
                            r.migration_date = datetime()
                        """

                    self._execute_query(
                        update_rel_query,
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "evidence_sources": evidence_sources,
                            "evidence_details": evidence_details,
                            "confidence_scores": confidence_scores,
                            "extraction_methods": extraction_methods,
                            "avg_confidence": avg_confidence,
                            "evidence_count": len(evidence_sources),
                        },
                    )

                    # Link to AbstractDocument nodes
                    for source_doc in set(evidence_sources):
                        link_query = f"""
                            MATCH (s {{id: $source_id}})-[r:{rel_type}]->(t {{id: $target_id}})
                            MATCH (doc:AbstractDocument {{id: $source_doc}})
                            MERGE (r)-[:SUPPORTED_BY]->(doc)
                            """

                        self._execute_query(
                            link_query,
                            {
                                "source_id": source_id,
                                "target_id": target_id,
                                "source_doc": source_doc,
                            },
                        )

                stats["relationships_updated"] += 1

            logger.info(
                f"Updated {stats['relationships_updated']} relationships with evidence"
            )

        except Exception as e:
            error_msg = f"Failed to migrate relationship evidence: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    def _cleanup_old_structure(self, stats: dict[str, Any]) -> None:
        """Remove RelationshipOccurrence nodes and HAS_OCCURRENCE relationships."""
        try:
            # Count occurrences before deletion
            count_query = """
                MATCH (occ:RelationshipOccurrence)
                RETURN count(occ) AS total_occurrences
                """
            count_result = self._execute_query(count_query)
            total_occurrences = (
                count_result[0]["total_occurrences"] if count_result else 0
            )

            # Delete HAS_OCCURRENCE relationships first
            delete_has_occ_query = """
                MATCH ()-[r:HAS_OCCURRENCE]->()
                DELETE r
                """
            self._execute_query(delete_has_occ_query)

            # Delete RelationshipOccurrence nodes
            delete_occ_query = """
                MATCH (occ:RelationshipOccurrence)
                DELETE occ
                """
            self._execute_query(delete_occ_query)

            stats["occurrence_nodes_processed"] = total_occurrences
            logger.info(f"Cleaned up {total_occurrences} RelationshipOccurrence nodes")

        except Exception as e:
            error_msg = f"Failed to cleanup old structure: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    def verify_migration(self) -> dict[str, Any]:
        """Verify that the migration was successful."""
        try:
            verification = {}

            # Check that no RelationshipOccurrence nodes remain
            remaining_occ_query = """
                MATCH (occ:RelationshipOccurrence)
                RETURN count(occ) AS remaining_occurrences
                """
            remaining_result = self._execute_query(remaining_occ_query)
            verification["remaining_occurrence_nodes"] = (
                remaining_result[0]["remaining_occurrences"] if remaining_result else 0
            )

            # Check AbstractDocument nodes created
            doc_count_query = """
                MATCH (doc:AbstractDocument)
                RETURN count(doc) AS total_documents
                """
            doc_result = self._execute_query(doc_count_query)
            verification["abstract_documents"] = (
                doc_result[0]["total_documents"] if doc_result else 0
            )

            # Check relationships with evidence
            rel_evidence_query = """
                MATCH ()-[r]->()
                WHERE r.evidence_sources IS NOT NULL
                RETURN count(r) AS relationships_with_evidence
                """
            rel_result = self._execute_query(rel_evidence_query)
            verification["relationships_with_evidence"] = (
                rel_result[0]["relationships_with_evidence"] if rel_result else 0
            )

            # Check SUPPORTED_BY relationships
            supported_by_query = """
                MATCH ()-[r:SUPPORTED_BY]->(:AbstractDocument)
                RETURN count(r) AS supported_by_relationships
                """
            supported_result = self._execute_query(supported_by_query)
            verification["supported_by_relationships"] = (
                supported_result[0]["supported_by_relationships"]
                if supported_result
                else 0
            )

            logger.info(f"Migration verification: {verification}")
            return verification

        except Exception as e:
            logger.error(f"Failed to verify migration: {e}", exc_info=True)
            return {"error": str(e)}

    def close(self) -> None:
        """Close database connections."""
        if self.db:
            self.db.close()
        logger.info("RelationshipMigration connections closed")


def main() -> None:
    """Command line interface for relationship migration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate RelationshipOccurrence nodes to new architecture"
    )
    parser.add_argument("--database", default="neo4j", help="Database name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Perform dry run without making changes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the migration (overrides --dry-run)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze current structure, don't migrate",
    )

    args = parser.parse_args()

    # Override dry_run if --execute is specified
    dry_run = not args.execute if args.execute else args.dry_run

    try:
        from pipeline.backends.neo4j_store import Neo4jGraphStore
        migrator = RelationshipMigration(database=args.database, dry_run=dry_run, graph_store=Neo4jGraphStore(database=args.database))

        print("🔍 Analyzing current RelationshipOccurrence structure...")
        analysis = migrator.analyze_current_structure()

        print("\n📊 Current Structure Analysis:")
        print(
            f"  - Total RelationshipOccurrence nodes: {analysis.get('total_occurrence_nodes', 0)}"
        )
        print(
            f"  - Relationships with occurrences: {analysis.get('relationships_with_occurrences', 0)}"
        )
        print(
            f"  - Unique source documents: {analysis.get('unique_source_documents', 0)}"
        )

        if analysis.get("relationship_types"):
            print("  - Relationship types with occurrences:")
            for rel_type in analysis["relationship_types"][:5]:  # Show top 5
                print(
                    f"    * {rel_type['rel_type']}: {rel_type['occurrence_count']} occurrences"
                )

        if args.analyze_only:
            print("\n✅ Analysis complete (analyze-only mode)")
            return

        if analysis.get("total_occurrence_nodes", 0) == 0:
            print("\n✅ No RelationshipOccurrence nodes found - migration not needed")
            return

        print(f"\n🔄 Starting migration (dry_run={dry_run})...")
        migration_result = migrator.migrate_to_new_structure()

        print("\n📈 Migration Results:")
        print(
            f"  - AbstractDocument nodes created: {migration_result.get('abstract_documents_created', 0)}"
        )
        print(
            f"  - Relationships updated: {migration_result.get('relationships_updated', 0)}"
        )
        print(
            f"  - Occurrence nodes processed: {migration_result.get('occurrence_nodes_processed', 0)}"
        )

        if migration_result.get("errors"):
            print(f"  - Errors: {len(migration_result['errors'])}")
            for error in migration_result["errors"]:
                print(f"    * {error}")

        if not dry_run:
            print("\n🔍 Verifying migration...")
            verification = migrator.verify_migration()
            print(
                f"  - Remaining occurrence nodes: {verification.get('remaining_occurrence_nodes', 0)}"
            )
            print(
                f"  - AbstractDocument nodes: {verification.get('abstract_documents', 0)}"
            )
            print(
                f"  - Relationships with evidence: {verification.get('relationships_with_evidence', 0)}"
            )
            print(
                f"  - SUPPORTED_BY relationships: {verification.get('supported_by_relationships', 0)}"
            )

        migrator.close()

        print(
            f"\n✅ Migration {'completed' if not dry_run else 'simulation completed'}"
        )

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
