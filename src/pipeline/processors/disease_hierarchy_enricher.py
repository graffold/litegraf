#!/usr/bin/env python3
"""
Disease Hierarchy Enricher

Enriches disease nodes with hierarchical relationships from MONDO ontology.
Creates parent-child relationships for generic querying (e.g., "heart disease" includes "atrial fibrillation").
"""

import logging
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pipeline.interfaces import GraphStore
# Optional dependencies
try:
    import networkx as nx
    import obonet

    HAS_ONTOLOGY_DEPS = True
except ImportError:
    obonet = None
    nx = None
    HAS_ONTOLOGY_DEPS = False

logger = logging.getLogger(__name__)
class DiseaseHierarchyEnricher:
    """
    Enriches disease nodes with hierarchical relationships from MONDO ontology.
    Creates parent-child relationships for generic querying (e.g., "heart disease" includes "atrial fibrillation").
    """

    def __init__(
        self,
        database: str = "cvd1",
        obo_path: str | None = None,
        db: GraphStore | None = None,
    ):
        self.database = database
        if db is not None:
            self.db = db
        else:
            from pipeline.backends.neo4j_store import Neo4jGraphStore
            self.db = Neo4jGraphStore(database=database)

        # Default to mondo.obo in utils directory
        if obo_path is None:
            current_dir = Path(__file__).parent
            obo_path = str(current_dir / ".." / "utils" / "mondo.obo")

        self.obo_path = str(obo_path)
        self.mondo_graph: Any | None = None
        self.term_to_id: dict[str, str] = {}  # Map disease names to MONDO IDs
        self.id_to_term: dict[str, str] = {}  # Map MONDO IDs to primary names
        self.synonyms: dict[str, str] = {}  # Map synonyms to MONDO IDs
        self.hierarchy: dict[str, Any] = {}  # Store parent-child relationships

        # Load MONDO ontology
        self._load_mondo_ontology()

        logger.info(f"Initialized DiseaseHierarchyEnricher for database: {database}")

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against Neo4j."""
        return self.db.execute_query(query, parameters)

    def _load_mondo_ontology(self):
        """Load MONDO ontology and build mappings."""
        if not HAS_ONTOLOGY_DEPS:
            logger.warning(
                "obonet and networkx not available - disease hierarchy enrichment disabled"
            )
            return

        try:
            if not os.path.exists(self.obo_path):
                logger.warning(f"MONDO ontology file not found at {self.obo_path}")
                logger.warning("Disease hierarchy enrichment will be disabled")
                return

            logger.info(f"Loading MONDO ontology from {self.obo_path}")
            self.mondo_graph = obonet.read_obo(self.obo_path)  # type: ignore

            # Build term mappings
            for term_id, data in self.mondo_graph.nodes(data=True):  # type: ignore
                if not term_id.startswith("MONDO:"):
                    continue

                # Get primary name
                name = data.get("name", "")
                if name:
                    self.term_to_id[name.lower()] = term_id
                    self.id_to_term[term_id] = name

                # Get synonyms
                synonyms = data.get("synonym", [])
                for synonym in synonyms:
                    # Extract synonym text (remove quotes and type info)
                    if isinstance(synonym, str):
                        syn_text = synonym.split('"')[1] if '"' in synonym else synonym
                        self.synonyms[syn_text.lower()] = term_id
                        self.term_to_id[syn_text.lower()] = term_id

            logger.info(
                f"Loaded {len(self.id_to_term)} MONDO terms with {len(self.synonyms)} synonyms"
            )

        except Exception as e:
            logger.error(f"Failed to load MONDO ontology: {e}")
            logger.warning("Disease hierarchy enrichment will be disabled")
            self.mondo_graph = None

    def _normalize_disease_name(self, name: str) -> str:
        """Normalize disease name for matching."""
        if not name:
            return ""

        # Convert to lowercase and remove common suffixes/prefixes
        normalized = name.lower().strip()

        # Remove common parenthetical information
        normalized = re.sub(r"\s*\([^)]*\)", "", normalized)

        # Remove common abbreviations and suffixes
        normalized = re.sub(
            r"\s+(disease|syndrome|disorder|condition|cancer|tumor|carcinoma)s?$",
            "",
            normalized,
        )

        # Normalize spaces and punctuation
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _find_mondo_match(self, disease_name: str) -> str | None:
        """Find MONDO ID for a disease name using various matching strategies."""
        if not disease_name or not self.mondo_graph:
            return None

        # Strategy 1: Exact match
        normalized = self._normalize_disease_name(disease_name)
        if normalized in self.term_to_id:
            return self.term_to_id[normalized]

        # Strategy 2: Original name exact match
        if disease_name.lower() in self.term_to_id:
            return self.term_to_id[disease_name.lower()]

        # Strategy 3: Synonym match
        if normalized in self.synonyms:
            return self.synonyms[normalized]

        # Strategy 4: Optimized fuzzy matching (limit search scope)
        best_match = None
        best_score = 0.85  # Minimum similarity threshold

        # Extract key words to limit search scope
        disease_words = set(normalized.split())

        # Check against primary terms (only those with word overlap)
        candidates_checked = 0
        max_candidates = 1000  # Limit to avoid performance issues

        for term, mondo_id in self.term_to_id.items():
            term_words = set(term.split())
            # Only check if there's word overlap or terms are short
            if disease_words & term_words or len(term.split()) <= 2:
                similarity = SequenceMatcher(None, normalized, term).ratio()
                if similarity > best_score:
                    best_score = similarity
                    best_match = mondo_id
                candidates_checked += 1
                if candidates_checked >= max_candidates:
                    break

        # Check against synonyms (limited scope)
        if not best_match or best_score < 0.9:  # Only if no good match found
            candidates_checked = 0
            for synonym, mondo_id in self.synonyms.items():
                synonym_words = set(synonym.split())
                if disease_words & synonym_words or len(synonym.split()) <= 2:
                    similarity = SequenceMatcher(None, normalized, synonym).ratio()
                    if similarity > best_score:
                        best_score = similarity
                        best_match = mondo_id
                    candidates_checked += 1
                    if candidates_checked >= max_candidates:
                        break

        if best_match:
            logger.debug(
                f"Fuzzy matched '{disease_name}' to MONDO ID {best_match} (score: {best_score:.3f})"
            )

        return best_match

    def _get_parents(self, mondo_id: str) -> list[tuple[str, str]]:
        """Get parent terms for a MONDO ID."""
        parents = []

        if self.mondo_graph and mondo_id in self.mondo_graph:
            # Get direct parents via 'is_a' relationships
            for parent_id in self.mondo_graph.predecessors(mondo_id):  # type: ignore
                if parent_id.startswith("MONDO:") and parent_id in self.id_to_term:
                    parent_name = self.id_to_term[parent_id]
                    parents.append((parent_id, parent_name))

        return parents

    def _get_all_ancestors(
        self, mondo_id: str, max_depth: int = 5
    ) -> list[tuple[str, str, int]]:
        """Get all ancestor terms for a MONDO ID with depth information."""
        ancestors: list[tuple[str, str, int]] = []

        if not self.mondo_graph:
            return ancestors

        visited = set()
        queue = [(mondo_id, 0)]  # (id, depth)

        while queue and len(ancestors) < 50:  # Limit to prevent infinite loops
            current_id, depth = queue.pop(0)

            if current_id in visited or depth >= max_depth:
                continue

            visited.add(current_id)

            # Get direct parents
            for parent_id in self.mondo_graph.predecessors(current_id):  # type: ignore
                if (
                    parent_id.startswith("MONDO:")
                    and parent_id in self.id_to_term
                    and parent_id not in visited
                ):
                    parent_name = self.id_to_term[parent_id]
                    ancestors.append((parent_id, parent_name, depth + 1))
                    queue.append((parent_id, depth + 1))

        return ancestors

    def consolidate_extracted_diseases_with_hierarchy(
        self, dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Consolidate disease nodes extracted from abstracts with MONDO hierarchy nodes.
        This ensures relationships from abstracts connect to hierarchical disease nodes.
        """
        logger.info(
            f"Consolidating extracted diseases with MONDO hierarchy (dry_run={dry_run})"
        )

        stats: dict[str, Any] = {
            "extracted_diseases_processed": 0,
            "diseases_consolidated": 0,
            "relationships_transferred": 0,
            "nodes_merged": 0,
            "consolidation_examples": [],
        }

        # Find extracted disease nodes that could match hierarchy nodes
        # Use more precise matching strategies to avoid false matches
        result = self._execute_query("""
            MATCH (extracted:Disease)
            WHERE extracted.mondo_id IS NULL
              AND extracted.source IS NULL
              AND extracted.name IS NOT NULL
              AND trim(extracted.name) <> ''
            MATCH (hierarchy:Disease)
            WHERE hierarchy.mondo_id IS NOT NULL
              AND hierarchy.name IS NOT NULL
              AND (
                // Exact match (highest priority)
                toLower(extracted.name) = toLower(hierarchy.name)
                OR
                // Match without spaces and special characters (for formatting differences)
                toLower(replace(replace(replace(extracted.name, ' ', ''), '(', ''), ')', '')) =
                toLower(replace(replace(replace(hierarchy.name, ' ', ''), '(', ''), ')', ''))
                OR
                // Specific disease type matches (more precise than keyword matching)
                (
                  // Cardiovascular diseases
                  (toLower(extracted.name) CONTAINS 'myocardial infarction' AND toLower(hierarchy.name) CONTAINS 'myocardial infarction') OR
                  (toLower(extracted.name) CONTAINS 'heart failure' AND toLower(hierarchy.name) CONTAINS 'heart failure') OR
                  (toLower(extracted.name) CONTAINS 'coronary artery disease' AND toLower(hierarchy.name) CONTAINS 'coronary artery disease') OR
                  // Diabetes
                  (toLower(extracted.name) CONTAINS 'diabetes mellitus' AND toLower(hierarchy.name) CONTAINS 'diabetes mellitus') OR
                  (toLower(extracted.name) CONTAINS 'type 1 diabetes' AND toLower(hierarchy.name) CONTAINS 'type 1 diabetes') OR
                  (toLower(extracted.name) CONTAINS 'type 2 diabetes' AND toLower(hierarchy.name) CONTAINS 'type 2 diabetes') OR
                  // Hypertension
                  (toLower(extracted.name) CONTAINS 'hypertension' AND toLower(hierarchy.name) CONTAINS 'hypertension' AND
                   size(split(toLower(extracted.name), ' ')) <= 3 AND size(split(toLower(hierarchy.name), ' ')) <= 3) OR
                  // Cancer types (require more specific matching)
                  (toLower(extracted.name) CONTAINS 'breast cancer' AND toLower(hierarchy.name) CONTAINS 'breast cancer') OR
                  (toLower(extracted.name) CONTAINS 'lung cancer' AND toLower(hierarchy.name) CONTAINS 'lung cancer') OR
                  (toLower(extracted.name) CONTAINS 'prostate cancer' AND toLower(hierarchy.name) CONTAINS 'prostate cancer') OR
                  // Neurological diseases
                  (toLower(extracted.name) CONTAINS 'alzheimer' AND toLower(hierarchy.name) CONTAINS 'alzheimer') OR
                  (toLower(extracted.name) CONTAINS 'parkinson' AND toLower(hierarchy.name) CONTAINS 'parkinson')
                )
              )

            // Count relationships that would be transferred
            OPTIONAL MATCH (extracted)-[out_rel]->(target)
            OPTIONAL MATCH (source)-[in_rel]->(extracted)

            RETURN extracted.id AS extracted_id,
                   extracted.name AS extracted_name,
                   hierarchy.id AS hierarchy_id,
                   hierarchy.name AS hierarchy_name,
                   hierarchy.mondo_id AS mondo_id,
                   count(DISTINCT out_rel) + count(DISTINCT in_rel) AS relationship_count
            ORDER BY relationship_count DESC
        """)

        for record in result:
            stats["extracted_diseases_processed"] += 1

            consolidation_info = {
                "extracted_name": record["extracted_name"],
                "hierarchy_name": record["hierarchy_name"],
                "mondo_id": record["mondo_id"],
                "relationships_count": record["relationship_count"],
            }
            stats["consolidation_examples"].append(consolidation_info)

            if not dry_run and record["relationship_count"] > 0:
                # Transfer relationships from extracted node to hierarchy node in steps

                # Step 1: Get all relationships to transfer
                rel_data = self._execute_query(
                    """
                        MATCH (extracted:Disease {id: $extracted_id})

                        // Get outgoing relationships
                        OPTIONAL MATCH (extracted)-[out_rel]->(target)
                        WITH extracted, collect({target_id: target.id, rel_type: type(out_rel), props: properties(out_rel)}) AS outgoing_rels

                        // Get incoming relationships
                        OPTIONAL MATCH (source)-[in_rel]->(extracted)
                        WITH extracted, outgoing_rels, collect({source_id: source.id, rel_type: type(in_rel), props: properties(in_rel)}) AS incoming_rels

                        RETURN outgoing_rels, incoming_rels,
                               size(outgoing_rels) + size(incoming_rels) AS total_rels
                    """,
                    {"extracted_id": record["extracted_id"]},
                )

                if not rel_data or len(rel_data) == 0:
                    continue

                rel_record = rel_data[0]

                total_transferred = 0
                outgoing_rels = rel_record.get("outgoing_rels", [])
                incoming_rels = rel_record.get("incoming_rels", [])

                # Step 2: Transfer outgoing relationships
                if outgoing_rels:
                    for rel in outgoing_rels:
                        if rel.get("target_id"):
                            self._execute_query(
                                """
                                    MATCH (hierarchy:Disease {id: $hierarchy_id})
                                    MATCH (target {id: $target_id})
                                    MERGE (hierarchy)-[new_rel:ASSOCIATED_WITH]->(target)
                                    SET new_rel += $props
                                """,
                                {
                                    "hierarchy_id": record["hierarchy_id"],
                                    "target_id": rel["target_id"],
                                    "props": rel.get("props", {}),
                                },
                            )
                            total_transferred += 1

                # Step 3: Transfer incoming relationships
                if incoming_rels:
                    for rel in incoming_rels:
                        if rel.get("source_id"):
                            self._execute_query(
                                """
                                    MATCH (hierarchy:Disease {id: $hierarchy_id})
                                    MATCH (source {id: $source_id})
                                    MERGE (source)-[new_rel:ASSOCIATED_WITH]->(hierarchy)
                                    SET new_rel += $props
                                """,
                                {
                                    "hierarchy_id": record["hierarchy_id"],
                                    "source_id": rel["source_id"],
                                    "props": rel.get("props", {}),
                                },
                            )
                            total_transferred += 1

                # Step 4: Copy properties and delete extracted node
                self._execute_query(
                    """
                        MATCH (extracted:Disease {id: $extracted_id})
                        MATCH (hierarchy:Disease {id: $hierarchy_id})

                        // Copy useful properties
                        SET hierarchy.embedding = COALESCE(hierarchy.embedding, extracted.embedding),
                            hierarchy.full_text = COALESCE(hierarchy.full_text, extracted.full_text),
                            hierarchy.abstract_text = COALESCE(hierarchy.abstract_text, extracted.abstract_text)

                        // Delete the extracted node and its relationships
                        DETACH DELETE extracted
                    """,
                    {
                        "extracted_id": record["extracted_id"],
                        "hierarchy_id": record["hierarchy_id"],
                    },
                )

                stats["relationships_transferred"] += total_transferred
                stats["nodes_merged"] += 1
                logger.debug(
                    f"Consolidated '{record['extracted_name']}' -> '{record['hierarchy_name']}' "
                    f"({total_transferred} relationships transferred)"
                )

            stats["diseases_consolidated"] += 1

        logger.info("Disease consolidation complete:")
        logger.info(
            f"  - Extracted diseases processed: {stats['extracted_diseases_processed']}"
        )
        logger.info(f"  - Diseases consolidated: {stats['diseases_consolidated']}")
        if not dry_run:
            logger.info(f"  - Nodes merged: {stats['nodes_merged']}")
            logger.info(
                f"  - Relationships transferred: {stats['relationships_transferred']}"
            )

        return stats

    def get_diseases_without_hierarchy(self) -> list[dict]:
        """Get disease nodes that don't have hierarchical relationships."""
        result = self._execute_query("""
            MATCH (d:Disease)
            WHERE NOT (d)-[:IS_A]->(:Disease)
            RETURN d.id AS id, d.name AS name, d.mondo_id AS mondo_id
            ORDER BY d.name
        """)

        diseases = []
        for record in result:
            diseases.append(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "mondo_id": record.get("mondo_id"),
                }
            )

        logger.info(
            f"Found {len(diseases)} diseases without hierarchical relationships"
        )
        return diseases

    def enrich_disease_hierarchy(
        self, dry_run: bool = True, consolidate_extracted: bool = True
    ) -> dict[str, Any]:
        """Add hierarchical relationships to disease nodes and consolidate with extracted diseases."""
        logger.info(
            f"Starting disease hierarchy enrichment (dry_run={dry_run}, consolidate_extracted={consolidate_extracted})"
        )

        if not self.mondo_graph:
            logger.warning("MONDO ontology not loaded - skipping hierarchy enrichment")
            return {
                "diseases_processed": 0,
                "mondo_matches_found": 0,
                "parent_relationships_created": 0,
                "ancestor_relationships_created": 0,
                "new_parent_nodes_created": 0,
                "enriched_diseases": [],
                "extracted_diseases_processed": 0,
                "diseases_consolidated": 0,
                "relationships_transferred": 0,
                "nodes_merged": 0,
            }

        # First, consolidate extracted diseases with any existing hierarchy
        consolidation_stats = {}
        if consolidate_extracted:
            logger.info(
                "Step 1: Consolidating extracted diseases with existing hierarchy"
            )
            consolidation_stats = self.consolidate_extracted_diseases_with_hierarchy(
                dry_run=dry_run
            )

        # Then, do the normal hierarchy enrichment
        logger.info("Step 2: Enriching diseases with MONDO hierarchy")
        hierarchy_stats = self._enrich_disease_hierarchy_core(dry_run=dry_run)

        # Merge the statistics
        combined_stats = hierarchy_stats.copy()
        combined_stats.update(consolidation_stats)

        return combined_stats

    def _enrich_disease_hierarchy_core(self, dry_run: bool = True) -> dict[str, Any]:
        """Core hierarchy enrichment logic (separated for consolidation flow)."""

        stats: dict[str, Any] = {
            "diseases_processed": 0,
            "mondo_matches_found": 0,
            "parent_relationships_created": 0,
            "ancestor_relationships_created": 0,
            "new_parent_nodes_created": 0,
            "enriched_diseases": [],
        }

        # Get diseases without hierarchy
        diseases = self.get_diseases_without_hierarchy()

        for disease in diseases:
            try:
                stats["diseases_processed"] += 1
                disease_name = disease["name"]
                disease_id = disease["id"]

                # Skip if already has MONDO ID
                if disease.get("mondo_id"):
                    logger.debug(
                        f"Disease {disease_name} already has MONDO ID: {disease['mondo_id']}"
                    )
                    continue

                # Find MONDO match
                mondo_id = self._find_mondo_match(disease_name)
                if not mondo_id:
                    logger.debug(f"No MONDO match found for: {disease_name}")
                    continue

                stats["mondo_matches_found"] += 1
                logger.info(f"Found MONDO match for '{disease_name}': {mondo_id}")

                # Get all ancestors
                ancestors = self._get_all_ancestors(mondo_id)

                if not dry_run:
                    # Update disease with MONDO ID
                    self._execute_query(
                        """
                        MATCH (d:Disease {id: $disease_id})
                        SET d.mondo_id = $mondo_id
                    """,
                        {"disease_id": disease_id, "mondo_id": mondo_id},
                    )

                    # Create parent relationships
                    for parent_mondo_id, parent_name, depth in ancestors:
                        # Create or update parent disease node
                        parent_result = self._execute_query(
                            """
                            MERGE (parent:Disease:Entity {mondo_id: $parent_mondo_id})
                            ON CREATE SET parent.name = $parent_name,
                                        parent.id = $parent_mondo_id,
                                        parent.type = 'Disease',
                                        parent.source = 'MONDO_hierarchy'
                            RETURN parent.id AS parent_id,
                                   CASE WHEN parent.source = 'MONDO_hierarchy' THEN true ELSE false END AS newly_created
                        """,
                            {
                                "parent_mondo_id": parent_mondo_id,
                                "parent_name": parent_name,
                            },
                        )

                        if parent_result and len(parent_result) > 0:
                            parent_record = parent_result[0]
                            if parent_record and parent_record.get("newly_created"):
                                stats["new_parent_nodes_created"] += 1

                        # Create IS_A relationship
                        self._execute_query(
                            """
                            MATCH (d:Disease {id: $disease_id})
                            MATCH (parent:Disease {mondo_id: $parent_mondo_id})
                            MERGE (d)-[r:IS_A]->(parent)
                            SET r.depth = $depth, r.source = 'MONDO_hierarchy'
                        """,
                            {
                                "disease_id": disease_id,
                                "parent_mondo_id": parent_mondo_id,
                                "depth": depth,
                            },
                        )

                        if depth == 1:
                            stats["parent_relationships_created"] += 1
                        else:
                            stats["ancestor_relationships_created"] += 1

                        logger.debug(
                            f"Created IS_A relationship: {disease_name} -> {parent_name} (depth: {depth})"
                        )

                # Track enriched disease
                stats["enriched_diseases"].append(
                    {
                        "name": disease_name,
                        "mondo_id": mondo_id,
                        "ancestors_count": len(ancestors),
                        "direct_parents": len([a for a in ancestors if a[2] == 1]),
                    }
                )

                if stats["diseases_processed"] % 10 == 0:
                    logger.info(
                        f"Processed {stats['diseases_processed']}/{len(diseases)} diseases"
                    )

            except Exception as e:
                logger.error(f"Failed to process disease {disease['name']}: {e}")
                continue

        logger.info("Disease hierarchy enrichment complete:")
        logger.info(f"  - Diseases processed: {stats['diseases_processed']}")
        logger.info(f"  - MONDO matches found: {stats['mondo_matches_found']}")
        logger.info(
            f"  - Parent relationships created: {stats['parent_relationships_created']}"
        )
        logger.info(
            f"  - Ancestor relationships created: {stats['ancestor_relationships_created']}"
        )
        logger.info(
            f"  - New parent nodes created: {stats['new_parent_nodes_created']}"
        )

        return stats

    def query_disease_hierarchy(
        self, query_term: str, max_depth: int = 3
    ) -> list[dict[str, Any]]:
        """Query for diseases including hierarchical matches."""
        logger.info(
            f"Querying disease hierarchy for: '{query_term}' (max_depth: {max_depth})"
        )

        # First try exact match
        result = self._execute_query(
            """
            MATCH (d:Disease)
            WHERE toLower(d.name) = toLower($query_term)
            RETURN d.id AS id,
                   d.name AS name,
                   d.mondo_id AS mondo_id,
                   'exact' AS match_type,
                   0 AS hierarchy_depth
            ORDER BY d.name
        """,
            {"query_term": query_term},
        )

        diseases = []
        for record in result:
            diseases.append(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "mondo_id": record.get("mondo_id"),
                    "match_type": record["match_type"],
                    "hierarchy_depth": record["hierarchy_depth"],
                }
            )

        # If no exact matches, try hierarchical matches (children of the query term)
        if not diseases:
            result = self._execute_query(
                """
                MATCH (query:Disease)-[:IS_A*1..$max_depth]->(child:Disease)
                WHERE toLower(query.name) = toLower($query_term)
                RETURN DISTINCT child.id AS id,
                       child.name AS name,
                       child.mondo_id AS mondo_id,
                       'hierarchical_child' AS match_type,
                       length((query)-[:IS_A*]->(child)) AS hierarchy_depth
                ORDER BY hierarchy_depth, child.name
            """,
                {"query_term": query_term, "max_depth": max_depth},
            )

            for record in result:
                diseases.append(
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "mondo_id": record.get("mondo_id"),
                        "match_type": record["match_type"],
                        "hierarchy_depth": record["hierarchy_depth"],
                    }
                )

        # If still no matches, try fuzzy matching as fallback
        if not diseases:
            mondo_id = self._find_mondo_match(query_term)
            if mondo_id:
                result = self._execute_query(
                    """
                    MATCH (d:Disease {mondo_id: $mondo_id})
                    RETURN d.id AS id,
                           d.name AS name,
                           d.mondo_id AS mondo_id,
                           'fuzzy_match' AS match_type,
                           0 AS hierarchy_depth
                """,
                    {"mondo_id": mondo_id},
                )

                for record in result:
                    diseases.append(
                        {
                            "id": record["id"],
                            "name": record["name"],
                            "mondo_id": record.get("mondo_id"),
                            "match_type": record["match_type"],
                            "hierarchy_depth": record["hierarchy_depth"],
                        }
                    )

        logger.info(f"Found {len(diseases)} diseases for query '{query_term}'")
        return diseases

    def get_hierarchy_statistics(self) -> dict[str, Any]:
        """Get statistics about disease hierarchy in the database."""
        stats: dict[str, Any] = {}

        # Total diseases
        result = self._execute_query("MATCH (d:Disease) RETURN count(d) AS count")
        stats["total_diseases"] = result[0]["count"] if result else 0

        # Diseases with MONDO IDs
        result = self._execute_query(
            "MATCH (d:Disease) WHERE d.mondo_id IS NOT NULL RETURN count(d) AS count"
        )
        stats["diseases_with_mondo"] = result[0]["count"] if result else 0

        # Diseases with hierarchical relationships
        result = self._execute_query(
            "MATCH (d:Disease)-[:IS_A]->() RETURN count(DISTINCT d) AS count"
        )
        stats["diseases_with_hierarchy"] = result[0]["count"] if result else 0

        # Total IS_A relationships
        result = self._execute_query("MATCH ()-[r:IS_A]->() RETURN count(r) AS count")
        stats["is_a_relationships"] = result[0]["count"] if result else 0

        # Average hierarchy depth
        result = self._execute_query("""
            MATCH (d:Disease)-[r:IS_A]->()
            WHERE r.depth IS NOT NULL
            RETURN avg(r.depth) AS avg_depth, max(r.depth) AS max_depth
        """)

        if result:
            stats["avg_hierarchy_depth"] = result[0]["avg_depth"] or 0
            stats["max_hierarchy_depth"] = result[0]["max_depth"] or 0
        else:
            stats["avg_hierarchy_depth"] = 0
            stats["max_hierarchy_depth"] = 0

        # Top-level disease categories
        result = self._execute_query("""
            MATCH (parent:Disease)<-[:IS_A]-(child:Disease)
            WHERE NOT (parent)-[:IS_A]->()
            RETURN parent.name AS category, count(child) AS child_count
            ORDER BY child_count DESC
            LIMIT 10
        """)

        stats["top_categories"] = [
            {"category": record["category"], "child_count": record["child_count"]}
            for record in result
        ]

        return stats

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()


# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich disease nodes with hierarchical relationships from MONDO ontology"
    )
    parser.add_argument("--database", "-d", default="cvd1", help="Database name")
    parser.add_argument("--obo-path", help="Path to MONDO.obo file")
    parser.add_argument(
        "--operation",
        choices=["enrich", "query", "stats"],
        default="enrich",
        help="Operation to perform",
    )
    parser.add_argument(
        "--query-term", help="Disease term to query for hierarchical matches"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without making changes"
    )
    parser.add_argument(
        "--max-depth", type=int, default=3, help="Maximum hierarchy depth for queries"
    )
    parser.add_argument(
        "--consolidate-extracted",
        action="store_true",
        default=True,
        help="Consolidate extracted diseases with hierarchy (default: True)",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip consolidation of extracted diseases",
    )

    args = parser.parse_args()

    # Handle consolidation flag logic
    consolidate_extracted = args.consolidate_extracted and not args.no_consolidate

    enricher = DiseaseHierarchyEnricher(database=args.database, obo_path=args.obo_path)

    try:
        if args.operation == "enrich":
            stats = enricher.enrich_disease_hierarchy(
                dry_run=args.dry_run, consolidate_extracted=consolidate_extracted
            )
            print("\nEnrichment Results:")
            print(f"  Diseases processed: {stats['diseases_processed']}")
            print(f"  MONDO matches found: {stats['mondo_matches_found']}")
            print(f"  Parent relationships: {stats['parent_relationships_created']}")
            print(
                f"  Ancestor relationships: {stats['ancestor_relationships_created']}"
            )
            print(f"  New parent nodes: {stats['new_parent_nodes_created']}")
            if consolidate_extracted:
                print(
                    f"  Extracted diseases processed: {stats.get('extracted_diseases_processed', 0)}"
                )
                print(
                    f"  Diseases consolidated: {stats.get('diseases_consolidated', 0)}"
                )
                if not args.dry_run:
                    print(f"  Nodes merged: {stats.get('nodes_merged', 0)}")
                    print(
                        f"  Relationships transferred: {stats.get('relationships_transferred', 0)}"
                    )

        elif args.operation == "query":
            if not args.query_term:
                print("--query-term required for query operation")
                sys.exit(1)
            diseases = enricher.query_disease_hierarchy(args.query_term, args.max_depth)
            print(f"\nHierarchical query results for '{args.query_term}':")
            for disease in diseases:
                print(
                    f"  {disease['match_type']}: {disease['name']} (depth: {disease['hierarchy_depth']})"
                )

        elif args.operation == "stats":
            stats = enricher.get_hierarchy_statistics()
            print("\nDisease Hierarchy Statistics:")
            print(f"  Total diseases: {stats['total_diseases']}")
            print(f"  With MONDO IDs: {stats['diseases_with_mondo']}")
            print(f"  With hierarchy: {stats['diseases_with_hierarchy']}")
            print(f"  IS_A relationships: {stats['is_a_relationships']}")
            print(f"  Avg hierarchy depth: {stats['avg_hierarchy_depth']:.2f}")
            print(f"  Max hierarchy depth: {stats['max_hierarchy_depth']}")
            print("\nTop disease categories:")
            for cat in stats["top_categories"]:
                print(f"    {cat['category']}: {cat['child_count']} children")

    finally:
        enricher.close()
