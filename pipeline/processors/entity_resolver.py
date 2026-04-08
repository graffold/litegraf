#!/usr/bin/env python3
"""
Enhanced Entity Resolution System for Disease Consolidation and Ontology Mapping

This module provides comprehensive entity resolution capab                            // Transfer incoming relationships
                            OPTIONAL MATCH (other)-[r]->(duplicate)
                            WHERE other.id <> $canonical_id
                            MERGE (other)-[new_rel_in:ASSOCIATED_WITH]->(canonical)
                            DELETE rs including:
1. Disease name consolidation using MONDO ontology
2. Hierarchical relationship creation (parent-child disease relationships)
3. Cross-reference mapping for disease variants
4. Protein-disease relationship enhancement
5. Node label merging (e.g., UniprotSubcellularLocation -> SubcellularLocation)
"""

import difflib
import random
import re
import time
from collections import defaultdict
from typing import Any

from pipeline.processors.ontology_filter import OntologyFilter
from src.core.database import Neo4jDatabase
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class EntityResolver:
    """
    Advanced entity resolution system that consolidates entities using ontology mappings
    and creates hierarchical relationships between related concepts.

    Capabilities:
    - Disease name consolidation using MONDO ontology
    - Hierarchical relationship creation (parent-child disease relationships)
    - Cross-reference mapping for disease variants
    - Protein-disease relationship enhancement
    - Node label merging (e.g., UniprotSubcellularLocation -> SubcellularLocation)
    """

    def __init__(
        self,
        db: Neo4jDatabase | None = None,
        ontology_filter: OntologyFilter | None = None,
        database: str = "cvd1",
    ):
        self.db = db or Neo4jDatabase(database=database)
        self.ontology_filter = ontology_filter or OntologyFilter()
        self.disease_ontology = self.ontology_filter.disease_ontology
        self.protein_ontology = self.ontology_filter.protein_ontology
        logger.info(
            f"Initialized EntityResolver with ontology mappings, database: {database}"
        )

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using Neo4j."""
        return self.db._execute_cypher(query, parameters)

    def _normalize_name(self, name: str) -> str:
        """Normalize entity names for comparison."""
        if not name:
            return ""
        # Convert to lowercase, remove extra spaces
        normalized = re.sub(r"\s+", " ", name.lower().strip())

        # Handle different punctuation and symbols
        normalized = normalized.replace("'", "'").replace(
            "'", "'"
        )  # Normalize apostrophes
        normalized = re.sub(
            r"[^\w\s\(\)\-\']", "", normalized
        )  # Keep only letters, numbers, spaces, parens, hyphens, apostrophes

        # Remove content in parentheses only if it's abbreviations (like (AD), (PH))
        # Keep the main disease name
        if "(" in normalized:
            # If there's an abbreviation in parentheses, remove it
            main_part = re.sub(r"\s*\([^)]*\)\s*", " ", normalized).strip()
            if main_part:  # Only use main part if it's not empty
                normalized = main_part

        return normalized

    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate string similarity between two names."""
        norm1, norm2 = self._normalize_name(name1), self._normalize_name(name2)
        return difflib.SequenceMatcher(None, norm1, norm2).ratio()

    def _find_canonical_disease(self, disease_name: str) -> dict[str, Any] | None:
        """Find canonical disease form using MONDO ontology."""
        name_lower = self._normalize_name(disease_name)

        # Exact matches first
        for term_id, data in self.disease_ontology.items():
            canonical_name = data.get("name", "")
            if self._normalize_name(canonical_name) == name_lower:
                return {
                    "id": term_id,
                    "name": canonical_name,
                    "synonyms": data.get("synonyms", []),
                }

            # Check synonyms
            for synonym in data.get("synonyms", []):
                if self._normalize_name(synonym) == name_lower:
                    return {
                        "id": term_id,
                        "name": canonical_name,
                        "synonyms": data.get("synonyms", []),
                    }

        # Fuzzy matching for close variants
        best_match = None
        best_score = 0.85  # Threshold for fuzzy matching

        for term_id, data in self.disease_ontology.items():
            canonical_name = data.get("name", "")
            score = self._calculate_similarity(disease_name, canonical_name)
            if score > best_score:
                best_match = {
                    "id": term_id,
                    "name": canonical_name,
                    "synonyms": data.get("synonyms", []),
                }
                best_score = score

            # Check synonyms too
            for synonym in data.get("synonyms", []):
                score = self._calculate_similarity(disease_name, synonym)
                if score > best_score:
                    best_match = {
                        "id": term_id,
                        "name": canonical_name,
                        "synonyms": data.get("synonyms", []),
                    }
                    best_score = score

        return best_match

    def consolidate_entities(
        self, strategy: str = "uniprot_id", dry_run: bool = False
    ) -> dict[str, Any]:
        """
        Unified entity consolidation method with multiple strategies.

        Args:
            strategy: Consolidation strategy to use:
                - "uniprot_id": Merge proteins by UniProt ID (fastest, most accurate)
                - "name_features": Merge by name, gene_symbol, synonyms (comprehensive)
                - "name": Merge by primary name only (basic)
                - "full": Run all strategies in sequence
            dry_run: If True, only report what would be done without making changes

        Returns:
            Dictionary with consolidation statistics including merged_count
        """
        logger.info(
            f"Starting entity consolidation with strategy='{strategy}' (dry_run={dry_run})"
        )

        if strategy == "uniprot_id":
            return self.consolidate_proteins_by_uniprot_id(dry_run=dry_run)
        if strategy == "name_features":
            return self.consolidate_entities_by_name_features(dry_run=dry_run)
        if strategy == "name":
            return self.consolidate_entities_by_name(dry_run=dry_run)
        if strategy == "full":
            # Run all strategies in sequence
            uniprot_stats = self.consolidate_proteins_by_uniprot_id(dry_run=dry_run)
            name_features_stats = self.consolidate_entities_by_name_features(
                dry_run=dry_run
            )
            name_stats = self.consolidate_entities_by_name(dry_run=dry_run)

            # Combine statistics
            return {
                "merged_count": (
                    uniprot_stats.get("nodes_merged", 0)
                    + name_features_stats.get("nodes_merged", 0)
                    + name_stats.get("nodes_merged", 0)
                ),
                "uniprot_stats": uniprot_stats,
                "name_features_stats": name_features_stats,
                "name_stats": name_stats,
            }
        raise ValueError(
            f"Unknown consolidation strategy: {strategy}. Use 'uniprot_id', 'name_features', 'name', or 'full'"
        )

    def consolidate_proteins_by_uniprot_id(
        self, dry_run: bool = False
    ) -> dict[str, int]:
        """
        Consolidate protein nodes that have the same UniProt ID but different names.
        This ensures proteins are deduplicated based on their canonical UniProt identifiers.
        """
        logger.info(
            f"Starting UniProt-based protein consolidation (dry_run={dry_run})..."
        )

        stats = {"proteins_consolidated": 0, "nodes_merged": 0}

        # Find proteins grouped by UniProt ID
        query = """
        MATCH (p:Protein)
        WHERE p.uniprotID IS NOT NULL
        RETURN p.uniprotID AS uniprot_id,
               collect({id: p.id, name: p.name}) AS protein_instances
        ORDER BY p.uniprotID
        """

        try:
            results = self._execute_query(query)

            for result in results:
                uniprot_id = result["uniprot_id"]
                protein_instances = result["protein_instances"]

                if len(protein_instances) <= 1:
                    continue  # No duplicates to consolidate

                # Sort by preference: keep the one with the most complete name
                protein_instances.sort(
                    key=lambda x: (len(x.get("name", "")), x.get("name", "").lower()),
                    reverse=True,
                )

                canonical_protein = protein_instances[0]
                duplicates = protein_instances[1:]

                if dry_run:
                    logger.info(
                        f"Would consolidate {len(duplicates)} proteins under UniProt ID '{uniprot_id}':"
                    )
                    logger.info(
                        f"  Canonical: '{canonical_protein['name']}' (id: {canonical_protein['id']})"
                    )
                    for dup in duplicates:
                        logger.info(f"  Duplicate: '{dup['name']}' (id: {dup['id']})")
                else:
                    # Consolidate the duplicates
                    for dup_protein in duplicates:
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                # Validate IDs exist
                                if not dup_protein.get(
                                    "id"
                                ) or not canonical_protein.get("id"):
                                    logger.warning(
                                        f"Skipping protein merge due to missing ID: dup_id={dup_protein.get('id')}, canonical_id={canonical_protein.get('id')}"
                                    )
                                    break

                                # Neo4j consolidation for proteins - Split into smaller transactions

                                # 1. Transfer incoming relationships
                                transfer_in_query = """
                                MATCH (dup:Protein {id: $dup_id})
                                MATCH (canonical:Protein {id: $canonical_id})
                                OPTIONAL MATCH (other)-[r]->(dup)
                                WHERE other.id <> $canonical_id
                                WITH dup, canonical, other, r
                                WHERE r IS NOT NULL
                                MERGE (other)-[new_rel_in:ASSOCIATED_WITH]->(canonical)
                                SET new_rel_in = properties(r)
                                DELETE r
                                """
                                self._execute_query(
                                    transfer_in_query,
                                    {
                                        "dup_id": dup_protein["id"],
                                        "canonical_id": canonical_protein["id"],
                                    },
                                )

                                # 2. Transfer outgoing relationships
                                transfer_out_query = """
                                MATCH (dup:Protein {id: $dup_id})
                                MATCH (canonical:Protein {id: $canonical_id})
                                OPTIONAL MATCH (dup)-[r]->(other)
                                WHERE other.id <> $canonical_id
                                WITH dup, canonical, other, r
                                WHERE r IS NOT NULL
                                MERGE (canonical)-[new_rel_out:ASSOCIATED_WITH]->(other)
                                SET new_rel_out = properties(r)
                                DELETE r
                                """
                                self._execute_query(
                                    transfer_out_query,
                                    {
                                        "dup_id": dup_protein["id"],
                                        "canonical_id": canonical_protein["id"],
                                    },
                                )

                                # 3. Update canonical and delete duplicate
                                finalize_query = """
                                MATCH (dup:Protein {id: $dup_id})
                                MATCH (canonical:Protein {id: $canonical_id})
                                SET canonical.consolidated_from = COALESCE(canonical.consolidated_from, []) + [$dup_name],
                                    canonical.consolidated_ids = COALESCE(canonical.consolidated_ids, []) + [$dup_id]
                                DETACH DELETE dup
                                """

                                self._execute_query(
                                    finalize_query,
                                    {
                                        "dup_id": dup_protein["id"],
                                        "canonical_id": canonical_protein["id"],
                                        "dup_name": dup_protein["name"],
                                    },
                                )

                                stats["nodes_merged"] += 1
                                logger.info(
                                    f"Merged protein '{dup_protein['name']}' into '{canonical_protein['name']}' (UniProt: {uniprot_id})"
                                )
                                break  # Success, exit retry loop

                            except Exception as e:
                                if (
                                    "DeadlockDetected" in str(e)
                                    and attempt < max_retries - 1
                                ):
                                    sleep_time = random.uniform(0.1, 1.0) * (
                                        attempt + 1
                                    )
                                    logger.warning(
                                        f"Deadlock detected merging {dup_protein['name']}. Retrying in {sleep_time:.2f}s..."
                                    )
                                    time.sleep(sleep_time)
                                else:
                                    logger.error(
                                        f"Failed to merge protein '{dup_protein['name']}': {e}"
                                    )
                                    break

                    stats["proteins_consolidated"] += 1

        except Exception as e:
            logger.error(f"Error during UniProt-based protein consolidation: {e}")

        logger.info(f"UniProt-based protein consolidation complete: {stats}")
        return stats

    def consolidate_entities_by_name_features(
        self, dry_run: bool = False
    ) -> dict[str, int]:
        """
        Consolidate entities based on multiple name-related features, not just primary names.
        This provides more comprehensive deduplication by considering:
        - Proteins: name, uniprot_gene_name, gene_symbol, synonyms
        - Diseases: name, synonyms, alternative_names

        This is more aggressive than name-based consolidation and should be run after UniProt-based consolidation.
        """
        logger.info(
            f"Starting comprehensive name features consolidation (dry_run={dry_run})..."
        )

        stats = {
            "proteins_consolidated": 0,
            "diseases_consolidated": 0,
            "nodes_merged": 0,
            "feature_matches": {},
        }

        # Define name features for each entity type
        name_features = {
            "Protein": ["name", "uniprot_gene_name", "gene_symbol"],
            "Disease": ["name"],
        }

        # Process each entity type
        for entity_type, features in name_features.items():
            logger.info(f"Processing {entity_type} name features: {features}")

            # Get all entities of this type with their properties
            # Build return clause dynamically
            return_parts = ["n.id AS id", "n.name AS name"]
            additional_features = [
                f"n.{feature} AS {feature}" for feature in features if feature != "name"
            ]
            if additional_features:
                return_parts.extend(additional_features)

            query = f"""
            MATCH (n:{entity_type})
            RETURN {", ".join(return_parts)}
            """

            try:
                entities = self._execute_query(query)

                # Group entities by normalized name features
                feature_groups = defaultdict(list)

                for entity in entities:
                    entity_id = entity.get("id")
                    if not entity_id:
                        continue

                    # Collect all non-null name features for this entity
                    name_values = []
                    for feature in features:
                        value = entity.get(feature)
                        if value and isinstance(value, str):
                            # Normalize the name feature
                            normalized = self._normalize_name(value)
                            if normalized:
                                name_values.append(normalized)

                    # Also check synonyms if available
                    synonyms = entity.get("synonyms", [])
                    if isinstance(synonyms, list):
                        for synonym in synonyms:
                            if synonym and isinstance(synonym, str):
                                normalized_syn = self._normalize_name(synonym)
                                if normalized_syn:
                                    name_values.append(normalized_syn)

                    # Use the first name value as the grouping key (most representative)
                    if name_values:
                        group_key = name_values[0]  # Use first normalized name as key
                        feature_groups[group_key].append(
                            {
                                "id": entity_id,
                                "name": entity.get("name", ""),
                                "name_values": name_values,
                                "all_features": {
                                    k: v for k, v in entity.items() if k != "id"
                                },
                            }
                        )

                # Process groups with multiple entities
                for group_key, entity_group in feature_groups.items():
                    if len(entity_group) <= 1:
                        continue

                    # Find entities that share name features
                    consolidated_group = []
                    processed_ids = set()

                    for entity in entity_group:
                        if entity["id"] in processed_ids:
                            continue

                        # Find other entities that share name features with this one
                        matching_entities = [entity]

                        for other_entity in entity_group:
                            if (
                                other_entity["id"] in processed_ids
                                or other_entity["id"] == entity["id"]
                            ):
                                continue

                            # Check if they share any name features
                            shared_features = set(entity["name_values"]) & set(
                                other_entity["name_values"]
                            )
                            if shared_features:
                                matching_entities.append(other_entity)
                                processed_ids.add(other_entity["id"])

                        if len(matching_entities) > 1:
                            consolidated_group.extend(matching_entities)
                            processed_ids.add(entity["id"])

                    # Consolidate each group of matching entities
                    if consolidated_group:
                        # Sort by completeness (prefer entities with more complete information)
                        consolidated_group.sort(
                            key=lambda x: (
                                len(
                                    [v for v in x["all_features"].values() if v]
                                ),  # More properties
                                len(x["name"] or ""),  # Longer names
                                x["name"] or "",  # Alphabetical
                            ),
                            reverse=True,
                        )

                        canonical_entity = consolidated_group[0]
                        duplicates = consolidated_group[1:]

                        if dry_run:
                            logger.info(
                                f"Would consolidate {len(duplicates)} {entity_type} entities by name features:"
                            )
                            logger.info(
                                f"  Canonical: '{canonical_entity['name']}' (id: {canonical_entity['id']})"
                            )
                            for dup in duplicates:
                                shared = set(canonical_entity["name_values"]) & set(
                                    dup["name_values"]
                                )
                                logger.info(
                                    f"  Duplicate: '{dup['name']}' (id: {dup['id']}) - shared features: {list(shared)}"
                                )
                        else:
                            # Perform consolidation
                            for dup_entity in duplicates:
                                max_retries = 3
                                for attempt in range(max_retries):
                                    try:
                                        # Validate IDs exist and names are not null
                                        if not dup_entity.get(
                                            "id"
                                        ) or not canonical_entity.get("id"):
                                            logger.warning(
                                                f"Skipping {entity_type} merge due to missing ID"
                                            )
                                            break

                                        dup_name = dup_entity.get("name") or "Unknown"
                                        if dup_name == "None":
                                            dup_name = "Unknown"

                                        if not isinstance(dup_name, str):
                                            dup_name = str(dup_name)

                                        # Neo4j-compatible consolidation - Split into smaller transactions

                                        # 1. Transfer incoming relationships
                                        transfer_in_query = f"""
                                        MATCH (dup:{entity_type} {{id: $dup_id}})
                                        MATCH (canonical:{entity_type} {{id: $canonical_id}})
                                        OPTIONAL MATCH (other)-[r]->(dup)
                                        WHERE other.id <> $canonical_id
                                        WITH dup, canonical, other, r
                                        WHERE r IS NOT NULL
                                        MERGE (other)-[new_rel_in:ASSOCIATED_WITH]->(canonical)
                                        SET new_rel_in = properties(r)
                                        DELETE r
                                        """
                                        self._execute_query(
                                            transfer_in_query,
                                            {
                                                "dup_id": dup_entity["id"],
                                                "canonical_id": canonical_entity["id"],
                                            },
                                        )

                                        # 2. Transfer outgoing relationships
                                        transfer_out_query = f"""
                                        MATCH (dup:{entity_type} {{id: $dup_id}})
                                        MATCH (canonical:{entity_type} {{id: $canonical_id}})
                                        OPTIONAL MATCH (dup)-[r]->(other)
                                        WHERE other.id <> $canonical_id
                                        WITH dup, canonical, other, r
                                        WHERE r IS NOT NULL
                                        MERGE (canonical)-[new_rel_out:ASSOCIATED_WITH]->(other)
                                        SET new_rel_out = properties(r)
                                        DELETE r
                                        """
                                        self._execute_query(
                                            transfer_out_query,
                                            {
                                                "dup_id": dup_entity["id"],
                                                "canonical_id": canonical_entity["id"],
                                            },
                                        )

                                        # 3. Update canonical and delete duplicate
                                        finalize_query = f"""
                                        MATCH (dup:{entity_type} {{id: $dup_id}})
                                        MATCH (canonical:{entity_type} {{id: $canonical_id}})
                                        SET canonical.consolidated_from = COALESCE(canonical.consolidated_from, []) + [$dup_name],
                                            canonical.consolidated_ids = COALESCE(canonical.consolidated_ids, []) + [$dup_id],
                                            canonical.consolidated_by = COALESCE(canonical.consolidated_by, []) + ['name_features']
                                        DETACH DELETE dup
                                        """

                                        self._execute_query(
                                            finalize_query,
                                            {
                                                "dup_id": dup_entity["id"],
                                                "canonical_id": canonical_entity["id"],
                                                "dup_name": dup_name,
                                            },
                                        )

                                        stats["nodes_merged"] += 1
                                        shared_features = set(
                                            canonical_entity["name_values"]
                                        ) & set(dup_entity["name_values"])
                                        logger.info(
                                            f"Merged {entity_type} '{dup_entity['name']}' into '{canonical_entity['name']}' (shared features: {list(shared_features)})"
                                        )
                                        break  # Success

                                    except Exception as e:
                                        if (
                                            "DeadlockDetected" in str(e)
                                            and attempt < max_retries - 1
                                        ):
                                            sleep_time = random.uniform(0.1, 1.0) * (
                                                attempt + 1
                                            )
                                            logger.warning(
                                                f"Deadlock detected merging {dup_entity['name']}. Retrying in {sleep_time:.2f}s..."
                                            )
                                            time.sleep(sleep_time)
                                        else:
                                            logger.error(
                                                f"Failed to merge {entity_type} '{dup_entity['name']}': {e}"
                                            )
                                            break

                                if entity_type == "Protein":
                                    stats["proteins_consolidated"] += 1
                                elif entity_type == "Disease":
                                    stats["diseases_consolidated"] += 1

                            # Track feature matches
                            feature_key = f"{entity_type.lower()}_name_features"
                            if feature_key not in stats["feature_matches"]:
                                stats["feature_matches"][feature_key] = 0
                            stats["feature_matches"][feature_key] += len(duplicates)

            except Exception as e:
                logger.error(
                    f"Error during {entity_type} name features consolidation: {e}"
                )
                continue

        logger.info(f"Name features consolidation complete: {stats}")
        return stats

    def consolidate_entities_by_name(self, dry_run: bool = False) -> dict[str, int]:
        """
        Consolidate entities that have very similar names but different IDs.
        This addresses the specific case like 'Pulmonary Hypertension (PH)' vs 'pulmonary hypertension (PH)'
        """
        logger.info(f"Starting name-based entity consolidation (dry_run={dry_run})...")

        # Get all disease nodes
        query = """
        MATCH (n)
        WHERE n:Disease OR n:Entity
        RETURN n.id AS id, n.name AS name, labels(n) AS labels
        ORDER BY n.name
        """

        nodes = self._execute_query(query)

        # Group by normalized names
        name_groups = defaultdict(list)
        for node in nodes:
            if node.get("name"):
                normalized = self._normalize_name(node["name"])
                name_groups[normalized].append(node)

        stats = {"diseases_consolidated": 0, "nodes_merged": 0}

        # Process groups with multiple nodes
        for node_group in name_groups.values():
            if len(node_group) <= 1:
                continue

            # Sort by preference: Disease label first, then by ID (handle None IDs)
            node_group.sort(
                key=lambda x: (
                    0 if "Disease" in x.get("labels", []) else 1,
                    x.get("id", "") or "",  # Convert None to empty string
                )
            )

            canonical_node = node_group[0]
            duplicates = node_group[1:]

            if dry_run:
                logger.info(
                    f"Would consolidate {len(duplicates)} nodes under '{canonical_node['name']}':"
                )
                for dup in duplicates:
                    logger.info(f"  - '{dup['name']}' (id: {dup['id']})")
            else:
                # Consolidate the duplicates
                for dup_node in duplicates:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # Validate that both nodes exist and have valid IDs
                            if not dup_node.get("id") or not canonical_node.get("id"):
                                logger.warning(
                                    f"Skipping merge due to missing ID: dup_id={dup_node.get('id')}, canonical_id={canonical_node.get('id')}"
                                )
                                break

                            # Verify both nodes still exist in the database
                            check_query = """
                            OPTIONAL MATCH (d {id: $dup_id})
                            OPTIONAL MATCH (c {id: $canonical_id})
                            RETURN
                              CASE WHEN d IS NOT NULL THEN true ELSE false END as dup_exists,
                              CASE WHEN c IS NOT NULL THEN true ELSE false END as canonical_exists
                            """

                            check_result = self._execute_query(
                                check_query,
                                {
                                    "dup_id": dup_node["id"],
                                    "canonical_id": canonical_node["id"],
                                },
                            )

                            if (
                                not check_result
                                or not check_result[0].get("dup_exists", False)
                                or not check_result[0].get("canonical_exists", False)
                            ):
                                logger.warning(
                                    f"Skipping merge - nodes don't exist: dup_id={dup_node['id']}, canonical_id={canonical_node['id']}"
                                )
                                break

                            # Neo4j-compatible consolidation query - Split into smaller transactions

                            # 1. Transfer incoming relationships
                            transfer_in_query = """
                            MATCH (duplicate {id: $dup_id})
                            MATCH (canonical {id: $canonical_id})
                            OPTIONAL MATCH (other)-[r]->(duplicate)
                            WHERE other.id <> $canonical_id
                            WITH duplicate, canonical, other, r
                            WHERE r IS NOT NULL
                            MERGE (other)-[new_rel_in:ASSOCIATED_WITH]->(canonical)
                            SET new_rel_in = properties(r)
                            DELETE r
                            """
                            self._execute_query(
                                transfer_in_query,
                                {
                                    "dup_id": dup_node["id"],
                                    "canonical_id": canonical_node["id"],
                                },
                            )

                            # 2. Transfer outgoing relationships
                            transfer_out_query = """
                            MATCH (duplicate {id: $dup_id})
                            MATCH (canonical {id: $canonical_id})
                            OPTIONAL MATCH (duplicate)-[r]->(other)
                            WHERE other.id <> $canonical_id
                            WITH duplicate, canonical, other, r
                            WHERE r IS NOT NULL
                            MERGE (canonical)-[new_rel_out:ASSOCIATED_WITH]->(other)
                            SET new_rel_out = properties(r)
                            DELETE r
                            """
                            self._execute_query(
                                transfer_out_query,
                                {
                                    "dup_id": dup_node["id"],
                                    "canonical_id": canonical_node["id"],
                                },
                            )

                            # 3. Update canonical and delete duplicate
                            finalize_query = """
                            MATCH (duplicate {id: $dup_id})
                            MATCH (canonical {id: $canonical_id})
                            SET canonical.name = $canonical_name,
                                canonical.consolidated_from = COALESCE(canonical.consolidated_from, []) + [$dup_name],
                                canonical.consolidated_ids = COALESCE(canonical.consolidated_ids, []) + [$dup_id]
                            REMOVE canonical:Entity
                            SET canonical:Disease
                            DETACH DELETE duplicate
                            """

                            self._execute_query(
                                finalize_query,
                                {
                                    "dup_id": dup_node["id"],
                                    "canonical_id": canonical_node["id"],
                                    "canonical_name": canonical_node["name"],
                                    "dup_name": dup_node["name"],
                                },
                            )

                            # Verify the consolidation worked
                            check_dup = self._execute_query(
                                "MATCH (n {id: $id}) RETURN count(n) as exists",
                                {"id": dup_node["id"]},
                            )
                            if check_dup and check_dup[0]["exists"] == 0:
                                logger.info(
                                    f"✓ Successfully deleted duplicate '{dup_node['name']}'"
                                )
                            else:
                                logger.error(
                                    f"✗ Failed to delete duplicate '{dup_node['name']}' - still exists"
                                )

                            stats["nodes_merged"] += 1
                            stats["diseases_consolidated"] += 1
                            logger.info(
                                f"Merged '{dup_node['name']}' into '{canonical_node['name']}'"
                            )
                            break  # Success, exit retry loop

                        except Exception as e:
                            if (
                                "DeadlockDetected" in str(e)
                                and attempt < max_retries - 1
                            ):
                                sleep_time = random.uniform(0.1, 1.0) * (attempt + 1)
                                logger.warning(
                                    f"Deadlock detected merging {dup_node['name']}. Retrying in {sleep_time:.2f}s..."
                                )
                                time.sleep(sleep_time)
                            else:
                                logger.error(
                                    f"Failed to merge node '{dup_node['name']}': {e}"
                                )
                                break

                stats["diseases_consolidated"] += 1

        return stats

    def create_pulmonary_hypertension_hierarchy(
        self, dry_run: bool = False
    ) -> dict[str, int]:
        """
        Create the specific Pulmonary Hypertension hierarchy as requested.

        Creates relationships between:
        - Pulmonary Hypertension (parent)
        - Group 1: Pulmonary Arterial Hypertension (PAH)
        - Group 2: Pulmonary Hypertension due to Left Heart Disease
        - Group 3: Pulmonary Hypertension due to Lung Diseases and/or Hypoxia
        - Group 4: Chronic Thromboembolic Pulmonary Hypertension (CTEPH)
        - Group 5: Pulmonary Hypertension with Unclear Multifactorial Mechanisms
        """
        logger.info(f"Creating Pulmonary Hypertension hierarchy (dry_run={dry_run})...")

        # Define the PH hierarchy
        ph_hierarchy = {
            "parent": "Pulmonary Hypertension",
            "subtypes": [
                {
                    "name": "Pulmonary Arterial Hypertension",
                    "group": 1,
                    "abbreviation": "PAH",
                },
                {
                    "name": "Idiopathic Pulmonary Arterial Hypertension",
                    "group": 1,
                    "abbreviation": "IPAH",
                    "parent_group": "PAH",
                },
                {
                    "name": "Heritable Pulmonary Arterial Hypertension",
                    "group": 1,
                    "abbreviation": "HPAH",
                    "parent_group": "PAH",
                },
                {
                    "name": "Pulmonary Hypertension due to Left Heart Disease",
                    "group": 2,
                    "abbreviation": "PH-LHD",
                },
                {
                    "name": "Pulmonary Hypertension due to Lung Diseases",
                    "group": 3,
                    "abbreviation": "PH-LD",
                },
                {
                    "name": "Pulmonary Hypertension due to Hypoxia",
                    "group": 3,
                    "abbreviation": "PH-Hypoxia",
                },
                {
                    "name": "Chronic Thromboembolic Pulmonary Hypertension",
                    "group": 4,
                    "abbreviation": "CTEPH",
                },
                {
                    "name": "Pulmonary Hypertension with Unclear Multifactorial Mechanisms",
                    "group": 5,
                    "abbreviation": "PH-UMM",
                },
            ],
        }

        stats = {"hierarchy_created": 0, "nodes_enhanced": 0}

        # First, ensure parent node exists or create it
        parent_name = ph_hierarchy["parent"]

        if not dry_run:
            # Create or update parent PH node
            parent_query = """
            MERGE (ph:Disease:Entity {name: $parent_name})
            ON CREATE SET ph.id = 'PH_PARENT_' + toString(timestamp())
            SET ph.mondo_id = 'MONDO:0002081',
                ph.is_parent_category = true,
                ph.WHO_classification = 'Pulmonary Hypertension Groups 1-5',
                ph.canonical_name = $parent_name
            RETURN ph.id as id, ph.name as name
            """

            parent_result = self._execute_query(
                parent_query, {"parent_name": parent_name}
            )
            parent_id = parent_result[0]["id"]
            stats["nodes_enhanced"] += 1

        # Process each subtype
        for subtype in ph_hierarchy["subtypes"]:
            subtype_name = subtype["name"]
            group_num = subtype["group"]
            abbreviation = subtype["abbreviation"]

            if dry_run:
                logger.info(
                    f"Would create/enhance: Group {group_num}: {subtype_name} ({abbreviation})"
                )
                continue

            # Check if this disease node exists in database (fuzzy matching)
            find_query = """
            MATCH (d)
            WHERE d:Disease OR d:Entity
            RETURN d.id as id, d.name as name, labels(d) as labels
            """

            existing_nodes = self._execute_query(find_query)

            # Find best match
            best_match = None
            best_score = 0.7  # Threshold for matching

            for node in existing_nodes:
                node_name = node.get("name", "")
                if not node_name:  # Skip nodes with empty names
                    continue

                score = self._calculate_similarity(node_name, subtype_name)

                # Also check for abbreviation matches
                if abbreviation and abbreviation.lower() in node_name.lower():
                    score += 0.3

                # Check for key terms
                if (
                    "pulmonary" in node_name.lower()
                    and "hypertension" in node_name.lower()
                ):
                    score += 0.2

                if score > best_score:
                    best_match = node
                    best_score = score

            if best_match:
                # Update existing node
                update_query = """
                MATCH (d {id: $node_id})
                SET d.name = $canonical_name,
                    d.WHO_group = $group_num,
                    d.abbreviation = $abbreviation,
                    d.canonical_name = $canonical_name,
                    d.original_name = d.name
                REMOVE d:Entity
                SET d:Disease
                RETURN d.id as id
                """

                self._execute_query(
                    update_query,
                    {
                        "node_id": best_match["id"],
                        "canonical_name": subtype_name,
                        "group_num": group_num,
                        "abbreviation": abbreviation,
                    },
                )

                # Create relationship to parent
                rel_query = """
                MATCH (parent:Disease {id: $parent_id})
                MATCH (child:Disease {id: $child_id})
                MERGE (parent)-[r:ONTOLOGY_RELATIONSHIP]->(child)
                SET r.type = 'WHO_CLASSIFICATION',
                    r.group_number = $group_num,
                    r.source = 'WHO_PH_Guidelines',
                    r.relationship = 'PARENT_OF'
                """

                self._execute_query(
                    rel_query,
                    {
                        "parent_id": parent_id,
                        "child_id": best_match["id"],
                        "group_num": group_num,
                    },
                )

                stats["hierarchy_created"] += 1
                stats["nodes_enhanced"] += 1
                logger.info(
                    f"Enhanced and linked: {subtype_name} (Group {group_num}) - matched with '{best_match['name']}'"
                )
            else:
                # Create new node if not found
                create_query = """
                CREATE (d:Disease {
                    id: 'PH_G' + toString($group_num) + '_' + toString(timestamp()),
                    name: $subtype_name,
                    WHO_group: $group_num,
                    abbreviation: $abbreviation,
                    canonical_name: $subtype_name,
                    source: 'WHO_PH_Guidelines'
                })
                RETURN d.id as id
                """

                result = self._execute_query(
                    create_query,
                    {
                        "subtype_name": subtype_name,
                        "group_num": group_num,
                        "abbreviation": abbreviation,
                    },
                )

                new_node_id = result[0]["id"]

                # Link to parent
                rel_query = """
                MATCH (parent:Disease {id: $parent_id})
                MATCH (child:Disease {id: $child_id})
                MERGE (parent)-[r:ONTOLOGY_RELATIONSHIP]->(child)
                SET r.type = 'WHO_CLASSIFICATION',
                    r.group_number = $group_num,
                    r.source = 'WHO_PH_Guidelines',
                    r.relationship = 'PARENT_OF'
                """

                self._execute_query(
                    rel_query,
                    {
                        "parent_id": parent_id,
                        "child_id": new_node_id,
                        "group_num": group_num,
                    },
                )

                stats["hierarchy_created"] += 1
                stats["nodes_enhanced"] += 1
                logger.info(f"Created and linked: {subtype_name} (Group {group_num})")

        logger.info(f"PH hierarchy creation complete: {stats}")
        return stats

    def merge_node_labels(
        self, label_mapping: dict[str, str], dry_run: bool = False
    ) -> dict[str, int]:
        """
        Merge node labels by changing old labels to new labels.

        Args:
            label_mapping: Dictionary mapping old labels to new labels
                e.g., {"UniprotSubcellularLocation": "SubcellularLocation", "DiseaseName": "Disease"}
            dry_run: If True, only report what would be done

        Returns:
            Statistics about the merge operations
        """
        logger.info(f"Starting node label merging (dry_run={dry_run})...")
        logger.info(f"Label mapping: {label_mapping}")

        stats = {"total_nodes_processed": 0, "labels_merged": 0}

        for old_label, new_label in label_mapping.items():
            logger.info(f"Processing label merge: {old_label} -> {new_label}")

            # First, count nodes with the old label
            count_query = f"""
            MATCH (n:{old_label})
            RETURN count(n) as node_count
            """

            try:
                count_result = self._execute_query(count_query)
                node_count = count_result[0]["node_count"] if count_result else 0

                if node_count == 0:
                    logger.info(f"No nodes found with label '{old_label}' - skipping")
                    continue

                if dry_run:
                    logger.info(
                        f"Would merge {node_count} nodes from '{old_label}' to '{new_label}'"
                    )
                    stats["total_nodes_processed"] += node_count
                    continue

                # Perform the label merge
                # Neo4j approach: Also use SET and REMOVE
                merge_query = f"""
                MATCH (n:{old_label})
                SET n:{new_label}
                REMOVE n:{old_label}
                RETURN count(n) as merged_count
                """

                merge_result = self._execute_query(merge_query)
                merged_count = merge_result[0]["merged_count"] if merge_result else 0

                stats["total_nodes_processed"] += merged_count
                stats["labels_merged"] += 1

                logger.info(
                    f"Successfully merged {merged_count} nodes from '{old_label}' to '{new_label}'"
                )

            except Exception as e:
                logger.error(
                    f"Failed to merge label '{old_label}' to '{new_label}': {e}"
                )
                continue

        logger.info(f"Node label merging complete: {stats}")
        return stats

    def run_full_entity_resolution(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Run the complete entity resolution pipeline.

        1. Map proteins to UniProt IDs (critical for consolidation)
        2. Consolidate proteins by UniProt ID (most accurate deduplication)
        3. Consolidate entities by comprehensive name features (proteins: name/gene_symbol/synonyms, diseases: name/synonyms)
        4. Consolidate entities by name using UniProt IDs as canonical identifiers
        5. Create the Pulmonary Hypertension hierarchy

        Args:
            dry_run: If True, only report what would be done

        Returns:
            Complete statistics from all operations
        """
        logger.info(f"Starting full entity resolution pipeline (dry_run={dry_run})...")

        all_stats = {}

        # Step 1: Map proteins to UniProt IDs (critical prerequisite for consolidation)
        logger.info("Step 1: Mapping proteins to UniProt IDs...")
        try:
            from src.utils.enhanced_uniprot_mapper import EnhancedUniProtMapper

            mapper = EnhancedUniProtMapper(
                database=self.db.database if self.db else None
            )
            if dry_run:
                uniprot_stats = mapper.run_mapping(dry_run=True)
            else:
                uniprot_stats = mapper.run_mapping(dry_run=False)

            all_stats["uniprot_mapping"] = uniprot_stats
            logger.info(f"Enhanced UniProt mapping completed: {uniprot_stats}")

        except Exception as e:
            logger.error(f"UniProt mapping failed: {e}")
            all_stats["uniprot_mapping"] = {"error": str(e)}

        # Step 2: Consolidate proteins by UniProt ID first (most accurate)
        logger.info("Step 2: Consolidating proteins by UniProt ID...")
        protein_stats = self.consolidate_proteins_by_uniprot_id(dry_run=dry_run)
        all_stats["protein_consolidation"] = protein_stats

        # Step 3: Consolidate entities by comprehensive name features (proteins: name/gene_symbol/synonyms, diseases: name/synonyms)
        logger.info("Step 3: Consolidating entities by comprehensive name features...")
        name_features_stats = self.consolidate_entities_by_name_features(
            dry_run=dry_run
        )
        all_stats["name_features_consolidation"] = name_features_stats

        # Step 4: Consolidate entities by name (now using UniProt IDs as canonical identifiers)
        logger.info("Step 4: Consolidating entities by name...")
        consolidation_stats = self.consolidate_entities_by_name(dry_run=dry_run)
        all_stats["consolidation"] = consolidation_stats

        # Step 5: Create specific PH hierarchy
        logger.info("Step 5: Creating Pulmonary Hypertension hierarchy...")
        ph_stats = self.create_pulmonary_hypertension_hierarchy(dry_run=dry_run)
        all_stats["ph_hierarchy"] = ph_stats

        # Summary
        total_operations = (
            all_stats.get("uniprot_mapping", {}).get("mapped_proteins", 0)
            + protein_stats.get("proteins_consolidated", 0)
            + protein_stats.get("nodes_merged", 0)
            + name_features_stats.get("proteins_consolidated", 0)
            + name_features_stats.get("diseases_consolidated", 0)
            + name_features_stats.get("nodes_merged", 0)
            + consolidation_stats.get("diseases_consolidated", 0)
            + consolidation_stats.get("nodes_merged", 0)
            + ph_stats.get("hierarchy_created", 0)
            + ph_stats.get("nodes_enhanced", 0)
        )

        all_stats["summary"] = {
            "total_operations": total_operations,
            "dry_run": dry_run,
        }

        logger.info(f"Entity resolution complete. Total operations: {total_operations}")
        return all_stats

    def consolidate_duplicate_properties(
        self,
        property_mappings: dict[str, list[str]],
        node_label: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Consolidate duplicate properties on nodes by merging values from multiple property names.

        Args:
            property_mappings: Dictionary mapping canonical property name to list of duplicate property names
                e.g., {"uniprot_id": ["uniprotID"], "uniprot_gene_name": ["name"]}
            node_label: Optional node label to filter nodes (e.g., "Protein")
            dry_run: If True, only report what would be done

        Returns:
            Statistics about the consolidation operations
        """
        logger.info(f"Starting property consolidation (dry_run={dry_run})...")
        logger.info(f"Property mappings: {property_mappings}")
        if node_label:
            logger.info(f"Filtering to nodes with label: {node_label}")

        stats = {
            "nodes_processed": 0,
            "properties_consolidated": 0,
            "nodes_updated": 0,
            "consolidation_details": {},
        }

        # Build the node match clause
        match_clause = f"MATCH (n:{node_label})" if node_label else "MATCH (n)"

        for canonical_prop, duplicate_props in property_mappings.items():
            logger.info(
                f"Processing property consolidation: {canonical_prop} <- {duplicate_props}"
            )

            # Find nodes that have duplicate properties
            conditions = []
            for dup_prop in duplicate_props:
                conditions.append(f"n.{dup_prop} IS NOT NULL")

            if not conditions:
                continue

            # Query to find nodes with duplicate properties
            # Handle duplicate column names by aliasing conflicts
            standard_columns = {"id", "name"}
            dup_prop_selects = []
            for prop in duplicate_props:
                if prop in standard_columns:
                    # Avoid duplicate column names by aliasing
                    dup_prop_selects.append(f"n.{prop} as dup_{prop}")
                else:
                    dup_prop_selects.append(f"n.{prop} as {prop}")

            find_query = f"""
            {match_clause}
            WHERE {" OR ".join(conditions)}
            RETURN n.id as id, n.name as name,
                   n.{canonical_prop} as canonical_value,
                   {", ".join(dup_prop_selects)}
            """

            try:
                nodes_with_duplicates = self._execute_query(find_query)

                consolidation_count = 0
                update_count = 0

                for node in nodes_with_duplicates:
                    node_id = node.get("id")
                    node_name = node.get("name", "unknown")
                    canonical_value = node.get("canonical_value")

                    # Collect all non-null values from duplicate properties
                    duplicate_values = []
                    standard_columns = {"id", "name"}
                    for dup_prop in duplicate_props:
                        # Handle aliased column names
                        if dup_prop in standard_columns:
                            column_name = f"dup_{dup_prop}"
                        else:
                            column_name = dup_prop

                        dup_value = node.get(column_name)
                        if dup_value and dup_value != canonical_value:
                            duplicate_values.append((dup_prop, dup_value))

                    if not duplicate_values:
                        continue

                    consolidation_count += len(duplicate_values)

                    if dry_run:
                        logger.info(
                            f"Would consolidate properties on node '{node_name}' (id: {node_id}):"
                        )
                        logger.info(f"  Canonical {canonical_prop}: {canonical_value}")
                        for dup_prop, dup_value in duplicate_values:
                            logger.info(
                                f"  Would merge {dup_prop}: {dup_value} -> {canonical_prop}"
                            )
                    else:
                        # Perform the consolidation

                        # Choose the value to keep (prefer canonical if it exists, otherwise use first duplicate)
                        final_value = canonical_value or duplicate_values[0][1]

                        # Build the update query
                        set_clauses = [f"n.{canonical_prop} = $final_value"]
                        remove_clauses = []

                        for dup_prop, dup_value in duplicate_values:
                            if (
                                dup_prop != canonical_prop
                            ):  # Don't remove canonical property
                                remove_clauses.append(f"n.{dup_prop}")

                        if node_label:
                            update_query = f"""
                            MATCH (n:{node_label} {{id: $node_id}})
                            SET {", ".join(set_clauses)}
                            {("REMOVE " + ", ".join(remove_clauses)) if remove_clauses else ""}
                            RETURN count(n) as updated_count
                            """
                        else:
                            update_query = f"""
                            MATCH (n {{id: $node_id}})
                            SET {", ".join(set_clauses)}
                            {("REMOVE " + ", ".join(remove_clauses)) if remove_clauses else ""}
                            RETURN count(n) as updated_count
                            """

                        update_result = self._execute_query(
                            update_query,
                            {"node_id": node_id, "final_value": final_value},
                        )

                        if (
                            update_result
                            and update_result[0].get("updated_count", 0) > 0
                        ):
                            update_count += 1
                            logger.info(
                                f"Consolidated properties on '{node_name}': {canonical_prop}={final_value}"
                            )
                            for dup_prop, dup_value in duplicate_values:
                                logger.info(
                                    f"  Merged {dup_prop}:{dup_value} -> {canonical_prop}"
                                )

                stats["properties_consolidated"] += consolidation_count
                stats["nodes_updated"] += update_count
                stats["consolidation_details"][canonical_prop] = {
                    "duplicate_properties": duplicate_props,
                    "properties_consolidated": consolidation_count,
                    "nodes_updated": update_count,
                }

                logger.info(
                    f"Property '{canonical_prop}' consolidation: {consolidation_count} properties on {update_count} nodes"
                )

            except Exception as e:
                logger.error(f"Error consolidating property '{canonical_prop}': {e}")
                continue

        stats["nodes_processed"] = sum(
            detail["nodes_updated"]
            for detail in stats["consolidation_details"].values()
        )

        logger.info(f"Property consolidation complete: {stats}")
        return stats

    def close(self):
        """Close database connections."""
        self.db.close()
        logger.info("EntityResolver closed")


def main():
    """Command line interface for entity resolution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Entity Resolution with Ontology Mapping"
    )
    parser.add_argument("--database", default="cvd1", help="Database name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--operation",
        choices=[
            "consolidate",
            "consolidate-uniprot",
            "consolidate-name-features",
            "ph-hierarchy",
            "merge-labels",
            "merge-properties",
            "full",
        ],
        default="full",
        help="Operation to perform",
    )
    parser.add_argument(
        "--label-mapping",
        nargs="*",
        help="Label mapping for merge operation (format: old_label:new_label old_label2:new_label2)",
    )
    parser.add_argument(
        "--property-mapping",
        nargs="*",
        help="Property mapping for merge-properties operation (format: canonical_prop:dup_prop1,dup_prop2)",
    )
    parser.add_argument(
        "--node-label",
        help="Node label to filter for property consolidation (e.g., Protein)",
    )

    args = parser.parse_args()

    # Parse label mapping if provided
    label_mapping = {}
    if args.label_mapping:
        for mapping in args.label_mapping:
            if ":" not in mapping:
                raise ValueError(
                    f"Invalid label mapping format: {mapping}. Use 'old_label:new_label' format."
                )
            old_label, new_label = mapping.split(":", 1)
            label_mapping[old_label.strip()] = new_label.strip()

    # Parse property mapping if provided
    property_mapping = {}
    if args.property_mapping:
        for mapping in args.property_mapping:
            if ":" not in mapping:
                raise ValueError(
                    f"Invalid property mapping format: {mapping}. Use 'canonical_prop:dup_prop1,dup_prop2' format."
                )
            canonical_prop, dup_props_str = mapping.split(":", 1)
            dup_props = [prop.strip() for prop in dup_props_str.split(",")]
            property_mapping[canonical_prop.strip()] = dup_props

    try:
        ontology_filter = OntologyFilter()
        resolver = EntityResolver(
            db=None, ontology_filter=ontology_filter, database=args.database
        )

        if args.operation == "consolidate":
            stats = resolver.consolidate_entities_by_name(dry_run=args.dry_run)
        elif args.operation == "consolidate-uniprot":
            stats = resolver.consolidate_proteins_by_uniprot_id(dry_run=args.dry_run)
        elif args.operation == "consolidate-name-features":
            stats = resolver.consolidate_entities_by_name_features(dry_run=args.dry_run)
        elif args.operation == "ph-hierarchy":
            stats = resolver.create_pulmonary_hypertension_hierarchy(
                dry_run=args.dry_run
            )
        elif args.operation == "merge-labels":
            if not label_mapping:
                raise ValueError(
                    "Label mapping is required for merge-labels operation. Use --label-mapping old:new"
                )
            stats = resolver.merge_node_labels(label_mapping, dry_run=args.dry_run)
        elif args.operation == "merge-properties":
            if not property_mapping:
                raise ValueError(
                    "Property mapping is required for merge-properties operation. Use --property-mapping canonical:dup1,dup2"
                )
            stats = resolver.consolidate_duplicate_properties(
                property_mapping, args.node_label, dry_run=args.dry_run
            )
        elif args.operation == "full":
            stats = resolver.run_full_entity_resolution(dry_run=args.dry_run)

        print("\nEntity Resolution Results:")
        print(f"Operation: {args.operation}")
        print(f"Dry Run: {args.dry_run}")
        if args.operation == "merge-labels":
            print(f"Label Mapping: {label_mapping}")
        elif args.operation == "merge-properties":
            print(f"Property Mapping: {property_mapping}")
            if args.node_label:
                print(f"Node Label Filter: {args.node_label}")
        print(f"Statistics: {stats}")

        resolver.close()

    except Exception as e:
        logger.error(f"Entity resolution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
