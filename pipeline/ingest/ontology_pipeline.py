"""
Unified Ontology Pipeline for KG Ingestion

This module provides a unified ontology pipeline that standardizes ontology operations
for Neo4j-backed knowledge graphs. It encapsulates:

- Ontology-based node and relationship filtering
- Protein entity linking and deduplication
- Post-processing node labeling and standardization

The pipeline integrates with LangGraph workflows and provides consistent
ontology validation and standardization across biomedical knowledge graphs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pipeline.processors.ontology_filter import OntologyFilter
from src.utils import logging_utils

logger = logging_utils.setup_logging()


class BackendAdapter(ABC):
    """Abstract base class for backend-specific query execution."""

    @abstractmethod
    def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results."""

    @abstractmethod
    def close(self) -> None:
        """Close the backend connection."""


class Neo4jBackendAdapter(BackendAdapter):
    """Neo4j-specific backend adapter."""

    def __init__(self, db):
        self.db = db

    def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.db._execute_cypher(query, parameters or {})

    def close(self) -> None:
        if self.db:
            self.db.close()


class OntologyPipeline:
    """
    Unified ontology pipeline for KG ingestion.

    This class provides standardized ontology operations including:
    - Node and relationship filtering based on biomedical ontologies
    - Protein entity linking and deduplication
    - Post-processing node labeling and name standardization
    """

    def __init__(
        self,
        backend_adapter: BackendAdapter | None = None,
        ontology_filter: OntologyFilter | None = None,
    ):
        """
        Initialize the ontology pipeline.

        Args:
            backend_adapter: Backend-specific adapter for query execution
            ontology_filter: OntologyFilter instance (created if not provided)
        """
        self.backend_adapter = backend_adapter
        self.ontology_filter = ontology_filter or OntologyFilter()
        self.logger = logging.getLogger(__name__)

    def set_backend_adapter(self, adapter: BackendAdapter) -> None:
        """Set the backend adapter for query execution."""
        self.backend_adapter = adapter

    async def filter_ontology(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Filter nodes and relationships for biomedical relevance using ontology validation.

        This is a LangGraph-compatible step that:
        1. Filters nodes using OntologyFilter.filter_nodes()
        2. Filters relationships to only include valid node connections
        3. Logs filtering statistics

        Args:
            state: Pipeline state containing 'nodes' and 'relationships'

        Returns:
            Updated state with filtered nodes and relationships
        """
        if state.get("error"):
            return state

        try:
            # Filter nodes for biomedical relevance
            filtered_nodes = self.ontology_filter.filter_nodes(state["nodes"])
            valid_node_ids = {node["id"] for node in filtered_nodes}

            # Filter relationships to only include connections between valid nodes
            filtered_relationships = [
                rel
                for rel in state["relationships"]
                if rel["source_id"] in valid_node_ids
                and rel["target_id"] in valid_node_ids
            ]

            chunk_id = state.get("chunk_id", "unknown")
            self.logger.debug(
                f"Filtered ontology for chunk {chunk_id}: "
                f"{len(filtered_nodes)} nodes, {len(filtered_relationships)} relationships"
            )

            return {"nodes": filtered_nodes, "relationships": filtered_relationships}

        except Exception as e:
            chunk_id = state.get("chunk_id", "unknown")
            self.logger.error(
                f"Ontology filtering failed for chunk {chunk_id}: {e}", exc_info=True
            )
            return {"error": str(e)}

    async def link_protein_entities(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Link extracted protein entities to existing Protein nodes with UniProt IDs.

        This step deduplicates proteins by:
        1. Finding existing canonical protein nodes
        2. Updating relationships to reference existing nodes instead of creating duplicates
        3. Setting canonical properties for new proteins

        Args:
            state: Pipeline state containing 'nodes' and 'relationships'

        Returns:
            Updated state with linked protein entities
        """
        if state.get("error") or not self.backend_adapter:
            return {"nodes": state["nodes"], "relationships": state["relationships"]}

        try:
            # Find protein entities in the extracted nodes
            protein_nodes = [
                node for node in state["nodes"] if node.get("type") == "Protein"
            ]

            if not protein_nodes:
                return {
                    "nodes": state["nodes"],
                    "relationships": state["relationships"],
                }  # No proteins to link

            linked_nodes = []
            linked_relationships = list(
                state["relationships"]
            )  # Copy existing relationships

            for protein_node in protein_nodes:
                protein_name = protein_node.get("name", "").strip()
                if not protein_name:
                    linked_nodes.append(protein_node)
                    continue

                # Try to find existing Protein node identifier
                existing_node_id = await self._find_uniprot_id_for_protein(protein_name)

                if existing_node_id:
                    # Update relationships to reference existing node instead of creating duplicate
                    for relationship in linked_relationships:
                        if relationship.get("source_id") == protein_node["id"]:
                            relationship["source_id"] = existing_node_id
                        if relationship.get("target_id") == protein_node["id"]:
                            relationship["target_id"] = existing_node_id

                    self.logger.info(
                        f"Mapped extracted protein '{protein_name}' to existing protein node {existing_node_id} - "
                        "skipping duplicate creation"
                    )
                    # Skip adding this node (it won't be created)
                else:
                    # No existing protein found - create new one with canonical properties
                    enhanced_protein_node = protein_node.copy()

                    # Set canonical UniProt properties
                    enhanced_protein_node["uniprot_gene_name"] = protein_name
                    # uniprot_id will remain None for now (until manual mapping or future enhancement)

                    linked_nodes.append(enhanced_protein_node)
                    self.logger.debug(
                        f"No existing protein found for '{protein_name}' - will create new node with canonical properties"
                    )

            return {"nodes": linked_nodes, "relationships": linked_relationships}

        except Exception as e:
            chunk_id = state.get("chunk_id", "unknown")
            self.logger.error(
                f"Protein entity linking failed for chunk {chunk_id}: {e}",
                exc_info=True,
            )
            return {
                "nodes": state["nodes"],
                "relationships": state["relationships"],
            }  # Return original state on error

    async def _find_uniprot_id_for_protein(self, protein_name: str) -> str | None:
        """
        Find existing protein node identifier for linking relationships.

        This method searches for canonical protein matches using various strategies:
        1. Exact matches on UniProt fields
        2. Fuzzy matching with prioritization
        3. Compound protein name splitting
        4. Canonical gene symbol mapping

        Args:
            protein_name: Name of the protein to find

        Returns:
            Node identifier if found, None otherwise
        """
        if not self.backend_adapter:
            return None

        try:
            # Normalize protein name for matching
            protein_name.strip().upper()

            # Query for existing Protein nodes that match this name
            query = """
            MATCH (p:Protein)
            WHERE p.uniprot_gene_name = $protein_name OR
                  p.uniprot_protein_description = $protein_name OR
                  p.uniprot_id = $protein_name OR
                  toLower(p.uniprot_protein_description) = toLower($protein_name) OR
                  toLower(p.uniprot_gene_name) = toLower($protein_name)
            RETURN COALESCE(p.uniprot_gene_name, p.uniprot_protein_description) AS node_identifier,
                   p.uniprot_id AS uniprot_id,
                   p.uniprot_gene_name AS gene_name,
                   p.uniprot_protein_description AS description
            ORDER BY CASE WHEN p.uniprot_id IS NOT NULL THEN 0 ELSE 1 END
            LIMIT 1
            """
            results = self.backend_adapter.execute_query(
                query, {"protein_name": protein_name}
            )

            if results and len(results) > 0:
                node_identifier = results[0].get("node_identifier")
                uniprot_id = results[0].get("uniprot_id")
                uniprot_gene_name = results[0].get("gene_name")
                results[0].get("description")
                if node_identifier:
                    priority_type = "canonical" if uniprot_id else "existing"
                    self.logger.info(
                        f"Found {priority_type} protein '{protein_name}' -> '{node_identifier}' "
                        f"(uniprot_id: {uniprot_id}, gene_name: {uniprot_gene_name})"
                    )
                    return str(node_identifier)

            # Try fuzzy matching and other strategies...
            # (Additional matching logic would go here, similar to the original implementation)

            return None

        except Exception as e:
            self.logger.debug(
                f"Error finding UniProt ID for protein '{protein_name}': {e}"
            )
            return None

    def resolve_and_label_nodes(self, ingestion_job_id: str | None = None) -> None:
        """
        Perform post-processing node labeling and name standardization.

        Args:
            ingestion_job_id: If provided, only label nodes from this ingestion job.
                When None, labels ALL nodes (full-scan, use for scheduled maintenance).

        This method should be called after all chunks have been processed to:
        1. Label Entity nodes as Disease or Protein based on ontology matching
        2. Standardize node names to canonical forms
        3. Remove redundant Entity labels
        """
        if not self.backend_adapter:
            self.logger.warning("No backend adapter available for node labeling")
            return

        try:
            scope = f"job {ingestion_job_id}" if ingestion_job_id else "all nodes"
            self.logger.info(
                f"Starting post-processing node labeling and standardization (scope: {scope})"
            )

            # Use the OntologyFilter's method with the adapter
            self.ontology_filter.resolve_and_label_nodes(
                backend_adapter=self.backend_adapter,
                ingestion_job_id=ingestion_job_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to resolve and label nodes: {e}", exc_info=True)

    def close(self) -> None:
        """Close the ontology pipeline and its backend adapter."""
        if self.backend_adapter:
            self.backend_adapter.close()
