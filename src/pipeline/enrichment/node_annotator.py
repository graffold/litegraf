"""
Node Annotator - Handles simple property and node class enrichment.

This class manages the final phase of enrichment for non-text-rich columns:
1. Add properties to existing protein nodes
2. Create new node classes for categorical data
3. Handle 1:1 mappings between name and ID columns
4. Process multi-value columns appropriately
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from pipeline.interfaces import GraphStore

from .base import (
    BaseEnrichmentProcessor,
    ColumnAnalysis,
    ColumnStrategy,
    ColumnType,
    EnrichmentStats,
)

logger = logging.getLogger(__name__)


class NodeAnnotator(BaseEnrichmentProcessor):
    """Handles property addition and node class creation for non-text columns."""

    def __init__(self, graph_store: GraphStore, *, database: str = "cvd1"):
        """
        Initialize node annotator.

        Args:
            graph_store: Graph database backend
            database: Database name
        """
        if not isinstance(graph_store, GraphStore):
            raise TypeError(
                f"graph_store must be a GraphStore, got {type(graph_store)}"
            )
        self.db = graph_store

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using the graph store."""
        return self.db.execute_query(query, parameters)

    async def process(
        self, data: list[dict[str, Any]], analysis: dict[str, ColumnAnalysis]
    ) -> dict[str, Any]:
        """Process data according to the node annotation logic."""
        # Extract metadata
        uniprot_column = None
        for col_name, col_analysis in analysis.items():
            if col_analysis.column_type.value == "uniprot_id":
                uniprot_column = col_name
                break

        # Process columns
        stats = await self.process_columns(
            analysis_results=analysis, csv_data=data, uniprot_column=uniprot_column
        )

        return {"processor_type": "node_annotation", "stats": stats, "success": True}

    async def process_columns(
        self,
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None = None,
        **kwargs,
    ) -> EnrichmentStats:
        """
        Process columns for property addition and node class creation.

        Args:
            analysis_results: Column analysis results
            csv_data: CSV data rows
            uniprot_column: Name of UniProt ID column if available
            **kwargs: Additional processing parameters

        Returns:
            EnrichmentStats with processing results
        """
        logger.info("Starting node annotation processing")
        start_time = datetime.now()

        # Categorize columns by strategy
        property_columns = []
        meaningful_node_class_columns = []

        for col_name, analysis in analysis_results.items():
            if analysis.strategy == ColumnStrategy.PROPERTY:
                property_columns.append(col_name)
            elif analysis.strategy == ColumnStrategy.MEANINGFUL_NODE_CLASS:
                meaningful_node_class_columns.append(col_name)

        # Initialize stats
        stats = EnrichmentStats()

        # Process property columns
        if property_columns:
            logger.info(
                f"Processing {len(property_columns)} property columns: {property_columns}"
            )
            property_stats = await self._process_property_columns(
                property_columns, analysis_results, csv_data, uniprot_column
            )
            stats.proteins_processed += property_stats.proteins_processed
            stats.properties_added += property_stats.properties_added

        # Process meaningful node class columns
        if meaningful_node_class_columns:
            logger.info(
                f"Processing {len(meaningful_node_class_columns)} meaningful node class columns: {meaningful_node_class_columns}"
            )
            meaningful_stats = await self._process_meaningful_node_class_columns(
                meaningful_node_class_columns,
                analysis_results,
                csv_data,
                uniprot_column,
            )
            stats.node_classes_created += meaningful_stats.node_classes_created
            stats.individual_nodes_created += meaningful_stats.individual_nodes_created
            stats.relationships_created += meaningful_stats.relationships_created

        processing_time = (datetime.now() - start_time).total_seconds()

        # Log summary
        logger.info("Node annotation processing complete:")
        logger.info(f"  • Property columns: {len(property_columns)}")
        logger.info(
            f"  • Meaningful node class columns: {len(meaningful_node_class_columns)}"
        )
        logger.info(f"  • Proteins processed: {stats.proteins_processed}")
        logger.info(f"  • Properties added: {stats.properties_added}")
        logger.info(f"  • Node classes created: {stats.node_classes_created}")
        logger.info(f"  • Individual nodes created: {stats.individual_nodes_created}")
        logger.info(f"  • Relationships created: {stats.relationships_created}")
        logger.info(f"  • Processing time: {processing_time:.2f}s")

        return stats

    async def _process_property_columns(
        self,
        columns: list[str],
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
    ) -> EnrichmentStats:
        """Process columns that should be added as properties to existing nodes - EFFICIENTLY with bulk operations."""

        if not uniprot_column:
            logger.warning(
                "No UniProt column found, cannot add properties to protein nodes"
            )
            return EnrichmentStats()

        stats = EnrichmentStats()

        # Step 1: Create all protein nodes in one bulk operation
        logger.info("Creating protein nodes from UniProt IDs...")
        unique_uniprot_ids = set()
        for row in csv_data:
            uniprot_id = row.get(uniprot_column, "").strip()
            if uniprot_id and uniprot_id.upper() != "NA":
                unique_uniprot_ids.add(uniprot_id)

        if unique_uniprot_ids:
            # Bulk create protein nodes
            uniprot_list = list(unique_uniprot_ids)

            # Process in large batches (1000 at a time for efficiency)
            batch_size = 1000
            for i in range(0, len(uniprot_list), batch_size):
                batch_ids = uniprot_list[i : i + batch_size]

                # Create protein nodes, but first check for existing ones by gene name
                # Build lookup data with uniprot_id -> gene_name mapping
                uniprot_gene_map = {}
                for row in csv_data:
                    if row.get(uniprot_column) in batch_ids:
                        gene_name = row.get("uniprot_gene_name", "").strip()
                        if gene_name and gene_name.upper() != "NA":
                            uniprot_gene_map[row.get(uniprot_column)] = gene_name

                # Query to handle protein deduplication by gene name
                # Separate update and create operations
                update_query = """
                UNWIND $protein_data AS data
                MATCH (existing:Protein) WHERE existing.uniprot_gene_name = data.gene_name
                SET existing.uniprot_id = data.uniprot_id
                RETURN existing
                """

                create_query = """
                UNWIND $protein_data AS data
                MERGE (p:Protein:Entity {uniprot_id: data.uniprot_id})
                ON CREATE SET p.uniprot_gene_name = data.gene_name
                RETURN p
                """

                # Prepare data for query
                protein_data = []
                for uniprot_id in batch_ids:
                    protein_data.append(
                        {
                            "uniprot_id": uniprot_id,
                            "gene_name": uniprot_gene_map.get(uniprot_id),
                        }
                    )

                try:
                    # First update existing proteins
                    update_result = self._execute_query(
                        update_query, {"protein_data": protein_data}
                    )
                    # Then create new proteins
                    create_result = self._execute_query(
                        create_query, {"protein_data": protein_data}
                    )
                    if update_result or create_result:
                        logger.info(
                            f"Created/ensured {len(batch_ids)} protein nodes (with gene name deduplication)"
                        )
                except Exception as e:
                    logger.error(f"Failed to create protein nodes batch: {e}")

        stats.proteins_processed = len(unique_uniprot_ids)

        # Step 2: Bulk set properties for each column
        for column in columns:
            analysis = analysis_results[column]
            logger.info(f"Bulk setting property {column} for all proteins")

            # Sanitize property name for Cypher
            property_name = self._sanitize_property_name(column)

            # Collect column data efficiently
            property_updates = []
            for row in csv_data:
                uniprot_id = row.get(uniprot_column, "").strip()
                value = row.get(column, "").strip()

                if not uniprot_id or not value or value.upper() == "NA":
                    continue

                # Process value based on type
                if analysis.has_multiple_values:
                    values = self._split_multi_value(value)
                    if values:
                        # Convert list to semicolon-separated string
                        processed_value = ";".join(str(v) for v in values)
                        property_updates.append(
                            {"uniprot_id": uniprot_id, "value": processed_value}
                        )
                else:
                    processed_value = self._convert_value_type(
                        value, analysis.column_type
                    )
                    property_updates.append(
                        {"uniprot_id": uniprot_id, "value": str(processed_value)}
                    )

            # Bulk update properties in large batches
            if property_updates:
                batch_size = 1000  # Much larger batches for efficiency

                for i in range(0, len(property_updates), batch_size):
                    batch = property_updates[i : i + batch_size]

                    # Single UNWIND query to update all properties in batch
                    query = f"""
                    UNWIND $updates AS update
                    MATCH (p:Protein {{uniprot_id: update.uniprot_id}})
                    SET p.`{property_name}` = update.value
                    RETURN count(p) as updated_count
                    """

                    try:
                        result = self._execute_query(query, {"updates": batch})
                        if result and result[0].get("updated_count", 0) > 0:
                            updated_count = result[0]["updated_count"]
                            stats.properties_added += updated_count
                            logger.info(
                                f"Bulk updated {updated_count} proteins with {column} property"
                            )
                    except Exception as e:
                        logger.error(f"Failed to bulk update property {column}: {e}")
                        # Try individual updates as fallback
                        logger.info(f"Falling back to individual updates for {column}")
                        for update in batch:
                            try:
                                individual_query = f"""
                                MATCH (p:Protein {{uniprot_id: $uniprot_id}})
                                SET p.`{property_name}` = $value
                                """
                                self._execute_query(
                                    individual_query,
                                    {
                                        "uniprot_id": update["uniprot_id"],
                                        "value": update["value"],
                                    },
                                )
                                stats.properties_added += 1
                            except Exception as individual_error:
                                logger.warning(
                                    f"Failed individual update for {update['uniprot_id']}: {individual_error}"
                                )

        return stats

    async def _process_node_class_columns(
        self,
        columns: list[str],
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
    ) -> EnrichmentStats:
        """Process columns that should create new node classes."""

        stats = EnrichmentStats()

        for column in columns:
            logger.info(f"Creating node class for column: {column}")

            # Create class name from column name
            class_name = self._create_class_name(column)

            # Get unique values for this column
            unique_values = set()
            for row in csv_data:
                value = row.get(column, "").strip()
                if value and value.upper() != "NA":
                    unique_values.add(value)

            # Create node class
            try:
                # Create the class node
                class_query = """
                MERGE (c:Class {name: $class_name, type: 'enrichment_class'})
                SET c.source = $source,
                    c.unique_value_count = $value_count
                RETURN c
                """
                class_result = self._execute_query(
                    class_query,
                    {
                        "class_name": class_name,
                        "source": f"enrichment:{column}",
                        "value_count": len(unique_values),
                    },
                )

                if class_result:
                    stats.node_classes_created += 1
                    logger.info(f"✅ Created class node: {class_name}")

                # Create individual nodes for each unique value
                node_stats = await self._create_individual_nodes(
                    class_name, unique_values, column, csv_data, uniprot_column
                )
                stats.individual_nodes_created += node_stats.individual_nodes_created
                stats.relationships_created += node_stats.relationships_created

            except Exception as e:
                logger.error(f"Failed to create node class for {column}: {e}")
                continue

        return stats

    async def _process_multi_value_columns(
        self,
        columns: list[str],
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
    ) -> EnrichmentStats:
        """Process multi-value columns that should create node classes."""

        stats = EnrichmentStats()

        for column in columns:
            logger.info(f"Creating node class for multi-value column: {column}")

            class_name = self._create_class_name(column)

            # Collect all unique values from multi-value cells
            unique_values = set()
            for row in csv_data:
                value = row.get(column, "").strip()
                if value and value.upper() != "NA":
                    values = self._split_multi_value(value)
                    unique_values.update(values)

            # Create class and nodes
            try:
                # Create the class node
                class_query = """
                MERGE (c:Class {name: $class_name, type: 'enrichment_class'})
                SET c.source = $source,
                    c.unique_value_count = $value_count,
                    c.multi_value = true
                RETURN c
                """
                class_result = self._execute_query(
                    class_query,
                    {
                        "class_name": class_name,
                        "source": f"enrichment:{column}",
                        "value_count": len(unique_values),
                    },
                )

                if class_result:
                    stats.node_classes_created += 1

                # Create individual nodes and relationships
                node_stats = await self._create_individual_nodes(
                    class_name,
                    unique_values,
                    column,
                    csv_data,
                    uniprot_column,
                    multi_value=True,
                )
                stats.individual_nodes_created += node_stats.individual_nodes_created
                stats.relationships_created += node_stats.relationships_created

            except Exception as e:
                logger.error(
                    f"Failed to create multi-value node class for {column}: {e}"
                )
                continue

        return stats

    async def _process_one_to_one_mappings(
        self,
        columns: list[str],
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
    ) -> EnrichmentStats:
        """Process 1:1 name-ID mapping columns."""

        stats = EnrichmentStats()

        for column in columns:
            analysis = analysis_results[column]
            partner_column = analysis.partner_column

            if not partner_column:
                logger.warning(
                    f"No partner column found for 1:1 mapping column {column}"
                )
                continue

            logger.info(f"Processing 1:1 mapping: {column} <-> {partner_column}")

            class_name = self._create_class_name(column)

            # Collect unique name-ID pairs
            name_id_pairs = {}
            for row in csv_data:
                name_value = row.get(column, "").strip()
                id_value = row.get(partner_column, "").strip()

                if (
                    name_value
                    and id_value
                    and name_value.upper() != "NA"
                    and id_value.upper() != "NA"
                ):
                    name_id_pairs[name_value] = id_value

            try:
                # Create the class node
                class_query = """
                MERGE (c:Class {name: $class_name, type: 'enrichment_class'})
                SET c.source_column = $column,
                    c.partner_column = $partner_column,
                    c.created_at = datetime(),
                    c.unique_value_count = $value_count,
                    c.mapping_type = '1:1'
                RETURN c
                """
                class_result = self._execute_query(
                    class_query,
                    {
                        "class_name": class_name,
                        "source": f"enrichment:{column}",
                        "partner_column": partner_column,
                        "value_count": len(name_id_pairs),
                    },
                )

                if class_result:
                    stats.node_classes_created += 1

                # Create individual nodes with both name and ID
                for name, external_id in name_id_pairs.items():
                    # Sanitize name for use as node identifier
                    self._sanitize_node_name(name)

                    node_query = f"""
                    MERGE (n:{class_name} {{name: $name}})
                    SET n.external_id = $external_id,
                        n.source_column = $source_column,
                        n.created_at = datetime()
                    RETURN n
                    """

                    node_result = self._execute_query(
                        node_query,
                        {
                            "name": name,
                            "external_id": external_id,
                            "source_column": column,
                        },
                    )

                    if node_result:
                        stats.individual_nodes_created += 1

                # Create relationships to proteins if UniProt column exists
                if uniprot_column:
                    relationship_stats = await self._create_protein_relationships(
                        class_name, column, csv_data, uniprot_column
                    )
                    stats.relationships_created += (
                        relationship_stats.relationships_created
                    )

            except Exception as e:
                logger.error(f"Failed to process 1:1 mapping for {column}: {e}")
                continue

        return stats

    async def _process_property_with_relationship_columns(
        self,
        columns: list[str],
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
    ) -> EnrichmentStats:
        """
        Process columns that should be both properties AND create meaningful node classes with relationships.

        This creates the pattern described by the user:
        - Protein gets annotation property (e.g., hpa_protein_location = "intracellular")
        - Protein gets relationship to meaningful node class (e.g., -> cytoplasm:uniprot_subcellular_location)
        """

        if not uniprot_column:
            logger.warning(
                "No UniProt column found, cannot process property+relationship columns"
            )
            return EnrichmentStats()

        stats = EnrichmentStats()
        proteins_updated = set()

        for column in columns:
            analysis = analysis_results[column]
            logger.info(
                f"Processing property+relationship column {column} with {analysis.unique_count} unique values"
            )

            # Sanitize property name for Cypher
            property_name = self._sanitize_property_name(column)

            # Determine node class name based on column (e.g., "hpa_protein_location" -> "uniprot_subcellular_location")
            node_class_name = self._infer_node_class_name(column)

            # Collect unique values for node creation
            unique_values = set()
            property_updates = []

            for row in csv_data:
                uniprot_id = row.get(uniprot_column, "").strip()
                value = row.get(column, "").strip()

                if not uniprot_id or not value or value.upper() == "NA":
                    continue

                unique_values.add(value)

                # Add property update
                property_updates.append({"uniprot_id": uniprot_id, "value": value})

            # Create node class if it doesn't exist
            class_query = """
            MERGE (c:NodeClass {name: $class_name})
            SET c.description = $description,
                c.source_column = $source_column,
                c.created_at = datetime()
            RETURN c
            """

            class_result = self._execute_query(
                class_query,
                {
                    "class_name": node_class_name,
                    "description": f"Node class for {column} with {len(unique_values)} subclasses",
                    "source_column": column,
                },
            )

            if class_result:
                stats.node_classes_created += 1

            # Create individual nodes for each unique value
            for value in unique_values:
                self._sanitize_node_name(value)

                node_query = f"""
                MERGE (n:{node_class_name} {{name: $name}})
                SET n.source_column = $source_column,
                    n.created_at = datetime()
                RETURN n
                """

                node_result = self._execute_query(
                    node_query, {"name": value, "source_column": column}
                )

                if node_result:
                    stats.individual_nodes_created += 1

            # Add properties to proteins and create relationships in batches
            batch_size = 100
            for i in range(0, len(property_updates), batch_size):
                batch = property_updates[i : i + batch_size]

                # Add properties
                for update in batch:
                    prop_query = f"""
                    MERGE (p:Protein:Entity {{uniprot_id: $uniprot_id}})
                    SET p.{property_name} = $value
                    RETURN p
                    """

                    prop_result = self._execute_query(
                        prop_query,
                        {"uniprot_id": update["uniprot_id"], "value": update["value"]},
                    )

                    if prop_result:
                        stats.properties_added += 1
                        proteins_updated.add(update["uniprot_id"])

                # Create relationship to node class
                rel_query = f"""
                MERGE (p:Protein:Entity {{uniprot_id: $uniprot_id}})
                MATCH (n:{node_class_name} {{name: $value}})
                MERGE (p)-[:BELONGS_TO]->(n)
                RETURN p, n
                """
                rel_result = self._execute_query(
                    rel_query,
                    {"uniprot_id": update["uniprot_id"], "value": update["value"]},
                )

                if rel_result:
                    stats.relationships_created += 1

            logger.info(
                f"Created {len(unique_values)} nodes in class {node_class_name} with {len(property_updates)} property+relationship updates"
            )

        stats.proteins_processed = len(proteins_updated)
        return stats

    async def _process_meaningful_node_class_columns(
        self,
        columns: list[str],
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
    ) -> EnrichmentStats:
        """
        Process columns that should create meaningful node classes with relationships - EFFICIENTLY.

        Creates CLASSIFICATION NODES that are LINKED to protein nodes via BELONGS_TO relationships.
        Uses bulk operations instead of processing one node/relationship at a time.
        """

        stats = EnrichmentStats()
        created_classes = set()

        for column in columns:
            analysis = analysis_results[column]
            logger.info(
                f"Bulk processing meaningful node class column {column} with {analysis.unique_count} unique values"
            )

            # Determine node class name
            node_class_name = self._infer_node_class_name(column)

            # Collect unique values and protein mappings efficiently
            unique_values = set()
            relationships_to_create = []  # List of (uniprot_id, value) pairs

            for row in csv_data:
                value = row.get(column, "").strip()
                if value and value.upper() != "NA":
                    # Split by semicolon and clean up each value
                    split_values = [v.strip() for v in value.split(";") if v.strip()]

                    # Get uniprot_id for this row
                    uniprot_id = (
                        row.get(uniprot_column, "").strip() if uniprot_column else None
                    )

                    if uniprot_id:
                        for split_value in split_values:
                            unique_values.add(split_value)
                            relationships_to_create.append(
                                {"uniprot_id": uniprot_id, "value": split_value}
                            )

            if node_class_name not in created_classes:
                created_classes.add(node_class_name)
                stats.node_classes_created += 1

            # Step 1: Bulk create all nodes for this class
            if unique_values:
                node_data = [
                    {"name": value, "source": f"enrichment:{column}"}
                    for value in unique_values
                ]

                bulk_node_query = f"""
                UNWIND $node_data AS data
                MERGE (n:{node_class_name} {{name: data.name}})
                SET n.source = data.source
                RETURN n
                """

                try:
                    result = self._execute_query(
                        bulk_node_query, {"node_data": node_data}
                    )
                    if result:
                        stats.individual_nodes_created += len(unique_values)
                        logger.info(
                            f"Bulk created {len(unique_values)} {node_class_name} nodes"
                        )
                except Exception as e:
                    logger.error(f"Failed to bulk create {node_class_name} nodes: {e}")
                    continue

            # Step 2: Bulk create all relationships
            if relationships_to_create and uniprot_column:
                # Process relationships in large batches
                batch_size = 1000

                for i in range(0, len(relationships_to_create), batch_size):
                    batch = relationships_to_create[i : i + batch_size]

                    bulk_rel_query = f"""
                    UNWIND $relationships AS rel
                    MATCH (p:Protein {{uniprot_id: rel.uniprot_id}})
                    MATCH (n:{node_class_name} {{name: rel.value}})
                    MERGE (p)-[:BELONGS_TO]->(n)
                    RETURN count(*) as created_count
                    """

                    try:
                        result = self._execute_query(
                            bulk_rel_query, {"relationships": batch}
                        )
                        if result:
                            created_count = result[0].get("created_count", 0)
                            stats.relationships_created += created_count
                            logger.info(
                                f"Bulk created {created_count} BELONGS_TO relationships for {node_class_name}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to bulk create relationships for {node_class_name}: {e}"
                        )
                        # Fallback to individual relationship creation
                        logger.info(
                            f"Falling back to individual relationship creation for {node_class_name}"
                        )
                        for rel in batch:
                            try:
                                individual_rel_query = f"""
                                MATCH (p:Protein {{uniprot_id: $uniprot_id}})
                                MATCH (n:{node_class_name} {{name: $value}})
                                MERGE (p)-[:BELONGS_TO]->(n)
                                """
                                self._execute_query(
                                    individual_rel_query,
                                    {
                                        "uniprot_id": rel["uniprot_id"],
                                        "value": rel["value"],
                                    },
                                )
                                stats.relationships_created += 1
                            except Exception as individual_error:
                                logger.warning(
                                    f"Failed individual relationship for {rel['uniprot_id']} -> {rel['value']}: {individual_error}"
                                )

        return stats

    def _infer_node_class_name(self, column_name: str) -> str:
        """
        Infer an appropriate node class name from a column name.

        Examples:
        - "hpa_protein_location" -> "uniprot_subcellular_location"
        - "pathway" -> "biological_pathway"
        - "tissue" -> "expression_tissue"
        """
        column_lower = column_name.lower()

        # Specific mappings for known biological concepts
        if "location" in column_lower or "localization" in column_lower:
            return "uniprot_subcellular_location"
        if "secretome" in column_lower:
            return "secretome_location"
        if "blood_cell_lineage" in column_lower or "blood_lineage" in column_lower:
            return "blood_cell_lineage"
        if "pathway" in column_lower:
            return "biological_pathway"
        if "tissue" in column_lower or "organ" in column_lower:
            return "expression_tissue"
        if "function" in column_lower:
            return "molecular_function"
        if "process" in column_lower:
            return "biological_process"
        if "component" in column_lower:
            return "cellular_component"
        if "disease" in column_lower:
            return "associated_disease"
        if "drug" in column_lower or "compound" in column_lower:
            return "drug_target"
        # Generic fallback
        return f"{column_name.replace('_', '')}"

    async def _create_individual_nodes(
        self,
        class_name: str,
        unique_values: set[str],
        column: str,
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
        multi_value: bool = False,
    ) -> EnrichmentStats:
        """Create individual nodes for a class."""

        stats = EnrichmentStats()

        # Create individual nodes for each unique value using batching
        if unique_values:
            node_data = [
                {"value": value, "source_column": column} for value in unique_values
            ]
            batch_size = 10  # Reduced from 100 to 10

            for i in range(0, len(node_data), batch_size):
                batch = node_data[i : i + batch_size]
                logger.info(
                    f"Creating node batch {i // batch_size + 1}/{(len(node_data) + batch_size - 1) // batch_size} for {class_name}"
                )

                try:
                    query = f"""
                    UNWIND $nodes AS node
                    MERGE (n:{class_name} {{name: node.value}})
                    SET n.source_column = node.source_column,
                        n.created_at = datetime()
                    RETURN count(n) as node_count
                    """

                    result = self._execute_query(query, {"nodes": batch})

                    if result and result[0].get("node_count", 0) > 0:
                        stats.individual_nodes_created += result[0]["node_count"]

                    # Add delay between batches to reduce database load
                    if i + batch_size < len(node_data):
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Failed to create node batch for {class_name}: {e}")
                    continue

        # Create relationships to proteins
        if uniprot_column:
            relationship_stats = await self._create_protein_relationships(
                class_name, column, csv_data, uniprot_column, multi_value
            )
            stats.relationships_created += relationship_stats.relationships_created

        return stats

    async def _create_protein_relationships(
        self,
        class_name: str,
        column: str,
        csv_data: list[dict[str, Any]],
        uniprot_column: str,
        multi_value: bool = False,
    ) -> EnrichmentStats:
        """Create relationships between proteins and class nodes."""

        stats = EnrichmentStats()
        relationship_name = f"HAS_{class_name.upper()}"

        # Collect all relationships for batching
        relationships = []

        for row in csv_data:
            uniprot_id = row.get(uniprot_column, "").strip()
            value = row.get(column, "").strip()

            if not uniprot_id or not value or value.upper() == "NA":
                continue

            if multi_value:
                # Handle multiple values
                values = self._split_multi_value(value)
                for single_value in values:
                    relationships.append(
                        {
                            "uniprot_id": uniprot_id,
                            "value": single_value.strip(),
                            "source_column": column,
                        }
                    )
            else:
                # Single value
                relationships.append(
                    {"uniprot_id": uniprot_id, "value": value, "source_column": column}
                )

        # Process relationships in batches
        batch_size = 10  # Reduced from 100 to 10

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]
            logger.info(
                f"Processing relationship batch {i // batch_size + 1}/{(len(relationships) + batch_size - 1) // batch_size} for {class_name}"
            )

            try:
                if multi_value:
                    # For multi-value, we already expanded the relationships above
                    query = f"""
                    UNWIND $relationships AS rel
                    MERGE (p:Protein:Entity {{uniprot_id: rel.uniprot_id}})
                    MATCH (n:{class_name} {{name: rel.value}})
                    MERGE (p)-[r:{relationship_name}]->(n)
                    SET r.source_column = rel.source_column
                    RETURN count(r) as relationship_count
                    """
                else:
                    query = f"""
                    UNWIND $relationships AS rel
                    MERGE (p:Protein:Entity {{uniprot_id: rel.uniprot_id}})
                    MATCH (n:{class_name} {{name: rel.value}})
                    MERGE (p)-[r:{relationship_name}]->(n)
                    SET r.source_column = rel.source_column
                    RETURN count(r) as relationship_count
                    """

                result = self._execute_query(query, {"relationships": batch})

                if result and result[0].get("relationship_count", 0) > 0:
                    stats.relationships_created += result[0]["relationship_count"]

                # Add delay between batches to reduce database load
                if i + batch_size < len(relationships):
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(
                    f"Failed to process relationship batch for {class_name}: {e}"
                )
                continue

        return stats

    def _sanitize_property_name(self, name: str) -> str:
        """Sanitize column name for use as Cypher property."""
        # Remove special characters and spaces, replace with underscores
        sanitized = re.sub(r"[^\w]", "_", name)
        # Remove multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"col_{sanitized}"
        return sanitized or "unknown_property"

    def _create_class_name(self, column: str) -> str:
        """Create a class name from column name."""
        # Convert to title case and remove special characters
        words = re.findall(r"\w+", column)
        class_name = "".join(word.capitalize() for word in words)
        return class_name or "UnknownClass"

    def _sanitize_node_name(self, name: str) -> str:
        """Sanitize value for use as node name."""
        # Keep original value but limit length
        return name[:100] if len(name) > 100 else name

    def _split_multi_value(self, value: str) -> list[str]:
        """Split multi-value string into individual values."""
        separators = [";", "|", ",", "/"]

        for sep in separators:
            if sep in value:
                return [v.strip() for v in value.split(sep) if v.strip()]

        return [value.strip()] if value.strip() else []

    def _convert_value_type(self, value: str, column_type: ColumnType) -> str:
        """Convert string value to appropriate type representation."""

        # Normalize the values but keep them as strings

        if column_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            try:
                # Validate it's a number but return as string
                if "." not in value:
                    int(value)  # Validate integer
                else:
                    float(value)  # Validate float
                return value  # Return original string
            except ValueError:
                return value  # Return as string if conversion fails

        elif column_type == ColumnType.BINARY:
            # Normalize binary values to consistent strings
            value_lower = value.lower().strip()
            if value_lower in ["true", "yes", "1", "y", "positive", "pos"]:
                return "true"
            if value_lower in ["false", "no", "0", "n", "negative", "neg"]:
                return "false"
            return value

        else:
            return value  # Return as string for all other types

    def get_supported_strategies(self) -> list[ColumnStrategy]:
        """Return list of column strategies this processor supports."""
        return [ColumnStrategy.PROPERTY, ColumnStrategy.MEANINGFUL_NODE_CLASS]
