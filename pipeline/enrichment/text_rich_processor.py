"""
Text Rich Processor - Handles KG extraction from text-rich columns.

This class manages the second phase of enrichment for columns identified as text-rich:
1. Process biological text descriptions using LLM services
2. Extract entities and relationships from text
3. Create new nodes and classes in the knowledge graph
4. Handle batch processing and error recovery
"""

import json
import logging
from datetime import datetime
from typing import Any, cast

from src.core.database import Neo4jDatabase
from src.factories.database_factory import DatabaseFactory

from .base import (
    BaseEnrichmentProcessor,
    ColumnAnalysis,
    ColumnStrategy,
    EnrichmentStats,
)

logger = logging.getLogger(__name__)


class TextRichProcessor(BaseEnrichmentProcessor):
    """Processes text-rich columns using KG extraction."""

    def __init__(self, database_factory: DatabaseFactory, database: str = "cvd1"):
        """
        Initialize text-rich processor.

        Args:
            database_factory: Factory for database connections
            database: Database name to use
        """
        self.database_factory = database_factory
        self.database = database
        self.extraction_counts = {"extractions": 0, "entities": 0, "relationships": 0}

        # Initialize Neo4j database connection
        self.db = Neo4jDatabase(database=database)

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using Neo4j."""
        return cast("Neo4jDatabase", self.db)._execute_cypher(query, parameters)

    async def process(
        self, data: list[dict[str, Any]], analysis: dict[str, ColumnAnalysis]
    ) -> dict[str, Any]:
        """Process data according to the text-rich processing logic."""
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

        return {"processor_type": "text_rich", "stats": stats, "success": True}

    async def process_columns(
        self,
        analysis_results: dict[str, ColumnAnalysis],
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None = None,
        service: str = "llama3",
        **kwargs,
    ) -> EnrichmentStats:
        """
        Process all text-rich columns using KG extraction.

        Args:
            analysis_results: Column analysis results
            csv_data: CSV data rows
            uniprot_column: Name of UniProt ID column if available
            service: LLM service to use for extraction
            **kwargs: Additional processing parameters

        Returns:
            EnrichmentStats with processing results
        """
        logger.info("Starting text-rich column processing with KG extraction")
        start_time = datetime.now()

        # Find text-rich columns
        text_rich_columns = [
            col_name
            for col_name, analysis in analysis_results.items()
            if analysis.strategy == ColumnStrategy.KG_EXTRACTION
        ]

        if not text_rich_columns:
            logger.info("No text-rich columns found for KG extraction")
            return EnrichmentStats(
                proteins_processed=0,
                properties_added=0,
                node_classes_created=0,
                individual_nodes_created=0,
                relationships_created=0,
                kg_extractions_performed=0,
            )

        logger.info(
            f"Processing {len(text_rich_columns)} text-rich columns: {text_rich_columns}"
        )

        # Process each text-rich column
        total_extractions = 0
        total_entities = 0
        total_relationships = 0
        errors = []

        for column in text_rich_columns:
            try:
                # Check if this column was manually set for KG extraction
                analysis = analysis_results.get(column)
                force_kg = False
                if analysis:
                    logger.info(
                        f"Column {column} analysis: type={analysis.column_type}, strategy={analysis.strategy}, reason='{analysis.reason}'"
                    )
                    if analysis.reason and "Manually set" in analysis.reason:
                        force_kg = True
                        logger.info(
                            f"Column {column} was manually set for KG extraction - bypassing heuristics"
                        )
                    elif analysis.strategy == ColumnStrategy.KG_EXTRACTION:
                        force_kg = True
                        logger.info(
                            f"Column {column} has KG_EXTRACTION strategy - forcing KG extraction"
                        )

                extractions, entities, relationships = await self._process_text_column(
                    column, csv_data, uniprot_column, service, force_kg=force_kg
                )
                total_extractions += extractions
                total_entities += entities
                total_relationships += relationships

                logger.info(
                    f"✅ Column {column}: {extractions} extractions, {entities} entities, {relationships} relationships"
                )

            except Exception as e:
                error_msg = f"Error processing column {column}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Log summary
        logger.info("Text-rich processing complete:")
        logger.info(
            f"  • Columns processed: {len(text_rich_columns) - len(errors)}/{len(text_rich_columns)}"
        )
        logger.info(f"  • Total extractions: {total_extractions}")
        logger.info(f"  • Entities created: {total_entities}")
        logger.info(f"  • Relationships created: {total_relationships}")
        logger.info(f"  • Processing time: {processing_time:.2f}s")

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during processing")
            for error in errors:
                logger.warning(f"  • {error}")

        return EnrichmentStats(
            proteins_processed=len(text_rich_columns) - len(errors),
            individual_nodes_created=total_entities,
            relationships_created=total_relationships,
            kg_extractions_performed=total_extractions,
            errors=errors or None,
        )

    async def _process_text_column(
        self,
        column: str,
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
        service: str,
        force_kg: bool = False,
    ) -> tuple[int, int, int]:
        """
        Process a single text-rich column for KG extraction.

        Args:
            column: Column name to process
            csv_data: CSV data rows
            uniprot_column: UniProt ID column for linking
            service: LLM service to use
            force_kg: Whether to force KG extraction even for short text

        Returns:
            Tuple of (extractions_count, entities_count, relationships_count)
        """
        logger.info(f"Processing text column: {column} (force_kg={force_kg})")

        # Collect text values and analyze their characteristics
        text_values = []
        for row in csv_data:
            text_value = row.get(column, "").strip()
            if text_value and text_value.upper() != "NA":
                text_values.append(text_value)

        if not text_values:
            logger.warning(f"No valid text found in column {column}")
            return 0, 0, 0

        # Analyze text characteristics to determine processing strategy
        avg_length = sum(len(val) for val in text_values) / len(text_values)
        max_length = max(len(val) for val in text_values)

        # Check if this looks like short protein descriptions or similar structured data
        has_keyword = any(
            keyword in column.lower()
            for keyword in ["protein", "gene", "name", "description"]
        )

        logger.info(f"Column {column} analysis:")
        logger.info(f"  • Average length: {avg_length:.1f} (threshold: <100)")
        logger.info(f"  • Max length: {max_length} (threshold: <300)")
        logger.info(f"  • Entry count: {len(text_values)} (threshold: >10)")
        logger.info(
            f"  • Has keyword: {has_keyword} (keywords: protein, gene, name, description)"
        )

        is_short_structured = (
            avg_length < 100  # Short average length
            and max_length < 300  # No very long texts
            and len(text_values)
            > 10  # Reasonable number of entries (lowered threshold)
            and has_keyword
        )

        if is_short_structured and not force_kg:
            logger.info(
                f"✅ Column {column} qualifies for structured data optimization - using efficient property-based processing"
            )
            return await self._process_as_structured_properties(
                column, csv_data, uniprot_column, text_values
            )

        if is_short_structured and force_kg:
            logger.info(
                f"⚠️ Column {column} qualifies for optimization but force_kg is enabled - proceeding with full KG extraction"
            )

        logger.info(
            f"Column {column} contains complex text (avg_len={avg_length:.1f}), using full KG extraction"
        )

        # Group documents by text content to avoid duplicate KG extraction
        text_to_documents = {}
        for i, row in enumerate(csv_data):
            text_value = row.get(column, "").strip()

            if not text_value or text_value.upper() == "NA":
                continue

            # Handle protein ID arrays - create one document per protein ID
            uniprot_ids = []
            if uniprot_column and row.get(uniprot_column):
                uniprot_value = row.get(uniprot_column)
                if isinstance(uniprot_value, list):
                    # Already a list/array
                    uniprot_ids = uniprot_value
                elif isinstance(uniprot_value, str):
                    # Try to parse as JSON array or comma-separated string
                    try:
                        # Try JSON parsing first
                        parsed = json.loads(uniprot_value)
                        uniprot_ids = parsed if isinstance(parsed, list) else [parsed]
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to comma-separated parsing
                        uniprot_ids = [
                            id.strip() for id in uniprot_value.split(",") if id.strip()
                        ]
                else:
                    # Single value, convert to list
                    uniprot_ids = [str(uniprot_value)]
            else:
                # No protein column, use row index
                uniprot_ids = [f"row_{i}"]

            # Create one document per protein ID
            for protein_id in uniprot_ids:
                doc_id = f"{protein_id}_{column}_{i}"

                document = {
                    "id": doc_id,
                    "text": text_value,
                    "metadata": {
                        "source_column": column,
                        "row_index": i,
                        "uniprot_id": protein_id,
                        "all_protein_ids_in_row": uniprot_ids,  # Keep track of all IDs in this row
                    },
                }

                # Group documents by text content
                if text_value not in text_to_documents:
                    text_to_documents[text_value] = []
                text_to_documents[text_value].append(document)

        # Create unique documents (one per unique text)
        unique_documents = []
        for text_value, documents in text_to_documents.items():
            # Use the first document's ID as the representative
            unique_doc = documents[0].copy()
            unique_doc["metadata"]["duplicate_count"] = len(documents)
            unique_doc["metadata"]["original_documents"] = documents
            unique_documents.append(unique_doc)

        logger.info(
            f"Created {len(unique_documents)} unique documents from {len(text_to_documents)} unique texts "
            f"(total {sum(len(docs) for docs in text_to_documents.values())} documents, "
            f"saved {sum(len(docs) for docs in text_to_documents.values()) - len(unique_documents)} duplicate extractions)"
        )

        # Process unique documents through KG pipeline
        try:
            extraction_results = await self._extract_kg_from_documents(
                unique_documents, service
            )

            # Distribute results back to all original documents
            all_extraction_results = []
            for unique_doc, extraction_result in zip(
                unique_documents, extraction_results, strict=False
            ):
                original_documents = unique_doc["metadata"]["original_documents"]

                # Create a copy of the extraction result for each original document
                for original_doc in original_documents:
                    result_copy = extraction_result.copy()
                    result_copy["doc_id"] = original_doc[
                        "id"
                    ]  # Update doc_id to match original
                    all_extraction_results.append(result_copy)

            # Count results across all documents
            total_entities = 0
            total_relationships = 0

            for result in all_extraction_results:
                if "entities" in result:
                    total_entities += len(result["entities"])
                if "relationships" in result:
                    total_relationships += len(result["relationships"])

            return len(all_extraction_results), total_entities, total_relationships

        except Exception as e:
            logger.error(f"KG extraction failed for column {column}: {e}")
            raise

    async def _extract_kg_from_documents(
        self, documents: list[dict[str, Any]], service: str
    ) -> list[dict[str, Any]]:
        """
        Extract knowledge graph data from documents using the KG pipeline.

        Args:
            documents: List of document dictionaries with id, text, and metadata
            service: LLM service to use for extraction

        Returns:
            List of extraction results with entities and relationships
        """
        try:
            # Import required classes (lazy import to avoid circular dependencies)
            from pipeline.ingest.ingestor import Chunk, ProcessedDocument
            from pipeline.ingest.kg_pipeline import KGPipeline

            logger.info("Using KGPipeline for KG extraction")

            kg_pipeline = KGPipeline(
                service=service,
                database=self.database,
                enable_consolidation=True,
            )

            # Convert documents to ProcessedDocument format
            processed_docs = []
            for doc in documents:
                # Create a single chunk from the document text
                chunk = Chunk(chunk_id=f"{doc['id']}_chunk_0", text=doc["text"])

                # Create ProcessedDocument
                processed_doc = ProcessedDocument(
                    doc_id=doc["id"],
                    source=doc[
                        "text"
                    ],  # Use the actual text content, not the column name
                    metadata=doc.get("metadata", {}),
                    chunks=[chunk],
                )
                processed_docs.append(processed_doc)

            # Process documents through KG pipeline
            logger.info(
                f"Processing {len(processed_docs)} documents through KG pipeline"
            )
            results = await kg_pipeline.process_documents(
                processed_docs, cleanup_existing=False
            )

            # Convert results to expected format and count entities/relationships
            extraction_results = []
            for result_doc in results:
                extraction_result = {
                    "doc_id": result_doc.doc_id,
                    "entities": [],
                    "relationships": [],
                }

                # Count entities and relationships from chunks
                for chunk in result_doc.chunks:
                    # Count entities (these would be stored as separate nodes)
                    if hasattr(chunk, "nodes") and chunk.nodes:
                        extraction_result["entities"].extend(chunk.nodes)

                    # Count relationships
                    if hasattr(chunk, "relationships") and chunk.relationships:
                        extraction_result["relationships"].extend(chunk.relationships)

                extraction_results.append(extraction_result)

            logger.info(
                f"Successfully processed {len(extraction_results)}/{len(documents)} documents"
            )
            return extraction_results

        except ImportError as e:
            logger.error(f"Could not import KG pipeline dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"KG extraction failed: {e}")
            raise

    async def _process_as_structured_properties(
        self,
        column: str,
        csv_data: list[dict[str, Any]],
        uniprot_column: str | None,
        text_values: list[str],
    ) -> tuple[int, int, int]:
        """
        Process short structured text (like protein descriptions) as simple properties.

        This is much more efficient than full KG extraction for structured data like
        protein names, gene descriptions, etc.

        Args:
            column: Column name being processed
            csv_data: CSV data rows
            uniprot_column: UniProt ID column for linking
            text_values: List of text values from the column

        Returns:
            Tuple of (properties_added_count, 0, 0) - no entities or relationships created
        """
        logger.info(
            f"Processing {len(text_values)} structured properties for column {column}"
        )

        try:
            # Get database connection using stored database name
            self.database_factory.create_database(database=self.database)

            properties_added = 0
            batch_size = 100

            # Process in batches for efficiency
            for i in range(0, len(csv_data), batch_size):
                batch_rows = csv_data[i : i + batch_size]

                # Prepare batch updates
                updates = []
                for row in batch_rows:
                    text_value = row.get(column, "").strip()

                    # Handle protein ID arrays
                    uniprot_ids = []
                    if uniprot_column and row.get(uniprot_column):
                        uniprot_value = row.get(uniprot_column)
                        if isinstance(uniprot_value, list):
                            # Already a list/array
                            uniprot_ids = uniprot_value
                        elif isinstance(uniprot_value, str):
                            # Try to parse as JSON array or comma-separated string
                            try:
                                # Try JSON parsing first
                                parsed = json.loads(uniprot_value)
                                if isinstance(parsed, list):
                                    uniprot_ids = parsed
                                else:
                                    uniprot_ids = [parsed]
                            except (json.JSONDecodeError, TypeError):
                                # Fall back to comma-separated parsing
                                uniprot_ids = [
                                    id.strip()
                                    for id in uniprot_value.split(",")
                                    if id.strip()
                                ]
                        else:
                            # Single value, convert to list
                            uniprot_ids = [str(uniprot_value)]

                    # Create one update per protein ID
                    for protein_id in uniprot_ids:
                        if text_value and text_value.upper() != "NA" and protein_id:
                            updates.append(
                                {
                                    "uniprot_id": protein_id,
                                    "property_name": column,
                                    "property_value": text_value,
                                }
                            )

                if updates:
                    # Add properties to existing protein nodes
                    self._batch_add_properties(updates)
                    properties_added += len(updates)

                    logger.debug(
                        f"Added {len(updates)} properties in batch {i // batch_size + 1}"
                    )

            logger.info(
                f"Successfully added {properties_added} properties for column {column}"
            )
            return properties_added, 0, 0  # (extractions, entities, relationships)

        except Exception as e:
            logger.error(
                f"Failed to process structured properties for column {column}: {e}"
            )
            raise

    def _batch_add_properties(self, updates: list[dict[str, str]]):
        """Add properties to existing nodes individually."""
        if not updates:
            return

        for update in updates:
            try:
                # Use string formatting for dynamic property names
                property_name = update["property_name"]
                individual_query = f"""
                MERGE (p:Protein:Entity {{uniprot_id: $uniprot_id}})
                SET p.`{property_name}` = $property_value
                """
                self._execute_query(
                    individual_query,
                    {
                        "uniprot_id": update["uniprot_id"],
                        "property_value": update["property_value"],
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to add property {update['property_name']} to {update['uniprot_id']}: {e}"
                )

    def get_supported_strategies(self) -> list[ColumnStrategy]:
        """Return list of column strategies this processor supports."""
        return [ColumnStrategy.KG_EXTRACTION]
