"""
Table Extraction and Processing
Extracts structured data from tables in biomedical papers,
maps to entity types, and integrates with knowledge graph.
"""

import json
from typing import Any

from pipeline.enrichment.column_analyzer import ColumnAnalyzer
from pipeline.processors.multimodal_processor import ExtractedTable, MultimodalDocument
from src.factories.llm_factory import get_llm
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class TableProcessor:
    """
    Processes tables extracted from biomedical documents.
    Uses LLM to interpret structure and map to biomedical entities.
    """

    def __init__(self, llm_service: str = "bedrock", model_id: str | None = None):
        """
        Initialize table processor.

        Args:
            llm_service: LLM service for table interpretation
            model_id: Optional model ID override
        """
        self.llm_service = llm_service
        self.model_id = model_id

        # Initialize LLM for table interpretation
        llm_kwargs = {}
        if model_id:
            llm_kwargs["model_id"] = model_id

        self.llm = get_llm(llm_service, **llm_kwargs)
        self.column_analyzer = ColumnAnalyzer(self.llm)

        logger.info(f"Initialized TableProcessor with {llm_service}")

    def analyze_table(
        self, table: ExtractedTable, context: str | None = None
    ) -> dict[str, Any]:
        """
        Analyze table structure and content to extract biomedical entities.

        Args:
            table: Extracted table data
            context: Optional surrounding text for context

        Returns:
            Dict with table analysis and extracted entities
        """
        logger.info(f"Analyzing table {table.table_id} ({len(table.rows)} rows)")

        # Step 1: Classify columns
        column_types = self._classify_columns(table)

        # Step 2: Identify entity columns (Protein, Disease, etc.)
        entity_columns = self._identify_entity_columns(table, column_types, context)

        # Step 3: Extract entities from table
        entities = self._extract_entities_from_table(table, entity_columns)

        # Step 4: Extract relationships (if table shows associations)
        relationships = self._extract_relationships_from_table(table, entity_columns)

        result = {
            "table_id": table.table_id,
            "page": table.page_number,
            "caption": table.caption,
            "column_count": len(table.headers),
            "row_count": len(table.rows),
            "column_types": column_types,
            "entity_columns": entity_columns,
            "entities": entities,
            "relationships": relationships,
            "summary": self._generate_table_summary(table, entities, relationships),
        }

        logger.info(
            f"Table analysis complete: {len(entities)} entities, "
            f"{len(relationships)} relationships"
        )

        return result

    def _classify_columns(self, table: ExtractedTable) -> dict[str, str]:
        """
        Classify each column's data type.

        Args:
            table: Extracted table

        Returns:
            Dict mapping column name to type
        """
        column_types = {}

        for col_idx, header in enumerate(table.headers):
            # Extract column values
            col_values = [
                row[col_idx] if col_idx < len(row) else "" for row in table.rows
            ]

            # Simple heuristic classification
            if self._is_numeric_column(col_values):
                column_types[header] = "numeric"
            elif self._is_identifier_column(col_values):
                column_types[header] = "identifier"
            elif self._is_categorical_column(col_values):
                column_types[header] = "categorical"
            else:
                column_types[header] = "text"

        return column_types

    def _identify_entity_columns(
        self,
        table: ExtractedTable,
        column_types: dict[str, str],
        context: str | None = None,
    ) -> dict[str, str]:
        """
        Use LLM to identify which columns contain biomedical entities.

        Args:
            table: Extracted table
            context: Optional surrounding text
            column_types: Column type classifications

        Returns:
            Dict mapping column name to entity type (Protein, Disease, etc.)
        """
        # Create prompt for LLM
        prompt = self._create_entity_identification_prompt(table, column_types, context)

        try:
            response = self.llm.invoke(prompt)

            # Parse LLM response
            return self._parse_entity_identification_response(response)

        except Exception as e:
            logger.warning(f"LLM entity identification failed: {e}")
            # Fallback: use header name heuristics
            return self._fallback_entity_identification(table.headers)

    def _create_entity_identification_prompt(
        self,
        table: ExtractedTable,
        column_types: dict[str, str],
        context: str | None = None,
    ) -> str:
        """Create prompt for LLM to identify entity columns."""

        # Show first few rows as examples
        sample_rows = table.rows[:5]
        table_preview = f"Headers: {table.headers}\n"
        table_preview += f"Column types: {column_types}\n"
        table_preview += "Sample rows:\n"
        for row in sample_rows:
            table_preview += f"  {row}\n"

        return f"""Analyze this biomedical table and identify which columns contain specific entity types.

{table_preview}

{f"Table caption: {table.caption}" if table.caption else ""}

{f"Surrounding context: {context[:500]}" if context else ""}

For each column, determine if it contains:
- Protein (protein names, gene names, UniProt IDs)
- Disease (disease names, conditions, phenotypes)
- Pathway (signaling pathways, biological processes)
- Drug (drug names, compounds, treatments)
- CellType (cell types, tissue types)
- Measurement (numerical measurements, expression levels, concentrations)
- Other (anything else)

Return your analysis as JSON:
{{
  "entity_columns": {{
    "column_name": "entity_type",
    ...
  }},
  "key_columns": ["primary entity column", "secondary entity column"],
  "relationship_type": "protein_disease_association|protein_protein_interaction|expression_data|clinical_data|other"
}}

Provide valid JSON only:"""

    def _parse_entity_identification_response(self, response: str) -> dict[str, str]:
        """Parse LLM response for entity column identification."""

        try:
            # Extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            else:
                json_text = response

            parsed = json.loads(json_text)
            return parsed.get("entity_columns", {})

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity identification JSON: {e}")
            return {}

    def _fallback_entity_identification(self, headers: list[str]) -> dict[str, str]:
        """Fallback heuristic-based entity identification."""

        entity_columns = {}

        for header in headers:
            header_lower = header.lower()

            if any(
                term in header_lower
                for term in ["protein", "gene", "uniprot", "symbol"]
            ):
                entity_columns[header] = "Protein"
            elif any(
                term in header_lower
                for term in ["disease", "condition", "phenotype", "disorder"]
            ):
                entity_columns[header] = "Disease"
            elif any(
                term in header_lower for term in ["pathway", "process", "function"]
            ):
                entity_columns[header] = "Pathway"
            elif any(
                term in header_lower
                for term in ["drug", "compound", "treatment", "inhibitor"]
            ):
                entity_columns[header] = "Drug"
            elif any(term in header_lower for term in ["cell", "tissue", "organ"]):
                entity_columns[header] = "CellType"
            elif any(
                term in header_lower
                for term in [
                    "fold",
                    "expression",
                    "level",
                    "concentration",
                    "p-value",
                    "fc",
                ]
            ):
                entity_columns[header] = "Measurement"

        return entity_columns

    def _extract_entities_from_table(
        self, table: ExtractedTable, entity_columns: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Extract individual entities from table rows."""

        entities = []

        # Get column indices for entity columns
        entity_col_indices = {
            header: idx
            for idx, header in enumerate(table.headers)
            if header in entity_columns
        }

        for row_idx, row in enumerate(table.rows):
            for header, col_idx in entity_col_indices.items():
                if col_idx < len(row):
                    entity_value = row[col_idx]
                    entity_type = entity_columns[header]

                    if entity_value and entity_value.strip():
                        entity = {
                            "name": entity_value.strip(),
                            "type": entity_type,
                            "source_table": table.table_id,
                            "source_column": header,
                            "row_index": row_idx,
                            "metadata": self._extract_row_metadata(table, row, row_idx),
                        }
                        entities.append(entity)

        return entities

    def _extract_relationships_from_table(
        self, table: ExtractedTable, entity_columns: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Extract relationships between entities in table."""

        relationships = []

        # If table has 2+ entity columns, create relationships
        entity_cols = list(entity_columns)

        if len(entity_cols) >= 2:
            # Assume first two entity columns are source and target
            source_col = entity_cols[0]
            target_col = entity_cols[1]

            source_idx = table.headers.index(source_col)
            target_idx = table.headers.index(target_col)

            for row_idx, row in enumerate(table.rows):
                if source_idx < len(row) and target_idx < len(row):
                    source_entity = row[source_idx]
                    target_entity = row[target_idx]

                    if source_entity and target_entity:
                        # Determine relationship type based on entity types
                        source_type = entity_columns[source_col]
                        target_type = entity_columns[target_col]
                        rel_type = self._infer_relationship_type(
                            source_type, target_type
                        )

                        relationship = {
                            "source": source_entity.strip(),
                            "source_type": source_type,
                            "target": target_entity.strip(),
                            "target_type": target_type,
                            "relationship_type": rel_type,
                            "source_table": table.table_id,
                            "row_index": row_idx,
                            "metadata": self._extract_row_metadata(table, row, row_idx),
                        }
                        relationships.append(relationship)

        return relationships

    def _infer_relationship_type(self, source_type: str, target_type: str) -> str:
        """Infer relationship type based on entity types."""

        if source_type == "Protein" and target_type == "Disease":
            return "ASSOCIATED_WITH"
        if source_type == "Protein" and target_type == "Protein":
            return "INTERACTS_WITH"
        if source_type == "Protein" and target_type == "Pathway":
            return "PARTICIPATES_IN"
        if source_type == "Drug" and target_type == "Protein":
            return "TARGETS"
        return "RELATED_TO"

    def _extract_row_metadata(
        self, table: ExtractedTable, row: list[str], row_idx: int
    ) -> dict[str, Any]:
        """Extract metadata from row (e.g., p-values, fold changes)."""

        metadata = {}

        for col_idx, header in enumerate(table.headers):
            if col_idx < len(row):
                value = row[col_idx]
                header_lower = header.lower()

                # Capture common experimental metrics
                if "p-value" in header_lower or "pvalue" in header_lower:
                    try:
                        metadata["p_value"] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif "fold" in header_lower or "fc" in header_lower:
                    try:
                        metadata["fold_change"] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif "confidence" in header_lower:
                    try:
                        metadata["confidence"] = float(value)
                    except (ValueError, TypeError):
                        pass

        return metadata

    def _generate_table_summary(
        self,
        table: ExtractedTable,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> str:
        """Generate human-readable summary of table contents."""

        entity_counts = {}
        for entity in entities:
            entity_type = entity["type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        summary = (
            f"Table with {len(table.rows)} rows and {len(table.headers)} columns. "
        )
        summary += f"Contains {len(entities)} entities: "
        summary += ", ".join(
            f"{count} {etype}" for etype, count in entity_counts.items()
        )

        if relationships:
            summary += f". {len(relationships)} relationships identified."

        return summary

    def _is_numeric_column(self, values: list[str]) -> bool:
        """Check if column contains primarily numeric values."""
        numeric_count = sum(1 for v in values if self._is_numeric(v))
        return numeric_count / len(values) > 0.7 if values else False

    def _is_identifier_column(self, values: list[str]) -> bool:
        """Check if column contains identifiers (UniProt IDs, etc.)."""
        # Simple heuristic: alphanumeric patterns
        identifier_patterns = ["P[0-9]{5}", "ENSG", "Q[0-9]{5}"]
        return any(
            any(pattern in v for pattern in identifier_patterns) for v in values[:10]
        )

    def _is_categorical_column(self, values: list[str]) -> bool:
        """Check if column contains categorical values."""
        unique_ratio = len(set(values)) / len(values) if values else 0
        return unique_ratio < 0.3

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Check if string represents a number."""
        try:
            float(value.replace(",", "").replace("%", ""))
            return True
        except (ValueError, AttributeError):
            return False

    def process_document_tables(
        self, multimodal_doc: MultimodalDocument, max_tables: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Process all tables in a multimodal document.

        Args:
            multimodal_doc: Document containing tables
            max_tables: Optional limit on number of tables

        Returns:
            List of table analyses
        """
        results = []
        tables_to_process = (
            multimodal_doc.tables[:max_tables] if max_tables else multimodal_doc.tables
        )

        logger.info(
            f"Processing {len(tables_to_process)} tables from document {multimodal_doc.doc_id}"
        )

        for table in tables_to_process:
            try:
                # Get context from surrounding text
                context = multimodal_doc.text_content[:1000]

                analysis = self.analyze_table(table, context=context)
                results.append(analysis)

            except Exception as e:
                logger.error(f"Failed to process table {table.table_id}: {e}")
                results.append({"table_id": table.table_id, "error": str(e)})

        logger.info(f"Completed analysis of {len(results)} tables")
        return results
