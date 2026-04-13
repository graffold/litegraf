"""
Column Analyzer - Analyzes CSV columns to determine data types and enrichment strategies.

This class handles the first phase of the enrichment hierarchy:
1. Detect column sparsity and data quality
2. Identify text-rich columns for KG extraction
3. Classify columns by data type and content
4. Suggest appropriate enrichment strategies
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any

from .base import ColumnAnalysis, ColumnStrategy, ColumnType, ManualColumnHandler

logger = logging.getLogger(__name__)


class ColumnAnalyzer:
    """Analyzes CSV columns to determine optimal enrichment strategies."""

    def __init__(
        self,
        empty_threshold: float = 90.0,
        unique_threshold: float = 20.0,
        meaningful_class_min: int = 11,
        meaningful_class_max: int = 200,
    ):
        """
        Initialize column analyzer with thresholds.

        Args:
            empty_threshold: Percentage threshold for considering columns too empty
            unique_threshold: Percentage threshold for unique values classification
            meaningful_class_min: Minimum unique values for meaningful node classes
            meaningful_class_max: Maximum unique values for meaningful node classes
        """
        self.empty_threshold = empty_threshold
        self.unique_threshold = unique_threshold
        self.meaningful_class_min = meaningful_class_min
        self.meaningful_class_max = meaningful_class_max

        # Manual column handlers override automatic analysis
        self.manual_handlers: dict[int, ManualColumnHandler] = {}

        # Biological keywords for text-rich detection
        self.biological_keywords = [
            "protein",
            "gene",
            "pathway",
            "enzyme",
            "binding",
            "regulation",
            "expression",
            "signaling",
            "metabolism",
            "catalyzes",
            "involved",
            "interaction",
            "phosphorylation",
            "transcription",
            "cell",
            "membrane",
        ]

        # Text-rich column name patterns
        self.text_rich_keywords = [
            "function",
            "description",
            "summary",
            "abstract",
            "mechanism",
            "pathway",
            "interaction",
            "role",
            "activity",
            "process",
            "annotation",
            "comment",
            "note",
            "text",
            "detail",
        ]

    def set_manual_handlers(self, handlers: dict[int, ManualColumnHandler]) -> None:
        """
        Set manual column handlers to override automatic analysis.

        Args:
            handlers: Dictionary mapping column index to ManualColumnHandler
        """
        self.manual_handlers = handlers.copy()
        logger.info(f"Set {len(handlers)} manual column handlers: {handlers}")

    def parse_handler_string(self, handler_str: str) -> ManualColumnHandler:
        """
        Parse shorthand handler string to ManualColumnHandler enum.

        Args:
            handler_str: Shorthand string like 'protein-id', 'prop', 'kg', etc.

        Returns:
            Corresponding ManualColumnHandler enum value

        Raises:
            ValueError: If handler_str is not recognized
        """
        handler_map = {
            # Protein matching
            "protein-id": ManualColumnHandler.PROTEIN_ID,
            "match-protein": ManualColumnHandler.PROTEIN_ID,
            "protein": ManualColumnHandler.PROTEIN_ID,
            "uniprot-id": ManualColumnHandler.PROTEIN_ID,
            # Protein properties
            "protein-prop": ManualColumnHandler.PROTEIN_PROPERTY,
            "prop": ManualColumnHandler.PROTEIN_PROPERTY,
            "property": ManualColumnHandler.PROTEIN_PROPERTY,
            # Knowledge graph extraction
            "kg": ManualColumnHandler.KG_EXTRACTION,
            "kg-extraction": ManualColumnHandler.KG_EXTRACTION,
            "extraction": ManualColumnHandler.KG_EXTRACTION,
            # Node classes
            "node-class": ManualColumnHandler.NODE_CLASS,
            "class": ManualColumnHandler.NODE_CLASS,
            "nodes": ManualColumnHandler.NODE_CLASS,
            # Skip
            "skip": ManualColumnHandler.SKIP,
            "ignore": ManualColumnHandler.SKIP,
            # Future: relationships
            "relationship": ManualColumnHandler.RELATIONSHIP,
            "rel": ManualColumnHandler.RELATIONSHIP,
        }

        handler_str_lower = handler_str.lower().strip()
        if handler_str_lower not in handler_map:
            valid_handlers = ", ".join(sorted(handler_map.keys()))
            raise ValueError(
                f"Unknown handler '{handler_str}'. Valid handlers: {valid_handlers}"
            )

        return handler_map[handler_str_lower]

    def _apply_manual_handler(
        self, analysis: ColumnAnalysis, handler: ManualColumnHandler
    ) -> ColumnAnalysis:
        """
        Apply manual handler override to column analysis.

        Args:
            analysis: Original column analysis
            handler: Manual handler to apply

        Returns:
            Modified column analysis with manual overrides
        """
        if handler == ManualColumnHandler.SKIP:
            return ColumnAnalysis(
                column_name=analysis.column_name,
                column_type=ColumnType.INSUFFICIENT_DATA,
                strategy=ColumnStrategy.SKIP,
                unique_count=analysis.unique_count,
                total_values=analysis.total_values,
                empty_percentage=analysis.empty_percentage,
                na_percentage=analysis.na_percentage,
                total_missing_percentage=analysis.total_missing_percentage,
                reason="Manually set to skip",
            )

        if handler == ManualColumnHandler.PROTEIN_ID:
            return ColumnAnalysis(
                column_name=analysis.column_name,
                column_type=ColumnType.UNIPROT_ID,
                strategy=ColumnStrategy.PROPERTY,  # Will be used to match existing proteins
                unique_count=analysis.unique_count,
                total_values=analysis.total_values,
                empty_percentage=analysis.empty_percentage,
                na_percentage=analysis.na_percentage,
                total_missing_percentage=analysis.total_missing_percentage,
                reason="Manually set as protein ID matcher",
            )

        if handler == ManualColumnHandler.PROTEIN_PROPERTY:
            return ColumnAnalysis(
                column_name=analysis.column_name,
                column_type=analysis.column_type,  # Keep original type detection
                strategy=ColumnStrategy.PROPERTY,
                unique_count=analysis.unique_count,
                total_values=analysis.total_values,
                empty_percentage=analysis.empty_percentage,
                na_percentage=analysis.na_percentage,
                total_missing_percentage=analysis.total_missing_percentage,
                reason="Manually set as protein property",
            )

        if handler == ManualColumnHandler.KG_EXTRACTION:
            return ColumnAnalysis(
                column_name=analysis.column_name,
                column_type=ColumnType.TEXT_RICH,
                strategy=ColumnStrategy.KG_EXTRACTION,
                unique_count=analysis.unique_count,
                total_values=analysis.total_values,
                empty_percentage=analysis.empty_percentage,
                na_percentage=analysis.na_percentage,
                total_missing_percentage=analysis.total_missing_percentage,
                reason="Manually set for KG extraction",
            )

        if handler == ManualColumnHandler.NODE_CLASS:
            return ColumnAnalysis(
                column_name=analysis.column_name,
                column_type=analysis.column_type,  # Keep original type detection
                strategy=ColumnStrategy.MEANINGFUL_NODE_CLASS,
                unique_count=analysis.unique_count,
                total_values=analysis.total_values,
                empty_percentage=analysis.empty_percentage,
                na_percentage=analysis.na_percentage,
                total_missing_percentage=analysis.total_missing_percentage,
                reason="Manually set to create node classes",
            )

        if handler == ManualColumnHandler.RELATIONSHIP:
            # For now, treat as property - future enhancement could create relationships
            return ColumnAnalysis(
                column_name=analysis.column_name,
                column_type=analysis.column_type,
                strategy=ColumnStrategy.PROPERTY,
                unique_count=analysis.unique_count,
                total_values=analysis.total_values,
                empty_percentage=analysis.empty_percentage,
                na_percentage=analysis.na_percentage,
                total_missing_percentage=analysis.total_missing_percentage,
                reason="Manually set for relationship creation (treated as property for now)",
            )

        logger.warning(f"Unknown manual handler {handler}, keeping original analysis")
        return analysis

    async def analyze_csv(
        self, file_path: str
    ) -> tuple[dict[str, ColumnAnalysis], dict[str, Any]]:
        """
        Analyze a CSV file and return column analyses and metadata.

        Args:
            file_path: Path to the CSV file to analyze

        Returns:
            Tuple of (column_analyses, file_metadata)
        """
        logger.info(f"Analyzing CSV structure for {file_path}")

        # Read and parse CSV
        data, headers = self._read_csv(file_path)
        total_rows = len(data)

        # Analyze each column
        column_analyses = {}
        uniprot_column = None
        ensembl_columns = []

        for column in headers:
            column_index = headers.index(column)

            # If manual handlers are specified, only process explicitly listed columns
            if self.manual_handlers and column_index not in self.manual_handlers:
                analysis = ColumnAnalysis(
                    column_name=column,
                    column_type=ColumnType.INSUFFICIENT_DATA,
                    strategy=ColumnStrategy.SKIP,
                    unique_count=0,
                    total_values=0,
                    empty_percentage=100.0,
                    na_percentage=0.0,
                    total_missing_percentage=100.0,
                    reason="Skipped - not in manual column handlers",
                )
                logger.debug(
                    f"Skipping column {column_index} ({column}) - not in manual handlers"
                )
            else:
                analysis = self._analyze_column(column, data, total_rows)

                # Check for manual handler override
                if column_index in self.manual_handlers:
                    manual_handler = self.manual_handlers[column_index]
                    analysis = self._apply_manual_handler(analysis, manual_handler)
                    logger.info(
                        f"Applied manual handler {manual_handler.value} to column {column_index} ({column})"
                    )

            column_analyses[column] = analysis

            # Track special columns
            if analysis.column_type == ColumnType.UNIPROT_ID:
                # Prefer columns that are exactly "uniprot" or "uniprot_id"
                if (
                    column.lower() in ["uniprot", "uniprot_id"]
                    or uniprot_column is None
                ):
                    uniprot_column = column
            elif analysis.column_type == ColumnType.ENSEMBL_ID:
                ensembl_columns.append(column)

        # Detect 1:1 mappings between columns
        # DISABLED: Complex auto-detection of relationships. Only hardcoded columns should create node classes.
        # self._detect_one_to_one_mappings(column_analyses, data)

        # Build metadata
        metadata = {
            "headers": headers,
            "uniprot_column": uniprot_column,
            "ensembl_columns": ensembl_columns,
            "total_rows": total_rows,
            "column_analysis": {
                k: self._analysis_to_dict(v) for k, v in column_analyses.items()
            },
        }

        return column_analyses, metadata

    def _read_csv(self, file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Read CSV, Parquet, or TXT file and return data with headers."""
        file_path_obj = Path(file_path)

        # Detect file type based on extension
        if file_path_obj.suffix.lower() in [".parquet", ".pq"]:
            return self._read_parquet(file_path)
        if file_path_obj.suffix.lower() in [".txt", ".tsv"]:
            return self._read_txt_file(file_path)
        return self._read_csv_file(file_path)

    def _read_csv_file(self, file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Read CSV file and return data with headers."""
        try:
            with open(file_path, encoding="utf-8") as file:
                # Try to detect delimiter
                sample = file.read(1024)
                file.seek(0)

                try:
                    delimiter = csv.Sniffer().sniff(sample).delimiter
                except csv.Error:
                    logger.warning(
                        "Could not auto-detect delimiter: Could not determine delimiter. Trying common delimiters..."
                    )
                    delimiter = ","

                logger.info(f"Using delimiter: '{delimiter}'")

                reader = csv.DictReader(file, delimiter=delimiter)
                headers = list(reader.fieldnames) if reader.fieldnames else []
                data = list(reader)

        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

        return data, headers

    def _read_parquet(self, file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Read Parquet file and return data with headers."""
        try:
            import polars as pl

            logger.info(f"Reading Parquet file: {file_path}")
            df = pl.read_parquet(file_path)

            headers = df.columns

            # Convert to list of dicts (same format as CSV reader)
            # Ensure string keys and convert all values to strings for consistency
            data = []
            for row in df.iter_rows(named=True):
                converted_row = {}
                for k, v in row.items():
                    if isinstance(v, list):
                        v = ", ".join(str(item) for item in v)
                    elif v is None:
                        v = ""
                    else:
                        v = str(v)
                    converted_row[k] = v
                data.append(converted_row)

            logger.info(
                f"Loaded {len(data)} rows with {len(headers)} columns from Parquet file"
            )

        except ImportError:
            raise ImportError(
                "polars is required to read Parquet files. Please install with: pip install polars"
            )
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            raise

        return data, headers

    def _read_txt_file(self, file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Read TXT/TSV file and return data with headers."""
        try:
            with open(file_path, encoding="utf-8") as file:
                # Read first few lines to detect format
                sample_lines = []
                for i, line in enumerate(file):
                    sample_lines.append(line.strip())
                    if i >= 10:  # Read first 11 lines for format detection
                        break

                # Reset file pointer
                file.seek(0)

                # Detect delimiter - common delimiters for text files
                delimiters = ["\t", "|", ",", ";", " "]
                best_delimiter = "\t"  # Default to tab for .txt/.tsv files
                max_consistent_columns = 0

                for delimiter in delimiters:
                    column_counts = []
                    for line in sample_lines:
                        if line.strip():  # Skip empty lines
                            parts = line.split(delimiter)
                            column_counts.append(len(parts))

                    # Check consistency - all lines should have same number of columns
                    if column_counts:
                        most_common_count = max(
                            set(column_counts), key=column_counts.count
                        )
                        consistent_lines = sum(
                            1 for count in column_counts if count == most_common_count
                        )

                        if (
                            consistent_lines > max_consistent_columns
                            and most_common_count > 1
                        ):
                            max_consistent_columns = consistent_lines
                            best_delimiter = delimiter

                logger.info(f"Detected delimiter for TXT file: '{best_delimiter}'")

                # Read the file with detected delimiter
                reader = csv.DictReader(file, delimiter=best_delimiter)
                headers = list(reader.fieldnames) if reader.fieldnames else []
                data = list(reader)

                # If no headers detected, create generic ones
                if not headers and data:
                    first_row = data[0] if data else {}
                    num_columns = len(first_row)
                    headers = [f"column_{i + 1}" for i in range(num_columns)]

                    # Convert first row of data to use generic headers
                    for i, row in enumerate(data):
                        new_row = {}
                        for j, (_key, value) in enumerate(row.items()):
                            new_header = (
                                headers[j] if j < len(headers) else f"column_{j + 1}"
                            )
                            new_row[new_header] = value
                        data[i] = new_row

                logger.info(
                    f"Loaded {len(data)} rows with {len(headers)} columns from TXT file"
                )

        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            raise

        return data, headers

    def _analyze_column(
        self, column: str, data: list[dict[str, Any]], total_rows: int
    ) -> ColumnAnalysis:
        """Analyze a single column and determine its characteristics."""

        # Get all values for this column
        values = [row.get(column, "") for row in data]

        # Convert all values to strings for consistent analysis
        # Handle complex data types (lists, etc.) by converting to string representation
        str_values = []
        has_complex_data = False
        for v in values:
            if isinstance(v, list):
                # Convert lists to comma-separated strings
                str_val = ", ".join(str(item) for item in v)
                has_complex_data = True
            else:
                str_val = str(v) if v is not None else ""
            str_values.append(str_val)

        # Use string values for analysis
        values = str_values

        # Calculate basic statistics
        non_empty_values = [
            v for v in values if v and v.strip() and v.strip().upper() != "NA"
        ]
        empty_count = total_rows - len(non_empty_values)

        empty_percentage = (empty_count / total_rows * 100) if total_rows > 0 else 0
        na_count = sum(1 for v in values if v.strip().upper() == "NA")
        na_percentage = (na_count / total_rows * 100) if total_rows > 0 else 0
        total_missing_percentage = empty_percentage + na_percentage

        # Check if column has insufficient data
        if total_missing_percentage >= self.empty_threshold:
            return ColumnAnalysis(
                column_name=column,
                column_type=ColumnType.INSUFFICIENT_DATA,
                strategy=ColumnStrategy.SKIP,
                unique_count=0,
                total_values=len(non_empty_values),
                empty_percentage=empty_percentage,
                na_percentage=na_percentage,
                total_missing_percentage=total_missing_percentage,
                reason=f"Column has {total_missing_percentage:.1f}% missing values (>= {self.empty_threshold}% threshold)",
            )

        if not non_empty_values:
            return ColumnAnalysis(
                column_name=column,
                column_type=ColumnType.INSUFFICIENT_DATA,
                strategy=ColumnStrategy.SKIP,
                unique_count=0,
                total_values=0,
                empty_percentage=empty_percentage,
                na_percentage=na_percentage,
                total_missing_percentage=total_missing_percentage,
                reason="No non-empty values found",
            )

        # Detect column type and strategy
        column_type, strategy, extra_info = self._classify_column(
            column, non_empty_values
        )

        # If column contains complex data (lists), mark it appropriately
        if has_complex_data and column_type == ColumnType.TEXT_RICH:
            # Complex data in text-rich columns might be problematic for KG extraction
            logger.warning(
                f"Column '{column}' contains complex data (lists) and is marked as text-rich. This may cause issues during KG extraction."
            )

        unique_values = list(set(non_empty_values))
        unique_count = len(unique_values)

        return ColumnAnalysis(
            column_name=column,
            column_type=column_type,
            strategy=strategy,
            unique_count=unique_count,
            total_values=len(non_empty_values),
            empty_percentage=empty_percentage,
            na_percentage=na_percentage,
            total_missing_percentage=total_missing_percentage,
            sample_values=unique_values[:10],  # First 10 unique values as samples
            **extra_info,
        )

    def _classify_column(
        self, column: str, values: list[str]
    ) -> tuple[ColumnType, ColumnStrategy, dict[str, Any]]:
        """Classify column type and determine enrichment strategy."""

        # Check for special ID columns first
        if self._is_uniprot_column(column, values):
            return ColumnType.UNIPROT_ID, ColumnStrategy.SKIP, {}

        if self._is_ensembl_column(column, values):
            return ColumnType.ENSEMBL_ID, ColumnStrategy.PROPERTY, {}

        # Check for metadata/timestamp columns that should be skipped
        if self._is_metadata_column(column, values):
            return (
                ColumnType.INSUFFICIENT_DATA,
                ColumnStrategy.SKIP,
                {"reason": f"Metadata column: {column}"},
            )

        # Check for redundant columns where all values equal the column name
        if self._is_redundant_column(column, values):
            return (
                ColumnType.INSUFFICIENT_DATA,
                ColumnStrategy.SKIP,
                {
                    "reason": f'Redundant column: all values equal column name "{column}"'
                },
            )

        # Check for HPA columns that should be meaningful node classes
        if self._is_hpa_node_class_column(column, values):
            return (
                ColumnType.CATEGORICAL_LARGE,
                ColumnStrategy.MEANINGFUL_NODE_CLASS,
                {},
            )

        # Check if column contains multiple values per cell
        has_multiple_values, expanded_count = self._check_multiple_values(values)
        extra_info = {
            "has_multiple_values": has_multiple_values,
            "expanded_unique_count": expanded_count if has_multiple_values else None,
        }

        # Check if text-rich
        if self._is_text_rich_column(column, values):
            avg_length = sum(len(v) for v in values) / len(values)
            extra_info["avg_text_length"] = int(avg_length)
            return ColumnType.TEXT_RICH, ColumnStrategy.KG_EXTRACTION, extra_info

        # Everything else becomes a property on the protein node
        # Determine the appropriate column type based on data characteristics
        unique_count = len(set(values))
        total_count = len(values)
        unique_percentage = (unique_count / total_count * 100) if total_count > 0 else 0

        # Numeric detection
        if self._is_numeric_column(values):
            if self._is_binary_numeric(values):
                return ColumnType.BINARY, ColumnStrategy.PROPERTY, extra_info
            if unique_count < 20:
                return ColumnType.NUMERIC_DISCRETE, ColumnStrategy.PROPERTY, extra_info
            return ColumnType.NUMERIC_CONTINUOUS, ColumnStrategy.PROPERTY, extra_info

        # Binary detection
        if unique_count <= 2:
            return ColumnType.BINARY, ColumnStrategy.PROPERTY, extra_info

        # Multi-value categorical
        if has_multiple_values:
            if expanded_count <= 10:
                return (
                    ColumnType.MULTI_VALUE_CATEGORICAL_SMALL,
                    ColumnStrategy.PROPERTY,
                    extra_info,
                )
            return (
                ColumnType.MULTI_VALUE_CATEGORICAL_LARGE,
                ColumnStrategy.PROPERTY,
                extra_info,
            )

        # Single-value categorical - determine if meaningful node class or property
        # LOW cardinality (few unique values) → Properties (simple attributes)
        # MEDIUM-HIGH cardinality (many unique values) → Node Classes (meaningful categories)

        if unique_count <= 10:  # Low cardinality - always properties
            if unique_count <= 2:
                return ColumnType.BINARY, ColumnStrategy.PROPERTY, extra_info
            return ColumnType.CATEGORICAL_SMALL, ColumnStrategy.PROPERTY, extra_info
        if unique_percentage >= 80.0:  # Too unique - becomes property (like IDs)
            return ColumnType.CATEGORICAL_LARGE, ColumnStrategy.PROPERTY, extra_info
        if self.meaningful_class_min <= unique_count <= self.meaningful_class_max:
            # Medium cardinality - meaningful node classes for exploration
            return (
                ColumnType.CATEGORICAL_LARGE,
                ColumnStrategy.MEANINGFUL_NODE_CLASS,
                extra_info,
            )
        # Outside meaningful range (too many unique values) - becomes property
        return ColumnType.CATEGORICAL_LARGE, ColumnStrategy.PROPERTY, extra_info

    def _is_uniprot_column(self, column: str, values: list[str]) -> bool:
        """Check if column contains UniProt IDs."""
        column_lower = column.lower()
        if "uniprot" in column_lower and not any(
            x in column_lower for x in ["secondary", "description", "name"]
        ):
            return True

        # Check value patterns (UniProt IDs are typically 6-10 alphanumeric)
        if values:
            sample_values = values[:10]
            uniprot_pattern = re.compile(r"^[A-Z0-9]{6,10}$")
            matches = sum(1 for v in sample_values if uniprot_pattern.match(v.strip()))
            return matches / len(sample_values) > 0.8

        return False

    def _is_ensembl_column(self, column: str, values: list[str]) -> bool:
        """Check if column contains Ensembl IDs."""
        column_lower = column.lower()
        if "ensembl" in column_lower:
            return True

        # Check for ENSG pattern
        if values:
            sample_values = values[:10]
            ensembl_pattern = re.compile(r"^ENSG\d{11}$")
            matches = sum(1 for v in sample_values if ensembl_pattern.match(v.strip()))
            return matches / len(sample_values) > 0.8

        return False

    def _is_metadata_column(self, column: str, values: list[str]) -> bool:
        """Check if column contains metadata that should be skipped."""
        column_lower = column.lower()

        # Timestamp and metadata column patterns
        metadata_patterns = [
            "created_at",
            "updated_at",
            "modified_at",
            "timestamp",
            "date",
            "time",
            "created",
            "updated",
            "modified",
            "inserted",
            "processed",
            "imported",
            "source_file",
            "source_url",
            "source",
            "origin",
            "provenance",
            "version",
            "revision",
            "id",
            "index",
            "row_id",
            "uuid",
            "hash",
        ]

        # Check column name
        if any(pattern in column_lower for pattern in metadata_patterns):
            return True

        # Check if values look like timestamps or IDs
        if values:
            sample_values = values[:10]

            # Check for timestamp patterns (ISO dates, Unix timestamps, etc.)
            timestamp_patterns = [
                re.compile(r"^\d{4}-\d{2}-\d{2}"),  # YYYY-MM-DD
                re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),  # ISO datetime
                re.compile(r"^\d{10,13}$"),  # Unix timestamp (10-13 digits)
                re.compile(r"^\d{2}/\d{2}/\d{4}"),  # MM/DD/YYYY
                re.compile(r"^\d{2}-\d{2}-\d{4}"),  # MM-DD-YYYY
            ]

            timestamp_matches = 0
            for val in sample_values:
                val_str = str(val).strip()
                if any(pattern.match(val_str) for pattern in timestamp_patterns):
                    timestamp_matches += 1

            # If >50% of values look like timestamps, skip the column
            if timestamp_matches > len(sample_values) * 0.5:
                return True

        return False

    def _is_redundant_column(self, column: str, values: list[str]) -> bool:
        """Check if column contains only the column name as values (redundant)."""
        if not values:
            return False

        # Check if all non-empty values are exactly the column name
        column_name = column.lower().strip()
        for val in values:
            val_str = str(val).strip()
            if val_str and val_str.upper() != "NA":  # Skip NA values
                if val_str.lower() != column_name:
                    return False

        return True

    def _is_hpa_node_class_column(self, column: str, values: list[str]) -> bool:
        """Check if column contains HPA data that should be meaningful node classes.

        These columns create classification nodes that are LINKED to protein nodes
        via BELONGS_TO relationships, adding structured relationships to the knowledge graph.
        They do NOT create unlinked standalone nodes.
        """
        column_lower = column.lower()

        # HPA columns that represent biological classifications and should be node classes
        # Each creates nodes linked to proteins via uniprot_id/gene_name relationships
        hpa_node_class_columns = [
            "hpa_secretome_location",  # Links proteins to secretion locations
            "hpa_which_blood_cell_lineage",  # Links proteins to blood cell types
            "hpa_which_cell_type",  # Links proteins to cell types
            "hpa_which_tissue",  # Links proteins to tissue expression
            "uniprot_molecular_function",  # Links proteins to molecular functions
            "uniprot_biological_process",  # Links proteins to biological processes
            "uniprot_cellular_component",  # Links proteins to cellular components
        ]

        return column_lower in hpa_node_class_columns

    def _is_text_rich_column(self, column: str, values: list[str]) -> bool:
        """Determine if a column contains rich text suitable for KG extraction."""

        # Check column name
        column_lower = column.lower()

        # Exclude specific columns that should NOT undergo KG extraction or embedding
        # These columns should be treated as simple properties instead
        excluded_text_columns = [
            "uniprot_protein_description",  # Simple protein descriptions - no embeddings needed
            "protein_description",  # Generic protein descriptions
            "gene_description",  # Gene descriptions
            "protein_name",  # Just names, not rich text
            "gene_name",  # Just names, not rich text
        ]

        if column_lower in excluded_text_columns:
            return False

        has_text_keyword = any(
            keyword in column_lower for keyword in self.text_rich_keywords
        )

        if not values:
            return False

        # Calculate average text length
        avg_length = sum(len(val) for val in values) / len(values)

        # Sample some values to check for biological content
        sample_size = min(10, len(values))
        sample_values = values[:sample_size]
        biological_content_count = 0

        for val in sample_values:
            val_lower = val.lower()
            if any(keyword in val_lower for keyword in self.biological_keywords):
                biological_content_count += 1

        biological_content_ratio = biological_content_count / sample_size

        # Criteria for text-rich column:
        # 1. Has text-related column name OR
        # 2. Average length > 80 characters AND contains biological keywords in >30% of samples
        return has_text_keyword or (avg_length > 80 and biological_content_ratio > 0.3)

    def _check_multiple_values(self, values: list[str]) -> tuple[bool, int]:
        """Check if column contains multiple values separated by delimiters."""
        separators = [";", "|", ",", "/"]

        multi_value_count = 0
        expanded_values: set[str] = set()

        for value in values[:20]:  # Sample first 20 values
            for sep in separators:
                if sep in value and len(value.split(sep)) > 1:
                    multi_value_count += 1
                    # Add expanded values to set
                    expanded_values.update(
                        v.strip() for v in value.split(sep) if v.strip()
                    )
                    break

        has_multiple = (
            multi_value_count > len(values) * 0.1
        )  # >10% have multiple values
        return has_multiple, len(expanded_values)

    def _is_numeric_column(self, values: list[str]) -> bool:
        """Check if column contains numeric values."""
        numeric_count = 0

        for value in values[:20]:  # Sample first 20 values
            try:
                float(value.strip())
                numeric_count += 1
            except ValueError:
                continue

        return numeric_count > len(values[:20]) * 0.8  # >80% are numeric

    def _is_binary_numeric(self, values: list[str]) -> bool:
        """Check if numeric column is binary (0/1 or similar)."""
        unique_values = set()
        for value in values:
            try:
                num = float(value.strip())
                unique_values.add(num)
                if len(unique_values) > 2:
                    return False
            except ValueError:
                return False

        return len(unique_values) <= 2

    def _analysis_to_dict(self, analysis: ColumnAnalysis) -> dict[str, Any]:
        """Convert ColumnAnalysis to dictionary for serialization."""
        return {
            "type": analysis.column_type.value,
            "unique_count": analysis.unique_count,
            "total_values": analysis.total_values,
            "empty_percentage": analysis.empty_percentage,
            "na_percentage": analysis.na_percentage,
            "total_missing_percentage": analysis.total_missing_percentage,
            "strategy": analysis.strategy.value,
            "has_multiple_values": analysis.has_multiple_values,
            "expanded_unique_count": analysis.expanded_unique_count,
            "avg_text_length": analysis.avg_text_length,
            "sample_values": analysis.sample_values,
            "partner_column": analysis.partner_column,
            "reason": analysis.reason,
        }
