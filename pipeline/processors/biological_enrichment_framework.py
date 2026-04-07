#!/usr/bin/env python3
"""
General Biological Enrichment Framework

This module provides a flexible framework for enriching the knowledge graph with
various types of biological data. Supports different enrichment types like:
- Subcellular locations
- Protein domains
- Gene Ontology terms
- Generic CSV enrichment (with Ensembl ID exclusion)
"""

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.database import Neo4jDatabase
from pipeline.ingest.ingestor import Chunk, ProcessedDocument
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class EnrichmentProcessor(ABC):
    """
    Abstract base class for biological data enrichment processors.
    """

    def __init__(self, database: str = "cvd1", db: Neo4jDatabase | None = None):
        self.db = db or Neo4jDatabase(database=database)

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using Neo4j."""
        return self.db._execute_cypher(query, parameters)

    @abstractmethod
    def get_enrichment_type(self) -> str:
        """Return the type of enrichment this processor handles."""

    @abstractmethod
    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """Parse the input data file and return structured data."""

    @abstractmethod
    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create the enrichment-specific nodes (e.g., SubcellularLocation, Domain, etc.)."""

    @abstractmethod
    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create relationships between proteins and enrichment nodes."""

    def enrich_from_file(self, file_path: str) -> dict[str, Any]:
        """
        Main enrichment method that orchestrates the entire process.

        Args:
            file_path: Path to the data file to process

        Returns:
            Dictionary with enrichment statistics
        """
        logger.info(
            f"Starting {self.get_enrichment_type()} enrichment from {file_path}"
        )

        stats = {
            "enrichment_type": self.get_enrichment_type(),
            "file_path": file_path,
            "parsing_stats": {},
            "node_creation_stats": {},
            "relationship_stats": {},
            "errors": [],
        }

        try:
            # Step 1: Parse the data file
            logger.info("Parsing data file...")
            parsed_data = self.parse_data_file(file_path)
            stats["parsing_stats"] = parsed_data.get("stats", {})

            # Step 2: Create enrichment nodes
            logger.info("Creating enrichment nodes...")
            node_stats = self.create_enrichment_nodes(parsed_data)
            stats["node_creation_stats"] = node_stats

            # Step 3: Create relationships
            logger.info("Creating relationships...")
            relationship_stats = self.create_relationships(parsed_data)
            stats["relationship_stats"] = relationship_stats

            logger.info(
                f"{self.get_enrichment_type()} enrichment completed successfully"
            )
            return stats

        except Exception as e:
            error_msg = f"{self.get_enrichment_type()} enrichment failed: {e}"
            logger.error(error_msg, exc_info=True)
            stats["errors"].append(error_msg)
            return stats

    def get_current_stats(self) -> dict[str, Any]:
        """Get current statistics about this enrichment type in the graph."""
        # Default implementation - subclasses can override for specific stats
        return {}

    def close(self):
        """Close database connections."""
        if self.db:
            self.db.close()
        logger.info(f"{self.get_enrichment_type()} enricher closed")


class SubcellularLocationProcessor(EnrichmentProcessor):
    """Processor for subcellular location enrichment."""

    def get_enrichment_type(self) -> str:
        return "subcellular_location"

    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """Parse subcellular location CSV file."""
        protein_locations = {}  # uniprot_id -> list of locations
        all_locations = set()
        stats = {"proteins_processed": 0, "locations_found": 0, "unique_locations": 0}

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprot_id = row.get("uniprot", "").strip()
                location_string = row.get("uniprot_subcellular_location", "").strip()

                if uniprot_id:
                    locations = self._parse_locations(location_string)
                    if locations:
                        protein_locations[uniprot_id] = locations
                        all_locations.update(locations)

        stats["proteins_processed"] = len(protein_locations)
        stats["locations_found"] = sum(len(locs) for locs in protein_locations.values())
        stats["unique_locations"] = len(all_locations)

        return {
            "protein_locations": protein_locations,
            "all_locations": all_locations,
            "stats": stats,
        }

    def _parse_locations(self, location_string: str) -> list[str]:
        """Parse location string handling semicolons."""
        if not location_string:
            return []
        locations = [
            loc.strip().strip('"') for loc in location_string.split(";") if loc.strip()
        ]
        return list(set(locations))  # Remove duplicates

    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create SubcellularLocation nodes."""
        locations = data["all_locations"]
        stats = {"created": 0, "updated": 0}

        # Create nodes in batches
        batch_size = 100
        location_list = list(locations)

        for i in range(0, len(location_list), batch_size):
            batch = location_list[i : i + batch_size]

            # Check existing
            existing_query = """
            UNWIND $locations AS location_name
            OPTIONAL MATCH (loc:SubcellularLocation {name: location_name})
            RETURN location_name, loc IS NOT NULL AS exists
            """
            existing_results = self._execute_query(existing_query, {"locations": batch})
            existing_locations = {
                r["location_name"] for r in existing_results if r.get("exists")
            }

            # Create new ones
            new_locations = [loc for loc in batch if loc not in existing_locations]
            if new_locations:
                create_query = """
                UNWIND $locations AS location_name
                CREATE (loc:SubcellularLocation {
                    name: location_name,
                    id: location_name,
                    category: CASE
                        WHEN location_name CONTAINS 'membrane' THEN 'membrane'
                        WHEN location_name IN ['Nucleus', 'Cytoplasm', 'Mitochondrion', 'Endoplasmic reticulum', 'Golgi apparatus'] THEN 'organelle'
                        WHEN location_name = 'Secreted' THEN 'extracellular'
                        ELSE 'other'
                    END
                })
                """
                self._execute_query(create_query, {"locations": new_locations})
                stats["created"] += len(new_locations)

            stats["updated"] += len(existing_locations)

        # Create hierarchy
        hierarchy_stats = self._create_hierarchy(locations)
        stats["hierarchy_created"] = hierarchy_stats

        return stats

    def _create_hierarchy(self, locations: set[str]) -> int:
        """Create hierarchical relationships."""
        hierarchy = {
            "Mitochondrion inner membrane": "Mitochondrion",
            "Mitochondrion outer membrane": "Mitochondrion",
            "Endoplasmic reticulum membrane": "Endoplasmic reticulum",
            "Golgi apparatus membrane": "Golgi apparatus",
        }

        relationships_created = 0
        for child, parent in hierarchy.items():
            if child in locations and parent in locations:
                query = """
                MATCH (child:SubcellularLocation {name: $child_name})
                MATCH (parent:SubcellularLocation {name: $parent_name})
                MERGE (child)-[r:PART_OF]->(parent)
                SET r.relationship_type = 'hierarchical'
                """
                self._execute_query(query, {"child_name": child, "parent_name": parent})
                relationships_created += 1

        return relationships_created

    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create LOCATED_IN relationships."""
        protein_locations = data["protein_locations"]
        stats = {
            "relationships_created": 0,
            "proteins_matched": 0,
            "proteins_not_found": 0,
        }

        # Process in batches
        batch_size = 50
        items = list(protein_locations.items())

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            uniprot_ids = [pid for pid, _ in batch]

            # Check which proteins exist
            protein_check_query = """
            UNWIND $uniprot_ids AS uniprot_id
            OPTIONAL MATCH (p) WHERE p.uniprotID = uniprot_id
            RETURN uniprot_id, p IS NOT NULL AS exists
            """
            protein_results = self._execute_query(
                protein_check_query, {"uniprot_ids": uniprot_ids}
            )
            existing_proteins = {
                r["uniprot_id"] for r in protein_results if r.get("exists")
            }

            stats["proteins_matched"] += len(existing_proteins)
            stats["proteins_not_found"] += len(uniprot_ids) - len(existing_proteins)

            # Create relationships
            for protein_id, locations in batch:
                if protein_id not in existing_proteins:
                    continue

                for location in locations:
                    relationship_query = """
                    MATCH (p) WHERE p.uniprotID = $uniprot_id
                    MATCH (loc:SubcellularLocation {name: $location_name})
                    MERGE (p)-[r:LOCATED_IN]->(loc)
                    SET r.source = 'UniProt'
                    """
                    self._execute_query(
                        relationship_query,
                        {"uniprot_id": protein_id, "location_name": location},
                    )
                    stats["relationships_created"] += 1

        return stats


class GenericCSVEnrichmentProcessor:
    """
    Generic processor that can enrich the graph from any CSV file.

    Logic:
    - Match UniProt IDs to protein nodes
    - Ensembl ID columns are ALWAYS treated as protein properties (never create node classes)
    - For each column besides UniProt ID and Ensembl ID columns:
      - Continuous numeric values → add as protein property
      - Binary values → add as protein property
      - < unique_threshold_percentage of total rows unique values → add as protein property
      - >= unique_threshold_percentage of total rows unique values → create new node class with relationships
    """

    def __init__(
        self,
        database: str = "cvd1",
        service: str = "local",
        db: Neo4jDatabase | None = None,
        empty_threshold: float = 40.0,
        unique_threshold_percentage: float = 5.0,
    ):
        self.service = service
        self.db = db or Neo4jDatabase(database=database)

        self.uniprot_column = None  # Will be detected automatically
        self.empty_threshold = (
            empty_threshold  # Maximum percentage of empty values allowed
        )
        self.unique_threshold_percentage = unique_threshold_percentage  # Minimum percentage of unique values for node class creation

        self.kg_pipeline = None

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using Neo4j."""
        return self.db._execute_cypher(query, parameters or {})

    async def enrich_from_file(self, file_path: str) -> dict[str, Any]:
        """Enrich from a CSV file or folder of CSV files using generic logic."""
        path = Path(file_path)

        if path.is_dir():
            return await self._enrich_from_folder(path)
        return await self._enrich_from_single_file(file_path)

    async def _enrich_from_single_file(self, file_path: str) -> dict[str, Any]:
        """Enrich from a single CSV file using generic logic."""
        logger.info(f"Starting generic CSV enrichment from {file_path}")

        # Parse and analyze the CSV
        data_analysis = self._analyze_csv(file_path)

        # Create enrichment based on column types
        enrichment_stats = await self._perform_enrichment(data_analysis)

        # Remove verbose data from results for clean terminal output
        clean_data_analysis = {
            "headers": data_analysis["headers"],
            "uniprot_column": data_analysis["uniprot_column"],
            "total_rows": data_analysis["total_rows"],
            "column_analysis": data_analysis["column_analysis"],
            # Exclude 'data' array to prevent verbose terminal output
        }

        stats = {
            "enrichment_type": "generic_csv",
            "file_path": file_path,
            "data_analysis": clean_data_analysis,
            "enrichment_stats": enrichment_stats,
        }

        logger.info("Generic CSV enrichment completed successfully")
        return stats

    async def _enrich_from_folder(self, folder_path: Path) -> dict[str, Any]:
        """Enrich from all CSV files in a folder."""
        logger.info(f"Starting generic CSV enrichment from folder {folder_path}")

        # Find all CSV files
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in folder {folder_path}")

        logger.info(f"Found {len(csv_files)} CSV files to process")

        # Process each file
        folder_stats = {
            "enrichment_type": "generic_csv_folder",
            "folder_path": str(folder_path),
            "total_files": len(csv_files),
            "processed_files": 0,
            "skipped_files": 0,
            "file_results": [],
            "aggregate_stats": {
                "total_proteins_processed": 0,
                "total_properties_added": 0,
                "total_node_classes_created": 0,
                "total_individual_nodes_created": 0,
                "total_relationships_created": 0,
                "total_errors": 0,
            },
        }

        for csv_file in csv_files:
            try:
                logger.info(f"Processing file: {csv_file.name}")
                file_result = await self._enrich_from_single_file(str(csv_file))

                # Clean up file result to remove verbose data
                clean_file_result = {
                    "enrichment_type": file_result.get("enrichment_type"),
                    "file_path": file_result.get("file_path"),
                    "data_analysis": {
                        "total_rows": file_result.get("data_analysis", {}).get(
                            "total_rows", 0
                        ),
                        "uniprot_column": file_result.get("data_analysis", {}).get(
                            "uniprot_column", "N/A"
                        ),
                    },
                    "enrichment_stats": file_result.get("enrichment_stats", {}),
                }

                folder_stats["file_results"].append(clean_file_result)
                folder_stats["processed_files"] += 1

                # Aggregate stats
                enrichment_stats = file_result.get("enrichment_stats", {})
                folder_stats["aggregate_stats"]["total_proteins_processed"] += (
                    enrichment_stats.get("proteins_processed", 0)
                )
                folder_stats["aggregate_stats"]["total_properties_added"] += (
                    enrichment_stats.get("properties_added", 0)
                )
                folder_stats["aggregate_stats"]["total_node_classes_created"] += (
                    enrichment_stats.get("node_classes_created", 0)
                )
                folder_stats["aggregate_stats"]["total_individual_nodes_created"] += (
                    enrichment_stats.get("individual_nodes_created", 0)
                )
                folder_stats["aggregate_stats"]["total_relationships_created"] += (
                    enrichment_stats.get("relationships_created", 0)
                )
                folder_stats["aggregate_stats"]["total_errors"] += len(
                    enrichment_stats.get("errors", [])
                )

            except Exception as e:
                logger.error(f"Failed to process {csv_file.name}: {e}")
                folder_stats["skipped_files"] += 1
                folder_stats["file_results"].append(
                    {"file_path": str(csv_file), "error": str(e)}
                )

        logger.info(
            f"Folder enrichment completed: {folder_stats['processed_files']} processed, {folder_stats['skipped_files']} skipped"
        )
        return folder_stats

    def _analyze_csv(self, file_path: str) -> dict[str, Any]:
        """Analyze CSV structure and determine enrichment strategy."""
        logger.info("Analyzing CSV structure...")

        data = []
        headers = None

        with open(file_path, encoding="utf-8") as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()

            # Try to detect delimiter, with fallback to common delimiters
            delimiter = None
            try:
                delimiter = sniffer.sniff(sample).delimiter
                logger.info(f"Detected delimiter: '{delimiter}'")
            except csv.Error as e:
                logger.warning(
                    f"Could not auto-detect delimiter: {e}. Trying common delimiters..."
                )
                # Try common delimiters in order of preference
                for delim in [",", "\t", ";", "|"]:
                    try:
                        f.seek(0)
                        test_reader = csv.DictReader(f, delimiter=delim)
                        test_headers = test_reader.fieldnames
                        if test_headers and len(test_headers) > 1:
                            # Check if we can read at least one row
                            test_data = list(test_reader)
                            if test_data:
                                delimiter = delim
                                logger.info(f"Using fallback delimiter: '{delimiter}'")
                                break
                    except Exception:
                        continue

                if delimiter is None:
                    raise ValueError(
                        "Could not determine CSV delimiter. File may not be a valid CSV or may have unusual formatting."
                    )

            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames
            data = list(reader)

        if not headers or not data:
            raise ValueError("CSV file must have headers and at least one row of data")

        # Find UniProt column
        uniprot_column = self._find_uniprot_column(list(headers))
        if not uniprot_column:
            raise ValueError(
                "Could not find UniProt ID column in CSV. Expected column names containing 'uniprot', 'accession', or similar"
            )

        # Find Ensembl columns (these will be treated as properties, not node classes)
        ensembl_columns = self._find_ensembl_columns(list(headers))

        logger.info(f"Found UniProt column: {uniprot_column}")
        if ensembl_columns:
            logger.info(
                f"Found Ensembl columns (will be treated as properties): {ensembl_columns}"
            )

        # Detect 1:1 column mappings (e.g., ReactomePathwayName + ReactomePathwayId)
        column_mappings = self._detect_one_to_one_mappings(
            data, list(headers), uniprot_column, ensembl_columns
        )

        # Analyze each column
        column_analysis = {}
        for col in headers:
            if col == uniprot_column:
                continue  # Skip UniProt column
            if col in ensembl_columns:
                # Ensembl columns are always treated as properties, never as node classes
                column_analysis[col] = self._analyze_ensembl_column(data, col)
                continue

            # Check if this column is part of a 1:1 mapping
            mapping_info = column_mappings.get(col)
            if mapping_info:
                column_analysis[col] = self._analyze_mapped_column(
                    data, col, mapping_info
                )
            else:
                column_analysis[col] = self._analyze_column(data, col)

        return {
            "headers": headers,
            "uniprot_column": uniprot_column,
            "total_rows": len(data),
            "column_analysis": column_analysis,
            "data": data,  # Keep data for processing, but don't include in final stats
        }

    def _find_uniprot_column(self, headers: list[str]) -> str | None:
        """Find the column containing UniProt IDs."""
        uniprot_keywords = [
            "uniprot",
            "accession",
            "protein_id",
            "proteinid",
            "uniprot_id",
            "uniprotid",
        ]

        for header in headers:
            header_lower = header.lower().replace(" ", "").replace("_", "")
            if any(keyword in header_lower for keyword in uniprot_keywords):
                return header

        return None

    def _find_ensembl_columns(self, headers: list[str]) -> list[str]:
        """Find columns containing Ensembl IDs."""
        ensembl_keywords = [
            "ensembl",
            "ensembl_id",
            "ensemblid",
            "ensg",
            "enst",
            "ensembl_gene",
            "ensembl_transcript",
        ]
        ensembl_columns = []

        for header in headers:
            header_lower = header.lower().replace(" ", "").replace("_", "")
            if any(keyword in header_lower for keyword in ensembl_keywords):
                ensembl_columns.append(header)

        return ensembl_columns

    def _detect_one_to_one_mappings(
        self,
        data: list[dict[str, Any]],
        headers: list[str],
        uniprot_column: str,
        ensembl_columns: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Detect 1:1 mappings between columns (e.g., PathwayName + PathwayId)."""
        mappings = {}

        # Skip system columns
        skip_columns = {uniprot_column} | set(ensembl_columns)
        candidate_columns = [col for col in headers if col not in skip_columns]

        # Look for name/id pairs
        name_keywords = ["name", "title", "description", "label"]
        id_keywords = ["id", "identifier", "code", "key"]

        for i, col1 in enumerate(candidate_columns):
            for _j, col2 in enumerate(candidate_columns[i + 1 :], i + 1):
                # Check if one looks like a name and the other like an ID
                col1_lower = col1.lower().replace(" ", "").replace("_", "")
                col2_lower = col2.lower().replace(" ", "").replace("_", "")

                is_name_id_pair = False
                name_col = None
                id_col = None

                # Check if col1 is name and col2 is id
                if any(name_kw in col1_lower for name_kw in name_keywords) and any(
                    id_kw in col2_lower for id_kw in id_keywords
                ):
                    name_col, id_col = col1, col2
                    is_name_id_pair = True
                # Check if col2 is name and col1 is id
                elif any(name_kw in col2_lower for name_kw in name_keywords) and any(
                    id_kw in col1_lower for id_kw in id_keywords
                ):
                    name_col, id_col = col2, col1
                    is_name_id_pair = True

                if is_name_id_pair and name_col and id_col:
                    logger.info(
                        f"Found potential name-ID pair: {name_col} (name) <-> {id_col} (id)"
                    )

                    # For obvious biological name-ID pairs, be flexible about perfect correspondence
                    # Focus on semantic correctness over data perfection
                    name_to_id = {}
                    valid_pairs_count = 0
                    total_rows_with_data = 0

                    for row in data:
                        name_val = row.get(name_col, "").strip()
                        id_val = row.get(id_col, "").strip()

                        # Skip NA values
                        if (
                            not name_val
                            or name_val.lower() == "na"
                            or not id_val
                            or id_val.lower() == "na"
                        ):
                            continue

                        total_rows_with_data += 1

                        # Handle semicolon-separated values flexibly
                        if ";" in name_val and ";" in id_val:
                            names = [
                                n.strip() for n in name_val.split(";") if n.strip()
                            ]
                            ids = [i.strip() for i in id_val.split(";") if i.strip()]

                            # Use available pairs even if lengths don't match perfectly
                            min_len = min(len(names), len(ids))
                            for i in range(min_len):
                                name = names[i]
                                id_item = ids[i]

                                # Store mapping, preferring first occurrence for conflicts
                                if name not in name_to_id:
                                    name_to_id[name] = id_item
                                    valid_pairs_count += 1
                        # Single values - store if not already mapped
                        elif name_val not in name_to_id:
                            name_to_id[name_val] = id_val
                            valid_pairs_count += 1

                    # Accept as 1:1 mapping if we have reasonable coverage and multiple unique pairs
                    coverage_ratio = valid_pairs_count / max(total_rows_with_data, 1)

                    if (
                        len(name_to_id) >= 2 and coverage_ratio >= 0.1
                    ):  # At least 2 unique pairs and 10% coverage
                        logger.info(
                            f"✅ Accepted semantic 1:1 mapping: {name_col} <-> {id_col} ({len(name_to_id)} unique pairs, {coverage_ratio:.1%} coverage)"
                        )
                        mappings[name_col] = {
                            "type": "1:1_mapping_name",
                            "partner_column": id_col,
                            "role": "name",
                        }
                        mappings[id_col] = {
                            "type": "1:1_mapping_id",
                            "partner_column": name_col,
                            "role": "id",
                        }
                    else:
                        logger.info(
                            f"❌ Insufficient data for semantic 1:1: {name_col} <-> {id_col} ({len(name_to_id)} unique pairs, {coverage_ratio:.1%} coverage)"
                        )

        return mappings

    def _analyze_ensembl_column(
        self, data: list[dict[str, Any]], column: str
    ) -> dict[str, Any]:
        """Analyze an Ensembl ID column - always treat as property, never create node classes."""
        values = []
        empty_count = 0

        for row in data:
            val = row.get(column, "").strip()
            if not val:  # Empty value
                empty_count += 1
            else:  # Valid value
                values.append(val)

        total_rows = len(data)
        empty_percentage = (empty_count / total_rows) * 100
        unique_values = set(values)
        unique_count = len(unique_values)

        return {
            "type": "ensembl_id",
            "unique_count": unique_count,
            "total_values": len(values),
            "empty_percentage": empty_percentage,
            "na_percentage": 0.0,  # Ensembl IDs don't use NA/null indicators
            "total_missing_percentage": empty_percentage,
            "strategy": "property",  # Always treat as property, never create node classes
            "sample_values": list(unique_values)[:5],
        }

    def _analyze_mapped_column(
        self, data: list[dict[str, Any]], column: str, mapping_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze a column that is part of a 1:1 mapping."""
        values = []
        empty_count = 0
        na_count = 0

        for row in data:
            val = row.get(column, "").strip()
            if not val:  # Empty value
                empty_count += 1
            elif val.lower() in [
                "na",
                "n/a",
                "null",
                "none",
                "nan",
                "unknown",
                "missing",
            ]:  # NA/null value
                na_count += 1
            else:  # Valid value
                values.append(val)

        total_rows = len(data)
        empty_percentage = (empty_count / total_rows) * 100
        na_percentage = (na_count / total_rows) * 100
        total_missing_percentage = empty_percentage + na_percentage

        unique_values = set(values)
        unique_count = len(unique_values)

        if mapping_info["role"] == "name":
            # Name column becomes the node class (if sufficient data)
            if total_missing_percentage >= self.empty_threshold:
                return {
                    "type": "1:1_mapping_name_insufficient_data",
                    "unique_count": unique_count,
                    "total_values": len(values),
                    "empty_percentage": empty_percentage,
                    "na_percentage": na_percentage,
                    "total_missing_percentage": total_missing_percentage,
                    "strategy": "skip",
                    "partner_column": mapping_info["partner_column"],
                    "reason": f"Column has {total_missing_percentage:.1f}% missing values (>= {self.empty_threshold}% threshold)",
                }
            return {
                "type": "1:1_mapping_name",
                "unique_count": unique_count,
                "total_values": len(values),
                "empty_percentage": empty_percentage,
                "na_percentage": na_percentage,
                "total_missing_percentage": total_missing_percentage,
                "strategy": "node_class_with_id_property",
                "partner_column": mapping_info["partner_column"],
                "sample_values": list(unique_values)[:5],
            }
        # role == 'id'
        # ID column becomes a property of the name column's nodes
        return {
            "type": "1:1_mapping_id",
            "unique_count": unique_count,
            "total_values": len(values),
            "empty_percentage": empty_percentage,
            "na_percentage": na_percentage,
            "total_missing_percentage": total_missing_percentage,
            "strategy": "skip_handled_by_partner",  # Handled when processing the name column
            "partner_column": mapping_info["partner_column"],
            "sample_values": list(unique_values)[:5],
        }

    def _analyze_column(
        self, data: list[dict[str, Any]], column: str
    ) -> dict[str, Any]:
        """Analyze a column to determine its type and enrichment strategy."""
        values = []
        empty_count = 0
        na_count = 0

        for row in data:
            val = row.get(column, "").strip()
            if not val:  # Empty value
                empty_count += 1
            elif val.lower() in [
                "na",
                "n/a",
                "null",
                "none",
                "nan",
                "unknown",
                "missing",
            ]:  # NA/null value
                na_count += 1
            else:  # Valid value
                values.append(val)

        total_rows = len(data)
        empty_percentage = (empty_count / total_rows) * 100
        na_percentage = (na_count / total_rows) * 100
        total_missing_percentage = empty_percentage + na_percentage

        # Check data completeness requirement: >empty_threshold% cannot be empty/NA (so >=empty_threshold% missing → skip)
        if total_missing_percentage >= self.empty_threshold:
            return {
                "type": "insufficient_data",
                "unique_count": len(set(values)),
                "total_values": len(values),
                "empty_percentage": empty_percentage,
                "na_percentage": na_percentage,
                "total_missing_percentage": total_missing_percentage,
                "strategy": "skip",
                "reason": f"Column has {total_missing_percentage:.1f}% missing values (>= {self.empty_threshold}% threshold)",
            }

        unique_values = set(values)
        unique_count = len(unique_values)

        # Check if this is a text-rich column suitable for KG extraction
        if self._is_text_rich_column(column, values):
            return {
                "type": "text_rich",
                "unique_count": unique_count,
                "total_values": len(values),
                "empty_percentage": empty_percentage,
                "na_percentage": na_percentage,
                "total_missing_percentage": total_missing_percentage,
                "strategy": "kg_extraction" if self.kg_pipeline else "property",
                "avg_text_length": sum(len(val) for val in values) / len(values),
                "sample_values": list(unique_values)[:3],
            }

        # Check for semicolon-separated values
        has_multiple_values = any(";" in val for val in values)
        if has_multiple_values:
            # Split semicolon-separated values and analyze the expanded set
            expanded_values = []
            for val in values:
                expanded_values.extend([v.strip() for v in val.split(";") if v.strip()])
            unique_expanded = set(expanded_values)
            expanded_unique_count = len(unique_expanded)

            # Use the number of unique multi-value combinations for threshold decision
            unique_multi_value_combinations = len(set(values))

            # Apply threshold to multi-value columns too
            unique_threshold_count = max(
                1, int((self.unique_threshold_percentage / 100.0) * total_rows)
            )

            if unique_multi_value_combinations < unique_threshold_count:
                return {
                    "type": "multi_value_categorical_small",
                    "unique_count": unique_multi_value_combinations,
                    "total_values": len(values),
                    "expanded_unique_count": expanded_unique_count,
                    "empty_percentage": empty_percentage,
                    "na_percentage": na_percentage,
                    "total_missing_percentage": total_missing_percentage,
                    "has_multiple_values": True,
                    "strategy": "property",
                    "sample_values": list(set(values))[:5],
                }
            return {
                "type": "multi_value_categorical_large",
                "unique_count": unique_multi_value_combinations,
                "total_values": len(values),
                "expanded_unique_count": expanded_unique_count,
                "empty_percentage": empty_percentage,
                "na_percentage": na_percentage,
                "total_missing_percentage": total_missing_percentage,
                "has_multiple_values": True,
                "strategy": "node_class_multi",
                "sample_values": list(unique_expanded)[:10],
            }

        unique_values = set(values)
        unique_count = len(unique_values)

        # Check if numeric
        numeric_values = []
        for val in values:
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                break
        else:
            # All values are numeric
            if len(numeric_values) > 0:
                is_continuous = (
                    len(set(numeric_values)) > 10
                )  # Consider continuous if >10 unique numeric values
                return {
                    "type": "continuous_numeric"
                    if is_continuous
                    else "discrete_numeric",
                    "unique_count": unique_count,
                    "total_values": len(values),
                    "empty_percentage": empty_percentage,
                    "na_percentage": na_percentage,
                    "total_missing_percentage": total_missing_percentage,
                    "strategy": "property",
                    "sample_values": list(unique_values)[:5],
                }

        # Check if binary
        if unique_count == 2:
            return {
                "type": "binary",
                "unique_count": unique_count,
                "total_values": len(values),
                "empty_percentage": empty_percentage,
                "na_percentage": na_percentage,
                "total_missing_percentage": total_missing_percentage,
                "strategy": "property",
                "values": sorted(unique_values),
            }

        # Categorical
        # Calculate threshold based on percentage of total rows
        unique_threshold_count = max(
            1, int((self.unique_threshold_percentage / 100.0) * total_rows)
        )

        if unique_count < unique_threshold_count:
            return {
                "type": "categorical_small",
                "unique_count": unique_count,
                "total_values": len(values),
                "empty_percentage": empty_percentage,
                "na_percentage": na_percentage,
                "total_missing_percentage": total_missing_percentage,
                "strategy": "property",
                "values": sorted(unique_values),
            }
        return {
            "type": "categorical_large",
            "unique_count": unique_count,
            "total_values": len(values),
            "empty_percentage": empty_percentage,
            "na_percentage": na_percentage,
            "total_missing_percentage": total_missing_percentage,
            "strategy": "node_class",
            "sample_values": list(unique_values)[:10],
        }

    async def _perform_enrichment(
        self, data_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform the actual enrichment based on analysis."""
        logger.info("Performing enrichment...")

        stats = {
            "proteins_processed": 0,
            "properties_added": 0,
            "node_classes_created": 0,  # Number of unique node labels created
            "individual_nodes_created": 0,  # Number of individual nodes created
            "relationships_created": 0,
            "kg_extractions_performed": 0,  # Number of KG extractions performed
            "errors": [],
        }

        data = data_analysis["data"]
        uniprot_column = data_analysis["uniprot_column"]
        column_analysis = data_analysis["column_analysis"]

        # Log skipped columns once at the beginning
        for column, analysis in column_analysis.items():
            if analysis.get("strategy") == "skip":
                logger.info(
                    f"Skipping column {column}: {analysis.get('reason', 'insufficient data')}"
                )

        # Process in batches
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # Get UniProt IDs for this batch
            uniprot_ids = [
                row[uniprot_column]
                for row in batch
                if row.get(uniprot_column, "").strip()
            ]

            if not uniprot_ids:
                continue

            # Check which proteins exist
            protein_check_query = """
            UNWIND $uniprot_ids AS uniprot_id
            OPTIONAL MATCH (p) WHERE p.uniprotID = uniprot_id
            RETURN uniprot_id, p IS NOT NULL AS exists
            """
            try:
                protein_results = self._execute_query(
                    protein_check_query, {"uniprot_ids": uniprot_ids}
                )
                existing_proteins = {
                    r["uniprot_id"] for r in protein_results if r.get("exists")
                }

                stats["proteins_processed"] += len(existing_proteins)

                # Process each column
                for column, analysis in column_analysis.items():
                    strategy = analysis.get("strategy")
                    if strategy == "skip" or strategy == "skip_handled_by_partner":
                        continue  # Already logged above or handled by partner column
                    if strategy == "property":
                        self._add_properties(
                            batch, column, analysis, existing_proteins, stats
                        )
                    elif strategy == "node_class":
                        self._create_node_class(
                            batch, column, analysis, existing_proteins, stats
                        )
                    elif strategy == "node_class_multi":
                        self._create_node_class_multi(
                            batch, column, analysis, existing_proteins, stats
                        )
                    elif strategy == "node_class_with_id_property":
                        self._create_node_class_with_id_property(
                            batch,
                            column,
                            analysis,
                            existing_proteins,
                            stats,
                            data_analysis,
                        )
                    elif strategy == "kg_extraction":
                        await self._extract_kg_from_text(
                            batch,
                            column,
                            analysis,
                            existing_proteins,
                            stats,
                            data_analysis,
                        )

            except Exception as e:
                error_msg = f"Error processing batch {i // batch_size + 1}: {e!s}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        return stats

    def _add_properties(
        self,
        batch: list[dict[str, Any]],
        column: str,
        analysis: dict[str, Any],
        existing_proteins: set[str],
        stats: dict[str, Any],
    ):
        """Add column values as properties to protein nodes."""
        property_name = self._sanitize_property_name(column)

        for row in batch:
            uniprot_id = row.get("uniprot", "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue

            value = row.get(column, "").strip()
            if not value:
                continue

            # Convert value based on type
            if analysis["type"] in ["continuous_numeric", "discrete_numeric"]:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
            elif analysis["type"] == "binary":
                # Normalize binary values
                value = self._normalize_binary_value(value)

            # Add property to protein
            query = f"""
            MATCH (p) WHERE p.uniprotID = $uniprot_id
            SET p.{property_name} = $value
            """

            try:
                self._execute_query(query, {"uniprot_id": uniprot_id, "value": value})
                stats["properties_added"] += 1
            except Exception as e:
                logger.error(
                    f"Error adding property {property_name} to protein {uniprot_id}: {e}"
                )

    def _create_node_class(
        self,
        batch: list[dict[str, Any]],
        column: str,
        analysis: dict[str, Any],
        existing_proteins: set[str],
        stats: dict[str, Any],
    ):
        """Create a new node class and relationships for categorical data with many values."""
        node_label = self._create_node_label(column)
        relationship_type = self._create_relationship_type(column)

        # Only collect values that correspond to existing proteins
        values_for_existing_proteins = set()
        for row in batch:
            uniprot_id = row.get("uniprot", "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue  # Skip rows for proteins that don't exist in the graph

            value = row.get(column, "").strip()
            if value:
                values_for_existing_proteins.add(value)

        # Create nodes only for values associated with existing proteins
        if values_for_existing_proteins:
            create_nodes_query = f"""
            UNWIND $values AS value
            MERGE (n:{node_label} {{name: value, id: value}})
            SET n.source_column = $column_name
            """

            try:
                self._execute_query(
                    create_nodes_query,
                    {
                        "values": list(values_for_existing_proteins),
                        "column_name": column,
                    },
                )
                stats["individual_nodes_created"] += len(values_for_existing_proteins)
                # Only increment node_classes_created once per column (not per individual node)
                if len(values_for_existing_proteins) > 0:
                    stats["node_classes_created"] += 1
            except Exception as e:
                logger.error(f"Error creating {node_label} nodes: {e}")
                return

        # Create relationships
        for row in batch:
            uniprot_id = row.get("uniprot", "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue

            value = row.get(column, "").strip()
            if not value:
                continue

            # Create relationships
            relationship_query = f"""
            MATCH (p) WHERE p.uniprotID = $uniprot_id
            MATCH (n:{node_label} {{name: $value}})
            MERGE (p)-[r:{relationship_type}]->(n)
            SET r.source_column = $column_name
            """

            try:
                self._execute_query(
                    relationship_query,
                    {"uniprot_id": uniprot_id, "value": value, "column_name": column},
                )
                stats["relationships_created"] += 1
            except Exception as e:
                logger.error(
                    f"Error creating relationship for {uniprot_id} -> {value}: {e}"
                )

    def _create_node_class_multi(
        self,
        batch: list[dict[str, Any]],
        column: str,
        analysis: dict[str, Any],
        existing_proteins: set[str],
        stats: dict[str, Any],
    ):
        """Create a new node class and multiple relationships for semicolon-separated values."""
        node_label = self._create_node_label(column)
        relationship_type = self._create_relationship_type(column)

        # Only collect values from rows that correspond to existing proteins
        values_for_existing_proteins = set()
        for row in batch:
            uniprot_id = row.get("uniprot", "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue  # Skip rows for proteins that don't exist in the graph

            value = row.get(column, "").strip()
            if value:
                # Split by semicolon and add all individual values
                individual_values = [v.strip() for v in value.split(";") if v.strip()]
                values_for_existing_proteins.update(individual_values)

        # Create nodes only for values associated with existing proteins
        if values_for_existing_proteins:
            create_nodes_query = f"""
            UNWIND $values AS value
            MERGE (n:{node_label} {{name: value, id: value}})
            SET n.source_column = $column_name
            """

            try:
                self._execute_query(
                    create_nodes_query,
                    {
                        "values": list(values_for_existing_proteins),
                        "column_name": column,
                    },
                )
                stats["individual_nodes_created"] += len(values_for_existing_proteins)
                # Only increment node_classes_created once per column (not per individual node)
                if len(values_for_existing_proteins) > 0:
                    stats["node_classes_created"] += 1
            except Exception as e:
                logger.error(f"Error creating {node_label} nodes: {e}")
                return

        # Create relationships (multiple per protein if semicolon-separated)
        for row in batch:
            uniprot_id = row.get("uniprot", "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue

            value = row.get(column, "").strip()
            if not value:
                continue

            # Split semicolon-separated values and create relationship for each
            individual_values = [v.strip() for v in value.split(";") if v.strip()]

            for individual_value in individual_values:
                relationship_query = f"""
                MATCH (p) WHERE p.uniprotID = $uniprot_id
                MATCH (n:{node_label} {{name: $value}})
                MERGE (p)-[r:{relationship_type}]->(n)
                SET r.source_column = $column_name
                """

                try:
                    self._execute_query(
                        relationship_query,
                        {
                            "uniprot_id": uniprot_id,
                            "value": individual_value,
                            "column_name": column,
                        },
                    )
                    stats["relationships_created"] += 1
                except Exception as e:
                    logger.error(
                        f"Error creating relationship for {uniprot_id} -> {individual_value}: {e}"
                    )

    def _create_node_class_with_id_property(
        self,
        batch: list[dict[str, Any]],
        column: str,
        analysis: dict[str, Any],
        existing_proteins: set[str],
        stats: dict[str, Any],
        data_analysis: dict[str, Any],
    ):
        """Create a new node class with ID properties for 1:1 mapped columns."""
        node_label = self._create_node_label(column)
        relationship_type = self._create_relationship_type(column)
        partner_column = analysis["partner_column"]

        # Collect name-id pairs for existing proteins only, handling semicolon-separated values
        name_id_pairs = {}
        protein_name_relationships = []  # Track which protein relates to which names

        for row in batch:
            uniprot_id = row.get(data_analysis["uniprot_column"], "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue

            name_value = row.get(column, "").strip()
            id_value = row.get(partner_column, "").strip()

            # Skip NA values
            if (
                not name_value
                or name_value.lower() == "na"
                or not id_value
                or id_value.lower() == "na"
            ):
                continue

            # Handle semicolon-separated values flexibly
            if ";" in name_value and ";" in id_value:
                names = [n.strip() for n in name_value.split(";") if n.strip()]
                ids = [i.strip() for i in id_value.split(";") if i.strip()]

                # Use available pairs even if lengths don't match perfectly
                min_len = min(len(names), len(ids))
                for i in range(min_len):
                    name = names[i]
                    id_val = ids[i]
                    name_id_pairs[name] = id_val
                    protein_name_relationships.append((uniprot_id, name))

                # Handle remaining names without IDs (use empty ID)
                for i in range(min_len, len(names)):
                    name = names[i]
                    if name not in name_id_pairs:  # Don't override existing mappings
                        name_id_pairs[name] = ""  # Empty ID for names without IDs
                        protein_name_relationships.append((uniprot_id, name))
            else:
                # Single values
                name_id_pairs[name_value] = id_value
                protein_name_relationships.append((uniprot_id, name_value))

        # Create nodes with both name and ID properties
        if name_id_pairs:
            # Prepare data for batch creation, handling empty IDs gracefully
            node_data = []
            for name, id_val in name_id_pairs.items():
                node_data.append(
                    {
                        "name": name,
                        "id_value": id_val or None,  # Use None for empty IDs
                    }
                )

            # Create separate queries for nodes with and without IDs
            nodes_with_ids = [item for item in node_data if item["id_value"]]
            nodes_without_ids = [item for item in node_data if not item["id_value"]]

            try:
                # Create nodes with IDs
                if nodes_with_ids:
                    create_nodes_with_ids_query = f"""
                    UNWIND $node_data AS data
                    MERGE (n:{node_label} {{name: data.name}})
                    SET n.{self._sanitize_property_name(partner_column)} = data.id_value,
                        n.source_column = $column_name,
                        n.id_column = $partner_column
                    """

                    self._execute_query(
                        create_nodes_with_ids_query,
                        {
                            "node_data": nodes_with_ids,
                            "column_name": column,
                            "partner_column": partner_column,
                        },
                    )

                # Create nodes without IDs
                if nodes_without_ids:
                    create_nodes_without_ids_query = f"""
                    UNWIND $node_data AS data
                    MERGE (n:{node_label} {{name: data.name}})
                    SET n.source_column = $column_name,
                        n.id_column = $partner_column
                    """

                    self._execute_query(
                        create_nodes_without_ids_query,
                        {
                            "node_data": nodes_without_ids,
                            "column_name": column,
                            "partner_column": partner_column,
                        },
                    )

                stats["individual_nodes_created"] += len(name_id_pairs)
                stats["node_classes_created"] += 1
                logger.info(
                    f"Created {len(name_id_pairs)} {node_label} nodes ({len(nodes_with_ids)} with {partner_column} properties, {len(nodes_without_ids)} without)"
                )

            except Exception as e:
                logger.error(
                    f"Error creating {node_label} nodes with ID properties: {e}"
                )
                return

        # Create relationships for each protein-name pair
        for uniprot_id, name_value in protein_name_relationships:
            relationship_query = f"""
            MATCH (p) WHERE p.uniprotID = $uniprot_id
            MATCH (n:{node_label} {{name: $name_value}})
            MERGE (p)-[r:{relationship_type}]->(n)
            SET r.source_column = $column_name
            """

            try:
                self._execute_query(
                    relationship_query,
                    {
                        "uniprot_id": uniprot_id,
                        "name_value": name_value,
                        "column_name": column,
                    },
                )
                stats["relationships_created"] += 1
            except Exception as e:
                logger.error(
                    f"Error creating relationship for {uniprot_id} -> {name_value}: {e}"
                )

    async def _extract_kg_from_text(
        self,
        batch: list[dict[str, Any]],
        column: str,
        analysis: dict[str, Any],
        existing_proteins: set[str],
        stats: dict[str, Any],
        data_analysis: dict[str, Any],
    ):
        """Extract knowledge graph entities and relationships from text-rich columns."""
        if not self.kg_pipeline:
            logger.warning(f"KG pipeline not available, treating {column} as property")
            self._add_properties(batch, column, analysis, existing_proteins, stats)
            return

        uniprot_column = data_analysis["uniprot_column"]
        extraction_count = 0

        for row in batch:
            uniprot_id = row.get(uniprot_column, "").strip()
            if not uniprot_id or uniprot_id not in existing_proteins:
                continue

            text_content = row.get(column, "").strip()
            if not text_content or len(text_content) < 50:  # Skip very short text
                continue

            try:
                # Create a temporary document for KG extraction
                chunk = Chunk(chunk_id=f"{uniprot_id}_{column}", text=text_content)

                doc = ProcessedDocument(
                    doc_id=f"{uniprot_id}_{column}",
                    source=f"csv_enrichment_{column}",
                    chunks=[chunk],
                    metadata={
                        "source_protein": uniprot_id,
                        "source_column": column,
                        "extraction_type": "csv_text_field",
                    },
                )

                # Process with KG pipeline
                logger.info(
                    f"Extracting KG from {column} for protein {uniprot_id} (text length: {len(text_content)})"
                )
                processed_docs = await self.kg_pipeline.process_documents([doc])

                if processed_docs and processed_docs[0].chunks:
                    extracted_chunk = processed_docs[0].chunks[0]
                    nodes_count = (
                        len(extracted_chunk.nodes)
                        if hasattr(extracted_chunk, "nodes")
                        else 0
                    )
                    rels_count = (
                        len(extracted_chunk.relationships)
                        if hasattr(extracted_chunk, "relationships")
                        else 0
                    )

                    if nodes_count > 0 or rels_count > 0:
                        extraction_count += 1
                        logger.info(
                            f"Extracted {nodes_count} nodes and {rels_count} relationships from {column} for {uniprot_id}"
                        )

                        # Also store the text as a property on the protein
                        property_name = self._sanitize_property_name(column)
                        update_query = f"""
                        MATCH (p) WHERE p.uniprotID = $uniprot_id
                        SET p.{property_name} = $text_content
                        """

                        self._execute_query(
                            update_query,
                            {"uniprot_id": uniprot_id, "text_content": text_content},
                        )
                        stats["properties_added"] += 1

            except Exception as e:
                logger.error(f"Error extracting KG from {column} for {uniprot_id}: {e}")
                # Fallback to storing as property
                property_name = self._sanitize_property_name(column)
                update_query = f"""
                MATCH (p) WHERE p.uniprotID = $uniprot_id
                SET p.{property_name} = $text_content
                """

                try:
                    self._execute_query(
                        update_query,
                        {"uniprot_id": uniprot_id, "text_content": text_content},
                    )
                    stats["properties_added"] += 1
                except Exception as prop_error:
                    logger.error(
                        f"Error storing text as property for {uniprot_id}: {prop_error}"
                    )

        stats["kg_extractions_performed"] += extraction_count
        logger.info(
            f"Completed KG extraction for column {column}: {extraction_count} extractions performed"
        )

    def _is_text_rich_column(self, column: str, values: list[str]) -> bool:
        """Determine if a column contains rich text suitable for KG extraction."""
        # Column name patterns that suggest rich text content
        text_rich_keywords = [
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

        # Check column name
        column_lower = column.lower()
        has_text_keyword = any(
            keyword in column_lower for keyword in text_rich_keywords
        )

        # Calculate average text length
        if not values:
            return False

        avg_length = sum(len(val) for val in values) / len(values)

        # Check for biological content indicators
        biological_keywords = [
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

        # Sample some values to check for biological content
        sample_size = min(10, len(values))
        sample_values = values[:sample_size]
        biological_content_ratio = 0

        for val in sample_values:
            val_lower = val.lower()
            if any(keyword in val_lower for keyword in biological_keywords):
                biological_content_ratio += 1

        biological_content_ratio /= sample_size

        # Criteria for text-rich column:
        # 1. Has text-related column name OR
        # 2. Average length > 80 characters AND contains biological keywords in >30% of samples
        return has_text_keyword or (avg_length > 80 and biological_content_ratio > 0.3)

    def _sanitize_property_name(self, column: str) -> str:
        """Sanitize column name to be a valid Cypher property name."""
        # Replace spaces and special chars with underscores, make lowercase
        return (
            column.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def _create_node_label(self, column: str) -> str:
        """Create a node label from column name."""
        # Capitalize words and remove special characters
        words = column.replace("_", " ").replace("-", " ").split()
        return "".join(word.capitalize() for word in words)

    def _create_relationship_type(self, column: str) -> str:
        """Create a relationship type from column name."""
        # Use uppercase with underscores
        return self._sanitize_property_name(column).upper()

    def _normalize_binary_value(self, value: str) -> bool:
        """Normalize binary values to boolean."""
        value_lower = value.lower().strip()
        if value_lower in ["yes", "true", "1", "y", "t", "present", "positive"]:
            return True
        if value_lower in ["no", "false", "0", "n", "f", "absent", "negative"]:
            return False
        # Default to True for any other value
        return True

    def get_current_stats(self) -> dict[str, Any]:
        """Get current enrichment statistics."""
        # This is a generic processor, so we don't have specific stats to return
        return {}

    def close(self):
        """Clean up resources."""
        if self.db:
            self.db.close()


class EnrichmentManager:
    """
    Manager class that coordinates different enrichment processors.
    """

    def __init__(self, database: str = "cvd1", service: str = "local"):
        from pipeline.processors.cell_type_enricher import CellTypeEnricher
        from pipeline.processors.protein_function_llm_processor import (
            ProteinFunctionLLMProcessor,
        )

        self.database = database
        self.service = service
        # Use the generic processor for all enrichment types
        self.processors = {
            "generic_csv": GenericCSVEnrichmentProcessor,
            # Keep legacy processors for backward compatibility
            "subcellular_location": SubcellularLocationProcessor,
            "cell_type": CellTypeEnricher,
            "protein_function_llm": ProteinFunctionLLMProcessor,
        }

    def get_available_enrichments(self) -> list[str]:
        """Get list of available enrichment types."""
        return list(self.processors.keys())

    async def enrich(
        self,
        enrichment_type: str,
        file_path: str,
        empty_threshold: float = 40.0,
        unique_threshold_percentage: float = 5.0,
    ) -> dict[str, Any]:
        """
        Perform enrichment using the specified processor.

        Args:
            enrichment_type: Type of enrichment ('generic_csv', 'subcellular_location', etc.)
            file_path: Path to the data file or folder
            empty_threshold: Maximum percentage of empty values allowed (default: 40.0)
            unique_threshold_percentage: Minimum percentage of unique values for node class creation (default: 5.0)

        Returns:
            Enrichment statistics
        """
        if enrichment_type not in self.processors:
            raise ValueError(f"Unknown enrichment type: {enrichment_type}")

        processor_class = self.processors[enrichment_type]
        if enrichment_type == "generic_csv":
            processor = processor_class(
                database=self.database,
                service=self.service,
                empty_threshold=empty_threshold,
                unique_threshold_percentage=unique_threshold_percentage,
            )
        elif enrichment_type == "protein_function_llm":
            processor = processor_class(database=self.database, service=self.service)
        else:
            processor = processor_class(database=self.database)

        try:
            if enrichment_type == "generic_csv":
                stats = await processor.enrich_from_file(file_path)
            else:
                # Legacy processors (may need to be made async if they use KG extraction)
                stats = processor.enrich_from_file(file_path)
            return stats
        finally:
            processor.close()

    def get_enrichment_stats(self, enrichment_type: str) -> dict[str, Any]:
        """Get current statistics for an enrichment type."""
        if enrichment_type not in self.processors:
            raise ValueError(f"Unknown enrichment type: {enrichment_type}")

        processor_class = self.processors[enrichment_type]
        if enrichment_type == "protein_function_llm":
            processor = processor_class(database=self.database, service=self.service)
        else:
            processor = processor_class(database=self.database)

        try:
            return processor.get_current_stats()
        finally:
            processor.close()


async def main():
    """Command line interface for biological enrichment."""
    import argparse

    parser = argparse.ArgumentParser(description="Biological Data Enrichment Framework")
    parser.add_argument(
        "--enrichment-type",
        choices=[
            "generic_csv",
            "subcellular_location",
            "cell_type",
            "protein_function_llm",
            "list",
        ],
        help="Type of enrichment to perform (or 'list' to show available types)",
    )
    parser.add_argument(
        "--file",
        help="Path to data file or folder containing CSV files (not required for protein_function_llm)",
    )
    parser.add_argument(
        "--empty-threshold",
        type=float,
        default=40.0,
        help="Maximum percentage of empty values allowed in a column (default: 40.0)",
    )
    parser.add_argument(
        "--unique-threshold-percentage",
        type=float,
        default=5.0,
        help="Minimum percentage of unique values for creating node classes instead of properties (default: 5.0)",
    )
    parser.add_argument("--database", default="cvd1", help="Database name")
    parser.add_argument(
        "--service",
        choices=[
            "local",
            "openai",
            "hf-inference",
            "sagemaker",
            "sagemaker-llama3",
            "bedrock",
        ],
        default="local",
        help="LLM service for KG extraction",
    )

    args = parser.parse_args()

    try:
        manager = EnrichmentManager(database=args.database, service=args.service)

        if args.enrichment_type == "list":
            print("Available enrichment types:")
            for enrichment_type in manager.get_available_enrichments():
                print(f"  - {enrichment_type}")
            return

        if not args.enrichment_type:
            parser.error("--enrichment-type is required (or use 'list' to see options)")

        if not args.file and args.enrichment_type != "protein_function_llm":
            parser.error("--file is required")

        print(f"Performing {args.enrichment_type} enrichment...")
        print(f"File: {args.file}")

        stats = await manager.enrich(
            args.enrichment_type,
            args.file,
            empty_threshold=args.empty_threshold,
            unique_threshold_percentage=args.unique_threshold_percentage,
        )

        print("\n📊 Enrichment Results:")
        print(f"Type: {stats['enrichment_type']}")

        if stats["enrichment_type"] == "generic_csv_folder":
            # Folder results
            print(f"Folder: {stats['folder_path']}")
            print(f"Total files: {stats['total_files']}")
            print(f"Processed files: {stats['processed_files']}")
            print(f"Skipped files: {stats['skipped_files']}")

            agg = stats["aggregate_stats"]
            print(f"Total proteins processed: {agg['total_proteins_processed']}")
            print(f"Total properties added: {agg['total_properties_added']}")
            print(f"Total node classes created: {agg['total_node_classes_created']}")
            print(
                f"Total individual nodes created: {agg['total_individual_nodes_created']}"
            )
            print(f"Total relationships created: {agg['total_relationships_created']}")
            if agg["total_errors"] > 0:
                print(f"Total errors: {agg['total_errors']}")

            # # Show individual file results
            # print("\n📁 Individual File Results:")
            # for file_result in stats['file_results']:
            #     if 'error' in file_result:
            #         print(f"❌ {Path(file_result['file_path']).name}: {file_result['error']}")
            #     else:
            #         data_analysis = file_result.get('data_analysis', {})
            #         enrichment_stats = file_result.get('enrichment_stats', {})
            #         print(f"✅ {Path(file_result['file_path']).name}: {data_analysis.get('total_rows', 0)} rows, {enrichment_stats.get('properties_added', 0)} properties, {enrichment_stats.get('relationships_created', 0)} relationships")

        elif stats["enrichment_type"] == "generic_csv":
            # Single file results
            data_analysis = stats.get("data_analysis", {})
            enrichment_stats = stats.get("enrichment_stats", {})

            print(f"File: {stats['file_path']}")
            print(f"Data parsed: {data_analysis.get('total_rows', 0)} rows")
            print(f"UniProt column: {data_analysis.get('uniprot_column', 'N/A')}")

            # Show column analysis (commented out for brief output)
            column_analysis = data_analysis.get("column_analysis", {})
            print(f"Columns analyzed: {len(column_analysis)}")
            # Detailed analysis output commented out to keep terminal responses brief
            # for col, analysis in column_analysis.items():
            #     strategy = analysis.get('strategy', 'unknown')
            #     if strategy == 'skip':
            #         reason = analysis.get('reason', 'unknown')
            #         print(f"  - {col}: {analysis.get('type', 'unknown')} → SKIPPED ({reason})")
            #     else:
            #         unique_count = analysis.get('unique_count', 0)
            #         empty_pct = analysis.get('empty_percentage', 0)
            #         na_pct = analysis.get('na_percentage', 0)
            #         has_multi = analysis.get('has_multiple_values', False)
            #         multi_indicator = " (multi-value)" if has_multi else ""
            #         print(f"  - {col}: {analysis.get('type', 'unknown')} ({unique_count} unique, {empty_pct:.1f}% empty){multi_indicator} → {strategy}")

            # Show enrichment stats
            print(
                f"Proteins processed: {enrichment_stats.get('proteins_processed', 0)}"
            )
            print(f"Properties added: {enrichment_stats.get('properties_added', 0)}")
            print(
                f"Node classes created: {enrichment_stats.get('node_classes_created', 0)}"
            )
            print(
                f"Individual nodes created: {enrichment_stats.get('individual_nodes_created', 0)}"
            )
            print(
                f"Relationships created: {enrichment_stats.get('relationships_created', 0)}"
            )

            if enrichment_stats.get("errors"):
                print(f"Errors: {len(enrichment_stats['errors'])}")
        else:
            # Legacy format
            if "parsing_stats" in stats:
                parsing = stats["parsing_stats"]
                print(
                    f"Data parsed: {parsing.get('proteins_processed', 0)} proteins, {parsing.get('unique_locations', 0)} unique items"
                )

            if "node_creation_stats" in stats:
                nodes = stats["node_creation_stats"]
                print(f"Nodes created: {nodes.get('created', 0)}")

            if "relationship_stats" in stats:
                rels = stats["relationship_stats"]
                print(f"Relationships created: {rels.get('relationships_created', 0)}")
                print(f"Entities matched: {rels.get('proteins_matched', 0)}")

            if stats.get("errors"):
                print(f"Errors: {len(stats['errors'])}")

        # Show current stats
        current_stats = manager.get_enrichment_stats(args.enrichment_type)
        if current_stats:
            print("\n📈 Current Graph Stats:")
            for key, value in current_stats.items():
                print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Enrichment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
