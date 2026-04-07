"""
Pipeline for integrating Protein-Protein Interaction (PPI) data into the knowledge graph.
Handles the complete workflow from CSV processing to database storage.
"""

import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.ppi_database import PPIDatabaseInterface
from src.models.ppi_models import PPIBatchData, PPINetworkStats
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class PPIIntegrationPipeline:
    """
    Complete pipeline for PPI data integration.
    Handles CSV processing, validation, and database storage.
    """

    def __init__(
        self,
        database_uri: str | None = None,
        database_user: str | None = None,
        database_password: str | None = None,
        database_name: str | None = None,
        min_confidence_threshold: float = 0.0,
        validate_uniprotIDs: bool = True,
        batch_size: int = 1000,
    ):
        """
        Initialize PPI integration pipeline.

        Args:
            database_uri: Neo4j database URI
            database_user: Neo4j username
            database_password: Neo4j password
            database_name: Neo4j database name
            min_confidence_threshold: Minimum confidence score for interactions
            validate_uniprotIDs: Whether to validate UniProt ID format
            batch_size: Number of interactions to process in each batch
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.validate_uniprotIDs = validate_uniprotIDs
        self.batch_size = batch_size

        # Initialize CSV processor
        from pipeline.processors.ppi_csv_processor import PPICSVProcessor

        self.csv_processor = PPICSVProcessor(
            uniprot_validation=validate_uniprotIDs,
            min_confidence_threshold=min_confidence_threshold,
            deduplicate=True,
        )

        # Initialize database interface
        self.db = PPIDatabaseInterface(
            uri=database_uri,
            user=database_user,
            password=database_password,
            database=database_name,
        )

        # Processing statistics
        self.processing_stats: dict[str, Any] = {
            "files_processed": 0,
            "total_interactions_processed": 0,
            "total_interactions_stored": 0,
            "total_proteins_created": 0,
            "total_evidence_records": 0,
            "processing_errors": [],
            "start_time": None,
            "end_time": None,
        }

        logger.info(
            f"Initialized PPI integration pipeline with batch_size={batch_size}"
        )

    def process_csv_file(
        self, csv_path: str, source_name: str | None = None
    ) -> dict[str, Any]:
        """
        Process a single CSV file containing PPI data.

        Args:
            csv_path: Path to CSV file
            source_name: Optional name for the data source

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing PPI CSV file: {csv_path}")
        start_time = time.time()

        try:
            # Validate file exists
            if not os.path.exists(csv_path):
                error = f"CSV file not found: {csv_path}"
                logger.error(error)
                return {"success": False, "error": error}

            # Process CSV file
            batch_data = self.csv_processor.process_csv_file(csv_path)

            # Check validation results
            if (
                batch_data.validation_result
                and not batch_data.validation_result.is_valid
            ):
                error = f"CSV validation failed: {batch_data.validation_result.errors}"
                logger.error(error)
                return {
                    "success": False,
                    "error": error,
                    "warnings": batch_data.validation_result.warnings
                    if batch_data.validation_result
                    else [],
                    "batch_data": batch_data,
                }

            # Update source database if provided
            if source_name:
                for interaction in batch_data.interactions:
                    interaction.source_database = source_name

            # Store in database
            if batch_data.interactions:
                storage_stats = self._store_interactions_batch(batch_data)

                # Update processing statistics
                self.processing_stats["files_processed"] += 1
                self.processing_stats["total_interactions_processed"] += len(
                    batch_data.interactions
                )
                self.processing_stats["total_interactions_stored"] += storage_stats.get(
                    "interactions_created", 0
                )
                self.processing_stats["total_proteins_created"] += storage_stats.get(
                    "proteins_created", 0
                )
                self.processing_stats["total_evidence_records"] += storage_stats.get(
                    "evidence_records_created", 0
                )

                processing_time = time.time() - start_time

                result = {
                    "success": True,
                    "interactions_processed": len(batch_data.interactions),
                    "storage_stats": storage_stats,
                    "processing_time_seconds": processing_time,
                    "warnings": batch_data.validation_result.warnings
                    if batch_data.validation_result
                    else [],
                    "batch_data": batch_data,
                }

                logger.info(
                    f"Successfully processed {csv_path}: {len(batch_data.interactions)} interactions in {processing_time:.2f}s"
                )
                return result
            warning = f"No valid interactions found in {csv_path}"
            logger.warning(warning)
            return {
                "success": True,
                "interactions_processed": 0,
                "warning": warning,
                "batch_data": batch_data,
            }

        except Exception as e:
            error = f"Failed to process {csv_path}: {e!s}"
            logger.error(error, exc_info=True)
            self.processing_stats["processing_errors"].append(error)
            return {"success": False, "error": error}

    def process_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.csv",
        source_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Process all CSV/TSV files in a directory.

        Args:
            directory_path: Path to directory containing CSV/TSV files
            file_pattern: File pattern to match (default: *.csv, will also look for *.tsv)
            source_name: Optional name for the data source

        Returns:
            Dictionary with overall processing results
        """
        logger.info(f"Processing PPI CSV/TSV files in directory: {directory_path}")
        self.processing_stats["start_time"] = datetime.now()

        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                error = f"Directory not found or not a directory: {directory_path}"
                logger.error(error)
                return {"success": False, "error": error}

            # Find CSV and TSV files
            csv_files = list(directory.glob("*.csv"))
            tsv_files = list(directory.glob("*.tsv"))
            all_files = csv_files + tsv_files

            if not all_files:
                warning = f"No CSV/TSV files found in {directory_path}"
                logger.warning(warning)
                return {"success": True, "warning": warning, "files_processed": 0}

            logger.info(
                f"Found {len(all_files)} files to process ({len(csv_files)} CSV, {len(tsv_files)} TSV)"
            )

            # Process each file
            file_results: list[dict[str, Any]] = []
            for data_file in all_files:
                file_source_name = source_name or data_file.stem
                result = self.process_csv_file(str(data_file), file_source_name)
                file_results.append({"file": str(data_file), "result": result})

            self.processing_stats["end_time"] = datetime.now()

            # Generate summary
            successful_files = sum(
                1 for r in file_results if r["result"].get("success", False)
            )
            total_interactions = sum(
                r["result"].get("interactions_processed", 0) for r in file_results
            )

            summary = {
                "success": True,
                "files_found": len(csv_files),
                "files_processed_successfully": successful_files,
                "total_interactions_processed": total_interactions,
                "file_results": file_results,
                "processing_stats": self.processing_stats.copy(),
            }

            logger.info(
                f"Directory processing complete: {successful_files}/{len(csv_files)} files successful, {total_interactions} interactions"
            )
            return summary

        except Exception as e:
            error = f"Failed to process directory {directory_path}: {e!s}"
            logger.error(error, exc_info=True)
            self.processing_stats["processing_errors"].append(error)
            return {"success": False, "error": error}

    def _store_interactions_batch(self, batch_data: PPIBatchData) -> dict[str, Any]:
        """Store a batch of interactions with batching support."""
        if not batch_data.interactions:
            return {
                "interactions_created": 0,
                "proteins_created": 0,
                "evidence_records_created": 0,
            }

        total_stats = {
            "interactions_created": 0,
            "proteins_created": 0,
            "proteins_updated": 0,
            "evidence_records_created": 0,
            "errors": [],
        }

        # Process in batches to avoid memory issues
        interactions = batch_data.interactions
        for i in range(0, len(interactions), self.batch_size):
            batch = interactions[i : i + self.batch_size]
            batch_subset = PPIBatchData(
                interactions=batch,
                source_file=batch_data.source_file,
                total_rows=len(batch),
                processed_rows=len(batch),
            )

            batch_stats = self.db.store_ppi_batch(batch_subset)

            # Aggregate statistics
            for key in [
                "interactions_created",
                "proteins_created",
                "proteins_updated",
                "evidence_records_created",
            ]:
                total_stats[key] += batch_stats.get(key, 0)

            total_stats["errors"].extend(batch_stats.get("errors", []))

            logger.debug(
                f"Processed batch {i // self.batch_size + 1}: {len(batch)} interactions"
            )

        return total_stats

    def get_network_statistics(self) -> PPINetworkStats:
        """Get current network statistics from the database."""
        return self.db.get_ppi_network_stats()

    def export_processing_report(self, output_path: str) -> None:
        """Export processing statistics to a file."""
        try:
            import json

            report = {
                "pipeline_stats": self.processing_stats,
                "network_stats": asdict(self.get_network_statistics()),
                "pipeline_config": {
                    "min_confidence_threshold": self.min_confidence_threshold,
                    "validate_uniprotIDs": self.validate_uniprotIDs,
                    "batch_size": self.batch_size,
                },
                "generated_at": datetime.now().isoformat(),
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Processing report exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export processing report: {e}")

    def validate_csv_format(self, csv_path: str) -> dict[str, Any]:
        """
        Validate CSV format without processing.

        Args:
            csv_path: Path to CSV file

        Returns:
            Validation results
        """
        try:
            import polars as pl

            # Determine separator based on file extension
            file_path = Path(csv_path)
            separator = "\t" if file_path.suffix.lower() == ".tsv" else ","

            df = pl.read_csv(csv_path, separator=separator, infer_schema_length=10000)
            validation_result = self.csv_processor._validate_csv_structure(df)
            column_mapping = self.csv_processor._map_columns(df.columns, df)

            return {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "total_rows": df.height,
                "columns": df.columns,
                "column_mapping": column_mapping,
                "sample_data": df.head(3).to_dicts() if df.height > 0 else [],
            }

        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Failed to validate CSV: {e!s}"],
                "warnings": [],
            }

    def cleanup_database(self, source_database: str | None = None) -> int:
        """
        Clean up PPI data from database.

        Args:
            source_database: If provided, only clean up data from this source

        Returns:
            Number of interactions deleted
        """
        logger.info(
            "Cleaning up PPI data"
            + (f" from source: {source_database}" if source_database else "")
        )
        return self.db.delete_ppi_data(source_database)

    def close(self):
        """Close database connections."""
        self.db.close()
        logger.info("PPI integration pipeline closed")


# Example usage functions for testing
def create_sample_ppi_csv(output_path: str, num_interactions: int = 100) -> None:
    """Create a sample PPI CSV file for testing."""
    import random

    import polars as pl

    # Sample UniProt IDs (real human proteins)
    sample_proteins = [
        "P04637",
        "P53_HUMAN",
        "Q13547",
        "P01308",
        "P02768",
        "P00738",
        "P01133",
        "P10636",
        "P35222",
        "P04049",
        "P28482",
        "P51587",
        "P35354",
        "P29474",
        "P08047",
        "P07900",
        "P11142",
        "P13569",
        "P20226",
        "P35228",
        "P04626",
        "P07948",
        "P06493",
        "P08069",
    ]

    interactions = []
    for _i in range(num_interactions):
        protein_a = random.choice(sample_proteins)
        protein_b = random.choice([p for p in sample_proteins if p != protein_a])

        interactions.append(
            {
                "protein_a_uniprot": protein_a,
                "protein_b_uniprot": protein_b,
                "confidence_score": round(random.uniform(0.1, 1.0), 3),
                "evidence_type": random.choice(
                    ["experimental", "computational", "literature", "database"]
                ),
                "interaction_type": random.choice(
                    ["binding", "regulation", "catalysis", "complex_formation"]
                ),
                "source": random.choice(["STRING", "BioGRID", "IntAct", "MINT"]),
                "method": random.choice(["Y2H", "Co-IP", "PCA", "FRET", "MS"]),
                "pubmed_id": f"{random.randint(10000000, 35000000)}"
                if random.random() > 0.3
                else None,
            }
        )

    df = pl.DataFrame(interactions)
    df.write_csv(output_path)
    logger.info(
        f"Created sample PPI CSV with {num_interactions} interactions: {output_path}"
    )


def run_sample_pipeline():
    """Run a sample pipeline for demonstration."""
    # Create sample data
    sample_csv = "/tmp/sample_ppi_data.csv"
    create_sample_ppi_csv(sample_csv, 50)

    # Initialize pipeline
    pipeline = PPIIntegrationPipeline(
        min_confidence_threshold=0.3, validate_uniprotIDs=True, batch_size=25
    )

    try:
        # Validate CSV format
        validation_result = pipeline.validate_csv_format(sample_csv)
        print("Validation Result:", validation_result)

        # Process CSV file
        processing_result = pipeline.process_csv_file(sample_csv, "sample_data")
        print("Processing Result:", processing_result)

        # Get network statistics
        stats = pipeline.get_network_statistics()
        print("Network Stats:", asdict(stats))

        # Export report
        pipeline.export_processing_report("/tmp/ppi_processing_report.json")

    finally:
        pipeline.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPI Integration Pipeline")
    parser.add_argument("--csv-file", required=True, help="Path to PPI CSV file")
    parser.add_argument("--database", default="neo4j", help="Database name")
    parser.add_argument(
        "--min-confidence", type=float, default=0.0, help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for processing"
    )
    parser.add_argument(
        "--validate-uniprot",
        action="store_true",
        default=True,
        help="Validate UniProt IDs",
    )
    parser.add_argument(
        "--no-validate-uniprot",
        action="store_false",
        dest="validate_uniprot",
        help="Skip UniProt ID validation",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PPIIntegrationPipeline(
        database_name=args.database,
        min_confidence_threshold=args.min_confidence,
        validate_uniprotIDs=args.validate_uniprot,
        batch_size=args.batch_size,
    )

    try:
        print(f"Processing PPI data from: {args.csv_file}")
        print(f"Min confidence: {args.min_confidence}")
        print(f"Batch size: {args.batch_size}")

        # Validate CSV format
        print("Validating CSV format...")
        validation_result = pipeline.validate_csv_format(args.csv_file)
        if not validation_result.get("is_valid", False):
            print(f"❌ CSV validation failed: {validation_result.get('errors', [])}")
            sys.exit(1)
        print("✅ CSV format is valid")

        # Process CSV file
        print("Processing CSV file...")
        processing_result = pipeline.process_csv_file(args.csv_file, "ppi_data")

        if processing_result["success"]:
            print("✅ PPI data processed successfully!")
            print(
                f"   Proteins created: {processing_result['storage_stats']['proteins_created']}"
            )
            print(
                f"   Interactions stored: {processing_result['storage_stats']['interactions_created']}"
            )
            print(
                f"   Evidence records: {processing_result['storage_stats']['evidence_records_created']}"
            )

            # Get network statistics
            stats = pipeline.get_network_statistics()
            print("\n📊 Network Statistics:")
            print(f"   Total interactions: {stats.total_interactions}")
            print(f"   Unique proteins: {stats.unique_proteins}")
            print(f"   Interaction types: {list(stats.interaction_type_counts.keys())}")

            # Export report
            report_path = "/tmp/ppi_processing_report.json"
            pipeline.export_processing_report(report_path)
            print(f"📄 Report exported to: {report_path}")

        else:
            print(
                f"❌ Processing failed: {processing_result.get('error', 'Unknown error')}"
            )
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        pipeline.close()
