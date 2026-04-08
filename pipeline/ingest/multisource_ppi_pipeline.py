"""
Enhanced PPI integration pipeline for multi-source data management.
Handles multiple CSV files from different sources (e.g., FunCoup, STRING, BioGRID).
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.processors.ppi_csv_processor import PPICSVProcessor
from src.core.multisource_ppi_database import MultiSourcePPIDatabaseInterface
from src.models.ppi_models import PPIBatchData
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class MultiSourcePPIPipeline:
    """
    Enhanced pipeline for processing multiple PPI data sources.
    Handles source attribution, evidence merging, and multi-source analytics.
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
        merge_strategy: str = "merge_evidence",
    ):
        """
        Initialize multi-source PPI integration pipeline.

        Args:
            database_uri: Neo4j database URI
            database_user: Neo4j username
            database_password: Neo4j password
            database_name: Neo4j database name
            min_confidence_threshold: Minimum confidence score for interactions
            validate_uniprotIDs: Whether to validate UniProt ID format
            batch_size: Number of interactions to process in each batch
            merge_strategy: How to handle overlapping interactions between sources
                - "merge_evidence": Combine evidence from all sources
                - "update_best": Keep highest confidence interaction
                - "preserve_all": Keep separate relationships per source
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.validate_uniprotIDs = validate_uniprotIDs
        self.batch_size = batch_size
        self.merge_strategy = merge_strategy

        # Initialize CSV processor
        self.csv_processor = PPICSVProcessor(
            uniprot_validation=validate_uniprotIDs,
            min_confidence_threshold=min_confidence_threshold,
            deduplicate=True,
        )

        # Initialize multi-source database interface
        self.db = MultiSourcePPIDatabaseInterface(
            uri=database_uri,
            user=database_user,
            password=database_password,
            database=database_name,
        )

        # Multi-source processing statistics
        self.source_stats = {}  # type: Dict[str, Dict[str, Any]]
        self.overall_stats = {
            "sources_processed": 0,
            "total_files_processed": 0,
            "total_interactions_processed": 0,
            "total_interactions_stored": 0,
            "merge_operations": 0,
            "source_comparison_results": {},
            "start_time": None,
            "end_time": None,
        }

        logger.info(
            f"Initialized MultiSourcePPIPipeline with merge_strategy='{merge_strategy}'"
        )

    def process_source_directory(
        self, source_configs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Process multiple data sources from different directories or files.

        Args:
            source_configs: List of source configuration dictionaries with keys:
                - 'name': Source name (e.g., 'STRING', 'FunCoup', 'BioGRID')
                - 'path': Path to CSV file or directory
                - 'file_pattern': File pattern if path is directory (default: '*.csv')
                - 'confidence_column': Custom confidence column name (optional)
                - 'evidence_type': Default evidence type for this source (optional)

        Returns:
            Dictionary with processing results for all sources
        """
        logger.info(f"Processing {len(source_configs)} data sources")
        self.overall_stats["start_time"] = datetime.now()

        results = {
            "success": True,
            "sources_processed": 0,
            "source_results": [],
            "comparison_results": {},
            "errors": [],
        }  # type: Dict[str, Any]

        try:
            # Process each source
            for source_config in source_configs:
                source_name = source_config["name"]
                source_path = source_config["path"]

                logger.info(f"Processing source: {source_name} from {source_path}")

                # Configure processor for this source
                if "evidence_type" in source_config:
                    # Could customize processor per source if needed
                    pass

                # Process source data
                if os.path.isfile(source_path):
                    source_result = self._process_single_source_file(source_config)
                elif os.path.isdir(source_path):
                    source_result = self._process_source_directory(source_config)
                else:
                    error = f"Source path not found: {source_path}"
                    logger.error(error)
                    source_result = {"success": False, "error": error}

                # Store source results
                source_result["source_name"] = source_name
                results["source_results"].append(source_result)

                if source_result["success"]:
                    results["sources_processed"] += 1
                    self.source_stats[source_name] = source_result
                else:
                    results["errors"].append(
                        f"{source_name}: {source_result.get('error', 'Unknown error')}"
                    )

            # Perform cross-source analysis
            if len(self.source_stats) >= 2:
                logger.info("Performing cross-source comparison analysis")
                comparison_results = self._perform_source_comparisons()
                results["comparison_results"] = comparison_results
                self.overall_stats["source_comparison_results"] = comparison_results

            self.overall_stats["end_time"] = datetime.now()
            self.overall_stats["sources_processed"] = results["sources_processed"]

            results["overall_stats"] = self.overall_stats.copy()
            results["network_stats"] = self.get_multisource_network_stats()

            logger.info(
                f"Multi-source processing complete: {results['sources_processed']}/{len(source_configs)} sources successful"
            )
            return results

        except Exception as e:
            error = f"Multi-source processing failed: {e!s}"
            logger.error(error, exc_info=True)
            results["success"] = False
            results["errors"].append(error)
            return results

    def _process_single_source_file(
        self, source_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a single CSV file for a source."""
        source_name = source_config["name"]
        file_path = source_config["path"]

        try:
            # Process CSV file
            batch_data = self.csv_processor.process_csv_file(file_path)

            # Set source name for all interactions
            for interaction in batch_data.interactions:
                interaction.source_database = source_name

            # Store in database with multi-source strategy
            if batch_data.interactions:
                storage_stats = self._store_multisource_batch(batch_data)

                result = {
                    "success": True,
                    "file_path": file_path,
                    "interactions_processed": len(batch_data.interactions),
                    "storage_stats": storage_stats,
                    "validation_warnings": batch_data.validation_result.warnings
                    if batch_data.validation_result
                    else [],
                }
            else:
                result = {
                    "success": True,
                    "file_path": file_path,
                    "interactions_processed": 0,
                    "warning": "No valid interactions found",
                }

            return result

        except Exception as e:
            error = f"Failed to process source file {file_path}: {e!s}"
            logger.error(error)
            return {"success": False, "error": error, "file_path": file_path}

    def _process_source_directory(
        self, source_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Process all CSV files in a source directory."""
        source_config["name"]
        directory_path = source_config["path"]
        file_pattern = source_config.get("file_pattern", "*.csv")

        try:
            directory = Path(directory_path)
            csv_files = list(directory.glob(file_pattern))

            if not csv_files:
                return {
                    "success": True,
                    "warning": f"No files found matching pattern {file_pattern}",
                    "files_processed": 0,
                }

            total_interactions = 0
            file_results = []

            for csv_file in csv_files:
                file_config = source_config.copy()
                file_config["path"] = str(csv_file)

                file_result = self._process_single_source_file(file_config)
                file_results.append(file_result)

                if file_result["success"]:
                    total_interactions += file_result.get("interactions_processed", 0)

            return {
                "success": True,
                "directory_path": directory_path,
                "files_found": len(csv_files),
                "files_processed": len([r for r in file_results if r["success"]]),
                "total_interactions_processed": total_interactions,
                "file_results": file_results,
            }

        except Exception as e:
            error = f"Failed to process source directory {directory_path}: {e!s}"
            logger.error(error)
            return {"success": False, "error": error}

    def _store_multisource_batch(self, batch_data: PPIBatchData) -> dict[str, Any]:
        """Store batch using multi-source strategy."""
        return self.db.store_ppi_batch_multisource(batch_data, self.merge_strategy)

    def _perform_source_comparisons(self) -> dict[str, Any]:
        """Perform pairwise comparisons between data sources."""
        comparisons = {}
        source_names = list(self.source_stats.keys())

        for i, source_a in enumerate(source_names):
            for source_b in source_names[i + 1 :]:
                comparison_key = f"{source_a}_vs_{source_b}"
                logger.info(f"Comparing {source_a} vs {source_b}")

                comparison_result = self.db.get_source_comparison(source_a, source_b)
                comparisons[comparison_key] = comparison_result

        return comparisons

    def get_multisource_network_stats(self) -> dict[str, Any]:
        """Get enhanced network statistics for multi-source data."""
        return self.db.get_multisource_network_stats()

    def get_source_specific_interactions(
        self, protein_a: str, protein_b: str
    ) -> list[dict[str, Any]]:
        """Get all interactions between two proteins showing source-specific attributes."""
        return self.db.get_interaction_by_sources(protein_a, protein_b)

    def export_multisource_report(self, output_path: str) -> None:
        """Export comprehensive multi-source processing report."""
        try:
            import json

            report = {
                "pipeline_config": {
                    "merge_strategy": self.merge_strategy,
                    "min_confidence_threshold": self.min_confidence_threshold,
                    "validate_uniprotIDs": self.validate_uniprotIDs,
                    "batch_size": self.batch_size,
                },
                "overall_stats": self.overall_stats,
                "source_stats": self.source_stats,
                "network_stats": self.get_multisource_network_stats(),
                "generated_at": datetime.now().isoformat(),
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Multi-source processing report exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export multi-source report: {e}")

    def close(self):
        """Close database connections."""
        self.db.close()
        logger.info("Multi-source PPI pipeline closed")


def create_example_multisource_config():
    """Create example configuration for multiple PPI sources."""
    return [
        {
            "name": "STRING",
            "path": "/data/ppi_sources/string_interactions.csv",
            "evidence_type": "computational",
            "confidence_column": "combined_score",
        },
        {
            "name": "FunCoup",
            "path": "/data/ppi_sources/funcoup_interactions.csv",
            "evidence_type": "computational",
            "confidence_column": "confidence",
        },
        {
            "name": "BioGRID",
            "path": "/data/ppi_sources/biogrid_directory/",
            "file_pattern": "BIOGRID*.csv",
            "evidence_type": "experimental",
        },
        {
            "name": "IntAct",
            "path": "/data/ppi_sources/intact_interactions.csv",
            "evidence_type": "experimental",
        },
    ]


def run_multisource_example():
    """Example of processing multiple PPI sources."""
    # Initialize pipeline
    pipeline = MultiSourcePPIPipeline(
        min_confidence_threshold=0.2,
        merge_strategy="merge_evidence",  # Combine evidence from all sources
        batch_size=1000,
    )

    try:
        # Configure sources
        source_configs = create_example_multisource_config()

        # Process all sources
        results = pipeline.process_source_directory(source_configs)

        if results["success"]:
            print(f"✓ Processed {results['sources_processed']} sources successfully")

            # Show network statistics
            network_stats = results["network_stats"]
            print(f"Total relationships: {network_stats.get('total_relationships', 0)}")
            print(f"Unique sources: {network_stats.get('unique_sources', 0)}")
            print(
                f"Multi-source protein pairs: {network_stats.get('multi_source_pairs', 0)}"
            )

            # Show source distribution
            source_dist = network_stats.get("source_distribution", {})
            print("Source distribution:")
            for source, count in source_dist.items():
                print(f"  {source}: {count} interactions")

            # Show comparisons
            comparisons = results.get("comparison_results", {})
            print("Source comparisons:")
            for comparison, data in comparisons.items():
                print(
                    f"  {comparison}: {data.get('common_interactions', 0)} common interactions"
                )

        else:
            print(f"✗ Processing failed: {results['errors']}")

        # Export report
        pipeline.export_multisource_report("/tmp/multisource_ppi_report.json")

    finally:
        pipeline.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Source PPI Integration Pipeline"
    )
    parser.add_argument(
        "--source-dir", required=True, help="Directory containing PPI source files"
    )
    parser.add_argument("--database", default="neo4j", help="Database name")
    parser.add_argument(
        "--merge-strategy",
        choices=["merge_evidence", "update_best", "preserve_all"],
        default="merge_evidence",
        help="How to handle overlapping interactions",
    )
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
    pipeline = MultiSourcePPIPipeline(
        database_name=args.database,
        min_confidence_threshold=args.min_confidence,
        validate_uniprotIDs=args.validate_uniprot,
        batch_size=args.batch_size,
        merge_strategy=args.merge_strategy,
    )

    try:
        print(f"Processing multi-source PPI data from: {args.source_dir}")
        print(f"Merge strategy: {args.merge_strategy}")
        print(f"Min confidence: {args.min_confidence}")
        print(f"Batch size: {args.batch_size}")

        # Configure sources (auto-detect from directory)
        import os
        from pathlib import Path

        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"❌ Source directory not found: {args.source_dir}")
            sys.exit(1)

        # Auto-detect source files
        source_configs = []
        for csv_file in source_dir.glob("*.csv"):
            source_name = csv_file.stem.replace("_", " ").title()
            source_configs.append(
                {
                    "name": source_name,
                    "path": str(csv_file),
                    "evidence_type": "experimental"
                    if "biogrid" in source_name.lower()
                    or "intact" in source_name.lower()
                    else "computational",
                }
            )

        if not source_configs:
            print(f"❌ No CSV files found in {args.source_dir}")
            sys.exit(1)

        print(f"Found {len(source_configs)} source files:")
        for config in source_configs:
            print(f"  - {config['name']}: {config['path']}")

        # Process all sources
        print("\nProcessing sources...")
        results = pipeline.process_source_directory(source_configs)

        if results["success"]:
            print("✅ Multi-source PPI data processed successfully!")
            print(f"   Sources processed: {results['sources_processed']}")
            print(
                f"   Total interactions: {results['overall_stats']['total_interactions_processed']}"
            )

            # Show network statistics
            network_stats = results["network_stats"]
            print("\n📊 Network Statistics:")
            print(
                f"   Total relationships: {network_stats.get('total_relationships', 0)}"
            )
            print(f"   Unique sources: {network_stats.get('unique_sources', 0)}")
            print(
                f"   Multi-source pairs: {network_stats.get('multi_source_pairs', 0)}"
            )

            # Show source distribution
            source_dist = network_stats.get("source_distribution", {})
            if source_dist:
                print("   Source distribution:")
                for source, count in source_dist.items():
                    print(f"     {source}: {count} interactions")

            # Show comparisons
            comparisons = results.get("comparison_results", {})
            if comparisons:
                print("   Source comparisons:")
                for comparison, data in comparisons.items():
                    print(
                        f"     {comparison}: {data.get('common_interactions', 0)} common interactions"
                    )

            # Export report
            report_path = "/tmp/multisource_ppi_report.json"
            pipeline.export_multisource_report(report_path)
            print(f"📄 Report exported to: {report_path}")

        else:
            print(f"❌ Processing failed: {results['errors']}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        pipeline.close()
