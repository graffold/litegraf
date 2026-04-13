"""
Enrichment Orchestrator - Main coordinator for CSV enrichment pipeline.

This class orchestrates the complete enrichment process following the hierarchy:
1. Column Analysis - Analyze columns to determine data types and strategies
2. Text Rich Processing - Extract KG from biological text descriptions
3. Node Annotation - Add properties and create node classes

It provides a unified interface for CSV enrichment with proper error handling,
logging, and progress tracking.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.interfaces import GraphStore, LLMProvider

from .base import EnrichmentStats
from .column_analyzer import ColumnAnalyzer
from .node_annotator import NodeAnnotator
from .text_rich_processor import TextRichProcessor

logger = logging.getLogger(__name__)


class EnrichmentOrchestrator:
    """Main orchestrator for CSV enrichment operations."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm_provider: LLMProvider,
        *,
        database: str = "cvd1",
    ):
        """
        Initialize enrichment orchestrator.

        Args:
            graph_store: Graph database backend
            llm_provider: LLM backend for KG extraction
            database: Database name
        """
        if not isinstance(graph_store, GraphStore):
            raise TypeError(
                f"graph_store must be a GraphStore, got {type(graph_store)}"
            )
        if not isinstance(llm_provider, LLMProvider):
            raise TypeError(
                f"llm_provider must be a LLMProvider, got {type(llm_provider)}"
            )
        self.db = graph_store
        self.llm = llm_provider
        self.database = database

        # Initialize components
        self.column_analyzer = ColumnAnalyzer()

        # Initialize processors with injected graph store
        self.text_processor = TextRichProcessor(
            graph_store, database=database
        )
        self.node_annotator = NodeAnnotator(graph_store, database=database)

    async def enrich_csv(
        self,
        file_path: str,
        skip_text_rich: bool = False,
        skip_node_annotation: bool = False,
        resume: bool = False,
        offset: int = 0,
        limit: int | None = None,
        batch_size: int = 1000,
        manual_column_handlers: dict[int, str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform complete CSV enrichment following the hierarchy.

        Args:
            file_path: Path to CSV file to enrich
            skip_text_rich: Skip KG extraction from text-rich columns
            skip_node_annotation: Skip property and node class creation
            resume: Resume processing from where it left off (checks existing proteins)
            offset: Start processing from row N (0-based index)
            limit: Process only N rows (useful for testing or batching)
            batch_size: Process CSV in batches of N rows
            manual_column_handlers: Dictionary mapping column index to handler shorthand (e.g., {0: 'protein-id', 4: 'prop'})
            **kwargs: Additional parameters for processors

        Returns:
            Dictionary with enrichment results and statistics
        """
        logger.info(f"Starting CSV enrichment for {file_path}")
        start_time = datetime.now()

        try:
            # Set manual column handlers if provided
            if manual_column_handlers:
                parsed_handlers = {}
                for col_idx, handler_str in manual_column_handlers.items():
                    try:
                        parsed_handlers[col_idx] = (
                            self.column_analyzer.parse_handler_string(handler_str)
                        )
                    except ValueError as e:
                        logger.error(
                            f"Invalid handler '{handler_str}' for column {col_idx}: {e}"
                        )
                        return {
                            "success": False,
                            "error": f"Invalid column handler: {e}",
                            "file_path": file_path,
                        }

                self.column_analyzer.set_manual_handlers(parsed_handlers)
                logger.info(f"Applied {len(parsed_handlers)} manual column handlers")

            # Phase 1: Column Analysis
            logger.info("Phase 1: Analyzing CSV columns...")
            analysis_start = time.perf_counter()

            column_analyses, metadata = await self.column_analyzer.analyze_csv(
                file_path
            )

            analysis_time = time.perf_counter() - analysis_start
            logger.info(f"✅ Column analysis complete in {analysis_time:.2f}s")

            # Log column analysis summary
            self._log_analysis_summary(column_analyses)

            # Check progress file if resume is True
            progress_file = Path(file_path).with_suffix(".progress")
            if resume and offset == 0 and progress_file.exists():
                try:
                    with open(progress_file) as f:
                        content = f.read().strip()
                        if content:
                            saved_offset = int(content)
                            logger.info(
                                f"Found progress file. Resuming from row {saved_offset}"
                            )
                            offset = saved_offset
                except ValueError:
                    logger.warning("Invalid progress file found, ignoring.")

            # Get CSV data for processing with resume/batch options
            csv_data, skip_info = await self._get_csv_data_for_resume(
                file_path, resume, offset, limit, batch_size, metadata
            )
            uniprot_column = metadata.get("uniprot_column")

            if skip_info:
                logger.info(f"Resume processing: {skip_info['processed_message']}")
                if skip_info["skipped_rows"] > 0:
                    logger.info(
                        f"⏭️  Skipped {skip_info['skipped_rows']} rows (offset: {offset})"
                    )
                if skip_info["already_processed"] > 0:
                    logger.info(
                        f"🔄 Skipped {skip_info['already_processed']} already processed proteins"
                    )

            # Initialize combined stats
            combined_stats = EnrichmentStats()
            phase_results = {}

            # Process data in batches to allow progress tracking and resumability
            total_rows = len(csv_data)
            if total_rows > 0:
                logger.info(f"Processing {total_rows} rows in batches of {batch_size}")

                for i in range(0, total_rows, batch_size):
                    batch_data = csv_data[i : i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (total_rows + batch_size - 1) // batch_size
                    logger.info(
                        f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_data)} rows)"
                    )

                    # Phase 2: Text Rich Processing (KG Extraction)
                    if not skip_text_rich:
                        logger.info(
                            f"Phase 2 (Batch {batch_num}): Processing text-rich columns..."
                        )
                        text_start = time.perf_counter()

                        text_stats = await self.text_processor.process_columns(
                            analysis_results=column_analyses,
                            csv_data=batch_data,
                            uniprot_column=uniprot_column,
                            service=self.service,
                            **kwargs,
                        )

                        text_time = time.perf_counter() - text_start
                        logger.info(
                            f"✅ Batch {batch_num} text processing complete in {text_time:.2f}s"
                        )

                        # Merge stats
                        combined_stats.kg_extractions_performed += (
                            text_stats.kg_extractions_performed
                        )
                        combined_stats.individual_nodes_created += (
                            text_stats.individual_nodes_created
                        )
                        combined_stats.relationships_created += (
                            text_stats.relationships_created
                        )
                        if text_stats.errors:
                            if combined_stats.errors is None:
                                combined_stats.errors = []
                            combined_stats.errors.extend(text_stats.errors)

                        # Accumulate phase results
                        if "text_rich" not in phase_results:
                            phase_results["text_rich"] = {
                                "processing_time": 0.0,
                                "stats": EnrichmentStats(),
                            }
                        phase_results["text_rich"]["processing_time"] += text_time

                    # Phase 3: Node Annotation (Properties and Classes)
                    if not skip_node_annotation:
                        logger.info(
                            f"Phase 3 (Batch {batch_num}): Processing node annotation..."
                        )
                        annotation_start = time.perf_counter()

                        annotation_stats = await self.node_annotator.process_columns(
                            analysis_results=column_analyses,
                            csv_data=batch_data,
                            uniprot_column=uniprot_column,
                            **kwargs,
                        )

                        annotation_time = time.perf_counter() - annotation_start
                        logger.info(
                            f"✅ Batch {batch_num} node annotation complete in {annotation_time:.2f}s"
                        )

                        # Merge stats
                        combined_stats.proteins_processed += (
                            annotation_stats.proteins_processed
                        )
                        combined_stats.properties_added += (
                            annotation_stats.properties_added
                        )
                        combined_stats.node_classes_created += (
                            annotation_stats.node_classes_created
                        )
                        combined_stats.individual_nodes_created += (
                            annotation_stats.individual_nodes_created
                        )
                        combined_stats.relationships_created += (
                            annotation_stats.relationships_created
                        )
                        if annotation_stats.errors:
                            if combined_stats.errors is None:
                                combined_stats.errors = []
                            combined_stats.errors.extend(annotation_stats.errors)

                        # Accumulate phase results
                        if "node_annotation" not in phase_results:
                            phase_results["node_annotation"] = {
                                "processing_time": 0.0,
                                "stats": EnrichmentStats(),
                            }
                        phase_results["node_annotation"]["processing_time"] += (
                            annotation_time
                        )

                    # Update progress file
                    current_progress = offset + i + len(batch_data)
                    try:
                        with open(progress_file, "w") as f:
                            f.write(str(current_progress))
                        logger.info(
                            f"💾 Progress saved: {current_progress} rows processed"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save progress: {e}")

            # Clean up progress file on successful completion
            if progress_file.exists():
                try:
                    progress_file.unlink()
                    logger.info("✨ Processing complete - removed progress file")
                except Exception as e:
                    logger.warning(f"Failed to remove progress file: {e}")

            total_time = (datetime.now() - start_time).total_seconds()

            # Build final results
            results = {
                "success": True,
                "file_path": file_path,
                "total_processing_time": total_time,
                "analysis_time": analysis_time,
                "column_analyses": {
                    k: self._analysis_to_dict(v) for k, v in column_analyses.items()
                },
                "metadata": metadata,
                "combined_stats": self._stats_to_dict(combined_stats),
                "phase_results": phase_results,
                "summary": self._create_summary(
                    combined_stats, total_time, len(column_analyses)
                ),
            }

            # Log final summary
            logger.info("CSV enrichment complete!")
            logger.info(f"  • Total time: {total_time:.2f}s")
            logger.info(f"  • Columns analyzed: {len(column_analyses)}")
            logger.info(f"  • Proteins processed: {combined_stats.proteins_processed}")
            logger.info(f"  • Properties added: {combined_stats.properties_added}")
            logger.info(
                f"  • Node classes created: {combined_stats.node_classes_created}"
            )
            logger.info(
                f"  • Individual nodes: {combined_stats.individual_nodes_created}"
            )
            logger.info(f"  • Relationships: {combined_stats.relationships_created}")
            logger.info(
                f"  • KG extractions: {combined_stats.kg_extractions_performed}"
            )

            return results

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"CSV enrichment failed after {total_time:.2f}s: {e}")

            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "total_processing_time": total_time,
            }

    def _get_csv_data(self, file_path: str) -> list[dict[str, Any]]:
        """Read CSV data for processing."""
        # This reuses the column analyzer's CSV reading logic
        data, _ = self.column_analyzer._read_csv(file_path)
        return data

    async def _get_csv_data_for_resume(
        self,
        file_path: str,
        resume: bool,
        offset: int,
        limit: int | None,
        batch_size: int,
        metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get CSV data with resume/batch processing support.

        Args:
            file_path: Path to CSV file
            resume: Whether to skip already processed proteins
            offset: Start from row N
            limit: Maximum rows to process
            batch_size: Batch size for processing
            metadata: CSV metadata including uniprot column info

        Returns:
            Tuple of (csv_data, skip_info)
        """
        # Read full CSV data
        full_data, _ = self.column_analyzer._read_csv(file_path)
        total_rows = len(full_data)

        # Apply offset
        if offset > 0:
            if offset >= total_rows:
                logger.warning(f"Offset {offset} exceeds total rows {total_rows}")
                return [], {
                    "processed_message": f"Offset {offset} exceeds data size",
                    "skipped_rows": total_rows,
                    "already_processed": 0,
                }
            full_data = full_data[offset:]

        # Apply limit
        if limit:
            full_data = full_data[:limit]

        skip_info = {
            "processed_message": f"Processing {len(full_data)} rows from total {total_rows}",
            "skipped_rows": offset,
            "already_processed": 0,
        }

        # If resume is enabled, filter out already processed proteins
        if resume:
            processed_data, already_processed = await self._filter_processed_proteins(
                full_data, metadata
            )
            skip_info["already_processed"] = already_processed
            skip_info["processed_message"] = (
                f"Processing {len(processed_data)} new rows ({already_processed} already processed)"
            )
            return processed_data, skip_info

        return full_data, skip_info

    async def _filter_processed_proteins(
        self, csv_data: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Filter out proteins that have already been processed.

        Args:
            csv_data: Full CSV data
            metadata: CSV metadata with uniprot column info

        Returns:
            Tuple of (filtered_data, count_already_processed)
        """
        try:
            # Use the injected graph store
            # Determine which column to use for protein identification
            uniprot_column = metadata.get("uniprot_column")

            # Create a set of existing protein identifiers
            existing_proteins = set()

            if uniprot_column:
                # Check by UniProt ID
                query = "MATCH (p:Protein) WHERE p.uniprot IS NOT NULL RETURN p.uniprot as id"

                results = self.db.execute_query(query)
                existing_proteins.update(
                    result["id"] for result in results if result["id"]
                )

                logger.info(
                    f"Found {len(existing_proteins)} existing proteins with UniProt IDs"
                )

                # Filter CSV data
                filtered_data = []
                already_processed = 0

                for row in csv_data:
                    uniprot_id = row.get(uniprot_column)
                    if uniprot_id and uniprot_id in existing_proteins:
                        already_processed += 1
                    else:
                        filtered_data.append(row)

                return filtered_data, already_processed

            # Fallback: check by gene name or name
            name_columns = ["gene_name", "protein_name", "name", "uniprot_gene_name"]
            name_column = None

            for col in name_columns:
                if col in csv_data[0]:
                    name_column = col
                    break

            if name_column:
                query = f"MATCH (p:Protein) WHERE p.{name_column} IS NOT NULL RETURN p.{name_column} as name"

                results = self.db.execute_query(query)
                existing_proteins.update(
                    result["name"].lower() for result in results if result["name"]
                )

                logger.info(
                    f"Found {len(existing_proteins)} existing proteins by {name_column}"
                )

                # Filter CSV data (case-insensitive matching)
                filtered_data = []
                already_processed = 0

                for row in csv_data:
                    name = row.get(name_column)
                    if name and name.lower() in existing_proteins:
                        already_processed += 1
                    else:
                        filtered_data.append(row)

                return filtered_data, already_processed
            logger.warning(
                "No suitable column found for resume checking - processing all rows"
            )
            return csv_data, 0

        except Exception as e:
            logger.error(f"Error filtering processed proteins: {e}")
            logger.info("Proceeding with all rows due to filter error")
            return csv_data, 0

    def _log_analysis_summary(self, column_analyses: dict[str, Any]) -> None:
        """Log a summary of column analysis results."""

        strategy_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for analysis in column_analyses.values():
            strategy = analysis.strategy.value
            col_type = analysis.column_type.value

            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            type_counts[col_type] = type_counts.get(col_type, 0) + 1

        logger.info("Column Analysis Summary:")
        logger.info("  Strategies:")
        for strategy, count in sorted(strategy_counts.items()):
            logger.info(f"    • {strategy}: {count}")

        logger.info("  Types:")
        for col_type, count in sorted(type_counts.items()):
            logger.info(f"    • {col_type}: {count}")

    def _analysis_to_dict(self, analysis) -> dict[str, Any]:
        """Convert ColumnAnalysis to dictionary."""
        return {
            "column_type": analysis.column_type.value,
            "strategy": analysis.strategy.value,
            "unique_count": analysis.unique_count,
            "total_values": analysis.total_values,
            "empty_percentage": analysis.empty_percentage,
            "na_percentage": analysis.na_percentage,
            "total_missing_percentage": analysis.total_missing_percentage,
            "has_multiple_values": analysis.has_multiple_values,
            "expanded_unique_count": analysis.expanded_unique_count,
            "avg_text_length": analysis.avg_text_length,
            "sample_values": analysis.sample_values[:5]
            if analysis.sample_values
            else None,
            "partner_column": analysis.partner_column,
            "reason": analysis.reason,
        }

    def _stats_to_dict(self, stats: EnrichmentStats) -> dict[str, Any]:
        """Convert EnrichmentStats to dictionary."""
        return {
            "proteins_processed": stats.proteins_processed,
            "properties_added": stats.properties_added,
            "node_classes_created": stats.node_classes_created,
            "individual_nodes_created": stats.individual_nodes_created,
            "relationships_created": stats.relationships_created,
            "kg_extractions_performed": stats.kg_extractions_performed,
            "errors": stats.errors,
        }

    def _create_summary(
        self, stats: EnrichmentStats, total_time: float, column_count: int
    ) -> str:
        """Create a human-readable summary of the enrichment results."""

        summary_lines = [
            f"Processed {column_count} columns in {total_time:.1f} seconds",
            f"Enhanced {stats.proteins_processed} proteins with {stats.properties_added} properties",
            f"Created {stats.node_classes_created} node classes with {stats.individual_nodes_created} instances",
            f"Generated {stats.relationships_created} relationships and {stats.kg_extractions_performed} KG extractions",
        ]

        if stats.errors:
            summary_lines.append(
                f"Encountered {len(stats.errors)} errors during processing"
            )

        return " | ".join(summary_lines)

    async def analyze_only(
        self, file_path: str, manual_column_handlers: dict[int, str] | None = None
    ) -> dict[str, Any]:
        """
        Perform only column analysis without any enrichment.

        Args:
            file_path: Path to CSV file to analyze
            manual_column_handlers: Optional manual column handlers

        Returns:
            Dictionary with analysis results only
        """
        logger.info(f"Analyzing CSV structure: {file_path}")

        try:
            # Set manual column handlers if provided
            if manual_column_handlers:
                parsed_handlers = {}
                for col_idx, handler_str in manual_column_handlers.items():
                    try:
                        parsed_handlers[col_idx] = (
                            self.column_analyzer.parse_handler_string(handler_str)
                        )
                    except ValueError as e:
                        logger.error(
                            f"Invalid handler '{handler_str}' for column {col_idx}: {e}"
                        )
                        return {
                            "success": False,
                            "error": f"Invalid column handler: {e}",
                            "file_path": file_path,
                        }

                self.column_analyzer.set_manual_handlers(parsed_handlers)
                logger.info(
                    f"Applied {len(parsed_handlers)} manual column handlers for analysis"
                )

            column_analyses, metadata = await self.column_analyzer.analyze_csv(
                file_path
            )

            self._log_analysis_summary(column_analyses)

            return {
                "success": True,
                "file_path": file_path,
                "column_analyses": {
                    k: self._analysis_to_dict(v) for k, v in column_analyses.items()
                },
                "metadata": metadata,
                "summary": f"Analyzed {len(column_analyses)} columns",
            }

        except Exception as e:
            logger.error(f"CSV analysis failed: {e}")
            return {"success": False, "error": str(e), "file_path": file_path}
