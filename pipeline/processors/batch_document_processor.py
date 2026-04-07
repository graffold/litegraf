"""
Batch Document Processor
Handles parallel processing of multiple PDF documents with checkpoint/resume capability.
Integrates with multimodal processors for vision and table extraction.
"""

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import Config
from pipeline.ingest.ingestor import ProcessedDocument
from pipeline.processors.multimodal_processor import MultimodalProcessor
from pipeline.processors.table_processor import TableProcessor
from pipeline.processors.vision_processor import VisionProcessor
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class ProcessingState(Enum):
    """State of document processing."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentProcessingResult:
    """Result of processing a single document."""

    file_path: str
    doc_id: str
    state: ProcessingState
    processed_doc: ProcessedDocument | None = None
    image_count: int = 0
    table_count: int = 0
    entity_count: int = 0
    error_message: str | None = None
    processing_time_seconds: float = 0.0


class BatchDocumentProcessor:
    """
    Processes multiple documents in parallel with checkpoint/resume capability.
    """

    def __init__(
        self,
        checkpoint_db_path: str = "./checkpoints/batch_processing.db",
        max_workers: int = 4,
        enable_vision: bool = True,
        enable_tables: bool = True,
        vision_model_id: str | None = None,
        llm_service: str = "bedrock",
    ):
        """
        Initialize batch processor.

        Args:
            checkpoint_db_path: Path to SQLite checkpoint database
            max_workers: Maximum parallel workers
            enable_vision: Enable vision processing for images
            enable_tables: Enable table extraction
            vision_model_id: Vision model ID (defaults to config)
            llm_service: LLM service for table processing
        """
        self.checkpoint_db_path = checkpoint_db_path
        self.max_workers = max_workers
        self.enable_vision = enable_vision
        self.enable_tables = enable_tables

        # Create checkpoint directory
        Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint database
        self._init_checkpoint_db()

        # Initialize processors
        self.multimodal_processor = MultimodalProcessor()

        if enable_vision:
            vision_model = (
                vision_model_id
                or Config.get_config("VISION_MODEL_ID")
            )
            self.vision_processor = VisionProcessor(vision_model_id=vision_model)
        else:
            self.vision_processor = None

        if enable_tables:
            self.table_processor = TableProcessor(llm_service=llm_service)
        else:
            self.table_processor = None

        logger.info(
            f"Initialized BatchDocumentProcessor "
            f"(workers={max_workers}, vision={enable_vision}, tables={enable_tables})"
        )

    def _init_checkpoint_db(self):
        """Initialize SQLite checkpoint database."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_state (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                doc_id TEXT,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                image_count INTEGER DEFAULT 0,
                table_count INTEGER DEFAULT 0,
                entity_count INTEGER DEFAULT 0,
                error_message TEXT,
                processing_time_seconds REAL DEFAULT 0.0
            )
        """)

        conn.commit()
        conn.close()

        logger.info(f"Initialized checkpoint database at {self.checkpoint_db_path}")

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_processing_state(self, file_path: str) -> ProcessingState | None:
        """Get current processing state for a file."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT state, file_hash FROM processing_state WHERE file_path = ?",
            (file_path,),
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            stored_state, stored_hash = result
            current_hash = self._get_file_hash(file_path)

            # If file changed, reset state
            if stored_hash != current_hash:
                logger.info(f"File {file_path} modified, resetting state")
                return None

            return ProcessingState(stored_state)

        return None

    def _update_processing_state(
        self,
        file_path: str,
        state: ProcessingState,
        doc_id: str | None = None,
        image_count: int = 0,
        table_count: int = 0,
        entity_count: int = 0,
        error_message: str | None = None,
        processing_time: float = 0.0,
    ):
        """Update processing state in checkpoint database."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        file_hash = self._get_file_hash(file_path)
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO processing_state
            (file_path, file_hash, doc_id, state, created_at, updated_at,
             image_count, table_count, entity_count, error_message, processing_time_seconds)
            VALUES (?, ?, ?, ?,
                    COALESCE((SELECT created_at FROM processing_state WHERE file_path = ?), ?),
                    ?, ?, ?, ?, ?, ?)
        """,
            (
                file_path,
                file_hash,
                doc_id,
                state.value,
                file_path,
                now,
                now,
                image_count,
                table_count,
                entity_count,
                error_message,
                processing_time,
            ),
        )

        conn.commit()
        conn.close()

    def process_folder(
        self,
        folder_path: str,
        file_extensions: list[str] = None,
        recursive: bool = True,
        skip_completed: bool = True,
    ) -> list[DocumentProcessingResult]:
        """
        Process all documents in a folder.

        Args:
            folder_path: Path to folder containing documents
            file_extensions: List of file extensions to process
            recursive: Search recursively in subdirectories
            skip_completed: Skip files that were successfully processed

        Returns:
            List of processing results
        """
        if file_extensions is None:
            file_extensions = [".pdf"]
        folder = Path(folder_path)

        # Find all matching files
        files = []
        for ext in file_extensions:
            if recursive:
                files.extend(folder.rglob(f"*{ext}"))
            else:
                files.extend(folder.glob(f"*{ext}"))

        file_paths = [str(f) for f in files]

        logger.info(f"Found {len(file_paths)} documents in {folder_path}")

        # Filter based on checkpoint state
        if skip_completed:
            files_to_process = []
            for file_path in file_paths:
                state = self._get_processing_state(file_path)
                if state != ProcessingState.COMPLETED:
                    files_to_process.append(file_path)
                else:
                    logger.debug(f"Skipping completed file: {file_path}")

            logger.info(
                f"After checkpoint filtering: {len(files_to_process)} files to process "
                f"({len(file_paths) - len(files_to_process)} skipped)"
            )
            file_paths = files_to_process

        # Process in parallel
        return self.process_files(file_paths)

    def process_files(self, file_paths: list[str]) -> list[DocumentProcessingResult]:
        """
        Process multiple files in parallel with retry logic.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of processing results
        """
        results = []

        logger.info(f"Starting batch processing of {len(file_paths)} files")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_file, path): path
                for path in file_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)

                    logger.info(
                        f"Completed {file_path}: "
                        f"{result.state.value} "
                        f"({result.image_count} images, {result.table_count} tables) "
                        f"in {result.processing_time_seconds:.2f}s"
                    )
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                    results.append(
                        DocumentProcessingResult(
                            file_path=file_path,
                            doc_id="",
                            state=ProcessingState.FAILED,
                            error_message=str(e),
                        )
                    )

        # Summary
        completed = sum(1 for r in results if r.state == ProcessingState.COMPLETED)
        failed = sum(1 for r in results if r.state == ProcessingState.FAILED)

        logger.info(
            f"Batch processing complete: "
            f"{completed} succeeded, {failed} failed out of {len(results)} total"
        )

        return results

    def _process_single_file(self, file_path: str) -> DocumentProcessingResult:
        """
        Process a single document file.

        Args:
            file_path: Path to document

        Returns:
            Processing result
        """
        start_time = datetime.now()

        try:
            # Update state to in_progress
            self._update_processing_state(file_path, ProcessingState.IN_PROGRESS)

            logger.info(f"Processing document: {file_path}")

            # Step 1: Extract multimodal content
            multimodal_doc = self.multimodal_processor.process_pdf(file_path)

            image_analyses = []
            table_analyses = []
            total_entities = 0

            # Step 2: Process images with vision model
            if self.enable_vision and self.vision_processor and multimodal_doc.images:
                max_images = int(Config.get_config("MAX_IMAGES_PER_DOCUMENT") or 20)
                logger.info(
                    f"Analyzing {min(len(multimodal_doc.images), max_images)} images"
                )

                image_analyses = self.vision_processor.process_document_images(
                    multimodal_doc, max_images=max_images
                )

                # Count extracted entities from images
                for analysis in image_analyses:
                    if "analysis" in analysis and "entities" in analysis["analysis"]:
                        total_entities += len(analysis["analysis"]["entities"])

            # Step 3: Process tables
            if self.enable_tables and self.table_processor and multimodal_doc.tables:
                max_tables = int(Config.get_config("MAX_TABLES_PER_DOCUMENT") or 10)
                logger.info(
                    f"Analyzing {min(len(multimodal_doc.tables), max_tables)} tables"
                )

                table_analyses = self.table_processor.process_document_tables(
                    multimodal_doc, max_tables=max_tables
                )

                # Count extracted entities from tables
                for analysis in table_analyses:
                    if "entities" in analysis:
                        total_entities += len(analysis["entities"])

            # Step 4: Convert to ProcessedDocument
            processed_doc = self.multimodal_processor.to_processed_document(
                multimodal_doc
            )

            # Enhance with multimodal analyses
            processed_doc.metadata["image_analyses"] = image_analyses
            processed_doc.metadata["table_analyses"] = table_analyses
            processed_doc.metadata["total_extracted_entities"] = total_entities

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Update checkpoint
            self._update_processing_state(
                file_path,
                ProcessingState.COMPLETED,
                doc_id=multimodal_doc.doc_id,
                image_count=len(multimodal_doc.images),
                table_count=len(multimodal_doc.tables),
                entity_count=total_entities,
                processing_time=processing_time,
            )

            return DocumentProcessingResult(
                file_path=file_path,
                doc_id=multimodal_doc.doc_id,
                state=ProcessingState.COMPLETED,
                processed_doc=processed_doc,
                image_count=len(multimodal_doc.images),
                table_count=len(multimodal_doc.tables),
                entity_count=total_entities,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)

            processing_time = (datetime.now() - start_time).total_seconds()

            self._update_processing_state(
                file_path,
                ProcessingState.FAILED,
                error_message=str(e),
                processing_time=processing_time,
            )

            return DocumentProcessingResult(
                file_path=file_path,
                doc_id="",
                state=ProcessingState.FAILED,
                error_message=str(e),
                processing_time_seconds=processing_time,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics from checkpoint database."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        stats = {}

        # Count by state
        cursor.execute("""
            SELECT state, COUNT(*),
                   SUM(image_count),
                   SUM(table_count),
                   SUM(entity_count),
                   AVG(processing_time_seconds)
            FROM processing_state
            GROUP BY state
        """)

        for row in cursor.fetchall():
            state, count, images, tables, entities, avg_time = row
            stats[state] = {
                "count": count,
                "total_images": images or 0,
                "total_tables": tables or 0,
                "total_entities": entities or 0,
                "avg_processing_time": avg_time or 0.0,
            }

        conn.close()

        return stats

    def reset_failed(self):
        """Reset all failed documents to pending for retry."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE processing_state
            SET state = ?, updated_at = ?
            WHERE state = ?
        """,
            (
                ProcessingState.PENDING.value,
                datetime.now().isoformat(),
                ProcessingState.FAILED.value,
            ),
        )

        rows_updated = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Reset {rows_updated} failed documents to pending")
        return rows_updated
