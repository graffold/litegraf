"""
Customer Publication Processor
Extracts and links customer study results from PDF corpus to proteins and diseases.
"""

import logging
from pathlib import Path
from typing import Any

from pipeline.processors.batch_document_processor import (
    BatchDocumentProcessor,
    ProcessingState,
)
from pipeline.processors.multimodal_processor import MultimodalProcessor
from pipeline.interfaces import GraphStore
logger = logging.getLogger(__name__)
class CustomerPublicationProcessor:
    """
    Processes customer publications (PDFs) and links them to proteins and diseases in the graph.
    """

    def __init__(
        self,
        db: GraphStore,
        checkpoint_db_path: str = "./checkpoints/customer_publications.db",
        max_workers: int = 4,
    ):
        """
        Initialize customer publication processor.

        Args:
            db: Graph store connection
            checkpoint_db_path: Path to SQLite checkpoint database
            max_workers: Maximum parallel workers for PDF processing
        """
        self.db = db
        self.batch_processor = BatchDocumentProcessor(
            checkpoint_db_path=checkpoint_db_path,
            max_workers=max_workers,
            enable_vision=False,  # Not needed for customer publications
            enable_tables=True,  # Extract tables for results data
        )
        self.multimodal_processor = MultimodalProcessor()
        logger.info("Initialized CustomerPublicationProcessor")

    def process_pdf_directory(
        self,
        pdf_dir: Path,
        skip_completed: bool = True,
    ) -> dict[str, Any]:
        """
        Process all PDFs in a directory and link to graph.

        Args:
            pdf_dir: Directory containing customer publication PDFs
            skip_completed: Skip already-processed PDFs

        Returns:
            Summary statistics
        """
        logger.info(f"Processing customer publications from {pdf_dir}")

        # Get all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        # Process PDFs in batch
        results = self.batch_processor.batch_process_documents(
            file_paths=[str(f) for f in pdf_files],
            skip_completed=skip_completed,
        )

        # Link processed documents to graph
        linked_count = 0
        for result in results:
            if result.state == ProcessingState.COMPLETED and result.processed_doc:
                try:
                    self._link_publication_to_graph(
                        result.processed_doc, result.file_path
                    )
                    linked_count += 1
                except Exception as e:
                    logger.error(f"Failed to link {result.file_path}: {e}")

        summary = {
            "total_pdfs": len(pdf_files),
            "processed": sum(
                1 for r in results if r.state == ProcessingState.COMPLETED
            ),
            "failed": sum(1 for r in results if r.state == ProcessingState.FAILED),
            "skipped": sum(1 for r in results if r.state == ProcessingState.SKIPPED),
            "linked_to_graph": linked_count,
        }

        logger.info(f"Processing complete: {summary}")
        return summary

    def _link_publication_to_graph(self, processed_doc: Any, file_path: str) -> None:
        """
        Link processed publication to proteins and diseases in graph.

        Args:
            processed_doc: Processed document with extracted content
            file_path: Original PDF file path
        """
        # Create CustomerPublication node
        query = """
        MERGE (pub:CustomerPublication {doc_id: $doc_id})
        SET pub.title = $title,
            pub.file_path = $file_path,
            pub.text_content = $text_content,
            pub.metadata = $metadata,
            pub.data_source = "CustomerPublication",
            pub.processed_date = datetime()
        RETURN pub
        """

        params = {
            "doc_id": processed_doc.doc_id,
            "title": processed_doc.metadata.get("title", Path(file_path).stem),
            "file_path": file_path,
            "text_content": processed_doc.text_content[:10000],  # Store first 10K chars
            "metadata": processed_doc.metadata,
        }

        self.db.execute_query(query, params)
        logger.info(f"Created CustomerPublication node: {processed_doc.doc_id}")

        # Link to proteins mentioned in text
        self._link_to_proteins(processed_doc.doc_id, processed_doc.text_content)

        # Link to diseases mentioned in text
        self._link_to_diseases(processed_doc.doc_id, processed_doc.text_content)

    def _link_to_proteins(self, doc_id: str, text_content: str) -> None:
        """
        Link publication to proteins mentioned in text.

        Args:
            doc_id: Document ID
            text_content: Full text content
        """
        # Find proteins mentioned in text (case-insensitive matching)
        query = """
        MATCH (pub:CustomerPublication {doc_id: $doc_id})
        MATCH (p:Protein)
        WHERE p.name IS NOT NULL
          AND toLower($text_content) CONTAINS toLower(p.name)
        MERGE (pub)-[r:MENTIONS_PROTEIN]->(p)
        SET r.data_source = "CustomerPublication"
        RETURN count(r) as link_count
        """

        result = self.db.execute_query(
            query, {"doc_id": doc_id, "text_content": text_content}
        )
        link_count = result[0]["link_count"] if result else 0
        logger.info(f"Linked {link_count} proteins to {doc_id}")

    def _link_to_diseases(self, doc_id: str, text_content: str) -> None:
        """
        Link publication to diseases mentioned in text.

        Args:
            doc_id: Document ID
            text_content: Full text content
        """
        # Find diseases mentioned in text (case-insensitive matching)
        query = """
        MATCH (pub:CustomerPublication {doc_id: $doc_id})
        MATCH (d:Disease)
        WHERE d.name IS NOT NULL
          AND toLower($text_content) CONTAINS toLower(d.name)
        MERGE (pub)-[r:MENTIONS_DISEASE]->(d)
        SET r.data_source = "CustomerPublication"
        RETURN count(r) as link_count
        """

        result = self.db.execute_query(
            query, {"doc_id": doc_id, "text_content": text_content}
        )
        link_count = result[0]["link_count"] if result else 0
        logger.info(f"Linked {link_count} diseases to {doc_id}")

    def get_publications_for_protein(self, protein_name: str) -> list[dict[str, Any]]:
        """
        Get all customer publications mentioning a protein.

        Args:
            protein_name: Protein name

        Returns:
            List of publication metadata
        """
        query = """
        MATCH (p:Protein {name: $protein_name})<-[:MENTIONS_PROTEIN]-(pub:CustomerPublication)
        RETURN pub.doc_id as doc_id,
               pub.title as title,
               pub.file_path as file_path,
               pub.metadata as metadata
        ORDER BY pub.processed_date DESC
        """

        return self.db.execute_query(query, {"protein_name": protein_name})

    def get_publications_for_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """
        Get all customer publications mentioning a disease.

        Args:
            disease_name: Disease name

        Returns:
            List of publication metadata
        """
        query = """
        MATCH (d:Disease {name: $disease_name})<-[:MENTIONS_DISEASE]-(pub:CustomerPublication)
        RETURN pub.doc_id as doc_id,
               pub.title as title,
               pub.file_path as file_path,
               pub.metadata as metadata
        ORDER BY pub.processed_date DESC
        """

        return self.db.execute_query(query, {"disease_name": disease_name})
