#!/usr/bin/env python3
"""
DisGeNET Database Integrator

Ingests gene-disease associations from DisGeNET TSV files into the knowledge graph.
DisGeNET provides curated gene-disease associations with evidence scores.

File format: TSV with columns including geneId, diseaseId, score, source, pmids
"""

import logging
import csv
from typing import Any

from pipeline.interfaces import GraphStore
logger = logging.getLogger(__name__)
class DisGeNETIntegrator:
    """Integrates DisGeNET gene-disease associations into the knowledge graph."""

    def __init__(
        self,
        backend: str = "neo4j",
        database: str = "olink1",
        db: GraphStore | None = None,
    ):
        """
        Initialize DisGeNET integrator.

        Args:
            backend: Database backend ("neo4j" or "neptune")
            database: Database name
            db: Optional GraphStore instance
        """
        self.backend = backend
        self.database = database

        if backend == "neptune":
            from archive.neptune_legacy.core.opencypher_query_executor import (
                create_opencypher_executor,
            )

            self.executor = create_opencypher_executor(database_type="neptune")
            self.db = None
        else:
            if db is not None:
                self.db = db
            else:
                from pipeline.backends.neo4j_store import Neo4jGraphStore
                self.db = Neo4jGraphStore(database=database)
            self.executor = None

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query using appropriate backend."""
        if self.backend == "neptune":
            return self.executor.execute_query(query, parameters)
        return self.db.execute_query(query, parameters)

    def parse_disgenet_file(
        self, file_path: str, score_threshold: float = 0.06
    ) -> dict[str, Any]:
        """
        Parse DisGeNET TSV file.

        Args:
            file_path: Path to DisGeNET TSV file
            score_threshold: Minimum score threshold (default 0.06 for curated sources)

        Returns:
            Dictionary with parsed associations and statistics
        """
        logger.info(f"Parsing DisGeNET file: {file_path}")

        associations = []
        total_rows = 0
        filtered_rows = 0

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                total_rows += 1

                # Extract and validate score
                try:
                    score = float(row.get("score", 0))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid score in row {total_rows}: {row.get('score')}"
                    )
                    continue

                # Filter by score threshold
                if score < score_threshold:
                    filtered_rows += 1
                    continue

                # Extract fields
                gene_id = row.get("geneId", "").strip()
                disease_id = row.get("diseaseId", "").strip()
                pmids = row.get("pmids", "").strip()
                source = row.get("source", "").strip()

                if not gene_id or not disease_id:
                    logger.warning(f"Missing gene or disease ID in row {total_rows}")
                    continue

                associations.append(
                    {
                        "gene_id": gene_id,
                        "disease_id": disease_id,
                        "score": score,
                        "pmids": pmids.split("|") if pmids else [],
                        "source": source,
                    }
                )

        logger.info(
            f"Parsed {len(associations)} associations (filtered {filtered_rows} below threshold {score_threshold})"
        )

        return {
            "associations": associations,
            "stats": {
                "total_rows": total_rows,
                "filtered_rows": filtered_rows,
                "associations_count": len(associations),
                "score_threshold": score_threshold,
            },
        }

    def ingest_associations(self, associations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Ingest gene-disease associations into the knowledge graph.

        Args:
            associations: List of association dictionaries

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Ingesting {len(associations)} DisGeNET associations")

        genes_created = 0
        diseases_created = 0
        relationships_created = 0

        for assoc in associations:
            # Create or merge gene node
            gene_query = """
            MERGE (g:Gene {entrez_id: $gene_id})
            ON CREATE SET g.source = 'DisGeNET', g.created_at = datetime()
            RETURN g
            """
            result = self._execute_query(gene_query, {"gene_id": assoc["gene_id"]})
            if result:
                genes_created += 1

            # Create or merge disease node
            disease_query = """
            MERGE (d:Disease {umls_id: $disease_id})
            ON CREATE SET d.source = 'DisGeNET', d.created_at = datetime()
            RETURN d
            """
            result = self._execute_query(
                disease_query, {"disease_id": assoc["disease_id"]}
            )
            if result:
                diseases_created += 1

            # Create relationship with provenance
            rel_query = """
            MATCH (g:Gene {entrez_id: $gene_id})
            MATCH (d:Disease {umls_id: $disease_id})
            MERGE (g)-[r:ASSOCIATES_WITH_DISEASE]->(d)
            ON CREATE SET
                r.disgenet_score = $score,
                r.pmids = $pmids,
                r.source = $source,
                r.data_source = 'DisGeNET',
                r.created_at = datetime()
            ON MATCH SET
                r.disgenet_score = $score,
                r.pmids = $pmids,
                r.source = $source
            RETURN r
            """
            result = self._execute_query(
                rel_query,
                {
                    "gene_id": assoc["gene_id"],
                    "disease_id": assoc["disease_id"],
                    "score": assoc["score"],
                    "pmids": assoc["pmids"],
                    "source": assoc["source"],
                },
            )
            if result:
                relationships_created += 1

        logger.info(
            f"Ingestion complete: {genes_created} genes, {diseases_created} diseases, {relationships_created} relationships"
        )

        return {
            "genes_created": genes_created,
            "diseases_created": diseases_created,
            "relationships_created": relationships_created,
        }

    def ingest_from_file(
        self, file_path: str, score_threshold: float = 0.06
    ) -> dict[str, Any]:
        """
        Complete ingestion workflow from file.

        Args:
            file_path: Path to DisGeNET TSV file
            score_threshold: Minimum score threshold

        Returns:
            Dictionary with complete statistics
        """
        # Parse file
        parsed_data = self.parse_disgenet_file(file_path, score_threshold)

        # Ingest associations
        ingestion_stats = self.ingest_associations(parsed_data["associations"])

        return {
            "parsing_stats": parsed_data["stats"],
            "ingestion_stats": ingestion_stats,
        }


def main():
    """CLI entry point for DisGeNET integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest DisGeNET gene-disease associations"
    )
    parser.add_argument("--file", required=True, help="Path to DisGeNET TSV file")
    parser.add_argument("--database", default="olink1", help="Database name")
    parser.add_argument(
        "--backend",
        default="neo4j",
        choices=["neo4j", "neptune"],
        help="Database backend",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.06, help="Minimum score threshold"
    )

    args = parser.parse_args()

    integrator = DisGeNETIntegrator(backend=args.backend, database=args.database)
    stats = integrator.ingest_from_file(args.file, args.score_threshold)

    logger.info(f"DisGeNET integration complete: {stats}")


if __name__ == "__main__":
    main()
