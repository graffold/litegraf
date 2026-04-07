"""Preprint version tracker for managing version history and publication links."""

import logging
from datetime import datetime

from src.core.database import Neo4jDatabase
from pipeline.ingest.preprint_fetcher import PreprintMetadata

logger = logging.getLogger(__name__)


class PreprintVersionTracker:
    """Track preprint versions and link to published papers."""

    def __init__(self, db: Neo4jDatabase):
        """Initialize version tracker.

        Args:
            db: Neo4j database connection
        """
        self.db = db

    def store_preprint_version(self, metadata: PreprintMetadata) -> None:
        """Store a preprint version in the graph.

        Creates PreprintVersion node and links to Preprint node.
        If this is the first version, creates the Preprint node.
        """
        query = """
        // Create or update Preprint node (represents all versions)
        MERGE (p:Preprint {doi: $doi})
        SET p.title = $title,
            p.server = $server,
            p.latest_version = $version,
            p.updated_at = $updated_at

        // Create PreprintVersion node for this specific version
        MERGE (v:PreprintVersion {doi: $doi, version: $version})
        SET v.title = $title,
            v.authors = $authors,
            v.posting_date = $posting_date,
            v.content_url = $content_url,
            v.abstract = $abstract,
            v.subject_area = $subject_area,
            v.server = $server,
            v.ingested_at = $ingested_at

        // Link version to preprint
        MERGE (p)-[:HAS_VERSION]->(v)
        """

        params = {
            "doi": metadata.doi,
            "version": metadata.version,
            "title": metadata.title,
            "authors": metadata.authors,
            "posting_date": metadata.posting_date,
            "content_url": metadata.content_url,
            "abstract": metadata.abstract,
            "subject_area": metadata.subject_area,
            "server": metadata.server,
            "updated_at": datetime.utcnow().isoformat(),
            "ingested_at": datetime.utcnow().isoformat(),
        }

        try:
            self.db._execute_cypher(query, params)
            logger.info(
                f"Stored preprint version {metadata.version} for DOI {metadata.doi}"
            )
        except Exception as e:
            logger.error(f"Failed to store preprint version: {e}")
            raise

    def link_to_publication(
        self,
        preprint_doi: str,
        pmid: str | None = None,
        published_doi: str | None = None,
    ) -> None:
        """Link preprint to published paper.

        Creates PUBLISHED_AS relationship from Preprint to Abstract (PMID) or published DOI.
        """
        if not pmid and not published_doi:
            logger.warning(
                f"No PMID or published DOI provided for preprint {preprint_doi}"
            )
            return

        if pmid:
            query = """
            MATCH (p:Preprint {doi: $preprint_doi})
            MATCH (a:Abstract {pmid: $pmid})
            MERGE (p)-[r:PUBLISHED_AS]->(a)
            SET r.linked_at = $linked_at,
                r.pmid = $pmid
            SET p.publication_status = "published",
                p.published_pmid = $pmid
            """
            params = {
                "preprint_doi": preprint_doi,
                "pmid": pmid,
                "linked_at": datetime.utcnow().isoformat(),
            }
        else:
            query = """
            MATCH (p:Preprint {doi: $preprint_doi})
            MERGE (p)-[r:PUBLISHED_AS_DOI {doi: $published_doi}]->(p)
            SET r.linked_at = $linked_at
            SET p.publication_status = "published",
                p.published_doi = $published_doi
            """
            params = {
                "preprint_doi": preprint_doi,
                "published_doi": published_doi,
                "linked_at": datetime.utcnow().isoformat(),
            }

        try:
            self.db._execute_cypher(query, params)
            logger.info(
                f"Linked preprint {preprint_doi} to publication (PMID: {pmid}, DOI: {published_doi})"
            )
        except Exception as e:
            logger.error(f"Failed to link preprint to publication: {e}")
            raise

    def store_version_history(self, versions: list[PreprintMetadata]) -> None:
        """Store complete version history for a preprint.

        Args:
            versions: List of PreprintMetadata sorted by version (oldest to newest)
        """
        if not versions:
            logger.warning("No versions provided")
            return

        # Store each version
        for version in versions:
            self.store_preprint_version(version)

        # Link to publication if available
        latest = versions[-1]
        if latest.published_pmid or latest.published_doi:
            self.link_to_publication(
                preprint_doi=latest.doi,
                pmid=latest.published_pmid,
                published_doi=latest.published_doi,
            )

    def get_version_history(self, doi: str) -> list[dict]:
        """Get version history for a preprint.

        Returns list of version metadata sorted by version number.
        """
        query = """
        MATCH (p:Preprint {doi: $doi})-[:HAS_VERSION]->(v:PreprintVersion)
        RETURN v.version AS version,
               v.title AS title,
               v.posting_date AS posting_date,
               v.content_url AS content_url,
               v.ingested_at AS ingested_at
        ORDER BY toInteger(v.version)
        """

        try:
            result = self.db._execute_cypher(query, {"doi": doi})
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to get version history for {doi}: {e}")
            return []

    def get_publication_status(self, doi: str) -> dict:
        """Get publication status for a preprint.

        Returns dict with publication_status, published_pmid, published_doi.
        """
        query = """
        MATCH (p:Preprint {doi: $doi})
        OPTIONAL MATCH (p)-[:PUBLISHED_AS]->(a:Abstract)
        RETURN p.publication_status AS status,
               p.published_pmid AS pmid,
               p.published_doi AS doi,
               a.pmid AS linked_pmid
        """

        try:
            result = self.db._execute_cypher(query, {"doi": doi})
            record = result.single()
            if record:
                return {
                    "publication_status": record.get("status", "preprint"),
                    "published_pmid": record.get("pmid") or record.get("linked_pmid"),
                    "published_doi": record.get("doi"),
                }
            return {
                "publication_status": "preprint",
                "published_pmid": None,
                "published_doi": None,
            }
        except Exception as e:
            logger.error(f"Failed to get publication status for {doi}: {e}")
            return {
                "publication_status": "unknown",
                "published_pmid": None,
                "published_doi": None,
            }
