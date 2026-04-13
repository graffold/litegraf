"""Deduplication logic for bioRxiv papers against existing database content."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from pipeline.ingest.biorxiv_fetcher import BioRxivPaperMetadata
logger = logging.getLogger(__name__)
def _normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip whitespace and punctuation."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


class BioRxivDeduplicator:
    """Handles deduplication between bioRxiv papers and existing database content."""

    def __init__(self, db: Any, backend: str = "neo4j"):
        self.db = db
        self.backend = backend

    def get_existing_dois(self) -> set[str]:
        """Query the database for DOIs already ingested as BioRxivPaper nodes."""
        try:
            query = "MATCH (p:BioRxivPaper) RETURN p.doi AS doi"
            results = self.db._execute_cypher(query)
            dois = {record["doi"] for record in results if record.get("doi")}
            logger.debug(f"Retrieved {len(dois)} existing bioRxiv DOIs")
            return dois
        except Exception as e:
            logger.error(f"Failed to retrieve existing bioRxiv DOIs: {e}")
            return set()

    def filter_new_papers(
        self, papers: list[BioRxivPaperMetadata], force: bool = False
    ) -> list[BioRxivPaperMetadata]:
        """Filter out papers whose DOIs already exist in the database.

        Args:
            papers: List of paper metadata to filter.
            force: If True, return all papers regardless of existing DOIs.

        Returns:
            List of papers not yet in the database (or all if force=True).
        """
        if force:
            logger.info(f"Force mode enabled, returning all {len(papers)} papers")
            return papers

        existing_dois = self.get_existing_dois()
        if not existing_dois:
            return papers

        new_papers = [p for p in papers if p.doi not in existing_dois]
        skipped = len(papers) - len(new_papers)
        if skipped > 0:
            logger.info(
                f"Filtered out {skipped} papers already in database, "
                f"{len(new_papers)} new papers remaining"
            )
        return new_papers

    def link_published_counterparts(self, dois: list[str]) -> int:
        """Create PUBLISHED_AS relationships between BioRxivPaper and Abstract nodes.

        For each DOI, checks if there's a matching Abstract node by:
        1. Exact title match (normalized, case-insensitive) via Cypher
        2. Fuzzy title matching (similarity > 0.85) via Python difflib

        Args:
            dois: List of DOIs to check for published counterparts.

        Returns:
            Number of PUBLISHED_AS links created.
        """
        links_created = 0

        for doi in dois:
            try:
                links_created += self._link_single_paper(doi)
            except Exception as e:
                logger.error(f"Failed to link published counterpart for DOI {doi}: {e}")

        logger.info(f"Created {links_created} PUBLISHED_AS links for {len(dois)} DOIs")
        return links_created

    def _link_single_paper(self, doi: str) -> int:
        """Attempt to link a single BioRxivPaper to its published Abstract counterpart.

        Returns 1 if a link was created, 0 otherwise.
        """
        # First, get the bioRxiv paper title
        biorxiv_query = "MATCH (b:BioRxivPaper {doi: $doi}) RETURN b.title AS title"
        try:
            biorxiv_results = self.db._execute_cypher(biorxiv_query, {"doi": doi})
        except Exception as e:
            logger.error(f"Failed to query BioRxivPaper for DOI {doi}: {e}")
            return 0

        if not biorxiv_results:
            logger.debug(f"No BioRxivPaper node found for DOI {doi}")
            return 0

        biorxiv_title = biorxiv_results[0].get("title", "")
        if not biorxiv_title:
            return 0

        # Try exact normalized title match via Cypher
        exact_query = (
            "MATCH (b:BioRxivPaper {doi: $doi}), (a:Abstract) "
            "WHERE toLower(a.title) = toLower(b.title) "
            "MERGE (b)-[:PUBLISHED_AS]->(a) "
            "RETURN count(*) AS cnt"
        )
        try:
            exact_results = self.db._execute_cypher(exact_query, {"doi": doi})
            if exact_results and exact_results[0].get("cnt", 0) > 0:
                logger.info(f"Exact title match found for DOI {doi}")
                return 1
        except Exception as e:
            logger.warning(f"Exact title match query failed for DOI {doi}: {e}")

        # Fallback: fuzzy title matching via Python
        return self._fuzzy_match_and_link(doi, biorxiv_title)

    def _fuzzy_match_and_link(self, doi: str, biorxiv_title: str) -> int:
        """Fuzzy match a bioRxiv paper title against all Abstract titles.

        Returns 1 if a link was created, 0 otherwise.
        """
        try:
            abstract_query = (
                "MATCH (a:Abstract) WHERE a.title IS NOT NULL "
                "RETURN a.title AS title, elementId(a) AS node_id"
            )
            abstracts = self.db._execute_cypher(abstract_query)
        except Exception as e:
            logger.error(f"Failed to query Abstract titles for fuzzy matching: {e}")
            return 0

        if not abstracts:
            return 0

        normalized_biorxiv = _normalize_title(biorxiv_title)
        best_ratio = 0.0
        best_abstract_title = None

        for record in abstracts:
            abstract_title = record.get("title", "")
            if not abstract_title:
                continue
            normalized_abstract = _normalize_title(abstract_title)
            ratio = SequenceMatcher(
                None, normalized_biorxiv, normalized_abstract
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_abstract_title = abstract_title

        if best_ratio > 0.85 and best_abstract_title is not None:
            merge_query = (
                "MATCH (b:BioRxivPaper {doi: $doi}), (a:Abstract {title: $title}) "
                "MERGE (b)-[:PUBLISHED_AS]->(a) "
                "RETURN count(*) AS cnt"
            )
            try:
                result = self.db._execute_cypher(
                    merge_query, {"doi": doi, "title": best_abstract_title}
                )
                if result and result[0].get("cnt", 0) > 0:
                    logger.info(
                        f"Fuzzy title match for DOI {doi} "
                        f"(similarity: {best_ratio:.2f})"
                    )
                    return 1
            except Exception as e:
                logger.error(f"Failed to create PUBLISHED_AS link for DOI {doi}: {e}")

        return 0
