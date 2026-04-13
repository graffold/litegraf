"""
Retraction Watch API integration for flagging retracted papers.

Fetches retraction notices from Retraction Watch database and flags affected papers in the knowledge graph.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class RetractionNotice:
    """Retraction notice metadata from Retraction Watch."""

    retraction_watch_id: str
    pmid: str | None
    doi: str | None
    title: str
    journal: str
    retraction_date: str | None
    retraction_reason: str | None
    retraction_nature: str | None  # "Retraction", "Expression of Concern", "Correction"
    original_paper_date: str | None
    link: str


class RetractionWatchIntegrator:
    """
    Integrates Retraction Watch database to flag retracted papers.

    Retraction Watch maintains a database of retracted papers with reasons and metadata.
    This integrator fetches retraction notices and flags affected papers in the knowledge graph.
    """

    def __init__(
        self,
        db: Any,
        api_key: str | None = None,
        rate_limit: float = 1.0,
    ):
        """
        Initialize Retraction Watch integrator.

        Args:
            db: Neo4j database instance
            api_key: Retraction Watch API key (optional, for higher rate limits)
            rate_limit: Delay between API requests in seconds (default 1.0)
        """
        self.db = db
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.base_url = "http://api.retractionwatch.com/v1"
        self.last_request_time = 0.0

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch_retraction(
        self,
        session: aiohttp.ClientSession,
        identifier: str,
        identifier_type: str = "pmid",
    ) -> dict[str, Any] | None:
        """
        Fetch retraction notice from Retraction Watch API.

        Args:
            session: aiohttp ClientSession
            identifier: PMID or DOI
            identifier_type: "pmid" or "doi"

        Returns:
            Retraction notice data or None if not found
        """
        await self._enforce_rate_limit()

        # Build API endpoint
        if identifier_type == "pmid":
            url = f"{self.base_url}/retractions/pmid/{identifier}"
        else:
            url = f"{self.base_url}/retractions/doi/{identifier}"

        # Add API key if provided
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with session.get(url, params=params) as response:
                if response.status == 404:
                    # Not retracted - this is success, not error
                    return None
                if response.status == 429:
                    # Rate limit - retry
                    logger.warning(f"Rate limit hit for {identifier}, retrying...")
                    raise aiohttp.ClientError("Rate limit exceeded")
                if response.status != 200:
                    logger.error(f"API error {response.status} for {identifier}")
                    return None

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching {identifier}: {e}")
            raise

    def _parse_retraction(self, data: dict[str, Any]) -> RetractionNotice | None:
        """
        Parse Retraction Watch API response.

        Args:
            data: API response JSON

        Returns:
            RetractionNotice or None if invalid
        """
        try:
            return RetractionNotice(
                retraction_watch_id=data.get("RetractionWatchID", ""),
                pmid=data.get("PMID"),
                doi=data.get("DOI"),
                title=data.get("Title", ""),
                journal=data.get("Journal", ""),
                retraction_date=data.get("RetractionDate"),
                retraction_reason=data.get("Reason"),
                retraction_nature=data.get("RetractionNature", "Retraction"),
                original_paper_date=data.get("OriginalPaperDate"),
                link=data.get("RetractionWatchURL", ""),
            )
        except Exception as e:
            logger.error(f"Error parsing retraction data: {e}")
            return None

    async def fetch_retraction_for_pmid(self, pmid: str) -> RetractionNotice | None:
        """
        Fetch retraction notice for a single PMID.

        Args:
            pmid: PubMed ID

        Returns:
            RetractionNotice or None if not retracted
        """
        async with aiohttp.ClientSession() as session:
            data = await self._fetch_retraction(session, pmid, "pmid")
            if data:
                return self._parse_retraction(data)
            return None

    async def fetch_retraction_for_doi(self, doi: str) -> RetractionNotice | None:
        """
        Fetch retraction notice for a single DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            RetractionNotice or None if not retracted
        """
        async with aiohttp.ClientSession() as session:
            data = await self._fetch_retraction(session, doi, "doi")
            if data:
                return self._parse_retraction(data)
            return None

    async def batch_fetch_retractions(
        self,
        pmids: list[str] | None = None,
        dois: list[str] | None = None,
    ) -> list[RetractionNotice]:
        """
        Batch fetch retraction notices for multiple publications.

        Args:
            pmids: List of PMIDs to check
            dois: List of DOIs to check

        Returns:
            List of RetractionNotice objects (only retracted papers)
        """
        pmids = pmids or []
        dois = dois or []

        retractions = []

        async with aiohttp.ClientSession() as session:
            # Fetch PMIDs
            for pmid in pmids:
                try:
                    data = await self._fetch_retraction(session, pmid, "pmid")
                    if data:
                        notice = self._parse_retraction(data)
                        if notice:
                            retractions.append(notice)
                except Exception as e:
                    logger.error(f"Error fetching retraction for PMID {pmid}: {e}")

            # Fetch DOIs
            for doi in dois:
                try:
                    data = await self._fetch_retraction(session, doi, "doi")
                    if data:
                        notice = self._parse_retraction(data)
                        if notice:
                            retractions.append(notice)
                except Exception as e:
                    logger.error(f"Error fetching retraction for DOI {doi}: {e}")

        return retractions

    def store_retraction(self, notice: RetractionNotice) -> bool:
        """
        Store retraction notice in graph and flag affected paper.

        Creates RetractionNotice node and HAS_RETRACTION relationship from Abstract.

        Args:
            notice: RetractionNotice to store

        Returns:
            True if stored successfully, False otherwise
        """
        query = """
        // Find Abstract by PMID or DOI
        OPTIONAL MATCH (a:Abstract {pmid: $pmid})
        WITH a, $pmid, $doi
        OPTIONAL MATCH (a2:Abstract {doi: $doi})
        WITH COALESCE(a, a2) AS abstract
        WHERE abstract IS NOT NULL

        // Create RetractionNotice node
        MERGE (r:RetractionNotice {retraction_watch_id: $retraction_watch_id})
        ON CREATE SET
            r.pmid = $pmid,
            r.doi = $doi,
            r.title = $title,
            r.journal = $journal,
            r.retraction_date = $retraction_date,
            r.retraction_reason = $retraction_reason,
            r.retraction_nature = $retraction_nature,
            r.original_paper_date = $original_paper_date,
            r.link = $link,
            r.data_source = 'RetractionWatch',
            r.created_at = datetime()

        // Create relationship and flag Abstract
        MERGE (abstract)-[rel:HAS_RETRACTION]->(r)
        SET abstract.is_retracted = true,
            abstract.retraction_date = $retraction_date,
            abstract.retraction_reason = $retraction_reason,
            abstract.retraction_nature = $retraction_nature

        RETURN abstract.pmid AS pmid, r.retraction_watch_id AS retraction_id
        """

        params = {
            "retraction_watch_id": notice.retraction_watch_id,
            "pmid": notice.pmid,
            "doi": notice.doi,
            "title": notice.title,
            "journal": notice.journal,
            "retraction_date": notice.retraction_date,
            "retraction_reason": notice.retraction_reason,
            "retraction_nature": notice.retraction_nature,
            "original_paper_date": notice.original_paper_date,
            "link": notice.link,
        }

        try:
            result = self.db.execute_query(query, params)
            if result:
                logger.info(f"Stored retraction notice for PMID {notice.pmid}")
                return True
            logger.warning(f"Abstract not found for PMID {notice.pmid}")
            return False
        except Exception as e:
            logger.error(f"Error storing retraction notice: {e}")
            return False

    async def enrich_abstracts_with_retractions(
        self,
        pmids: list[str] | None = None,
        dois: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Enrich abstracts with retraction notices.

        Args:
            pmids: List of PMIDs to check (if None, checks all abstracts in graph)
            dois: List of DOIs to check

        Returns:
            Summary dict with counts
        """
        # If no PMIDs/DOIs provided, fetch all from graph
        if pmids is None and dois is None:
            pmids = self.get_all_pmids()

        # Fetch retraction notices
        logger.info(
            f"Checking {len(pmids or [])} PMIDs and {len(dois or [])} DOIs for retractions..."
        )
        retractions = await self.batch_fetch_retractions(pmids=pmids, dois=dois)

        # Store retraction notices
        stored_count = 0
        for notice in retractions:
            if self.store_retraction(notice):
                stored_count += 1

        logger.info(f"Found {len(retractions)} retractions, stored {stored_count}")

        return {
            "checked_pmids": len(pmids or []),
            "checked_dois": len(dois or []),
            "retractions_found": len(retractions),
            "retractions_stored": stored_count,
        }

    def get_all_pmids(self) -> list[str]:
        """
        Get all PMIDs from Abstract nodes in graph.

        Returns:
            List of PMIDs
        """
        query = """
        MATCH (a:Abstract)
        WHERE a.pmid IS NOT NULL
        RETURN a.pmid AS pmid
        """

        try:
            result = self.db.execute_query(query)
            return [record["pmid"] for record in result]
        except Exception as e:
            logger.error(f"Error fetching PMIDs: {e}")
            return []

    def get_retracted_abstracts(self) -> list[dict[str, Any]]:
        """
        Get all retracted abstracts from graph.

        Returns:
            List of dicts with abstract and retraction info
        """
        query = """
        MATCH (a:Abstract)-[:HAS_RETRACTION]->(r:RetractionNotice)
        RETURN a.pmid AS pmid,
               a.title AS title,
               r.retraction_date AS retraction_date,
               r.retraction_reason AS retraction_reason,
               r.retraction_nature AS retraction_nature,
               r.link AS link
        ORDER BY r.retraction_date DESC
        """

        try:
            result = self.db.execute_query(query)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error fetching retracted abstracts: {e}")
            return []

    def flag_affected_relationships(self) -> int:
        """
        Flag relationships derived from retracted papers.

        Adds is_from_retracted_paper flag to all relationships connected to retracted abstracts.

        Returns:
            Count of flagged relationships
        """
        query = """
        MATCH (a:Abstract {is_retracted: true})-[:HAS_CHUNK]->(c:Chunk)
        MATCH (c)-[r]->(entity)
        WHERE type(r) IN ['ASSOCIATES_WITH', 'INTERACTS_WITH', 'REGULATES', 'TREATS']
        SET r.is_from_retracted_paper = true,
            r.retraction_flagged_at = datetime()
        RETURN count(r) AS flagged_count
        """

        try:
            result = self.db.execute_query(query)
            count = result[0]["flagged_count"] if result else 0
            logger.info(f"Flagged {count} relationships from retracted papers")
            return count
        except Exception as e:
            logger.error(f"Error flagging relationships: {e}")
            return 0
