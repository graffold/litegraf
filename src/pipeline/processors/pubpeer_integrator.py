"""PubPeer API integration for post-publication peer review comments."""

import asyncio
import logging
from dataclasses import dataclass

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class PubPeerComment:
    """PubPeer comment metadata."""

    comment_id: str
    publication_doi: str | None
    publication_pmid: str | None
    comment_text: str
    author: str  # "Anonymous" or pseudonym
    created_at: str
    link: str
    total_comments: int


class PubPeerIntegrator:
    """Fetch post-publication peer review comments from PubPeer."""

    API_BASE = "https://api.pubpeer.com/v1"

    def __init__(
        self,
        db,
        api_key: str,
        rate_limit: float = 1.0,
    ):
        """Initialize PubPeer integrator.

        Args:
            db: Neo4j database instance
            api_key: PubPeer API key (request from contact@pubpeer.com)
            rate_limit: Delay between API requests in seconds (default 1.0)
        """
        self.db = db
        self.api_key = api_key
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self._last_request_time = asyncio.get_event_loop().time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
    )
    async def _fetch_publication(
        self,
        session: aiohttp.ClientSession,
        doi: str | None = None,
        pmid: str | None = None,
    ) -> dict | None:
        """Fetch publication data from PubPeer API.

        Args:
            session: aiohttp client session
            doi: Publication DOI
            pmid: Publication PMID

        Returns:
            Publication data dict or None if not found
        """
        await self._enforce_rate_limit()

        # PubPeer API uses DOI or PMID as identifier
        identifier = doi or f"pmid:{pmid}"
        url = f"{self.API_BASE}/publications/{identifier}"
        params = {"devkey": self.api_key}

        try:
            async with session.get(url, params=params) as response:
                if response.status == 404:
                    logger.debug(f"No PubPeer comments found for {identifier}")
                    return None
                if response.status == 429:
                    logger.warning("Rate limit exceeded, retrying...")
                    raise aiohttp.ClientError("Rate limit exceeded")
                if response.status != 200:
                    logger.error(f"API error {response.status} for {identifier}")
                    return None

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request failed for {identifier}: {e}")
            raise

    def _parse_publication(self, data: dict) -> list[PubPeerComment]:
        """Parse publication data into PubPeerComment objects.

        Args:
            data: Publication data from API

        Returns:
            List of PubPeerComment objects
        """
        if not data or "comments" not in data:
            return []

        comments = []
        pub_doi = data.get("doi")
        pub_pmid = data.get("pmid")
        pub_link = data.get("link", "")
        total_comments = data.get("total_comments", 0)

        for comment_data in data.get("comments", []):
            comment = PubPeerComment(
                comment_id=comment_data.get("id", ""),
                publication_doi=pub_doi,
                publication_pmid=pub_pmid,
                comment_text=comment_data.get("text", ""),
                author=comment_data.get("author", "Anonymous"),
                created_at=comment_data.get("created_at", ""),
                link=pub_link,
                total_comments=total_comments,
            )
            comments.append(comment)

        return comments

    async def fetch_comments_for_pmid(self, pmid: str) -> list[PubPeerComment]:
        """Fetch PubPeer comments for a single PMID.

        Args:
            pmid: PubMed ID

        Returns:
            List of PubPeerComment objects
        """
        async with aiohttp.ClientSession() as session:
            data = await self._fetch_publication(session, pmid=pmid)
            if data:
                return self._parse_publication(data)
            return []

    async def fetch_comments_for_doi(self, doi: str) -> list[PubPeerComment]:
        """Fetch PubPeer comments for a single DOI.

        Args:
            doi: Publication DOI

        Returns:
            List of PubPeerComment objects
        """
        async with aiohttp.ClientSession() as session:
            data = await self._fetch_publication(session, doi=doi)
            if data:
                return self._parse_publication(data)
            return []

    async def batch_fetch_comments(
        self, pmids: list[str] = None, dois: list[str] = None
    ) -> dict[str, list[PubPeerComment]]:
        """Fetch comments for multiple publications.

        Args:
            pmids: List of PMIDs
            dois: List of DOIs

        Returns:
            Dict mapping identifier to list of comments
        """
        results = {}
        async with aiohttp.ClientSession() as session:
            # Fetch by PMID
            if pmids:
                for pmid in pmids:
                    data = await self._fetch_publication(session, pmid=pmid)
                    if data:
                        comments = self._parse_publication(data)
                        results[f"pmid:{pmid}"] = comments

            # Fetch by DOI
            if dois:
                for doi in dois:
                    data = await self._fetch_publication(session, doi=doi)
                    if data:
                        comments = self._parse_publication(data)
                        results[doi] = comments

        return results

    def store_comments(self, comments: list[PubPeerComment]) -> int:
        """Store PubPeer comments in graph.

        Creates PubPeerComment nodes and HAS_PUBPEER_COMMENT relationships
        from Abstract nodes.

        Args:
            comments: List of PubPeerComment objects

        Returns:
            Number of comments stored
        """
        if not comments:
            return 0

        query = """
        UNWIND $comments AS comment
        MATCH (a:Abstract)
        WHERE a.pmid = comment.pmid OR a.doi = comment.doi
        MERGE (pc:PubPeerComment {comment_id: comment.comment_id})
        ON CREATE SET
            pc.text = comment.text,
            pc.author = comment.author,
            pc.created_at = comment.created_at,
            pc.link = comment.link,
            pc.total_comments = comment.total_comments,
            pc.data_source = "PubPeer"
        MERGE (a)-[:HAS_PUBPEER_COMMENT]->(pc)
        RETURN count(pc) as stored_count
        """

        comment_dicts = [
            {
                "comment_id": c.comment_id,
                "pmid": c.publication_pmid,
                "doi": c.publication_doi,
                "text": c.comment_text,
                "author": c.author,
                "created_at": c.created_at,
                "link": c.link,
                "total_comments": c.total_comments,
            }
            for c in comments
        ]

        result = self.db.execute_query(query, {"comments": comment_dicts})
        stored_count = result[0]["stored_count"] if result else 0
        logger.info(f"Stored {stored_count} PubPeer comments")
        return stored_count

    async def enrich_abstracts_with_comments(
        self, pmids: list[str] = None, dois: list[str] = None
    ) -> dict:
        """Fetch and store PubPeer comments for abstracts.

        Args:
            pmids: List of PMIDs to enrich
            dois: List of DOIs to enrich

        Returns:
            Summary dict with counts
        """
        logger.info(
            f"Fetching PubPeer comments for {len(pmids or [])} PMIDs, {len(dois or [])} DOIs"
        )

        # Fetch comments
        results = await self.batch_fetch_comments(pmids=pmids, dois=dois)

        # Flatten comments
        all_comments = []
        for comments in results.values():
            all_comments.extend(comments)

        # Store in graph
        stored_count = self.store_comments(all_comments)

        summary = {
            "total_publications_checked": len(results),
            "publications_with_comments": len([c for c in results.values() if c]),
            "total_comments_fetched": len(all_comments),
            "comments_stored": stored_count,
        }

        logger.info(f"PubPeer enrichment complete: {summary}")
        return summary

    def get_abstracts_with_comments(self) -> list[dict]:
        """Query abstracts that have PubPeer comments.

        Returns:
            List of dicts with abstract and comment info
        """
        query = """
        MATCH (a:Abstract)-[:HAS_PUBPEER_COMMENT]->(pc:PubPeerComment)
        RETURN a.pmid as pmid, a.title as title,
               count(pc) as comment_count,
               collect(pc.link)[0] as pubpeer_link
        ORDER BY comment_count DESC
        """
        return self.db.execute_query(query)
