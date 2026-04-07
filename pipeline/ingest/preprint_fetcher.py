"""Unified preprint fetcher for bioRxiv and medRxiv APIs."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import aiohttp

logger = logging.getLogger(__name__)

ServerType = Literal["biorxiv", "medrxiv"]


@dataclass
class PreprintMetadata:
    """Metadata for a preprint paper from bioRxiv or medRxiv."""

    doi: str
    title: str
    authors: list[str]
    posting_date: str
    content_url: str
    abstract: str | None = None
    subject_area: str | None = None
    server: ServerType = "biorxiv"
    version: str = "1"
    published_doi: str | None = None
    published_pmid: str | None = None


class PreprintFetcher:
    """Fetches paper metadata from bioRxiv and medRxiv APIs.

    Both bioRxiv and medRxiv use the same API infrastructure, differing only
    in the server parameter. This class provides a unified interface for both.
    """

    BASE_URL = "https://api.biorxiv.org"
    CONTENT_BASE_BIORXIV = "https://www.biorxiv.org/content"
    CONTENT_BASE_MEDRXIV = "https://www.medrxiv.org/content"

    def __init__(
        self,
        server: ServerType = "biorxiv",
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ):
        """Initialize the fetcher.

        Args:
            server: Which preprint server to query ("biorxiv" or "medrxiv")
            rate_limit_delay: Delay in seconds between API requests
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.server = server
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.content_base = (
            self.CONTENT_BASE_MEDRXIV
            if server == "medrxiv"
            else self.CONTENT_BASE_BIORXIV
        )

    async def search(
        self, query: str, max_results: int = 100
    ) -> list[PreprintMetadata]:
        """Search preprints by query string.

        The API doesn't have a direct text search endpoint, so this
        uses the details endpoint with a recent date range and filters results
        by matching the query against title and abstract fields.
        """
        query_lower = query.lower()
        date_to = datetime.now().strftime("%Y-%m-%d")
        date_from = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        papers = await self._fetch_papers_paginated(
            date_from, date_to, max_results=max_results * 5
        )

        results: list[PreprintMetadata] = []
        for paper in papers:
            title_match = query_lower in paper.title.lower()
            abstract_match = (
                paper.abstract is not None and query_lower in paper.abstract.lower()
            )
            if title_match or abstract_match:
                results.append(paper)
                if len(results) >= max_results:
                    break

        return results

    async def fetch_by_dois(self, dois: list[str]) -> list[PreprintMetadata]:
        """Fetch metadata for specific DOIs (latest version only)."""
        results: list[PreprintMetadata] = []
        for doi in dois:
            url = f"{self.BASE_URL}/details/{self.server}/{doi}"
            try:
                data = await self._make_request(url)
                collection = data.get("collection", [])
                if collection:
                    # Use the latest version (last entry)
                    paper = self._parse_paper(collection[-1])
                    if paper is not None:
                        results.append(paper)
                    else:
                        logger.warning(f"Could not parse paper data for DOI: {doi}")
                else:
                    logger.warning(f"No results found for DOI: {doi}")
            except Exception as e:
                logger.error(f"Failed to fetch DOI {doi}: {e}")
            await asyncio.sleep(self.rate_limit_delay)
        return results

    async def fetch_all_versions(self, doi: str) -> list[PreprintMetadata]:
        """Fetch all versions of a preprint by DOI.

        Returns list of PreprintMetadata sorted by version (oldest to newest).
        """
        url = f"{self.BASE_URL}/details/{self.server}/{doi}"
        try:
            data = await self._make_request(url)
            collection = data.get("collection", [])
            if not collection:
                logger.warning(f"No versions found for DOI: {doi}")
                return []

            # Parse all versions
            versions = []
            for version_data in collection:
                paper = self._parse_paper(version_data)
                if paper is not None:
                    versions.append(paper)

            # Sort by version number
            versions.sort(key=lambda p: int(p.version))
            return versions
        except Exception as e:
            logger.error(f"Failed to fetch versions for DOI {doi}: {e}")
            return []

    async def fetch_by_date_range(
        self,
        date_from: str,
        date_to: str,
        subject: str | None = None,
        max_results: int = 100,
    ) -> list[PreprintMetadata]:
        """Fetch papers posted within a date range, optionally filtered by subject area."""
        papers = await self._fetch_papers_paginated(date_from, date_to, max_results)

        if subject is not None:
            subject_lower = subject.lower()
            papers = [
                p
                for p in papers
                if p.subject_area is not None
                and p.subject_area.lower() == subject_lower
            ]

        return papers[:max_results]

    async def _fetch_papers_paginated(
        self, date_from: str, date_to: str, max_results: int
    ) -> list[PreprintMetadata]:
        """Fetch papers using the details endpoint with pagination."""
        papers: list[PreprintMetadata] = []
        cursor = 0

        while len(papers) < max_results:
            url = (
                f"{self.BASE_URL}/details/{self.server}/{date_from}/{date_to}/{cursor}"
            )
            try:
                data = await self._make_request(url)
            except Exception as e:
                logger.error(f"Failed to fetch papers at cursor {cursor}: {e}")
                break

            collection = data.get("collection", [])
            if not collection:
                break

            for item in collection:
                paper = self._parse_paper(item)
                if paper is not None:
                    papers.append(paper)
                    if len(papers) >= max_results:
                        break

            # API returns up to 100 results per page
            if len(collection) < 100:
                break

            cursor += len(collection)
            await asyncio.sleep(self.rate_limit_delay)

        return papers[:max_results]

    async def _make_request(self, url: str) -> dict:
        """Make an HTTP request with exponential backoff retry."""
        delay = 1.0
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.json()
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Request to {url} failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2.0

        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"Request to {url} failed after {self.max_retries} attempts")

    def _parse_paper(self, data: dict) -> PreprintMetadata | None:
        """Parse API response dict into PreprintMetadata.

        Returns None if required fields are missing.
        """
        doi = data.get("doi", "").strip()
        title = data.get("title", "").strip()
        authors_raw = data.get("authors", "").strip()
        date = data.get("date", "").strip()
        version = data.get("version", "1")

        if not doi or not title or not authors_raw or not date:
            return None

        authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
        if not authors:
            return None

        content_url = f"{self.content_base}/{doi}v{version}"

        abstract = data.get("abstract", "")
        if abstract is not None:
            abstract = abstract.strip() or None

        category = data.get("category", "")
        subject_area = category.strip() if category else None

        # Extract publication information if available
        published_doi = data.get("published_doi", "")
        published_doi = published_doi.strip() if published_doi else None

        published_pmid = data.get("published_pmid", "")
        published_pmid = published_pmid.strip() if published_pmid else None

        return PreprintMetadata(
            doi=doi,
            title=title,
            authors=authors,
            posting_date=date,
            content_url=content_url,
            abstract=abstract,
            subject_area=subject_area,
            server=self.server,
            version=str(version),
            published_doi=published_doi,
            published_pmid=published_pmid,
        )
