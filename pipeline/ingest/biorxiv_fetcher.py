"""bioRxiv paper fetcher for retrieving preprint metadata and content URLs."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiohttp

from src.utils.logging_utils import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class BioRxivPaperMetadata:
    """Metadata for a bioRxiv paper retrieved from the bioRxiv API."""

    doi: str
    title: str
    authors: list[str]
    posting_date: str  # ISO format YYYY-MM-DD
    content_url: str  # URL to full paper HTML
    abstract: str | None = None
    subject_area: str | None = None


class BioRxivFetcher:
    """Fetches paper metadata and content URLs from the bioRxiv API."""

    BASE_URL = "https://api.biorxiv.org"
    CONTENT_BASE = "https://www.biorxiv.org/content"

    def __init__(self, rate_limit_delay: float = 1.0, max_retries: int = 3):
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

    async def search(
        self, query: str, max_results: int = 100
    ) -> list[BioRxivPaperMetadata]:
        """Search bioRxiv by query string.

        The bioRxiv API doesn't have a direct text search endpoint, so this
        uses the details endpoint with a recent date range and filters results
        by matching the query against title and abstract fields.
        """
        query_lower = query.lower()
        date_to = datetime.now().strftime("%Y-%m-%d")
        date_from = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        papers = await self._fetch_papers_paginated(
            date_from, date_to, max_results=max_results * 10
        )

        results: list[BioRxivPaperMetadata] = []
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

    async def fetch_by_dois(self, dois: list[str]) -> list[BioRxivPaperMetadata]:
        """Fetch metadata for specific DOIs."""
        results: list[BioRxivPaperMetadata] = []
        for doi in dois:
            url = f"{self.BASE_URL}/details/biorxiv/{doi}"
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

    async def fetch_by_date_range(
        self,
        date_from: str,
        date_to: str,
        subject: str | None = None,
        max_results: int = 100,
    ) -> list[BioRxivPaperMetadata]:
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
    ) -> list[BioRxivPaperMetadata]:
        """Fetch papers using the details endpoint with pagination."""
        papers: list[BioRxivPaperMetadata] = []
        cursor = 0

        while len(papers) < max_results:
            url = f"{self.BASE_URL}/details/biorxiv/{date_from}/{date_to}/{cursor}"
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

            # bioRxiv API returns up to 100 results per page
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

        raise last_exception  # type: ignore[misc]

    def _parse_paper(self, data: dict) -> BioRxivPaperMetadata | None:
        """Parse API response dict into BioRxivPaperMetadata.

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

        content_url = f"{self.CONTENT_BASE}/{doi}v{version}"

        abstract = data.get("abstract", "")
        if abstract is not None:
            abstract = abstract.strip() or None

        category = data.get("category", "")
        subject_area = category.strip() if category else None

        return BioRxivPaperMetadata(
            doi=doi,
            title=title,
            authors=authors,
            posting_date=date,
            content_url=content_url,
            abstract=abstract,
            subject_area=subject_area,
        )
