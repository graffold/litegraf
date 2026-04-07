"""PubMed Central (PMC) full-text fetcher.

Primary strategy: Entrez efetch (XML) — works for all PMC OA articles.
Fallback: BioC JSON API — for articles not yet in efetch.
"""

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from src.utils.logging_utils import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class PMCArticle:
    """Metadata and full text for a PMC open-access article."""

    pmcid: str
    pmid: str | None = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    abstract: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(s["text"] for s in self.sections if s.get("text"))


class PMCFetcher:
    """Fetches full-text articles from PMC Open Access via Entrez efetch."""

    def __init__(self, rate_limit_delay: float = 0.35, batch_size: int = 10):
        self.rate_limit_delay = rate_limit_delay
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_and_fetch(
        self, query: str, max_results: int = 50
    ) -> list[PMCArticle]:
        """Search PMC for OA articles and fetch full text via efetch XML."""
        pmcids = await self.search(query, max_results)
        if not pmcids:
            return []
        return await self._fetch_batch_efetch(pmcids)

    async def search(self, query: str, max_results: int = 50) -> list[str]:
        """Search PMC for OA article IDs (numeric, without PMC prefix).

        Returns a list of numeric PMC ID strings suitable for efetch.
        """
        from Bio import Entrez

        from src.config import Config

        Entrez.email = Config.get_config("ENTREZ_EMAIL") or "user@example.com"
        Entrez.api_key = Config.get_config("ENTREZ_API_KEY")

        oa_query = f'({query}) AND "open access"[filter]'
        logger.info(f"Searching PMC Open Access for: {query} (max {max_results})")

        try:
            handle = Entrez.esearch(
                db="pmc", term=oa_query, retmax=max_results, usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.error(f"PMC esearch failed: {e}")
            return []

        pmcids: list[str] = record.get("IdList", [])
        if not pmcids:
            logger.info("No PMC results found")
        else:
            logger.info(f"Found {len(pmcids)} PMC IDs")
        return pmcids

    async def fetch_batch(self, pmcids: list[str]) -> list[PMCArticle]:
        """Fetch a batch of articles by numeric PMC IDs.

        Public wrapper around _fetch_batch_efetch for use by the ingestor's
        streaming pipeline.
        """
        return await self._fetch_batch_efetch(pmcids)

    async def fetch_by_pmcids(self, pmcids: list[str]) -> list[PMCArticle]:
        """Fetch full text for a list of PMC IDs."""
        # Strip "PMC" prefix if present — efetch wants numeric IDs
        numeric_ids = [p.replace("PMC", "") for p in pmcids]
        return await self._fetch_batch_efetch(numeric_ids)

    # ------------------------------------------------------------------
    # Batch efetch (primary strategy)
    # ------------------------------------------------------------------

    async def _fetch_batch_efetch(self, pmcids: list[str]) -> list[PMCArticle]:
        """Fetch articles in batches using Entrez efetch XML."""
        from Bio import Entrez

        from src.config import Config

        Entrez.email = Config.get_config("ENTREZ_EMAIL") or "user@example.com"
        Entrez.api_key = Config.get_config("ENTREZ_API_KEY")

        articles: list[PMCArticle] = []
        total = len(pmcids)

        for i in range(0, total, self.batch_size):
            batch = pmcids[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            logger.info(
                f"Fetching batch {batch_num}/{total_batches} ({len(batch)} articles)"
            )

            try:
                handle = Entrez.efetch(
                    db="pmc", id=",".join(batch), rettype="xml", retmode="xml"
                )
                xml_text = handle.read()
                handle.close()

                if isinstance(xml_text, bytes):
                    xml_text = xml_text.decode("utf-8")

                batch_articles = self._parse_pmc_xml(xml_text)
                articles.extend(batch_articles)
                logger.info(f"Batch {batch_num}: parsed {len(batch_articles)} articles")

            except Exception as e:
                logger.error(f"efetch failed for batch {batch_num}: {e}")
                # Fall back to one-by-one for this batch
                for pmcid in batch:
                    try:
                        handle = Entrez.efetch(
                            db="pmc", id=pmcid, rettype="xml", retmode="xml"
                        )
                        xml_text = handle.read()
                        handle.close()
                        if isinstance(xml_text, bytes):
                            xml_text = xml_text.decode("utf-8")
                        single = self._parse_pmc_xml(xml_text)
                        articles.extend(single)
                    except Exception as e2:
                        logger.warning(f"Failed to fetch PMC{pmcid}: {e2}")
                    await asyncio.sleep(self.rate_limit_delay)

            await asyncio.sleep(self.rate_limit_delay)

        return articles

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------

    def _parse_pmc_xml(self, xml_text: str) -> list[PMCArticle]:
        """Parse PMC efetch XML into PMCArticle objects."""
        articles: list[PMCArticle] = []

        try:
            root = ET.fromstring(xml_text)  # noqa: S314
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return []

        # efetch returns <pmc-articleset> with <article> children
        article_elements = root.findall(".//article")
        if not article_elements:
            # Single article might be the root itself
            if root.tag == "article":
                article_elements = [root]

        for art_el in article_elements:
            article = self._parse_article_element(art_el)
            if article:
                articles.append(article)

        return articles

    def _parse_article_element(self, art: ET.Element) -> PMCArticle | None:
        """Parse a single <article> element."""
        front = art.find(".//front")
        if front is None:
            return None

        # PMC ID
        pmcid = ""
        pmid = None
        for aid in front.findall(".//article-id"):
            id_type = aid.get("pub-id-type", "")
            if id_type in ("pmc", "pmcid"):
                raw = (aid.text or "").strip()
                # Normalise to "PMCnnnnn" regardless of whether the XML
                # already includes the prefix (pmcid type) or not (pmc type).
                pmcid = raw if raw.startswith("PMC") else f"PMC{raw}"
            elif id_type == "pmid":
                pmid = aid.text

        if not pmcid:
            return None

        # Title
        title_el = front.find(".//article-title")
        title = self._get_text(title_el) if title_el is not None else ""

        # Authors
        authors: list[str] = []
        for contrib in front.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.findtext("name/surname", "")
            given = contrib.findtext("name/given-names", "")
            if surname:
                authors.append(f"{given} {surname}".strip())

        # Journal
        journal = front.findtext(".//journal-title", "")

        # Year
        year = ""
        pub_date = front.find(".//pub-date")
        if pub_date is not None:
            year = pub_date.findtext("year", "")

        # Abstract
        abstract_parts: list[str] = []
        for abs_el in front.findall(".//abstract"):
            for p in abs_el.findall(".//p"):
                text = self._get_text(p)
                if text:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Body sections
        sections: list[dict[str, str]] = []
        body = art.find(".//body")
        if body is not None:
            for sec in body.findall(".//sec"):
                heading_el = sec.find("title")
                heading = self._get_text(heading_el) if heading_el is not None else ""
                paragraphs: list[str] = []
                for p in sec.findall(".//p"):
                    text = self._get_text(p)
                    if text:
                        paragraphs.append(text)
                if paragraphs:
                    sections.append({"heading": heading, "text": " ".join(paragraphs)})

            # Also grab paragraphs directly under <body> (not in a <sec>)
            for p in body.findall("./p"):
                text = self._get_text(p)
                if text:
                    sections.append({"heading": "", "text": text})

        if not title and not sections:
            return None

        return PMCArticle(
            pmcid=pmcid,
            pmid=pmid,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            sections=sections,
            abstract=abstract,
        )

    @staticmethod
    def _get_text(el: ET.Element) -> str:
        """Recursively extract all text from an XML element, stripping tags."""
        return "".join(el.itertext()).strip()
