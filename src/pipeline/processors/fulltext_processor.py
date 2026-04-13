"""
FullTextProcessor - Fetch full-text articles from PMC Open Access.

This module provides functionality to fetch full-text articles from PubMed Central (PMC)
Open Access subset using the NCBI E-utilities API.
"""

import logging
import sqlite3
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from Bio import Entrez
except ImportError:
    Entrez = None

from pipeline.config import PipelineConfig as Config

logger = logging.getLogger(__name__)


class FullTextProcessor:
    """
    Fetch and parse full-text articles from PMC Open Access.

    Uses NCBI E-utilities API to fetch full-text XML from PMC Open Access subset.
    Handles rate limiting, retries, and XML parsing with parallel processing support.
    """

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        rate_limit_delay: float = 0.34,
        checkpoint_db_path: str = "./data/checkpoints/fulltext_processing.db",
        max_workers: int = 3,
    ):
        """
        Initialize FullTextProcessor.

        Args:
            email: Email for NCBI Entrez (required by NCBI)
            api_key: Optional NCBI API key for higher rate limits
            rate_limit_delay: Delay between API calls (default 0.34s = ~3 requests/sec)
            checkpoint_db_path: Path to SQLite checkpoint database for incremental processing
            max_workers: Maximum parallel workers for batch processing (default 3)
        """
        if Entrez is None:
            raise ImportError(
                "Biopython is required for PMC API access. Install with: pip install biopython"
            )

        self.email = email or Config.ENTREZ_EMAIL
        self.api_key = api_key or Config.ENTREZ_API_KEY
        self.rate_limit_delay = rate_limit_delay
        self.checkpoint_db_path = checkpoint_db_path
        self.max_workers = max_workers

        # Configure Entrez
        Entrez.email = self.email
        if self.api_key and self.api_key != "your-entrez-api-key":
            Entrez.api_key = self.api_key
            logger.info("Using NCBI API key for higher rate limits")

        self._last_request_time = 0.0

        # Initialize checkpoint database
        if checkpoint_db_path:
            Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_checkpoint_db()

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _init_checkpoint_db(self) -> None:
        """Initialize SQLite checkpoint database for incremental processing."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fulltext_processing (
                pmc_id TEXT PRIMARY KEY,
                pmid TEXT,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT,
                processing_time_seconds REAL DEFAULT 0.0
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Initialized checkpoint database at {self.checkpoint_db_path}")

    def _get_processing_state(self, pmc_id: str) -> str | None:
        """Get current processing state for a PMC ID."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT state FROM fulltext_processing WHERE pmc_id = ?", (pmc_id,)
        )
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def _update_processing_state(
        self,
        pmc_id: str,
        state: str,
        pmid: str | None = None,
        error_message: str | None = None,
        processing_time: float = 0.0,
    ) -> None:
        """Update processing state in checkpoint database."""
        conn = sqlite3.connect(self.checkpoint_db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO fulltext_processing
            (pmc_id, pmid, state, created_at, updated_at, error_message, processing_time_seconds)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM fulltext_processing WHERE pmc_id = ?), ?), ?, ?, ?)
        """,
            (pmc_id, pmid, state, pmc_id, now, now, error_message, processing_time),
        )

        conn.commit()
        conn.close()

    def batch_fetch_fulltext(
        self, pmc_ids: list[str], skip_completed: bool = True
    ) -> dict[str, dict[str, Any] | None]:
        """
        Fetch multiple full-text articles in parallel with checkpoint support.

        Args:
            pmc_ids: List of PMC IDs to fetch
            skip_completed: Skip PMC IDs that are already completed in checkpoint database

        Returns:
            Dictionary mapping PMC ID to parsed full-text data (or None if failed)
        """
        results: dict[str, dict[str, Any] | None] = {}

        # Filter out completed PMC IDs if requested
        pmc_ids_to_process = []
        for pmc_id in pmc_ids:
            normalized_id = pmc_id.strip()
            if not normalized_id.startswith("PMC"):
                normalized_id = f"PMC{normalized_id}"

            if skip_completed and self.checkpoint_db_path:
                state = self._get_processing_state(normalized_id)
                if state == "completed":
                    logger.info(f"Skipping {normalized_id} (already completed)")
                    continue

            pmc_ids_to_process.append(normalized_id)

        if not pmc_ids_to_process:
            logger.info("No PMC IDs to process (all completed)")
            return results

        logger.info(
            f"Processing {len(pmc_ids_to_process)} PMC IDs with {self.max_workers} workers"
        )

        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pmc_id = {
                executor.submit(self._fetch_with_checkpoint, pmc_id): pmc_id
                for pmc_id in pmc_ids_to_process
            }

            for future in as_completed(future_to_pmc_id):
                pmc_id = future_to_pmc_id[future]
                try:
                    result = future.result()
                    results[pmc_id] = result
                except Exception as e:
                    logger.error(f"Error processing {pmc_id}: {e}")
                    results[pmc_id] = None

        return results

    def _fetch_with_checkpoint(self, pmc_id: str) -> dict[str, Any] | None:
        """Fetch full-text with checkpoint tracking."""
        start_time = time.time()

        try:
            # Update state to in_progress
            if self.checkpoint_db_path:
                self._update_processing_state(pmc_id, "in_progress")

            # Fetch full-text
            result = self.fetch_fulltext(pmc_id)

            # Update state to completed
            if self.checkpoint_db_path:
                processing_time = time.time() - start_time
                pmid = result.get("pmid") if result else None
                self._update_processing_state(
                    pmc_id, "completed", pmid=pmid, processing_time=processing_time
                )

            return result

        except Exception as e:
            # Update state to failed
            if self.checkpoint_db_path:
                processing_time = time.time() - start_time
                self._update_processing_state(
                    pmc_id,
                    "failed",
                    error_message=str(e),
                    processing_time=processing_time,
                )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((HTTPError, ConnectionError)),
    )
    def fetch_fulltext(self, pmc_id: str) -> dict[str, Any] | None:
        """
        Fetch full-text article from PMC Open Access.

        Args:
            pmc_id: PMC ID (e.g., "PMC1234567" or "1234567")

        Returns:
            Dictionary with parsed full-text data or None if not available
            {
                "pmc_id": str,
                "pmid": str | None,
                "title": str,
                "abstract": str,
                "body": str,
                "sections": list[dict],  # [{"title": str, "content": str}]
                "authors": list[str],
                "journal": str,
                "publication_date": str,
                "doi": str | None,
                "keywords": list[str]
            }
        """
        # Normalize PMC ID
        pmc_id = pmc_id.strip()
        if not pmc_id.startswith("PMC"):
            pmc_id = f"PMC{pmc_id}"

        try:
            self._rate_limit()

            # Fetch full-text XML from PMC
            logger.info(f"Fetching full-text for {pmc_id}")
            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml", retmode="xml")
            xml_content = handle.read()
            handle.close()

            # Parse XML
            return self._parse_pmc_xml(xml_content, pmc_id)

        except HTTPError as e:
            if e.code == 404:
                logger.warning(f"Full-text not available for {pmc_id} (404)")
                return None
            if e.code == 429:
                logger.warning(f"Rate limit exceeded for {pmc_id}, retrying...")
                raise  # Retry via tenacity
            logger.error(f"HTTP error fetching {pmc_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching full-text for {pmc_id}: {e}")
            return None

    def _parse_pmc_xml(self, xml_content: str | bytes, pmc_id: str) -> dict[str, Any]:
        """
        Parse PMC XML to extract full-text content.

        Args:
            xml_content: Raw XML content from PMC
            pmc_id: PMC ID for logging

        Returns:
            Dictionary with parsed full-text data
        """
        try:
            root = ET.fromstring(xml_content)  # noqa: S314

            # Extract metadata
            article = root.find(".//article")
            if article is None:
                logger.warning(f"No article element found in {pmc_id}")
                return self._empty_result(pmc_id)

            # Extract front matter (metadata)
            front = article.find("front")
            metadata = self._extract_metadata(front) if front is not None else {}

            # Extract abstract
            abstract = self._extract_abstract(front) if front is not None else ""

            # Extract body (main content)
            body_elem = article.find("body")
            body_text, sections = (
                self._extract_body(body_elem) if body_elem is not None else ("", [])
            )

            # Extract figures and tables in parallel for better performance
            figures = []
            tables = []

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_figures = executor.submit(self._extract_figures, article)
                future_tables = executor.submit(self._extract_tables, article)

                figures = future_figures.result()
                tables = future_tables.result()

            return {
                "pmc_id": pmc_id,
                "pmid": metadata.get("pmid"),
                "title": metadata.get("title", ""),
                "abstract": abstract,
                "body": body_text,
                "sections": sections,
                "figures": figures,
                "tables": tables,
                "authors": metadata.get("authors", []),
                "journal": metadata.get("journal", ""),
                "publication_date": metadata.get("publication_date", ""),
                "doi": metadata.get("doi"),
                "keywords": metadata.get("keywords", []),
            }

        except ET.ParseError as e:
            logger.error(f"XML parsing error for {pmc_id}: {e}")
            return self._empty_result(pmc_id)
        except Exception as e:
            logger.error(f"Error parsing PMC XML for {pmc_id}: {e}")
            return self._empty_result(pmc_id)

    def _extract_metadata(self, front: ET.Element) -> dict[str, Any]:
        """Extract metadata from front matter."""
        metadata: dict[str, Any] = {}

        # Title
        title_elem = front.find(".//article-title")
        if title_elem is not None:
            metadata["title"] = self._get_text(title_elem)

        # PMID
        pmid_elem = front.find(".//article-id[@pub-id-type='pmid']")
        if pmid_elem is not None:
            metadata["pmid"] = pmid_elem.text

        # DOI
        doi_elem = front.find(".//article-id[@pub-id-type='doi']")
        if doi_elem is not None:
            metadata["doi"] = doi_elem.text

        # Authors
        authors = []
        for contrib in front.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.find(".//surname")
            given_names = contrib.find(".//given-names")
            if surname is not None:
                author = surname.text or ""
                if given_names is not None and given_names.text:
                    author = f"{given_names.text} {author}"
                authors.append(author)
        metadata["authors"] = authors

        # Journal
        journal_elem = front.find(".//journal-title")
        if journal_elem is not None:
            metadata["journal"] = journal_elem.text or ""

        # Publication date
        pub_date = front.find(".//pub-date")
        if pub_date is not None:
            year = pub_date.find("year")
            month = pub_date.find("month")
            day = pub_date.find("day")
            date_parts = []
            if year is not None and year.text:
                date_parts.append(year.text)
            if month is not None and month.text:
                date_parts.append(month.text.zfill(2))
            if day is not None and day.text:
                date_parts.append(day.text.zfill(2))
            metadata["publication_date"] = "-".join(date_parts)

        # Keywords
        keywords = []
        for kwd in front.findall(".//kwd"):
            if kwd.text:
                keywords.append(kwd.text)
        metadata["keywords"] = keywords

        return metadata

    def _extract_abstract(self, front: ET.Element) -> str:
        """Extract abstract text."""
        abstract_elem = front.find(".//abstract")
        if abstract_elem is not None:
            return self._get_text(abstract_elem)
        return ""

    def _extract_body(self, body: ET.Element) -> tuple[str, list[dict[str, str]]]:
        """
        Extract body text and sections.

        Returns:
            Tuple of (full_body_text, sections_list)
        """
        sections = []
        full_text_parts = []

        for sec in body.findall(".//sec"):
            # Extract section title
            title_elem = sec.find("title")
            section_title = (
                self._get_text(title_elem)
                if title_elem is not None
                else "Untitled Section"
            )

            # Extract section content (all paragraphs)
            paragraphs = []
            for p in sec.findall(".//p"):
                para_text = self._get_text(p)
                if para_text:
                    paragraphs.append(para_text)

            section_content = "\n\n".join(paragraphs)

            if section_content:
                sections.append({"title": section_title, "content": section_content})
                full_text_parts.append(f"## {section_title}\n\n{section_content}")

        full_body_text = "\n\n".join(full_text_parts)
        return full_body_text, sections

    def _get_text(self, element: ET.Element) -> str:
        """
        Extract all text from an element, including nested elements.

        Handles inline elements like <italic>, <bold>, etc.
        """
        text_parts = []

        # Get text before first child
        if element.text:
            text_parts.append(element.text)

        # Get text from children
        for child in element:
            text_parts.append(self._get_text(child))
            # Get tail text after child
            if child.tail:
                text_parts.append(child.tail)

        return "".join(text_parts).strip()

    def _empty_result(self, pmc_id: str) -> dict[str, Any]:
        """Return empty result structure."""
        return {
            "pmc_id": pmc_id,
            "pmid": None,
            "title": "",
            "abstract": "",
            "body": "",
            "sections": [],
            "figures": [],
            "tables": [],
            "authors": [],
            "journal": "",
            "publication_date": "",
            "doi": None,
            "keywords": [],
        }

    def _extract_figures(self, article: ET.Element) -> list[dict[str, Any]]:
        """
        Extract figures from article.

        Returns:
            List of figure dictionaries with metadata and image URLs
        """
        figures = []

        for fig in article.findall(".//fig"):
            figure_data: dict[str, Any] = {}

            # Extract figure ID
            figure_data["id"] = fig.get("id", "")

            # Extract label (e.g., "Fig. 1")
            label_elem = fig.find("label")
            figure_data["label"] = (
                self._get_text(label_elem) if label_elem is not None else ""
            )

            # Extract caption
            caption_elem = fig.find("caption")
            if caption_elem is not None:
                # Extract title from caption
                title_elem = caption_elem.find("title")
                figure_data["title"] = (
                    self._get_text(title_elem) if title_elem is not None else ""
                )

                # Extract caption text (all paragraphs)
                caption_paragraphs = []
                for p in caption_elem.findall("p"):
                    para_text = self._get_text(p)
                    if para_text:
                        caption_paragraphs.append(para_text)
                figure_data["caption"] = " ".join(caption_paragraphs)
            else:
                figure_data["title"] = ""
                figure_data["caption"] = ""

            # Extract graphic URLs
            graphics = []

            # Check for alternatives (multiple representations)
            alternatives = fig.find("alternatives")
            if alternatives is not None:
                for graphic in alternatives.findall("graphic"):
                    graphic_info = self._extract_graphic_info(graphic)
                    if graphic_info:
                        graphics.append(graphic_info)
            else:
                # Single graphic
                graphic = fig.find("graphic")
                if graphic is not None:
                    graphic_info = self._extract_graphic_info(graphic)
                    if graphic_info:
                        graphics.append(graphic_info)

            figure_data["graphics"] = graphics

            figures.append(figure_data)

        return figures

    def _extract_tables(self, article: ET.Element) -> list[dict[str, Any]]:
        """
        Extract tables from article.

        Returns:
            List of table dictionaries with metadata and content
        """
        tables = []

        for table_wrap in article.findall(".//table-wrap"):
            table_data: dict[str, Any] = {}

            # Extract table ID
            table_data["id"] = table_wrap.get("id", "")

            # Extract label (e.g., "Table 1")
            label_elem = table_wrap.find("label")
            table_data["label"] = (
                self._get_text(label_elem) if label_elem is not None else ""
            )

            # Extract caption
            caption_elem = table_wrap.find("caption")
            if caption_elem is not None:
                # Extract title from caption
                title_elem = caption_elem.find("title")
                table_data["title"] = (
                    self._get_text(title_elem) if title_elem is not None else ""
                )

                # Extract caption text (all paragraphs)
                caption_paragraphs = []
                for p in caption_elem.findall("p"):
                    para_text = self._get_text(p)
                    if para_text:
                        caption_paragraphs.append(para_text)
                table_data["caption"] = " ".join(caption_paragraphs)
            else:
                table_data["title"] = ""
                table_data["caption"] = ""

            # Extract table content
            # Check for alternatives (table as graphic vs structured table)
            alternatives = table_wrap.find("alternatives")
            if alternatives is not None:
                # Extract both graphic and structured table if available
                graphic = alternatives.find("graphic")
                if graphic is not None:
                    table_data["graphic"] = self._extract_graphic_info(graphic)

                table_elem = alternatives.find("table")
                if table_elem is not None:
                    table_data["structured_content"] = self._extract_table_structure(
                        table_elem
                    )
            else:
                # Check for graphic representation
                graphic = table_wrap.find("graphic")
                if graphic is not None:
                    table_data["graphic"] = self._extract_graphic_info(graphic)

                # Check for structured table
                table_elem = table_wrap.find("table")
                if table_elem is not None:
                    table_data["structured_content"] = self._extract_table_structure(
                        table_elem
                    )

            tables.append(table_data)

        return tables

    def _extract_graphic_info(self, graphic: ET.Element) -> dict[str, str] | None:
        """
        Extract graphic information (URL, mimetype, etc.).

        Returns:
            Dictionary with graphic metadata or None if no URL found
        """
        # Extract xlink:href attribute (image URL)
        href = graphic.get("{http://www.w3.org/1999/xlink}href") or graphic.get("href")
        if not href:
            return None

        return {
            "url": href,
            "mimetype": graphic.get("mimetype", ""),
            "mime_subtype": graphic.get("mime-subtype", ""),
            "specific_use": graphic.get("specific-use", ""),
        }

    def _extract_table_structure(self, table: ET.Element) -> list[list[str]]:
        """
        Extract structured table content as 2D array.

        Returns:
            List of rows, where each row is a list of cell values
        """
        rows = []

        # Extract header rows
        thead = table.find("thead")
        if thead is not None:
            for tr in thead.findall("tr"):
                row = []
                for cell in tr.findall("th") + tr.findall("td"):
                    row.append(self._get_text(cell))
                if row:
                    rows.append(row)

        # Extract body rows
        tbody = table.find("tbody")
        if tbody is not None:
            for tr in tbody.findall("tr"):
                row = []
                for cell in tr.findall("td") + tr.findall("th"):
                    row.append(self._get_text(cell))
                if row:
                    rows.append(row)

        # If no thead/tbody, extract rows directly
        if not rows:
            for tr in table.findall("tr"):
                row = []
                for cell in tr.findall("td") + tr.findall("th"):
                    row.append(self._get_text(cell))
                if row:
                    rows.append(row)

        return rows

    def is_available_in_pmc(self, pmid: str) -> str | None:
        """
        Check if a PMID has full-text available in PMC Open Access.

        Args:
            pmid: PubMed ID

        Returns:
            PMC ID if available, None otherwise
        """
        try:
            self._rate_limit()

            # Search for PMC ID using PMID
            handle = Entrez.elink(
                dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc"
            )
            result = Entrez.read(handle)
            handle.close()

            # Extract PMC ID from result
            if result and len(result) > 0:
                link_set = result[0]
                if "LinkSetDb" in link_set and len(link_set["LinkSetDb"]) > 0:
                    links = link_set["LinkSetDb"][0]
                    if "Link" in links and len(links["Link"]) > 0:
                        pmc_id = links["Link"][0]["Id"]
                        return f"PMC{pmc_id}"

            return None

        except Exception as e:
            logger.error(f"Error checking PMC availability for PMID {pmid}: {e}")
            return None
