"""
Large-scale PubMed scraper with advanced deduplication, checkpoint recovery, and distributed processing.

This module provides enterprise-grade PubMed scraping capabilities designed for processing
the entire PubMed database (35+ million abstracts) with robust error handling, progress tracking,
and resume capabilities.
"""

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from Bio import Entrez
except ImportError:
    print("Warning: BioPython not installed. Install with: pip install biopython")
    Entrez = None

try:
    import aiofiles
except ImportError:
    print("Warning: aiofiles not installed. Install with: pip install aiofiles")
    aiofiles = None

try:
    import aiohttp
except ImportError:
    print("Warning: aiohttp not installed. Install with: pip install aiohttp")
    aiohttp = None

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:
    print("Warning: tenacity not installed. Install with: pip install tenacity")

    # Create dummy decorators for compatibility
    def retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    stop_after_attempt = retry
    wait_exponential = retry
    retry_if_exception_type = retry

import signal
from enum import Enum
from urllib.error import HTTPError

from src.utils.logging_utils import setup_logging

try:
    from src.models.document import Chunk, ProcessedDocument
except ImportError:
    # Create basic classes for compatibility
    @dataclass
    class ProcessedDocument:
        doc_id: str
        source: str
        metadata: dict[str, Any]
        chunks: list["Chunk"]

    @dataclass
    class Chunk:
        chunk_id: str
        text: str


from src.config import Config
from src.core.database import DatabaseInterface

logger = setup_logging()


class ScrapingState(Enum):
    """Enum for tracking scraping states"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScrapingCheckpoint:
    """Checkpoint data structure for resume capability"""

    session_id: str
    search_term: str
    total_expected: int
    processed_count: int
    failed_count: int
    skipped_count: int
    current_batch_start: int
    batch_size: int
    last_update: datetime
    estimated_completion: datetime | None = None
    error_rate: float = 0.0
    avg_processing_time: float = 0.0


@dataclass
class PMIDRecord:
    """PMID processing record with state tracking"""

    pmid: str
    state: ScrapingState
    title: str | None = None
    abstract_hash: str | None = None
    processing_attempts: int = 0
    last_attempt: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)


class PubMedMassScraper:
    """
    Enterprise-grade PubMed scraper with Factory Pattern initialization and Strategy Pattern
    for different processing approaches.
    """

    def __init__(
        self,
        database: DatabaseInterface,
        checkpoint_dir: str = "pubmed_checkpoints",
        batch_size: int = 500,
        max_concurrent_batches: int = 3,
        rate_limit_delay: float = 0.5,
        enable_distributed: bool = False,
    ):
        """
        Initialize PubMed mass scraper with configurable parameters.

        Args:
            database: Database interface for storage
            checkpoint_dir: Directory for checkpoint files
            batch_size: Number of PMIDs per batch
            max_concurrent_batches: Maximum concurrent batch processing
            rate_limit_delay: Base delay between API calls
            enable_distributed: Enable distributed processing capabilities
        """
        self.database = database
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.rate_limit_delay = rate_limit_delay
        self.enable_distributed = enable_distributed

        # Initialize checkpoint database
        self.checkpoint_db = self._init_checkpoint_database()

        # Processing statistics
        self.stats: dict[str, int | float | datetime | None] = {
            "total_processed": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_duplicates": 0,
            "processing_start_time": None,
            "last_checkpoint_time": None,
            "error_rate": 0.0,
            "avg_batch_time": 0.0,
        }

        # Rate limiting and error handling
        self.adaptive_rate_limiter = AdaptiveRateLimiter(base_delay=rate_limit_delay)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=300)

        # Graceful shutdown handling
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Configure Entrez
        Entrez.email = Config.ENTREZ_EMAIL

    def _init_checkpoint_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for checkpoint and progress tracking"""
        db_path = self.checkpoint_dir / "scraping_progress.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create tables for progress tracking
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id TEXT PRIMARY KEY,
                search_term TEXT NOT NULL,
                total_expected INTEGER,
                processed_count INTEGER DEFAULT 0,
                failed_count INTEGER DEFAULT 0,
                skipped_count INTEGER DEFAULT 0,
                current_batch_start INTEGER DEFAULT 0,
                batch_size INTEGER,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checkpoint_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pmid_records (
                pmid TEXT PRIMARY KEY,
                session_id TEXT,
                state TEXT NOT NULL,
                title TEXT,
                abstract_hash TEXT,
                processing_attempts INTEGER DEFAULT 0,
                last_attempt TIMESTAMP,
                error_message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES checkpoints(session_id)
            );

            CREATE TABLE IF NOT EXISTS batch_progress (
                batch_id TEXT PRIMARY KEY,
                session_id TEXT,
                batch_start INTEGER,
                batch_end INTEGER,
                status TEXT,
                processing_time REAL,
                error_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES checkpoints(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_pmid_state ON pmid_records(state);
            CREATE INDEX IF NOT EXISTS idx_session_pmids ON pmid_records(session_id);
            CREATE INDEX IF NOT EXISTS idx_batch_session ON batch_progress(session_id);
        """)

        conn.commit()
        return conn

    def _signal_handler(self, signum: int, frame):
        """Handle graceful shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

    async def scrape_entire_pubmed(
        self,
        search_query: str = "*",
        resume_session_id: str | None = None,
        max_total_results: int | None = None,
    ) -> str:
        """
        Scrape the entire PubMed database with checkpoint recovery.

        Args:
            search_query: PubMed search query (default: "*" for all)
            resume_session_id: Session ID to resume from checkpoint
            max_total_results: Maximum results to process (for testing)

        Returns:
            Session ID for tracking progress
        """
        if resume_session_id:
            checkpoint = self._load_checkpoint(resume_session_id)
            if checkpoint:
                logger.info(f"Resuming session {resume_session_id} from checkpoint")
                return await self._resume_scraping(checkpoint)
            logger.warning(
                f"No checkpoint found for session {resume_session_id}, starting new session"
            )

        # Start new scraping session
        session_id = self._generate_session_id()
        logger.info(f"Starting new PubMed scraping session: {session_id}")

        # Get total count
        total_count = await self._get_total_count(search_query)
        if max_total_results:
            total_count = min(total_count, max_total_results)

        logger.info(f"Total abstracts to process: {total_count:,}")

        # Create initial checkpoint
        checkpoint = ScrapingCheckpoint(
            session_id=session_id,
            search_term=search_query,
            total_expected=total_count,
            processed_count=0,
            failed_count=0,
            skipped_count=0,
            current_batch_start=0,
            batch_size=self.batch_size,
            last_update=datetime.now(),
        )

        self._save_checkpoint(checkpoint)
        self.stats["processing_start_time"] = datetime.now()

        try:
            await self._process_in_batches(checkpoint)
        except Exception as e:
            logger.error(f"Scraping session {session_id} failed: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_session(session_id)

        return session_id

    async def _get_total_count(self, search_query: str) -> int:
        """Get total count of results for search query"""
        try:
            handle = Entrez.esearch(db="pubmed", term=search_query, retmax=0)
            record = Entrez.read(handle)
            handle.close()
            return int(record.get("Count", 0))
        except Exception as e:
            logger.error(f"Failed to get total count: {e}")
            raise

    async def _process_in_batches(self, checkpoint: ScrapingCheckpoint):
        """Process PMIDs in batches with concurrent processing"""
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        while (
            checkpoint.current_batch_start < checkpoint.total_expected
            and not self._shutdown_requested
        ):
            # Calculate batch range
            batch_end = min(
                checkpoint.current_batch_start + checkpoint.batch_size,
                checkpoint.total_expected,
            )

            # Create batch tasks with semaphore
            async with semaphore:
                batch_id = (
                    f"{checkpoint.session_id}_batch_{checkpoint.current_batch_start}"
                )

                try:
                    batch_results = await self._process_batch(
                        checkpoint, checkpoint.current_batch_start, batch_end, batch_id
                    )

                    # Update checkpoint
                    checkpoint.processed_count += batch_results["processed"]
                    checkpoint.failed_count += batch_results["failed"]
                    checkpoint.skipped_count += batch_results["skipped"]
                    checkpoint.current_batch_start = batch_end
                    checkpoint.last_update = datetime.now()

                    self._save_checkpoint(checkpoint)

                    # Log progress
                    progress_pct = (
                        checkpoint.processed_count / checkpoint.total_expected
                    ) * 100
                    logger.info(
                        f"Progress: {checkpoint.processed_count:,}/{checkpoint.total_expected:,} "
                        f"({progress_pct:.2f}%) - Batch {batch_id} completed"
                    )

                    # Adaptive rate limiting
                    await self.adaptive_rate_limiter.wait()

                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}", exc_info=True)
                    checkpoint.failed_count += checkpoint.batch_size
                    checkpoint.current_batch_start = batch_end
                    self._save_checkpoint(checkpoint)

    async def _process_batch(
        self,
        checkpoint: ScrapingCheckpoint,
        batch_start: int,
        batch_end: int,
        batch_id: str,
    ) -> dict[str, int]:
        """Process a single batch of PMIDs"""
        batch_start_time = time.time()

        try:
            # Get PMIDs for this batch
            pmids = await self._get_pmid_batch(
                checkpoint.search_term, batch_start, batch_end - batch_start
            )

            if not pmids:
                logger.warning(f"No PMIDs retrieved for batch {batch_id}")
                return {"processed": 0, "failed": 0, "skipped": 0}

            # Filter out already processed PMIDs
            new_pmids = await self._filter_existing_pmids(pmids, checkpoint.session_id)
            skipped_count = len(pmids) - len(new_pmids)

            if not new_pmids:
                logger.info(f"All PMIDs in batch {batch_id} already processed")
                return {"processed": 0, "failed": 0, "skipped": skipped_count}

            # Fetch and process abstracts
            results = await self._fetch_and_process_abstracts(
                new_pmids, checkpoint.session_id
            )

            # Update batch progress
            batch_time = time.time() - batch_start_time
            self._record_batch_progress(
                batch_id,
                checkpoint.session_id,
                batch_start,
                batch_end,
                results,
                batch_time,
            )

            return {
                "processed": results["successful"],
                "failed": results["failed"],
                "skipped": skipped_count,
            }

        except Exception as e:
            logger.error(f"Batch {batch_id} processing failed: {e}", exc_info=True)
            return {"processed": 0, "failed": batch_end - batch_start, "skipped": 0}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(HTTPError),
    )
    async def _get_pmid_batch(
        self, search_term: str, start: int, count: int
    ) -> list[str]:
        """Get batch of PMIDs with retry logic"""
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=search_term,
                retstart=start,
                retmax=count,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            logger.warning(f"Failed to get PMID batch {start}-{start + count}: {e}")
            raise

    async def _filter_existing_pmids(
        self, pmids: list[str], session_id: str
    ) -> list[str]:
        """Filter out PMIDs that already exist in database or processing records"""
        # Check database for existing PMIDs
        existing_in_db = set()
        try:
            result = self.database._execute_cypher(
                "MATCH (a:Abstract) WHERE exists(a.pmid) RETURN a.pmid AS pmid"
            )
            existing_in_db = {record["pmid"] for record in result if record.get("pmid")}
            logger.debug(f"Found {len(existing_in_db)} existing PMIDs in database")
        except Exception as e:
            logger.warning(f"Failed to query existing PMIDs from database: {e}")

        # Check checkpoint database for PMIDs in this session
        cursor = self.checkpoint_db.cursor()
        cursor.execute(
            "SELECT pmid FROM pmid_records WHERE session_id = ? AND state IN (?, ?)",
            (
                session_id,
                ScrapingState.COMPLETED.value,
                ScrapingState.IN_PROGRESS.value,
            ),
        )
        existing_in_session = {row[0] for row in cursor.fetchall()}

        # Return PMIDs that don't exist in either place
        all_existing = existing_in_db.union(existing_in_session)
        return [pmid for pmid in pmids if pmid not in all_existing]

    async def _fetch_and_process_abstracts(
        self, pmids: list[str], session_id: str
    ) -> dict[str, int]:
        """Fetch and process abstracts for a list of PMIDs"""
        results = {"successful": 0, "failed": 0}

        # Mark PMIDs as in progress
        await self._mark_pmids_in_progress(pmids, session_id)

        try:
            # Fetch abstracts from PubMed
            abstracts = await self._fetch_abstracts_batch(pmids)

            # Process each abstract
            for pmid, abstract_data in abstracts.items():
                try:
                    if abstract_data:
                        # Create ProcessedDocument with complete metadata
                        doc = ProcessedDocument(
                            doc_id=f"pubmed_{pmid}",
                            source=abstract_data["abstract"],
                            metadata={
                                "pmid": pmid,
                                "title": abstract_data["title"],
                                "authors": abstract_data["authors"],
                                "journal": abstract_data["journal"],
                                "publication_date": abstract_data["publication_date"],
                                "doi": abstract_data["doi"],
                                "keywords": abstract_data["keywords"],
                                "source": "pubmed_mass_scraper",
                            },
                            chunks=[
                                Chunk(
                                    chunk_id=f"{pmid}_1", text=abstract_data["abstract"]
                                )
                            ],
                        )

                        # Store in database
                        await self._store_document(doc)

                        # Update PMID record
                        await self._mark_pmid_completed(pmid, session_id, abstract_data)
                        results["successful"] += 1
                    else:
                        await self._mark_pmid_failed(
                            pmid, session_id, "No abstract data retrieved"
                        )
                        results["failed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process PMID {pmid}: {e}")
                    await self._mark_pmid_failed(pmid, session_id, str(e))
                    results["failed"] += 1

        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Mark all PMIDs as failed
            for pmid in pmids:
                await self._mark_pmid_failed(pmid, session_id, str(e))
            results["failed"] = len(pmids)

        return results

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((HTTPError, ConnectionError)),
    )
    async def _fetch_abstracts_batch(
        self, pmids: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        """Fetch abstracts for batch of PMIDs with comprehensive error handling"""
        results: dict[str, dict[str, Any] | None] = {}

        try:
            handle = Entrez.efetch(
                db="pubmed", id=pmids, rettype="abstract", retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            for article in records.get("PubmedArticle", []):
                try:
                    pmid = str(article["MedlineCitation"]["PMID"])

                    # Extract abstract
                    abstract_parts = article["MedlineCitation"]["Article"]["Abstract"][
                        "AbstractText"
                    ]
                    abstract_text = " ".join([str(part) for part in abstract_parts])

                    # Extract title
                    title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])

                    # Extract authors
                    authors = []
                    author_list = article["MedlineCitation"]["Article"].get(
                        "AuthorList", []
                    )
                    if author_list:
                        for author in author_list:
                            if isinstance(author, dict):
                                # Handle structured author data
                                last_name = author.get("LastName", "")
                                fore_name = author.get("ForeName", "")
                                if last_name or fore_name:
                                    full_name = f"{fore_name} {last_name}".strip()
                                    authors.append(full_name)
                            else:
                                # Handle string author data
                                authors.append(str(author))

                    # Extract journal information
                    journal_info = article["MedlineCitation"]["Article"].get(
                        "Journal", {}
                    )
                    journal = ""
                    if isinstance(journal_info, dict):
                        journal = journal_info.get("Title", "")

                    # Extract publication date
                    pub_date = ""
                    journal_issue = (
                        journal_info.get("JournalIssue", {})
                        if isinstance(journal_info, dict)
                        else {}
                    )
                    if isinstance(journal_issue, dict):
                        pub_date_info = journal_issue.get("PubDate", {})
                        if isinstance(pub_date_info, dict):
                            year = pub_date_info.get("Year", "")
                            month = pub_date_info.get("Month", "")
                            day = pub_date_info.get("Day", "")
                            pub_date = f"{year} {month} {day}".strip()

                    # Extract DOI
                    doi = ""
                    elocation_ids = article["MedlineCitation"]["Article"].get(
                        "ELocationID", []
                    )
                    if elocation_ids:
                        for eloc_id in elocation_ids:
                            if (
                                isinstance(eloc_id, dict)
                                and eloc_id.get("EIdType") == "doi"
                            ):
                                doi = eloc_id.get("#text", "")
                                break
                            if isinstance(eloc_id, str) and eloc_id.startswith("10."):
                                doi = eloc_id
                                break

                    # Extract keywords
                    keywords = []
                    keyword_list = article["MedlineCitation"].get("KeywordList", [])
                    if keyword_list:
                        for keyword_group in keyword_list:
                            if isinstance(keyword_group, list):
                                for keyword in keyword_group:
                                    if isinstance(keyword, dict):
                                        keywords.append(keyword.get("#text", ""))
                                    else:
                                        keywords.append(str(keyword))

                    results[pmid] = {
                        "abstract": abstract_text,
                        "title": title,
                        "authors": authors,
                        "journal": journal,
                        "publication_date": pub_date,
                        "doi": doi,
                        "keywords": keywords,
                        "abstract_hash": hashlib.md5(
                            abstract_text.encode()
                        ).hexdigest(),
                    }

                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse article data: {e}")
                    if (
                        "MedlineCitation" in article
                        and "PMID" in article["MedlineCitation"]
                    ):
                        pmid = str(article["MedlineCitation"]["PMID"])
                        results[pmid] = None

            # Mark any missing PMIDs as None
            for pmid in pmids:
                if pmid not in results:
                    results[pmid] = None

        except Exception as e:
            logger.error(f"Failed to fetch abstracts for PMIDs {pmids[:5]}...: {e}")
            # Mark all as failed
            for pmid in pmids:
                results[pmid] = None

        return results

    async def _store_document(self, doc: ProcessedDocument):
        """Store processed document in database"""
        try:
            pmid = doc.metadata["pmid"]
            text = doc.source
            title = doc.metadata["title"]
            chunk_id = doc.chunks[0].chunk_id

            # Use MERGE to create or update the abstract node
            query = """
            MERGE (a:Abstract {pmid: $pmid})
            SET a.text = $text, a.title = $title, a.chunk_id = $chunk_id
            """

            self.database._execute_cypher(
                query,
                {"pmid": pmid, "text": text, "title": title, "chunk_id": chunk_id},
            )

            logger.debug(f"Stored Abstract node for PMID {pmid}")
        except Exception as e:
            logger.error(f"Failed to store document {doc.doc_id}: {e}")
            raise

    async def _mark_pmids_in_progress(self, pmids: list[str], session_id: str):
        """Mark PMIDs as in progress in checkpoint database"""
        cursor = self.checkpoint_db.cursor()
        for pmid in pmids:
            cursor.execute(
                """INSERT OR REPLACE INTO pmid_records
                   (pmid, session_id, state, processing_attempts, last_attempt, updated_at)
                   VALUES (?, ?, ?, 1, ?, ?)""",
                (
                    pmid,
                    session_id,
                    ScrapingState.IN_PROGRESS.value,
                    datetime.now(),
                    datetime.now(),
                ),
            )
        self.checkpoint_db.commit()

    async def _mark_pmid_completed(
        self, pmid: str, session_id: str, abstract_data: dict[str, str]
    ):
        """Mark PMID as completed in checkpoint database"""
        cursor = self.checkpoint_db.cursor()
        cursor.execute(
            """UPDATE pmid_records SET
               state = ?, title = ?, abstract_hash = ?, updated_at = ?
               WHERE pmid = ? AND session_id = ?""",
            (
                ScrapingState.COMPLETED.value,
                abstract_data["title"],
                abstract_data["abstract_hash"],
                datetime.now(),
                pmid,
                session_id,
            ),
        )
        self.checkpoint_db.commit()

    async def _mark_pmid_failed(self, pmid: str, session_id: str, error_message: str):
        """Mark PMID as failed in checkpoint database"""
        cursor = self.checkpoint_db.cursor()
        cursor.execute(
            """UPDATE pmid_records SET
               state = ?, error_message = ?, updated_at = ?
               WHERE pmid = ? AND session_id = ?""",
            (
                ScrapingState.FAILED.value,
                error_message,
                datetime.now(),
                pmid,
                session_id,
            ),
        )
        self.checkpoint_db.commit()

    def _record_batch_progress(
        self,
        batch_id: str,
        session_id: str,
        batch_start: int,
        batch_end: int,
        results: dict[str, int],
        processing_time: float,
    ):
        """Record batch progress in checkpoint database"""
        cursor = self.checkpoint_db.cursor()
        cursor.execute(
            """INSERT INTO batch_progress
               (batch_id, session_id, batch_start, batch_end, status, processing_time,
                error_count, success_count, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                session_id,
                batch_start,
                batch_end,
                "completed",
                processing_time,
                results["failed"],
                results["successful"],
                datetime.now(),
            ),
        )
        self.checkpoint_db.commit()

    def _save_checkpoint(self, checkpoint: ScrapingCheckpoint):
        """Save checkpoint to database"""
        cursor = self.checkpoint_db.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO checkpoints
               (session_id, search_term, total_expected, processed_count, failed_count,
                skipped_count, current_batch_start, batch_size, last_update, checkpoint_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.session_id,
                checkpoint.search_term,
                checkpoint.total_expected,
                checkpoint.processed_count,
                checkpoint.failed_count,
                checkpoint.skipped_count,
                checkpoint.current_batch_start,
                checkpoint.batch_size,
                checkpoint.last_update,
                json.dumps(asdict(checkpoint), default=str),
            ),
        )
        self.checkpoint_db.commit()

    def _load_checkpoint(self, session_id: str) -> ScrapingCheckpoint | None:
        """Load checkpoint from database"""
        cursor = self.checkpoint_db.cursor()
        cursor.execute(
            "SELECT checkpoint_data FROM checkpoints WHERE session_id = ?",
            (session_id,),
        )
        result = cursor.fetchone()

        if result:
            checkpoint_data = json.loads(result[0])
            # Convert datetime strings back to datetime objects
            for key, value in checkpoint_data.items():
                if key in ["last_update", "estimated_completion"] and value:
                    checkpoint_data[key] = datetime.fromisoformat(value)
            return ScrapingCheckpoint(**checkpoint_data)
        return None

    async def _resume_scraping(self, checkpoint: ScrapingCheckpoint) -> str:
        """Resume scraping from checkpoint"""
        logger.info(f"Resuming scraping session {checkpoint.session_id}")
        logger.info(
            f"Progress: {checkpoint.processed_count:,}/{checkpoint.total_expected:,}"
        )

        self.stats["processing_start_time"] = datetime.now()

        try:
            await self._process_in_batches(checkpoint)
        except Exception as e:
            logger.error(f"Resume scraping failed: {e}", exc_info=True)
            raise

        return checkpoint.session_id

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"pubmed_scraping_{timestamp}"

    def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        logger.info(f"Cleaning up session {session_id}")
        # Keep checkpoint data for resumption, only clean up memory

    def get_session_progress(self, session_id: str) -> dict[str, Any]:
        """Get detailed progress information for a session"""
        cursor = self.checkpoint_db.cursor()

        # Get checkpoint info
        cursor.execute("SELECT * FROM checkpoints WHERE session_id = ?", (session_id,))
        checkpoint_row = cursor.fetchone()

        if not checkpoint_row:
            return {"error": f"Session {session_id} not found"}

        # Get batch progress
        cursor.execute(
            """SELECT COUNT(*) as total_batches,
               AVG(processing_time) as avg_batch_time,
               SUM(success_count) as total_success,
               SUM(error_count) as total_errors
               FROM batch_progress WHERE session_id = ?""",
            (session_id,),
        )
        batch_stats = cursor.fetchone()

        # Get PMID state distribution
        cursor.execute(
            """SELECT state, COUNT(*) as count
               FROM pmid_records WHERE session_id = ?
               GROUP BY state""",
            (session_id,),
        )
        state_distribution = dict(cursor.fetchall())

        progress_pct = (
            (checkpoint_row[3] / checkpoint_row[2]) * 100
            if checkpoint_row[2] > 0
            else 0
        )

        return {
            "session_id": session_id,
            "search_term": checkpoint_row[1],
            "total_expected": checkpoint_row[2],
            "processed_count": checkpoint_row[3],
            "failed_count": checkpoint_row[4],
            "skipped_count": checkpoint_row[5],
            "progress_percentage": round(progress_pct, 2),
            "current_batch_start": checkpoint_row[6],
            "last_update": checkpoint_row[8],
            "batch_stats": {
                "total_batches": batch_stats[0] or 0,
                "avg_batch_time": round(batch_stats[1] or 0, 2),
                "total_success": batch_stats[2] or 0,
                "total_errors": batch_stats[3] or 0,
            },
            "pmid_state_distribution": state_distribution,
        }

    def get_scraped_documents(self, session_id: str) -> list[ProcessedDocument]:
        """Retrieve successfully scraped documents from a session for KG processing"""
        cursor = self.checkpoint_db.cursor()

        # Get all successfully completed PMIDs with their abstracts
        cursor.execute(
            """SELECT pmid, title, abstract_hash FROM pmid_records
               WHERE session_id = ? AND state = ? AND title IS NOT NULL
               ORDER BY updated_at""",
            (session_id, ScrapingState.COMPLETED.value),
        )

        completed_records = cursor.fetchall()
        logger.info(
            f"Found {len(completed_records)} completed abstracts in session {session_id}"
        )

        documents = []
        for pmid, title, abstract_hash in completed_records:
            try:
                # Retrieve the actual abstract text and metadata from the database
                doc_data = self._get_document_data_from_db(pmid)
                if not doc_data or not doc_data.get("abstract"):
                    logger.warning(
                        f"No document data found in database for PMID {pmid}, skipping"
                    )
                    continue

                # Create a proper ProcessedDocument with the actual abstract text and metadata
                doc = ProcessedDocument(
                    doc_id=f"pubmed_{pmid}",
                    source=doc_data["abstract"],  # Use the actual abstract text
                    metadata={
                        "pmid": pmid,
                        "title": doc_data.get("title", title or f"PMID {pmid}"),
                        "authors": doc_data.get("authors", []),
                        "journal": doc_data.get("journal", ""),
                        "publication_date": doc_data.get("publication_date", ""),
                        "doi": doc_data.get("doi", ""),
                        "keywords": doc_data.get("keywords", []),
                        "source": "pubmed_mass_scraper",
                        "abstract_hash": abstract_hash,
                    },
                    chunks=[
                        Chunk(
                            chunk_id=f"pubmed_{pmid}_chunk_0",
                            text=doc_data["abstract"],  # Use the actual abstract text
                        )
                    ],
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to create document for PMID {pmid}: {e}")
                continue

        logger.info(
            f"Successfully created {len(documents)} ProcessedDocument objects with full abstract text and metadata"
        )
        return documents

    def _get_abstract_text_from_db(self, pmid: str) -> str | None:
        """Retrieve abstract text from the database for a given PMID"""
        try:
            result = self.database._execute_cypher(
                "MATCH (a:Abstract {pmid: $pmid}) RETURN a.text AS text", {"pmid": pmid}
            )

            if result and len(result) > 0:
                return result[0].get("text")

            logger.warning(f"No abstract text found in database for PMID {pmid}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve abstract text for PMID {pmid}: {e}")
            return None

    def _get_document_data_from_db(self, pmid: str) -> dict[str, Any] | None:
        """Retrieve complete document data including metadata from the database for a given PMID"""
        try:
            result = self.database._execute_cypher(
                """MATCH (a:Abstract {pmid: $pmid})
                   RETURN a.text AS text, a.title AS title, a.authors AS authors,
                          a.journal AS journal, a.publication_date AS publication_date,
                          a.doi AS doi, a.keywords AS keywords""",
                {"pmid": pmid},
            )

            if result and len(result) > 0:
                data = result[0]
                authors = data.get("authors", [])
                keywords = data.get("keywords", [])

                return {
                    "abstract": data.get("text", ""),
                    "title": data.get("title", ""),
                    "authors": authors,
                    "journal": data.get("journal", ""),
                    "publication_date": data.get("publication_date", ""),
                    "doi": data.get("doi", ""),
                    "keywords": keywords,
                }

            logger.warning(f"No document data found in database for PMID {pmid}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve document data for PMID {pmid}: {e}")
            return None

    def close(self):
        """Close all connections and cleanup resources"""
        if self.checkpoint_db:
            self.checkpoint_db.close()


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts delays based on API response times"""

    def __init__(self, base_delay: float = 0.5, max_delay: float = 10.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.current_delay = base_delay
        self.response_times: list[float] = []
        self.error_count = 0

    async def wait(self):
        """Wait based on current delay"""
        await asyncio.sleep(self.current_delay)

    def record_success(self, response_time: float):
        """Record successful API call"""
        self.response_times.append(response_time)
        self.error_count = 0

        # Keep only recent response times
        if len(self.response_times) > 10:
            self.response_times = self.response_times[-10:]

        # Adjust delay based on response times
        avg_response_time = sum(self.response_times) / len(self.response_times)
        if avg_response_time > 5.0:  # Slow responses
            self.current_delay = min(self.current_delay * 1.2, self.max_delay)
        elif avg_response_time < 1.0:  # Fast responses
            self.current_delay = max(self.current_delay * 0.9, self.base_delay)

    def record_error(self):
        """Record API error"""
        self.error_count += 1
        self.current_delay = min(self.current_delay * 2.0, self.max_delay)


class CircuitBreaker:
    """Circuit breaker pattern for handling API failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "closed"  # closed, open, half-open

    def can_proceed(self) -> bool:
        """Check if operation can proceed"""
        if self.state == "closed":
            return True
        if self.state == "open":
            if (
                self.last_failure_time
                and (datetime.now() - self.last_failure_time).seconds
                >= self.recovery_timeout
            ):
                self.state = "half-open"
                return True
            return False
        # half-open
        return True

    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Factory function for creating scraper instances
def create_pubmed_scraper(database: DatabaseInterface, **kwargs) -> PubMedMassScraper:
    """Factory function for creating PubMed scraper instances"""
    return PubMedMassScraper(database, **kwargs)
