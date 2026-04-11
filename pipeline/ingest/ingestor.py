import logging
import time
from dataclasses import dataclass, field

from Bio import Entrez

from pipeline.interfaces import GraphStore
from pipeline.config import PipelineConfig as Config
logger = logging.getLogger(__name__)
Entrez.email = Config.get_config("ENTREZ_EMAIL")
Entrez.api_key = Config.get_config("ENTREZ_API_KEY")


@dataclass
class Chunk:
    chunk_id: str
    text: str  # Only chunk-specific text, not full abstract
    pmid: str | None = None
    title: str | None = None
    publication_year: int | None = None
    embedding: list[float] | None = None
    nodes: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    doc_id: str
    source: str
    metadata: dict
    chunks: list[Chunk] = field(default_factory=list)


class Ingestor:
    def __init__(self, graph_store: GraphStore | None = None):
        """Initialize ingestor with Entrez email and graph store."""
        self.email = Config.get_config("ENTREZ_EMAIL") or "your.email@example.com"
        Entrez.email = self.email
        Entrez.api_key = Config.get_config("ENTREZ_API_KEY")
        if graph_store is None:
            from pipeline.backends.neo4j_store import Neo4jGraphStore
            graph_store = Neo4jGraphStore()
        self.db = graph_store
        logger.info("Initialized Ingestor with graph store")

    def _get_existing_pmids(self) -> set[str]:
        """Query Neo4j for existing PMIDs."""
        return self.db.get_existing_pmids()

    def _store_abstract_node(
        self,
        pmid: str,
        text: str,
        title: str,
        chunk_id: str,
        publication_year: int | None = None,
    ):
        """Store Abstract node in Neo4j."""
        if publication_year is not None:
            self.db.store_abstract_node(pmid, text, title, chunk_id, publication_year)
        else:
            self.db.store_abstract_node(pmid, text, title, chunk_id)

    def fetch_pubmed_abstracts(
        self, search_terms: list[str], max_results: int = 1000
    ) -> list[ProcessedDocument]:
        """Fetch new PubMed abstracts, excluding existing PMIDs."""
        logger.info(f"Fetching abstracts for terms: {search_terms}")
        query = " AND ".join([f"{term}[TIAB]" for term in search_terms])
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]
        logger.info(f"Found {len(pmids)} PMIDs")

        # Filter out existing PMIDs
        existing_pmids = self._get_existing_pmids()
        new_pmids = [pmid for pmid in pmids if pmid not in existing_pmids]
        logger.info(f"After filtering, {len(new_pmids)} new PMIDs remain")

        if not new_pmids:
            logger.info("No new abstracts to fetch.")
            return []

        documents = []
        batch_size = 100
        for start in range(0, len(new_pmids), batch_size):
            end = min(start + batch_size, len(new_pmids))
            batch_pmids = new_pmids[start:end]
            logger.info(f"Fetching batch: {start} to {end}")
            retries = 3
            for attempt in range(retries):
                try:
                    handle = Entrez.efetch(db="pubmed", id=batch_pmids, retmode="xml")
                    records = Entrez.read(handle)
                    handle.close()
                    break
                except Exception as e:
                    logger.warning(
                        f"Entrez batch fetch failed (attempt {attempt + 1}/{retries}): {e}"
                    )
                    time.sleep(2)
                    if attempt == retries - 1:
                        logger.error(f"Failed to fetch batch {start}-{end}")
                        records = {"PubmedArticle": []}
            for article in records["PubmedArticle"]:
                try:
                    abstract_parts = article["MedlineCitation"]["Article"]["Abstract"][
                        "AbstractText"
                    ]
                    abstract_text = " ".join([str(part) for part in abstract_parts])
                    title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                    pmid = str(article["MedlineCitation"]["PMID"])

                    # Extract publication year
                    pub_year = None
                    try:
                        # Try to get year from PubDate
                        pub_date = article["MedlineCitation"]["Article"]["Journal"][
                            "JournalIssue"
                        ]["PubDate"]
                        if "Year" in pub_date:
                            pub_year = int(pub_date["Year"])
                        elif "MedlineDate" in pub_date:
                            # Extract year from MedlineDate format like "2020 Jan-Feb"
                            import re

                            year_match = re.search(
                                r"\b(19|20)\d{2}\b", pub_date["MedlineDate"]
                            )
                            if year_match:
                                pub_year = int(year_match.group())
                    except (KeyError, ValueError, TypeError) as year_error:
                        logger.debug(
                            f"Could not extract publication year for PMID {pmid}: {year_error}"
                        )
                        pub_year = None

                    chunk_id = f"{pmid}_1"
                    doc = ProcessedDocument(
                        doc_id=f"pubmed_{pmid}",
                        source=abstract_text,
                        metadata={
                            "pmid": pmid,
                            "title": title,
                            "publication_year": pub_year,
                        },
                        chunks=[
                            Chunk(
                                chunk_id=chunk_id,
                                text=f"Abstract from PMID {pmid}",
                                pmid=pmid,
                                title=title,
                                publication_year=pub_year,
                            )
                        ],
                    )
                    # Store Abstract node in Neo4j with publication year
                    self._store_abstract_node(
                        pmid, abstract_text, title, chunk_id, pub_year
                    )
                    documents.append(doc)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping article PMID {pmid}: {e}")
            time.sleep(0.1)
        logger.info(f"Collected {len(documents)} new documents")
        return documents

    def fetch_abstracts(
        self, search_term: str, max_results: int = 10
    ) -> list[ProcessedDocument]:
        """Fetch new abstracts from Entrez, excluding existing PMIDs."""
        try:
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            pmids = record["IdList"]
            logger.info(f"Found {len(pmids)} PMIDs for search term: {search_term}")

            # Filter out existing PMIDs
            existing_pmids = self._get_existing_pmids()
            new_pmids = [pmid for pmid in pmids if pmid not in existing_pmids]
            logger.info(f"After filtering, {len(new_pmids)} new PMIDs")

            if not new_pmids:
                logger.info("No new abstracts found.")
                return []

            docs = []
            handle = Entrez.efetch(
                db="pubmed", id=new_pmids, rettype="abstract", retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            for article in records["PubmedArticle"]:
                try:
                    abstract_parts = article["MedlineCitation"]["Article"]["Abstract"][
                        "AbstractText"
                    ]
                    abstract_text = " ".join([str(part) for part in abstract_parts])
                    title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                    pmid = str(article["MedlineCitation"]["PMID"])

                    # Extract publication year
                    pub_year = None
                    try:
                        # Try to get year from PubDate
                        pub_date = article["MedlineCitation"]["Article"]["Journal"][
                            "JournalIssue"
                        ]["PubDate"]
                        if "Year" in pub_date:
                            pub_year = int(pub_date["Year"])
                        elif "MedlineDate" in pub_date:
                            # Extract year from MedlineDate format like "2020 Jan-Feb"
                            import re

                            year_match = re.search(
                                r"\b(19|20)\d{2}\b", pub_date["MedlineDate"]
                            )
                            if year_match:
                                pub_year = int(year_match.group())
                    except (KeyError, ValueError, TypeError) as year_error:
                        logger.debug(
                            f"Could not extract publication year for PMID {pmid}: {year_error}"
                        )
                        pub_year = None

                    chunk_id = f"{pmid}_1"
                    doc = ProcessedDocument(
                        doc_id=f"pubmed_{pmid}",
                        source=abstract_text,
                        metadata={
                            "pmid": pmid,
                            "title": title,
                            "publication_year": pub_year,
                        },
                        chunks=[
                            Chunk(
                                chunk_id=chunk_id,
                                text=f"Abstract from PMID {pmid}",
                                pmid=pmid,
                                title=title,
                                publication_year=pub_year,
                            )
                        ],
                    )
                    # Store Abstract node in Neo4j with publication year
                    self._store_abstract_node(
                        pmid, abstract_text, title, chunk_id, pub_year
                    )
                    docs.append(doc)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping article PMID {pmid}: {e}")
            logger.info(f"Collected {len(docs)} new documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to fetch abstracts: {e}")
            return []

    def close(self):
        """Close database connection."""
        self.db.close()
        logger.info("Closed Ingestor database connection")
