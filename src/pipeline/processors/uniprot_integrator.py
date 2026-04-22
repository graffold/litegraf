#!/usr/bin/env python3
"""
UniProt API integration for enriching protein nodes with functional annotations.

Fetches protein annotations from UniProt REST API and adds them to the knowledge graph.
Supports batch processing with rate limiting and retry logic.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class UniProtAnnotation:
    """UniProt protein annotation data."""

    uniprot_id: str
    protein_name: str | None = None
    gene_names: list[str] | None = None
    organism: str | None = None
    function: str | None = None
    subcellular_location: list[str] | None = None
    keywords: list[str] | None = None
    go_terms: dict[str, list[str]] | None = None  # category -> list of terms
    pathway: list[str] | None = None
    disease_involvement: list[str] | None = None
    sequence_length: int | None = None


class UniProtIntegrator:
    """Fetch and integrate UniProt functional annotations into knowledge graph."""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(
        self,
        db: Any,
        rate_limit: float = 0.2,  # 5 requests/sec (UniProt allows 10/sec)
        batch_size: int = 100,
    ):
        """
        Initialize UniProt integrator.

        Args:
            db: Database instance (Neo4j or Neptune)
            rate_limit: Delay between requests in seconds (default 0.2s = 5 req/sec)
            batch_size: Number of proteins to process in parallel (default 100)
        """
        self.db = db
        self.rate_limit = rate_limit
        self.batch_size = batch_size
        self._last_request_time = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    async def _fetch_annotation(
        self, session: aiohttp.ClientSession, uniprot_id: str
    ) -> UniProtAnnotation | None:
        """
        Fetch annotation for a single UniProt ID.

        Args:
            session: aiohttp ClientSession
            uniprot_id: UniProt accession ID

        Returns:
            UniProtAnnotation or None if not found
        """
        # Rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self._last_request_time = asyncio.get_event_loop().time()

        url = f"{self.BASE_URL}/{uniprot_id}.json"
        logger.debug(f"Fetching UniProt annotation for {uniprot_id}")

        async with session.get(url) as response:
            if response.status == 404:
                logger.warning(f"UniProt ID not found: {uniprot_id}")
                return None
            if response.status != 200:
                logger.error(f"UniProt API error for {uniprot_id}: {response.status}")
                response.raise_for_status()

            data = await response.json()
            return self._parse_annotation(uniprot_id, data)

    def _parse_annotation(
        self, uniprot_id: str, data: dict[str, Any]
    ) -> UniProtAnnotation:
        """Parse UniProt JSON response into annotation object."""
        # Protein name
        protein_name = None
        if "proteinDescription" in data:
            rec_name = data["proteinDescription"].get("recommendedName", {})
            protein_name = rec_name.get("fullName", {}).get("value")

        # Gene names
        gene_names = []
        if "genes" in data:
            for gene in data["genes"]:
                if "geneName" in gene:
                    gene_names.append(gene["geneName"]["value"])

        # Organism
        organism = None
        if "organism" in data:
            organism = data["organism"].get("scientificName")

        # Function
        function = None
        if "comments" in data:
            for comment in data["comments"]:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function = texts[0].get("value")
                    break

        # Subcellular location
        subcellular_location = []
        if "comments" in data:
            for comment in data["comments"]:
                if comment.get("commentType") == "SUBCELLULAR LOCATION":
                    for loc in comment.get("subcellularLocations", []):
                        if "location" in loc:
                            subcellular_location.append(loc["location"]["value"])

        # Keywords
        keywords = []
        if "keywords" in data:
            keywords = [kw["name"] for kw in data["keywords"]]

        # GO terms
        go_terms: dict[str, list[str]] = {
            "biological_process": [],
            "molecular_function": [],
            "cellular_component": [],
        }
        if "uniProtKBCrossReferences" in data:
            for xref in data["uniProtKBCrossReferences"]:
                if xref.get("database") == "GO":
                    props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                    term = props.get("GoTerm", "")
                    props.get("GoEvidenceType", "")
                    if "P:" in term:
                        go_terms["biological_process"].append(term)
                    elif "F:" in term:
                        go_terms["molecular_function"].append(term)
                    elif "C:" in term:
                        go_terms["cellular_component"].append(term)

        # Pathway
        pathway = []
        if "comments" in data:
            for comment in data["comments"]:
                if comment.get("commentType") == "PATHWAY":
                    texts = comment.get("texts", [])
                    if texts:
                        pathway.append(texts[0].get("value", ""))

        # Disease involvement
        disease_involvement = []
        if "comments" in data:
            for comment in data["comments"]:
                if comment.get("commentType") == "DISEASE":
                    disease_name = comment.get("disease", {}).get("diseaseId", "")
                    if disease_name:
                        disease_involvement.append(disease_name)

        # Sequence length
        sequence_length = None
        if "sequence" in data:
            sequence_length = data["sequence"].get("length")

        return UniProtAnnotation(
            uniprot_id=uniprot_id,
            protein_name=protein_name,
            gene_names=gene_names,
            organism=organism,
            function=function,
            subcellular_location=subcellular_location,
            keywords=keywords,
            go_terms=go_terms,
            pathway=pathway,
            disease_involvement=disease_involvement,
            sequence_length=sequence_length,
        )

    async def fetch_annotations(
        self, uniprot_ids: list[str]
    ) -> dict[str, UniProtAnnotation]:
        """
        Fetch annotations for multiple UniProt IDs in parallel batches.

        Args:
            uniprot_ids: List of UniProt accession IDs

        Returns:
            Dict mapping UniProt ID to annotation (excludes not found)
        """
        annotations: dict[str, UniProtAnnotation] = {}
        total = len(uniprot_ids)

        async with aiohttp.ClientSession() as session:
            for i in range(0, total, self.batch_size):
                batch = uniprot_ids[i : i + self.batch_size]
                logger.info(
                    f"Fetching batch {i // self.batch_size + 1}/{(total + self.batch_size - 1) // self.batch_size} "
                    f"({len(batch)} proteins)"
                )

                tasks = [
                    self._fetch_annotation(session, uniprot_id) for uniprot_id in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for uniprot_id, result in zip(batch, results, strict=False):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {uniprot_id}: {result}")
                    elif result is not None:
                        annotations[uniprot_id] = result

        logger.info(f"Fetched {len(annotations)}/{total} annotations successfully")
        return annotations

    def store_annotation(self, annotation: UniProtAnnotation) -> None:
        """
        Store annotation in knowledge graph.

        Args:
            annotation: UniProtAnnotation to store
        """
        # Build properties dict
        props = {
            "uniprot_id": annotation.uniprot_id,
            "source": "UniProt",
        }
        if annotation.protein_name:
            props["protein_name"] = annotation.protein_name
        if annotation.gene_names:
            props["gene_names"] = annotation.gene_names
        if annotation.organism:
            props["organism"] = annotation.organism
        if annotation.function:
            props["function"] = annotation.function
        if annotation.subcellular_location:
            props["subcellular_location"] = annotation.subcellular_location
        if annotation.keywords:
            props["keywords"] = annotation.keywords
        if annotation.go_terms:
            for category, terms in annotation.go_terms.items():
                if terms:
                    props[f"go_{category}"] = terms
        if annotation.pathway:
            props["pathway"] = annotation.pathway
        if annotation.disease_involvement:
            props["disease_involvement"] = annotation.disease_involvement
        if annotation.sequence_length:
            props["sequence_length"] = annotation.sequence_length

        # Update protein node
        query = """
        MATCH (p:Protein)
        WHERE p.uniprot_id = $uniprot_id OR p.uniprotID = $uniprot_id
        SET p += $props
        RETURN p.uniprot_id as id
        """
        params = {"uniprot_id": annotation.uniprot_id, "props": props}

        result = self.db.execute_query(query, params)
        if result:
            logger.debug(f"Updated protein {annotation.uniprot_id}")
        else:
            logger.warning(
                f"Protein node not found for UniProt ID: {annotation.uniprot_id}"
            )

    async def enrich_proteins(self, uniprot_ids: list[str] | None = None) -> int:
        """
        Enrich protein nodes with UniProt annotations.

        Args:
            uniprot_ids: List of UniProt IDs to enrich (if None, fetch all from graph)

        Returns:
            Number of proteins enriched
        """
        # Get UniProt IDs from graph if not provided
        if uniprot_ids is None:
            logger.info("Fetching UniProt IDs from graph...")
            query = """
            MATCH (p:Protein)
            WHERE p.uniprot_id IS NOT NULL OR p.uniprotID IS NOT NULL
            RETURN COALESCE(p.uniprot_id, p.uniprotID) as uniprot_id
            """
            result = self.db.execute_query(query)
            uniprot_ids = [row["uniprot_id"] for row in result]
            logger.info(f"Found {len(uniprot_ids)} proteins with UniProt IDs")

        if not uniprot_ids:
            logger.warning("No UniProt IDs to enrich")
            return 0

        # Fetch annotations
        annotations = await self.fetch_annotations(uniprot_ids)

        # Store annotations
        logger.info(f"Storing {len(annotations)} annotations in graph...")
        for annotation in annotations.values():
            self.store_annotation(annotation)

        logger.info(f"Enriched {len(annotations)} proteins with UniProt data")
        return len(annotations)


async def main():
    """CLI entry point for UniProt enrichment."""
    import argparse
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from pipeline.dx.registry import BackendRegistry

    parser = argparse.ArgumentParser(
        description="Enrich protein nodes with UniProt functional annotations"
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Database name (default: neo4j)",
    )
    parser.add_argument(
        "--backend",
        default="neo4j",
        choices=["neo4j"],
        help="Database backend (default: neo4j)",
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j user (default: neo4j)",
    )
    parser.add_argument(
        "--password",
        help="Neo4j password (reads from NEO4J_PASSWORD env if not provided)",
    )
    parser.add_argument(
        "--uniprot-ids",
        nargs="+",
        help="Specific UniProt IDs to enrich (if not provided, enriches all)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.2,
        help="Delay between API requests in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of proteins to process in parallel (default: 100)",
    )

    args = parser.parse_args()

    # Get password from env if not provided
    if not args.password:
        import os

        args.password = os.getenv("NEO4J_PASSWORD")
        if not args.password:
            logger.error("Neo4j password required (--password or NEO4J_PASSWORD env)")
            sys.exit(1)

    # Connect to database
    logger.info(f"Connecting to {args.backend} database: {args.database}")
    db = BackendRegistry.resolve_graph_store(args.backend, uri=args.uri, user=args.user, password=args.password, database=args.database)

    # Create integrator
    integrator = UniProtIntegrator(
        db=db,
        rate_limit=args.rate_limit,
        batch_size=args.batch_size,
    )

    # Enrich proteins
    count = await integrator.enrich_proteins(uniprot_ids=args.uniprot_ids)
    logger.info(f"Successfully enriched {count} proteins")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
