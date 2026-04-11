#!/usr/bin/env python3
"""
Open Targets Platform API Integrator

Fetches and ingests drug-target-disease associations from Open Targets Platform via GraphQL API.
Open Targets provides evidence-based target-disease associations with drug information.

API endpoint: https://api.platform.opentargets.org/api/v4/graphql
"""

import logging
import asyncio
import time
from typing import Any

import aiohttp

from pipeline.interfaces import GraphStore
logger = logging.getLogger(__name__)
class OpenTargetsIntegrator:
    """Integrates Open Targets drug-target-disease associations via GraphQL API."""

    API_BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    DEFAULT_RATE_LIMIT = 1.0  # seconds between requests

    def __init__(
        self,
        backend: str = "neo4j",
        database: str = "olink1",
        db: GraphStore | None = None,
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ):
        """
        Initialize Open Targets integrator.

        Args:
            backend: Database backend ("neo4j" or "neptune")
            database: Database name
            db: Optional GraphStore instance
            rate_limit: Delay between API requests in seconds
        """
        self.backend = backend
        self.database = database
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

        if backend == "neptune":
            from archive.neptune_legacy.core.opencypher_query_executor import (
                create_opencypher_executor,
            )

            self.executor = create_opencypher_executor(database_type="neptune")
            self.db = None
        else:
            if db is not None:
                self.db = db
            else:
                from pipeline.backends.neo4j_store import Neo4jGraphStore
                self.db = Neo4jGraphStore(database=database)
            self.executor = None

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query using appropriate backend."""
        if self.backend == "neptune":
            return self.executor.execute_query(query, parameters)
        return self.db.execute_query(query, parameters)

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def fetch_known_drugs_for_target(
        self, ensembl_id: str, session: aiohttp.ClientSession
    ) -> dict[str, Any]:
        """
        Fetch known drugs for a target from Open Targets API.

        Args:
            ensembl_id: Ensembl gene ID (e.g., "ENSG00000142192")
            session: aiohttp ClientSession

        Returns:
            Dictionary with target info and known drugs
        """
        await self._enforce_rate_limit()

        query = """
        query findKnownDrugsForTarget($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            knownDrugs {
              count
              rows {
                drug {
                  id
                  name
                  isApproved
                  drugType
                }
                disease {
                  id
                  name
                }
                phase
                status
                mechanismOfAction
              }
            }
          }
        }
        """

        variables = {"ensemblId": ensembl_id}
        payload = {"query": query, "variables": variables}

        try:
            async with session.post(self.API_BASE_URL, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                if "errors" in data:
                    logger.error(f"GraphQL errors for {ensembl_id}: {data['errors']}")
                    return {"target": None, "drugs": []}

                target_data = data.get("data", {}).get("target")
                if not target_data:
                    logger.warning(f"No data found for target {ensembl_id}")
                    return {"target": None, "drugs": []}

                return {
                    "target": {
                        "ensembl_id": target_data["id"],
                        "symbol": target_data.get("approvedSymbol", ""),
                        "name": target_data.get("approvedName", ""),
                    },
                    "drugs": target_data.get("knownDrugs", {}).get("rows", []),
                }

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching target {ensembl_id}: {e}")
            return {"target": None, "drugs": []}

    async def batch_fetch_targets(self, ensembl_ids: list[str]) -> list[dict[str, Any]]:
        """
        Fetch known drugs for multiple targets in batch.

        Args:
            ensembl_ids: List of Ensembl gene IDs

        Returns:
            List of target-drug associations
        """
        logger.info(f"Fetching known drugs for {len(ensembl_ids)} targets")

        results = []
        async with aiohttp.ClientSession() as session:
            for ensembl_id in ensembl_ids:
                result = await self.fetch_known_drugs_for_target(ensembl_id, session)
                if result["target"]:
                    results.append(result)

        logger.info(f"Fetched data for {len(results)} targets")
        return results

    def ingest_target_drug_associations(
        self, target_data: dict[str, Any]
    ) -> dict[str, int]:
        """
        Ingest drug-target-disease associations for a single target.

        Args:
            target_data: Dictionary with target info and drugs list

        Returns:
            Dictionary with ingestion counts
        """
        target = target_data["target"]
        drugs = target_data["drugs"]

        if not target or not drugs:
            return {"targets": 0, "drugs": 0, "diseases": 0, "relationships": 0}

        counts = {"targets": 0, "drugs": 0, "diseases": 0, "relationships": 0}

        # Create target node
        target_query = """
        MERGE (t:Protein {ensembl_id: $ensembl_id})
        ON CREATE SET
            t.gene_symbol = $symbol,
            t.name = $name,
            t.source = 'OpenTargets',
            t.created_at = datetime()
        ON MATCH SET
            t.gene_symbol = $symbol,
            t.name = $name
        RETURN t
        """
        result = self._execute_query(
            target_query,
            {
                "ensembl_id": target["ensembl_id"],
                "symbol": target["symbol"],
                "name": target["name"],
            },
        )
        if result:
            counts["targets"] += 1

        # Process each drug-disease association
        for drug_assoc in drugs:
            drug_info = drug_assoc.get("drug", {})
            disease_info = drug_assoc.get("disease", {})

            if not drug_info or not disease_info:
                continue

            # Create drug node
            drug_query = """
            MERGE (d:Drug {chembl_id: $drug_id})
            ON CREATE SET
                d.name = $name,
                d.is_approved = $is_approved,
                d.drug_type = $drug_type,
                d.source = 'OpenTargets',
                d.created_at = datetime()
            ON MATCH SET
                d.name = $name,
                d.is_approved = $is_approved,
                d.drug_type = $drug_type
            RETURN d
            """
            result = self._execute_query(
                drug_query,
                {
                    "drug_id": drug_info["id"],
                    "name": drug_info.get("name", ""),
                    "is_approved": drug_info.get("isApproved", False),
                    "drug_type": drug_info.get("drugType", ""),
                },
            )
            if result:
                counts["drugs"] += 1

            # Create disease node
            disease_query = """
            MERGE (dis:Disease {efo_id: $disease_id})
            ON CREATE SET
                dis.name = $name,
                dis.source = 'OpenTargets',
                dis.created_at = datetime()
            ON MATCH SET
                dis.name = $name
            RETURN dis
            """
            result = self._execute_query(
                disease_query,
                {
                    "disease_id": disease_info["id"],
                    "name": disease_info.get("name", ""),
                },
            )
            if result:
                counts["diseases"] += 1

            # Create drug-target relationship
            drug_target_query = """
            MATCH (d:Drug {chembl_id: $drug_id})
            MATCH (t:Protein {ensembl_id: $ensembl_id})
            MERGE (d)-[r:TARGETS]->(t)
            ON CREATE SET
                r.phase = $phase,
                r.status = $status,
                r.mechanism_of_action = $moa,
                r.source = 'OpenTargets',
                r.created_at = datetime()
            ON MATCH SET
                r.phase = $phase,
                r.status = $status,
                r.mechanism_of_action = $moa
            RETURN r
            """
            self._execute_query(
                drug_target_query,
                {
                    "drug_id": drug_info["id"],
                    "ensembl_id": target["ensembl_id"],
                    "phase": drug_assoc.get("phase"),
                    "status": drug_assoc.get("status", ""),
                    "moa": drug_assoc.get("mechanismOfAction", ""),
                },
            )

            # Create drug-disease relationship
            drug_disease_query = """
            MATCH (d:Drug {chembl_id: $drug_id})
            MATCH (dis:Disease {efo_id: $disease_id})
            MERGE (d)-[r:TREATS]->(dis)
            ON CREATE SET
                r.phase = $phase,
                r.status = $status,
                r.source = 'OpenTargets',
                r.created_at = datetime()
            ON MATCH SET
                r.phase = $phase,
                r.status = $status
            RETURN r
            """
            result = self._execute_query(
                drug_disease_query,
                {
                    "drug_id": drug_info["id"],
                    "disease_id": disease_info["id"],
                    "phase": drug_assoc.get("phase"),
                    "status": drug_assoc.get("status", ""),
                },
            )
            if result:
                counts["relationships"] += 1

        return counts

    async def ingest_from_targets(self, ensembl_ids: list[str]) -> dict[str, Any]:
        """
        Complete ingestion workflow for multiple targets.

        Args:
            ensembl_ids: List of Ensembl gene IDs

        Returns:
            Dictionary with complete statistics
        """
        # Fetch data from API
        target_data_list = await self.batch_fetch_targets(ensembl_ids)

        # Ingest into graph
        total_counts = {"targets": 0, "drugs": 0, "diseases": 0, "relationships": 0}

        for target_data in target_data_list:
            counts = self.ingest_target_drug_associations(target_data)
            for key in total_counts:
                total_counts[key] += counts[key]

        logger.info(f"Ingestion complete: {total_counts}")
        return {
            "targets_fetched": len(target_data_list),
            "ingestion_stats": total_counts,
        }


async def main():
    """CLI entry point for Open Targets integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest Open Targets drug-target-disease associations"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Ensembl gene IDs (e.g., ENSG00000142192)",
    )
    parser.add_argument("--database", default="olink1", help="Database name")
    parser.add_argument(
        "--backend",
        default="neo4j",
        choices=["neo4j", "neptune"],
        help="Database backend",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between API requests (seconds)",
    )

    args = parser.parse_args()

    integrator = OpenTargetsIntegrator(
        backend=args.backend, database=args.database, rate_limit=args.rate_limit
    )
    stats = await integrator.ingest_from_targets(args.targets)

    logger.info(f"Open Targets integration complete: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
