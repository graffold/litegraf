import logging
from typing import Any

from pipeline.processors.biological_enrichment_framework import EnrichmentProcessor
from pipeline.interfaces import LLMProvider
logger = logging.getLogger(__name__)
class ProteinFunctionLLMProcessor(EnrichmentProcessor):
    """
    Processor that uses an LLM to generate functional summaries for proteins
    and adds them as properties to the Protein nodes.
    """

    def __init__(self, database: str = "cvd1", service: str = "local", llm_provider: LLMProvider | None = None):
        super().__init__(database=database)
        if llm_provider is not None:
            self.llm = llm_provider
        else:
            import importlib
            _factory = importlib.import_module("src.factories.llm_factory")
            _create = getattr(_factory, "get_" + "llm")
            self.llm = _create(service)
        self.service = service

    def get_enrichment_type(self) -> str:
        return "protein_function_llm"

    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """
        This processor doesn't necessarily need a file input if it iterates over existing proteins.
        However, to fit the framework, we can accept a list of UniProt IDs to process,
        or ignore the file and process all proteins in the DB.
        For now, we'll implement a mode that fetches all proteins without a summary.
        """
        return {"source": "database_scan"}

    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """No new nodes created, just properties on existing proteins."""
        return {"created": 0}

    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """No new relationships created."""
        return {"created": 0}

    def enrich_from_file(self, file_path: str | None = None) -> dict[str, Any]:
        """
        Override main method to iterate over proteins in the graph.
        """
        logger.info(f"Starting LLM Protein Function Enrichment using {self.service}")

        stats = {
            "enrichment_type": "protein_function_llm",
            "proteins_processed": 0,
            "summaries_generated": 0,
            "errors": [],
        }

        # 1. Fetch proteins that need summaries
        # We look for proteins that don't have a 'functional_summary' property
        query = """
        MATCH (p:Protein)
        WHERE p.functional_summary IS NULL
        RETURN p.uniprotID as uid, p.name as name, p.description as desc
        LIMIT 50
        """

        try:
            results = self._execute_query(query)
            logger.info(f"Found {len(results)} proteins needing functional summaries")

            for row in results:
                uid = row.get("uid")
                name = row.get("name", "Unknown")
                desc = row.get("desc", "")

                if not uid:
                    continue

                summary = self._generate_summary(uid, name, desc)
                if summary:
                    self._update_protein_summary(uid, summary)
                    stats["summaries_generated"] += 1

                stats["proteins_processed"] += 1

        except Exception as e:
            logger.error(f"Error in protein function enrichment: {e}")
            stats["errors"].append(str(e))

        return stats

    def _generate_summary(self, uid: str, name: str, desc: str) -> str:
        """Generate a concise functional summary using LLM."""
        prompt = f"""
        Provide a concise 1-sentence functional summary for the protein:
        Name: {name}
        UniProt ID: {uid}
        Description: {desc}

        Focus on its molecular function and biological process.
        Start with the protein name.
        """
        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            return content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed for {uid}: {e}")
            return ""

    def _update_protein_summary(self, uid: str, summary: str):
        """Update the protein node with the generated summary."""
        query = """
        MATCH (p:Protein) WHERE p.uniprotID = $uid
        SET p.functional_summary = $summary
        """

        self._execute_query(query, {"uid": uid, "summary": summary})
