import csv
from typing import Any

from pipeline.processors.biological_enrichment_framework import EnrichmentProcessor
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class CellTypeEnricher(EnrichmentProcessor):
    """
    Processor for enriching proteins with cell type expression data.
    Links proteins to CellType nodes via EXPRESSED_IN relationships.
    """

    def get_enrichment_type(self) -> str:
        return "cell_type"

    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """
        Parse cell type expression CSV file.
        Expected format: uniprot_id, cell_type, [expression_level, reliability]
        """
        protein_cell_types = {}  # uniprot_id -> list of {cell_type, level, reliability}
        all_cell_types = set()
        stats = {"proteins_processed": 0, "cell_types_found": 0, "unique_cell_types": 0}

        try:
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support various column naming conventions
                    uniprot_id = row.get(
                        "uniprot", row.get("uniprot_id", row.get("Protein", ""))
                    ).strip()
                    cell_type = row.get("cell_type", row.get("Cell Type", "")).strip()

                    if uniprot_id and cell_type:
                        if uniprot_id not in protein_cell_types:
                            protein_cell_types[uniprot_id] = []

                        entry = {
                            "name": cell_type,
                            "level": row.get(
                                "level", row.get("expression_level", "Unknown")
                            ).strip(),
                            "reliability": row.get("reliability", "Unknown").strip(),
                        }

                        protein_cell_types[uniprot_id].append(entry)
                        all_cell_types.add(cell_type)

            stats["proteins_processed"] = len(protein_cell_types)
            stats["cell_types_found"] = sum(len(v) for v in protein_cell_types.values())
            stats["unique_cell_types"] = len(all_cell_types)

            return {
                "protein_cell_types": protein_cell_types,
                "all_cell_types": list(all_cell_types),
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Failed to parse cell type file: {e}")
            return {"stats": stats, "protein_cell_types": {}, "all_cell_types": []}

    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create CellType nodes."""
        cell_types = data.get("all_cell_types", [])
        stats = {"created": 0, "updated": 0}

        if not cell_types:
            return stats

        # Create nodes in batches
        batch_size = 100
        for i in range(0, len(cell_types), batch_size):
            batch = cell_types[i : i + batch_size]

            query = """
            UNWIND $cell_types AS ct_name
            MERGE (c:CellType {name: ct_name, id: ct_name})
            """
            try:
                self._execute_query(query, {"cell_types": batch})
                stats["created"] += len(batch)
            except Exception as e:
                logger.error(f"Error creating cell type nodes: {e}")

        return stats

    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create EXPRESSED_IN relationships between Proteins and CellTypes."""
        protein_cell_types = data.get("protein_cell_types", {})
        stats = {"relationships_created": 0, "proteins_matched": 0}

        # Process in batches
        batch_size = 50
        items = list(protein_cell_types.items())

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            for uniprot_id, cell_type_entries in batch:
                # Check if protein exists first to avoid creating disconnected nodes if desired,
                # or just MERGE. Here we assume we want to attach to existing proteins.

                for entry in cell_type_entries:
                    ct_name = entry["name"]
                    level = entry["level"]
                    reliability = entry["reliability"]

                    query = """
                    MATCH (p:Protein) WHERE p.uniprotID = $uniprot_id
                    MATCH (c:CellType {name: $ct_name})
                    MERGE (p)-[r:EXPRESSED_IN]->(c)
                    SET r.expression_level = $level,
                        r.reliability = $reliability,
                        r.source = 'CellTypeEnrichment'
                    """

                    try:
                        self._execute_query(
                            query,
                            {
                                "uniprot_id": uniprot_id,
                                "ct_name": ct_name,
                                "level": level,
                                "reliability": reliability,
                            },
                        )
                        stats["relationships_created"] += 1
                    except Exception as e:
                        logger.error(f"Error linking {uniprot_id} to {ct_name}: {e}")

        return stats
