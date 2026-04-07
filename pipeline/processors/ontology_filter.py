import argparse
import csv
import os
from typing import Any

from obonet import read_obo

from src.core.database import Neo4jDatabase
from src.utils import logging_utils

logger = logging_utils.setup_logging()

# Fallback dictionary-based ontology
FALLBACK_ONTOLOGY = {
    "cvd": "Cardiovascular Disease",
    "heart attack": "Cardiovascular Disease",
    "myocardial infarction": "Myocarditis",
    "alzheimer's": "Alzheimer's Disease",
    "alzheimers": "Alzheimer's Disease",
    "diabetes": "Diabetes",
    "diabetes mellitus": "disease",
    "cancer": "Cancer",
    "breast cancer": "Breast Cancer",
    "lung cancer": "Lung Cancer",
    "stroke": "Stroke",
    "hypertension": "Hypertension",
    "covid": "COVID-19",
    "corona virus": "COVID-19",
}

# Generic terms to exclude for Protein nodes
SPURIOUS_PROTEIN_TERMS = {
    "protein",
    "enzyme",
    "receptor",
    "molecule",
    "compound",
    "substance",
    "isoform",
    "fragment",
    "subunit",
    "domain",
    "region",
    "peptide",
    "polypeptide",
}

# Generic terms to exclude for any node
SPURIOUS_GENERIC_TERMS = {
    "disease",
    "syndrome",
    "disorder",
    "condition",
    "patient",
    "subject",
    "cohort",
    "group",
    "population",
    "study",
    "analysis",
    "data",
    "result",
    "finding",
    "control",
    "case",
    "sample",
    "specimen",
    "level",
    "concentration",
    "expression",
    "isoform 1",
    "isoform 2",
    "isoform 3",
    "isoform 4",
}


class OntologyFilter:
    def __init__(self, mondo_path: str = None, protein_ontology_path: str = None):
        """Initialize with MonDO and protein ontologies."""
        self.mondo = None
        self.disease_ontology = {}
        self.protein_ontology = {}
        self.mondo_path = mondo_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "utils", "mondo.obo")
        )
        self.protein_ontology_path = protein_ontology_path or os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "utils", "uniprot_ids_human.csv"
            )
        )
        self.backend_adapter = None  # Backend adapter for query execution
        self._load_ontologies()
        logger.info("Initialized OntologyFilter with disease and protein ontologies")

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query using the appropriate backend."""
        if hasattr(self, "backend_adapter") and self.backend_adapter is not None:
            return self.backend_adapter.execute_query(query, parameters)
        if hasattr(self, "db") and self.db is not None:
            return self.db._execute_cypher(query, parameters)
        raise ValueError(
            "No database connection available. Call resolve_and_label_nodes with appropriate backend parameters first."
        )

    def _load_ontologies(self):
        """Load disease and protein ontologies."""
        # Load disease ontology (MonDO)
        try:
            self.mondo = read_obo(self.mondo_path)
            self.disease_ontology = self._load_disease_ontology()
            logger.info(
                f"Successfully loaded MonDO ontology with {len(self.disease_ontology)} entries"
            )
        except Exception as e:
            logger.error(
                f"Failed to load MonDO ontology: {e}. Using fallback dictionary."
            )
            self.disease_ontology = {
                k.lower(): {"name": v, "synonyms": []}
                for k, v in FALLBACK_ONTOLOGY.items()
            }

        # Load protein ontology
        try:
            self.protein_ontology = self._load_protein_ontology()
            logger.info(f"Loaded {len(self.protein_ontology)} protein ontology entries")
        except Exception as e:
            logger.error(f"Failed to load protein ontology: {e}")
            self.protein_ontology = {}

    def _load_disease_ontology(self) -> dict[str, dict[str, Any]]:
        """Load disease ontology from mondo.obo file."""
        ontology = {}
        if self.mondo is not None:
            for term_id, term in self.mondo.nodes(data=True):
                canonical_name = term.get("name")
                if canonical_name:
                    ontology[term_id] = {
                        "name": canonical_name,
                        "synonyms": [
                            s.split("[")[0].strip().strip('"')
                            for s in term.get("synonym", [])
                        ],
                    }
        return ontology

    def _load_protein_ontology(self) -> dict[str, dict[str, Any]]:
        """Load protein ontology from CSV file."""
        protein_ontology = {}
        with open(self.protein_ontology_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprot_id = row["uniprot"]
                gene_name = row["gene_name"]
                if uniprot_id and gene_name:
                    protein_ontology[uniprot_id] = {
                        "name": gene_name,
                        "synonyms": [],
                    }
        return protein_ontology

    def is_valid_disease(self, name: str) -> bool:
        """Check if a disease name is valid using MonDO or fallback ontology."""
        name_lower = name.lower().strip()

        # First check ontology matches
        for data in self.disease_ontology.values():
            if name_lower == data["name"].lower():
                return True
            if any(name_lower == s.lower() for s in data.get("synonyms", [])):
                return True
        if name_lower in [k.lower() for k in FALLBACK_ONTOLOGY]:
            return True

        # For biomedical KG, be more permissive with disease names
        # Allow diseases that are reasonably long and contain medical keywords
        return bool(
            len(name_lower) > 5
            and any(
                keyword in name_lower
                for keyword in [
                    "disease",
                    "disorder",
                    "syndrome",
                    "cancer",
                    "tumor",
                    "deficiency",
                    "failure",
                    "insufficiency",
                    "hypertension",
                    "diabetes",
                    "arthritis",
                    "infection",
                    "inflammation",
                    "fever",
                    "pain",
                    "fracture",
                    "injury",
                ]
            )
        )

    def is_valid_protein(self, name: str) -> bool:
        """Check if a protein name is valid, excluding generic terms."""
        name_lower = name.lower().strip()
        if name_lower in SPURIOUS_PROTEIN_TERMS or name_lower in SPURIOUS_GENERIC_TERMS:
            return False
        # Also filter out "Isoform X" if it's just that
        if name_lower.startswith("isoform ") and len(name_lower.split()) <= 2:
            return False
        return len(name_lower) > 2 and not name_lower.startswith("generic")

    def standardize_disease(self, name: str) -> str:
        """Map disease name to canonical form using MonDO or fallback ontology."""
        name_lower = name.lower().strip()
        for data in self.disease_ontology.values():
            if name_lower == data["name"].lower():
                return data["name"]
            if any(name_lower == s.lower() for s in data.get("synonyms", [])):
                return data["name"]
        return FALLBACK_ONTOLOGY.get(name_lower, name)

    def standardize_protein(self, name: str) -> str:
        """Map protein name to canonical form using protein ontology."""
        name_lower = name.lower().strip()
        for data in self.protein_ontology.values():
            if name_lower == data["name"].lower():
                return data["name"]
            if any(name_lower == s.lower() for s in data.get("synonyms", [])):
                return data["name"]
        return name

    def filter_nodes(self, nodes: list) -> list:
        """Filter nodes for biomedical relevance."""
        valid_nodes = []
        for node in nodes:
            # Handle both Neo4j nodes and dict nodes
            if isinstance(node, dict):
                name = node.get("name", "")
                label = node.get("type", "")
                is_dict = True
            else:
                name = getattr(node, "properties", {}).get("name", "")
                label = getattr(node, "label", "")
                is_dict = False

            name_lower = name.lower().strip()
            if name_lower in SPURIOUS_GENERIC_TERMS:
                logger.debug(f"Removed spurious generic node: {name}")
                continue

            if label == "Disease":
                if self.is_valid_disease(name):
                    standardized_name = self.standardize_disease(name)
                    if is_dict:
                        node["name"] = standardized_name
                    else:
                        node.properties["name"] = standardized_name  # type: ignore
                    valid_nodes.append(node)
                else:
                    logger.debug(f"Removed spurious Disease node: {name}")
            elif label == "Protein":
                if self.is_valid_protein(name):
                    standardized_name = self.standardize_protein(name)
                    if is_dict:
                        node["name"] = standardized_name
                    else:
                        node.properties["name"] = standardized_name  # type: ignore
                    valid_nodes.append(node)
                else:
                    logger.debug(f"Removed spurious Protein node: {name}")
            else:
                # For generic Entity nodes, also check against spurious terms
                if (
                    name_lower in SPURIOUS_PROTEIN_TERMS
                    or name_lower in SPURIOUS_GENERIC_TERMS
                ):
                    logger.debug(f"Removed spurious Entity node: {name}")
                    continue
                # Also filter out "Isoform X" if it's just that
                if name_lower.startswith("isoform ") and len(name_lower.split()) <= 2:
                    logger.debug(f"Removed spurious Isoform node: {name}")
                    continue

                logger.debug(f"Removed invalid node label: {label}")
        return valid_nodes

    def filter_relationships(
        self, relationships: list, valid_node_ids: set[str]
    ) -> list:
        """Filter relationships to include only valid types and nodes."""
        valid_relationships = []
        for rel in relationships:
            rel_type = getattr(rel, "type", "")
            start_id = getattr(rel, "start_node_id", "")
            end_id = getattr(rel, "end_node_id", "")
            if rel_type in ["ASSOCIATED_WITH", "CAUSES", "TREATS"]:
                if start_id in valid_node_ids and end_id in valid_node_ids:
                    valid_relationships.append(rel)
                else:
                    logger.debug(
                        f"Removed relationship {rel_type} with invalid nodes: {start_id} -> {end_id}"
                    )
            else:
                logger.debug(f"Removed invalid relationship type: {rel_type}")
        return valid_relationships

    def resolve_and_label_nodes(
        self,
        db=None,
        backend_adapter=None,
        batch_size: int = 1000,
        ingestion_job_id: str | None = None,
    ):
        """Resolve and label nodes as Disease or Protein, standardizing names in database.

        Args:
            db: Database connection (alternative to backend_adapter).
            backend_adapter: Backend adapter for query execution.
            batch_size: Number of nodes to update per write batch.
            ingestion_job_id: If provided, only process nodes from this ingestion job.
                When None, processes ALL nodes in the database (legacy full-scan behavior).
        """
        scope = f"job {ingestion_job_id}" if ingestion_job_id else "ALL nodes"
        logger.info(f"Resolving and labeling nodes in Neo4j database (scope: {scope})")

        # Set up query execution
        if backend_adapter is not None:
            self.backend_adapter = backend_adapter
        else:
            self.db = db
            if not self.db:
                raise ValueError(
                    "Database connection required. Provide db or backend_adapter."
                )

        offset = 0
        fetch_size = 2000  # Fetch in smaller chunks to avoid OOM
        total_updates = 0

        # Build WHERE clause: scope to current job if provided
        if ingestion_job_id:
            where_clause = f"WHERE n.ingestion_job_id = '{ingestion_job_id}'"
        else:
            where_clause = "WHERE n:Entity OR n:Disease OR n:Protein OR NOT (n:Disease OR n:Protein)"

        while True:
            query = f"MATCH (n) {where_clause} RETURN n.id AS id, n.name AS name, labels(n) AS labels ORDER BY n.id SKIP {offset} LIMIT {fetch_size}"

            results = self._execute_query(query)
            if not results:
                break

            updates = []
            seen_ids = set()

            for record in results:
                node_id = record["id"]
                if node_id in seen_ids:
                    continue
                seen_ids.add(node_id)
                node_name = record["name"]
                labels = record["labels"]
                node_type = labels[0] if labels else "Entity"

                if not node_name:
                    logger.debug(f"Skipping node with missing name: {node_id}")
                    continue

                canonical_node = None
                if node_type in ["Disease", "Entity", "Unknown"]:
                    canonical_node = self._find_canonical_disease(node_name)
                    if canonical_node:
                        node_type = "Disease"
                if not canonical_node and node_type in ["Protein", "Entity", "Unknown"]:
                    canonical_node = self._find_canonical_protein(node_name)
                    if canonical_node:
                        node_type = "Protein"

                if canonical_node:
                    updates.append(
                        {
                            "id": node_id,
                            "canonical_name": canonical_node["name"],
                            "type": node_type,
                        }
                    )
                    logger.debug(
                        f"Resolved {node_type} '{node_name}' to '{canonical_node['name']}' (id: {node_id})"
                    )
                else:
                    logger.debug(
                        f"No canonical match for '{node_name}' (id: {node_id})"
                    )

            # Update in batches
            if updates:
                for i in range(0, len(updates), batch_size):
                    batch = updates[i : i + batch_size]
                    update_query = """
                    UNWIND $batch AS update
                    MATCH (n {id: update.id})
                    SET n.name = update.canonical_name
                    FOREACH (x IN CASE WHEN update.type = 'Disease' THEN [1] ELSE [] END |
                        SET n:Disease
                        REMOVE n:Entity
                    )
                    FOREACH (x IN CASE WHEN update.type = 'Protein' THEN [1] ELSE [] END |
                        SET n:Protein
                        REMOVE n:Entity
                    )
                    """
                    self._execute_query(update_query, parameters={"batch": batch})
                    logger.info(
                        f"Updated and labeled {len(batch)} nodes in current fetch batch"
                    )
                total_updates += len(updates)

            offset += fetch_size
            logger.info(f"Processed {offset} nodes so far...")

        logger.info(f"Resolved and labeled total {total_updates} nodes")

    def _find_canonical_disease(self, name: str) -> dict[str, Any] | None:
        """Find canonical disease ID and name based on input name."""
        name_lower = name.lower().strip()
        for disease_id, data in self.disease_ontology.items():
            if name_lower == data["name"].lower():
                return {"id": disease_id, "name": data["name"], "type": "Disease"}
            for synonym in data.get("synonyms", []):
                if name_lower == synonym.lower():
                    return {"id": disease_id, "name": data["name"], "type": "Disease"}
        return None

    def _find_canonical_protein(self, name: str) -> dict[str, Any] | None:
        """Find canonical protein ID and name based on input name."""
        name_lower = name.lower().strip()
        for protein_id, data in self.protein_ontology.items():
            if name_lower == data["name"].lower():
                return {"id": protein_id, "name": data["name"], "type": "Protein"}
            for synonym in data.get("synonyms", []):
                if name_lower == synonym.lower():
                    return {"id": protein_id, "name": data["name"], "type": "Protein"}
        return None

    def label_existing_disease_nodes(self, db=None):
        """Retroactively label existing Entity nodes as Disease if they match MonDO ontology."""
        logger.info("Retroactively labeling Entity nodes as Disease in Neo4j database")

        self.db = db
        if not self.db:
            raise ValueError("Database connection required for Neo4j backend")

        offset = 0
        fetch_size = 2000
        labeled_count = 0

        while True:
            # Get all Entity nodes and check each one against ontology
            query = f"MATCH (n:Entity) RETURN n.id AS id, n.name AS name ORDER BY n.id SKIP {offset} LIMIT {fetch_size}"
            results = self._execute_query(query)

            if not results:
                break

            for record in results:
                node_id = record["id"]
                node_name = record["name"]

                if not node_name:
                    continue

                # Check if this node matches any disease in ontology (exact match only)
                canonical_disease = self._find_canonical_disease(node_name)
                if canonical_disease:
                    # Label the node as Disease and update name
                    update_query = """
                    MATCH (n {id: $id})
                    REMOVE n:Entity
                    SET n:Disease, n.name = $canonical_name
                    """
                    self._execute_query(
                        update_query,
                        parameters={
                            "id": node_id,
                            "canonical_name": canonical_disease["name"],
                        },
                    )
                    labeled_count += 1
                    logger.debug(
                        f"Labeled Entity node '{node_name}' as Disease '{canonical_disease['name']}'"
                    )

            offset += fetch_size
            logger.info(f"Processed {offset} nodes for disease labeling...")

    def label_existing_protein_nodes(self, db=None):
        """Retroactively label existing Entity nodes as Protein if they match protein ontology."""
        logger.info("Retroactively labeling Entity nodes as Protein in Neo4j database")

        self.db = db
        if not self.db:
            raise ValueError("Database connection required for Neo4j backend")

        offset = 0
        fetch_size = 2000
        labeled_count = 0

        while True:
            # Get all Entity nodes and check each one against ontology
            query = f"MATCH (n:Entity) RETURN n.id AS id, n.name AS name ORDER BY n.id SKIP {offset} LIMIT {fetch_size}"
            results = self._execute_query(query)

            if not results:
                break

            for record in results:
                node_id = record["id"]
                node_name = record["name"]

                if not node_name:
                    continue

                # Check if this node matches any protein in ontology (exact match only)
                canonical_protein = self._find_canonical_protein(node_name)
                if canonical_protein:
                    # Label the node as Protein and update name
                    update_query = """
                    MATCH (n {id: $id})
                    REMOVE n:Entity
                    SET n:Protein, n.name = $canonical_name
                    """
                    self._execute_query(
                        update_query,
                        parameters={
                            "id": node_id,
                            "canonical_name": canonical_protein["name"],
                        },
                    )
                    labeled_count += 1
                    logger.debug(
                        f"Labeled Entity node '{node_name}' as Protein '{canonical_protein['name']}'"
                    )

            offset += fetch_size
            logger.info(f"Processed {offset} nodes for protein labeling...")

    def cleanup_database(self, db=None, conservative: bool = False):
        """Clean up database by removing only problematic nodes and relationships."""
        logger.info("Cleaning up Neo4j database")

        self.db = db
        if not self.db:
            raise ValueError("Database connection required for Neo4j backend")

        try:
            if conservative:
                # Conservative cleanup for invalid nodes (safer for production)
                cleanup_query = """
            MATCH (n)
            WHERE n.name IS NULL AND n.id IS NULL
              AND size(keys(n)) <= 1
              AND NOT (n)-[]-()
            DETACH DELETE n
            """
            else:
                # Original cleanup (for testing/legacy compatibility)
                cleanup_query = "MATCH (n) WHERE NOT (n)-[]-() DETACH DELETE n"

            self._execute_query(cleanup_query)

            # Remove relationships with missing source or target IDs + duplicates
            combined_rel_cleanup = """
            MATCH ()-[r]-()
            WHERE r.source_id IS NULL OR r.target_id IS NULL
            DELETE r
            """
            self._execute_query(combined_rel_cleanup)

            cleanup_type = "conservative" if conservative else "full"
            logger.info(f"Database cleanup completed ({cleanup_type} cleanup)")
        except Exception as e:
            logger.error(f"Failed to clean up database: {e}")


def main():
    """Run ontology operations from the command line."""
    parser = argparse.ArgumentParser(
        description="Standardize and label nodes in database using MonDO and protein ontologies."
    )
    parser.add_argument(
        "--mondo-path",
        default="src/utils/mondo.obo",
        help="Path to mondo.obo file (default: ../utils/mondo.obo)",
    )
    parser.add_argument(
        "--protein-path",
        default="src/utils/uniprot_ids_human.csv",
        help="Path to uniprot_ids_human.csv file (default: ../utils/uniprot_ids_human.csv)",
    )
    parser.add_argument(
        "--database", default="neo4j", help="Database name (default: neo4j)"
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user", default="neo4j", help="Neo4j username (default: neo4j)"
    )
    parser.add_argument("--password", default="testpass", help="Neo4j password")
    parser.add_argument(
        "--operation",
        choices=["standardize", "label_diseases", "label_proteins", "cleanup"],
        default="standardize",
        help="Operation to perform: standardize (names and labels), label_diseases, label_proteins, or cleanup (default: standardize)",
    )
    args = parser.parse_args()

    try:
        db = Neo4jDatabase(
            uri=args.uri, user=args.user, password=args.password, database=args.database
        )

        ontology_filter = OntologyFilter(
            mondo_path=args.mondo_path, protein_ontology_path=args.protein_path
        )

        if args.operation == "standardize":
            ontology_filter.resolve_and_label_nodes(db=db)
        elif args.operation == "label_diseases":
            ontology_filter.label_existing_disease_nodes(db=db)
        elif args.operation == "label_proteins":
            ontology_filter.label_existing_protein_nodes(db=db)
        elif args.operation == "cleanup":
            ontology_filter.cleanup_database(db=db, conservative=True)

        db.close()
    except Exception as e:
        logger.error(
            f"Failed to perform operation '{args.operation}': {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
