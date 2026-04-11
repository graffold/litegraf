#!/usr/bin/env python3
"""
OBO Structure Processor

Directly ingests OBO ontology files into the Knowledge Graph.
Creates nodes for terms and relationships for hierarchy and other defined connections.
Deterministic, fast, and structure-preserving.
"""

import logging
import re
from typing import Any

from pipeline.interfaces import GraphStore
logger = logging.getLogger(__name__)
class OBOStructureProcessor:
    """
    Parses OBO files and builds a Knowledge Graph directly from the ontology structure.
    """

    def __init__(self, database: str = "cvd1", graph_store: GraphStore | None = None):
        self.database = database
        if graph_store is not None:
            self.db = graph_store
        else:
            from pipeline.backends.neo4j_store import Neo4jGraphStore
            self.db = Neo4jGraphStore(database=database)

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query."""
        return self.db.execute_query(query, parameters)

    def parse_obo_file(
        self, file_path: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Parses an OBO file and returns lists of nodes and relationships.

        Returns:
            Tuple containing:
            - List of node dictionaries (id, name, def, synonyms, etc.)
            - List of relationship dictionaries (source, target, type)
        """
        nodes = []
        relationships = []

        current_term = {}
        in_term = False

        logger.info(f"Parsing OBO file: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if line == "[Term]":
                        if in_term and "id" in current_term:
                            self._process_term(current_term, nodes, relationships)
                        current_term = {
                            "synonyms": [],
                            "xrefs": [],
                            "is_a": [],
                            "relationships": [],
                        }
                        in_term = True
                    elif line == "[Typedef]":
                        if in_term and "id" in current_term:
                            self._process_term(current_term, nodes, relationships)
                        in_term = False
                    elif in_term and line:
                        self._parse_line(line, current_term)

            # Process last term
            if in_term and "id" in current_term:
                self._process_term(current_term, nodes, relationships)

            logger.info(
                f"Parsed {len(nodes)} terms and {len(relationships)} relationships"
            )
            return nodes, relationships

        except Exception as e:
            logger.error(f"Failed to parse OBO file: {e}")
            return [], []

    def _parse_line(self, line: str, current_term: dict[str, Any]):
        """Parses a single line of an OBO term definition."""
        if ":" not in line:
            return

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "id":
            current_term["id"] = value
        elif key == "name":
            current_term["name"] = value
        elif key == "def":
            # Extract definition text "Definition" [Reference]
            match = re.match(r'"(.*?)"', value)
            if match:
                current_term["def"] = match.group(1)
        elif key == "synonym":
            # Extract synonym "Synonym" SCOPE [Reference]
            # Example: "nephropathy" EXACT [MGI:...]
            match = re.match(r'"(.*?)"\s+([A-Z]+)\s+\[.*\]', value)
            if match:
                current_term["synonyms"].append(
                    {"text": match.group(1), "scope": match.group(2)}
                )
            else:
                # Fallback for simple format
                match = re.match(r'"(.*?)"', value)
                if match:
                    current_term["synonyms"].append(
                        {"text": match.group(1), "scope": "RELATED"}
                    )
        elif key == "xref":
            current_term["xrefs"].append(value.split(" ")[0])  # Take ID part
        elif key == "is_a":
            # is_a: ID ! Comment
            parent_id = value.split("!")[0].strip()
            current_term["is_a"].append(parent_id)
        elif key == "relationship":
            # relationship: type ID ! Comment
            parts = value.split(" ")
            if len(parts) >= 2:
                rel_type = parts[0]
                target_id = parts[1]
                comment = ""
                if "!" in value:
                    comment = value.split("!", 1)[1].strip()
                current_term["relationships"].append(
                    {"type": rel_type, "target": target_id, "comment": comment}
                )

    def _process_term(
        self,
        term: dict[str, Any],
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ):
        """Processes a parsed term into node and relationship structures."""
        # Filter synonyms
        exact_synonyms = []
        other_synonyms = []

        for syn in term["synonyms"]:
            if syn["scope"] == "EXACT":
                exact_synonyms.append(syn["text"])
            elif syn["scope"] in ["BROAD", "RELATED", "NARROW"]:
                other_synonyms.append(syn)

        # Create node
        node = {
            "id": term["id"],
            "name": term.get("name", term["id"]),
            "definition": term.get("def", ""),
            "synonyms": exact_synonyms,
            "xrefs": term["xrefs"],
            "source": "OBO_Ontology",
        }
        nodes.append(node)

        # Create IS_A relationships
        for parent_id in term["is_a"]:
            relationships.append(
                {
                    "source": term["id"],
                    "target": parent_id,
                    "type": "IS_A",
                    "target_type": "Term",
                }
            )

        # Create synonym relationships (BROAD, RELATED, NARROW)
        for syn in other_synonyms:
            relationships.append(
                {
                    "source": term["id"],
                    "target": syn["text"],
                    "type": f"HAS_{syn['scope']}_SYNONYM",
                    "target_type": "Synonym",
                }
            )

        # Create other relationships
        for rel in term["relationships"]:
            relationships.append(
                {
                    "source": term["id"],
                    "target": rel["target"],
                    "type": rel["type"]
                    .upper()
                    .replace("_", "_"),  # Normalize relationship type
                    "target_label": rel.get("comment", ""),
                    "target_type": "Term",
                }
            )

    def ingest_ontology(self, file_path: str, batch_size: int = 1000):
        """
        Main method to ingest the ontology.
        """
        nodes, relationships = self.parse_obo_file(file_path)

        if not nodes:
            logger.warning("No nodes parsed from OBO file.")
            return

        # 1. Create Nodes
        logger.info(f"Creating {len(nodes)} ontology nodes...")
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            self._create_nodes_batch(batch)
            logger.info(
                f"Created nodes batch {i // batch_size + 1}/{(len(nodes) + batch_size - 1) // batch_size}"
            )

        # 2. Create Relationships
        logger.info(f"Creating {len(relationships)} ontology relationships...")
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]
            self._create_relationships_batch(batch)
            logger.info(
                f"Created relationships batch {i // batch_size + 1}/{(len(relationships) + batch_size - 1) // batch_size}"
            )

        logger.info("Ontology ingestion complete.")

    def _create_nodes_batch(self, batch: list[dict[str, Any]]):
        """Creates a batch of nodes in the database."""
        # Determine label based on ID prefix (e.g., MONDO -> Disease)
        # For now, we'll use a generic 'OntologyTerm' and specific labels if possible

        # We can't easily dynamic label in UNWIND in pure Cypher without APOC sometimes
        # So we'll assume 'Disease' for MONDO, or generic 'OntologyTerm'

        query = """
        UNWIND $batch AS item
        MERGE (n:Disease:Entity {id: item.id})
        SET n.name = item.name,
            n.definition = item.definition,
            n.synonyms = item.synonyms,
            n.xrefs = item.xrefs,
            n.source = item.source,
            n.mondo_id = item.id
        """

        # If not MONDO, maybe use a different label?
        # For this specific use case (MONDO), 'Disease' is appropriate.

        self._execute_query(query, {"batch": batch})

    def _create_relationships_batch(self, batch: list[dict[str, Any]]):
        """Creates a batch of relationships in the database."""
        # Group by relationship type and target type to optimize
        rels_by_type = {}

        for rel in batch:
            key = (rel["type"], rel["target_type"])
            if key not in rels_by_type:
                rels_by_type[key] = []
            rels_by_type[key].append(rel)

        for (rel_type, target_type), rels in rels_by_type.items():
            if target_type == "Synonym":
                # Create Synonym nodes and relationships
                query = f"""
                UNWIND $batch AS item
                MATCH (source:Disease {{id: item.source}})
                MERGE (target:Synonym {{name: item.target}})
                MERGE (source)-[r:`{rel_type}`]->(target)
                SET r.source = 'OBO_Ontology'
                """
                self._execute_query(query, {"batch": rels})
            else:
                # Standard Term relationships
                # We need to handle different target labels based on ID prefix
                # But for batching efficiency, we'll use a generic approach or split further
                # For now, let's try to infer label in Cypher or just use a generic merge

                # To support GO terms, etc., we should ideally know the label.
                # Let's assume if it's not MONDO, it might be something else.
                # But MERGE requires a label usually for efficiency.
                # We'll use a dynamic approach where we try to match existing, or create generic.

                # Simplified approach: Use 'OntologyTerm' for targets that might not be Diseases
                # But we need to respect if they ARE Diseases (MONDO).

                # Let's split this batch further by ID prefix for better labeling
                rels_by_prefix = {"MONDO": [], "GO": [], "HP": [], "OTHER": []}
                for r in rels:
                    tid = r["target"]
                    if tid.startswith("MONDO:"):
                        rels_by_prefix["MONDO"].append(r)
                    elif tid.startswith("GO:"):
                        rels_by_prefix["GO"].append(r)
                    elif tid.startswith("HP:"):
                        rels_by_prefix["HP"].append(r)
                    else:
                        rels_by_prefix["OTHER"].append(r)

                # Execute for each prefix type
                for prefix, sub_batch in rels_by_prefix.items():
                    if not sub_batch:
                        continue

                    target_label = "Disease"
                    if prefix == "GO":
                        target_label = "GeneOntologyTerm"
                    elif prefix == "HP":
                        target_label = "Phenotype"
                    elif prefix == "OTHER":
                        target_label = "OntologyTerm"

                    query = f"""
                    UNWIND $batch AS item
                    MATCH (source:Disease {{id: item.source}})
                    MERGE (target:{target_label} {{id: item.target}})
                    ON CREATE SET target.name = COALESCE(item.target_label, item.target), target.source = 'OBO_Reference'
                    MERGE (source)-[r:`{rel_type}`]->(target)
                    SET r.source = 'OBO_Ontology'
                    """
                    self._execute_query(query, {"batch": sub_batch})

    def close(self):
        if self.db:
            self.db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest OBO ontology")
    parser.add_argument("--file", required=True, help="Path to OBO file")
    parser.add_argument("--database", default="cvd1")

    args = parser.parse_args()

    from pipeline.backends.neo4j_store import Neo4jGraphStore
    processor = OBOStructureProcessor(database=args.database, graph_store=Neo4jGraphStore(database=args.database))
    try:
        processor.ingest_ontology(args.file)
    finally:
        processor.close()
