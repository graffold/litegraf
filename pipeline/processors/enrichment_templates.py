#!/usr/bin/env python3
"""
Template for creating new biological enrichment processors.

This file shows how to implement a new enrichment processor by extending
the EnrichmentProcessor base class. Use this as a starting point for
adding new types of biological data enrichment.

Example implementations:
- Protein domains (Pfam, InterPro)
- Gene Ontology (GO) terms
- Protein families
- Post-translational modifications
- Tissue expression data
- Disease associations
"""

import csv
from collections import defaultdict
from typing import Any

from pipeline.processors.biological_enrichment_framework import EnrichmentProcessor


class ProteinDomainProcessor(EnrichmentProcessor):
    """
    Example processor for protein domain enrichment.
    Processes data like: uniprotID,domain_name,start_position,end_position,source
    """

    def get_enrichment_type(self) -> str:
        return "protein_domains"

    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """Parse protein domain data file."""
        protein_domains = defaultdict(list)  # uniprotID -> list of domains
        all_domains = set()
        stats = {"proteins_processed": 0, "domains_found": 0, "unique_domains": 0}

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprotID = row.get("uniprotID", "").strip()
                domain_name = row.get("domain_name", "").strip()

                if uniprotID and domain_name:
                    domain_info = {
                        "name": domain_name,
                        "start": int(row.get("start_position", 0)),
                        "end": int(row.get("end_position", 0)),
                        "source": row.get("source", "Unknown"),
                        "accession": row.get("domain_accession", ""),
                    }

                    protein_domains[uniprotID].append(domain_info)
                    all_domains.add(domain_name)

        stats["proteins_processed"] = len(protein_domains)
        stats["domains_found"] = sum(
            len(domains) for domains in protein_domains.values()
        )
        stats["unique_domains"] = len(all_domains)

        return {
            "protein_domains": dict(protein_domains),
            "all_domains": all_domains,
            "stats": stats,
        }

    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create ProteinDomain nodes."""
        domains = data["all_domains"]
        stats = {"created": 0, "updated": 0}

        # Create domain nodes in batches
        batch_size = 100
        domain_list = list(domains)

        for i in range(0, len(domain_list), batch_size):
            batch = domain_list[i : i + batch_size]

            # Check existing domains
            existing_query = """
            UNWIND $domains AS domain_name
            OPTIONAL MATCH (d:ProteinDomain {name: domain_name})
            RETURN domain_name, d IS NOT NULL AS exists
            """
            existing_results = self._execute_query(existing_query, {"domains": batch})
            existing_domains = {
                r["domain_name"] for r in existing_results if r.get("exists")
            }

            # Create new domains
            new_domains = [d for d in batch if d not in existing_domains]
            if new_domains:
                create_query = """
                UNWIND $domains AS domain_name
                CREATE (d:ProteinDomain {
                    name: domain_name,
                    id: domain_name,
                    category: 'protein_domain',
                    created_at: datetime()
                })
                """
                self._execute_query(create_query, {"domains": new_domains})
                stats["created"] += len(new_domains)

            stats["updated"] += len(existing_domains)

        return stats

    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create HAS_DOMAIN relationships."""
        protein_domains = data["protein_domains"]
        stats = {
            "relationships_created": 0,
            "proteins_matched": 0,
            "proteins_not_found": 0,
        }

        # Process in batches
        batch_size = 50
        items = list(protein_domains.items())

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            uniprotIDs = [pid for pid, _ in batch]

            # Check which proteins exist
            protein_check_query = """
            UNWIND $uniprotIDs AS uniprotID
            OPTIONAL MATCH (p) WHERE p.uniprotID = uniprotID
            RETURN uniprotID, p IS NOT NULL AS exists
            """
            protein_results = self._execute_query(
                protein_check_query, {"uniprotIDs": uniprotIDs}
            )
            existing_proteins = {
                r["uniprotID"] for r in protein_results if r.get("exists")
            }

            stats["proteins_matched"] += len(existing_proteins)
            stats["proteins_not_found"] += len(uniprotIDs) - len(existing_proteins)

            # Create relationships
            for protein_id, domains in batch:
                if protein_id not in existing_proteins:
                    continue

                for domain in domains:
                    relationship_query = """
                    MATCH (p) WHERE p.uniprotID = $uniprotID
                    MATCH (d:ProteinDomain {name: $domain_name})
                    MERGE (p)-[r:HAS_DOMAIN]->(d)
                    SET r.start_position = $start,
                        r.end_position = $end,
                        r.source = $source,
                        r.accession = $accession,
                        r.created_at = datetime()
                    """
                    self._execute_query(
                        relationship_query,
                        {
                            "uniprotID": protein_id,
                            "domain_name": domain["name"],
                            "start": domain["start"],
                            "end": domain["end"],
                            "source": domain["source"],
                            "accession": domain["accession"],
                        },
                    )
                    stats["relationships_created"] += 1

        return stats


class GeneOntologyProcessor(EnrichmentProcessor):
    """
    Example processor for Gene Ontology (GO) term enrichment.
    Processes data like: uniprotID,go_term,go_id,evidence_code,aspect
    """

    def get_enrichment_type(self) -> str:
        return "gene_ontology"

    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """Parse GO term data file."""
        protein_go_terms = defaultdict(list)  # uniprotID -> list of GO terms
        all_go_terms = set()
        stats = {"proteins_processed": 0, "go_terms_found": 0, "unique_go_terms": 0}

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprotID = row.get("uniprotID", "").strip()
                go_term = row.get("go_term", "").strip()
                go_id = row.get("go_id", "").strip()

                if uniprotID and go_term and go_id:
                    go_info = {
                        "term": go_term,
                        "id": go_id,
                        "evidence": row.get("evidence_code", ""),
                        "aspect": row.get(
                            "aspect", ""
                        ),  # P (process), F (function), C (component)
                        "source": row.get("source", "UniProt"),
                    }

                    protein_go_terms[uniprotID].append(go_info)
                    all_go_terms.add((go_id, go_term))

        stats["proteins_processed"] = len(protein_go_terms)
        stats["go_terms_found"] = sum(len(terms) for terms in protein_go_terms.values())
        stats["unique_go_terms"] = len(all_go_terms)

        return {
            "protein_go_terms": dict(protein_go_terms),
            "all_go_terms": all_go_terms,
            "stats": stats,
        }

    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create GOTerm nodes."""
        go_terms = data["all_go_terms"]  # Set of (go_id, go_term) tuples
        stats = {"created": 0, "updated": 0}

        # Convert to list of dicts for processing
        go_term_list = [{"id": go_id, "term": go_term} for go_id, go_term in go_terms]

        # Create GO term nodes in batches
        batch_size = 100

        for i in range(0, len(go_term_list), batch_size):
            batch = go_term_list[i : i + batch_size]
            go_ids = [item["id"] for item in batch]

            # Check existing GO terms
            existing_query = """
            UNWIND $go_ids AS go_id
            OPTIONAL MATCH (go:GOTerm {id: go_id})
            RETURN go_id, go IS NOT NULL AS exists
            """
            existing_results = self._execute_query(existing_query, {"go_ids": go_ids})
            existing_go_ids = {r["go_id"] for r in existing_results if r.get("exists")}

            # Create new GO terms
            new_terms = [item for item in batch if item["id"] not in existing_go_ids]
            if new_terms:
                create_query = """
                UNWIND $go_terms AS go_data
                CREATE (go:GOTerm {
                    id: go_data.id,
                    name: go_data.term,
                    aspect: CASE
                        WHEN go_data.term CONTAINS 'biological_process' THEN 'P'
                        WHEN go_data.term CONTAINS 'molecular_function' THEN 'F'
                        WHEN go_data.term CONTAINS 'cellular_component' THEN 'C'
                        ELSE 'unknown'
                    END,
                    created_at: datetime()
                })
                """
                self._execute_query(create_query, {"go_terms": new_terms})
                stats["created"] += len(new_terms)

            stats["updated"] += len(existing_go_ids)

        return stats

    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create ASSOCIATED_WITH relationships to GO terms."""
        protein_go_terms = data["protein_go_terms"]
        stats = {
            "relationships_created": 0,
            "proteins_matched": 0,
            "proteins_not_found": 0,
        }

        # Process in batches
        batch_size = 50
        items = list(protein_go_terms.items())

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            uniprotIDs = [pid for pid, _ in batch]

            # Check which proteins exist
            protein_check_query = """
            UNWIND $uniprotIDs AS uniprotID
            OPTIONAL MATCH (p) WHERE p.uniprotID = uniprotID
            RETURN uniprotID, p IS NOT NULL AS exists
            """
            protein_results = self._execute_query(
                protein_check_query, {"uniprotIDs": uniprotIDs}
            )
            existing_proteins = {
                r["uniprotID"] for r in protein_results if r.get("exists")
            }

            stats["proteins_matched"] += len(existing_proteins)
            stats["proteins_not_found"] += len(uniprotIDs) - len(existing_proteins)

            # Create relationships
            for protein_id, go_terms in batch:
                if protein_id not in existing_proteins:
                    continue

                for go_term in go_terms:
                    relationship_query = """
                    MATCH (p) WHERE p.uniprotID = $uniprotID
                    MATCH (go:GOTerm {id: $go_id})
                    MERGE (p)-[r:ASSOCIATED_WITH]->(go)
                    SET r.evidence_code = $evidence,
                        r.aspect = $aspect,
                        r.source = $source,
                        r.created_at = datetime()
                    """
                    self._execute_query(
                        relationship_query,
                        {
                            "uniprotID": protein_id,
                            "go_id": go_term["id"],
                            "evidence": go_term["evidence"],
                            "aspect": go_term["aspect"],
                            "source": go_term["source"],
                        },
                    )
                    stats["relationships_created"] += 1

        return stats


# Template for creating new processors
class TemplateEnrichmentProcessor(EnrichmentProcessor):
    """
    Template for creating new enrichment processors.

    Replace the methods below with your specific implementation:

    1. get_enrichment_type() - Return a unique string identifier
    2. parse_data_file() - Parse your specific data format
    3. create_enrichment_nodes() - Create your specific node types
    4. create_relationships() - Create relationships to proteins

    Then add your processor to the EnrichmentManager.processors dict.
    """

    def get_enrichment_type(self) -> str:
        return "your_enrichment_type"

    def parse_data_file(self, file_path: str) -> dict[str, Any]:
        """Parse your data file format."""
        # Example structure - modify as needed
        protein_data = defaultdict(list)  # uniprotID -> list of data items
        all_items = set()
        stats = {"proteins_processed": 0, "items_found": 0, "unique_items": 0}

        with open(file_path, encoding="utf-8") as f:
            # Parse your file format (CSV, TSV, JSON, etc.)
            reader = csv.DictReader(f)  # or other parser
            for row in reader:
                uniprotID = row.get("uniprotID", "").strip()
                # Parse your specific data fields
                item_name = row.get("item_name", "").strip()

                if uniprotID and item_name:
                    item_info = {
                        "name": item_name,
                        # Add other fields as needed
                        "confidence": float(row.get("confidence", 1.0)),
                        "source": row.get("source", "Unknown"),
                    }

                    protein_data[uniprotID].append(item_info)
                    all_items.add(item_name)

        stats["proteins_processed"] = len(protein_data)
        stats["items_found"] = sum(len(items) for items in protein_data.values())
        stats["unique_items"] = len(all_items)

        return {
            "protein_data": dict(protein_data),
            "all_items": all_items,
            "stats": stats,
        }

    def create_enrichment_nodes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create your specific enrichment nodes."""
        items = data["all_items"]
        stats = {"created": 0, "updated": 0}

        # Create nodes in batches
        batch_size = 100
        item_list = list(items)

        for i in range(0, len(item_list), batch_size):
            batch = item_list[i : i + batch_size]

            # Check existing items
            existing_query = """
            UNWIND $items AS item_name
            OPTIONAL MATCH (n:YourNodeType {name: item_name})
            RETURN item_name, n IS NOT NULL AS exists
            """
            existing_results = self._execute_query(existing_query, {"items": batch})
            existing_items = {
                r["item_name"] for r in existing_results if r.get("exists")
            }

            # Create new items
            new_items = [item for item in batch if item not in existing_items]
            if new_items:
                create_query = """
                UNWIND $items AS item_name
                CREATE (n:YourNodeType {
                    name: item_name,
                    id: item_name,
                    category: 'your_category',
                    created_at: datetime()
                })
                """
                self._execute_query(create_query, {"items": new_items})
                stats["created"] += len(new_items)

            stats["updated"] += len(existing_items)

        return stats

    def create_relationships(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create relationships between proteins and your enrichment nodes."""
        protein_data = data["protein_data"]
        stats = {
            "relationships_created": 0,
            "proteins_matched": 0,
            "proteins_not_found": 0,
        }

        # Process in batches
        batch_size = 50
        items = list(protein_data.items())

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            uniprotIDs = [pid for pid, _ in batch]

            # Check which proteins exist
            protein_check_query = """
            UNWIND $uniprotIDs AS uniprotID
            OPTIONAL MATCH (p) WHERE p.uniprotID = uniprotID
            RETURN uniprotID, p IS NOT NULL AS exists
            """
            protein_results = self._execute_query(
                protein_check_query, {"uniprotIDs": uniprotIDs}
            )
            existing_proteins = {
                r["uniprotID"] for r in protein_results if r.get("exists")
            }

            stats["proteins_matched"] += len(existing_proteins)
            stats["proteins_not_found"] += len(uniprotIDs) - len(existing_proteins)

            # Create relationships
            for protein_id, data_items in batch:
                if protein_id not in existing_proteins:
                    continue

                for item in data_items:
                    relationship_query = """
                    MATCH (p) WHERE p.uniprotID = $uniprotID
                    MATCH (n:YourNodeType {name: $item_name})
                    MERGE (p)-[r:YOUR_RELATIONSHIP_TYPE]->(n)
                    SET r.confidence = $confidence,
                        r.source = $source,
                        r.created_at = datetime()
                    """
                    self._execute_query(
                        relationship_query,
                        {
                            "uniprotID": protein_id,
                            "item_name": item["name"],
                            "confidence": item.get("confidence", 1.0),
                            "source": item.get("source", "Unknown"),
                        },
                    )
                    stats["relationships_created"] += 1

        return stats
