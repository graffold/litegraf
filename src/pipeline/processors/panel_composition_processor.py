"""
Panel Composition Processor - Ingest Olink panel composition metadata.

This module provides functionality to create Panel nodes and link them to proteins
via CONTAINS_TARGET relationships, enabling panel-based filtering and queries.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PanelCompositionProcessor:
    """Process Olink panel composition metadata from JSON files."""

    def __init__(self, db: Any):
        """
        Initialize panel composition processor.

        Args:
            db: Database instance (Neo4j or Neptune)
        """
        self.db = db

    def ingest_panel_composition(self, json_path: Path) -> dict[str, int]:
        """
        Ingest Olink panel composition data from JSON file.

        Expected JSON format:
        {
          "panels": [
            {
              "panel_name": "Inflammation",
              "panel_id": "OL001",
              "target_count": 92,
              "description": "Inflammation panel",
              "targets": [
                {
                  "uniprot_id": "P01375",
                  "protein_name": "TNF-alpha",
                  "gene_symbol": "TNF",
                  "detection_limit": 0.5,
                  "dynamic_range": 4.5
                },
                ...
              ]
            },
            ...
          ]
        }

        Args:
            json_path: Path to JSON file with panel composition data

        Returns:
            Dict with counts: {"panels_created": N, "relationships_created": M}
        """
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return {"panels_created": 0, "relationships_created": 0}

        logger.info(f"Ingesting panel composition from {json_path}")

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            return {"panels_created": 0, "relationships_created": 0}

        panels = data.get("panels", [])
        if not panels:
            logger.warning("No panels found in JSON file")
            return {"panels_created": 0, "relationships_created": 0}

        panels_created = 0
        relationships_created = 0

        for panel in panels:
            panel_name = panel.get("panel_name", "").strip()
            if not panel_name:
                logger.warning(f"Skipping panel with missing panel_name: {panel}")
                continue

            # Create Panel node
            panel_props = {
                "name": panel_name,
                "data_source": "Olink",
            }
            if panel.get("panel_id"):
                panel_props["panel_id"] = panel["panel_id"].strip()
            if panel.get("target_count") is not None:
                panel_props["target_count"] = int(panel["target_count"])
            if panel.get("description"):
                panel_props["description"] = panel["description"].strip()

            create_panel_query = """
            MERGE (panel:Panel {name: $name})
            SET panel += $props
            RETURN panel.name as name
            """
            result = self.db.execute_query(
                create_panel_query, {"name": panel_name, "props": panel_props}
            )
            if result:
                panels_created += 1
                logger.debug(f"Created/updated Panel node: {panel_name}")

            # Link targets to panel
            targets = panel.get("targets", [])
            for target in targets:
                uniprot_id = target.get("uniprot_id", "").strip()
                if not uniprot_id:
                    logger.warning(
                        f"Skipping target with missing uniprot_id in panel {panel_name}"
                    )
                    continue

                # Build relationship properties
                rel_props = {"data_source": "Olink"}
                if target.get("detection_limit") is not None:
                    try:
                        rel_props["detection_limit"] = float(target["detection_limit"])
                    except ValueError:
                        logger.warning(
                            f"Invalid detection_limit for {uniprot_id}: {target['detection_limit']}"
                        )
                if target.get("dynamic_range") is not None:
                    try:
                        rel_props["dynamic_range"] = float(target["dynamic_range"])
                    except ValueError:
                        logger.warning(
                            f"Invalid dynamic_range for {uniprot_id}: {target['dynamic_range']}"
                        )

                # Create CONTAINS_TARGET relationship
                link_query = """
                MATCH (panel:Panel {name: $panel_name})
                MATCH (p:Protein)
                WHERE p.uniprot_id = $uniprot_id OR p.uniprotID = $uniprot_id
                MERGE (panel)-[r:CONTAINS_TARGET]->(p)
                SET r += $rel_props
                RETURN p.uniprot_id as protein_id
                """
                params = {
                    "panel_name": panel_name,
                    "uniprot_id": uniprot_id,
                    "rel_props": rel_props,
                }
                result = self.db.execute_query(link_query, params)
                if result:
                    relationships_created += 1
                    logger.debug(f"Linked protein {uniprot_id} to panel {panel_name}")
                else:
                    logger.warning(
                        f"Protein node not found for UniProt ID: {uniprot_id} (panel: {panel_name})"
                    )

        logger.info(
            f"Created {panels_created} panels and {relationships_created} CONTAINS_TARGET relationships"
        )
        return {
            "panels_created": panels_created,
            "relationships_created": relationships_created,
        }

    def get_panel_targets(self, panel_name: str) -> list[dict[str, Any]]:
        """
        Get all protein targets for a specific panel.

        Args:
            panel_name: Name of the panel

        Returns:
            List of dicts with protein info: uniprot_id, name, gene_symbol, detection_limit, dynamic_range
        """
        query = """
        MATCH (panel:Panel {name: $panel_name})-[r:CONTAINS_TARGET]->(p:Protein)
        RETURN p.uniprot_id as uniprot_id,
               p.name as protein_name,
               p.gene_symbol as gene_symbol,
               r.detection_limit as detection_limit,
               r.dynamic_range as dynamic_range
        ORDER BY p.name
        """
        results = self.db.execute_query(query, {"panel_name": panel_name})
        return [dict(row) for row in results] if results else []

    def get_protein_panels(self, uniprot_id: str) -> list[str]:
        """
        Get all panels containing a specific protein.

        Args:
            uniprot_id: UniProt accession ID

        Returns:
            List of panel names
        """
        query = """
        MATCH (panel:Panel)-[:CONTAINS_TARGET]->(p:Protein)
        WHERE p.uniprot_id = $uniprot_id OR p.uniprotID = $uniprot_id
        RETURN panel.name as panel_name
        ORDER BY panel.name
        """
        results = self.db.execute_query(query, {"uniprot_id": uniprot_id})
        return [row["panel_name"] for row in results] if results else []

    def get_all_panels(self) -> list[dict[str, Any]]:
        """
        Get all panels with metadata.

        Returns:
            List of dicts with panel info: name, panel_id, target_count, description
        """
        query = """
        MATCH (panel:Panel)
        RETURN panel.name as name,
               panel.panel_id as panel_id,
               panel.target_count as target_count,
               panel.description as description
        ORDER BY panel.name
        """
        results = self.db.execute_query(query, {})
        return [dict(row) for row in results] if results else []
