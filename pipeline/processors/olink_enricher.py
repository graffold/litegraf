"""
Olink Enricher - Add Olink assay validation data to protein nodes.

This module provides functionality to enrich protein nodes with Olink-specific
assay performance metrics from CSV validation data files.
"""

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OlinkEnricher:
    """Enrich protein nodes with Olink assay validation data."""

    def __init__(self, db: Any):
        """
        Initialize Olink enricher.

        Args:
            db: Database instance (Neo4j or Neptune)
        """
        self.db = db

    def ingest_validation_data(self, csv_path: Path) -> int:
        """
        Ingest Olink assay validation data from CSV file.

        Expected CSV format:
        - uniprot_id: UniProt accession ID
        - assay_name: Olink assay name
        - panel: Olink panel name
        - lod: Limit of detection (pg/mL)
        - lloq: Lower limit of quantification (pg/mL)
        - cv_intra: Intra-assay coefficient of variation (%)
        - cv_inter: Inter-assay coefficient of variation (%)
        - specificity: Assay specificity score (0-1)

        Args:
            csv_path: Path to CSV file with validation data

        Returns:
            Number of proteins enriched
        """
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0

        logger.info(f"Ingesting Olink validation data from {csv_path}")
        enriched_count = 0

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uniprot_id = row.get("uniprot_id", "").strip()
                if not uniprot_id:
                    logger.warning(f"Skipping row with missing uniprot_id: {row}")
                    continue

                # Build properties dict
                props = {"data_source": "Olink"}
                if row.get("assay_name"):
                    props["olink_assay_name"] = row["assay_name"].strip()
                if row.get("panel"):
                    props["olink_panel"] = row["panel"].strip()
                if row.get("lod"):
                    try:
                        props["olink_lod"] = float(row["lod"])
                    except ValueError:
                        logger.warning(
                            f"Invalid LOD value for {uniprot_id}: {row['lod']}"
                        )
                if row.get("lloq"):
                    try:
                        props["olink_lloq"] = float(row["lloq"])
                    except ValueError:
                        logger.warning(
                            f"Invalid LLOQ value for {uniprot_id}: {row['lloq']}"
                        )
                if row.get("cv_intra"):
                    try:
                        props["olink_cv_intra"] = float(row["cv_intra"])
                    except ValueError:
                        logger.warning(
                            f"Invalid CV_intra value for {uniprot_id}: {row['cv_intra']}"
                        )
                if row.get("cv_inter"):
                    try:
                        props["olink_cv_inter"] = float(row["cv_inter"])
                    except ValueError:
                        logger.warning(
                            f"Invalid CV_inter value for {uniprot_id}: {row['cv_inter']}"
                        )
                if row.get("specificity"):
                    try:
                        props["olink_specificity"] = float(row["specificity"])
                    except ValueError:
                        logger.warning(
                            f"Invalid specificity value for {uniprot_id}: {row['specificity']}"
                        )

                # Update protein node
                query = """
                MATCH (p:Protein)
                WHERE p.uniprot_id = $uniprot_id OR p.uniprotID = $uniprot_id
                SET p += $props
                RETURN p.uniprot_id as id
                """
                params = {"uniprot_id": uniprot_id, "props": props}

                result = self.db.execute_query(query, params)
                if result:
                    enriched_count += 1
                    logger.debug(f"Enriched protein {uniprot_id} with Olink data")
                else:
                    logger.warning(
                        f"Protein node not found for UniProt ID: {uniprot_id}"
                    )

        logger.info(f"Enriched {enriched_count} proteins with Olink validation data")
        return enriched_count
