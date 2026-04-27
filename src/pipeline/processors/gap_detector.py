"""Knowledge gap detection from graph structure.

Analyzes the graph after ingestion to surface:
- Isolated entities (degree ≤ 1)
- Sparse communities (low internal edge density)
- Bridge nodes (connecting 3+ clusters)
- Type imbalances (many nodes, few edges for a label)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pipeline.interfaces import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Structured report of knowledge graph gaps."""

    isolated_entities: list[dict[str, str]] = field(default_factory=list)
    sparse_communities: list[dict[str, Any]] = field(default_factory=list)
    bridge_nodes: list[dict[str, Any]] = field(default_factory=list)
    type_imbalances: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_gaps(self) -> int:
        return (
            len(self.isolated_entities)
            + len(self.sparse_communities)
            + len(self.bridge_nodes)
            + len(self.type_imbalances)
        )


class GapDetector:
    """Detect structural gaps in a knowledge graph."""

    def __init__(self, graph: GraphStore) -> None:
        self._graph = graph

    def detect(self, min_community_size: int = 3, cohesion_threshold: float = 0.15) -> GapReport:
        """Run all gap detection checks and return a GapReport."""
        report = GapReport()
        report.isolated_entities = self._find_isolated()
        report.sparse_communities = self._find_sparse_communities(min_community_size, cohesion_threshold)
        report.bridge_nodes = self._find_bridges()
        report.type_imbalances = self._find_type_imbalances()
        logger.info("Gap detection complete: %d total gaps found", report.total_gaps)
        return report

    def _find_isolated(self) -> list[dict[str, str]]:
        """Find entities with degree ≤ 1 (excluding Chunk nodes)."""
        try:
            results = self._graph.execute_query(
                "MATCH (n) WHERE n.name IS NOT NULL AND NOT 'Chunk' IN labels(n) "
                "WITH n, size((n)--()) AS degree WHERE degree <= 1 "
                "RETURN n.name AS name, labels(n)[0] AS type, degree "
                "ORDER BY degree ASC LIMIT 50"
            )
            return [{"name": r["name"], "type": r["type"], "degree": r["degree"]} for r in results]
        except Exception as e:
            logger.warning("Isolated entity detection failed: %s", e)
            return []

    def _find_sparse_communities(self, min_size: int, threshold: float) -> list[dict[str, Any]]:
        """Find communities with low internal edge density."""
        try:
            from pipeline.processors.community_detector import CommunityDetector
            detector = CommunityDetector(self._graph)
            communities = detector.detect_communities(min_community_size=min_size)

            sparse = []
            for comm in communities:
                if comm.size < min_size:
                    continue
                n = comm.size
                max_edges = n * (n - 1) / 2
                if max_edges == 0:
                    continue
                # Count internal edges
                try:
                    result = self._graph.execute_query(
                        "MATCH (a)-[r]-(b) WHERE a.community_id = $cid AND b.community_id = $cid "
                        "AND id(a) < id(b) RETURN count(r) AS edge_count",
                        {"cid": comm.community_id},
                    )
                    actual = result[0]["edge_count"] if result else 0
                except Exception:
                    actual = 0
                cohesion = actual / max_edges
                if cohesion < threshold:
                    sparse.append({
                        "community_id": comm.community_id,
                        "size": comm.size,
                        "cohesion": round(cohesion, 3),
                        "sample_entities": comm.entity_names[:5],
                    })
            return sparse
        except Exception as e:
            logger.warning("Sparse community detection failed: %s", e)
            return []

    def _find_bridges(self) -> list[dict[str, Any]]:
        """Find nodes connecting 3+ different communities."""
        try:
            results = self._graph.execute_query(
                "MATCH (n)--(m) WHERE n.community_id IS NOT NULL AND m.community_id IS NOT NULL "
                "AND n.name IS NOT NULL "
                "WITH n, collect(DISTINCT m.community_id) AS neighbor_communities "
                "WHERE size(neighbor_communities) >= 3 "
                "RETURN n.name AS name, labels(n)[0] AS type, "
                "size(neighbor_communities) AS communities_connected "
                "ORDER BY communities_connected DESC LIMIT 20"
            )
            return [dict(r) for r in results]
        except Exception as e:
            logger.warning("Bridge node detection failed: %s", e)
            return []

    def _find_type_imbalances(self) -> list[dict[str, Any]]:
        """Find entity types with many nodes but disproportionately few edges."""
        try:
            results = self._graph.execute_query(
                "MATCH (n) WHERE n.name IS NOT NULL AND NOT 'Chunk' IN labels(n) "
                "WITH labels(n)[0] AS type, count(n) AS node_count "
                "WHERE node_count >= 5 "
                "RETURN type, node_count ORDER BY node_count DESC"
            )
            imbalances = []
            for r in results:
                label = r["type"]
                try:
                    edge_result = self._graph.execute_query(
                        f"MATCH (n:{label})-[r]-() RETURN count(r) AS edge_count"
                    )
                    edge_count = edge_result[0]["edge_count"] if edge_result else 0
                except Exception:
                    edge_count = 0
                ratio = edge_count / r["node_count"] if r["node_count"] else 0
                if ratio < 1.0:  # fewer edges than nodes = sparse
                    imbalances.append({
                        "type": label,
                        "node_count": r["node_count"],
                        "edge_count": edge_count,
                        "edges_per_node": round(ratio, 2),
                    })
            return imbalances
        except Exception as e:
            logger.warning("Type imbalance detection failed: %s", e)
            return []
