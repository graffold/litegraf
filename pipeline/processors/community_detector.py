"""Community detection for thematic clustering of knowledge graph entities.

Uses NetworkX + python-louvain to extract the graph, run Louvain community
detection, and write community assignments back to the graph database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import community as community_louvain
import networkx as nx

from src.utils.logging_utils import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class Community:
    """A detected community of related entities."""

    community_id: int
    entity_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    summary: str | None = None
    size: int = 0


class CommunityDetector:
    """Detects communities in the knowledge graph using Louvain algorithm.

    Extracts the graph into a NetworkX representation, runs community
    detection, optionally generates LLM summaries, and writes community
    IDs back to entity nodes.
    """

    def __init__(self, db: Any, llm: Any | None = None) -> None:
        """Initialize with a database connection and optional LLM.

        Args:
            db: Database object with ``_execute_cypher(query, params)`` method.
            llm: Optional LLM with an ``invoke(prompt)`` method for summaries.
        """
        self.db = db
        self.llm = llm

    def _extract_graph(self) -> nx.Graph:
        """Extract entity nodes and relationships from the graph database.

        Returns:
            A NetworkX undirected graph with entity nodes and edges.

        Raises:
            ConnectionError: If the database query fails.
        """
        try:
            nodes = self.db._execute_cypher(
                "MATCH (n) WHERE n.name IS NOT NULL "
                "RETURN elementId(n) AS id, n.name AS name"
            )
        except Exception as e:
            msg = f"Failed to extract nodes from graph database: {e}"
            raise ConnectionError(msg) from e

        try:
            edges = self.db._execute_cypher(
                "MATCH (a)-[r]->(b) "
                "WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
                "RETURN elementId(a) AS source, elementId(b) AS target, "
                "type(r) AS rel_type"
            )
        except Exception as e:
            msg = f"Failed to extract relationships from graph database: {e}"
            raise ConnectionError(msg) from e

        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"], name=node["name"])
        for edge in edges:
            if edge["source"] in G and edge["target"] in G:
                G.add_edge(edge["source"], edge["target"], rel_type=edge["rel_type"])

        logger.info(
            f"Extracted graph with {G.number_of_nodes()} nodes "
            f"and {G.number_of_edges()} edges"
        )
        return G

    def detect_communities(self, min_community_size: int = 3) -> list[Community]:
        """Extract graph into NetworkX, run Louvain, return communities.

        Handles disconnected components separately by running Louvain on
        each connected component independently.

        Args:
            min_community_size: Minimum number of entities for a community
                to be included in results.

        Returns:
            List of Community dataclass instances, filtered by min size.

        Raises:
            ConnectionError: If graph extraction fails.
        """
        G = self._extract_graph()

        if G.number_of_nodes() == 0:
            logger.warning("Graph has no nodes, returning empty communities")
            return []

        # Run Louvain on each connected component separately
        node_to_community: dict[str, int] = {}
        community_offset = 0

        for component_nodes in nx.connected_components(G):
            subgraph = G.subgraph(component_nodes)
            if subgraph.number_of_nodes() < 2:
                # Single-node components get their own community ID
                for node_id in component_nodes:
                    node_to_community[node_id] = community_offset
                community_offset += 1
                continue

            partition = community_louvain.best_partition(subgraph)
            # Offset community IDs to avoid collisions across components
            max_id = max(partition.values()) if partition else 0
            for node_id, comm_id in partition.items():
                node_to_community[node_id] = comm_id + community_offset
            community_offset += max_id + 1

        # Group nodes by community
        communities_map: dict[int, Community] = {}
        for node_id, comm_id in node_to_community.items():
            if comm_id not in communities_map:
                communities_map[comm_id] = Community(
                    community_id=comm_id,
                    entity_ids=[],
                    entity_names=[],
                )
            comm = communities_map[comm_id]
            comm.entity_ids.append(node_id)
            name = G.nodes[node_id].get("name", "")
            comm.entity_names.append(name)

        # Set sizes and filter by min_community_size
        for comm in communities_map.values():
            comm.size = len(comm.entity_ids)

        result = [c for c in communities_map.values() if c.size >= min_community_size]
        result.sort(key=lambda c: c.community_id)

        logger.info(
            f"Detected {len(result)} communities with size >= {min_community_size} "
            f"(total partitions: {len(communities_map)})"
        )
        return result

    async def generate_summaries(self, communities: list[Community]) -> list[Community]:
        """Use LLM to generate a text summary for each community.

        If the LLM is not configured or fails for a specific community,
        that community's summary is set to None and processing continues.

        Args:
            communities: List of communities to summarize.

        Returns:
            The same list with ``summary`` fields populated where possible.
        """
        if self.llm is None:
            logger.warning("No LLM configured, skipping community summary generation")
            return communities

        for comm in communities:
            try:
                entity_list = ", ".join(comm.entity_names[:50])
                prompt = (
                    f"Summarize the following community of related entities "
                    f"from a biomedical knowledge graph. "
                    f"Community {comm.community_id} contains {comm.size} entities: "
                    f"{entity_list}.\n\n"
                    f"Provide a concise summary (2-3 sentences) describing "
                    f"the key theme, main entities, and their relationships."
                )
                response = self.llm.invoke(prompt)
                comm.summary = (
                    response.strip()
                    if isinstance(response, str)
                    else str(response).strip()
                )
            except Exception:
                logger.warning(
                    f"Failed to generate summary for community {comm.community_id}",
                    exc_info=True,
                )
                comm.summary = None

        summarized = sum(1 for c in communities if c.summary is not None)
        logger.info(
            f"Generated summaries for {summarized}/{len(communities)} communities"
        )
        return communities

    def write_community_ids(self, communities: list[Community]) -> int:
        """Write community_id property back to entity nodes in the graph DB.

        Args:
            communities: List of communities with entity_ids populated.

        Returns:
            Number of nodes updated.

        Raises:
            ConnectionError: If the database write fails.
        """
        total_updated = 0

        for comm in communities:
            if not comm.entity_ids:
                continue
            try:
                result = self.db._execute_cypher(
                    "UNWIND $entity_ids AS eid "
                    "MATCH (n) WHERE elementId(n) = eid "
                    "SET n.community_id = $community_id "
                    "RETURN count(n) AS updated",
                    {
                        "entity_ids": comm.entity_ids,
                        "community_id": comm.community_id,
                    },
                )
                if result:
                    total_updated += result[0].get("updated", 0)
            except Exception as e:
                msg = (
                    f"Failed to write community_id for community "
                    f"{comm.community_id}: {e}"
                )
                raise ConnectionError(msg) from e

        logger.info(f"Updated community_id on {total_updated} nodes")
        return total_updated

    def write_community_summaries(self, communities: list[Community]) -> int:
        """Write community summaries to the graph database.

        Creates or updates Community nodes with summaries.

        Args:
            communities: List of communities with summaries populated.

        Returns:
            Number of community nodes created/updated.

        Raises:
            ConnectionError: If the database write fails.
        """
        total_updated = 0

        for comm in communities:
            if comm.summary is None:
                continue
            try:
                result = self.db._execute_cypher(
                    "MERGE (c:Community {community_id: $community_id}) "
                    "SET c.summary = $summary, c.size = $size "
                    "RETURN count(c) AS updated",
                    {
                        "community_id": comm.community_id,
                        "summary": comm.summary,
                        "size": comm.size,
                    },
                )
                if result:
                    total_updated += result[0].get("updated", 0)
            except Exception as e:
                msg = f"Failed to write summary for community {comm.community_id}: {e}"
                raise ConnectionError(msg) from e

        logger.info(f"Created/updated {total_updated} Community nodes with summaries")
        return total_updated
