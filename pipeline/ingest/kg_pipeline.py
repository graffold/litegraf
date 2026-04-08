import asyncio
import os
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict

from Bio import Entrez
from langgraph.graph import END, START, StateGraph
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from pipeline.ingest.ingestor import Chunk, ProcessedDocument
from pipeline.ingest.ontology_pipeline import Neo4jBackendAdapter, OntologyPipeline
from pipeline.ingest.sentence_locator import annotate_relationships
from pipeline.ingest.token_chunker import TokenChunker
from pipeline.processors.entity_resolver import EntityResolver
from pipeline.processors.incremental_consolidation import IncrementalConsolidator
from pipeline.processors.ingestion_run_report import (
    ChunkObservation,
    IngestionRunReport,
)
from pipeline.processors.relationship_counter import RelationshipCounter
from src.cache import ingestion_subgraph_cache
from src.cache.redis_cache import get_redis_cache
from src.config import Config
from src.core.context_graph_interfaces import (
    ContextGraphManager as ContextGraphManagerBase,
)
from src.core.context_graph_interfaces import ProvenanceFactory as ProvenanceFactoryBase
from src.core.database import Neo4jDatabase
from src.factories.embedding_factory import get_embedder
from src.factories.llm_factory import get_llm
from src.utils import logging_utils
from src.utils.database_config import DatabaseConfig
from src.utils.unified_retry_utils import UnifiedRetryUtilities

logger = logging_utils.setup_logging()

os.environ["TOKENIZERS_PARALLELISM"] = Config.get_config(
    "TOKENIZERS_PARALLELISM"
).lower()


class PipelineState(TypedDict):
    text: str
    doc_id: str
    chunk_id: str
    index: int
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    error: str


class KGPipeline:
    def __init__(
        self,
        service: str = "local",
        database: str = "olink",
        enable_consolidation: bool = False,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        skip_node_labeling: bool = False,
        context_graph_manager: ContextGraphManagerBase | None = None,
        provenance_factory: ProvenanceFactoryBase | None = None,
        **kwargs,
    ):
        self.service = service  # Store service for later use
        self.database = database
        self.enable_consolidation = enable_consolidation
        self.skip_node_labeling = skip_node_labeling
        self.context_graph_manager = context_graph_manager
        self.provenance_factory = provenance_factory

        # Token-based chunker (replaces word-count Chunker)
        self.chunker = TokenChunker(
            max_tokens=max_tokens, overlap_tokens=overlap_tokens
        )

        # Get unified configuration
        self.config = DatabaseConfig.get_optimized_config("neo4j", "default")
        self.retry_utils = UnifiedRetryUtilities()

        # Initialize batch size from unified configuration
        self.batch_size = self.config.get("batch_size", 50)

        logger.info("🚀 Initialized KGPipeline with unified configuration")
        logger.info(
            f"📊 Batch size: {self.batch_size}, Consolidation: {enable_consolidation}"
        )

        # Neo4j mode with optional consolidation
        self.db = Neo4jDatabase(database=database)
        if enable_consolidation:
            self.db.enable_consolidation()
            logger.info(
                "🎯 Consolidation enabled for Neo4j database - duplicate relationships will be prevented"
            )

        logger.info(f"Initializing KGPipeline with service={service}")
        embed_kwargs = {}
        llm_kwargs = kwargs.copy()

        # Service configuration - removed OpenAI and HuggingFace Hub dependencies
        if service == "sagemaker-llama3":
            llm_type = "sagemaker-llama3"
            embed_type = "huggingface"
            endpoint_name = Config.get_config("SAGEMAKER_ENDPOINT_NAME")
            if not endpoint_name:
                raise ValueError(
                    "SAGEMAKER_ENDPOINT_NAME required for sagemaker-llama3"
                )
            llm_kwargs["endpoint_name"] = endpoint_name
        elif service == "bedrock":
            llm_type = "bedrock"
            embed_type = "huggingface"
            embed_kwargs["model_name"] = "all-mpnet-base-v2"
        else:
            llm_type = "local"
            embed_type = "huggingface"
            embed_kwargs["model_name"] = "all-mpnet-base-v2"

        # Log deprecation warning for removed services
        if service in ["openai", "hf-inference"]:
            logger.warning(
                f"Service '{service}' no longer supported (dependencies removed). Using local models."
            )

        logger.debug(f"Selected LLM type: {llm_type}, Embedder type: {embed_type}")
        self.embedder = get_embedder(embed_type, **embed_kwargs)
        self.llm = get_llm(llm_type, **llm_kwargs)
        self.entities = ["Protein", "Disease", "Entity"]
        self.relations = ["ASSOCIATED_WITH", "CAUSES", "TREATS"]
        self.potential_schema = [
            ("Protein", "ASSOCIATED_WITH", "Disease"),
            ("Protein", "CAUSES", "Disease"),
            ("Protein", "TREATS", "Disease"),
        ]

        # Initialize SimpleKGPipeline
        self.kg_pipeline = None
        try:
            self.kg_pipeline = SimpleKGPipeline(
                llm=self.llm,
                driver=self.db.driver,
                from_pdf=False,
                embedder=self.embedder,
                entities=self.entities,
                relations=self.relations,
                potential_schema=self.potential_schema,
                neo4j_database=self.database,
            )
        except Exception as e:
            logger.error(f"Failed to initialize SimpleKGPipeline: {e}", exc_info=True)
            raise

        # Use config batch size or default to 10 for better throughput on A100
        if self.batch_size < 10:
            self.batch_size = 10
            logger.info(
                f"Increased batch size to {self.batch_size} for better throughput"
            )

        # Initialize ontology pipeline with Neo4j backend adapter
        neo4j_adapter = Neo4jBackendAdapter(self.db)
        self.ontology_pipeline = OntologyPipeline(backend_adapter=neo4j_adapter)
        self.relationship_counter = RelationshipCounter(database=self.database)
        self.entity_resolver = EntityResolver(self.db)
        self.incremental_consolidator = IncrementalConsolidator(
            db=self.db,
            chunk_threshold=25,  # Run consolidation every 25 chunks
            skip_full_resolution=skip_node_labeling,
        )
        self.graph = self._build_graph()

        # Configure Entrez
        self.email = Config.get_config("ENTREZ_EMAIL")
        if (
            not self.email
            or self.email == "your-email@example.com"
            or self.email == "your.email@example.com"
        ):
            logger.warning(
                "Using default/placeholder email for Entrez. This may cause 400 errors from NCBI."
            )
            self.email = "tool.user@example.com"  # Fallback to something generic but valid format

        Entrez.email = self.email

        api_key = Config.get_config("ENTREZ_API_KEY")
        if api_key and api_key != "your-entrez-api-key":
            Entrez.api_key = api_key
        else:
            Entrez.api_key = None
            logger.info(
                "No valid Entrez API key found. Requests will be rate limited to 3/second."
            )

        self.failed_chunks: list[Any] = []
        self.run_report = IngestionRunReport(service=service, database=database)
        self.ingestion_job_id: str | None = None
        self.ingested_at: str | None = None

        # Redis cache for ephemeral sub-graph tracking (graceful degradation)
        self._redis = get_redis_cache(logger=logger)

    def _ensure_chunk_node(
        self,
        chunk_id: str,
        text: str,
        doc_id: str,
    ) -> None:
        """Create or update a Chunk node in Neo4j with text and job metadata.

        This ensures the chunk text is queryable for summary/Q&A even before
        the embedding pipeline runs.
        """
        try:
            self.db._execute_cypher(
                "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                "SET c.text = $text, "
                "    c.doc_id = $doc_id, "
                "    c.ingestion_job_id = $ingestion_job_id, "
                "    c.ingested_at = $ingested_at",
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "doc_id": doc_id,
                    "ingestion_job_id": self.ingestion_job_id,
                    "ingested_at": self.ingested_at,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to create Chunk node {chunk_id}: {e}")

    def _store_via_context_graph(
        self,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        doc_id: str,
        chunk_id: str,
    ) -> None:
        """Store nodes and relationships through ContextGraphManager.

        Builds a ProvenanceChain for each relationship and routes writes
        through ContextGraphManager for provenance, temporal, and confidence
        handling. Falls back to direct writes if context_graph_manager or
        provenance_factory is not configured.
        """
        assert self.context_graph_manager is not None
        assert self.provenance_factory is not None
        now = datetime.now(UTC)

        # Store nodes via ContextGraphManager
        for node in nodes:
            node_id = node["id"]
            node_type = node.get("type", "Entity")
            props = {k: v for k, v in node.items() if k not in ("id", "type")}
            try:
                self.context_graph_manager.store_node(
                    self.db,
                    node_id,
                    node_type,
                    props,
                    now,
                )
            except Exception as e:
                logger.warning(
                    f"ContextGraphManager.store_node failed for {node_id}, "
                    f"falling back to direct write: {e}"
                )
                # Fallback: let the existing db method handle it
                self.db._store_in_neo4j(
                    [node],
                    [],
                    chunk_id,
                    ingestion_job_id=self.ingestion_job_id,
                    ingested_at=self.ingested_at,
                )

        # Store relationships via ContextGraphManager with provenance
        for rel in relationships:
            source_id = rel["source_id"]
            target_id = rel["target_id"]
            rel_type = rel["type"]
            props = {
                k: v
                for k, v in rel.items()
                if k not in ("source_id", "target_id", "type")
            }
            props["chunk_id"] = chunk_id
            if self.ingestion_job_id:
                props["ingestion_job_id"] = self.ingestion_job_id

            try:
                provenance = self.provenance_factory.build_from_kg_pipeline(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    extraction_method="llm",
                    ingestion_job_id=self.ingestion_job_id or "",
                    timestamp=now,
                )
                self.context_graph_manager.store_relationship(
                    self.db,
                    source_id,
                    target_id,
                    rel_type,
                    props,
                    provenance,
                    now,
                )
            except Exception as e:
                logger.warning(
                    f"ContextGraphManager.store_relationship failed for "
                    f"{source_id}->{target_id}, falling back to direct write: {e}"
                )
                self.db._store_in_neo4j(
                    [],
                    [rel],
                    chunk_id,
                    ingestion_job_id=self.ingestion_job_id,
                    ingested_at=self.ingested_at,
                )

    def _store_abstract_metadata(
        self,
        pmid: str,
        abstract_text: str,
        title: str,
        chunk_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store abstract metadata node - abstracts are metadata containers, not text containers."""
        try:
            metadata = metadata or {}

            # Abstract nodes are metadata containers - they don't store full text
            # Full text goes in Chunk nodes. Abstract nodes just store metadata and reference chunks.

            abstract_query = """
            MERGE (a:Abstract {pmid: $pmid})
            SET a.title = $title,
                a.authors = $authors,
                a.journal = $journal,
                a.publication_date = $publication_date,
                a.doi = $doi,
                a.keywords = $keywords,
                a.ingestion_job_id = $ingestion_job_id,
                a.ingested_at = $ingested_at
            """

            params = {
                "pmid": pmid,
                "title": title,
                "authors": metadata.get("authors", []),
                "journal": metadata.get("journal", ""),
                "publication_date": metadata.get("publication_date", ""),
                "doi": metadata.get("doi", ""),
                "keywords": metadata.get("keywords", []),
                "ingestion_job_id": self.ingestion_job_id,
                "ingested_at": self.ingested_at,
            }

            self.db._execute_cypher(abstract_query, params)
            logger.debug(f"Stored abstract metadata node {pmid}")
        except Exception as e:
            logger.error(f"Failed to store abstract metadata {pmid}: {e}")

    async def _extract_kg_with_llm(self, text: str) -> dict[str, Any]:
        """Extract knowledge graph using LLM."""
        try:
            # Use a custom prompt that works with our LLM setup
            prompt = f"""Extract entities and relationships from the following text. Return a JSON object with this exact structure:

{{
    "nodes": [
        {{
            "id": "unique_entity_id",
            "name": "entity_name",
            "type": "entity_type"
        }}
    ],
    "relationships": [
        {{
            "source_id": "source_entity_id",
            "target_id": "target_entity_id",
            "type": "relationship_type",
            "source_sentence": "the exact sentence from the text that supports this relationship"
        }}
    ]
}}

Entity types should be: Protein, Disease, or Entity.
Relationship types should be: ASSOCIATED_WITH, CAUSES, TREATS, or RELATED_TO.
For each relationship, "source_sentence" MUST be the verbatim sentence from the input text that supports the extracted triple.

IMPORTANT: Do NOT extract generic terms as standalone entities. Examples of terms to IGNORE:
- "Isoform", "Isoform 1", "Isoform 2", etc. (unless fully qualified like "Isoform 1 of Protein X")
- "Fragment", "Subunit", "Domain", "Region"
- "Protein", "Enzyme", "Molecule", "Compound"
- "Disease", "Syndrome", "Disorder"
- "Patient", "Subject", "Cohort", "Group"
- "Study", "Analysis", "Data", "Result"

Only extract specific, named entities (e.g., "BRCA1", "Alzheimer's Disease", "Interleukin-6").

Text: {text}

Return only valid JSON:"""

            # DEBUG: Log the prompt being sent to LLM
            logger.info("=" * 80)
            logger.info("DEBUG: Sending prompt to LLM")
            logger.info(f"Input text length: {len(text)} characters")
            logger.info(f"Input text preview: {text[:300]}...")
            logger.info(f"Prompt length: {len(prompt)} characters")
            logger.debug(f"Full prompt:\n{prompt}")
            logger.info("=" * 80)

            # Call the LLM directly
            logger.info("Invoking LLM...")
            response = await self.llm.ainvoke(prompt)
            logger.info("LLM invocation completed")

            # Handle response being an object (CustomLLMWrapper) or string
            if hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            # DEBUG: Log the raw LLM response
            logger.info("=" * 80)
            logger.info("DEBUG: Received LLM response")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response text length: {len(response_text)} characters")
            logger.info(
                f"Response text preview (first 500 chars):\n{response_text[:500]}"
            )
            if len(response_text) > 500:
                logger.info(
                    f"Response text preview (last 500 chars):\n{response_text[-500:]}"
                )
            logger.debug(f"Full response text:\n{response_text}")
            logger.info("=" * 80)

            # Parse the JSON response
            try:
                import json
                import re

                logger.info("Attempting to parse LLM response as JSON...")

                # IMPROVED: Handle markdown code blocks more robustly
                json_str = response_text
                if "```json" in response_text:
                    logger.info("Found ```json code block, extracting...")
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    logger.info("Found ``` code block, searching for JSON...")
                    # Try to find the block that looks like JSON
                    parts = response_text.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("{") and "nodes" in part:
                            json_str = part
                            logger.info("Found JSON-like content in code block")
                            break

                # Try to find JSON object if not in code block or if code block extraction failed
                if not json_str.startswith("{") or not json_str.endswith("}"):
                    logger.info(
                        "JSON not properly bounded, searching for JSON object..."
                    )
                    match = re.search(r"\{.*\}", json_str, re.DOTALL)
                    if match:
                        json_str = match.group()
                        logger.info("Found JSON object via regex")

                logger.info(f"Extracted JSON string length: {len(json_str)} characters")
                logger.debug(f"JSON string to parse:\n{json_str}")

                # Clean up any potential trailing commas or comments if possible (basic cleanup)
                # json.loads is strict, so we rely on the LLM being good, but we can try to be lenient

                try:
                    result = json.loads(json_str)
                    logger.info("Successfully parsed JSON response")
                except json.JSONDecodeError as parse_error:
                    # Last ditch effort: sometimes LLMs put comments // in JSON
                    # We won't implement a full loose parser here, but we will log the failure clearly
                    logger.error(f"JSON parsing failed: {parse_error}")
                    logger.error(f"Failed to parse this JSON string:\n{json_str}")
                    raise

                # Validate the structure
                nodes = result.get("nodes", [])
                relationships = result.get("relationships", [])

                logger.info(
                    f"Parsed result contains {len(nodes)} nodes and {len(relationships)} relationships"
                )

                # Ensure nodes have required fields
                validated_nodes = []
                for node in nodes:
                    if isinstance(node, dict) and "id" in node and "name" in node:
                        validated_node = {
                            "id": str(node["id"]),
                            "name": str(node["name"]),
                            "type": str(node.get("type", "Entity")),
                        }
                        validated_nodes.append(validated_node)
                        logger.debug(f"Validated node: {validated_node}")
                    else:
                        logger.warning(
                            f"Dropping invalid node (missing id/name): {node}"
                        )

                # Ensure relationships have required fields
                validated_relationships = []
                for rel in relationships:
                    if (
                        isinstance(rel, dict)
                        and "source_id" in rel
                        and "target_id" in rel
                    ):
                        validated_rel = {
                            "source_id": str(rel["source_id"]),
                            "target_id": str(rel["target_id"]),
                            "type": str(rel.get("type", "RELATED_TO")),
                        }
                        if (
                            rel.get("source_sentence")
                            and Config.ENABLE_SENTENCE_PROVENANCE
                        ):
                            validated_rel["source_sentence"] = str(
                                rel["source_sentence"]
                            )
                        validated_relationships.append(validated_rel)
                        logger.debug(f"Validated relationship: {validated_rel}")
                    else:
                        logger.warning(
                            f"Dropping invalid relationship (missing source/target): {rel}"
                        )

                if not validated_nodes and not validated_relationships:
                    logger.warning("=" * 80)
                    logger.warning("WARNING: Extraction resulted in 0 items!")
                    logger.warning(f"Raw LLM response:\n{response_text}")
                    logger.warning(f"Parsed result: {result}")
                    logger.warning("=" * 80)

                logger.info(
                    f"Final result: {len(validated_nodes)} validated nodes, {len(validated_relationships)} validated relationships"
                )
                return {
                    "nodes": validated_nodes,
                    "relationships": validated_relationships,
                }

            except json.JSONDecodeError as e:
                logger.error("=" * 80)
                logger.error("JSON PARSING ERROR")
                logger.error(f"Error: {e}")
                logger.error(f"Raw response text:\n{response_text}")
                logger.error(f"Attempted to parse:\n{json_str}")
                logger.error("=" * 80)
                return {
                    "nodes": [],
                    "relationships": [],
                    "error": f"JSON parsing failed: {e}",
                }

        except Exception as e:
            logger.error("=" * 80)
            logger.error("EXTRACTION FAILED")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {e}")
            logger.error("=" * 80)
            logger.error("Full traceback:", exc_info=True)
            # Fallback to empty result for now
            return {"nodes": [], "relationships": [], "error": str(e)}

    def _cleanup_graphrag_state(self):
        """Clean up GraphRAG temporary labels and state that might cause merge conflicts."""
        try:
            # Remove temporary GraphRAG labels that can cause merge conflicts
            cleanup_queries = [
                "MATCH (n:__KGBuilder__) REMOVE n:__KGBuilder__",
                "MATCH (n:__Entity__) WHERE size(labels(n)) > 1 REMOVE n:__Entity__",  # Only remove if node has other labels
                "MATCH ()-[r]->() WHERE type(r) STARTS WITH '__' DELETE r",  # Remove temporary relationships
            ]

            for query in cleanup_queries:
                try:
                    self.db._execute_cypher(query)
                except Exception as e:
                    logger.debug(f"Cleanup query '{query}' warning: {e}")
        except Exception as e:
            logger.debug(f"GraphRAG state cleanup warning: {e}")
            # Don't fail the entire process for cleanup issues

    def _build_graph(self):
        graph = StateGraph(PipelineState)

        async def extract_kg(state: PipelineState) -> dict[str, Any]:
            try:
                # Always use custom extraction for better control and reliability with Llama 3
                # SimpleKGPipeline has proven unstable with Bedrock Llama 3
                t0 = time.perf_counter()
                result = await self._extract_kg_with_llm(state["text"])
                result["_extraction_time_s"] = time.perf_counter() - t0
                result["_raw_nodes"] = len(result.get("nodes", []))
                result["_raw_relationships"] = len(result.get("relationships", []))

                logger.debug(
                    f"Extracted {len(result['nodes'])} nodes and {len(result['relationships'])} relationships for chunk {state['chunk_id']}"
                )
                return result
            except Exception as e:
                logger.error(
                    f"KG extraction failed for chunk {state['chunk_id']}: {e}",
                    exc_info=True,
                )
                return {"nodes": [], "relationships": [], "error": str(e)}

        async def filter_ontology(state: PipelineState) -> dict[str, Any]:
            pre_nodes = len(state.get("nodes", []))
            pre_rels = len(state.get("relationships", []))
            result = await self.ontology_pipeline.filter_ontology(dict(state))
            result["_pre_filter_nodes"] = pre_nodes
            result["_pre_filter_relationships"] = pre_rels
            result["_post_filter_nodes"] = len(result.get("nodes", []))
            result["_post_filter_relationships"] = len(result.get("relationships", []))
            return result

        async def link_protein_entities(state: PipelineState) -> dict[str, Any]:
            """Link extracted protein entities to existing Protein nodes with UniProt IDs."""
            if state.get("error"):
                return {
                    "nodes": state["nodes"],
                    "relationships": state["relationships"],
                }
            try:
                # Find protein entities in the extracted nodes
                protein_nodes = [
                    node for node in state["nodes"] if node.get("type") == "Protein"
                ]

                if not protein_nodes:
                    return {
                        "nodes": state["nodes"],
                        "relationships": state["relationships"],
                    }  # No protein entities to link

                linked_nodes = []
                linked_relationships = list(
                    state["relationships"]
                )  # Copy existing relationships
                proteins_resolved = 0
                proteins_unresolved = 0

                for protein_node in protein_nodes:
                    protein_name = protein_node.get("name", "").strip()
                    if not protein_name:
                        linked_nodes.append(protein_node)
                        continue

                    # Try to find existing Protein node identifier
                    existing_node_id = await self._find_uniprot_id_for_protein(
                        protein_name
                    )

                    if existing_node_id:
                        # DON'T create a new protein node - use existing one instead
                        # Update any relationships that reference this protein_node["id"] to use the existing node identifier
                        for relationship in linked_relationships:
                            if relationship.get("source_id") == protein_node["id"]:
                                relationship["source_id"] = existing_node_id
                            if relationship.get("target_id") == protein_node["id"]:
                                relationship["target_id"] = existing_node_id

                        proteins_resolved += 1
                        logger.info(
                            f"Mapped extracted protein '{protein_name}' to existing protein node {existing_node_id} - skipping duplicate creation"
                        )
                        # Skip adding this node to linked_nodes (it won't be created)
                    else:
                        # No existing protein found - create new one with canonical properties
                        enhanced_protein_node = protein_node.copy()

                        # Set canonical UniProt properties
                        enhanced_protein_node["uniprot_gene_name"] = protein_name
                        # uniprot_id will remain None for now (until manual mapping or future enhancement)

                        proteins_unresolved += 1
                        linked_nodes.append(enhanced_protein_node)
                        logger.debug(
                            f"No existing protein found for '{protein_name}' - will create new node with canonical properties"
                        )

                return {
                    "nodes": linked_nodes,
                    "relationships": linked_relationships,
                    "_proteins_resolved": proteins_resolved,
                    "_proteins_unresolved": proteins_unresolved,
                }

            except Exception as e:
                logger.error(
                    f"Protein entity linking failed for chunk {state['chunk_id']}: {e}",
                    exc_info=True,
                )
                return {
                    "nodes": state["nodes"],
                    "relationships": state["relationships"],
                }  # Return original state on error

        graph.add_node("extract_kg", extract_kg)
        graph.add_node("filter_ontology", filter_ontology)
        graph.add_node("link_protein_entities", link_protein_entities)
        graph.add_edge(START, "extract_kg")
        graph.add_edge("extract_kg", "filter_ontology")
        graph.add_edge("filter_ontology", "link_protein_entities")
        graph.add_edge("link_protein_entities", END)
        return graph.compile()

    async def _find_uniprot_id_for_protein(self, protein_name: str) -> str | None:
        """Find existing protein node identifier for linking relationships."""
        try:
            # Normalize protein name for matching
            protein_name.strip().upper()

            # Query for existing Protein nodes that match this name
            # Return the node identifier that should be used for relationship linking
            # Match against standardized UniProt fields only
            query = """
            MATCH (p:Protein)
            WHERE p.uniprot_gene_name = $protein_name OR
                  p.uniprot_protein_description = $protein_name OR
                  p.uniprot_id = $protein_name OR
                  toLower(p.uniprot_protein_description) = toLower($protein_name) OR
                  toLower(p.uniprot_gene_name) = toLower($protein_name)
            RETURN COALESCE(p.uniprot_gene_name, p.uniprot_protein_description) AS node_identifier,
                   p.uniprot_id,
                   p.uniprot_gene_name
            LIMIT 1
            """
            results = self.db._execute_cypher(query, {"protein_name": protein_name})

            if results and len(results) > 0:
                node_identifier = results[0].get("node_identifier")
                uniprot_id = results[0].get("uniprot_id")
                uniprot_gene_name = results[0].get("gene_name") or results[0].get(
                    "uniprot_gene_name"
                )
                results[0].get("description")
                if node_identifier:
                    priority_type = "canonical" if uniprot_id else "existing"
                    logger.info(
                        f"Found {priority_type} protein '{protein_name}' -> '{node_identifier}' (uniprot_id: {uniprot_id}, gene_name: {uniprot_gene_name})"
                    )
                    return str(node_identifier)

            # Try fuzzy matching if exact match fails
            # First, try compound protein name splitting for cases like "TLR4/NF-κB"
            compound_proteins = self._extract_individual_proteins_from_compound(
                protein_name
            )
            for individual_protein in compound_proteins:
                if (
                    individual_protein != protein_name
                ):  # Only try if it's different from original
                    individual_query = """
                    MATCH (p:Protein)
                    WHERE p.uniprot_gene_name = $protein_name OR
                          p.uniprot_protein_description = $protein_name OR
                          p.uniprot_id = $protein_name OR
                          toLower(p.uniprot_protein_description) = toLower($protein_name) OR
                          toLower(p.uniprot_gene_name) = toLower($protein_name)
                    RETURN COALESCE(p.uniprot_gene_name, p.uniprot_protein_description) AS node_identifier,
                           p.uniprot_id,
                           p.uniprot_gene_name
                    LIMIT 1
                    """
                    individual_results = self.db._execute_cypher(
                        individual_query, {"protein_name": individual_protein.strip()}
                    )

                    if individual_results and len(individual_results) > 0:
                        node_identifier = individual_results[0].get("node_identifier")
                        uniprot_id = individual_results[0].get("p.uniprot_id")
                        uniprot_gene_name = individual_results[0].get(
                            "p.uniprot_gene_name"
                        )
                        if node_identifier:
                            logger.debug(
                                f"Found existing protein '{individual_protein}' from compound name '{protein_name}' with identifier {node_identifier} (uniprot_id: {uniprot_id}, uniprot_gene_name: {uniprot_gene_name})"
                            )
                            return str(node_identifier)

            # Try case-insensitive exact match on uniprot_protein_description for descriptive names
            description_exact_query = """
            MATCH (p:Protein)
            WHERE p.uniprot_id IS NOT NULL
              AND toLower(COALESCE(p.uniprot_protein_description, '')) = toLower($protein_name)
            RETURN id(p) AS node_identifier, p.uniprot_id, p.uniprot_gene_name
            ORDER BY p.uniprot_id
            LIMIT 1
            """
            description_results = self.db._execute_cypher(
                description_exact_query, {"protein_name": protein_name}
            )

            if description_results and len(description_results) > 0:
                node_identifier = description_results[0].get("node_identifier")
                uniprot_id = description_results[0].get("p.uniprot_id")
                uniprot_gene_name = description_results[0].get("p.uniprot_gene_name")
                if node_identifier:
                    logger.info(
                        f"Case-insensitive description match: '{protein_name}' -> canonical protein {node_identifier} (uniprot_id: {uniprot_id}, gene: {uniprot_gene_name})"
                    )
                    return str(node_identifier)

            # Try mapping descriptive protein names to canonical gene symbols using UniProt data
            canonical_gene_symbol = await self._find_canonical_gene_symbol(protein_name)
            if canonical_gene_symbol and canonical_gene_symbol != protein_name:
                canonical_query = """
                MATCH (p:Protein)
                WHERE p.uniprot_gene_name = $canonical_symbol OR
                      p.gene_name = $canonical_symbol OR
                      p.name = $canonical_symbol
                RETURN id(p) AS node_identifier, p.uniprot_id, p.uniprot_gene_name
                LIMIT 1
                """
                canonical_results = self.db._execute_cypher(
                    canonical_query, {"canonical_symbol": canonical_gene_symbol.upper()}
                )

                if canonical_results and len(canonical_results) > 0:
                    node_identifier = canonical_results[0].get("node_identifier")
                    uniprot_id = canonical_results[0].get("p.uniprot_id")
                    uniprot_gene_name = canonical_results[0].get("p.uniprot_gene_name")
                    if node_identifier:
                        logger.info(
                            f"Mapped descriptive protein name '{protein_name}' to canonical gene symbol '{canonical_gene_symbol}' -> existing protein {node_identifier} (uniprot_id: {uniprot_id})"
                        )
                        return str(node_identifier)

            # Standard fuzzy matching as fallback - prioritize canonical proteins with UniProt IDs
            # First try to find proteins with UniProt IDs (canonical proteins)
            fuzzy_query_canonical = """
            MATCH (p:Protein)
            WHERE (toLower(p.name) CONTAINS toLower($protein_name) OR
                   toLower(p.uniprot_gene_name) CONTAINS toLower($protein_name) OR
                   toLower(p.gene_name) CONTAINS toLower($protein_name))
              AND p.uniprot_id IS NOT NULL
            RETURN id(p) AS node_identifier, p.uniprot_id, p.uniprot_gene_name
            ORDER BY p.uniprot_id
            LIMIT 1
            """
            fuzzy_results = self.db._execute_cypher(
                fuzzy_query_canonical, {"protein_name": protein_name}
            )

            # If no canonical protein found, fall back to any matching protein
            if not fuzzy_results or len(fuzzy_results) == 0:
                fuzzy_query = """
                MATCH (p:Protein)
                WHERE toLower(p.name) CONTAINS toLower($protein_name) OR
                      toLower(p.uniprot_gene_name) CONTAINS toLower($protein_name) OR
                      toLower(p.gene_name) CONTAINS toLower($protein_name)
                RETURN id(p) AS node_identifier, p.uniprot_id, p.uniprot_gene_name
                LIMIT 1
                """
                fuzzy_results = self.db._execute_cypher(
                    fuzzy_query, {"protein_name": protein_name}
                )

            if fuzzy_results and len(fuzzy_results) > 0:
                node_identifier = fuzzy_results[0].get("node_identifier")
                uniprot_id = fuzzy_results[0].get("p.uniprot_id")
                uniprot_gene_name = fuzzy_results[0].get("p.uniprot_gene_name")
                if node_identifier:
                    priority_type = "canonical" if uniprot_id else "duplicate"
                    logger.info(
                        f"Found {priority_type} protein '{protein_name}' via fuzzy match with identifier {node_identifier} (uniprot_id: {uniprot_id}, uniprot_gene_name: {uniprot_gene_name})"
                    )
                    return str(node_identifier)

            return None

        except Exception as e:
            logger.debug(f"Error finding UniProt ID for protein '{protein_name}': {e}")
            return None

    def _extract_individual_proteins_from_compound(
        self, protein_name: str
    ) -> list[str]:
        """Extract individual protein names from compound protein names like 'TLR4/NF-κB'."""
        # Common separators used in compound protein names
        separators = ["/", "-", " and ", "&", "+", ":", ";"]

        # Start with the original name
        proteins = [protein_name]

        # Split by each separator
        for separator in separators:
            new_proteins = []
            for protein in proteins:
                if separator in protein:
                    # Split and clean up
                    parts = [part.strip() for part in protein.split(separator)]
                    # Only include parts that look like protein names (not too short)
                    parts = [
                        part for part in parts if len(part) >= 2 and not part.isdigit()
                    ]
                    new_proteins.extend(parts)
                else:
                    new_proteins.append(protein)
            proteins = new_proteins

        # Remove duplicates and sort by length (longer names first)
        unique_proteins = list(set(proteins))
        unique_proteins.sort(key=len, reverse=True)

        return unique_proteins

    async def _find_canonical_gene_symbol(self, protein_name: str) -> str | None:
        """Find canonical gene symbol for descriptive protein names using UniProt mapping data."""
        try:
            # Load UniProt mapping if not already loaded
            if not hasattr(self, "_uniprot_mapping"):
                self._uniprot_mapping = self._load_uniprot_mapping()

            if not self._uniprot_mapping:
                return None

            # Normalize the input protein name
            normalized_name = protein_name.strip().lower()

            # Direct exact match on protein name
            for uniprot_id, data in self._uniprot_mapping.items():
                protein_names = data.get("protein_names", [])
                for name in protein_names:
                    if name.lower() == normalized_name:
                        gene_symbol = data.get("gene_symbol", "")
                        if gene_symbol:
                            logger.debug(
                                f"Found exact match: '{protein_name}' -> '{gene_symbol}' (UniProt: {uniprot_id})"
                            )
                            return gene_symbol

            # Fuzzy matching for partial matches
            for uniprot_id, data in self._uniprot_mapping.items():
                protein_names = data.get("protein_names", [])
                for name in protein_names:
                    # Check if the input name is contained in the UniProt protein name
                    if (
                        normalized_name in name.lower() and len(normalized_name) > 10
                    ) or (name.lower() in normalized_name and len(name) > 10):
                        gene_symbol = data.get("gene_symbol", "")
                        if gene_symbol:
                            logger.debug(
                                f"Found fuzzy match: '{protein_name}' -> '{gene_symbol}' (UniProt: {uniprot_id}, matched name: '{name}')"
                            )
                            return gene_symbol

            return None

        except Exception as e:
            logger.debug(
                f"Error finding canonical gene symbol for '{protein_name}': {e}"
            )
            return None

    def _load_uniprot_mapping(self) -> dict[str, dict[str, Any]]:
        """Load UniProt mapping data for protein name resolution."""
        try:
            import csv
            import os

            # Path to UniProt mapping file
            uniprot_file = os.path.join(
                os.path.dirname(__file__), "..", "utils", "uniprot_ids_human.csv"
            )

            if not os.path.exists(uniprot_file):
                logger.warning(f"UniProt mapping file not found: {uniprot_file}")
                return {}

            mapping = {}
            with open(uniprot_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uniprot_id = row.get("uniprot_id", "").strip()
                    gene_symbol = row.get("gene_symbol", "").strip()
                    protein_name = row.get("protein_name", "").strip()

                    if uniprot_id and gene_symbol:
                        mapping[uniprot_id] = {
                            "gene_symbol": gene_symbol,
                            "protein_names": [protein_name] if protein_name else [],
                        }

            logger.debug(f"Loaded {len(mapping)} UniProt protein mappings")
            return mapping

        except Exception as e:
            logger.warning(f"Failed to load UniProt mapping: {e}")
            return {}

    async def _process_single_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Process a single item from the batch."""
        state: PipelineState = {
            "text": item["text"],
            "doc_id": item["doc_id"],
            "chunk_id": str(item["chunk_id"]),
            "index": item["index"],
            "nodes": [],
            "relationships": [],
            "error": "",
        }

        result_data = {
            "nodes": [],
            "relationships": [],
            "doc_id": item["doc_id"],
            "chunk_id": str(item["chunk_id"]),
            "error": None,
        }

        try:
            result = await self.graph.ainvoke(state)
        except Exception as graph_error:
            # Handle APOC procedure errors specifically
            error_msg = str(graph_error)
            if "apoc.refactor.mergeNodes" in error_msg and (
                "not found" in error_msg or "NotFoundException" in error_msg
            ):
                logger.warning(
                    f"APOC merge nodes error for chunk {item['chunk_id']}: {error_msg}. "
                    f"Node was likely deleted during previous merge operations. Attempting recovery..."
                )
                # Try to recover by cleaning GraphRAG state and retrying once
                try:
                    # Note: _cleanup_graphrag_state is sync and might affect other concurrent tasks if they rely on state
                    await asyncio.sleep(0.5)
                    result = await self.graph.ainvoke(state)
                    logger.info(
                        f"Successfully recovered chunk {item['chunk_id']} after retry"
                    )
                except Exception as retry_error:
                    logger.warning(
                        f"Recovery failed for chunk {item['chunk_id']}: {retry_error}. Skipping."
                    )
                    self.failed_chunks.append(
                        {
                            "chunk_id": item["chunk_id"],
                            "doc_id": item["doc_id"],
                            "error": f"APOC merge failed, retry failed: {retry_error}",
                        }
                    )
                    result_data["error"] = (
                        f"APOC merge failed, retry failed: {retry_error}"
                    )
                    self.run_report.record_chunk(
                        ChunkObservation(
                            chunk_id=str(item["chunk_id"]),
                            doc_id=item["doc_id"],
                            error_category=self.run_report.classify_error(
                                str(retry_error)
                            ),
                            error_detail=f"APOC merge failed, retry failed: {retry_error}",
                        )
                    )
                    return result_data
            elif "ProcedureCallFailed" in error_msg and "apoc" in error_msg.lower():
                logger.warning(
                    f"APOC procedure error for chunk {item['chunk_id']}: {error_msg}. "
                    f"This may be due to database conflicts. Skipping chunk."
                )
                self.failed_chunks.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "doc_id": item["doc_id"],
                        "error": str(graph_error),
                    }
                )
                result_data["error"] = str(graph_error)
                self.run_report.record_chunk(
                    ChunkObservation(
                        chunk_id=str(item["chunk_id"]),
                        doc_id=item["doc_id"],
                        error_category=self.run_report.classify_error(str(graph_error)),
                        error_detail=str(graph_error),
                    )
                )
                return result_data
            else:
                logger.error(
                    f"Graph execution failed for chunk {item['chunk_id']}: {graph_error}",
                    exc_info=True,
                )
                self.failed_chunks.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "doc_id": item["doc_id"],
                        "error": str(graph_error),
                    }
                )
                result_data["error"] = str(graph_error)
                self.run_report.record_chunk(
                    ChunkObservation(
                        chunk_id=str(item["chunk_id"]),
                        doc_id=item["doc_id"],
                        error_category=self.run_report.classify_error(str(graph_error)),
                        error_detail=str(graph_error),
                    )
                )
                return result_data

        if result.get("error"):
            logger.error(f"Error for chunk {item['chunk_id']}: {result['error']}")
            self.failed_chunks.append(
                {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "error": result["error"],
                }
            )
            obs = ChunkObservation(
                chunk_id=str(item["chunk_id"]),
                doc_id=item["doc_id"],
                extraction_error=result["error"],
                error_category=self.run_report.classify_error(result["error"]),
                error_detail=result["error"],
            )
            self.run_report.record_chunk(obs)
            result_data["error"] = result["error"]
            return result_data
        if result["nodes"] or result["relationships"]:
            # Add source_doc to relationships for tracking
            enhanced_relationships = []
            for rel in result["relationships"]:
                enhanced_rel = rel.copy()
                enhanced_rel["source_doc"] = item["doc_id"]
                enhanced_relationships.append(enhanced_rel)

            # Add publication year to nodes if available in metadata
            enhanced_nodes = []
            publication_year = item.get("metadata", {}).get("publication_year")
            for node in result["nodes"]:
                enhanced_node = node.copy()
                if publication_year is not None:
                    enhanced_node["publication_year"] = publication_year
                enhanced_nodes.append(enhanced_node)

            # Locate source sentences and record char-span provenance
            if Config.ENABLE_SENTENCE_PROVENANCE:
                annotate_relationships(
                    enhanced_relationships, str(item["chunk_id"]), item["text"]
                )

            # Store results (run in thread to avoid blocking the event loop)
            try:
                if self.context_graph_manager and self.provenance_factory:
                    # Route writes through ContextGraphManager for provenance/temporal/confidence
                    await asyncio.to_thread(
                        self._store_via_context_graph,
                        enhanced_nodes,
                        enhanced_relationships,
                        item["doc_id"],
                        str(item["chunk_id"]),
                    )
                else:
                    # Fall back to direct Neo4j writes (existing behavior)
                    await asyncio.to_thread(
                        self.db._store_in_neo4j,
                        enhanced_nodes,
                        enhanced_relationships,
                        item["chunk_id"],
                        ingestion_job_id=self.ingestion_job_id,
                        ingested_at=self.ingested_at,
                    )

                # Also create/update the Chunk node with text so summary/Q&A can find it
                await asyncio.to_thread(
                    self._ensure_chunk_node,
                    item["chunk_id"],
                    item["text"],
                    item["doc_id"],
                )

                # Track node/edge IDs in Redis ephemeral sub-graph cache
                if self.ingestion_job_id:
                    ingestion_subgraph_cache.add_nodes(
                        self._redis,
                        self.ingestion_job_id,
                        [n["id"] for n in enhanced_nodes],
                    )
                    ingestion_subgraph_cache.add_edges(
                        self._redis,
                        self.ingestion_job_id,
                        [
                            f"{r['source_id']}:{r['type']}:{r['target_id']}"
                            for r in enhanced_relationships
                        ],
                    )
            except Exception as db_error:
                logger.error(
                    f"Failed to store results for chunk {item['chunk_id']}: {db_error}"
                )
                self.failed_chunks.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "doc_id": item["doc_id"],
                        "error": f"DB storage failed: {db_error}",
                    }
                )
                result_data["error"] = f"DB storage failed: {db_error}"
                self.run_report.record_chunk(
                    ChunkObservation(
                        chunk_id=str(item["chunk_id"]),
                        doc_id=item["doc_id"],
                        raw_nodes=result.get("_raw_nodes", 0),
                        raw_relationships=result.get("_raw_relationships", 0),
                        error_category=self.run_report.classify_error(
                            f"DB storage failed: {db_error}"
                        ),
                        error_detail=f"DB storage failed: {db_error}",
                    )
                )
                return result_data

        logger.info(
            f"Stored {len(result['nodes'])} nodes and {len(result['relationships'])} relationships for chunk {item['chunk_id']}"
        )

        # Record chunk observation for the run report
        raw_n = result.get("_raw_nodes", len(result["nodes"]))
        raw_r = result.get("_raw_relationships", len(result["relationships"]))
        pre_filt_n = result.get("_pre_filter_nodes", raw_n)
        post_filt_n = result.get("_post_filter_nodes", len(result["nodes"]))
        post_filt_r = result.get(
            "_post_filter_relationships", len(result["relationships"])
        )
        obs = ChunkObservation(
            chunk_id=str(item["chunk_id"]),
            doc_id=item["doc_id"],
            raw_nodes=raw_n,
            raw_relationships=raw_r,
            validated_nodes=pre_filt_n,
            validated_relationships=result.get("_pre_filter_relationships", raw_r),
            dropped_nodes=raw_n - pre_filt_n,
            dropped_relationships=raw_r
            - result.get("_pre_filter_relationships", raw_r),
            post_filter_nodes=post_filt_n,
            post_filter_relationships=post_filt_r,
            proteins_resolved=result.get("_proteins_resolved", 0),
            proteins_unresolved=result.get("_proteins_unresolved", 0),
            extraction_time_s=result.get("_extraction_time_s", 0.0),
        )
        self.run_report.record_chunk(obs)

        return {
            "nodes": result["nodes"],
            "relationships": result["relationships"],
            "doc_id": item["doc_id"],
            "chunk_id": str(item["chunk_id"]),
            "error": None,
        }

    async def process_batch(
        self, batch_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process a batch of items with enhanced APOC error handling and dynamic batch sizing for memory resilience."""
        if not batch_items:
            return []

        # Pre-cleanup to prevent APOC merge conflicts
        try:
            await asyncio.to_thread(self._cleanup_graphrag_state)
        except Exception as cleanup_error:
            logger.debug(f"Pre-batch cleanup warning: {cleanup_error}")

        # Process items concurrently
        results = await asyncio.gather(
            *[self._process_single_item(item) for item in batch_items]
        )

        # Check for memory errors and retry if needed
        oom_indices = []
        for i, result in enumerate(results):
            error = result.get("error", "")
            if error and (
                "MemoryPoolOutOfMemoryError" in str(error)
                or "TransientError" in str(error)
            ):
                oom_indices.append(i)

        if oom_indices:
            # If we have OOM errors and batch size > 1, we should retry with smaller batches
            if len(batch_items) > 1:
                logger.warning(
                    f"Detected {len(oom_indices)} OOM errors in batch of {len(batch_items)}. Retrying with reduced batch size."
                )

                # Remove failed items from failed_chunks list since we are retrying
                failed_chunk_ids = {batch_items[i]["chunk_id"] for i in oom_indices}
                self.failed_chunks = [
                    fc
                    for fc in self.failed_chunks
                    if fc["chunk_id"] not in failed_chunk_ids
                ]

                # Identify items to retry
                items_to_retry = [batch_items[i] for i in oom_indices]

                retry_results = []

                # Split logic
                if len(items_to_retry) == 1:
                    # Just retry this one item
                    logger.info(
                        f"Retrying single item {items_to_retry[0]['chunk_id']} sequentially"
                    )
                    retry_results = await self.process_batch(items_to_retry)
                else:
                    mid = len(items_to_retry) // 2
                    batch1 = items_to_retry[:mid]
                    batch2 = items_to_retry[mid:]

                    logger.info(
                        f"Splitting failed batch into {len(batch1)} and {len(batch2)} items"
                    )

                    results1 = await self.process_batch(batch1)
                    results2 = await self.process_batch(batch2)
                    retry_results = results1 + results2

                # Map chunk_id to result for retry results
                retry_map = {str(r["chunk_id"]): r for r in retry_results}

                final_results = []
                for i, item in enumerate(batch_items):
                    if i in oom_indices:
                        # This was a failed item, check if we have a retry result
                        chunk_id = str(item["chunk_id"])
                        if chunk_id in retry_map:
                            final_results.append(retry_map[chunk_id])
                        else:
                            # Should not happen if logic is correct
                            final_results.append(results[i])
                    else:
                        final_results.append(results[i])

                return final_results
            logger.error(
                f"Item {batch_items[0]['chunk_id']} failed with OOM even with batch size 1. Skipping."
            )

        await asyncio.sleep(2)
        return results

    async def process_documents(
        self,
        docs: list[ProcessedDocument],
        cleanup_existing: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> list[ProcessedDocument]:
        # Use provided job_id from API if available, otherwise generate one
        self.ingestion_job_id = (
            metadata.get("job_id")
            if metadata and metadata.get("job_id")
            else str(uuid.uuid4())
        )
        self.ingested_at = datetime.now(UTC).isoformat()
        logger.info(
            f"Processing {len(docs)} documents with KG pipeline (job_id={self.ingestion_job_id})"
        )
        start_ns = time.perf_counter_ns()
        self.failed_chunks = []
        self.run_report = IngestionRunReport(
            service=self.service, database=self.database
        )

        # Re-chunk documents using token-based chunker
        docs = self.chunker.chunk_documents(docs)
        logger.info(
            f"Token-chunked {len(docs)} documents (max_tokens={self.chunker.max_tokens}, overlap={self.chunker.overlap_tokens})"
        )

        total_chunks = sum(len(doc.chunks) for doc in docs)
        self.run_report.set_corpus_size(docs=len(docs), chunks=total_chunks)

        if cleanup_existing:
            logger.info("Cleaning database")
            self.ontology_filter.cleanup_database(self.db, conservative=True)

        doc_chunk_map: dict[int, list[int]] = {}
        all_items = []
        chunk_index = 0

        for doc_idx, doc in enumerate(docs):
            doc_chunk_map[doc_idx] = []
            doc_metadata = (
                doc.metadata or metadata or {"doc_id": doc.doc_id, "source": "pubmed"}
            )
            for i, chunk in enumerate(doc.chunks):
                all_items.append(
                    {
                        "text": chunk.text,
                        "doc_id": doc.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "index": i,
                        "metadata": doc_metadata,
                    }
                )
                doc_chunk_map[doc_idx].append(chunk_index)
                chunk_index += 1

        all_results = []
        for i in range(0, len(all_items), self.batch_size):
            batch = all_items[i : i + self.batch_size]
            logger.info(
                f"Processing batch {i // self.batch_size + 1}/{(len(all_items) + self.batch_size - 1) // self.batch_size} with {len(batch)} chunks"
            )

            # Add a brief delay between batches to reduce database conflicts
            if i > 0:
                await asyncio.sleep(1)

            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)

            # Check for incremental consolidation after each batch
            for chunk_data in batch:
                consolidation_stats = self.incremental_consolidator.on_chunk_processed(
                    chunk_data["chunk_id"]
                )
                if consolidation_stats:
                    logger.info(
                        f"Incremental consolidation completed: {consolidation_stats}"
                    )
                    self.run_report.record_consolidation(consolidation_stats)

        for doc_idx, chunk_indices in doc_chunk_map.items():
            for i, chunk_idx in enumerate(chunk_indices):
                if chunk_idx < len(all_results):
                    docs[doc_idx].chunks[i].nodes = all_results[chunk_idx]["nodes"]
                    docs[doc_idx].chunks[i].relationships = all_results[chunk_idx][
                        "relationships"
                    ]

        if self.failed_chunks:
            logger.warning(
                f"Failed to process {len(self.failed_chunks)} chunks: {self.failed_chunks}"
            )

        # Run final consolidation after all chunks processed
        logger.info("Running final entity consolidation...")
        consolidation_stats = self.incremental_consolidator.final_consolidation()
        logger.info(f"Consolidation results: {consolidation_stats}")
        self.run_report.record_consolidation(consolidation_stats)

        # Node labeling — scope to current job's nodes only (not full DB scan)
        if self.skip_node_labeling:
            logger.info("Skipping node labeling (--skip-node-labeling)")
        else:
            self.ontology_pipeline.resolve_and_label_nodes(
                ingestion_job_id=self.ingestion_job_id
            )

        # Node processing completed - embeddings can be generated later with --add-graph-embeddings
        if metadata and metadata.get("publication_year"):
            logger.info(
                "Node processing with publication year completed. Use --add-graph-embeddings to generate embeddings for enriched nodes."
            )

        elapsed = timedelta(microseconds=(time.perf_counter_ns() - start_ns) // 1000)

        # ---- Ingestion summary ----
        total_nodes = 0
        total_rels = 0
        node_types: dict[str, int] = {}
        rel_types: dict[str, int] = {}
        for doc in docs:
            for chunk in doc.chunks:
                for node in chunk.nodes:
                    total_nodes += 1
                    ntype = node.get("type", node.get("label", "Entity"))
                    node_types[ntype] = node_types.get(ntype, 0) + 1
                for rel in chunk.relationships:
                    total_rels += 1
                    rtype = rel.get("type", "UNKNOWN")
                    rel_types[rtype] = rel_types.get(rtype, 0) + 1

        logger.info(f"{'=' * 60}")
        logger.info(f"INGESTION SUMMARY (job {self.ingestion_job_id})")
        logger.info(f"  Documents: {len(docs)}, Chunks: {total_chunks}")
        logger.info(
            f"  Nodes extracted: {total_nodes} ({', '.join(f'{v} {k}' for k, v in sorted(node_types.items(), key=lambda x: -x[1]))})"
        )
        logger.info(
            f"  Relationships extracted: {total_rels} ({', '.join(f'{v} {k}' for k, v in sorted(rel_types.items(), key=lambda x: -x[1])[:5])})"
        )
        if self.failed_chunks:
            logger.info(f"  Failed chunks: {len(self.failed_chunks)}")
        logger.info(f"  Time: {elapsed} (HH:MM:SS)")
        logger.info(f"{'=' * 60}")

        logger.info(f"Processed in {elapsed} (HH:MM:SS)")

        # Finalize and emit the TRL-3 ingestion run report
        self.last_run_report = self.run_report.finalize()

        # Set TTL on Redis sub-graph sets now that the job is complete
        if self.ingestion_job_id:
            ingestion_subgraph_cache.set_ttl(self._redis, self.ingestion_job_id)

        return docs

    async def fetch_new_abstracts(
        self, search_term: str, max_results: int = 10, force: bool = False
    ) -> list[ProcessedDocument]:
        """
        Fetch abstracts from PubMed.

        Args:
            search_term: The search query for PubMed
            max_results: Maximum number of results to fetch
            force: If True, fetch abstracts even if they already exist in the database
        """
        try:
            if force:
                logger.info(
                    "Force mode enabled: fetching abstracts regardless of existing PMIDs"
                )
                existing_pmids: set = set()  # Empty set to fetch all abstracts
            else:
                existing_pmids = self.db.get_existing_pmids()
                logger.info(f"Found {len(existing_pmids)} existing PMIDs in database")

            # Search PubMed for articles
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()

            # Filter PMIDs based on force flag
            if force:
                pmids = record["IdList"][:max_results]
                logger.info(f"Force mode: processing all {len(pmids)} PMIDs found")
            else:
                pmids = [
                    pmid for pmid in record["IdList"] if pmid not in existing_pmids
                ][:max_results]
                logger.info(
                    f"Normal mode: processing {len(pmids)} new PMIDs (filtered from {len(record['IdList'])} total)"
                )

            if not pmids:
                if force:
                    logger.info("No abstracts found for the search term.")
                else:
                    logger.info(
                        "No new abstracts found (all PMIDs already exist in database)."
                    )
                return []

            # Fetch abstract details
            logger.info(f"Fetching details for {len(pmids)} PMIDs...")
            handle = Entrez.efetch(
                db="pubmed", id=pmids, rettype="abstract", retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            logger.info(f"Entrez records keys: {records.keys()}")

            article_count = len(records.get("PubmedArticle", []))
            book_count = len(records.get("PubmedBookArticle", []))

            logger.info(
                f"Fetched {article_count} articles and {book_count} book articles from PubMed"
            )

            docs = []
            processed_count = 0
            skipped_count = 0

            # Process book articles
            for book in records.get("PubmedBookArticle", []):
                try:
                    if "BookDocument" not in book:
                        continue
                    book_doc = book["BookDocument"]
                    pmid = str(book_doc["PMID"])

                    # Extract abstract
                    abstract_text = ""
                    if (
                        "Abstract" in book_doc
                        and "AbstractText" in book_doc["Abstract"]
                    ):
                        abstract_parts = book_doc["Abstract"]["AbstractText"]
                        abstract_text = " ".join([str(part) for part in abstract_parts])

                    if not abstract_text:
                        continue

                    title = str(book_doc.get("ArticleTitle", ""))

                    metadata = {
                        "pmid": pmid,
                        "title": title,
                        "source": "pubmed",
                        "type": "book",
                    }

                    # Extract authors
                    try:
                        authors = []
                        if "AuthorList" in book_doc:
                            for author in book_doc["AuthorList"]:
                                if "LastName" in author and "ForeName" in author:
                                    authors.append(
                                        f"{author['ForeName']} {author['LastName']}"
                                    )
                                elif "LastName" in author:
                                    authors.append(author["LastName"])
                        metadata["authors"] = authors[:10]
                    except Exception:
                        metadata["authors"] = []

                    # Extract publication date
                    try:
                        pub_date = book_doc.get("Book", {}).get("PubDate", {})
                        if "Year" in pub_date:
                            metadata["publication_date"] = str(pub_date["Year"])
                    except Exception:
                        metadata["publication_date"] = ""

                    if not force and pmid in existing_pmids:
                        skipped_count += 1
                        continue

                    doc = ProcessedDocument(
                        doc_id=f"pubmed_{pmid}",
                        source=abstract_text,
                        metadata=metadata,
                        chunks=[
                            Chunk(
                                chunk_id=f"{pmid}_1",
                                text=abstract_text,
                                pmid=pmid,
                                title=title,
                            )
                        ],
                    )
                    docs.append(doc)

                    self.db.store_abstract_node(pmid, abstract_text, title, f"{pmid}_1")
                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing book article: {e}")
                    skipped_count += 1

            for article in records.get("PubmedArticle", []):
                try:
                    pmid = str(article["MedlineCitation"]["PMID"])

                    # Extract abstract text with fallback to OtherAbstract
                    abstract_text = ""
                    try:
                        if "Abstract" in article["MedlineCitation"]["Article"]:
                            abstract_parts = article["MedlineCitation"]["Article"][
                                "Abstract"
                            ]["AbstractText"]
                            abstract_text = " ".join(
                                [str(part) for part in abstract_parts]
                            )
                        elif (
                            "OtherAbstract" in article["MedlineCitation"]
                            and article["MedlineCitation"]["OtherAbstract"]
                        ):
                            other_abstract = article["MedlineCitation"][
                                "OtherAbstract"
                            ][0]
                            if "AbstractText" in other_abstract:
                                abstract_parts = other_abstract["AbstractText"]
                                abstract_text = " ".join(
                                    [str(part) for part in abstract_parts]
                                )
                    except Exception:
                        pass

                    if not abstract_text:
                        # If still no abstract, skip
                        raise KeyError("Abstract")

                    title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])

                    # Extract additional metadata
                    metadata = {"pmid": pmid, "title": title, "source": "pubmed"}

                    # Extract authors
                    try:
                        authors = []
                        author_list = article["MedlineCitation"]["Article"][
                            "AuthorList"
                        ]
                        for author in author_list:
                            if "LastName" in author and "ForeName" in author:
                                authors.append(
                                    f"{author['ForeName']} {author['LastName']}"
                                )
                            elif "LastName" in author:
                                authors.append(author["LastName"])
                        metadata["authors"] = authors[:10]  # Limit to first 10 authors
                    except (KeyError, TypeError):
                        metadata["authors"] = []

                    # Extract journal
                    try:
                        journal = article["MedlineCitation"]["Article"]["Journal"][
                            "Title"
                        ]
                        metadata["journal"] = str(journal)
                    except (KeyError, TypeError):
                        metadata["journal"] = ""

                    # Extract publication date
                    try:
                        pub_date = article["MedlineCitation"]["Article"]["Journal"][
                            "JournalIssue"
                        ]["PubDate"]
                        if "Year" in pub_date:
                            metadata["publication_date"] = str(pub_date["Year"])
                        elif "MedlineDate" in pub_date:
                            # Extract year from MedlineDate
                            import re

                            year_match = re.search(
                                r"\b(19|20)\d{2}\b", str(pub_date["MedlineDate"])
                            )
                            if year_match:
                                metadata["publication_date"] = year_match.group()
                    except (KeyError, TypeError):
                        metadata["publication_date"] = ""

                    # Extract DOI if available
                    try:
                        article_ids = article["PubmedData"]["ArticleIdList"]
                        for article_id in article_ids:
                            if article_id.attributes.get("IdType") == "doi":
                                metadata["doi"] = str(article_id)
                                break
                    except (KeyError, TypeError):
                        metadata["doi"] = ""

                    # Extract keywords if available
                    try:
                        keywords = []
                        keyword_list = article["MedlineCitation"]["KeywordList"]
                        for keyword_group in keyword_list:
                            for keyword in keyword_group:
                                keywords.append(str(keyword))
                        metadata["keywords"] = keywords[:20]  # Limit keywords
                    except (KeyError, TypeError):
                        metadata["keywords"] = []

                    # Check if we should process this abstract
                    if not force and pmid in existing_pmids:
                        logger.debug(f"Skipping existing PMID {pmid}")
                        skipped_count += 1
                        continue

                    doc = ProcessedDocument(
                        doc_id=f"pubmed_{pmid}",
                        source=abstract_text,
                        metadata=metadata,
                        chunks=[
                            Chunk(
                                chunk_id=f"{pmid}_1",
                                text=abstract_text,
                                pmid=pmid,
                                title=title,
                            )
                        ],
                    )
                    docs.append(doc)

                    # Store the abstract node
                    self.db.store_abstract_node(pmid, abstract_text, title, f"{pmid}_1")
                    processed_count += 1

                except (KeyError, TypeError) as e:
                    if skipped_count < 3:
                        logger.warning(f"Skipping article due to missing data: {e}")
                        # Log the structure of the first few failed articles to debug
                        try:
                            # MedlineCitation is usually a dict-like object, might need conversion
                            logger.warning(
                                f"Failed article structure keys: {article.keys()}"
                            )
                            if "MedlineCitation" in article:
                                logger.warning(
                                    f"MedlineCitation keys: {article['MedlineCitation'].keys()}"
                                )
                                if "Article" in article["MedlineCitation"]:
                                    logger.warning(
                                        f"Article keys: {article['MedlineCitation']['Article'].keys()}"
                                    )
                        except Exception as debug_e:
                            logger.warning(f"Failed to log debug info: {debug_e}")
                    else:
                        logger.warning(f"Skipping article due to missing data: {e}")
                    skipped_count += 1

            logger.info(
                f"Successfully processed {processed_count} abstracts, skipped {skipped_count}"
            )
            if force and processed_count > 0:
                logger.info(
                    f"Force mode: Re-processed {processed_count} abstracts (may include existing ones)"
                )

            return docs

        except Exception as e:
            logger.error(f"Failed to fetch abstracts: {e}", exc_info=True)
            return []

    async def load_stored_abstracts(
        self, limit: int | None = None
    ) -> list[ProcessedDocument]:
        """
        Load abstracts that have been pre-stored in the database (e.g., from mass scraper).

        Args:
            limit: Maximum number of abstracts to load (None for all)

        Returns:
            List of ProcessedDocument objects loaded from database
        """
        try:
            logger.info("Loading stored abstracts from database...")

            # Query to get stored abstracts
            abstracts = self.db.get_stored_abstracts(limit=limit)

            logger.info(f"Found {len(abstracts)} stored abstracts in database")

            docs = []
            for abstract in abstracts:
                try:
                    pmid = abstract["pmid"]
                    abstract_text = abstract["text"]
                    title = abstract["title"]

                    # Create basic metadata
                    metadata = {
                        "pmid": pmid,
                        "title": title,
                        "source": "pubmed",
                        "from_mass_scraper": True,
                    }

                    doc = ProcessedDocument(
                        doc_id=f"pubmed_{pmid}",
                        source=abstract_text,
                        metadata=metadata,
                        chunks=[
                            Chunk(
                                chunk_id=f"{pmid}_1",
                                text=abstract_text,
                                pmid=pmid,
                                title=title,
                            )
                        ],
                    )
                    docs.append(doc)

                except Exception as e:
                    logger.warning(
                        f"Error processing stored abstract {abstract.get('pmid', 'unknown')}: {e}"
                    )
                    continue

            logger.info(f"Successfully loaded {len(docs)} stored abstracts")
            return docs

        except Exception as e:
            logger.error(f"Failed to load stored abstracts: {e}", exc_info=True)
            return []

    async def run(
        self, search_term: str, max_results: int = 10, force: bool = False
    ) -> list[ProcessedDocument]:
        """
        Run the complete KG pipeline: fetch abstracts and process them.

        Args:
            search_term: The search query for PubMed
            max_results: Maximum number of results to fetch
            force: If True, fetch and process abstracts even if they already exist
        """
        docs = await self.fetch_new_abstracts(search_term, max_results, force=force)
        return await self.process_documents(docs) if docs else []

    def close(self):
        if self.db:
            self.db.close()

    def _extract_sentence_context(
        self, abstract_text: str, source_name: str, target_name: str
    ) -> str | None:
        """
        Extract the sentence containing both the source and target entities.
        This provides better context for relationship evidence.
        """
        if not abstract_text or not source_name or not target_name:
            return None

        # Split into sentences
        sentences = re.split(r"[.!?]+", abstract_text)

        # Find sentences containing both entities (case-insensitive)
        source_pattern = re.compile(re.escape(source_name), re.IGNORECASE)
        target_pattern = re.compile(re.escape(target_name), re.IGNORECASE)

        for sentence in sentences:
            if source_pattern.search(sentence) and target_pattern.search(sentence):
                return sentence.strip()

        return None

    def _calculate_relationship_confidence(
        self, relationship: dict[str, Any], abstract_text: str
    ) -> float:
        """
        Calculate a confidence score for the relationship based on various factors.

        Factors considered:
        - Distance between entities in text
        - Presence of relationship-indicating keywords
        - Length and quality of context
        """
        try:
            source_name = relationship.get("source_name", "")
            target_name = relationship.get("target_name", "")

            if not source_name or not target_name or not abstract_text:
                return 0.5  # Default moderate confidence

            confidence_score = 0.5  # Base score

            # Check if entities appear in the same sentence
            sentence_context = self._extract_sentence_context(
                abstract_text, source_name, target_name
            )
            if sentence_context:
                confidence_score += 0.2

                # Check for relationship keywords
                relationship_keywords = [
                    "associated",
                    "related",
                    "linked",
                    "connected",
                    "correlated",
                    "involved",
                    "implicated",
                    "contributes",
                    "causes",
                    "leads",
                    "affects",
                    "influences",
                    "regulates",
                    "controls",
                    "modulates",
                ]

                sentence_lower = sentence_context.lower()
                keyword_count = sum(
                    1 for keyword in relationship_keywords if keyword in sentence_lower
                )
                confidence_score += min(keyword_count * 0.1, 0.3)  # Max 0.3 bonus

            # Penalize if entities are very far apart
            source_pos = abstract_text.lower().find(source_name.lower())
            target_pos = abstract_text.lower().find(target_name.lower())
            if source_pos != -1 and target_pos != -1:
                distance = abs(source_pos - target_pos)
                if distance > 500:  # Very far apart
                    confidence_score -= 0.1
                elif distance > 200:  # Moderately far
                    confidence_score -= 0.05

            return max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]

        except Exception as e:
            logger.warning(f"Failed to calculate relationship confidence: {e}")
            return 0.5

    async def process_documents_enhanced(
        self,
        docs: list[ProcessedDocument],
        cleanup_existing: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> list[ProcessedDocument]:
        """
        Enhanced document processing that stores relationships with full context and confidence scores.
        """
        # First run the standard processing
        processed_docs = await self.process_documents(docs, cleanup_existing, metadata)

        # Collect confidence scores per chunk for the run report
        chunk_confidence_map: dict[str, list[float]] = {}

        # Then enhance with detailed relationship tracking
        for doc in processed_docs:
            for chunk in doc.chunks:
                if not chunk.relationships:
                    continue

                abstract_text = chunk.text

                for rel in chunk.relationships:
                    # Retry logic for OOM errors
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            # Calculate confidence score
                            confidence = self._calculate_relationship_confidence(
                                rel, abstract_text
                            )

                            # Track confidence for the run report
                            chunk_confidence_map.setdefault(chunk.chunk_id, []).append(
                                confidence
                            )

                            # Extract sentence context
                            sentence_context = self._extract_sentence_context(
                                abstract_text,
                                rel.get("source_name", ""),
                                rel.get("target_name", ""),
                            )

                            # Store enhanced relationship information
                            self.relationship_counter.store_relationship_with_context(
                                source_id=rel["source_id"],
                                target_id=rel["target_id"],
                                rel_type=rel.get("type", "ASSOCIATED_WITH"),
                                source_doc=doc.doc_id,
                                abstract_text=abstract_text,
                                sentence_context=sentence_context,
                                confidence_score=confidence,
                                extraction_method="llm_enhanced",
                                chunk_id=chunk.chunk_id,
                                metadata={
                                    "source_name": rel.get("source_name"),
                                    "target_name": rel.get("target_name"),
                                    "pmid": metadata.get("pmid") if metadata else None,
                                },
                            )

                            logger.debug(
                                f"Enhanced relationship stored: {rel['source_id']} -> {rel['target_id']} "
                                f"(confidence: {confidence:.2f})"
                            )
                            break  # Success, exit retry loop

                        except Exception as e:
                            is_oom = "MemoryPoolOutOfMemoryError" in str(
                                e
                            ) or "TransientError" in str(e)
                            if is_oom and attempt < max_retries - 1:
                                wait_time = 5 * (attempt + 1)
                                logger.warning(
                                    f"OOM error enhancing relationship {rel.get('source_id')} -> {rel.get('target_id')}. Retrying in {wait_time}s..."
                                )
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(
                                    f"Failed to enhance relationship {rel.get('source_id')} -> {rel.get('target_id')}: {e}"
                                )
                                break

        # Backfill confidence scores into the run report observations
        for scores in chunk_confidence_map.values():
            self.run_report.record_confidence_scores(scores)

        # Re-finalize the report with confidence data included
        self.last_run_report = self.run_report.finalize()

        return processed_docs

    def get_relationship_quality_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive quality report for extracted relationships.
        """
        try:
            report: dict[str, Any] = {
                "summary": {},
                "high_confidence_relationships": [],
                "pattern_analysis": {},
                "quality_metrics": {},
            }

            # Get high-confidence relationships
            high_conf_rels = (
                self.relationship_counter.get_high_confidence_relationships(
                    min_occurrences=2, min_avg_confidence=0.7
                )
            )
            report["high_confidence_relationships"] = high_conf_rels[:20]  # Top 20

            # Get pattern analysis
            patterns = self.relationship_counter.analyze_relationship_patterns()
            report["pattern_analysis"] = patterns

            # Calculate quality metrics
            total_rels = len(high_conf_rels) if high_conf_rels else 0
            if total_rels > 0:
                avg_confidence = (
                    sum(rel["avg_confidence"] for rel in high_conf_rels) / total_rels
                )
                avg_occurrences = (
                    sum(rel["occurrence_count"] for rel in high_conf_rels) / total_rels
                )

                report["quality_metrics"] = {
                    "total_high_confidence_relationships": total_rels,
                    "average_confidence_score": round(avg_confidence, 3),
                    "average_occurrences_per_relationship": round(avg_occurrences, 1),
                }

            # Summary statistics - updated for edge-based evidence storage
            if self.db:
                summary_stats = self.db._execute_cypher("""
                    MATCH (s)-[r]->(t)
                    RETURN
                        count(DISTINCT r) AS total_relationships,
                        sum(size(coalesce(r.evidence, []))) AS total_occurrences,
                        count(DISTINCT CASE WHEN r.source_docs IS NOT NULL THEN r END) AS relationships_with_sources,
                        avg(r.avg_confidence) AS overall_avg_confidence
                """)
            else:
                summary_stats = [
                    {
                        "total_relationships": 0,
                        "total_occurrences": 0,
                        "relationships_with_sources": 0,
                        "overall_avg_confidence": 0.0,
                    }
                ]

            if summary_stats:
                report["summary"] = summary_stats[0]

            logger.info("Generated relationship quality report")
            return report

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}", exc_info=True)
            return {"error": str(e)}

    # Enhanced Node Management Methods
    def update_enriched_embeddings(self, node_ids: list[str] | None = None):
        """
        Simple method to update node embeddings with enriched properties.
        Uses the existing EmbeddingPipeline pattern for consistency.

        Args:
            node_ids: Specific node IDs to update. If None, updates all nodes that need embeddings.
        """

        from pipeline.ingest.embedding_pipeline_neo4j import Neo4jEmbeddingPipeline

        embedding_pipeline = Neo4jEmbeddingPipeline(
            service=self.service,
            database=self.database,
        )
        try:
            if node_ids:
                # Update specific nodes
                for node_id in node_ids:
                    query = """
                    MATCH (n) WHERE n.id = $node_id AND n.type IS NOT NULL
                    RETURN n.id AS id, n.name AS name, n.type AS type,
                           properties(n) AS all_properties
                    """
                    if self.db:
                        result = self.db._execute_cypher(query, {"node_id": node_id})
                        if result:
                            # Generate embedding for enriched node context (stored in vector index, not as node feature)
                            node_data = result[0]
                            all_props = node_data.get("all_properties", {})

                            # Start with base node info
                            enriched_text = f"{all_props.get('type', 'Node')}: {all_props.get('name', 'Unknown')}"

                            # Dynamically add ALL properties except core ones (id, name, type)
                            excluded_props = {
                                "id",
                                "name",
                                "type",
                                "embedding",
                                "enriched_text",
                            }

                            for prop_key, prop_value in all_props.items():
                                if (
                                    prop_key not in excluded_props
                                    and prop_value is not None
                                    and str(prop_value).strip() != ""
                                ):
                                    # Format property name nicely
                                    formatted_key = prop_key.replace("_", " ").title()
                                    enriched_text += f" ({formatted_key}: {prop_value})"

                            embedding = embedding_pipeline.embedder.embed_query(
                                enriched_text
                            )
                            update_query = """
                            MATCH (n) WHERE n.id = $node_id
                            SET n.embedding = $embedding, n.enriched_text = $enriched_text
                            """
                            self.db._execute_cypher(
                                update_query,
                                {
                                    "node_id": all_props.get("id"),
                                    "embedding": embedding,
                                    "enriched_text": enriched_text,
                                },
                            )
                return {"updated": len(node_ids) if node_ids else 0}
            # Update all nodes that need embeddings - dynamically detect any property changes
            query = """
                MATCH (n)
                WHERE n.type IS NOT NULL AND (
                    n.embedding IS NULL OR
                    n.enriched_text IS NULL OR
                    size([key IN keys(n) WHERE key NOT IN ['id', 'name', 'type', 'embedding', 'enriched_text']
                          AND n[key] IS NOT NULL
                          AND NOT n.enriched_text CONTAINS toString(n[key])]) > 0
                )
                RETURN n.id AS id, properties(n) AS all_properties
                LIMIT 1000
                """
            results = self.db._execute_cypher(query) if self.db else []

            updated_count = 0
            for node_data in results:
                all_props = node_data.get("all_properties", {})

                # Start with base node info
                enriched_text = f"{all_props.get('type', 'Node')}: {all_props.get('name', 'Unknown')}"

                # Dynamically add ALL properties except core ones
                excluded_props = {"id", "name", "type", "embedding", "enriched_text"}

                for prop_key, prop_value in all_props.items():
                    if (
                        prop_key not in excluded_props
                        and prop_value is not None
                        and str(prop_value).strip() != ""
                    ):
                        # Format property name nicely
                        formatted_key = prop_key.replace("_", " ").title()
                        enriched_text += f" ({formatted_key}: {prop_value})"

                embedding = embedding_pipeline.embedder.embed_query(enriched_text)

                # Only store embedding as property if explicitly requested (for efficiency)
                if getattr(embedding_pipeline, "store_embedding_properties", False):
                    update_query = """
                        MATCH (n) WHERE n.id = $node_id
                        SET n.embedding = $embedding, n.enriched_text = $enriched_text
                        """
                    params = {
                        "node_id": all_props.get("id"),
                        "embedding": embedding,
                        "enriched_text": enriched_text,
                    }
                else:
                    # Only store text properties, embedding stored separately in vector index
                    update_query = """
                        MATCH (n) WHERE n.id = $node_id
                        SET n.enriched_text = $enriched_text
                        """
                    params = {
                        "node_id": all_props.get("id"),
                        "enriched_text": enriched_text,
                    }

                if self.db:
                    self.db._execute_cypher(update_query, params)
                updated_count += 1

            logger.info(
                f"Updated embeddings for {updated_count} nodes with enriched properties"
            )
            return {"updated": updated_count}
        finally:
            embedding_pipeline.close()
