# Changelog

All notable changes to litegraf are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-04-11

- `LiteGraf` single-entry-point dataclass with all-defaults constructor
- Content-hash deduplication for idempotent inserts
- LLM response caching (disk-based, hash-keyed)
- Async concurrency limiter for LLM calls
- Sync/async dual API (`insert()` / `ainsert()`, `query()` / `aquery()`)
- Pluggable backend resolution via string shorthands, class references, or instances
- `only_context` query mode for custom prompt chains
- Package renamed from biokg-ingest to litegraf

## [0.0.12] — 2026-04-08

- Abstract interfaces fully decoupled from proprietary codebase
- Zero `src/` imports in pipeline package
- Standalone `pipeline/config.py` replacing proprietary config
- Standalone `pipeline/utils/` replacing proprietary utilities

## [0.0.11] — 2026-04-08

- Abstract interfaces: `GraphStore`, `EmbeddingProvider`, `LLMProvider`, `JobStore`
- Default backends: Neo4jGraphStore, LocalEmbeddingProvider, OllamaLLMProvider, SQLiteJobStore
- Dependency injection across all pipeline modules
- Internal adapters for proprietary backends
- CLI entrypoints (`run`, `enrich`)
- `ProcessorBase` ABC and `discover_processors()` plugin architecture
- Property-based test suite (8 properties via Hypothesis)
- `pyproject.toml` package configuration

## [0.0.10] — 2026-04-04

- Dynamic context graph for provenance tracking
- Extraction metrics module for pipeline quality monitoring
- KG quality metrics module

## [0.0.9] — 2026-04-01

- Tabular data ingestion (CSV/Excel → KG)
- `TabularIngestionService` with core chunking logic
- Table source validation in ingestion API
- Sub-graph deletion endpoint
- Ingestion preview with sub-KG graph visualization
- Metadata extractor service for uploaded files

## [0.0.8] — 2026-03-26

- Generic entity type system (domain-agnostic extraction)
- Domain-adaptive LLM extraction prompts
- Parameterized `kg_pipeline.py` and `gleaning_extractor.py` prompts
- Generic `get_entity_details`, `get_entity_hierarchy`, `get_entity_categories` tools
- End-to-end generic pipeline CLI
- Decoupled ontology layer from bio-specific assumptions
- BYOK (bring your own key) LLM key management
- CSV/subgraph export functionality
- Generalized `ontology_pipeline.py` for non-OBO hierarchies
- Integration test for non-bio text ingestion

## [0.0.7] — 2026-03-23

- PMC (PubMed Central) full-text ingestion with batch-streaming
- Expansive PMC ingestion with section filtering
- Graph statistics tool
- Entity disambiguation tool
- Shared diseases/proteins discovery tools
- Disease hierarchy tool

## [0.0.6] — 2026-03-10

- BioRxiv full-paper ingestion pipeline
- PDF content extraction for preprints
- Section parsing for structured papers
- BioRxiv deduplication against existing graph
- LLM response cache for CI evaluation
- Fine-tuned model evaluation integration for extraction

## [0.0.5] — 2026-02-22

- PMC full-text fetcher and API integration
- Multimodal chunking strategy (figures + tables)
- Full-text ingestion performance optimization
- UniProt database integration
- Assay validation data ingestion
- Panel composition metadata ingestion
- Sample size extraction from studies
- Entity coverage measurement
- Async pipeline entry points standardized
- Reactome pathway importer

## [0.0.4] — 2026-02-10

- Refactored Neo4j query agent with graph language tools
- KNN network expansion for entity discovery
- Protein-to-disease relationship improvements
- Agentic `search_knowledge_graph` and `dynamic_cypher_query` tools
- Flexible, robust graph tools with expanded feature returns
- Community detection integration

## [0.0.3] — 2026-01-08

- Neo4j agent with agentic query logic
- Dynamic query protocol elements
- KNN expansion fixes
- Redis session manager for follow-up queries
- Multimodal processing fixes
- Test refactoring

## [0.0.2] — 2025-12-02

- Neo4j pipeline refactor with base class enhancements
- Neptune + Neo4j dual backend support
- Graph refiner for relationship quality
- Neo4j KNN application
- Bedrock LLM integration working
- Database backfill utilities
- Link via all node type features

## [0.0.1] — 2025-10-02

- CSV enrichment pipeline (text-rich column KG extraction)
- Enrichment orchestrator with column analysis
- Node annotator for property and class enrichment
- Chunk → Abstract → Document node hierarchy
- Sentence-level embeddings
- UniProt protein matching from abstracts
- Disease hierarchy enrichment from MONDO ontology
- Community detection pipeline
- Neptune graph database support
- Relationship occurrence tracking with confidence scores
- Major enrichment pipeline refactor

## [0.0.0-rc.4] — 2025-09-08

- Dynamic query agent
- PPI (protein-protein interaction) capabilities
- Embedding atlas DB viewer
- Graphistry visualization integration
- Neptune agent support
- Massive PubMed ingestion pipeline
- Node type enrichment (disease + protein)

## [0.0.0-rc.3] — 2025-08-20

- Enriched embeddings (embeddings after node enrichment)
- Publication year as node feature
- Hybrid retrieval system with node features
- Binary quantized embeddings
- Graph analysis refactoring
- Graphistry visualization

## [0.0.0-rc.2] — 2025-07-08

- Stable KG pipeline with query support
- LLM factory (Ollama, SageMaker)
- Embedding factory (HuggingFace sentence-transformers)
- Docker deployment with official Neo4j image
- Entity resolution with MONDO disease ontology
- Ontology filter for node validation
- Full abstract text stored on nodes
- Enrichment node creation
- Graph density improvements

## [0.0.0-rc.1] — 2025-06-16

- KG query pipeline working end-to-end
- KG → embedding pipeline (sequential)
- LangChain Ollama integration
- Retroactive node embeddings
- Graph plot in application
- Enrichment script for adding nodes
- Entity resolution with ontology harmonization

## [0.0.0-alpha.2] — 2025-05-29

- MONDO disease ontology integration
- Ontology filter for entity validation
- LangGraph parallel embed + graph pipeline
- Refactored logging and embedding

## [0.0.0-alpha.1] — 2025-05-20

- First working prototype: Ollama + Neo4j
- PubMed abstract ingestion
- Basic entity extraction with local LLM
- Neo4j graph storage
- Embedding generation with sentence-transformers

## [0.0.0-alpha.0] — 2025-05-09

- Initial experiments: local RAG working
- First entity-link search prototype
