"""CLI entrypoints for the biokg-ingest package.

Provides ``biokg-ingest run`` and ``biokg-ingest enrich`` subcommands that
construct default backends from CLI arguments merged with an optional YAML
config file, then execute the corresponding pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import yaml

from pipeline.interfaces import LLMProvider


def _load_config(config_path: str | None) -> dict[str, Any]:
    """Load backend configuration from a YAML file.

    Returns an empty dict when *config_path* is ``None`` or not provided.
    """
    if config_path:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_backends(args: argparse.Namespace, config: dict[str, Any]) -> tuple:
    """Construct backend instances from CLI args merged with config file values.

    CLI arguments take precedence over config file values, which take
    precedence over built-in defaults.
    """
    from pipeline.backends.local_embeddings import LocalEmbeddingProvider
    from pipeline.backends.sqlite_job_store import SQLiteJobStore
    from pipeline.dx.registry import BackendRegistry

    graph_store = BackendRegistry.resolve_graph_store(
        args.graph_backend or config.get("graph_backend", "neo4j"),
        uri=args.graph_uri or config.get("graph_uri", "bolt://localhost:7687"),
        auth=(
            args.graph_user or config.get("graph_user", "neo4j"),
            args.graph_password or config.get("graph_password", "password"),
        ),
        database=args.graph_database or config.get("graph_database", "neo4j"),
    )
    embedding_provider = LocalEmbeddingProvider(
        model_name=args.embedding_model
        or config.get("embedding_model", "all-mpnet-base-v2")
    )

    llm_provider_name = getattr(args, "llm_provider", None) or config.get("llm_provider", "ollama")
    llm_provider: LLMProvider
    if llm_provider_name == "cloudflare":
        from pipeline.backends.cloudflare_llm import CloudflareLLMProvider
        llm_provider = CloudflareLLMProvider(
            model=args.llm_model or config.get("llm_model", "@cf/meta/llama-3.1-8b-instruct"),
        )
    elif llm_provider_name == "bedrock":
        from pipeline.backends.bedrock_llm import BedrockLLMProvider
        llm_provider = BedrockLLMProvider(
            model=args.llm_model or config.get("llm_model", "eu.amazon.nova-lite-v1:0"),
        )
    else:
        from pipeline.backends.ollama_llm import OllamaLLMProvider
        llm_provider = OllamaLLMProvider(
            model=args.llm_model or config.get("llm_model", "llama3"),
            base_url=args.llm_url or config.get("llm_url", "http://localhost:11434"),
        )

    job_store = SQLiteJobStore()
    return graph_store, embedding_provider, llm_provider, job_store


def run_cmd(args: argparse.Namespace) -> None:
    """``biokg-ingest run`` — start the ingestion pipeline."""
    config = _load_config(args.config)
    graph_store, embedding_provider, llm_provider, _job_store = _build_backends(
        args, config
    )
    from pipeline.ingest.kg_pipeline import KGPipeline

    pipeline = KGPipeline(
        graph_store=graph_store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
    )
    asyncio.run(pipeline.run(search_term=args.query, max_results=args.max_results))


def enrich_cmd(args: argparse.Namespace) -> None:
    """``biokg-ingest enrich`` — start the enrichment pipeline."""
    config = _load_config(args.config)
    graph_store, _, llm_provider, _ = _build_backends(args, config)
    from pipeline.enrichment.enrichment_orchestrator import EnrichmentOrchestrator

    orchestrator = EnrichmentOrchestrator(
        graph_store=graph_store,
        llm_provider=llm_provider,
    )
    asyncio.run(orchestrator.enrich_csv(file_path=args.file))


def main() -> None:
    """Main CLI entrypoint (``biokg-ingest``)."""
    parser = argparse.ArgumentParser(prog="litegraf")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # --- run subcommand ---
    run_parser = subparsers.add_parser("run", help="Run ingestion pipeline")
    run_parser.add_argument("--query", required=True)
    run_parser.add_argument("--max-results", type=int, default=10)
    run_parser.add_argument("--config", help="Path to YAML/TOML config file")
    run_parser.add_argument("--graph-backend", default="neo4j", choices=["neo4j", "memgraph"])
    run_parser.add_argument("--graph-uri", help="Graph database URI")
    run_parser.add_argument("--graph-user", default="neo4j")
    run_parser.add_argument("--graph-password")
    run_parser.add_argument("--graph-database", default="neo4j")
    run_parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    run_parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "bedrock", "cloudflare"])
    run_parser.add_argument("--llm-model", default="llama3")
    run_parser.add_argument("--llm-url", default="http://localhost:11434")
    run_parser.set_defaults(func=run_cmd)

    # --- enrich subcommand ---
    enrich_parser = subparsers.add_parser("enrich", help="Run enrichment pipeline")
    enrich_parser.add_argument("--file", required=True)
    enrich_parser.add_argument("--config", help="Path to YAML/TOML config file")
    enrich_parser.add_argument("--graph-backend", default="neo4j", choices=["neo4j", "memgraph"])
    enrich_parser.add_argument("--graph-uri")
    enrich_parser.add_argument("--graph-user", default="neo4j")
    enrich_parser.add_argument("--graph-password")
    enrich_parser.add_argument("--graph-database", default="neo4j")
    enrich_parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "bedrock", "cloudflare"])
    enrich_parser.add_argument("--llm-model", default="llama3")
    enrich_parser.add_argument("--llm-url", default="http://localhost:11434")
    enrich_parser.set_defaults(func=enrich_cmd)

    args = parser.parse_args()

    if not args.command:
        # No subcommand — launch interactive TUI
        from pipeline.tui import _interactive
        _interactive()
        return

    args.func(args)
