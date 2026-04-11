"""Unit tests for pipeline.cli — argument parsing and config loading.

Validates Requirements 14.3, 14.4:
- CLI accepts command-line arguments for backend configurations
- CLI accepts --config flag for loading from YAML
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pipeline.cli import _load_config, main

# ---------------------------------------------------------------------------
# _load_config tests (Requirement 14.4)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Test YAML config loading via _load_config."""

    def test_returns_empty_dict_when_path_is_none(self) -> None:
        assert _load_config(None) == {}

    def test_loads_yaml_file(self, tmp_path: Path) -> None:
        config_data = {
            "graph_uri": "bolt://myhost:7687",
            "graph_user": "admin",
            "llm_model": "mistral",
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = _load_config(str(config_file))
        assert result == config_data

    def test_returns_empty_dict_for_empty_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = _load_config(str(config_file))
        assert result == {}

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# --help exits 0 (Requirement 14.3)
# ---------------------------------------------------------------------------


class TestHelpExitsZero:
    """Test that --help on subcommands exits with code 0."""

    def test_run_help_exits_zero(self) -> None:
        with patch.object(sys, "argv", ["biokg-ingest", "run", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_enrich_help_exits_zero(self) -> None:
        with patch.object(sys, "argv", ["biokg-ingest", "enrich", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_top_level_help_exits_zero(self) -> None:
        with patch.object(sys, "argv", ["biokg-ingest", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Argument / config merging (Requirement 14.3, 14.4)
# ---------------------------------------------------------------------------


class TestArgConfigMerging:
    """Test that CLI args override config file values in _build_backends."""

    def test_build_backends_uses_cli_args_over_config(self, tmp_path: Path) -> None:
        """CLI args take precedence over config file values."""
        import argparse

        from pipeline.cli import _build_backends

        config = {
            "graph_uri": "bolt://config-host:7687",
            "graph_user": "config-user",
            "graph_password": "config-pass",
            "graph_database": "config-db",
            "embedding_model": "config-model",
            "llm_model": "config-llm",
            "llm_url": "http://config-host:11434",
        }

        # Simulate CLI args that override some config values
        args = argparse.Namespace(
            graph_uri="bolt://cli-host:7687",
            graph_user="cli-user",
            graph_password="cli-pass",
            graph_database="cli-db",
            embedding_model=None,  # Not provided via CLI — should fall back to config
            llm_model=None,  # Not provided via CLI — should fall back to config
            llm_url=None,  # Not provided via CLI — should fall back to config
        )

        # Patch the backend constructors to capture what they receive
        with (
            patch("pipeline.backends.neo4j_store.Neo4jGraphStore") as MockGraph,
            patch(
                "pipeline.backends.local_embeddings.LocalEmbeddingProvider"
            ) as MockEmbed,
            patch("pipeline.backends.ollama_llm.OllamaLLMProvider") as MockLLM,
            patch("pipeline.backends.sqlite_job_store.SQLiteJobStore") as _MockJob,
        ):
            _build_backends(args, config)

            # CLI args should win for graph_uri, graph_user, graph_password, graph_database
            MockGraph.assert_called_once_with(
                uri="bolt://cli-host:7687",
                auth=("cli-user", "cli-pass"),
                database="cli-db",
            )

            # Config values should be used when CLI args are None/falsy
            MockEmbed.assert_called_once_with(model_name="config-model")
            MockLLM.assert_called_once_with(
                model="config-llm",
                base_url="http://config-host:11434",
            )

    def test_build_backends_uses_defaults_when_no_config(self) -> None:
        """When no config file and no CLI args, defaults are used."""
        import argparse

        from pipeline.cli import _build_backends

        args = argparse.Namespace(
            graph_uri=None,
            graph_user=None,
            graph_password=None,
            graph_database=None,
            embedding_model=None,
            llm_model=None,
            llm_url=None,
        )

        with (
            patch("pipeline.backends.neo4j_store.Neo4jGraphStore") as MockGraph,
            patch(
                "pipeline.backends.local_embeddings.LocalEmbeddingProvider"
            ) as MockEmbed,
            patch("pipeline.backends.ollama_llm.OllamaLLMProvider") as MockLLM,
            patch("pipeline.backends.sqlite_job_store.SQLiteJobStore"),
        ):
            _build_backends(args, {})

            MockGraph.assert_called_once_with(
                uri="bolt://localhost:7687",
                auth=("neo4j", "password"),
                database="neo4j",
            )
            MockEmbed.assert_called_once_with(model_name="all-mpnet-base-v2")
            MockLLM.assert_called_once_with(
                model="llama3",
                base_url="http://localhost:11434",
            )
