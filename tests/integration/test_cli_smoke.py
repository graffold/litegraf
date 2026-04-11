"""CLI entrypoint smoke tests.

Verifies that ``biokg-ingest run --help`` and ``biokg-ingest enrich --help``
exit with code 0, confirming the CLI is wired up correctly.

Validates: Requirements 15.3
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from pipeline.cli import main


@pytest.mark.integration
class TestCLISmokeHelp:
    """Smoke tests: --help on each subcommand exits 0."""

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

    def test_no_args_exits_nonzero(self) -> None:
        """Running with no subcommand should exit with an error."""
        with patch.object(sys, "argv", ["biokg-ingest"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
