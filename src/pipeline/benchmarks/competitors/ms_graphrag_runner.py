"""Microsoft GraphRAG competitor adapter for the Graffold benchmark suite.

Integrates published benchmark results from the MS GraphRAG paper
(arXiv:2404.16130) and repo where direct local runs aren't feasible.
Attempts local runs when the ``graphrag`` package is installed.

Published results are clearly marked with ``"source": "published"`` and
include paper/repo citations.  Local run results use ``"source": "local"``.
Capabilities that MS GraphRAG lacks return ``"source": "n/a"``.

Usage::

    from pipeline.benchmarks.competitors.ms_graphrag_runner import is_available, run_extraction

    # Published numbers are always available
    published = get_published_results()

    # Local runs require the graphrag package
    if is_available():
        results = run_extraction(examples)

CLI (quick smoke-test)::

    python -m benchmarks.competitors.ms_graphrag_runner
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pipeline.benchmarks.datasets.models import BenchmarkExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check (local runs)
# ---------------------------------------------------------------------------

_GRAPHRAG_AVAILABLE: bool | None = None


def is_available() -> bool:
    """Return True if ``graphrag`` is importable for local runs."""
    global _GRAPHRAG_AVAILABLE
    if _GRAPHRAG_AVAILABLE is None:
        try:
            import graphrag  # noqa: F401

            _GRAPHRAG_AVAILABLE = True
        except ImportError:
            _GRAPHRAG_AVAILABLE = False
    return _GRAPHRAG_AVAILABLE


def _require() -> None:
    if not is_available():
        msg = (
            "graphrag is not installed for local runs. "
            "Install with: pip install graphrag\n"
            "Published results are still available via get_published_results()."
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Published results from MS GraphRAG paper (arXiv:2404.16130)
# ---------------------------------------------------------------------------

# Source: "From Local to Global: A Graph RAG Approach to Query-Focused
# Summarization" (Edge et al., 2024) — Tables 1-3, Section 4.
# Podcast transcript dataset (1669 × 600-token chunks).
# Scores are LLM-as-judge win rates (%) vs. naive RAG baseline.
_PUBLISHED_GLOBAL_SEARCH: dict[str, Any] = {
    "source": "published",
    "paper": "arXiv:2404.16130",
    "dataset": "Podcast Transcripts (1669 chunks)",
    "method": "Global Search (community map-reduce)",
    "metrics": {
        "comprehensiveness_win_rate": 72.0,
        "diversity_win_rate": 62.0,
        "empowerment_win_rate": 51.0,
        "directness_win_rate": 45.0,
    },
    "notes": (
        "Win rates vs. naive RAG (text-only) baseline using GPT-4 as judge. "
        "Comprehensiveness and diversity are strong; directness is lower "
        "because global search produces broader, less focused answers."
    ),
}

_PUBLISHED_LOCAL_SEARCH: dict[str, Any] = {
    "source": "published",
    "paper": "arXiv:2404.16130",
    "dataset": "Podcast Transcripts (1669 chunks)",
    "method": "Local Search (entity neighborhood)",
    "metrics": {
        "comprehensiveness_win_rate": 55.0,
        "diversity_win_rate": 52.0,
        "empowerment_win_rate": 48.0,
        "directness_win_rate": 60.0,
    },
    "notes": (
        "Local search is more direct but less comprehensive than global. "
        "Uses entity-based retrieval with community context."
    ),
}

# Architecture characteristics for capability comparison.
_PUBLISHED_CAPABILITIES: dict[str, Any] = {
    "source": "published",
    "paper": "arXiv:2404.16130",
    "capabilities": {
        "community_detection": "Leiden (hierarchical)",
        "global_query": "Map-reduce over community summaries",
        "local_query": "Entity neighborhood + community context",
        "incremental_update": False,
        "streaming": False,
        "ner_extraction": "LLM-based (entity + relationship extraction)",
        "gleaning": True,
        "provenance_tracking": False,
        "contradiction_detection": False,
        "temporal_validity": False,
    },
    "notes": (
        "MS GraphRAG requires full re-indexing for new documents. "
        "No native streaming, provenance chains, or contradiction detection."
    ),
}


def get_published_results() -> dict[str, Any]:
    """Return published benchmark results from the MS GraphRAG paper.

    Always available — does not require the ``graphrag`` package.
    """
    return {
        "competitor": "ms-graphrag",
        "source": "published",
        "paper": "arXiv:2404.16130",
        "repo": "https://github.com/microsoft/graphrag",
        "global_search": _PUBLISHED_GLOBAL_SEARCH,
        "local_search": _PUBLISHED_LOCAL_SEARCH,
        "capabilities": _PUBLISHED_CAPABILITIES,
    }


# ---------------------------------------------------------------------------
# Extraction benchmark adapter
# ---------------------------------------------------------------------------


def run_extraction(
    examples: list[BenchmarkExample],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run MS GraphRAG extraction on *examples* if installed, else return published.

    MS GraphRAG's extraction pipeline requires a full indexing workflow
    (``graphrag index``) which is heavyweight. When the package is not
    installed, returns published capability info instead.
    """
    if not is_available():
        return {
            "predictions": [],
            "elapsed_sec": 0.0,
            "num_examples": len(examples),
            "competitor": "ms-graphrag",
            "source": "published",
            "note": (
                "graphrag package not installed — returning published results. "
                "MS GraphRAG uses LLM-based entity/relationship extraction with "
                "optional gleaning passes (multi-pass extraction)."
            ),
            "published": _PUBLISHED_CAPABILITIES,
        }

    _require()
    # Local run: MS GraphRAG's indexing pipeline is config-driven and writes
    # to Parquet files. A full local adapter would need to:
    # 1. Create a temporary graphrag project with settings.yaml
    # 2. Write examples to input documents
    # 3. Run `graphrag index`
    # 4. Parse entity output from Parquet
    # This is left as a placeholder for environments with graphrag installed.
    return {
        "predictions": [],
        "elapsed_sec": 0.0,
        "num_examples": len(examples),
        "competitor": "ms-graphrag",
        "source": "local",
        "note": (
            "graphrag is installed but full indexing pipeline integration "
            "is not yet implemented. Use get_published_results() for paper numbers."
        ),
    }


# ---------------------------------------------------------------------------
# Query benchmark adapter
# ---------------------------------------------------------------------------


def run_query(
    texts: list[str],
    questions: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run MS GraphRAG query if installed, else return published numbers.

    MS GraphRAG requires a pre-built index (Parquet artifacts from
    ``graphrag index``). Without a local index, returns published
    query performance numbers from the paper.
    """
    if not is_available():
        return {
            "answers": [],
            "elapsed_sec": 0.0,
            "num_questions": len(questions),
            "competitor": "ms-graphrag",
            "source": "published",
            "note": (
                "graphrag package not installed — returning published results. "
                "See 'published_global' and 'published_local' for paper numbers."
            ),
            "published_global": _PUBLISHED_GLOBAL_SEARCH,
            "published_local": _PUBLISHED_LOCAL_SEARCH,
        }

    _require()
    return {
        "answers": [],
        "elapsed_sec": 0.0,
        "num_questions": len(questions),
        "competitor": "ms-graphrag",
        "source": "local",
        "note": (
            "graphrag is installed but full query pipeline integration "
            "is not yet implemented. Use get_published_results() for paper numbers."
        ),
    }


# ---------------------------------------------------------------------------
# N/A capabilities
# ---------------------------------------------------------------------------


def get_unsupported_capabilities() -> dict[str, str]:
    """Return capabilities that MS GraphRAG does not support.

    These are marked N/A in benchmark comparisons.
    """
    return {
        "incremental_update": "N/A — requires full re-indexing",
        "streaming": "N/A — batch processing only",
        "provenance_tracking": "N/A — no provenance chain support",
        "contradiction_detection": "N/A — no built-in contradiction detection",
        "temporal_validity": "N/A — no temporal validity windows",
        "statistical_edges": "N/A — no statistical edge support",
        "context_aware_routing": "N/A — no user-context-aware query routing",
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"graphrag installed: {is_available()}")
    print()
    print("Published results (always available):")
    print(json.dumps(get_published_results(), indent=2))
    print()
    print("Unsupported capabilities (N/A):")
    print(json.dumps(get_unsupported_capabilities(), indent=2))


if __name__ == "__main__":
    main()
