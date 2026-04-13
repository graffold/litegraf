"""nano-graphrag competitor adapter for the Graffold benchmark suite.

Wraps the nano-graphrag library so it can be driven by the same datasets and
metrics used for Graffold.  Skips gracefully when the package is not installed.

Usage::

    from pipeline.benchmarks.competitors.nano_graphrag_runner import is_available, run_extraction

    if is_available():
        results = run_extraction(examples)

CLI (quick smoke-test)::

    python -m benchmarks.competitors.nano_graphrag_runner
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from typing import Any

from pipeline.benchmarks.datasets.models import BenchmarkExample, Entity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_NANO_GRAPHRAG_AVAILABLE: bool | None = None


def is_available() -> bool:
    """Return True if ``nano-graphrag`` is importable."""
    global _NANO_GRAPHRAG_AVAILABLE
    if _NANO_GRAPHRAG_AVAILABLE is None:
        try:
            import nano_graphrag  # noqa: F401

            _NANO_GRAPHRAG_AVAILABLE = True
        except ImportError:
            _NANO_GRAPHRAG_AVAILABLE = False
    return _NANO_GRAPHRAG_AVAILABLE


def _require() -> None:
    if not is_available():
        msg = "nano-graphrag is not installed. Install with: pip install nano-graphrag"
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Core insert + query helper
# ---------------------------------------------------------------------------


def _insert_and_query(
    texts: list[str],
    questions: list[str],
    *,
    working_dir: str,
    best_model_func: Callable[..., Any] | None = None,
    cheap_model_func: Callable[..., Any] | None = None,
    embedding_func: Any | None = None,
    query_mode: str = "local",
) -> list[str]:
    """Insert *texts* into a nano-graphrag instance and run *questions*."""
    _require()
    from nano_graphrag import GraphRAG, QueryParam

    kwargs: dict[str, Any] = {"working_dir": working_dir}
    if best_model_func is not None:
        kwargs["best_model_func"] = best_model_func
    if cheap_model_func is not None:
        kwargs["cheap_model_func"] = cheap_model_func
    if embedding_func is not None:
        kwargs["embedding_func"] = embedding_func

    rag = GraphRAG(**kwargs)

    for text in texts:
        rag.insert(text)

    answers: list[str] = []
    for q in questions:
        ans = rag.query(q, param=QueryParam(mode=query_mode))
        answers.append(str(ans))
    return answers


# ---------------------------------------------------------------------------
# Extraction benchmark adapter
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(
    r"\*\*([^*]+)\*\*\s*\(([^)]+)\)",  # **EntityName** (Type)
)


def _parse_entities_from_answer(answer: str, text: str) -> list[Entity]:
    """Best-effort entity extraction from a nano-graphrag answer string."""
    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for match in _ENTITY_RE.finditer(answer):
        name, etype = match.group(1).strip(), match.group(2).strip()
        key = (name.lower(), etype.lower())
        if key in seen:
            continue
        seen.add(key)
        start = text.lower().find(name.lower())
        end = start + len(name) if start >= 0 else -1
        entities.append(Entity(text=name, entity_type=etype, start=start, end=end))
    return entities


def run_extraction(
    examples: list[BenchmarkExample],
    *,
    best_model_func: Callable[..., Any] | None = None,
    cheap_model_func: Callable[..., Any] | None = None,
    embedding_func: Any | None = None,
    query_mode: str = "local",
) -> dict[str, Any]:
    """Run nano-graphrag on *examples* and return extraction-style results.

    Because nano-graphrag is a RAG system (not a dedicated NER tool), we:
    1. Insert all example texts into a fresh nano-graphrag index.
    2. For each example, query for entities with a structured prompt.
    3. Parse the free-text answer into ``Entity`` objects.
    4. Return predicted ``BenchmarkExample`` list + timing info.
    """
    _require()

    tmp = tempfile.mkdtemp(prefix="nano_graphrag_bench_")
    try:
        texts = [ex.text for ex in examples]
        questions = [
            f"List all biomedical entities (proteins, diseases, chemicals, genes) "
            f"mentioned in the following text. "
            f"Format each as **EntityName** (Type).\n\n{ex.text}"
            for ex in examples
        ]

        t0 = time.monotonic()
        answers = _insert_and_query(
            texts,
            questions,
            working_dir=tmp,
            best_model_func=best_model_func,
            cheap_model_func=cheap_model_func,
            embedding_func=embedding_func,
            query_mode=query_mode,
        )
        elapsed = time.monotonic() - t0

        predictions: list[BenchmarkExample] = []
        for ex, ans in zip(examples, answers, strict=True):
            entities = _parse_entities_from_answer(ans, ex.text)
            predictions.append(
                BenchmarkExample(doc_id=ex.doc_id, text=ex.text, entities=entities)
            )

        return {
            "predictions": predictions,
            "elapsed_sec": round(elapsed, 3),
            "num_examples": len(examples),
            "competitor": "nano-graphrag",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Query benchmark adapter
# ---------------------------------------------------------------------------


def run_query(
    texts: list[str],
    questions: list[str],
    *,
    best_model_func: Callable[..., Any] | None = None,
    cheap_model_func: Callable[..., Any] | None = None,
    embedding_func: Any | None = None,
    query_mode: str = "local",
) -> dict[str, Any]:
    """Index *texts* and answer *questions* via nano-graphrag.

    Returns answers and timing for comparison with Graffold query metrics.
    """
    _require()

    tmp = tempfile.mkdtemp(prefix="nano_graphrag_bench_")
    try:
        t0 = time.monotonic()
        answers = _insert_and_query(
            texts,
            questions,
            working_dir=tmp,
            best_model_func=best_model_func,
            cheap_model_func=cheap_model_func,
            embedding_func=embedding_func,
            query_mode=query_mode,
        )
        elapsed = time.monotonic() - t0

        return {
            "answers": answers,
            "elapsed_sec": round(elapsed, 3),
            "num_questions": len(questions),
            "competitor": "nano-graphrag",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not is_available():
        print("nano-graphrag is not installed — skipping.")
        print("Install with: pip install nano-graphrag")
        return

    print("nano-graphrag is available ✓")
    print(json.dumps({"available": True, "competitor": "nano-graphrag"}, indent=2))


if __name__ == "__main__":
    main()
