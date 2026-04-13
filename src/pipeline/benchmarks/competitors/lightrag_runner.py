"""LightRAG competitor adapter for the Graffold benchmark suite.

Wraps the LightRAG library (``lightrag-hku``) so it can be driven by the same
datasets and metrics used for Graffold.  Skips gracefully when the package is
not installed.

Usage::

    from pipeline.benchmarks.competitors.lightrag_runner import is_available, run_extraction

    if is_available():
        results = run_extraction(examples, llm_func=my_llm, embed_func=my_embed)

CLI (quick smoke-test)::

    python -m benchmarks.competitors.lightrag_runner
"""

from __future__ import annotations

import asyncio
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

_LIGHTRAG_AVAILABLE: bool | None = None


def is_available() -> bool:
    """Return True if ``lightrag-hku`` is importable."""
    global _LIGHTRAG_AVAILABLE
    if _LIGHTRAG_AVAILABLE is None:
        try:
            import lightrag  # noqa: F401

            _LIGHTRAG_AVAILABLE = True
        except ImportError:
            _LIGHTRAG_AVAILABLE = False
    return _LIGHTRAG_AVAILABLE


def _require() -> None:
    if not is_available():
        msg = "LightRAG is not installed. Install with: pip install lightrag-hku"
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Entity extraction via LightRAG's KG pipeline
# ---------------------------------------------------------------------------


async def _insert_and_query(
    texts: list[str],
    questions: list[str],
    *,
    working_dir: str,
    llm_model_func: Callable[..., Any] | None = None,
    embedding_func: Any | None = None,
    query_mode: str = "hybrid",
) -> list[str]:
    """Insert *texts* into a LightRAG instance and run *questions*.

    Returns a list of answer strings (one per question).
    """
    _require()
    from lightrag import LightRAG, QueryParam

    kwargs: dict[str, Any] = {"working_dir": working_dir}
    if llm_model_func is not None:
        kwargs["llm_model_func"] = llm_model_func
    if embedding_func is not None:
        kwargs["embedding_func"] = embedding_func

    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    try:
        if texts:
            await rag.ainsert(texts)
        answers: list[str] = []
        for q in questions:
            ans = await rag.aquery(q, param=QueryParam(mode=query_mode))
            answers.append(str(ans))
        return answers
    finally:
        await rag.finalize_storages()


def insert_and_query(
    texts: list[str],
    questions: list[str],
    **kwargs: Any,
) -> list[str]:
    """Synchronous wrapper around :func:`_insert_and_query`."""
    return asyncio.run(_insert_and_query(texts, questions, **kwargs))


# ---------------------------------------------------------------------------
# Extraction benchmark adapter
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(
    r"\*\*([^*]+)\*\*\s*\(([^)]+)\)",  # **EntityName** (Type)
)


def _parse_entities_from_answer(answer: str, text: str) -> list[Entity]:
    """Best-effort entity extraction from a LightRAG answer string.

    LightRAG returns free-text answers, not structured NER output.  We look
    for bold-marked entity names and try to locate them in the source text.
    """
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
    llm_model_func: Callable[..., Any] | None = None,
    embedding_func: Any | None = None,
    query_mode: str = "local",
) -> dict[str, Any]:
    """Run LightRAG on *examples* and return extraction-style results.

    Because LightRAG is a RAG system (not a dedicated NER tool), we:
    1. Insert all example texts into a fresh LightRAG index.
    2. For each example, query "Extract all biomedical entities from: <text>".
    3. Parse the free-text answer into ``Entity`` objects.
    4. Return predicted ``BenchmarkExample`` list + timing info.
    """
    _require()

    tmp = tempfile.mkdtemp(prefix="lightrag_bench_")
    try:
        texts = [ex.text for ex in examples]
        questions = [
            f"List all biomedical entities (proteins, diseases, chemicals, genes) "
            f"mentioned in the following text. "
            f"Format each as **EntityName** (Type).\n\n{ex.text}"
            for ex in examples
        ]

        t0 = time.monotonic()
        answers = insert_and_query(
            texts,
            questions,
            working_dir=tmp,
            llm_model_func=llm_model_func,
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
            "competitor": "lightrag",
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
    llm_model_func: Callable[..., Any] | None = None,
    embedding_func: Any | None = None,
    query_mode: str = "hybrid",
) -> dict[str, Any]:
    """Index *texts* and answer *questions* via LightRAG.

    Returns answers and timing for comparison with Graffold query metrics.
    """
    _require()

    tmp = tempfile.mkdtemp(prefix="lightrag_bench_")
    try:
        t0 = time.monotonic()
        answers = insert_and_query(
            texts,
            questions,
            working_dir=tmp,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            query_mode=query_mode,
        )
        elapsed = time.monotonic() - t0

        return {
            "answers": answers,
            "elapsed_sec": round(elapsed, 3),
            "num_questions": len(questions),
            "competitor": "lightrag",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not is_available():
        print("LightRAG is not installed — skipping.")
        print("Install with: pip install lightrag-hku")
        return

    print("LightRAG is available ✓")
    print(json.dumps({"available": True, "competitor": "lightrag"}, indent=2))


if __name__ == "__main__":
    main()
