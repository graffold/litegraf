"""Throughput metrics: ingestion rates, embedding speed, pipeline time, memory, and cost.

Measures pipeline throughput from structured run records. Each record represents
one pipeline execution (or sub-phase) with document counts, timing, token usage,
and resource measurements.

Metrics
-------
- **PubMed ingestion throughput**: abstracts processed per minute.
- **PMC / bioRxiv throughput**: full-text articles processed per minute.
- **Embedding generation rate**: vectors generated per second.
- **End-to-end pipeline time**: total wall-clock seconds for N documents.
- **Memory peak (max RSS)**: maximum resident set size in MB during the run.
- **Token cost per 1,000 documents**: estimated USD per LLM config.

Usage
-----
    from pipeline.benchmarks.metrics.throughput_metrics import evaluate_throughput

    records = [
        ThroughputRecord(source="pubmed", doc_count=500, duration_sec=120.0, ...),
        ...
    ]
    report = evaluate_throughput(records)
    print(json.dumps(report, indent=2))

CLI::

    python -m benchmarks.metrics.throughput_metrics --results run.json

Input JSON format (list of records)::

    [
      {
        "source": "pubmed",
        "doc_count": 500,
        "duration_sec": 120.5,
        "embedding_count": 1500,
        "embedding_duration_sec": 30.2,
        "prompt_tokens": 250000,
        "completion_tokens": 80000,
        "llm_config": "bedrock",
        "peak_rss_mb": 1024.5,
        "pipeline_phase": "full"
      }
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any

# Default cost rates per 1K tokens (USD), matching CostTracker.DEFAULT_RATES
DEFAULT_RATES: dict[str, dict[str, float]] = {
    "ollama": {"input": 0.0, "output": 0.0},
    "openai": {"input": 0.01, "output": 0.03},
    "bedrock": {"input": 0.008, "output": 0.024},
    "sagemaker": {"input": 0.005, "output": 0.015},
}


@dataclass
class ThroughputRecord:
    """A single pipeline run or phase measurement."""

    source: str = ""  # pubmed, pmc, biorxiv, pdf
    doc_count: int = 0
    duration_sec: float = 0.0
    embedding_count: int = 0
    embedding_duration_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_config: str = ""  # ollama, bedrock, openai, sagemaker
    peak_rss_mb: float = 0.0
    pipeline_phase: str = ""  # full, ingestion, embedding, consolidation
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def evaluate_ingestion_throughput(
    records: list[ThroughputRecord],
) -> dict[str, Any]:
    """Compute ingestion throughput (docs/min) grouped by source.

    Returns per-source and overall stats.
    """
    by_source: dict[str, dict[str, float]] = {}

    for r in records:
        if r.doc_count <= 0 or r.duration_sec <= 0:
            continue
        src = r.source.lower() or "unknown"
        if src not in by_source:
            by_source[src] = {"total_docs": 0, "total_sec": 0.0}
        by_source[src]["total_docs"] += r.doc_count
        by_source[src]["total_sec"] += r.duration_sec

    result: dict[str, Any] = {}
    total_docs = 0
    total_sec = 0.0

    for src, agg in sorted(by_source.items()):
        docs_per_min = agg["total_docs"] / agg["total_sec"] * 60
        result[src] = {
            "total_docs": int(agg["total_docs"]),
            "total_sec": round(agg["total_sec"], 2),
            "docs_per_min": round(docs_per_min, 2),
        }
        total_docs += int(agg["total_docs"])
        total_sec += agg["total_sec"]

    result["overall"] = {
        "total_docs": total_docs,
        "total_sec": round(total_sec, 2),
        "docs_per_min": round(total_docs / total_sec * 60, 2) if total_sec > 0 else 0.0,
    }
    return result


def evaluate_embedding_rate(records: list[ThroughputRecord]) -> dict[str, Any]:
    """Compute embedding generation rate (vectors/sec).

    Aggregates across all records that have embedding data.
    """
    total_vectors = 0
    total_sec = 0.0

    for r in records:
        if r.embedding_count > 0 and r.embedding_duration_sec > 0:
            total_vectors += r.embedding_count
            total_sec += r.embedding_duration_sec

    if total_sec == 0:
        return {"total_vectors": 0, "total_sec": 0.0, "vectors_per_sec": 0.0}

    return {
        "total_vectors": total_vectors,
        "total_sec": round(total_sec, 2),
        "vectors_per_sec": round(total_vectors / total_sec, 2),
    }


def evaluate_pipeline_time(records: list[ThroughputRecord]) -> dict[str, Any]:
    """Capture end-to-end pipeline time for N documents.

    Groups by pipeline_phase and reports total wall-clock time.
    """
    by_phase: dict[str, dict[str, float]] = {}
    total_docs = 0
    total_sec = 0.0

    for r in records:
        if r.duration_sec <= 0:
            continue
        phase = r.pipeline_phase or "unspecified"
        if phase not in by_phase:
            by_phase[phase] = {"total_docs": 0, "total_sec": 0.0}
        by_phase[phase]["total_docs"] += r.doc_count
        by_phase[phase]["total_sec"] += r.duration_sec
        total_docs += r.doc_count
        total_sec += r.duration_sec

    phases = {
        p: {
            "total_docs": int(v["total_docs"]),
            "total_sec": round(v["total_sec"], 2),
        }
        for p, v in sorted(by_phase.items())
    }

    return {
        "total_docs": total_docs,
        "total_sec": round(total_sec, 2),
        "by_phase": phases,
    }


def evaluate_memory_peak(records: list[ThroughputRecord]) -> dict[str, Any]:
    """Report peak memory (max RSS in MB) across all records."""
    rss_values = [r.peak_rss_mb for r in records if r.peak_rss_mb > 0]
    if not rss_values:
        return {"max_rss_mb": 0.0, "avg_rss_mb": 0.0, "count": 0}

    return {
        "max_rss_mb": round(max(rss_values), 2),
        "avg_rss_mb": round(sum(rss_values) / len(rss_values), 2),
        "min_rss_mb": round(min(rss_values), 2),
        "count": len(rss_values),
    }


def evaluate_token_cost(
    records: list[ThroughputRecord],
    rates: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Compute token cost per 1,000 documents, grouped by LLM config.

    Parameters
    ----------
    records:
        Pipeline run records with token counts and llm_config.
    rates:
        Optional override for per-1K-token rates. Defaults to ``DEFAULT_RATES``.

    Returns
    -------
    dict with per-config cost breakdown and cost_per_1k_docs.
    """
    effective_rates = dict(DEFAULT_RATES)
    if rates:
        effective_rates.update(rates)

    by_config: dict[str, dict[str, float]] = {}

    for r in records:
        if r.prompt_tokens <= 0 and r.completion_tokens <= 0:
            continue
        cfg = r.llm_config.lower() or "unknown"
        if cfg not in by_config:
            by_config[cfg] = {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_docs": 0,
                "total_cost": 0.0,
            }
        entry = by_config[cfg]
        entry["total_prompt_tokens"] += r.prompt_tokens
        entry["total_completion_tokens"] += r.completion_tokens
        entry["total_docs"] += r.doc_count

        cfg_rates = effective_rates.get(cfg, {"input": 0.0, "output": 0.0})
        cost = (
            r.prompt_tokens * cfg_rates["input"]
            + r.completion_tokens * cfg_rates["output"]
        ) / 1000
        entry["total_cost"] += cost

    result: dict[str, Any] = {}
    for cfg, agg in sorted(by_config.items()):
        cost_per_1k = (
            agg["total_cost"] / agg["total_docs"] * 1000
            if agg["total_docs"] > 0
            else 0.0
        )
        result[cfg] = {
            "total_prompt_tokens": int(agg["total_prompt_tokens"]),
            "total_completion_tokens": int(agg["total_completion_tokens"]),
            "total_docs": int(agg["total_docs"]),
            "total_cost_usd": round(agg["total_cost"], 4),
            "cost_per_1k_docs_usd": round(cost_per_1k, 4),
        }

    return result


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def evaluate_throughput(
    records: list[ThroughputRecord],
    rates: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run all throughput metrics and return a combined report.

    Parameters
    ----------
    records:
        List of pipeline run/phase records.
    rates:
        Optional cost rate overrides.

    Returns
    -------
    dict with sections for each metric group.
    """
    return {
        "num_records": len(records),
        "ingestion_throughput": evaluate_ingestion_throughput(records),
        "embedding_rate": evaluate_embedding_rate(records),
        "pipeline_time": evaluate_pipeline_time(records),
        "memory_peak": evaluate_memory_peak(records),
        "token_cost": evaluate_token_cost(records, rates),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_records(raw: list[dict[str, Any]]) -> list[ThroughputRecord]:
    """Parse raw JSON dicts into ThroughputRecord instances."""
    return [
        ThroughputRecord(
            source=item.get("source", ""),
            doc_count=item.get("doc_count", 0),
            duration_sec=item.get("duration_sec", 0.0),
            embedding_count=item.get("embedding_count", 0),
            embedding_duration_sec=item.get("embedding_duration_sec", 0.0),
            prompt_tokens=item.get("prompt_tokens", 0),
            completion_tokens=item.get("completion_tokens", 0),
            llm_config=item.get("llm_config", ""),
            peak_rss_mb=item.get("peak_rss_mb", 0.0),
            pipeline_phase=item.get("pipeline_phase", ""),
        )
        for item in raw
    ]


def main() -> None:
    """CLI entry point for evaluating throughput metrics.

    Usage:
        python -m benchmarks.metrics.throughput_metrics --results run.json
    """
    parser = argparse.ArgumentParser(description="Pipeline throughput metrics")
    parser.add_argument(
        "--results", required=True, help="Path to pipeline run results JSON file"
    )
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    with open(args.results) as f:
        raw = json.load(f)

    records = _parse_records(raw)
    report = evaluate_throughput(records)
    text = json.dumps(report, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
