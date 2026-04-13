"""Query performance metrics: relevance, retrieval recall, latency, mode routing, cache impact.

Evaluates query system performance from structured result records. Each record
represents one query execution with its answer, sources, timing, and mode info.

Metrics
-------
- **Answer relevance** (LLM-as-judge, 1–5 scale): An LLM rates how well the
  answer addresses the question given the retrieved context.
- **Retrieval recall**: Fraction of known relevant sources actually retrieved.
- **Multi-hop success rate**: Fraction of multi-hop queries that succeeded.
- **Mode routing accuracy**: Fraction of queries where the predicted mode
  matched the expected mode.
- **Latency percentiles**: P50, P95, P99 of query latency.
- **Cache hit rate impact**: Latency comparison between cache hits and misses.

Usage
-----
    from pipeline.benchmarks.metrics.query_metrics import evaluate_query_performance

    results = evaluate_query_performance(records)
    print(json.dumps(results, indent=2))

CLI::

    python -m benchmarks.metrics.query_metrics --results results.json

Input JSON format (list of records)::

    [
      {
        "question": "What proteins ...",
        "answer": "Several proteins ...",
        "expected_sources": ["PMID:123", "PMID:456"],
        "retrieved_sources": ["PMID:123", "PMID:789"],
        "expected_mode": "local",
        "predicted_mode": "local",
        "latency_ms": 340.5,
        "success": true,
        "cache_hit": false,
        "hop_depth": 2,
        "context": "Retrieved context text ..."
      }
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# LLM judge protocol
# ---------------------------------------------------------------------------


class LLMJudge(Protocol):
    """Protocol for an LLM that scores answer relevance."""

    def score_relevance(self, question: str, answer: str, context: str) -> float:
        """Return a relevance score from 1.0 to 5.0."""
        ...


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    """A single query execution result for benchmarking."""

    question: str
    answer: str = ""
    expected_sources: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    expected_mode: str = ""
    predicted_mode: str = ""
    latency_ms: float = 0.0
    success: bool = True
    cache_hit: bool = False
    hop_depth: int = 0
    context: str = ""


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def evaluate_answer_relevance(
    records: list[QueryRecord],
    judge: LLMJudge,
) -> dict[str, Any]:
    """Score answer relevance via LLM-as-judge (1–5 scale).

    Parameters
    ----------
    records:
        Query records with ``question``, ``answer``, and ``context``.
    judge:
        An object implementing ``score_relevance(question, answer, context) -> float``.

    Returns
    -------
    dict with ``avg_score``, ``min_score``, ``max_score``, ``scores``.
    """
    scores: list[float] = []
    for r in records:
        if r.answer:
            s = judge.score_relevance(r.question, r.answer, r.context)
            scores.append(max(1.0, min(5.0, s)))

    if not scores:
        return {"avg_score": 0.0, "min_score": 0.0, "max_score": 0.0, "count": 0}

    return {
        "avg_score": round(sum(scores) / len(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "count": len(scores),
    }


def evaluate_retrieval_recall(records: list[QueryRecord]) -> dict[str, Any]:
    """Compute retrieval recall against known relevant sources.

    Returns
    -------
    dict with ``avg_recall``, per-query recalls, and counts.
    """
    recalls: list[float] = []
    total_expected = 0
    total_retrieved = 0

    for r in records:
        if not r.expected_sources:
            continue
        expected = set(r.expected_sources)
        retrieved = set(r.retrieved_sources)
        hits = len(expected & retrieved)
        total_expected += len(expected)
        total_retrieved += hits
        recalls.append(hits / len(expected))

    if not recalls:
        return {
            "avg_recall": 0.0,
            "count": 0,
            "total_expected": 0,
            "total_retrieved": 0,
        }

    return {
        "avg_recall": round(sum(recalls) / len(recalls), 4),
        "count": len(recalls),
        "total_expected": total_expected,
        "total_retrieved": total_retrieved,
        "macro_recall": round(sum(recalls) / len(recalls), 4),
        "micro_recall": round(total_retrieved / total_expected, 4)
        if total_expected
        else 0.0,
    }


def evaluate_multi_hop_success(records: list[QueryRecord]) -> dict[str, Any]:
    """Measure multi-hop query success rate (hop_depth > 1).

    Returns
    -------
    dict with ``success_rate``, counts.
    """
    multi_hop = [r for r in records if r.hop_depth > 1]
    if not multi_hop:
        return {"total": 0, "success_rate": 0.0}

    successful = sum(1 for r in multi_hop if r.success)
    return {
        "total": len(multi_hop),
        "successful": successful,
        "failed": len(multi_hop) - successful,
        "success_rate": round(successful / len(multi_hop), 4),
    }


def evaluate_mode_routing(records: list[QueryRecord]) -> dict[str, Any]:
    """Report mode routing accuracy (predicted vs expected mode).

    Returns
    -------
    dict with ``accuracy``, per-mode breakdown, confusion counts.
    """
    with_expected = [r for r in records if r.expected_mode and r.predicted_mode]
    if not with_expected:
        return {"total": 0, "accuracy": 0.0, "per_mode": {}}

    correct = sum(1 for r in with_expected if r.predicted_mode == r.expected_mode)

    # Per-mode breakdown
    per_mode: dict[str, dict[str, int]] = {}
    for r in with_expected:
        mode = r.expected_mode
        if mode not in per_mode:
            per_mode[mode] = {"total": 0, "correct": 0}
        per_mode[mode]["total"] += 1
        if r.predicted_mode == r.expected_mode:
            per_mode[mode]["correct"] += 1

    per_mode_acc = {
        m: {**c, "accuracy": round(c["correct"] / c["total"], 4)}
        for m, c in sorted(per_mode.items())
    }

    return {
        "total": len(with_expected),
        "correct": correct,
        "accuracy": round(correct / len(with_expected), 4),
        "per_mode": per_mode_acc,
    }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the p-th percentile from a sorted list (0 <= p <= 1)."""
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def evaluate_latency(records: list[QueryRecord]) -> dict[str, Any]:
    """Compute latency percentiles (P50, P95, P99).

    Returns
    -------
    dict with ``p50_ms``, ``p95_ms``, ``p99_ms``, ``avg_ms``, ``count``.
    """
    latencies = sorted(r.latency_ms for r in records if r.success and r.latency_ms > 0)
    if not latencies:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "avg_ms": 0.0}

    return {
        "count": len(latencies),
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "p99_ms": round(_percentile(latencies, 0.99), 2),
        "avg_ms": round(sum(latencies) / len(latencies), 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
    }


def evaluate_cache_impact(records: list[QueryRecord]) -> dict[str, Any]:
    """Measure cache hit rate and its impact on latency.

    Returns
    -------
    dict with ``hit_rate``, ``avg_hit_latency_ms``, ``avg_miss_latency_ms``,
    ``speedup_ratio``.
    """
    hits = [r for r in records if r.cache_hit and r.latency_ms > 0]
    misses = [r for r in records if not r.cache_hit and r.latency_ms > 0]

    total = len(hits) + len(misses)
    hit_rate = len(hits) / total if total else 0.0

    avg_hit = sum(r.latency_ms for r in hits) / len(hits) if hits else 0.0
    avg_miss = sum(r.latency_ms for r in misses) / len(misses) if misses else 0.0
    speedup = avg_miss / avg_hit if avg_hit > 0 else 0.0

    return {
        "total_queries": total,
        "cache_hits": len(hits),
        "cache_misses": len(misses),
        "hit_rate": round(hit_rate, 4),
        "avg_hit_latency_ms": round(avg_hit, 2),
        "avg_miss_latency_ms": round(avg_miss, 2),
        "speedup_ratio": round(speedup, 2),
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def evaluate_query_performance(
    records: list[QueryRecord],
    judge: LLMJudge | None = None,
) -> dict[str, Any]:
    """Run all query performance metrics and return a combined report.

    Parameters
    ----------
    records:
        List of query execution records.
    judge:
        Optional LLM judge for answer relevance scoring. If ``None``,
        the ``answer_relevance`` section is omitted.

    Returns
    -------
    dict with sections for each metric group.
    """
    report: dict[str, Any] = {"num_queries": len(records)}

    if judge is not None:
        report["answer_relevance"] = evaluate_answer_relevance(records, judge)

    report["retrieval_recall"] = evaluate_retrieval_recall(records)
    report["multi_hop_success"] = evaluate_multi_hop_success(records)
    report["mode_routing"] = evaluate_mode_routing(records)
    report["latency"] = evaluate_latency(records)
    report["cache_impact"] = evaluate_cache_impact(records)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_records(raw: list[dict[str, Any]]) -> list[QueryRecord]:
    """Parse raw JSON dicts into QueryRecord instances."""
    return [
        QueryRecord(
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            expected_sources=item.get("expected_sources", []),
            retrieved_sources=item.get("retrieved_sources", []),
            expected_mode=item.get("expected_mode", ""),
            predicted_mode=item.get("predicted_mode", ""),
            latency_ms=item.get("latency_ms", 0.0),
            success=item.get("success", True),
            cache_hit=item.get("cache_hit", False),
            hop_depth=item.get("hop_depth", 0),
            context=item.get("context", ""),
        )
        for item in raw
    ]


def main() -> None:
    """CLI entry point for evaluating query performance metrics.

    Usage:
        python -m benchmarks.metrics.query_metrics --results results.json
    """
    parser = argparse.ArgumentParser(description="Query performance metrics")
    parser.add_argument(
        "--results", required=True, help="Path to query results JSON file"
    )
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    with open(args.results) as f:
        raw = json.load(f)

    records = _parse_records(raw)
    report = evaluate_query_performance(records, judge=None)
    text = json.dumps(report, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
