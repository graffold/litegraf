#!/usr/bin/env python3
"""
Ingestion Run Report — TRL-3 evidence collector for KG ingestion pipeline.

Aggregates per-run metrics using O(1) streaming accumulators.
No per-chunk data is retained after recording — memory footprint is constant
regardless of corpus size.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
@dataclass
class ChunkObservation:
    """Transient struct passed to record_chunk(). Not stored after aggregation."""

    chunk_id: str
    doc_id: str

    # Extraction stage
    raw_nodes: int = 0
    raw_relationships: int = 0
    extraction_time_s: float = 0.0
    extraction_error: str | None = None

    # Validation stage (post-schema check)
    validated_nodes: int = 0
    validated_relationships: int = 0
    dropped_nodes: int = 0
    dropped_relationships: int = 0

    # Ontology filter stage
    post_filter_nodes: int = 0
    post_filter_relationships: int = 0

    # Protein linking stage
    proteins_resolved: int = 0
    proteins_unresolved: int = 0

    # Confidence scores for relationships in this chunk
    confidence_scores: list[float] = field(default_factory=list)

    # Error category (None if success)
    error_category: str | None = None
    error_detail: str | None = None


class IngestionRunReport:
    """
    Streaming accumulator for ingestion run metrics.

    Memory usage is O(1) — each call to record_chunk() folds the observation
    into running totals and discards the raw data.  Confidence scores are
    bucketed into a fixed-size histogram on the fly.

    Usage:
        report = IngestionRunReport(service="bedrock", database="neo4j")
        report.record_chunk(obs)   # called per chunk — O(1) memory
        summary = report.finalize()
    """

    _ERROR_PATTERNS = [
        ("JSONDecodeError", "llm_json_parse"),
        ("JSON parsing failed", "llm_json_parse"),
        ("Extraction resulted in 0 items", "llm_empty_extraction"),
        ("APOC merge", "neo4j_apoc"),
        ("DeadlockDetected", "neo4j_deadlock"),
        ("MemoryPoolOutOfMemoryError", "neo4j_oom"),
        ("TransientError", "neo4j_transient"),
        ("DB storage failed", "neo4j_storage"),
        ("Entrez", "entrez_api"),
        ("HTTPError", "entrez_api"),
    ]

    _CONF_BUCKETS = ("0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0")

    def __init__(self, service: str = "", database: str = ""):
        self.service = service
        self.database = database
        self.run_start = time.time()

        self._total_docs = 0
        self._total_chunks = 0
        self._chunks_observed = 0

        # Extraction accumulators
        self._total_extraction_time = 0.0
        self._total_raw_nodes = 0
        self._total_raw_rels = 0
        self._total_val_nodes = 0
        self._total_val_rels = 0
        self._total_dropped_nodes = 0
        self._total_dropped_rels = 0
        self._empty_extraction_chunks = 0

        # Ontology filter accumulators
        self._total_post_filter_nodes = 0
        self._total_post_filter_rels = 0

        # Protein linking accumulators
        self._total_proteins_resolved = 0
        self._total_proteins_unresolved = 0

        # Confidence — streaming histogram + Welford online stats
        self._conf_count = 0
        self._conf_sum = 0.0
        self._conf_min = float("inf")
        self._conf_max = float("-inf")
        self._conf_histogram: dict[str, int] = dict.fromkeys(self._CONF_BUCKETS, 0)

        # Error accumulators
        self._failed_chunks = 0
        self._error_counts: Counter[str] = Counter()

        # Consolidation (merged from incremental + final)
        self._consolidation_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def set_corpus_size(self, docs: int, chunks: int) -> None:
        self._total_docs = docs
        self._total_chunks = chunks

    def record_chunk(self, obs: ChunkObservation) -> None:
        """Fold a chunk observation into running accumulators. O(1) memory."""
        self._chunks_observed += 1

        # Extraction
        self._total_extraction_time += obs.extraction_time_s
        self._total_raw_nodes += obs.raw_nodes
        self._total_raw_rels += obs.raw_relationships
        self._total_val_nodes += obs.validated_nodes
        self._total_val_rels += obs.validated_relationships
        self._total_dropped_nodes += obs.dropped_nodes
        self._total_dropped_rels += obs.dropped_relationships

        if (
            obs.raw_nodes == 0
            and obs.raw_relationships == 0
            and not obs.extraction_error
        ):
            self._empty_extraction_chunks += 1

        # Ontology filter
        self._total_post_filter_nodes += obs.post_filter_nodes
        self._total_post_filter_rels += obs.post_filter_relationships

        # Protein linking
        self._total_proteins_resolved += obs.proteins_resolved
        self._total_proteins_unresolved += obs.proteins_unresolved

        # Confidence — bucket each score immediately
        for score in obs.confidence_scores:
            self._record_confidence(score)

        # Errors
        if obs.error_category:
            self._failed_chunks += 1
            self._error_counts[obs.error_category] += 1

    def record_confidence_scores(self, scores: list[float]) -> None:
        """Record additional confidence scores (e.g. from enhanced processing)."""
        for score in scores:
            self._record_confidence(score)

    def record_consolidation(self, stats: dict[str, Any]) -> None:
        """Merge consolidation stats (incremental or final)."""
        for k, v in stats.items():
            if isinstance(v, int | float):
                self._consolidation_stats[k] = self._consolidation_stats.get(k, 0) + v
            else:
                self._consolidation_stats[k] = v

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------

    def classify_error(self, error_str: str) -> str:
        """Map an error string to a canonical category."""
        if not error_str:
            return "unknown"
        for pattern, category in self._ERROR_PATTERNS:
            if pattern in error_str:
                return category
        return "other"

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """Produce the aggregate report dict. Safe to call multiple times."""
        elapsed = time.time() - self.run_start
        n = self._chunks_observed

        if n == 0:
            return self._empty_report(elapsed)

        avg_extraction_time = self._total_extraction_time / n
        avg_nodes = self._total_val_nodes / n
        avg_rels = self._total_val_rels / n

        node_val_rate = (
            self._total_val_nodes / self._total_raw_nodes
            if self._total_raw_nodes
            else 0.0
        )
        rel_val_rate = (
            self._total_val_rels / self._total_raw_rels if self._total_raw_rels else 0.0
        )
        ont_node_rate = (
            self._total_post_filter_nodes / self._total_val_nodes
            if self._total_val_nodes
            else 0.0
        )
        ont_rel_rate = (
            self._total_post_filter_rels / self._total_val_rels
            if self._total_val_rels
            else 0.0
        )
        total_proteins = self._total_proteins_resolved + self._total_proteins_unresolved
        protein_rate = (
            self._total_proteins_resolved / total_proteins if total_proteins else 0.0
        )
        success_rate = (n - self._failed_chunks) / n

        report = {
            "run_metadata": {
                "service": self.service,
                "database": self.database,
                "wall_clock_s": round(elapsed, 2),
                "total_documents": self._total_docs,
                "total_chunks": self._total_chunks,
                "chunks_observed": n,
            },
            "throughput": {
                "total_extraction_time_s": round(self._total_extraction_time, 2),
                "avg_extraction_time_per_chunk_s": round(avg_extraction_time, 3),
                "chunks_per_minute": round((n / elapsed) * 60, 1) if elapsed > 0 else 0,
            },
            "extraction_yield": {
                "total_raw_nodes": self._total_raw_nodes,
                "total_raw_relationships": self._total_raw_rels,
                "total_validated_nodes": self._total_val_nodes,
                "total_validated_relationships": self._total_val_rels,
                "avg_nodes_per_chunk": round(avg_nodes, 2),
                "avg_relationships_per_chunk": round(avg_rels, 2),
                "node_validation_rate": round(node_val_rate, 4),
                "rel_validation_rate": round(rel_val_rate, 4),
                "dropped_nodes": self._total_dropped_nodes,
                "dropped_relationships": self._total_dropped_rels,
                "empty_extraction_chunks": self._empty_extraction_chunks,
            },
            "ontology_filter": {
                "post_filter_nodes": self._total_post_filter_nodes,
                "post_filter_relationships": self._total_post_filter_rels,
                "node_pass_rate": round(ont_node_rate, 4),
                "rel_pass_rate": round(ont_rel_rate, 4),
            },
            "protein_linking": {
                "total_proteins_seen": total_proteins,
                "resolved_to_canonical": self._total_proteins_resolved,
                "unresolved_new_nodes": self._total_proteins_unresolved,
                "link_rate": round(protein_rate, 4),
            },
            "confidence": self._confidence_summary(),
            "consolidation": self._consolidation_stats,
            "errors": {
                "total_failed_chunks": self._failed_chunks,
                "success_rate": round(success_rate, 4),
                "by_category": dict(self._error_counts.most_common()),
            },
        }

        self._log_report(report)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_confidence(self, score: float) -> None:
        """Bucket a single confidence score. O(1)."""
        self._conf_count += 1
        self._conf_sum += score
        self._conf_min = min(self._conf_min, score)
        self._conf_max = max(self._conf_max, score)

        if score < 0.3:
            self._conf_histogram["0.0-0.3"] += 1
        elif score < 0.5:
            self._conf_histogram["0.3-0.5"] += 1
        elif score < 0.7:
            self._conf_histogram["0.5-0.7"] += 1
        elif score < 0.9:
            self._conf_histogram["0.7-0.9"] += 1
        else:
            self._conf_histogram["0.9-1.0"] += 1

    def _confidence_summary(self) -> dict[str, Any]:
        if self._conf_count == 0:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "histogram": {}}
        return {
            "count": self._conf_count,
            "mean": round(self._conf_sum / self._conf_count, 4),
            "min": round(self._conf_min, 4),
            "max": round(self._conf_max, 4),
            "histogram": dict(self._conf_histogram),
        }

    def _empty_report(self, elapsed: float) -> dict[str, Any]:
        return {
            "run_metadata": {
                "service": self.service,
                "database": self.database,
                "wall_clock_s": round(elapsed, 2),
                "total_documents": self._total_docs,
                "total_chunks": self._total_chunks,
                "chunks_observed": 0,
            },
            "throughput": {},
            "extraction_yield": {},
            "ontology_filter": {},
            "protein_linking": {},
            "confidence": {},
            "consolidation": {},
            "errors": {
                "total_failed_chunks": 0,
                "success_rate": 1.0,
                "by_category": {},
            },
        }

    def _log_report(self, report: dict[str, Any]) -> None:
        logger.info("=" * 80)
        logger.info("📋 INGESTION RUN REPORT")
        logger.info("=" * 80)

        meta = report["run_metadata"]
        logger.info(
            f"Service: {meta['service']} | DB: {meta['database']} | "
            f"Docs: {meta['total_documents']} | Chunks: {meta['chunks_observed']} | "
            f"Wall clock: {meta['wall_clock_s']}s"
        )

        tp = report.get("throughput", {})
        if tp:
            logger.info(
                f"Throughput: {tp.get('chunks_per_minute', 0)} chunks/min | "
                f"Avg extraction: {tp.get('avg_extraction_time_per_chunk_s', 0)}s/chunk"
            )

        ey = report.get("extraction_yield", {})
        if ey:
            logger.info(
                f"Yield: {ey.get('total_validated_nodes', 0)} nodes, "
                f"{ey.get('total_validated_relationships', 0)} rels | "
                f"Avg/chunk: {ey.get('avg_nodes_per_chunk', 0)} nodes, "
                f"{ey.get('avg_relationships_per_chunk', 0)} rels | "
                f"Validation rate: nodes={ey.get('node_validation_rate', 0)}, "
                f"rels={ey.get('rel_validation_rate', 0)}"
            )
            if ey.get("empty_extraction_chunks", 0) > 0:
                logger.warning(
                    f"⚠️  {ey['empty_extraction_chunks']} chunks produced zero extractions"
                )

        of = report.get("ontology_filter", {})
        if of:
            logger.info(
                f"Ontology filter pass-through: nodes={of.get('node_pass_rate', 0)}, "
                f"rels={of.get('rel_pass_rate', 0)}"
            )

        pl = report.get("protein_linking", {})
        if pl and pl.get("total_proteins_seen", 0) > 0:
            logger.info(
                f"Protein linking: {pl['resolved_to_canonical']}/{pl['total_proteins_seen']} "
                f"resolved (rate={pl['link_rate']})"
            )

        conf = report.get("confidence", {})
        if conf and conf.get("count", 0) > 0:
            logger.info(
                f"Confidence: n={conf['count']} | mean={conf['mean']} | "
                f"min={conf['min']} | max={conf['max']}"
            )
            logger.info(f"  Histogram: {conf['histogram']}")

        errs = report.get("errors", {})
        logger.info(
            f"Errors: {errs.get('total_failed_chunks', 0)} failed | "
            f"Success rate: {errs.get('success_rate', 1.0)}"
        )
        if errs.get("by_category"):
            for cat, count in errs["by_category"].items():
                logger.info(f"  [{cat}]: {count}")

        logger.info("=" * 80)
