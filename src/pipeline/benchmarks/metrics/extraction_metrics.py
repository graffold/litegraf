"""Extraction metrics: precision, recall, F1, and span accuracy for NER evaluation.

Compares predicted entities against gold-standard entities from benchmark datasets.
Supports both exact-match and partial-match (overlap) span evaluation, with
per-entity-type breakdowns.

Methodology
-----------
**Exact match**: A predicted entity is a true positive iff its (start, end, entity_type)
triple matches a gold entity exactly.

**Partial match**: A predicted entity is a true positive iff it overlaps with a gold
entity of the same type (i.e. the character spans intersect and entity_type matches).
Each gold entity can match at most one prediction (greedy 1-to-1 by overlap size).

**Metrics**:
- Precision = TP / (TP + FP)
- Recall    = TP / (TP + FN)
- F1        = 2 * P * R / (P + R)

All metrics are computed per-entity-type and micro-averaged across types.

Usage
-----
    from pipeline.benchmarks.datasets.loader import load_dataset
    from pipeline.benchmarks.metrics.extraction_metrics import evaluate_extraction

    gold = load_dataset("bc5cdr")
    predicted = [...]  # list of BenchmarkExample with predicted entities

    results = evaluate_extraction(gold.splits["test"].examples, predicted)
    print(json.dumps(results, indent=2))
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from pipeline.benchmarks.datasets.models import BenchmarkExample, Entity

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
        }


def _spans_overlap(a: Entity, b: Entity) -> int:
    """Return overlap length between two entity spans, 0 if none."""
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    return max(0, end - start)


def _text_normalize(t: str) -> str:
    """Lowercase, strip, collapse whitespace for text-based matching."""
    return " ".join(t.lower().split())


def _exact_match_counts(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
    entity_type: str | None = None,
) -> _Counts:
    """Count exact-match TP/FP/FN.

    Uses span-based matching when offsets are available, otherwise falls
    back to normalized text + type matching.
    """
    gold = [
        e for e in gold_entities if entity_type is None or e.entity_type == entity_type
    ]
    pred = [
        e for e in pred_entities if entity_type is None or e.entity_type == entity_type
    ]

    # Check if predictions have valid offsets
    has_offsets = any(e.start >= 0 and e.end > e.start for e in pred)

    if has_offsets:
        gold_set = {(e.start, e.end, e.entity_type) for e in gold}
        pred_set = {(e.start, e.end, e.entity_type) for e in pred}
    else:
        # Text-based matching
        gold_set = {(_text_normalize(e.text), e.entity_type) for e in gold}
        pred_set = {(_text_normalize(e.text), e.entity_type) for e in pred}

    tp = len(gold_set & pred_set)
    return _Counts(tp=tp, fp=len(pred_set) - tp, fn=len(gold_set) - tp)


def _partial_match_counts(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
    entity_type: str | None = None,
) -> _Counts:
    """Count partial-match TP/FP/FN.

    Uses span overlap when offsets are available, otherwise uses substring/
    containment matching on normalized text.
    """
    gold = [
        e for e in gold_entities if entity_type is None or e.entity_type == entity_type
    ]
    pred = [
        e for e in pred_entities if entity_type is None or e.entity_type == entity_type
    ]

    has_offsets = any(e.start >= 0 and e.end > e.start for e in pred)

    pairs: list[tuple[int, int, int]] = []  # (score, gold_idx, pred_idx)
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g.entity_type == p.entity_type:
                if has_offsets:
                    ov = _spans_overlap(g, p)
                    if ov > 0:
                        pairs.append((ov, gi, pi))
                else:
                    # Text-based: score by substring containment or token overlap
                    gt = _text_normalize(g.text)
                    pt = _text_normalize(p.text)
                    if gt in pt or pt in gt:
                        pairs.append((max(len(gt), len(pt)), gi, pi))
                    else:
                        # Token overlap
                        g_tokens = set(gt.split())
                        p_tokens = set(pt.split())
                        overlap = len(g_tokens & p_tokens)
                        if overlap > 0:
                            pairs.append((overlap, gi, pi))

    pairs.sort(reverse=True)

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0
    for _, gi, pi in pairs:
        if gi not in matched_gold and pi not in matched_pred:
            tp += 1
            matched_gold.add(gi)
            matched_pred.add(pi)

    return _Counts(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_extraction(
    gold_examples: list[BenchmarkExample],
    pred_examples: list[BenchmarkExample],
) -> dict[str, Any]:
    """Evaluate predicted entities against gold standard.

    Parameters
    ----------
    gold_examples:
        Gold-standard examples (from a benchmark split).
    pred_examples:
        Predicted examples. Must have the same ``doc_id`` values as gold.
        Entities should have ``start``, ``end``, and ``entity_type`` populated.

    Returns
    -------
    dict
        Structured results with ``exact_match`` and ``partial_match`` sections,
        each containing per-type and micro-averaged scores.
    """
    pred_by_id = {ex.doc_id: ex for ex in pred_examples}

    # Discover all entity types across gold
    all_types: set[str] = set()
    for ex in gold_examples:
        for ent in ex.entities:
            all_types.add(ent.entity_type)

    # Accumulate per-type counts across documents
    exact_by_type: dict[str, _Counts] = {t: _Counts() for t in sorted(all_types)}
    partial_by_type: dict[str, _Counts] = {t: _Counts() for t in sorted(all_types)}
    exact_micro = _Counts()
    partial_micro = _Counts()

    for gold_ex in gold_examples:
        pred_ex = pred_by_id.get(gold_ex.doc_id)
        pred_ents = pred_ex.entities if pred_ex else []

        # Per-type
        for etype in sorted(all_types):
            ec = _exact_match_counts(gold_ex.entities, pred_ents, etype)
            exact_by_type[etype].tp += ec.tp
            exact_by_type[etype].fp += ec.fp
            exact_by_type[etype].fn += ec.fn

            pc = _partial_match_counts(gold_ex.entities, pred_ents, etype)
            partial_by_type[etype].tp += pc.tp
            partial_by_type[etype].fp += pc.fp
            partial_by_type[etype].fn += pc.fn

        # Micro (all types together)
        ec_all = _exact_match_counts(gold_ex.entities, pred_ents)
        exact_micro.tp += ec_all.tp
        exact_micro.fp += ec_all.fp
        exact_micro.fn += ec_all.fn

        pc_all = _partial_match_counts(gold_ex.entities, pred_ents)
        partial_micro.tp += pc_all.tp
        partial_micro.fp += pc_all.fp
        partial_micro.fn += pc_all.fn

    return {
        "num_documents": len(gold_examples),
        "entity_types": sorted(all_types),
        "exact_match": {
            "per_type": {t: c.to_dict() for t, c in exact_by_type.items()},
            "micro_avg": exact_micro.to_dict(),
        },
        "partial_match": {
            "per_type": {t: c.to_dict() for t, c in partial_by_type.items()},
            "micro_avg": partial_micro.to_dict(),
        },
    }


def evaluate_dataset(
    dataset_name: str,
    pred_examples: list[BenchmarkExample],
    split: str = "test",
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Convenience: load a gold dataset by name and evaluate predictions against it.

    Parameters
    ----------
    dataset_name:
        Name registered in ``benchmarks.datasets.loader.REGISTRY``.
    pred_examples:
        Predicted examples with entities.
    split:
        Which split to evaluate against (default ``"test"``).
    base_dir:
        Override data directory.

    Returns
    -------
    dict
        Results dict with ``dataset``, ``split``, and metric sections.
    """
    from pipeline.benchmarks.datasets.loader import load_dataset

    ds = load_dataset(dataset_name, base_dir)
    if split not in ds.splits:
        msg = f"Split '{split}' not found in {dataset_name}. Available: {list(ds.splits.keys())}"
        raise ValueError(msg)

    gold = ds.splits[split].examples
    results = evaluate_extraction(gold, pred_examples)
    return {"dataset": dataset_name, "split": split, **results}


# ---------------------------------------------------------------------------
# CLI: evaluate a predictions JSON file against a gold dataset
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for evaluating extraction predictions.

    Usage:
        python -m benchmarks.metrics.extraction_metrics \\
            --dataset bc5cdr --split test --predictions preds.json

    Predictions JSON format:
        [
            {
                "doc_id": "12345",
                "entities": [
                    {"text": "aspirin", "entity_type": "Chemical", "start": 0, "end": 7}
                ]
            }
        ]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate NER extraction metrics")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. bc5cdr)")
    parser.add_argument(
        "--split", default="test", help="Split to evaluate (default: test)"
    )
    parser.add_argument(
        "--predictions", required=True, help="Path to predictions JSON file"
    )
    parser.add_argument("--base-dir", help="Override dataset base directory")
    parser.add_argument("--output", help="Write results to file (default: stdout)")
    args = parser.parse_args()

    with open(args.predictions) as f:
        raw = json.load(f)

    pred_examples = [
        BenchmarkExample(
            doc_id=item["doc_id"],
            text=item.get("text", ""),
            entities=[
                Entity(
                    text=e.get("text", ""),
                    entity_type=e["entity_type"],
                    start=e["start"],
                    end=e["end"],
                    mesh_id=e.get("mesh_id", ""),
                )
                for e in item.get("entities", [])
            ],
        )
        for item in raw
    ]

    results = evaluate_dataset(args.dataset, pred_examples, args.split, args.base_dir)
    output = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
