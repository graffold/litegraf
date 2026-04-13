"""Knowledge graph quality metrics for consolidation, mapping, provenance, and dedup.

Evaluates KG construction quality against synthetic ground truth from the
generators in ``benchmarks.generators``:

- **Consolidation accuracy / false merge rate** — against ``consolidation_stress``
- **UniProt / MONDO mapping rates** — fraction of nodes with canonical IDs
- **Provenance completeness** — against ``provenance_annotator``
- **Relationship dedup rate** — before/after edge ratio
- **Evidence preservation** — PMIDs, confidence scores, excerpts retained
- **Contradiction detection recall** — against ``contradiction_pairs``

Usage:
    python -m benchmarks.metrics.kg_quality_metrics \\
        --consolidation consolidation.json \\
        --predicted-clusters pred_clusters.json

    python -m benchmarks.metrics.kg_quality_metrics \\
        --provenance provenance.json \\
        --predicted-provenance pred_prov.json

    python -m benchmarks.metrics.kg_quality_metrics \\
        --contradictions contradictions.json \\
        --predicted-contradictions pred_contra.json

Input formats are documented per function.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ---------------------------------------------------------------------------
# 1. Consolidation accuracy & false merge rate
# ---------------------------------------------------------------------------


def evaluate_consolidation(
    ground_truth: dict,
    predicted_clusters: dict[str, list[str]],
) -> dict[str, Any]:
    """Evaluate entity consolidation against ground truth clusters.

    Parameters
    ----------
    ground_truth:
        Output of ``consolidation_stress.generate()`` — must contain
        ``ground_truth.clusters`` mapping cluster_id → list of node IDs.
    predicted_clusters:
        Mapping of predicted cluster_id → list of node IDs.  Node IDs must
        match those in the ground truth.

    Returns
    -------
    dict with ``consolidation_accuracy``, ``false_merge_rate``, and details.
    """
    gt_clusters: dict[str, list[str]] = ground_truth["ground_truth"]["clusters"]

    # Build node → gt_cluster lookup
    node_to_gt: dict[str, str] = {}
    for cid, members in gt_clusters.items():
        for nid in members:
            node_to_gt[nid] = cid

    # Pairwise evaluation: for every pair of nodes in the same predicted
    # cluster, check if they belong to the same GT cluster.
    correct_merges = 0
    false_merges = 0
    total_pred_pairs = 0

    for members in predicted_clusters.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                total_pred_pairs += 1
                a, b = members[i], members[j]
                if node_to_gt.get(a) == node_to_gt.get(b) and a in node_to_gt:
                    correct_merges += 1
                else:
                    false_merges += 1

    # Recall: how many GT same-cluster pairs were captured?
    total_gt_pairs = 0
    for members in gt_clusters.values():
        n = len(members)
        total_gt_pairs += n * (n - 1) // 2

    # Build node → pred_cluster lookup for recall
    node_to_pred: dict[str, str] = {}
    for cid, members in predicted_clusters.items():
        for nid in members:
            node_to_pred[nid] = cid

    recalled_pairs = 0
    for members in gt_clusters.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if node_to_pred.get(a) == node_to_pred.get(b) and a in node_to_pred:
                    recalled_pairs += 1

    accuracy = correct_merges / total_pred_pairs if total_pred_pairs else 0.0
    false_merge_rate = false_merges / total_pred_pairs if total_pred_pairs else 0.0
    recall = recalled_pairs / total_gt_pairs if total_gt_pairs else 0.0

    return {
        "consolidation_accuracy": round(accuracy, 4),
        "false_merge_rate": round(false_merge_rate, 4),
        "merge_recall": round(recall, 4),
        "correct_merges": correct_merges,
        "false_merges": false_merges,
        "total_predicted_pairs": total_pred_pairs,
        "total_gt_pairs": total_gt_pairs,
        "recalled_pairs": recalled_pairs,
        "num_predicted_clusters": len(predicted_clusters),
        "num_gt_clusters": len(gt_clusters),
    }


# ---------------------------------------------------------------------------
# 2. UniProt / MONDO mapping rates
# ---------------------------------------------------------------------------


def evaluate_mapping_rates(nodes: list[dict]) -> dict[str, Any]:
    """Compute UniProt and MONDO mapping rates from a list of KG nodes.

    Parameters
    ----------
    nodes:
        List of node dicts, each with at least ``type`` and optionally
        ``uniprot_id`` / ``mondo_id``.

    Returns
    -------
    dict with ``uniprot_mapping_rate``, ``mondo_mapping_rate``, and counts.
    """
    proteins = [n for n in nodes if n.get("type") == "Protein"]
    diseases = [n for n in nodes if n.get("type") == "Disease"]

    uniprot_mapped = sum(1 for n in proteins if n.get("uniprot_id"))
    mondo_mapped = sum(1 for n in diseases if n.get("mondo_id"))

    return {
        "uniprot_mapping_rate": round(uniprot_mapped / len(proteins), 4)
        if proteins
        else 0.0,
        "mondo_mapping_rate": round(mondo_mapped / len(diseases), 4)
        if diseases
        else 0.0,
        "proteins_total": len(proteins),
        "proteins_with_uniprot": uniprot_mapped,
        "diseases_total": len(diseases),
        "diseases_with_mondo": mondo_mapped,
    }


# ---------------------------------------------------------------------------
# 3. Provenance completeness
# ---------------------------------------------------------------------------


def evaluate_provenance(
    ground_truth: dict,
    predicted_valid: dict[str, bool],
) -> dict[str, Any]:
    """Evaluate provenance chain completeness.

    Parameters
    ----------
    ground_truth:
        Output of ``provenance_annotator.generate()`` — contains ``chains``.
    predicted_valid:
        Mapping of chain_id → bool indicating whether the system judged
        the chain as valid (complete provenance).

    Returns
    -------
    dict with ``provenance_completeness``, accuracy, and breakdown.
    """
    chains = ground_truth["chains"]

    tp = fp = tn = fn = 0
    for chain in chains:
        cid = chain["chain_id"]
        gt_valid = chain["ground_truth"]["valid_chain"]
        pred_valid = predicted_valid.get(cid, False)

        if gt_valid and pred_valid:
            tp += 1
        elif gt_valid and not pred_valid:
            fn += 1
        elif not gt_valid and pred_valid:
            fp += 1
        else:
            tn += 1

    total = len(chains)
    gt_valid_count = sum(1 for c in chains if c["ground_truth"]["valid_chain"])
    accuracy = (tp + tn) / total if total else 0.0
    completeness = tp / gt_valid_count if gt_valid_count else 0.0

    return {
        "provenance_completeness": round(completeness, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_chains": total,
        "gt_valid_chains": gt_valid_count,
    }


# ---------------------------------------------------------------------------
# 4. Relationship dedup rate
# ---------------------------------------------------------------------------


def evaluate_relationship_dedup(
    edges_before: int,
    edges_after: int,
) -> dict[str, Any]:
    """Compute relationship deduplication rate.

    Parameters
    ----------
    edges_before:
        Number of edges before consolidation.
    edges_after:
        Number of edges after consolidation.

    Returns
    -------
    dict with ``dedup_rate`` (1 - after/before) and counts.
    """
    ratio = edges_after / edges_before if edges_before else 1.0
    return {
        "dedup_rate": round(1.0 - ratio, 4),
        "before_after_ratio": round(ratio, 4),
        "edges_before": edges_before,
        "edges_after": edges_after,
    }


# ---------------------------------------------------------------------------
# 5. Evidence preservation
# ---------------------------------------------------------------------------


def evaluate_evidence_preservation(
    original_edges: list[dict],
    consolidated_edges: list[dict],
) -> dict[str, Any]:
    """Measure how well consolidation preserves evidence.

    Parameters
    ----------
    original_edges:
        Pre-consolidation edges, each with ``pmid``, ``confidence``, and
        optionally ``excerpt``.
    consolidated_edges:
        Post-consolidation edges, each with ``pmids`` (list), ``confidence_scores``
        (list), and optionally ``abstract_excerpts`` (list).

    Returns
    -------
    dict with preservation rates for PMIDs, confidence scores, and excerpts.
    """
    # Collect all original evidence
    all_pmids = {e["pmid"] for e in original_edges if e.get("pmid")}
    all_confidences = {e["confidence"] for e in original_edges if "confidence" in e}
    all_excerpts = {e["excerpt"] for e in original_edges if e.get("excerpt")}

    # Collect preserved evidence
    preserved_pmids: set[str] = set()
    preserved_confidences: set[float] = set()
    preserved_excerpts: set[str] = set()
    for e in consolidated_edges:
        preserved_pmids.update(e.get("pmids", []))
        preserved_confidences.update(e.get("confidence_scores", []))
        preserved_excerpts.update(e.get("abstract_excerpts", []))

    pmid_rate = len(all_pmids & preserved_pmids) / len(all_pmids) if all_pmids else 1.0
    conf_rate = (
        len(all_confidences & preserved_confidences) / len(all_confidences)
        if all_confidences
        else 1.0
    )
    excerpt_rate = (
        len(all_excerpts & preserved_excerpts) / len(all_excerpts)
        if all_excerpts
        else 1.0
    )

    return {
        "pmid_preservation_rate": round(pmid_rate, 4),
        "confidence_preservation_rate": round(conf_rate, 4),
        "excerpt_preservation_rate": round(excerpt_rate, 4),
        "original_pmids": len(all_pmids),
        "preserved_pmids": len(all_pmids & preserved_pmids),
        "original_confidences": len(all_confidences),
        "preserved_confidences": len(all_confidences & preserved_confidences),
        "original_excerpts": len(all_excerpts),
        "preserved_excerpts": len(all_excerpts & preserved_excerpts),
    }


# ---------------------------------------------------------------------------
# 6. Contradiction detection recall
# ---------------------------------------------------------------------------


def evaluate_contradiction_detection(
    ground_truth: dict,
    predicted_contradictions: dict[str, bool],
) -> dict[str, Any]:
    """Evaluate contradiction detection recall.

    Parameters
    ----------
    ground_truth:
        Output of ``contradiction_pairs.generate()`` — contains ``pairs``.
    predicted_contradictions:
        Mapping of pair_id → bool indicating whether the system detected
        a contradiction.

    Returns
    -------
    dict with ``recall``, ``precision``, ``f1``, and breakdown by type.
    """
    pairs = ground_truth["pairs"]

    tp = fp = fn = 0
    by_type: dict[str, dict[str, int]] = {}

    for pair in pairs:
        pid = pair["pair_id"]
        ctype = pair["contradiction_type"]
        # All pairs in the generator are contradictory (label == "contradictory")
        gt_positive = pair.get("label") == "contradictory"
        pred_positive = predicted_contradictions.get(pid, False)

        if ctype not in by_type:
            by_type[ctype] = {"tp": 0, "fn": 0, "total": 0}
        by_type[ctype]["total"] += 1

        if gt_positive and pred_positive:
            tp += 1
            by_type[ctype]["tp"] += 1
        elif gt_positive and not pred_positive:
            fn += 1
            by_type[ctype]["fn"] += 1
        elif not gt_positive and pred_positive:
            fp += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_type = {}
    for ctype, counts in sorted(by_type.items()):
        t = counts["total"]
        r = counts["tp"] / t if t else 0.0
        per_type[ctype] = {"recall": round(r, 4), "detected": counts["tp"], "total": t}

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_pairs": len(pairs),
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def full_report(
    consolidation_gt: dict | None = None,
    predicted_clusters: dict[str, list[str]] | None = None,
    nodes: list[dict] | None = None,
    provenance_gt: dict | None = None,
    predicted_provenance: dict[str, bool] | None = None,
    edges_before: int | None = None,
    edges_after: int | None = None,
    original_edges: list[dict] | None = None,
    consolidated_edges: list[dict] | None = None,
    contradiction_gt: dict | None = None,
    predicted_contradictions: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Run all applicable metrics and return a combined report."""
    report: dict[str, Any] = {}

    if consolidation_gt is not None and predicted_clusters is not None:
        report["consolidation"] = evaluate_consolidation(
            consolidation_gt, predicted_clusters
        )

    if nodes is not None:
        report["mapping_rates"] = evaluate_mapping_rates(nodes)

    if provenance_gt is not None and predicted_provenance is not None:
        report["provenance"] = evaluate_provenance(provenance_gt, predicted_provenance)

    if edges_before is not None and edges_after is not None:
        report["relationship_dedup"] = evaluate_relationship_dedup(
            edges_before, edges_after
        )

    if original_edges is not None and consolidated_edges is not None:
        report["evidence_preservation"] = evaluate_evidence_preservation(
            original_edges, consolidated_edges
        )

    if contradiction_gt is not None and predicted_contradictions is not None:
        report["contradiction_detection"] = evaluate_contradiction_detection(
            contradiction_gt, predicted_contradictions
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="KG quality metrics evaluation")
    parser.add_argument("--consolidation", help="consolidation_stress JSON file")
    parser.add_argument("--predicted-clusters", help="Predicted clusters JSON file")
    parser.add_argument("--provenance", help="provenance_annotator JSON file")
    parser.add_argument("--predicted-provenance", help="Predicted provenance JSON file")
    parser.add_argument("--contradictions", help="contradiction_pairs JSON file")
    parser.add_argument(
        "--predicted-contradictions", help="Predicted contradictions JSON file"
    )
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    report: dict[str, Any] = {}

    if args.consolidation and args.predicted_clusters:
        with open(args.consolidation) as f:
            gt = json.load(f)
        with open(args.predicted_clusters) as f:
            pred = json.load(f)
        report["consolidation"] = evaluate_consolidation(gt, pred)
        # Also compute mapping rates from the consolidation nodes
        report["mapping_rates"] = evaluate_mapping_rates(gt["nodes"])

    if args.provenance and args.predicted_provenance:
        with open(args.provenance) as f:
            gt = json.load(f)
        with open(args.predicted_provenance) as f:
            pred = json.load(f)
        report["provenance"] = evaluate_provenance(gt, pred)

    if args.contradictions and args.predicted_contradictions:
        with open(args.contradictions) as f:
            gt = json.load(f)
        with open(args.predicted_contradictions) as f:
            pred = json.load(f)
        report["contradiction_detection"] = evaluate_contradiction_detection(gt, pred)

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
