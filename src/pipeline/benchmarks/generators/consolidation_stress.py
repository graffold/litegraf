"""Generate a synthetic knowledge graph with known duplicates for consolidation stress testing.

Produces protein and disease nodes with deliberate duplicates (typos, synonym
variants, case differences, UniProt/MONDO ID collisions) plus relationships
between them.  Ground-truth cluster assignments are included so consolidation
algorithms can be evaluated against a known answer.

Output format (JSON):
{
  "seed": 42,
  "nodes": [
    {
      "id": "n0",
      "name": "TP53",
      "type": "Protein",
      "uniprot_id": "P04637",
      "gene_symbol": "TP53",
      "synonyms": ["p53", "tumor protein p53"],
      "cluster_id": "c0"          # ground-truth duplicate cluster
    },
    ...
  ],
  "edges": [
    {
      "source": "n0",
      "target": "n10",
      "relation": "ASSOCIATES_WITH",
      "pmid": "PMID:00000001",
      "confidence": 0.92
    },
    ...
  ],
  "ground_truth": {
    "clusters": {
      "c0": ["n0", "n1", "n2"],   # these three nodes are the same entity
      ...
    },
    "total_unique_entities": 20,
    "total_duplicate_nodes": 40
  }
}

Usage:
    python -m benchmarks.generators.consolidation_stress
    python -m benchmarks.generators.consolidation_stress --seed 123 --proteins 30 --diseases 20 -o out.json
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical entity pools
# ---------------------------------------------------------------------------

_PROTEINS = [
    ("P04637", "TP53", "tumor protein p53", ["p53", "Li-Fraumeni"]),
    ("P01308", "INS", "insulin", ["proinsulin"]),
    ("P05231", "IL6", "interleukin-6", ["IL-6", "BSF-2"]),
    ("P01375", "TNF", "tumor necrosis factor", ["TNF-alpha", "cachectin"]),
    ("P01589", "IL2RA", "interleukin-2 receptor subunit alpha", ["CD25", "Tac"]),
    ("P10145", "CXCL8", "interleukin-8", ["IL-8", "NAP-1"]),
    ("P09038", "FGF2", "fibroblast growth factor 2", ["bFGF"]),
    ("P15692", "VEGFA", "vascular endothelial growth factor A", ["VEGF"]),
    ("P01137", "TGFB1", "transforming growth factor beta 1", ["TGF-beta1"]),
    ("P17936", "IGFBP3", "insulin-like growth factor binding protein 3", []),
    ("P02768", "ALB", "albumin", ["serum albumin"]),
    ("P68871", "HBB", "hemoglobin subunit beta", ["beta-globin"]),
    ("P00738", "HP", "haptoglobin", []),
    ("P02787", "TF", "transferrin", ["serotransferrin"]),
    ("P01024", "C3", "complement C3", []),
    ("P02647", "APOA1", "apolipoprotein A-I", ["ApoA1"]),
    ("P04114", "APOB", "apolipoprotein B", ["ApoB-100"]),
    ("P06727", "APOA4", "apolipoprotein A-IV", ["ApoA4"]),
    ("P02652", "APOA2", "apolipoprotein A-II", ["ApoA2"]),
    ("P02766", "TTR", "transthyretin", ["prealbumin"]),
]

_DISEASES = [
    ("MONDO:0004979", "asthma", ["bronchial asthma"]),
    ("MONDO:0005015", "diabetes mellitus", ["diabetes", "DM"]),
    ("MONDO:0005044", "hypertension", ["high blood pressure", "HTN"]),
    ("MONDO:0005148", "type 2 diabetes", ["T2D", "NIDDM"]),
    ("MONDO:0005010", "Alzheimer disease", ["AD", "Alzheimer's"]),
    ("MONDO:0005301", "multiple sclerosis", ["MS"]),
    ("MONDO:0005812", "lung cancer", ["NSCLC", "pulmonary neoplasm"]),
    ("MONDO:0005575", "colorectal cancer", ["CRC"]),
    ("MONDO:0005180", "Parkinson disease", ["PD", "Parkinson's"]),
    ("MONDO:0005090", "schizophrenia", []),
    ("MONDO:0005265", "inflammatory bowel disease", ["IBD"]),
    ("MONDO:0005011", "Crohn disease", ["Crohn's"]),
    ("MONDO:0005147", "type 1 diabetes", ["T1D", "IDDM"]),
    ("MONDO:0005041", "glaucoma", []),
    ("MONDO:0005083", "psoriasis", []),
    ("MONDO:0005027", "epilepsy", []),
    ("MONDO:0005046", "heart failure", ["CHF", "cardiac failure"]),
    ("MONDO:0005068", "myocardial infarction", ["MI", "heart attack"]),
    ("MONDO:0005020", "chronic kidney disease", ["CKD"]),
    ("MONDO:0005105", "melanoma", []),
]


# ---------------------------------------------------------------------------
# Duplicate-generation strategies
# ---------------------------------------------------------------------------


def _typo(name: str, rng: random.Random) -> str:
    """Introduce a single-character typo."""
    if len(name) < 3:
        return name + rng.choice(string.ascii_lowercase)
    idx = rng.randint(1, len(name) - 2)
    c = rng.choice(string.ascii_lowercase)
    return name[:idx] + c + name[idx + 1 :]


def _case_variant(name: str, rng: random.Random) -> str:
    return rng.choice([name.upper(), name.lower(), name.title()])


def _synonym_variant(synonyms: list[str], rng: random.Random) -> str | None:
    return rng.choice(synonyms) if synonyms else None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    id: str
    name: str
    type: str  # "Protein" | "Disease"
    cluster_id: str
    uniprot_id: str = ""
    mondo_id: str = ""
    gene_symbol: str = ""
    synonyms: list[str] = field(default_factory=list)


@dataclass
class _Edge:
    source: str
    target: str
    relation: str
    pmid: str
    confidence: float


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(
    *,
    seed: int = 42,
    n_proteins: int = 20,
    n_diseases: int = 20,
    dups_per_entity: int = 2,
    edges_per_cluster: int = 3,
) -> dict:
    """Return a dict matching the documented JSON schema."""
    rng = random.Random(seed)

    proteins = _PROTEINS[:n_proteins]
    diseases = _DISEASES[:n_diseases]

    nodes: list[_Node] = []
    clusters: dict[str, list[str]] = {}
    node_idx = 0

    strategies = [_typo, _case_variant]

    def _make_id() -> str:
        nonlocal node_idx
        nid = f"n{node_idx}"
        node_idx += 1
        return nid

    # --- Proteins ---
    for i, (uid, gene, name, syns) in enumerate(proteins):
        cid = f"cp{i}"
        cluster_ids: list[str] = []

        # canonical node
        nid = _make_id()
        nodes.append(
            _Node(
                id=nid,
                name=name,
                type="Protein",
                cluster_id=cid,
                uniprot_id=uid,
                gene_symbol=gene,
                synonyms=list(syns),
            )
        )
        cluster_ids.append(nid)

        # duplicate nodes
        for _ in range(dups_per_entity):
            nid = _make_id()
            strat = rng.choice(strategies)
            dup_name = _synonym_variant(syns, rng) or strat(name, rng)
            # some dups keep the uniprot_id, some don't
            keep_uid = rng.random() < 0.5
            nodes.append(
                _Node(
                    id=nid,
                    name=dup_name,
                    type="Protein",
                    cluster_id=cid,
                    uniprot_id=uid if keep_uid else "",
                    gene_symbol=gene if rng.random() < 0.5 else "",
                    synonyms=[],
                )
            )
            cluster_ids.append(nid)

        clusters[cid] = cluster_ids

    # --- Diseases ---
    for i, (mondo, name, syns) in enumerate(diseases):
        cid = f"cd{i}"
        cluster_ids = []

        nid = _make_id()
        nodes.append(
            _Node(
                id=nid,
                name=name,
                type="Disease",
                cluster_id=cid,
                mondo_id=mondo,
                synonyms=list(syns),
            )
        )
        cluster_ids.append(nid)

        for _ in range(dups_per_entity):
            nid = _make_id()
            strat = rng.choice(strategies)
            dup_name = _synonym_variant(syns, rng) or strat(name, rng)
            keep_mondo = rng.random() < 0.5
            nodes.append(
                _Node(
                    id=nid,
                    name=dup_name,
                    type="Disease",
                    cluster_id=cid,
                    mondo_id=mondo if keep_mondo else "",
                    synonyms=[],
                )
            )
            cluster_ids.append(nid)

        clusters[cid] = cluster_ids

    # --- Edges (between protein and disease clusters) ---
    edges: list[_Edge] = []
    protein_clusters = [c for c in clusters if c.startswith("cp")]
    disease_clusters = [c for c in clusters if c.startswith("cd")]

    for pc in protein_clusters:
        targets = rng.sample(
            disease_clusters, min(edges_per_cluster, len(disease_clusters))
        )
        for dc in targets:
            src = rng.choice(clusters[pc])
            tgt = rng.choice(clusters[dc])
            edges.append(
                _Edge(
                    source=src,
                    target=tgt,
                    relation="ASSOCIATES_WITH",
                    pmid=f"PMID:{rng.randint(10000000, 99999999)}",
                    confidence=round(rng.uniform(0.5, 1.0), 3),
                )
            )

    total_unique = len(proteins) + len(diseases)
    return {
        "seed": seed,
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "ground_truth": {
            "clusters": clusters,
            "total_unique_entities": total_unique,
            "total_duplicate_nodes": len(nodes) - total_unique,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate consolidation stress-test dataset"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proteins", type=int, default=20)
    parser.add_argument("--diseases", type=int, default=20)
    parser.add_argument("--dups", type=int, default=2, help="Duplicates per entity")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    data = generate(
        seed=args.seed,
        n_proteins=args.proteins,
        n_diseases=args.diseases,
        dups_per_entity=args.dups,
    )

    text = json.dumps(data, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
