"""Generate synthetic provenance-annotated chains from biomedical abstracts.

Each chain traces an entity or relationship back through a sequence of
extraction steps: source document → chunk → extraction pass → entity node.
Ground-truth annotations mark the correct provenance path so that
provenance-tracking systems can be evaluated.

Output format (JSON):
{
  "seed": 42,
  "chains": [
    {
      "chain_id": "chain_0",
      "entity": {
        "name": "IL-6",
        "type": "Protein",
        "uniprot_id": "P05231"
      },
      "steps": [
        {
          "step": 0,
          "type": "source",
          "pmid": "PMID:12345678",
          "title": "IL-6 in cardiovascular disease",
          "source_type": "pubmed"
        },
        {
          "step": 1,
          "type": "chunk",
          "chunk_index": 2,
          "text": "IL-6 levels were elevated in patients with heart failure...",
          "token_count": 128
        },
        {
          "step": 2,
          "type": "extraction",
          "pass_number": 1,
          "method": "llm_ner",
          "confidence": 0.94
        },
        {
          "step": 3,
          "type": "entity_node",
          "node_id": "prot_IL6",
          "properties": {"name": "IL-6", "uniprot_id": "P05231"}
        }
      ],
      "ground_truth": {
        "valid_chain": true,
        "expected_entity_name": "IL-6",
        "expected_source_pmid": "PMID:12345678"
      }
    },
    ...
  ]
}

Usage:
    python -m benchmarks.generators.provenance_annotator
    python -m benchmarks.generators.provenance_annotator --seed 7 --chains 50 -o out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Entity + abstract pools
# ---------------------------------------------------------------------------

_ENTITIES = [
    ("Protein", "IL-6", "P05231", "interleukin-6"),
    ("Protein", "TNF-alpha", "P01375", "tumor necrosis factor"),
    ("Protein", "VEGF", "P15692", "vascular endothelial growth factor A"),
    ("Protein", "TP53", "P04637", "tumor protein p53"),
    ("Protein", "insulin", "P01308", "insulin"),
    ("Protein", "albumin", "P02768", "serum albumin"),
    ("Protein", "CXCL8", "P10145", "interleukin-8"),
    ("Protein", "TGF-beta1", "P01137", "transforming growth factor beta 1"),
    ("Disease", "type 2 diabetes", "MONDO:0005148", "T2D"),
    ("Disease", "Alzheimer disease", "MONDO:0005010", "AD"),
    ("Disease", "heart failure", "MONDO:0005046", "CHF"),
    ("Disease", "lung cancer", "MONDO:0005812", "NSCLC"),
    ("Disease", "hypertension", "MONDO:0005044", "HTN"),
    ("Disease", "Parkinson disease", "MONDO:0005180", "PD"),
    ("Disease", "chronic kidney disease", "MONDO:0005020", "CKD"),
    ("Disease", "asthma", "MONDO:0004979", "bronchial asthma"),
]

_SOURCE_TYPES = ["pubmed", "pmc", "biorxiv", "pdf"]
_EXTRACTION_METHODS = ["llm_ner", "regex_fallback", "gleaning_pass"]

_ABSTRACT_TEMPLATES = [
    "{name} levels were significantly elevated in patients with {disease}, "
    "suggesting a role in disease pathogenesis (p < 0.001).",
    "We investigated the association between {name} and {disease} in a cohort "
    "of 500 patients. {name} was identified as a potential biomarker.",
    "Plasma {name} concentration was measured in {disease} patients and healthy "
    "controls. Results indicate {name} may serve as a diagnostic indicator.",
    "A proteomics analysis revealed {name} as differentially expressed in "
    "{disease}, with a fold change of 2.3 compared to controls.",
]

_DISEASES_FOR_CONTEXT = [
    "cardiovascular disease",
    "type 2 diabetes",
    "Alzheimer disease",
    "lung cancer",
    "chronic kidney disease",
    "heart failure",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(*, seed: int = 42, n_chains: int = 30) -> dict:
    """Return a dict matching the documented JSON schema."""
    rng = random.Random(seed)
    chains = []

    for i in range(n_chains):
        etype, name, eid, long_name = rng.choice(_ENTITIES)
        disease_ctx = rng.choice(_DISEASES_FOR_CONTEXT)
        source_type = rng.choice(_SOURCE_TYPES)
        pmid = f"PMID:{rng.randint(10000000, 99999999)}"
        method = rng.choice(_EXTRACTION_METHODS)
        confidence = round(rng.uniform(0.7, 1.0), 3)
        chunk_idx = rng.randint(0, 8)
        pass_number = 2 if method == "gleaning_pass" else 1

        abstract_text = rng.choice(_ABSTRACT_TEMPLATES).format(
            name=name, disease=disease_ctx
        )
        token_count = len(abstract_text.split()) * 4 // 3  # rough estimate

        # Decide if this chain is valid or has a corruption
        is_valid = rng.random() < 0.8
        corrupted_name = name
        if not is_valid:
            # Introduce a name mismatch to simulate broken provenance
            corrupted_name = long_name

        id_key = "uniprot_id" if etype == "Protein" else "mondo_id"
        node_id = f"{'prot' if etype == 'Protein' else 'dis'}_{name.replace(' ', '_').replace('-', '_')}"

        chain = {
            "chain_id": f"chain_{i}",
            "entity": {
                "name": name,
                "type": etype,
                id_key: eid,
            },
            "steps": [
                {
                    "step": 0,
                    "type": "source",
                    "pmid": pmid,
                    "title": f"{name} in {disease_ctx}",
                    "source_type": source_type,
                },
                {
                    "step": 1,
                    "type": "chunk",
                    "chunk_index": chunk_idx,
                    "text": abstract_text,
                    "token_count": token_count,
                },
                {
                    "step": 2,
                    "type": "extraction",
                    "pass_number": pass_number,
                    "method": method,
                    "confidence": confidence,
                },
                {
                    "step": 3,
                    "type": "entity_node",
                    "node_id": node_id,
                    "properties": {"name": corrupted_name, id_key: eid},
                },
            ],
            "ground_truth": {
                "valid_chain": is_valid,
                "expected_entity_name": name,
                "expected_source_pmid": pmid,
            },
        }
        chains.append(chain)

    return {"seed": seed, "chains": chains}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate provenance-annotated chain dataset"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chains", type=int, default=30)
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    data = generate(seed=args.seed, n_chains=args.chains)

    text = json.dumps(data, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
