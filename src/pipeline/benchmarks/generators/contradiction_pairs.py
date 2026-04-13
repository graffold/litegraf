"""Generate synthetic abstract pairs containing contradictory biomedical claims.

Each pair consists of two abstracts that make opposing assertions about the
same protein-disease relationship (e.g., "upregulated" vs "downregulated",
"associated" vs "no association").  Ground truth labels indicate the
contradiction type.

Output format (JSON):
{
  "seed": 42,
  "pairs": [
    {
      "pair_id": "pair_0",
      "abstract_a": {
        "pmid": "SYNTH:00000001",
        "title": "...",
        "text": "...",
        "claim": "IL-6 is upregulated in type 2 diabetes"
      },
      "abstract_b": {
        "pmid": "SYNTH:00000002",
        "title": "...",
        "text": "...",
        "claim": "IL-6 is downregulated in type 2 diabetes"
      },
      "protein": "IL-6",
      "disease": "type 2 diabetes",
      "contradiction_type": "direction",
      "label": "contradictory"
    },
    ...
  ]
}

Contradiction types:
  - "direction"   : up vs down regulation
  - "association"  : associated vs no association
  - "effect"       : protective vs harmful

Usage:
    python -m benchmarks.generators.contradiction_pairs
    python -m benchmarks.generators.contradiction_pairs --seed 7 --pairs 50 -o out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_PROTEINS = [
    "IL-6",
    "TNF-alpha",
    "VEGF",
    "TP53",
    "CXCL8",
    "TGF-beta1",
    "insulin",
    "albumin",
    "transferrin",
    "haptoglobin",
    "ApoA1",
    "ApoB",
    "FGF2",
    "IGFBP3",
    "hemoglobin beta",
    "complement C3",
    "IL-2R alpha",
    "transthyretin",
    "ApoA4",
    "ApoA2",
]

_DISEASES = [
    "type 2 diabetes",
    "Alzheimer disease",
    "lung cancer",
    "hypertension",
    "colorectal cancer",
    "Parkinson disease",
    "asthma",
    "multiple sclerosis",
    "heart failure",
    "chronic kidney disease",
    "inflammatory bowel disease",
    "myocardial infarction",
    "schizophrenia",
    "melanoma",
    "Crohn disease",
    "psoriasis",
    "epilepsy",
    "glaucoma",
    "type 1 diabetes",
]

_CONTRADICTION_TEMPLATES: list[tuple[str, str, str, str, str]] = [
    # (type, title_a, body_a, title_b, body_b)  — {P} and {D} are placeholders
    (
        "direction",
        "{P} is upregulated in {D}: a cohort study",
        "We measured plasma {P} levels in 200 patients with {D} and 200 controls. "
        "{P} was significantly upregulated in the {D} group (p < 0.001), suggesting "
        "a role in disease pathogenesis.",
        "{P} is downregulated in {D}: a population-based analysis",
        "In a population-based study of 500 individuals, serum {P} was significantly "
        "lower in {D} patients compared to healthy controls (p = 0.003), indicating "
        "downregulation in disease.",
    ),
    (
        "association",
        "{P} is associated with {D} risk",
        "Our genome-wide association study identified {P} as a significant risk "
        "factor for {D} (OR = 2.1, 95% CI 1.5-2.9). Elevated {P} levels predicted "
        "disease onset in a 10-year follow-up.",
        "No association between {P} and {D}",
        "We conducted a large-scale meta-analysis of 15 studies and found no "
        "statistically significant association between {P} levels and {D} risk "
        "(pooled OR = 1.02, 95% CI 0.88-1.18, p = 0.78).",
    ),
    (
        "effect",
        "{P} has a protective effect against {D}",
        "Administration of recombinant {P} in a mouse model of {D} reduced disease "
        "severity by 40% (p < 0.01). {P} appears to exert a protective effect "
        "through anti-inflammatory signaling.",
        "{P} exacerbates {D} progression",
        "Overexpression of {P} in transgenic mice accelerated {D} progression, "
        "with a 60% increase in disease markers at 12 weeks. {P} may promote "
        "pathological pathways in {D}.",
    ),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(*, seed: int = 42, n_pairs: int = 30) -> dict:
    """Return a dict matching the documented JSON schema."""
    rng = random.Random(seed)
    pairs = []
    pmid_counter = 1

    for i in range(n_pairs):
        protein = rng.choice(_PROTEINS)
        disease = rng.choice(_DISEASES)
        template = rng.choice(_CONTRADICTION_TEMPLATES)
        ctype, title_a, body_a, title_b, body_b = template

        def _fill(s: str, p: str = protein, d: str = disease) -> str:
            return s.replace("{P}", p).replace("{D}", d)

        pmid_a = f"SYNTH:{pmid_counter:08d}"
        pmid_counter += 1
        pmid_b = f"SYNTH:{pmid_counter:08d}"
        pmid_counter += 1

        pairs.append(
            {
                "pair_id": f"pair_{i}",
                "abstract_a": {
                    "pmid": pmid_a,
                    "title": _fill(title_a),
                    "text": _fill(body_a),
                    "claim": _fill(
                        title_a.split(":")[0] if ":" in title_a else title_a
                    ),
                },
                "abstract_b": {
                    "pmid": pmid_b,
                    "title": _fill(title_b),
                    "text": _fill(body_b),
                    "claim": _fill(
                        title_b.split(":")[0] if ":" in title_b else title_b
                    ),
                },
                "protein": protein,
                "disease": disease,
                "contradiction_type": ctype,
                "label": "contradictory",
            }
        )

    return {"seed": seed, "pairs": pairs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contradiction-pair dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    data = generate(seed=args.seed, n_pairs=args.pairs)

    text = json.dumps(data, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
