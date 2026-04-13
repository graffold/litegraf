# ChemProt — BioCreative VI Chemical-Protein Interaction Corpus

## Overview

ChemProt consists of 1,820 PubMed abstracts annotated with chemical-protein interactions by domain experts. It was the shared task dataset for BioCreative VI Track 5.

## Task

**Relation Extraction**: Given a PubMed abstract with annotated chemical and protein/gene entities, classify the relation between each chemical-protein pair into one of 5 CPR groups (or no relation).

## CPR Groups (Relation Types)

| Group | Relation Types |
|-------|---------------|
| CPR:3 | Upregulator, Activator, Indirect Upregulator |
| CPR:4 | Downregulator, Inhibitor, Indirect Downregulator |
| CPR:5 | Agonist, Agonist-Activator, Agonist-Inhibitor |
| CPR:6 | Antagonist |
| CPR:9 | Substrate, Product Of, Substrate Product Of |

## Splits

| Split | Documents |
|-------|-----------|
| Train | 1,020 |
| Dev | 612 |
| Test | 800 |

## Format

TSV files per split:
- `abstracts.tsv`: PMID, title, abstract
- `entities.tsv`: PMID, entity_id, type, start, end, text
- `relations.tsv`: PMID, CPR_group, eval_type, relation, arg1, arg2
- `gold_standard.tsv`: PMID, CPR_group, eval_type, relation, arg1, arg2

## Source & Citation

- **Paper**: Krallinger et al. (2017). "Overview of the BioCreative VI chemical-protein interaction Track." *Proceedings of BioCreative VI Workshop*.
- **BioCreative**: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/
- **HuggingFace**: https://huggingface.co/datasets/bigbio/chemprot

## License

The ChemProt corpus is available for research purposes. Original distribution requires BioCreative registration. The HuggingFace mirror (bigbio/chemprot) provides programmatic access.

## Download

```bash
# Via HuggingFace (requires `datasets` package)
python -m benchmarks.datasets.loader --download --dataset chemprot

# Manual: download from BioCreative and place TSV files under
# benchmarks/datasets/data/chemprot/chemprot_training/
# benchmarks/datasets/data/chemprot/chemprot_development/
# benchmarks/datasets/data/chemprot/chemprot_test_gs/
```

## Usage

```python
from benchmarks.datasets import chemprot

dataset = chemprot.load("benchmarks/datasets/data/chemprot")
print(dataset.summary())

for ex in dataset.test.examples[:3]:
    print(f"PMID {ex.doc_id}: {len(ex.entities)} entities, {len(ex.relations)} relations")
```

## Relevance to Graffold

ChemProt evaluates Graffold's ability to extract directional chemical-protein interactions — a superset of the protein-disease associations that are Graffold's primary focus. The CPR relation types (upregulator, inhibitor, agonist, etc.) map to Graffold's relationship extraction with typed edges.
