# BC5CDR — BioCreative V Chemical Disease Relation Corpus

## Overview

The BC5CDR corpus consists of 1,500 PubMed articles annotated with:
- **4,409 chemicals** and **5,818 diseases** (named entity recognition)
- **3,116 chemical-induced-disease (CID) relations** (relation extraction)

Entity annotations include mention text spans and normalized MeSH concept identifiers.

## Task

- **NER**: Identify chemical and disease entity mentions in biomedical text
- **RE**: Extract chemical-induced-disease (CID) relationships between entities

## Splits

| Split | Documents |
|-------|-----------|
| Train | 500 |
| Dev | 500 |
| Test | 500 |

## Format

PubTator format with three line types per document:
```
PMID|t|Title text
PMID|a|Abstract text
PMID  start  end  mention_text  entity_type  MeSH_ID
PMID  CID  Chemical_MeSH  Disease_MeSH
```

## Source & Citation

- **Paper**: Li et al. (2016). "BioCreative V CDR task corpus: a resource for chemical disease relation extraction." *Database*, baw068.
- **DOI**: https://doi.org/10.1093/database/baw068
- **PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4860626/
- **Data**: https://github.com/JHnlp/BioCreative-V-CDR-Corpus

## License

Public domain (NIH-funded research corpus). The corpus is freely available for research use.

## Download

```bash
python -m benchmarks.datasets.loader --download --dataset bc5cdr
```

## Usage

```python
from benchmarks.datasets import bc5cdr

# Download
bc5cdr.download("benchmarks/datasets/data/bc5cdr")

# Load
dataset = bc5cdr.load("benchmarks/datasets/data/bc5cdr")
print(dataset.summary())  # {'train': 500, 'dev': 500, 'test': 500}

# Access examples
for ex in dataset.test.examples[:3]:
    print(f"PMID {ex.doc_id}: {len(ex.entities)} entities, {len(ex.relations)} relations")
```

## Relevance to Graffold

BC5CDR directly evaluates Graffold's core capability: extracting chemical-disease relationships from PubMed abstracts. The CID relation task maps to Graffold's `ASSOCIATES_WITH` relationship extraction between chemicals/drugs and diseases.
