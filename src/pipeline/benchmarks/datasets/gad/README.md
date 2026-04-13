# GAD — Gene-Disease Associations Corpus

## Overview

The GAD corpus contains ~30,000 PubMed title/abstract sentences automatically labeled for gene-disease associations via distant supervision from the Genetic Association Database. It is a standard benchmark for binary biomedical relation classification.

## Task

**Binary Relation Classification**: Given a sentence containing a gene and a disease mention, predict whether the sentence describes a true gene-disease association (positive) or not (negative).

## Splits

| Split | Examples | Positive | Negative |
|-------|----------|----------|----------|
| Train | ~27,000 | ~50% | ~50% |
| Test | ~3,000 | ~50% | ~50% |

(Exact counts depend on the version; the BioBERT distribution is the standard benchmark split.)

## Format

TSV with columns:
```
index \t sentence \t label
```
Where label is `1` (positive association) or `0` (negative/no association).

## Source & Citation

- **Original Database**: Becker et al. (2004). "The Genetic Association Database." *Nature Genetics*, 36(5), 431-432.
- **NLP Benchmark**: Lee et al. (2020). "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." *Bioinformatics*, 36(4), 1234-1240.
- **Data (BioBERT)**: https://github.com/dmis-lab/biobert
- **HuggingFace**: https://huggingface.co/datasets/bigbio/gad

## License

The GAD corpus as distributed via BioBERT is available for research use. The original Genetic Association Database was a public NIH-funded resource.

## Download

```bash
python -m benchmarks.datasets.loader --download --dataset gad
```

## Usage

```python
from benchmarks.datasets import gad

# Download
gad.download("benchmarks/datasets/data/gad")

# Load
dataset = gad.load("benchmarks/datasets/data/gad")
print(dataset.summary())  # {'train': ~27000, 'test': ~3000}

# Access examples
for ex in dataset.test.examples[:3]:
    print(f"[{ex.label}] {ex.text[:80]}...")
```

## Relevance to Graffold

GAD directly evaluates Graffold's core mission: identifying gene/protein-disease associations from biomedical text. The binary classification task measures whether the system can distinguish true associations from co-occurrences — critical for knowledge graph quality.
