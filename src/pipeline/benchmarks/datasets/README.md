# Benchmark Datasets

Established biomedical NLP benchmark datasets for evaluating Graffold's entity extraction and relation extraction capabilities.

## Available Datasets

| Dataset | Task | Entities | Relations | Source |
|---------|------|----------|-----------|--------|
| [BC5CDR](bc5cdr/) | NER + RE | Chemical, Disease | CID (chemical-induced-disease) | BioCreative V |
| [ChemProt](chemprot/) | RE | Chemical, Gene/Protein | CPR:3-9 (5 interaction types) | BioCreative VI |
| [GAD](gad/) | Binary RE | Gene, Disease | positive/negative association | BioBERT / Genetic Association DB |

## Quick Start

```bash
# Download all datasets
python -m benchmarks.datasets.loader --download

# Download a specific dataset
python -m benchmarks.datasets.loader --download --dataset bc5cdr

# List available datasets
python -m benchmarks.datasets.loader --list
```

## Python API

```python
from benchmarks.datasets.loader import load_dataset, download_dataset

# Download and load
download_dataset("bc5cdr")
ds = load_dataset("bc5cdr")
print(ds.summary())  # {'train': 500, 'dev': 500, 'test': 500}

# Access splits
for example in ds.test.examples[:5]:
    print(f"{example.doc_id}: {len(example.entities)} entities")
```

## Data Storage

Downloaded data is stored under `benchmarks/datasets/data/`:
```
benchmarks/datasets/data/
├── bc5cdr/          # PubTator files
├── chemprot/        # TSV files or HuggingFace cache
└── gad/             # TSV files
```

This directory is gitignored — datasets are downloaded on demand.

## Adding New Datasets

1. Create a module in `benchmarks/datasets/` with `download()` and `load()` functions
2. Use the shared `BenchmarkDataset` / `BenchmarkExample` models from `models.py`
3. Register in `loader.py` REGISTRY
4. Add a README in `benchmarks/datasets/<name>/README.md`
