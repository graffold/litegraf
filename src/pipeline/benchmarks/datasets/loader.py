"""Unified entry point for downloading and loading benchmark datasets.

Usage:
    # Download all datasets
    python -m benchmarks.datasets.loader --download

    # Download specific dataset
    python -m benchmarks.datasets.loader --download --dataset bc5cdr

    # List available datasets
    python -m benchmarks.datasets.loader --list
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import ModuleType
from typing import TypedDict

from . import bc5cdr, chemprot, gad
from .models import BenchmarkDataset

logger = logging.getLogger(__name__)

_DATASETS_ROOT = Path(__file__).parent.parent.parent / "benchmarks" / "datasets"


class _DatasetEntry(TypedDict):
    module: ModuleType
    dir: str
    description: str


REGISTRY: dict[str, _DatasetEntry] = {
    "bc5cdr": {
        "module": bc5cdr,
        "dir": "bc5cdr",
        "description": "BioCreative V CDR — Chemical-Disease NER + Relation Extraction",
    },
    "chemprot": {
        "module": chemprot,
        "dir": "chemprot",
        "description": "BioCreative VI Track 5 — Chemical-Protein Interaction Extraction",
    },
    "gad": {
        "module": gad,
        "dir": "gad",
        "description": "Gene-Disease Association — Binary Relation Classification",
    },
}


def get_data_dir(dataset_name: str, base_dir: str | None = None) -> Path:
    """Return the data directory for a dataset."""
    root = Path(base_dir) if base_dir else _DATASETS_ROOT
    return root / "data" / REGISTRY[dataset_name]["dir"]


def download_dataset(name: str, base_dir: str | None = None) -> None:
    """Download a single dataset."""
    if name not in REGISTRY:
        msg = f"Unknown dataset: {name}. Available: {list(REGISTRY.keys())}"
        raise ValueError(msg)
    data_dir = get_data_dir(name, base_dir)
    REGISTRY[name]["module"].download(str(data_dir))


def download_all(base_dir: str | None = None) -> None:
    """Download all benchmark datasets."""
    for name in REGISTRY:
        logger.info("Downloading %s...", name)
        download_dataset(name, base_dir)


def load_dataset(name: str, base_dir: str | None = None) -> BenchmarkDataset:
    """Load a dataset by name."""
    if name not in REGISTRY:
        msg = f"Unknown dataset: {name}. Available: {list(REGISTRY.keys())}"
        raise ValueError(msg)
    data_dir = get_data_dir(name, base_dir)
    result: BenchmarkDataset = REGISTRY[name]["module"].load(str(data_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dataset manager")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--dataset", choices=list(REGISTRY.keys()), help="Specific dataset (default: all)")
    parser.add_argument("--base-dir", help="Override base directory for data storage")
    parser.add_argument("--list", action="store_true", dest="list_datasets", help="List available datasets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list_datasets:
        for name, info in REGISTRY.items():
            print(f"  {name:12s}  {info['description']}")
        return

    if args.download:
        if args.dataset:
            download_dataset(args.dataset, args.base_dir)
        else:
            download_all(args.base_dir)
        return

    # Default: load and print summary
    for name in REGISTRY:
        if args.dataset and args.dataset != name:
            continue
        try:
            ds = load_dataset(name, args.base_dir)
            print(f"{ds.name}: {ds.summary()}")
        except Exception:
            logger.exception("Failed to load %s", name)


if __name__ == "__main__":
    main()
