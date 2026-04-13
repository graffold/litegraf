"""GAD (Gene-Disease Associations) dataset loader.

Binary relation classification: given a sentence with a gene and disease mention,
predict whether a true gene-disease association is described (positive/negative).

The GAD corpus contains ~30K PubMed title/abstract sentences annotated via distant
supervision from the Genetic Association Database. The BioBERT distribution provides
10-fold cross-validation splits.

Source: https://github.com/dmis-lab/biobert (RE datasets)
Download: http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/REdata.zip
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

from .models import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkSplit,
)

logger = logging.getLogger(__name__)

_ZIP_URL = "http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/REdata.zip"


def download(target_dir: str) -> None:
    """Download GAD from BioBERT RE datasets zip."""
    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check if already extracted (fold 1 as sentinel)
    if (out / "1" / "train.tsv").exists():
        logger.info("GAD already downloaded: %s", out)
        return

    zip_path = out / "REdata.zip"
    logger.info("Downloading GAD (BioBERT RE datasets) from %s", _ZIP_URL)
    urllib.request.urlretrieve(_ZIP_URL, zip_path)  # noqa: S310

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Extract only GAD files: GAD/1/train.tsv etc.
            if member.startswith("GAD/") and not member.endswith("/"):
                # Strip the GAD/ prefix
                rel = member[len("GAD/"):]
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))

    zip_path.unlink(missing_ok=True)
    logger.info("GAD download complete → %s", out)


def _parse_tsv(path: Path) -> list[BenchmarkExample]:
    """Parse GAD TSV: index \\t sentence \\t label."""
    examples: list[BenchmarkExample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if i == 0 and line.startswith("index"):
            continue  # skip header
        parts = line.split("\t")
        if len(parts) >= 3:
            idx, sentence, label = parts[0], parts[1], parts[2]
            examples.append(
                BenchmarkExample(
                    doc_id=str(idx),
                    text=sentence,
                    label="positive" if label.strip() == "1" else "negative",
                )
            )
        elif len(parts) == 2:
            examples.append(
                BenchmarkExample(
                    doc_id=str(i),
                    text=parts[0],
                    label="positive" if parts[1].strip() == "1" else "negative",
                )
            )
    return examples


def load(data_dir: str, fold: int = 1) -> BenchmarkDataset:
    """Load GAD from data_dir. Uses 10-fold CV; default fold=1."""
    root = Path(data_dir)
    splits: dict[str, BenchmarkSplit] = {}

    fold_dir = root / str(fold)
    if not fold_dir.exists():
        logger.warning("GAD fold %d not found at %s — run download() first", fold, fold_dir)
        return BenchmarkDataset(
            name="GAD",
            task="binary_re",
            entity_types=["Gene", "Disease"],
            relation_types=["positive", "negative"],
            splits={},
        )

    for split_name in ("train", "dev", "test"):
        path = fold_dir / f"{split_name}.tsv"
        if not path.exists():
            continue
        examples = _parse_tsv(path)
        splits[split_name] = BenchmarkSplit(name=split_name, examples=examples)
        logger.info("GAD fold %d %s: %d examples", fold, split_name, len(examples))

    return BenchmarkDataset(
        name="GAD",
        task="binary_re",
        entity_types=["Gene", "Disease"],
        relation_types=["positive", "negative"],
        splits=splits,
    )
