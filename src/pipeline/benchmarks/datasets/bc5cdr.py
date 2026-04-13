"""BC5CDR dataset loader.

BioCreative V Chemical Disease Relation corpus.
1500 PubMed articles with chemical/disease NER and chemical-induced-disease (CID) relations.
Format: PubTator (title|abstract lines, entity annotations, relation annotations).

Source: https://github.com/JHnlp/BioCreative-V-CDR-Corpus
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
    Entity,
    Relation,
)

logger = logging.getLogger(__name__)

_ZIP_URL = "https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip"

_FILES = {
    "train": "CDR_TrainingSet.PubTator.txt",
    "dev": "CDR_DevelopmentSet.PubTator.txt",
    "test": "CDR_TestSet.PubTator.txt",
}


def download(target_dir: str) -> None:
    """Download and extract BC5CDR PubTator files."""
    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if all((out / f).exists() for f in _FILES.values()):
        logger.info("BC5CDR already downloaded: %s", out)
        return

    zip_path = out / "CDR_Data.zip"
    logger.info("Downloading BC5CDR from %s", _ZIP_URL)
    urllib.request.urlretrieve(_ZIP_URL, zip_path)  # noqa: S310

    # Extract PubTator files from the zip
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            basename = Path(member).name
            if basename in _FILES.values():
                # Extract to target dir with flat name
                data = zf.read(member)
                (out / basename).write_bytes(data)
                logger.info("Extracted %s", basename)

    zip_path.unlink(missing_ok=True)
    logger.info("BC5CDR download complete → %s", out)


def _parse_pubtator(path: Path) -> list[BenchmarkExample]:
    """Parse a PubTator-format file into BenchmarkExamples."""
    examples: list[BenchmarkExample] = []
    current_id = ""
    title = ""
    abstract = ""
    entities: list[Entity] = []
    relations: list[Relation] = []

    def _flush() -> None:
        nonlocal current_id, title, abstract, entities, relations
        if current_id:
            text = f"{title} {abstract}".strip()
            examples.append(
                BenchmarkExample(
                    doc_id=current_id,
                    text=text,
                    entities=list(entities),
                    relations=list(relations),
                )
            )
        current_id = ""
        title = ""
        abstract = ""
        entities = []
        relations = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            _flush()
            continue

        # Title line: PMID|t|Title text
        if "|t|" in line:
            parts = line.split("|t|", 1)
            current_id = parts[0]
            title = parts[1]
            continue

        # Abstract line: PMID|a|Abstract text
        if "|a|" in line:
            abstract = line.split("|a|", 1)[1]
            continue

        cols = line.split("\t")

        # Relation line: PMID \t CID \t Chemical_MeSH \t Disease_MeSH
        if len(cols) == 4 and cols[1] == "CID":
            relations.append(
                Relation(
                    head=cols[2],
                    tail=cols[3],
                    relation_type="CID",
                    head_type="Chemical",
                    tail_type="Disease",
                )
            )
            continue

        # Entity line: PMID \t start \t end \t text \t type \t MeSH_ID
        if len(cols) >= 6:
            mesh_id = cols[5] if len(cols) > 5 else ""
            ent = Entity(
                text=cols[3],
                entity_type=cols[4],
                start=int(cols[1]),
                end=int(cols[2]),
                mesh_id=mesh_id,
            )
            entities.append(ent)

    _flush()  # last document
    return examples


def load(data_dir: str) -> BenchmarkDataset:
    """Load BC5CDR from PubTator files in data_dir."""
    root = Path(data_dir)
    splits: dict[str, BenchmarkSplit] = {}
    for split_name, filename in _FILES.items():
        path = root / filename
        if not path.exists():
            logger.warning("BC5CDR %s not found at %s — run download() first", split_name, path)
            continue
        examples = _parse_pubtator(path)
        splits[split_name] = BenchmarkSplit(name=split_name, examples=examples)
        logger.info("BC5CDR %s: %d documents", split_name, len(examples))

    return BenchmarkDataset(
        name="BC5CDR",
        task="ner+re",
        entity_types=["Chemical", "Disease"],
        relation_types=["CID"],
        splits=splits,
    )
