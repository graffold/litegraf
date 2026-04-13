"""ChemProt dataset loader.

BioCreative VI Track 5 — Chemical-Protein Interaction extraction.
1820 PubMed abstracts with chemical-protein relation annotations.
Format: TSV files (abstracts, entities, relations, gold standard).

The original data requires BioCreative registration. This loader supports
the widely-used re-distribution via the bigbio/chemprot HuggingFace dataset
or manual download from BioCreative.

For manual setup, place these files per split directory:
  chemprot_training/  chemprot_development/  chemprot_test_gs/
    - chemprot_{split}_abstracts.tsv
    - chemprot_{split}_entities.tsv
    - chemprot_{split}_relations.tsv  (train/dev only)
    - chemprot_{split}_gold_standard.tsv

Source: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/
HuggingFace mirror: https://huggingface.co/datasets/bigbio/chemprot
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from .models import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkSplit,
    Entity,
    Relation,
)

logger = logging.getLogger(__name__)

# CPR groups used in evaluation (5 positive classes)
CPR_GROUPS = {
    "CPR:3": "UPREGULATOR|ACTIVATOR|INDIRECT_UPREGULATOR",
    "CPR:4": "DOWNREGULATOR|INHIBITOR|INDIRECT_DOWNREGULATOR",
    "CPR:5": "AGONIST|AGONIST-ACTIVATOR|AGONIST-INHIBITOR",
    "CPR:6": "ANTAGONIST",
    "CPR:9": "SUBSTRATE|PRODUCT_OF|SUBSTRATE_PRODUCT_OF",
}

_SPLIT_DIRS = {
    "train": "chemprot_training",
    "dev": "chemprot_development",
    "test": "chemprot_test_gs",
}


def download(target_dir: str) -> None:
    """Download ChemProt via HuggingFace datasets (requires `datasets` package).

    Falls back to instructions if the package is unavailable.
    """
    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        ds = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb", trust_remote_code=True)
        # Write a marker so load() knows HF format is available
        (out / ".hf_downloaded").touch()
        # Save to disk for offline use
        ds.save_to_disk(str(out / "hf_cache"))
        logger.info("ChemProt downloaded via HuggingFace → %s", out)
    except ImportError:
        msg = (
            "ChemProt requires manual download from BioCreative:\n"
            "  https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/\n"
            "Or install `datasets`: pip install datasets\n"
            f"Place files under: {out}"
        )
        logger.warning(msg)
        (out / "DOWNLOAD_INSTRUCTIONS.txt").write_text(msg)


def _load_tsv_split(split_dir: Path, split_name: str) -> list[BenchmarkExample]:
    """Load a ChemProt split from TSV files."""
    prefix_map = {"train": "training", "dev": "development", "test": "test_gs"}
    prefix = prefix_map.get(split_name, split_name)

    abstracts_file = split_dir / f"chemprot_{prefix}_abstracts.tsv"
    entities_file = split_dir / f"chemprot_{prefix}_entities.tsv"
    gold_file = split_dir / f"chemprot_{prefix}_gold_standard.tsv"

    if not abstracts_file.exists():
        return []

    # Load abstracts: PMID \t title \t abstract
    docs: dict[str, str] = {}
    for line in abstracts_file.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 3:
            docs[parts[0]] = f"{parts[1]} {parts[2]}"
        elif len(parts) == 2:
            docs[parts[0]] = parts[1]

    # Load entities: PMID \t entity_id \t type \t start \t end \t text
    doc_entities: dict[str, list[Entity]] = defaultdict(list)
    entity_ids: dict[str, dict[str, Entity]] = defaultdict(dict)
    if entities_file.exists():
        for line in entities_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 6:
                pmid, eid, etype = parts[0], parts[1], parts[2]
                ent = Entity(
                    text=parts[5],
                    entity_type=etype,
                    start=int(parts[3]),
                    end=int(parts[4]),
                )
                doc_entities[pmid].append(ent)
                entity_ids[pmid][eid] = ent

    # Load gold standard relations: PMID \t CPR_group \t eval \t relation \t arg1 \t arg2
    doc_relations: dict[str, list[Relation]] = defaultdict(list)
    if gold_file.exists():
        for line in gold_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 4:
                pmid, cpr_group = parts[0], parts[1]
                # arg1/arg2 format: "Arg1:T5"
                arg1_id = parts[4].split(":")[-1] if len(parts) > 4 else ""
                arg2_id = parts[5].split(":")[-1] if len(parts) > 5 else ""
                head_ent = entity_ids.get(pmid, {}).get(arg1_id)
                tail_ent = entity_ids.get(pmid, {}).get(arg2_id)
                doc_relations[pmid].append(
                    Relation(
                        head=head_ent.text if head_ent else arg1_id,
                        tail=tail_ent.text if tail_ent else arg2_id,
                        relation_type=cpr_group,
                        head_type=head_ent.entity_type if head_ent else "CHEMICAL",
                        tail_type=tail_ent.entity_type if tail_ent else "GENE",
                    )
                )

    examples = []
    for pmid, text in docs.items():
        examples.append(
            BenchmarkExample(
                doc_id=pmid,
                text=text,
                entities=doc_entities.get(pmid, []),
                relations=doc_relations.get(pmid, []),
            )
        )
    return examples


def _load_hf_split(cache_dir: Path) -> dict[str, list[BenchmarkExample]]:
    """Load from HuggingFace cached dataset."""
    from datasets import load_from_disk

    ds = load_from_disk(str(cache_dir))
    splits: dict[str, list[BenchmarkExample]] = {}
    split_map = {"train": "train", "validation": "dev", "test": "test"}

    for hf_name, our_name in split_map.items():
        if hf_name not in ds:
            continue
        examples = []
        for row in ds[hf_name]:
            entities = []
            for ent in row.get("entities", []):
                for i, text in enumerate(ent.get("text", [])):
                    offsets = ent.get("offsets", [[]])[i] if i < len(ent.get("offsets", [])) else [-1, -1]
                    entities.append(
                        Entity(
                            text=text,
                            entity_type=ent.get("type", ""),
                            start=offsets[0] if offsets else -1,
                            end=offsets[1] if len(offsets) > 1 else -1,
                        )
                    )
            relations = []
            for rel in row.get("relations", []):
                relations.append(
                    Relation(
                        head=rel.get("arg1_id", ""),
                        tail=rel.get("arg2_id", ""),
                        relation_type=rel.get("type", ""),
                    )
                )
            passages = row.get("passages", [])
            text = " ".join(p.get("text", [""])[0] for p in passages)
            examples.append(
                BenchmarkExample(
                    doc_id=row.get("document_id", row.get("id", "")),
                    text=text,
                    entities=entities,
                    relations=relations,
                )
            )
        splits[our_name] = examples
    return splits


def load(data_dir: str) -> BenchmarkDataset:
    """Load ChemProt from data_dir (TSV or HuggingFace cache)."""
    root = Path(data_dir)
    splits: dict[str, BenchmarkSplit] = {}

    # Try HuggingFace cache first
    hf_cache = root / "hf_cache"
    if (root / ".hf_downloaded").exists() and hf_cache.exists():
        hf_splits = _load_hf_split(hf_cache)
        for name, examples in hf_splits.items():
            splits[name] = BenchmarkSplit(name=name, examples=examples)
            logger.info("ChemProt %s (HF): %d documents", name, len(examples))
    else:
        # TSV format
        for split_name, dirname in _SPLIT_DIRS.items():
            split_dir = root / dirname
            if not split_dir.exists():
                logger.warning("ChemProt %s dir not found: %s", split_name, split_dir)
                continue
            examples = _load_tsv_split(split_dir, split_name)
            splits[split_name] = BenchmarkSplit(name=split_name, examples=examples)
            logger.info("ChemProt %s: %d documents", split_name, len(examples))

    return BenchmarkDataset(
        name="ChemProt",
        task="re",
        entity_types=["CHEMICAL", "GENE"],
        relation_types=list(CPR_GROUPS.keys()),
        splits=splits,
    )
