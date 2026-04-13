"""Shared data models for benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Entity:
    """A named entity mention in text."""

    text: str
    entity_type: str  # e.g. "Chemical", "Disease", "Gene", "Protein"
    start: int = -1
    end: int = -1
    mesh_id: str = ""


@dataclass
class Relation:
    """A relation between two entities."""

    head: str
    tail: str
    relation_type: str  # e.g. "CID", "CPR:4", "positive", "negative"
    head_type: str = ""
    tail_type: str = ""


@dataclass
class BenchmarkExample:
    """A single example from a benchmark dataset."""

    doc_id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    label: str = ""  # For binary classification datasets (GAD)


@dataclass
class BenchmarkSplit:
    """A dataset split (train/dev/test)."""

    name: str
    examples: list[BenchmarkExample]

    @property
    def size(self) -> int:
        return len(self.examples)


@dataclass
class BenchmarkDataset:
    """A complete benchmark dataset with splits."""

    name: str
    task: str  # "ner", "re", "binary_re"
    entity_types: list[str]
    relation_types: list[str]
    splits: dict[str, BenchmarkSplit]

    @property
    def train(self) -> BenchmarkSplit | None:
        return self.splits.get("train")

    @property
    def dev(self) -> BenchmarkSplit | None:
        return self.splits.get("dev")

    @property
    def test(self) -> BenchmarkSplit | None:
        return self.splits.get("test")

    def summary(self) -> dict[str, int]:
        return {name: split.size for name, split in self.splits.items()}


class DatasetLoader(Protocol):
    """Protocol for dataset loaders."""

    def load(self, data_dir: str) -> BenchmarkDataset: ...

    def download(self, target_dir: str) -> None: ...
