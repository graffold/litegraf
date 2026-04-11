"""Context graph interface stubs for standalone pipeline usage."""
from abc import ABC, abstractmethod
from typing import Any


class ContextGraphManager(ABC):
    @abstractmethod
    def store_context(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def store_node(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def store_relationship(self, *args, **kwargs) -> Any:
        ...


class ProvenanceFactory(ABC):
    @abstractmethod
    def create_provenance(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def build_from_kg_pipeline(self, *args, **kwargs) -> Any:
        ...
