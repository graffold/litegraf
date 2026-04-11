"""Base class for pipeline processors.

Defines the ProcessorBase ABC that all pipeline processors implement,
enabling a discoverable plugin architecture.
"""

from abc import ABC, abstractmethod
from typing import Any

from pipeline.interfaces import GraphStore


class ProcessorBase(ABC):
    """Base class for all pipeline processors.

    Every processor receives a GraphStore via its constructor and implements
    a standard ``process`` method plus a ``name`` property for discovery.
    """

    @abstractmethod
    def __init__(self, graph_store: GraphStore, **kwargs: Any) -> None:
        """Initialise the processor with a graph store backend.

        Args:
            graph_store: The graph database backend to use.
            **kwargs: Additional processor-specific configuration.
        """

    @abstractmethod
    async def process(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        """Run the processor on the given data.

        Args:
            data: Input data to process.
            **kwargs: Additional processing options.

        Returns:
            A dictionary containing the processing results.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable processor name used for discovery."""
