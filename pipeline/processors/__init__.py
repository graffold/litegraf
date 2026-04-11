"""Processors package for data processing and enrichment components."""

import importlib
import logging
import pkgutil

from .base import ProcessorBase
from .disease_hierarchy_enricher import DiseaseHierarchyEnricher
from .entity_resolver import EntityResolver
from .ontology_filter import OntologyFilter
from .relationship_counter import RelationshipCounter

logger = logging.getLogger(__name__)


def discover_processors() -> dict[str, type[ProcessorBase]]:
    """Scan pipeline.processors for ProcessorBase subclasses.

    Uses ``pkgutil.iter_modules`` to find all modules in the package,
    imports each one, and collects classes that inherit from
    :class:`ProcessorBase` (excluding ``ProcessorBase`` itself).

    Returns:
        A dictionary mapping each processor's ``name`` property to its class.
    """
    processors: dict[str, type[ProcessorBase]] = {}
    package = importlib.import_module("pipeline.processors")
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"pipeline.processors.{module_name}")
        except Exception:
            logger.warning("Failed to import processor module %s", module_name, exc_info=True)
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, ProcessorBase)
                and attr is not ProcessorBase
            ):
                # Access the name via the property descriptor on the class.
                # ``name`` is declared as an abstract property, so we read
                # through the descriptor's ``fget`` to avoid instantiation.
                try:
                    prop = getattr(attr, "name")
                    if isinstance(prop, property) and prop.fget is not None:
                        name_value: str = prop.fget(attr)
                    else:
                        # Fallback: use class name as key
                        name_value = attr.__name__
                    processors[name_value] = attr
                except Exception:
                    logger.warning(
                        "Processor class %s.%s has no usable 'name' property, skipping",
                        module_name,
                        attr_name,
                        exc_info=True,
                    )
    return processors


__all__ = [
    "DiseaseHierarchyEnricher",
    "EntityResolver",
    "OntologyFilter",
    "ProcessorBase",
    "RelationshipCounter",
    "discover_processors",
]
