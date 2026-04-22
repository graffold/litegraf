"""Entity schema base model for domain-specific extraction configuration.

Provides the EntitySchema Pydantic v2 base model that domain-specific schemas
extend, and the ValidationResult model for extraction validation output.
"""

from typing import Any

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of validating extracted entities and relationships."""

    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]  # each has "item" and "reason" keys


class EntitySchema(BaseModel):
    """Base model for domain-specific extraction configuration."""

    domain_name: str
    entity_types: list[str]
    relationship_types: list[str]
    extraction_prompt: str
    entity_properties: dict[str, list[str]] = Field(default_factory=dict)
    post_processing_hooks: list[Any] = Field(default_factory=list)

    def validate_extraction(
        self,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> ValidationResult:
        """Partition entities and relationships into accepted/rejected.

        Checks each entity's ``type`` against ``entity_types`` and each
        relationship's ``type`` against ``relationship_types``. Items with
        unknown types go into ``rejected`` with a reason string.
        """
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for entity in entities:
            if entity.get("type") in self.entity_types:
                accepted.append(entity)
            else:
                rejected.append(
                    {
                        "item": entity,
                        "reason": (
                            f"Entity type '{entity.get('type')}' not in"
                            f" allowed types: {self.entity_types}"
                        ),
                    }
                )

        for relationship in relationships:
            if relationship.get("type") in self.relationship_types:
                accepted.append(relationship)
            else:
                rejected.append(
                    {
                        "item": relationship,
                        "reason": (
                            f"Relationship type"
                            f" '{relationship.get('type')}' not in"
                            f" allowed types:"
                            f" {self.relationship_types}"
                        ),
                    }
                )

        return ValidationResult(accepted=accepted, rejected=rejected)
