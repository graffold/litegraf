"""Multi-pass entity extraction (gleaning) for improved knowledge graph completeness.

Performs iterative LLM passes over text to capture entities missed in the first pass.
Each subsequent pass prompts the LLM with already-found entities and asks for missed ones.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Any

from pipeline.config import PipelineConfig as Config
logger = logging.getLogger(__name__)
@dataclass
class GleaningResult:
    """Result of a multi-pass gleaning extraction."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    passes_performed: int = 0
    entities_per_pass: list[int] = field(default_factory=list)


class GleaningExtractor:
    """Multi-pass LLM entity extractor that iteratively finds missed entities.

    Uses an LLM with an `invoke(prompt: str) -> str` interface. Each pass after
    the first builds a gleaning prompt listing already-found entities and asking
    the LLM to identify any that were missed.

    Stops when:
    - A pass returns zero new entities, OR
    - max_passes is reached
    """

    def __init__(self, llm: Any, max_passes: int = 3) -> None:
        """Initialize the gleaning extractor.

        Args:
            llm: Any object with an `invoke(prompt: str) -> str` method.
                 The response is expected to be JSON with "entities" and
                 "relationships" arrays.
            max_passes: Maximum number of extraction passes (default 3, minimum 2).
        """
        self.llm = llm
        self.max_passes = max(max_passes, 2)

    async def extract(
        self, text: str, existing_entities: list[dict[str, Any]] | None = None
    ) -> GleaningResult:
        """Perform multi-pass entity extraction over the given text.

        Args:
            text: The text to extract entities and relationships from.
            existing_entities: Optional list of already-known entities to seed
                               the extraction context.

        Returns:
            GleaningResult with deduplicated entities, relationships, and metadata.
        """
        if not text or not text.strip():
            return GleaningResult(passes_performed=0)

        all_entities: list[dict[str, Any]] = list(existing_entities or [])
        all_relationships: list[dict[str, Any]] = []
        entities_per_pass: list[int] = []

        for pass_num in range(1, self.max_passes + 1):
            try:
                if pass_num == 1:
                    prompt = self._build_initial_prompt(text)
                else:
                    prompt = self._build_gleaning_prompt(text, all_entities)

                response_text = self.llm.invoke(prompt)
                entities, relationships = self._parse_response(response_text)

            except TimeoutError:
                logger.warning(
                    f"LLM timeout on gleaning pass {pass_num}, "
                    f"returning {len(all_entities)} entities collected so far"
                )
                break
            except Exception as e:
                logger.warning(
                    f"LLM error on gleaning pass {pass_num}: {e}, "
                    f"returning {len(all_entities)} entities collected so far"
                )
                break

            # Count genuinely new entities (not already in all_entities)
            new_entities = self._find_new_entities(entities, all_entities)
            entities_per_pass.append(len(new_entities))

            all_entities.extend(new_entities)
            all_relationships.extend(relationships)

            logger.info(
                f"Gleaning pass {pass_num}: found {len(new_entities)} new entities"
            )

            # Early termination: no new entities found
            if len(new_entities) == 0:
                break

        # Deduplicate and return
        deduped_entities = self._merge_entities(all_entities)
        deduped_relationships = self._merge_relationships(all_relationships)

        return GleaningResult(
            entities=deduped_entities,
            relationships=deduped_relationships,
            passes_performed=len(entities_per_pass),
            entities_per_pass=entities_per_pass,
        )

    def _build_initial_prompt(self, text: str) -> str:
        """Build the initial extraction prompt for pass 1."""
        return (
            "Extract all entities and relationships from the following text. "
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "entity_name", "type": "entity_type"}\n'
            "  ],\n"
            '  "relationships": [\n'
            '    {"source": "source_name", "target": "target_name", '
            '"type": "relationship_type", '
            '"source_sentence": "the exact sentence from the text"}\n'
            "  ]\n"
            "}\n\n"
            "Entity types should be: Protein, Disease, or Entity.\n"
            "Relationship types should be: ASSOCIATED_WITH, CAUSES, TREATS, "
            "or RELATED_TO.\n"
            'For each relationship, "source_sentence" MUST be the verbatim '
            "sentence from the input text that supports the extracted triple.\n\n"
            f"Text: {text}\n\n"
            "Return only valid JSON:"
        )

    def _build_gleaning_prompt(
        self, text: str, found_entities: list[dict[str, Any]]
    ) -> str:
        """Build a gleaning prompt that lists found entities and asks for missed ones.

        Args:
            text: The original text being analyzed.
            found_entities: Entities found in previous passes.

        Returns:
            A prompt string for the LLM.
        """
        entity_names = [e.get("name", "unknown") for e in found_entities]
        entity_list = ", ".join(entity_names) if entity_names else "(none)"

        return (
            "The following entities have already been extracted from the text below:\n"
            f"Already found: {entity_list}\n\n"
            "Please carefully re-read the text and identify any entities or "
            "relationships that were MISSED in the previous extraction. "
            "Only return NEW entities and relationships not already listed above.\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "entity_name", "type": "entity_type"}\n'
            "  ],\n"
            '  "relationships": [\n'
            '    {"source": "source_name", "target": "target_name", '
            '"type": "relationship_type", '
            '"source_sentence": "the exact sentence from the text"}\n'
            "  ]\n"
            "}\n\n"
            'For each relationship, "source_sentence" MUST be the verbatim '
            "sentence from the input text that supports the extracted triple.\n"
            "If no new entities or relationships are found, return:\n"
            '{"entities": [], "relationships": []}\n\n'
            f"Text: {text}\n\n"
            "Return only valid JSON:"
        )

    def _parse_response(
        self, response_text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse LLM response JSON into entities and relationships.

        Handles markdown code blocks and extracts JSON robustly.
        On malformed JSON, logs a warning and returns empty lists.

        Args:
            response_text: Raw LLM response string.

        Returns:
            Tuple of (entities, relationships).
        """
        import re

        if not response_text or not response_text.strip():
            return [], []

        json_str = response_text.strip()

        # Strip markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            parts = json_str.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") and (
                    "entities" in part or "relationships" in part
                ):
                    json_str = part
                    break

        # Try regex extraction if not cleanly bounded
        if not (json_str.startswith("{") and json_str.endswith("}")):
            match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if match:
                json_str = match.group()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed LLM response, skipping pass: {e}")
            return [], []

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Validate entity structure
        valid_entities = []
        for ent in entities:
            if isinstance(ent, dict) and "name" in ent:
                valid_entities.append(
                    {
                        "name": str(ent["name"]),
                        "type": str(ent.get("type", "Entity")),
                    }
                )

        valid_relationships = []
        for rel in relationships:
            if isinstance(rel, dict) and "source" in rel and "target" in rel:
                validated = {
                    "source": str(rel["source"]),
                    "target": str(rel["target"]),
                    "type": str(rel.get("type", "RELATED_TO")),
                }
                if rel.get("source_sentence") and Config.ENABLE_SENTENCE_PROVENANCE:
                    validated["source_sentence"] = str(rel["source_sentence"])
                valid_relationships.append(validated)

        return valid_entities, valid_relationships

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize an entity name for deduplication (lowercase, stripped)."""
        return name.strip().lower()

    def _find_new_entities(
        self,
        candidates: list[dict[str, Any]],
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only entities from candidates not already in existing (by normalized name).

        Args:
            candidates: Entities from the current pass.
            existing: Entities accumulated from previous passes.

        Returns:
            List of genuinely new entities.
        """
        existing_names = {self._normalize_name(e.get("name", "")) for e in existing}
        new = []
        for ent in candidates:
            norm = self._normalize_name(ent.get("name", ""))
            if norm and norm not in existing_names:
                new.append(ent)
                existing_names.add(norm)  # prevent duplicates within candidates
        return new

    def _merge_entities(
        self, all_entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate entities by normalized name, keeping the first occurrence.

        Args:
            all_entities: All entities collected across passes.

        Returns:
            Deduplicated entity list.
        """
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for ent in all_entities:
            norm = self._normalize_name(ent.get("name", ""))
            if norm and norm not in seen:
                seen.add(norm)
                merged.append(ent)
        return merged

    def _merge_relationships(
        self, all_relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate relationships by (source, target, type) normalized tuple.

        Args:
            all_relationships: All relationships collected across passes.

        Returns:
            Deduplicated relationship list.
        """
        seen: set[tuple[str, str, str]] = set()
        merged: list[dict[str, Any]] = []
        for rel in all_relationships:
            key = (
                self._normalize_name(rel.get("source", "")),
                self._normalize_name(rel.get("target", "")),
                self._normalize_name(rel.get("type", "")),
            )
            if key not in seen:
                seen.add(key)
                merged.append(rel)
        return merged
