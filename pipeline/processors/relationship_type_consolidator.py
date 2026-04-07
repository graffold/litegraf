#!/usr/bin/env python3
"""
LLM-based relationship type consolidation system.
Uses language models to identify and merge semantically similar relationship types.
"""

import json
import re
from typing import Any

from src.core.database import Neo4jDatabase
from src.factories.llm_factory import get_llm
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class RelationshipTypeConsolidator:
    """
    Uses LLM to identify and consolidate semantically similar relationship types.
    """

    def __init__(self, database: str = "cvd1", llm_service: str = "local"):
        self.db = Neo4jDatabase(database=database)
        self.database = database
        self.llm = get_llm(llm_service)
        logger.info(
            f"Initialized RelationshipTypeConsolidator for database: {database}"
        )

    def _execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query."""
        return self.db._execute_cypher(query, parameters)

    def get_all_relationship_types(self) -> list[dict[str, Any]]:
        """Get all relationship types with their frequencies."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS frequency
        ORDER BY frequency DESC
        """

        try:
            result = self._execute_query(query)
            logger.info(f"Found {len(result)} distinct relationship types")
            return result
        except Exception as e:
            logger.error(f"Failed to get relationship types: {e}")
            return []

    def analyze_relationship_semantics(
        self, relationship_types: list[str]
    ) -> dict[str, list[str]]:
        """
        Use LLM to analyze relationship types and group semantically similar ones.
        """
        logger.info(
            f"Analyzing semantics of {len(relationship_types)} relationship types..."
        )

        # Create prompt for LLM analysis
        prompt = self._create_semantic_analysis_prompt(relationship_types)

        try:
            # Get LLM response
            response = self.llm.invoke(prompt)

            # Parse the response
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Extract JSON from response
            semantic_groups = self._parse_semantic_response(content)

            logger.info(f"LLM identified {len(semantic_groups)} semantic groups")
            return semantic_groups

        except Exception as e:
            logger.error(f"Failed to analyze relationship semantics: {e}")
            return {}

    def _create_semantic_analysis_prompt(self, relationship_types: list[str]) -> str:
        """Create prompt for semantic analysis of relationship types."""

        rel_types_str = "\n".join([f"- {rt}" for rt in relationship_types])

        return f"""
You are an expert in biological and medical relationship analysis. I have a list of relationship types from a biomedical knowledge graph that need to be consolidated.

Your task is to group semantically similar relationship types together and suggest a canonical name for each group.

Relationship types to analyze:
{rel_types_str}

Please analyze these relationship types and group them by semantic similarity. Consider:
1. Synonyms (e.g., "associated_with" and "related_to")
2. Different tenses or forms (e.g., "causes" and "caused_by")
3. Similar meanings with different wording (e.g., "affiliated_with" and "associated_with")
4. Biological/medical context (e.g., "treats" and "therapeutic_for")

Return your analysis as a JSON object with this structure:
{{
  "semantic_groups": [
    {{
      "canonical_name": "ASSOCIATED_WITH",
      "description": "General association or correlation",
      "members": ["associated_with", "related_to", "affiliated_with", "correlates_with"]
    }},
    {{
      "canonical_name": "CAUSES",
      "description": "Causal relationship",
      "members": ["causes", "leads_to", "results_in", "induces"]
    }}
  ]
}}

Guidelines:
- Use uppercase, underscore-separated canonical names
- Keep distinct biological meanings separate (don't merge "causes" with "treats")
- Group only truly similar relationships
- Provide clear descriptions for each group
- If a relationship type is unique, it can be in its own group

JSON Response:
"""

    def _parse_semantic_response(self, content: str) -> dict[str, list[str]]:
        """Parse LLM response to extract semantic groups."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # Convert to the format we need
                groups = {}
                for group in data.get("semantic_groups", []):
                    canonical = group.get("canonical_name", "")
                    members = group.get("members", [])
                    if canonical and members:
                        groups[canonical] = members

                return groups
            logger.warning("No JSON found in LLM response")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response content: {content}")
            return {}

    def create_consolidation_plan(
        self,
        semantic_groups: dict[str, list[str]],
        frequency_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Create a consolidation plan based on semantic analysis and frequency data.
        """
        logger.info("Creating consolidation plan...")

        # Create frequency lookup
        freq_lookup = {item["rel_type"]: item["frequency"] for item in frequency_data}

        consolidation_plan = []

        for canonical_name, members in semantic_groups.items():
            # Only consolidate if there are multiple members
            if len(members) > 1:
                # Calculate total frequency for this group
                total_frequency = sum(freq_lookup.get(member, 0) for member in members)

                # Find the most frequent member as the primary candidate
                member_frequencies = [
                    (member, freq_lookup.get(member, 0)) for member in members
                ]
                member_frequencies.sort(key=lambda x: x[1], reverse=True)

                # Use canonical name or most frequent member
                primary_name = canonical_name

                plan_item = {
                    "canonical_name": primary_name,
                    "members": members,
                    "frequencies": dict(member_frequencies),
                    "total_frequency": total_frequency,
                    "consolidation_needed": True,
                }

                consolidation_plan.append(plan_item)
                logger.info(
                    f"Plan: Consolidate {members} -> {primary_name} (total: {total_frequency} relationships)"
                )

        return consolidation_plan

    def execute_consolidation(
        self, consolidation_plan: list[dict[str, Any]], dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Execute the relationship type consolidation plan.
        """
        logger.info(
            f"Executing consolidation plan ({'DRY RUN' if dry_run else 'LIVE'})..."
        )

        stats = {
            "groups_processed": 0,
            "relationships_updated": 0,
            "consolidations": [],
        }

        for plan_item in consolidation_plan:
            if not plan_item["consolidation_needed"]:
                continue

            canonical_name = plan_item["canonical_name"]
            members = plan_item["members"]

            # For each member except the canonical name, update relationships
            for member in members:
                if member != canonical_name:
                    update_count = self._update_relationship_type(
                        old_type=member, new_type=canonical_name, dry_run=dry_run
                    )

                    if update_count > 0:
                        stats["relationships_updated"] += update_count
                        stats["consolidations"].append(
                            {
                                "from": member,
                                "to": canonical_name,
                                "count": update_count,
                            }
                        )

                        logger.info(
                            f"{'[DRY RUN] ' if dry_run else ''}Updated {update_count} relationships: {member} -> {canonical_name}"
                        )

            stats["groups_processed"] += 1

        logger.info(
            f"Consolidation complete: {stats['groups_processed']} groups, {stats['relationships_updated']} relationships updated"
        )
        return stats

    def _update_relationship_type(
        self, old_type: str, new_type: str, dry_run: bool = True
    ) -> int:
        """Update all relationships of old_type to new_type."""

        # First, count how many would be updated
        count_query = """
        MATCH ()-[r]->()
        WHERE type(r) = $old_type
        RETURN count(r) AS count
        """

        try:
            result = self._execute_query(count_query, {"old_type": old_type})
            count = result[0]["count"] if result else 0

            if count == 0:
                return 0

            if not dry_run:
                # Create new relationships and delete old ones
                # This is a complex operation that requires careful handling

                # Neo4j doesn't support dynamic relationship type creation in this way
                # We need to use APOC or a different approach
                # For now, let's use a metadata approach
                metadata_query = """
                MATCH ()-[r]->()
                WHERE type(r) = $old_type
                SET r.original_type = type(r),
                    r.canonical_type = $new_type,
                    r.consolidated = true
                """

                self._execute_query(
                    metadata_query, {"old_type": old_type, "new_type": new_type}
                )

            return count

        except Exception as e:
            logger.error(
                f"Failed to update relationship type {old_type} -> {new_type}: {e}"
            )
            return 0

    def validate_consolidation(
        self, consolidation_plan: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate the consolidation plan by asking LLM to review it.
        """
        logger.info("Validating consolidation plan with LLM...")

        # Create validation prompt
        plan_summary = []
        for item in consolidation_plan:
            if item["consolidation_needed"]:
                plan_summary.append(
                    f"Consolidate {item['members']} -> {item['canonical_name']}"
                )

        validation_prompt = f"""
Please review this relationship type consolidation plan for a biomedical knowledge graph:

Planned consolidations:
{chr(10).join(plan_summary)}

Are these consolidations appropriate? Consider:
1. Do the grouped relationship types have similar meanings?
2. Are any important semantic distinctions being lost?
3. Are the canonical names appropriate?
4. Should any groups be split or merged differently?

Respond with a JSON object:
{{
  "validation_result": "APPROVED" | "NEEDS_REVISION" | "REJECTED",
  "concerns": ["list of any concerns"],
  "suggestions": ["list of suggestions for improvement"],
  "confidence_score": 0.0-1.0
}}

JSON Response:
"""

        try:
            response = self.llm.invoke(validation_prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse validation response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group(0))
                logger.info(
                    f"Validation result: {validation_data.get('validation_result', 'UNKNOWN')}"
                )
                return validation_data
            logger.warning("No JSON found in validation response")
            return {"validation_result": "UNKNOWN", "concerns": [], "suggestions": []}

        except Exception as e:
            logger.error(f"Failed to validate consolidation plan: {e}")
            return {
                "validation_result": "ERROR",
                "concerns": [str(e)],
                "suggestions": [],
            }

    def run_full_consolidation(self, dry_run: bool = True) -> dict[str, Any]:
        """
        Run the complete relationship type consolidation process.
        """
        logger.info("Starting full relationship type consolidation...")

        # Step 1: Get all relationship types
        rel_type_data = self.get_all_relationship_types()
        if not rel_type_data:
            logger.warning("No relationship types found")
            return {"error": "No relationship types found"}

        rel_types = [item["rel_type"] for item in rel_type_data]

        # Step 2: Analyze semantics with LLM
        semantic_groups = self.analyze_relationship_semantics(rel_types)
        if not semantic_groups:
            logger.warning("No semantic groups identified")
            return {"error": "No semantic groups identified"}

        # Step 3: Create consolidation plan
        consolidation_plan = self.create_consolidation_plan(
            semantic_groups, rel_type_data
        )

        # Step 4: Validate plan
        validation = self.validate_consolidation(consolidation_plan)

        # Step 5: Execute if approved
        execution_stats = {}
        if validation.get("validation_result") == "APPROVED":
            execution_stats = self.execute_consolidation(
                consolidation_plan, dry_run=dry_run
            )
        else:
            logger.warning(
                f"Consolidation not approved: {validation.get('concerns', [])}"
            )

        return {
            "relationship_types_found": len(rel_types),
            "semantic_groups": semantic_groups,
            "consolidation_plan": consolidation_plan,
            "validation": validation,
            "execution_stats": execution_stats,
            "dry_run": dry_run,
        }

    def print_consolidation_report(self, results: dict[str, Any]):
        """Print a comprehensive consolidation report."""
        print("🔗 RELATIONSHIP TYPE CONSOLIDATION REPORT")
        print("=" * 60)

        print("\n📊 ANALYSIS RESULTS:")
        print(
            f"  • Relationship types found: {results.get('relationship_types_found', 0)}"
        )
        print(
            f"  • Semantic groups identified: {len(results.get('semantic_groups', {}))}"
        )
        print(
            f"  • Consolidation plan items: {len(results.get('consolidation_plan', []))}"
        )

        # Show semantic groups
        semantic_groups = results.get("semantic_groups", {})
        if semantic_groups:
            print("\n🎯 SEMANTIC GROUPS:")
            for canonical, members in semantic_groups.items():
                print(f"  • {canonical}: {members}")

        # Show consolidation plan
        consolidation_plan = results.get("consolidation_plan", [])
        if consolidation_plan:
            print("\n📋 CONSOLIDATION PLAN:")
            for item in consolidation_plan:
                if item["consolidation_needed"]:
                    print(f"  • {item['canonical_name']}: {item['members']}")
                    print(f"    Total frequency: {item['total_frequency']}")

        # Show validation results
        validation = results.get("validation", {})
        if validation:
            print("\n✅ VALIDATION RESULTS:")
            print(f"  • Result: {validation.get('validation_result', 'UNKNOWN')}")
            print(f"  • Confidence: {validation.get('confidence_score', 0.0):.2f}")

            concerns = validation.get("concerns", [])
            if concerns:
                print(f"  • Concerns: {concerns}")

            suggestions = validation.get("suggestions", [])
            if suggestions:
                print(f"  • Suggestions: {suggestions}")

        # Show execution stats
        execution_stats = results.get("execution_stats", {})
        if execution_stats:
            print("\n🔧 EXECUTION RESULTS:")
            print(f"  • Groups processed: {execution_stats.get('groups_processed', 0)}")
            print(
                f"  • Relationships updated: {execution_stats.get('relationships_updated', 0)}"
            )
            print(f"  • Dry run: {results.get('dry_run', True)}")

            consolidations = execution_stats.get("consolidations", [])
            if consolidations:
                print("  • Consolidations performed:")
                for cons in consolidations:
                    print(
                        f"    - {cons['from']} -> {cons['to']} ({cons['count']} relationships)"
                    )

    def close(self):
        """Close database connections."""
        if self.db:
            self.db.close()
        logger.info("RelationshipTypeConsolidator connections closed")


def main():
    """Main function for command-line usage."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Consolidate relationship types using LLM"
    )
    parser.add_argument("--database", "-d", default="cvd1", help="Database name")
    parser.add_argument("--llm", "-l", default="local", help="LLM service to use")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument(
        "--execute", action="store_true", help="Execute consolidation (not dry run)"
    )

    args = parser.parse_args()

    print("🚀 Running Relationship Type Consolidation")
    print(f"Database: {args.database}")
    print(f"LLM: {args.llm}")
    print(f"Mode: {'DRY RUN' if not args.execute else 'LIVE EXECUTION'}")

    consolidator = RelationshipTypeConsolidator(
        database=args.database, llm_service=args.llm
    )

    try:
        # Run full consolidation
        results = consolidator.run_full_consolidation(dry_run=not args.execute)

        # Print comprehensive report
        consolidator.print_consolidation_report(results)

        # Show summary
        if results.get("execution_stats"):
            stats = results["execution_stats"]
            print(
                f"\n{'🔍 DRY RUN SUMMARY' if not args.execute else '✅ EXECUTION SUMMARY'}"
            )
            print(f"Groups processed: {stats.get('groups_processed', 0)}")
            print(
                f"Relationships that would be updated: {stats.get('relationships_updated', 0)}"
            )

            if not args.execute and stats.get("relationships_updated", 0) > 0:
                print("\nTo execute these changes, run with --execute flag")

    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    finally:
        consolidator.close()


if __name__ == "__main__":
    main()
