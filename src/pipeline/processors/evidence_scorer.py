"""
Evidence scoring for biomedical literature using LLM-based study design classification.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum

from pipeline.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class StudyDesign(Enum):
    """Types of study designs with associated quality weights."""

    RCT = ("randomized_controlled_trial", 1.0)  # Highest quality
    COHORT = ("cohort_study", 0.7)
    CASE_CONTROL = ("case_control_study", 0.6)
    CASE_REPORT = ("case_report", 0.3)  # Lowest quality
    OBSERVATIONAL = ("observational_study", 0.5)
    SYSTEMATIC_REVIEW = ("systematic_review", 0.9)
    META_ANALYSIS = ("meta_analysis", 0.95)
    UNKNOWN = ("unknown", 0.4)

    def __init__(self, study_type: str, weight: float):
        self.study_type = study_type
        self.weight = weight


@dataclass
class StudyClassification:
    """Result of study design classification."""

    study_design: StudyDesign
    confidence: float  # LLM confidence in classification (0.0-1.0)
    reasoning: str  # LLM explanation for classification


@dataclass
class SampleSizeExtraction:
    """Result of sample size extraction."""

    sample_size: int | None  # Number of participants/samples, None if not found
    confidence: float  # LLM confidence in extraction (0.0-1.0)
    reasoning: str  # LLM explanation for extraction


class EvidenceScorer:
    """
    LLM-based evidence scorer for biomedical literature.

    Classifies study designs and assigns quality weights for evidence aggregation.
    """

    CLASSIFICATION_PROMPT = """You are a biomedical research expert. Classify the study design of the following abstract.

Study Design Types:
- randomized_controlled_trial (RCT): Participants randomly assigned to treatment/control groups
- cohort_study: Follows a group over time to observe outcomes
- case_control_study: Compares cases (with disease) to controls (without disease)
- case_report: Describes individual patient case(s)
- observational_study: Observes without intervention
- systematic_review: Comprehensive review of existing studies
- meta_analysis: Statistical analysis combining multiple studies
- unknown: Cannot determine from abstract

Abstract:
{abstract}

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "study_design": "<one of the types above>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}"""

    SAMPLE_SIZE_PROMPT = """You are a biomedical research expert. Extract the sample size (number of participants, subjects, or samples) from the following abstract.

Look for:
- Number of participants/subjects/patients in the study
- Sample size mentioned in methods or results
- Cohort size, group sizes, or total N
- Phrases like "n=100", "100 patients", "cohort of 500"

Abstract:
{abstract}

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "sample_size": <integer or null if not found>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of where/how you found it, or why not found>"
}}"""

    def __init__(self, llm: LLMProvider):
        """
        Initialize evidence scorer.

        Args:
            llm: LLM instance for classification
        """
        self.llm = llm

    async def score_study_design(self, abstract: str) -> StudyClassification:
        """
        Classify study design from abstract text using LLM.

        Args:
            abstract: Abstract text to classify

        Returns:
            StudyClassification with design type, confidence, and reasoning
        """
        prompt = self.CLASSIFICATION_PROMPT.format(abstract=abstract)

        try:
            response = await self.llm.ainvoke(prompt)

            # Parse LLM response
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Clean markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split("\n")
                response_text = (
                    "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                )
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )

            result = json.loads(response_text)

            # Map string to StudyDesign enum
            design_str = result["study_design"].lower()
            study_design = self._map_to_study_design(design_str)

            return StudyClassification(
                study_design=study_design,
                confidence=float(result["confidence"]),
                reasoning=result["reasoning"],
            )

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse LLM response as JSON: {e}. Response: {response_text[:200]}"
            )
            return StudyClassification(
                study_design=StudyDesign.UNKNOWN,
                confidence=0.0,
                reasoning="Failed to parse LLM response",
            )
        except Exception as e:
            logger.error(f"Error classifying study design: {e}")
            return StudyClassification(
                study_design=StudyDesign.UNKNOWN,
                confidence=0.0,
                reasoning=f"Error: {e!s}",
            )

    def _map_to_study_design(self, design_str: str) -> StudyDesign:
        """Map string to StudyDesign enum."""
        design_map = {
            "randomized_controlled_trial": StudyDesign.RCT,
            "rct": StudyDesign.RCT,
            "cohort_study": StudyDesign.COHORT,
            "cohort": StudyDesign.COHORT,
            "case_control_study": StudyDesign.CASE_CONTROL,
            "case_control": StudyDesign.CASE_CONTROL,
            "case-control": StudyDesign.CASE_CONTROL,
            "case_report": StudyDesign.CASE_REPORT,
            "case report": StudyDesign.CASE_REPORT,
            "observational_study": StudyDesign.OBSERVATIONAL,
            "observational": StudyDesign.OBSERVATIONAL,
            "systematic_review": StudyDesign.SYSTEMATIC_REVIEW,
            "systematic review": StudyDesign.SYSTEMATIC_REVIEW,
            "meta_analysis": StudyDesign.META_ANALYSIS,
            "meta-analysis": StudyDesign.META_ANALYSIS,
            "meta analysis": StudyDesign.META_ANALYSIS,
            "unknown": StudyDesign.UNKNOWN,
        }
        return design_map.get(design_str.lower(), StudyDesign.UNKNOWN)

    def get_study_design_weight(self, study_design: StudyDesign) -> float:
        """
        Get quality weight for a study design.

        Args:
            study_design: Study design type

        Returns:
            Quality weight (0.0-1.0)
        """
        return study_design.weight

    async def extract_sample_size(self, abstract: str) -> SampleSizeExtraction:
        """
        Extract sample size from abstract text using LLM.

        Args:
            abstract: Abstract text to extract from

        Returns:
            SampleSizeExtraction with sample size, confidence, and reasoning
        """
        prompt = self.SAMPLE_SIZE_PROMPT.format(abstract=abstract)

        try:
            response = await self.llm.ainvoke(prompt)

            # Parse LLM response
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Clean markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = (
                    "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                )
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )

            result = json.loads(response_text)

            # Handle null/None sample size
            sample_size = result.get("sample_size")
            if sample_size is not None:
                sample_size = int(sample_size)

            return SampleSizeExtraction(
                sample_size=sample_size,
                confidence=float(result["confidence"]),
                reasoning=result["reasoning"],
            )

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse LLM response as JSON: {e}. Response: {response_text[:200]}"
            )
            return SampleSizeExtraction(
                sample_size=None,
                confidence=0.0,
                reasoning="Failed to parse LLM response",
            )
        except Exception as e:
            logger.error(f"Error extracting sample size: {e}")
            return SampleSizeExtraction(
                sample_size=None, confidence=0.0, reasoning=f"Error: {e!s}"
            )

    def calculate_evidence_confidence(
        self,
        study_design: StudyDesign,
        sample_size: int | None = None,
        replication_count: int = 1,
    ) -> float:
        """
        Calculate aggregate confidence score based on study design, sample size, and replication.

        Formula:
        - Base score from study design weight (0.3-1.0)
        - Sample size bonus: +0.1 for n≥100, +0.05 for n≥50
        - Replication bonus: +0.05 per additional study (max +0.15)
        - Final score capped at 1.0

        Args:
            study_design: Study design type
            sample_size: Number of participants (None if unknown)
            replication_count: Number of independent studies supporting this finding

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with study design weight
        score = study_design.weight

        # Sample size bonus
        if sample_size is not None:
            if sample_size >= 100:
                score += 0.1
            elif sample_size >= 50:
                score += 0.05

        # Replication bonus (max +0.15 for 3+ studies)
        if replication_count > 1:
            replication_bonus = min((replication_count - 1) * 0.05, 0.15)
            score += replication_bonus

        # Cap at 1.0
        return min(score, 1.0)
