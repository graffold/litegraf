"""Section parser for identifying and labeling logical sections of scientific papers."""

import re
from dataclasses import dataclass

from src.utils.logging_utils import setup_logging

logger = setup_logging(name=__name__)


@dataclass
class PaperSection:
    """A labeled section of a scientific paper."""

    label: str  # e.g., "Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "Body"
    text: str
    start_offset: int
    end_offset: int


class SectionParser:
    """Parses extracted text into labeled scientific paper sections."""

    STANDARD_SECTIONS = [
        "Abstract",
        "Introduction",
        "Methods",
        "Materials and Methods",
        "Results",
        "Discussion",
        "Conclusion",
        "Conclusions",
    ]
    DEFAULT_LABEL = "Body"

    def __init__(self) -> None:
        self._heading_pattern = self._build_heading_pattern()

    def _build_heading_pattern(self) -> re.Pattern[str]:
        """Build a single regex pattern that matches any heading format for standard sections."""
        # Escape section names for regex and join with alternation
        section_names = "|".join(re.escape(s) for s in self.STANDARD_SECTIONS)

        # Build pattern matching:
        # 1. Markdown headings: ^#{1,3}\s+(SectionName)
        # 2. Numbered headings: ^\d+[.)]\s*(SectionName)
        # 3. All-caps headings: ^(ABSTRACT|INTRODUCTION|...) on their own line
        # 4. Plain headings: ^(SectionName) at start of line
        all_caps = "|".join(re.escape(s.upper()) for s in self.STANDARD_SECTIONS)

        pattern = (
            rf"^(?:"
            rf"#{1, 3}\s+({section_names})"  # Markdown headings
            rf"|\d+[.)]\s*({section_names})"  # Numbered headings
            rf"|({all_caps})"  # All-caps headings
            rf"|({section_names})"  # Plain headings
            rf")\s*$"
        )
        return re.compile(pattern, re.MULTILINE | re.IGNORECASE)

    def _normalize_label(self, match: re.Match[str]) -> str:
        """Extract and normalize the section label from a regex match."""
        # The match has 4 groups; pick the first non-None one
        for group in match.groups():
            if group is not None:
                return group.strip().title()
        return self.DEFAULT_LABEL

    def parse(self, text: str) -> list[PaperSection]:
        """Identify and label sections from extracted paper text.

        Args:
            text: The full extracted text of a scientific paper.

        Returns:
            List of PaperSection objects with label, text, start_offset, end_offset.
            Empty list if text is empty.
        """
        if not text:
            return []

        matches = list(self._heading_pattern.finditer(text))

        if not matches:
            logger.debug(
                "No section headings found, assigning all text to '%s'",
                self.DEFAULT_LABEL,
            )
            return [
                PaperSection(
                    label=self.DEFAULT_LABEL,
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                )
            ]

        sections: list[PaperSection] = []

        # Text before the first heading gets the default label
        first_match_start = matches[0].start()
        if first_match_start > 0:
            pre_text = text[:first_match_start].strip()
            if pre_text:
                sections.append(
                    PaperSection(
                        label=self.DEFAULT_LABEL,
                        text=pre_text,
                        start_offset=0,
                        end_offset=first_match_start,
                    )
                )

        # Process each heading match
        for i, match in enumerate(matches):
            label = self._normalize_label(match)
            # Section text starts after the heading line
            section_start = match.end()
            # Section text ends at the start of the next heading (or end of text)
            section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_text = text[section_start:section_end].strip()
            sections.append(
                PaperSection(
                    label=label,
                    text=section_text,
                    start_offset=section_start,
                    end_offset=section_end,
                )
            )

        logger.debug("Parsed %d sections from text", len(sections))
        return sections

    @staticmethod
    def filter_sections(
        sections: list[PaperSection], labels: list[str]
    ) -> list[PaperSection]:
        """Filter sections to only include those whose labels are in the provided list.

        Args:
            sections: List of PaperSection objects to filter.
            labels: List of section labels to keep (case-insensitive comparison).

        Returns:
            Filtered list of PaperSection objects.
        """
        labels_lower = {label.lower() for label in labels}
        return [s for s in sections if s.label.lower() in labels_lower]
