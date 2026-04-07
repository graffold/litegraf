"""CSV and Excel extraction strategies for the My Files Data Explorer.

Implements ExtractionStrategy subclasses that parse CSV and Excel files,
returning the tabular content as text in ExtractionResult.
"""

import time

import pandas as pd

from pipeline.ingest.extraction_models import (
    ExtractionResult,
    ExtractionSource,
    SourceType,
)
from pipeline.ingest.extraction_strategies import ExtractionStrategy


class CSVExtractionStrategy(ExtractionStrategy):
    """Extract tabular content from CSV files using pandas."""

    @property
    def name(self) -> str:
        return "csv"

    def is_available(self) -> bool:
        return True

    def supports_source_type(self, source_type: SourceType) -> bool:
        return source_type == SourceType.CSV

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        start = time.time()
        try:
            df = pd.read_csv(source.content)
            if df.empty:
                return ExtractionResult(
                    success=False,
                    text="",
                    method=self.name,
                    execution_time=time.time() - start,
                    error="CSV file contains no parseable rows",
                    metadata=source.metadata,
                )
            return ExtractionResult(
                success=True,
                text=df.to_csv(index=False),
                method=self.name,
                execution_time=time.time() - start,
                metadata={**(source.metadata or {}), "rows": len(df), "columns": list(df.columns)},
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                text="",
                method=self.name,
                execution_time=time.time() - start,
                error=str(e),
                metadata=source.metadata,
            )


class ExcelExtractionStrategy(ExtractionStrategy):
    """Extract tabular content from Excel files (.xlsx, .xls) using pandas."""

    @property
    def name(self) -> str:
        return "excel"

    def is_available(self) -> bool:
        try:
            import openpyxl  # noqa: F401
            return True
        except ImportError:
            return False

    def supports_source_type(self, source_type: SourceType) -> bool:
        return source_type == SourceType.EXCEL

    def extract(self, source: ExtractionSource) -> ExtractionResult:
        start = time.time()
        try:
            engine = "xlrd" if source.content.endswith(".xls") else "openpyxl"
            df = pd.read_excel(source.content, engine=engine)
            if df.empty:
                return ExtractionResult(
                    success=False,
                    text="",
                    method=self.name,
                    execution_time=time.time() - start,
                    error="Excel file contains no parseable rows",
                    metadata=source.metadata,
                )
            return ExtractionResult(
                success=True,
                text=df.to_csv(index=False),
                method=self.name,
                execution_time=time.time() - start,
                metadata={**(source.metadata or {}), "rows": len(df), "columns": list(df.columns)},
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                text="",
                method=self.name,
                execution_time=time.time() - start,
                error=str(e),
                metadata=source.metadata,
            )
