"""Minimal retry utilities for standalone pipeline usage."""
import logging
import time

logger = logging.getLogger(__name__)


class UnifiedRetryUtilities:
    def execute_with_retry(
        self,
        operation,
        backend="neo4j",
        max_retries=3,
        retry_delay=2.0,
        operation_name="",
    ):
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {operation_name}: {e}"
                    )
                    time.sleep(retry_delay * (attempt + 1))
        raise last_error
