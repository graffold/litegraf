"""Minimal database configuration for standalone pipeline usage."""


class DatabaseConfig:
    @staticmethod
    def get_optimized_config(backend: str, operation_type: str) -> dict:
        return {"batch_size": 50}

    @staticmethod
    def get_timeout_config(backend: str, operation_type: str) -> dict:
        return {"read_timeout": 60}

    @staticmethod
    def estimate_query_complexity(query: str) -> str:
        if len(query) > 500:
            return "complex"
        return "simple"

    @staticmethod
    def serialize_complex_property(value):
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return value
        return str(value)
