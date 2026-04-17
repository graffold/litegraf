"""Cross-encoder reranker backend using sentence-transformers."""

from __future__ import annotations

from typing import Any

from pipeline.interfaces import RerankerProvider


class CrossEncoderReranker(RerankerProvider):
    """Reranker using a sentence-transformers CrossEncoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        return sorted(candidates, key=lambda c: c["score"], reverse=True)
