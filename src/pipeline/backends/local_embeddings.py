"""Local embedding provider using sentence-transformers.

Wraps the ``SentenceTransformer`` model to satisfy the
:class:`~pipeline.interfaces.EmbeddingProvider` ABC.  The default model is
``all-mpnet-base-v2`` (768-dimensional embeddings) but any model supported by
sentence-transformers can be passed via the *model_name* constructor argument.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from pipeline.interfaces import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by a local sentence-transformers model.

    Parameters
    ----------
    model_name:
        Name or path of the sentence-transformers model to load.
        Defaults to ``"all-mpnet-base-v2"`` (768 dimensions).
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    # -- EmbeddingProvider interface -----------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""
        vector: list[float] = self._model.encode(text).tolist()  # type: ignore[union-attr]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings. Returns a list of float vectors."""
        vectors: list[list[float]] = self._model.encode(texts).tolist()  # type: ignore[union-attr]
        return vectors
