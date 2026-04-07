"""Token-based document chunker with overlap and sentence boundary preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from pipeline.ingest.ingestor import Chunk, ProcessedDocument
from src.utils import logging_utils

logger = logging_utils.setup_logging()

# Encoding lookup consistent with TokenCounter
_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "llama": "cl100k_base",
    "mistral": "cl100k_base",
    "claude": "cl100k_base",
}


@dataclass
class TokenChunk:
    """A chunk of text with token-level positioning."""

    chunk_id: str
    text: str
    token_count: int
    start_token: int
    end_token: int


class TokenChunker:
    """Splits documents into token-counted chunks with configurable overlap.

    Uses tiktoken for token counting (consistent with TokenCounter) and
    preserves sentence boundaries using spaCy when available.

    The algorithm:
    1. Tokenize the full document into a single token stream.
    2. Split the document into sentences and map each sentence to its
       token span within the full stream.
    3. Greedily pack sentences into chunks up to *max_tokens*.
    4. When a single sentence exceeds *max_tokens*, split it at the
       token boundary.
    5. Consecutive chunks share an overlap of *overlap_tokens* tokens
       taken from the end of the previous chunk.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        model_name: str = "default",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model_name = model_name

        # Initialize tiktoken encoder
        encoding_name = self._resolve_encoding(model_name)
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(
                f"Failed to get encoding {encoding_name}, "
                f"falling back to cl100k_base: {e}"
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self._nlp = self._load_spacy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_encoding(model_name: str) -> str:
        lower = model_name.lower()
        for key, enc in _MODEL_ENCODINGS.items():
            if key in lower:
                return enc
        return "cl100k_base"

    @staticmethod
    def _load_spacy():  # type: ignore[return]
        """Try to load a spaCy model for sentence segmentation."""
        try:
            import spacy

            return spacy.load("en_core_sci_sm")
        except Exception:
            try:
                import spacy

                return spacy.load("en_core_web_sm")
            except Exception:
                logger.warning(
                    "spaCy model not available; "
                    "falling back to simple sentence splitting"
                )
                return None

    def _split_sentences(self, text: str) -> list[str]:
        """Split *text* into sentences, preferring spaCy."""
        if self._nlp is not None:
            doc = self._nlp(text)
            sents = [s.text for s in doc.sents if s.text.strip()]
            if sents:
                return sents
        # Fallback: split on sentence-ending punctuation followed by space
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in raw if s.strip()]

    def _encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def _decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, doc_id: str) -> list[TokenChunk]:
        """Split *text* into token-counted chunks with overlap.

        Returns an empty list for empty / whitespace-only input.
        """
        if not text or not text.strip():
            return []

        # 1. Tokenize the entire document once
        all_tokens = self._encode(text)
        if not all_tokens:
            return []

        # 2. Get sentence texts and map each to its token span
        sentences = self._split_sentences(text)
        sent_spans = self._map_sentence_spans(all_tokens, sentences)

        # 3. Build chunks by greedily packing sentence spans
        chunks: list[TokenChunk] = []
        start = 0  # current position in all_tokens

        while start < len(all_tokens):
            end = min(start + self.max_tokens, len(all_tokens))

            # Try to snap *end* back to a sentence boundary
            end = self._snap_to_sentence_boundary(start, end, sent_spans, all_tokens)

            chunk_tokens = all_tokens[start:end]
            chunk_text = self._decode(chunk_tokens)
            cid = f"{doc_id}_{len(chunks) + 1}"
            chunks.append(
                TokenChunk(
                    chunk_id=cid,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    start_token=start,
                    end_token=end,
                )
            )

            # Advance, applying overlap
            advance = len(chunk_tokens) - self.overlap_tokens
            if advance <= 0:
                # Overlap >= chunk size — just move forward by 1 to avoid
                # infinite loop
                advance = max(1, len(chunk_tokens))
            start += advance

        return chunks

    def _map_sentence_spans(
        self,
        all_tokens: list[int],
        sentences: list[str],
    ) -> list[tuple[int, int]]:
        """Return (start, end) token offsets for each sentence.

        We encode each sentence individually and walk through *all_tokens*
        to find where each sentence's tokens begin and end.  Because
        concatenated-sentence encoding may differ slightly from
        whole-document encoding, we use a greedy alignment approach.
        """
        spans: list[tuple[int, int]] = []
        pos = 0
        for sent in sentences:
            sent_tokens = self._encode(sent)
            sent_len = len(sent_tokens)
            if sent_len == 0:
                continue
            span_start = pos
            span_end = min(pos + sent_len, len(all_tokens))
            spans.append((span_start, span_end))
            pos = span_end
        return spans

    def _snap_to_sentence_boundary(
        self,
        chunk_start: int,
        chunk_end: int,
        sent_spans: list[tuple[int, int]],
        all_tokens: list[int],
    ) -> int:
        """Try to move *chunk_end* back to the nearest sentence boundary.

        If the chunk would contain only part of a sentence that exceeds
        max_tokens, we allow the hard cut and log a warning.
        """
        if chunk_end >= len(all_tokens):
            return chunk_end

        # Find the last sentence that ends at or before chunk_end
        best_end = chunk_start  # worst case: no sentence fits
        for s_start, s_end in sent_spans:
            if s_end <= chunk_start:
                continue
            if s_start >= chunk_end:
                break
            if s_end <= chunk_end:
                best_end = s_end

        if best_end > chunk_start:
            return best_end

        # No sentence boundary found within budget — a single sentence
        # spans beyond max_tokens.  Hard-cut at token boundary.
        sent_token_count = chunk_end - chunk_start
        logger.warning(
            f"Single sentence (~{sent_token_count} tokens) exceeds "
            f"max_tokens ({self.max_tokens}); splitting at token boundary"
        )
        return chunk_end

    # ------------------------------------------------------------------
    # Drop-in replacement for Chunker.chunk_documents
    # ------------------------------------------------------------------

    def chunk_documents(self, docs: list[ProcessedDocument]) -> list[ProcessedDocument]:
        """Drop-in replacement for ``Chunker.chunk_documents``."""
        for doc in docs:
            token_chunks = self.chunk_text(doc.source, doc.doc_id)
            doc.chunks = [
                Chunk(
                    chunk_id=tc.chunk_id,
                    text=tc.text,
                    pmid=doc.metadata.get("pmid"),
                    title=doc.metadata.get("title"),
                    publication_year=doc.metadata.get("publication_year"),
                )
                for tc in token_chunks
            ]
            logger.info(
                f"Document {doc.doc_id} chunked into {len(doc.chunks)} "
                f"chunks (max_tokens={self.max_tokens}, "
                f"overlap={self.overlap_tokens})"
            )
        return docs
