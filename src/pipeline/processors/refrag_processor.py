"""
REFRAG (Retrieval-Enhanced Few-shot RAG) Processor
Implements Meta's REFRAG approach for compressed retrieval and processing

This processor implements the 4-step REFRAG approach:
1. Compress: Encode 16-token chunks into dense embeddings
2. Shorten: Replace raw tokens with compressed embeddings
3. Accelerate: Reduce attention computation and KV cache
4. Select: Use RL policy to preserve critical chunks
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
# Module-level cache for transformer models used by ChunkEncoder
_chunk_encoder_models: dict[
    str, tuple[Any, Any]
] = {}  # model_name -> (tokenizer, model)


class ChunkEncoder:
    """Small, lightweight encoder for compressing 16-token chunks"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if model_name not in _chunk_encoder_models:
            logger.info(
                f"Loading ChunkEncoder model '{model_name}' (first time, will be cached)"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            _chunk_encoder_models[model_name] = (tokenizer, model)
        self.tokenizer, self.model = _chunk_encoder_models[model_name]
        self.chunk_size = 16  # tokens per chunk
        self.embedding_dim = 384  # dimension of compressed embeddings

    def encode_chunks(self, text: str) -> list[torch.Tensor]:
        """
        Compress every 16-token chunk into dense embeddings

        Args:
            text: Input text to be chunked and compressed

        Returns:
            List of chunk embeddings (16x compression ratio)
        """
        try:
            # Tokenize the full text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Split into 16-token chunks
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i : i + self.chunk_size]

                # Pad if necessary
                if len(chunk_tokens) < self.chunk_size:
                    chunk_tokens.extend(
                        [self.tokenizer.pad_token_id]
                        * (self.chunk_size - len(chunk_tokens))
                    )

                # Convert back to text and encode
                chunk_text = self.tokenizer.decode(
                    chunk_tokens, skip_special_tokens=True
                )

                # Get dense embedding for this chunk
                inputs = self.tokenizer(
                    chunk_text, return_tensors="pt", padding=True, truncation=True
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling for chunk embedding
                    chunk_embedding = outputs.last_hidden_state.mean(dim=1)

                chunks.append(chunk_embedding.squeeze())

            logger.info(
                f"Compressed {len(tokens)} tokens into {len(chunks)} chunk embeddings ({len(tokens) / len(chunks):.1f}x compression)"
            )
            return chunks

        except Exception as e:
            logger.error(f"Error encoding chunks: {e}")
            return []


class CriticalitySelector:
    """RL-based policy for selecting critical chunks to preserve uncompressed"""

    def __init__(self):
        # This would typically be a trained RL model
        # For now, we'll use heuristics as a placeholder
        self.critical_keywords = [
            "protein",
            "disease",
            "biomarker",
            "pathway",
            "interaction",
            "mutation",
            "expression",
            "function",
            "therapeutic",
            "clinical",
        ]

    def select_critical_chunks(
        self, chunks: list[str], embeddings: list[torch.Tensor]
    ) -> list[bool]:
        """
        Identify which chunks should remain uncompressed

        Args:
            chunks: Original text chunks
            embeddings: Corresponding chunk embeddings

        Returns:
            Boolean mask indicating which chunks are critical
        """
        critical_mask = []

        for i, chunk in enumerate(chunks):
            # Heuristic: chunks containing domain-specific keywords are critical
            is_critical = any(
                keyword in chunk.lower() for keyword in self.critical_keywords
            )

            # Additional heuristic: chunks with high embedding magnitude might be information-dense
            if len(embeddings) > i:
                embedding_magnitude = torch.norm(embeddings[i]).item()
                if embedding_magnitude > np.percentile(
                    [torch.norm(emb).item() for emb in embeddings], 75
                ):
                    is_critical = True

            critical_mask.append(is_critical)

        logger.info(
            f"Selected {sum(critical_mask)}/{len(critical_mask)} chunks as critical"
        )
        return critical_mask


class REFRAGProcessor:
    """Main REFRAG processor implementing the complete pipeline"""

    def __init__(self):
        self.chunk_encoder = ChunkEncoder()
        self.criticality_selector = CriticalitySelector()

    def process_retrieved_documents(self, documents: list[str]) -> dict[str, Any]:
        """
        Process retrieved documents using REFRAG approach

        Args:
            documents: List of retrieved document texts

        Returns:
            Dictionary containing compressed and preserved content
        """
        processed_docs = []
        total_compression_ratio = 0

        for doc_idx, document in enumerate(documents):
            logger.info(f"Processing document {doc_idx + 1}/{len(documents)}")

            # Step 1: Compress - Encode 16-token chunks
            chunk_embeddings = self.chunk_encoder.encode_chunks(document)

            # Create text chunks for selection
            tokens = self.chunk_encoder.tokenizer.encode(
                document, add_special_tokens=False
            )
            text_chunks = []
            for i in range(0, len(tokens), self.chunk_encoder.chunk_size):
                chunk_tokens = tokens[i : i + self.chunk_encoder.chunk_size]
                chunk_text = self.chunk_encoder.tokenizer.decode(
                    chunk_tokens, skip_special_tokens=True
                )
                text_chunks.append(chunk_text)

            # Step 4: Select - Identify critical chunks to preserve
            critical_mask = self.criticality_selector.select_critical_chunks(
                text_chunks, chunk_embeddings
            )

            # Separate compressed and preserved content
            compressed_chunks = []
            preserved_chunks = []

            for i, (chunk_text, chunk_embedding, is_critical) in enumerate(
                zip(text_chunks, chunk_embeddings, critical_mask, strict=False)
            ):
                if is_critical:
                    preserved_chunks.append(
                        {"position": i, "text": chunk_text, "type": "preserved"}
                    )
                else:
                    compressed_chunks.append(
                        {
                            "position": i,
                            "embedding": chunk_embedding,
                            "type": "compressed",
                        }
                    )

            # Calculate compression ratio for this document
            original_tokens = len(tokens)
            compressed_tokens = len(preserved_chunks) * self.chunk_encoder.chunk_size
            compression_ratio = (
                original_tokens / (compressed_tokens + len(compressed_chunks))
                if (compressed_tokens + len(compressed_chunks)) > 0
                else 1
            )
            total_compression_ratio += compression_ratio

            processed_docs.append(
                {
                    "original_length": original_tokens,
                    "compressed_chunks": compressed_chunks,
                    "preserved_chunks": preserved_chunks,
                    "compression_ratio": compression_ratio,
                }
            )

        avg_compression_ratio = (
            total_compression_ratio / len(documents) if documents else 1
        )

        return {
            "processed_documents": processed_docs,
            "total_documents": len(documents),
            "average_compression_ratio": avg_compression_ratio,
            "metadata": {
                "chunk_size": self.chunk_encoder.chunk_size,
                "embedding_dim": self.chunk_encoder.embedding_dim,
            },
        }

    def prepare_llm_input(
        self, processed_result: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """
        Prepare the compressed input for the main LLM

        Args:
            processed_result: Output from process_retrieved_documents
            query: Original user query

        Returns:
            Prepared input for LLM with compressed content
        """
        # Step 2: Shorten - Create shortened input sequence
        llm_input_parts = [f"Query: {query}\n\nRetrieved Information:"]

        for doc_idx, doc in enumerate(processed_result["processed_documents"]):
            llm_input_parts.append(f"\nDocument {doc_idx + 1}:")

            # Add preserved chunks (full text)
            for chunk in doc["preserved_chunks"]:
                llm_input_parts.append(f"[CRITICAL] {chunk['text']}")

            # Add markers for compressed chunks
            if doc["compressed_chunks"]:
                llm_input_parts.append(
                    f"[COMPRESSED] {len(doc['compressed_chunks'])} chunks compressed"
                )

        llm_input = "\n".join(llm_input_parts)

        # Step 3: Accelerate - The shortened input enables faster processing
        return {
            "input_text": llm_input,
            "compressed_embeddings": [
                chunk["embedding"]
                for doc in processed_result["processed_documents"]
                for chunk in doc["compressed_chunks"]
            ],
            "compression_stats": {
                "total_docs": processed_result["total_documents"],
                "avg_compression_ratio": processed_result["average_compression_ratio"],
            },
        }
