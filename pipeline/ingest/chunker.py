import spacy

from pipeline.ingest.ingestor import Chunk, ProcessedDocument
from src.utils import logging_utils

logger = logging_utils.setup_logging()


class Chunker:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = spacy.load("en_core_sci_sm")

    def chunk_documents(self, docs: list[ProcessedDocument]) -> list[ProcessedDocument]:
        for doc in docs:
            doc_text = doc.source
            spacy_doc = self.splitter(doc_text)
            sentences = [
                sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()
            ]

            chunks = []
            current_chunk = ""
            current_length = 0
            chunk_id = 1

            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.metadata['pmid']}_{chunk_id}",
                            text=current_chunk.strip(),
                        )
                    )
                    current_chunk = sentence
                    current_length = sentence_length
                    chunk_id += 1
                else:
                    current_chunk += " " + sentence
                    current_length += sentence_length

            if current_chunk:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.metadata['pmid']}_{chunk_id}",
                        text=current_chunk.strip(),
                    )
                )

            doc.chunks = chunks
            logger.info(f"Document {doc.doc_id} chunked into {len(doc.chunks)} chunks")
        return docs
