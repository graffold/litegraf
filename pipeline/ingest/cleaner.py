import logging
import re

from pipeline.ingest.ingestor import ProcessedDocument
logger = logging.getLogger(__name__)
class Cleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[\ufeff]", "", text)
        text = re.sub(r"[\x00-\x09]", " ", text)
        return text.strip()

    def clean_documents(self, docs: list[ProcessedDocument]) -> list[ProcessedDocument]:
        for doc in docs:
            doc.source = self.clean_text(doc.source)
            logger.debug(f"Cleaned document {doc.doc_id}")
        return docs
