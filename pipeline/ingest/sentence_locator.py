"""Locate source sentences within chunk text and record character offsets."""

from difflib import SequenceMatcher


def locate_sentence(chunk_text: str, sentence: str, threshold: float = 0.8) -> str:
    """Return 'char_start:char_end' for *sentence* inside *chunk_text*.

    Strategy:
      1. Exact substring match.
      2. Sliding-window fuzzy match (SequenceMatcher ratio >= *threshold*).
      3. Sentinel '0:0' when nothing matches.
    """
    if not sentence or not chunk_text:
        return "0:0"

    # 1. Exact match
    idx = chunk_text.find(sentence)
    if idx != -1:
        return f"{idx}:{idx + len(sentence)}"

    # 2. Fuzzy sliding window
    best_ratio = 0.0
    best_start = 0
    best_end = 0
    window = len(sentence)
    # Allow ±30 % window size variation
    min_w = max(1, int(window * 0.7))
    max_w = min(len(chunk_text), int(window * 1.3))

    for w in (window, min_w, max_w):
        for start in range(0, len(chunk_text) - w + 1, max(1, w // 4)):
            candidate = chunk_text[start : start + w]
            ratio = SequenceMatcher(None, sentence, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = start + w

    if best_ratio >= threshold:
        return f"{best_start}:{best_end}"

    # 3. Sentinel
    return "0:0"


def annotate_relationships(
    relationships: list[dict], chunk_id: str, chunk_text: str
) -> list[dict]:
    """Add evidence_chunk_ids and evidence_char_spans to each relationship."""
    for rel in relationships:
        sentence = rel.get("source_sentence", "")
        span = locate_sentence(chunk_text, sentence)
        rel["evidence_chunk_ids"] = [chunk_id]
        rel["evidence_char_spans"] = [span]
    return relationships
