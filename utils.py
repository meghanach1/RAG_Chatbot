# helpers (text split, cleaning)
"""
utils.py - helpers for text chunking
"""

import os
from dotenv import load_dotenv
load_dotenv()
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", 1500))

def chunk_text(text: str, max_chars: int = None, overlap: int = 200) -> list[str]:
    """
    Simple character-based chunker with overlap.
    - max_chars: max characters per chunk
    - overlap: number of overlapping characters between chunks
    """
    if max_chars is None:
        max_chars = MAX_CHUNK_CHARS

    text = text.replace("\r\n", "\n").strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
