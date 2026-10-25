import re

_WORD = re.compile(r"\S+")


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into chunks of ~`size` words with `overlap` words carried over."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = _WORD.findall(text)
    if not words:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks
