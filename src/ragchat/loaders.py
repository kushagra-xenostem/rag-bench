import io
from pathlib import PurePath

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".docx"}
MAX_BYTES = 25 * 1024 * 1024


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFile(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED))}")
    if len(data) > MAX_BYTES:
        raise UnsupportedFile("File is larger than 25 MB")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n\n".join(pages)
    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return data.decode("utf-8", errors="replace")
