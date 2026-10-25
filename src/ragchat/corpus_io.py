"""Portable corpus format: a zip with manifest.json + documents.jsonl, or a bare JSONL file.

Each JSONL line is {"name": str, "text": str, "chunk_size"?: int}. Any tool can produce this format,
so corpora can be moved between installations without depending on another system."""
import io
import json
import time
import zipfile

FORMAT = "ragchat-corpus"
FORMAT_VERSION = 1
MAX_IMPORT_DOCS = 5000


class InvalidCorpus(ValueError):
    pass


def export_zip(documents: list[dict], corpus_version: int) -> bytes:
    lines = [
        json.dumps({"name": d["name"], "text": d["text"], "chunk_size": d["chunk_size"], "sha256": d["sha256"]},
                   ensure_ascii=False)
        for d in documents
    ]
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "corpus_version": corpus_version,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "documents": len(documents),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("documents.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    return buf.getvalue()


def _parse_jsonl(text: str) -> list[dict]:
    docs = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise InvalidCorpus(f"Line {n}: invalid JSON ({e.msg})") from e
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(obj.get("text"), str):
            raise InvalidCorpus(f"Line {n}: each line needs string fields 'name' and 'text'")
        cs = obj.get("chunk_size")
        docs.append({"name": obj["name"][:200] or "untitled", "text": obj["text"],
                     "chunk_size": cs if isinstance(cs, int) else None})
        if len(docs) > MAX_IMPORT_DOCS:
            raise InvalidCorpus(f"More than {MAX_IMPORT_DOCS} documents")
    return docs


def parse_import(filename: str, data: bytes) -> list[dict]:
    if filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "documents.jsonl" not in z.namelist():
                    raise InvalidCorpus("Zip must contain documents.jsonl")
                if z.getinfo("documents.jsonl").file_size > 500 * 1024 * 1024:
                    raise InvalidCorpus("documents.jsonl is too large")
                return _parse_jsonl(z.read("documents.jsonl").decode("utf-8"))
        except zipfile.BadZipFile as e:
            raise InvalidCorpus("Not a valid zip file") from e
    if filename.lower().endswith((".jsonl", ".json")):
        return _parse_jsonl(data.decode("utf-8", errors="replace"))
    raise InvalidCorpus("Import accepts .zip (exported corpus) or .jsonl")
