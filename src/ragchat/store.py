import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, text TEXT NOT NULL,
    chunk_size INTEGER NOT NULL, n_chunks INTEGER NOT NULL, n_chars INTEGER NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB
);
CREATE TABLE IF NOT EXISTS corpus_log (
    version INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
    doc_id TEXT NOT NULL, doc_name TEXT NOT NULL, detail TEXT, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, summary TEXT, summarized_upto INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, sources TEXT, meta TEXT, created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created);
CREATE TABLE IF NOT EXISTS feedback (
    message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL, note TEXT, created REAL NOT NULL
);
"""


class DuplicateDocument(ValueError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def _blob(vec) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        current = self._db.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self._db.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        if has_tables and current != SCHEMA_VERSION:
            raise RuntimeError(f"Database {path} uses an older schema; delete it to start fresh.")
        self._db.executescript(SCHEMA)
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._lock = threading.RLock()

    # ---------- corpus ----------
    @property
    def version(self) -> int:
        row = self._db.execute("SELECT max(version) FROM corpus_log").fetchone()
        return row[0] or 0

    def _log(self, action: str, doc_id: str, name: str, detail: str | None = None) -> None:
        self._db.execute(
            "INSERT INTO corpus_log (action, doc_id, doc_name, detail, created) VALUES (?,?,?,?,?)",
            (action, doc_id, name, detail, time.time()),
        )

    def find_by_hash(self, sha: str) -> dict | None:
        row = self._db.execute("SELECT id, name FROM documents WHERE sha256=?", (sha,)).fetchone()
        return dict(row) if row else None

    def add_document(self, name: str, text: str, chunks: list[str], chunk_size: int,
                     embeddings: list | None = None) -> dict:
        sha = sha256_text(text)
        with self._lock, self._db:
            dup = self.find_by_hash(sha)
            if dup:
                raise DuplicateDocument(f"Same content as already indexed '{dup['name']}'")
            doc_id = uuid.uuid4().hex[:12]
            self._db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, name, sha, text, chunk_size, len(chunks), len(text), time.time()),
            )
            self._insert_chunks(doc_id, chunks, embeddings)
            self._log("add", doc_id, name, f"{len(chunks)} chunks @ {chunk_size} words")
        return self.get_document(doc_id)

    def _insert_chunks(self, doc_id: str, chunks: list[str], embeddings: list | None) -> None:
        embs = embeddings if embeddings is not None else [None] * len(chunks)
        self._db.executemany(
            "INSERT INTO chunks (doc_id, position, text, embedding) VALUES (?,?,?,?)",
            [(doc_id, i, c, _blob(e)) for i, (c, e) in enumerate(zip(chunks, embs))],
        )

    def replace_chunks(self, doc_id: str, chunks: list[str], chunk_size: int, embeddings: list | None) -> dict:
        with self._lock, self._db:
            doc = self.get_document(doc_id)
            self._db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._insert_chunks(doc_id, chunks, embeddings)
            self._db.execute("UPDATE documents SET chunk_size=?, n_chunks=? WHERE id=?",
                             (chunk_size, len(chunks), doc_id))
            self._log("reindex", doc_id, doc["name"], f"{len(chunks)} chunks @ {chunk_size} words")
        return self.get_document(doc_id)

    def get_document(self, doc_id: str, with_text: bool = False) -> dict | None:
        cols = "*" if with_text else "id, name, sha256, chunk_size, n_chunks, n_chars, created"
        row = self._db.execute(f"SELECT {cols} FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, with_text: bool = False) -> list[dict]:
        cols = "*" if with_text else "id, name, sha256, chunk_size, n_chunks, n_chars, created"
        return [dict(r) for r in self._db.execute(f"SELECT {cols} FROM documents ORDER BY created")]

    def document_chunks(self, doc_id: str) -> list[dict]:
        rows = self._db.execute("SELECT position, text FROM chunks WHERE doc_id=? ORDER BY position", (doc_id,))
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._db:
            doc = self.get_document(doc_id)
            if not doc:
                return False
            self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self._log("delete", doc_id, doc["name"])
        return True

    def corpus_log(self, limit: int = 100) -> list[dict]:
        rows = self._db.execute("SELECT * FROM corpus_log ORDER BY version DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def all_chunks(self) -> tuple[list[dict], np.ndarray | None]:
        sql = (
            "SELECT c.id, c.doc_id, c.position, c.text, c.embedding, d.name AS doc_name "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY d.created, c.position"
        )
        rows = [dict(r) for r in self._db.execute(sql)]
        blobs = [r.pop("embedding") for r in rows]
        if rows and all(b is not None for b in blobs):
            return rows, np.vstack([np.frombuffer(b, dtype=np.float32) for b in blobs])
        return rows, None

    def chunks_missing_embeddings(self) -> list[dict]:
        rows = self._db.execute("SELECT id, text FROM chunks WHERE embedding IS NULL")
        return [dict(r) for r in rows]

    def set_embeddings(self, pairs: list[tuple[int, object]]) -> None:
        with self._lock, self._db:
            self._db.executemany("UPDATE chunks SET embedding=? WHERE id=?", [(_blob(v), i) for i, v in pairs])

    # ---------- sessions & messages ----------
    def add_message(self, session_id: str, role: str, content: str, sources: list | None = None,
                    meta: dict | None = None) -> str:
        mid = uuid.uuid4().hex[:12]
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))
            self._db.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                (mid, session_id, role, content, json.dumps(sources) if sources else None,
                 json.dumps(meta) if meta else None, time.time()),
            )
        return mid

    def history(self, session_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT m.*, f.rating, f.note FROM messages m LEFT JOIN feedback f ON f.message_id = m.id "
            "WHERE session_id=? ORDER BY created",
            (session_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"]) if d["sources"] else []
            d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
            out.append(d)
        return out

    def get_session(self, session_id: str) -> dict:
        row = self._db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else {"id": session_id, "summary": None, "summarized_upto": 0}

    def set_summary(self, session_id: str, summary: str, upto: int) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))
            self._db.execute("UPDATE sessions SET summary=?, summarized_upto=? WHERE id=?",
                             (summary, upto, session_id))

    def clear_session(self, session_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def set_feedback(self, message_id: str, rating: int, note: str | None) -> bool:
        row = self._db.execute("SELECT role FROM messages WHERE id=?", (message_id,)).fetchone()
        if not row or row["role"] != "assistant":
            return False
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)", (message_id, rating, note, time.time())
            )
        return True

    def all_turns(self) -> list[dict]:
        """Every assistant message joined with the user question that preceded it and its feedback."""
        rows = self._db.execute(
            "SELECT m.id, m.session_id, m.content, m.sources, m.meta, m.created, f.rating, f.note, "
            "(SELECT q.content FROM messages q WHERE q.session_id = m.session_id AND q.role = 'user' "
            " AND q.created <= m.created ORDER BY q.created DESC LIMIT 1) AS question "
            "FROM messages m LEFT JOIN feedback f ON f.message_id = m.id "
            "WHERE m.role = 'assistant' ORDER BY m.created"
        )
        out = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"]) if d["sources"] else []
            d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
            out.append(d)
        return out
