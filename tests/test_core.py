import io

import pytest

from ragchat.bm25 import BM25, tokenize
from ragchat.chunking import chunk_text
from ragchat.loaders import UnsupportedFile, extract_text


def test_tokenize_drops_stopwords_and_punct():
    assert tokenize("What is the Capital of France?") == ["capital", "france"]


def test_bm25_ranks_relevant_first():
    idx = BM25(["cats purr loudly", "dogs bark at night", "the stock market fell"])
    assert idx.search("why do dogs bark")[0].index == 1


def test_bm25_no_match_returns_empty():
    assert BM25(["alpha beta"]).search("zeta") == []


def test_bm25_ties_break_by_index():
    idx = BM25(["same text here", "same text here", "other"])
    assert [h.index for h in idx.search("same text")] == [0, 1]


def test_bm25_empty_corpus():
    assert BM25([]).search("anything") == []


def test_bm25_respects_k():
    idx = BM25([f"apple {i}" for i in range(10)])
    assert len(idx.search("apple", k=3)) == 3


def test_chunk_overlap_and_coverage():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), size=10, overlap=2)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert chunks[-1].split()[-1] == "w24"


def test_chunk_empty_and_short():
    assert chunk_text("   ") == []
    assert chunk_text("one two", size=10, overlap=2) == ["one two"]


def test_chunk_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


def test_extract_txt_and_md():
    assert extract_text("a.txt", b"hello") == "hello"
    assert "Title" in extract_text("a.md", b"# Title")


def test_extract_docx():
    import docx

    d = docx.Document()
    d.add_paragraph("Quarterly revenue grew")
    buf = io.BytesIO()
    d.save(buf)
    assert "Quarterly revenue grew" in extract_text("r.docx", buf.getvalue())


def test_extract_rejects_unknown_type():
    with pytest.raises(UnsupportedFile):
        extract_text("x.exe", b"MZ")


# ---------- store & retrieval internals ----------
def test_store_rejects_old_schema(tmp_path):
    import sqlite3

    from ragchat.store import Store

    db = tmp_path / "old.db"
    sqlite3.connect(db).execute("CREATE TABLE documents (id TEXT)").connection.commit()
    with pytest.raises(RuntimeError, match="older schema"):
        Store(db)


def test_store_version_and_hash_normalisation(tmp_path):
    from ragchat.store import Store, sha256_text

    s = Store(tmp_path / "s.db")
    assert s.version == 0
    s.add_document("a", "x  y", ["x y"], 100)
    assert s.version == 1 and sha256_text("x y") == sha256_text(" x\n y ")


def test_warmup_backfills_missing_embeddings(tmp_path):
    from conftest import HashEmbedder

    from ragchat.api import create_app
    from ragchat.config import Settings
    from ragchat.store import Store

    db = tmp_path / "w.db"
    Store(db).add_document("a.txt", "giraffe facts", ["giraffe facts"], 100)  # stored with no embedding
    app = create_app(Settings("http://x/v1", "k", "m", db, 5), embedder=HashEmbedder(), use_local_models=False)
    app.state.warmup()
    assert Store(db).chunks_missing_embeddings() == []


def test_hybrid_is_deterministic(tmp_path):
    from conftest import HashEmbedder

    from ragchat.retrieval import RetrievalConfig, Retriever
    from ragchat.store import Store

    s = Store(tmp_path / "h.db")
    e = HashEmbedder()
    for i in range(5):
        txt = f"shared words here number{i}"
        s.add_document(f"d{i}", txt, [txt], 100, list(e.embed([txt])))
    r = Retriever(s, e)
    runs = {tuple(x["doc_name"] for x in r.search("shared words", RetrievalConfig("hybrid", 5))) for _ in range(3)}
    assert runs == {("d0", "d1", "d2", "d3", "d4")}
    with pytest.raises(ValueError):
        r.search("x", RetrievalConfig("nope"))
