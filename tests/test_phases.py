"""Phase 2 (analytics), 5 (corpus management) and 6 (evaluation lab)."""
import io
import json
import zipfile
from ragchat import llm
from conftest import ndjson
from ragchat.analytics import build_report
from ragchat.corpus_io import InvalidCorpus, export_zip, parse_import
from ragchat.evaluation import InvalidEvalSet, parse_eval_set, score_answer

DOC_A = b"The Zephyr-9 battery lasts 14 hours. The warranty period is two years."
DOC_B = b"The giraffe is the tallest land animal. Giraffes eat acacia leaves."


def setup_docs(c):
    c.post("/documents", files=[("files", ("a.txt", DOC_A)), ("files", ("b.txt", DOC_B))])


def chat(c, q, sid="s", **kw):
    return ndjson(c.post("/chat", json={"session_id": sid, "question": q, "method": "bm25", **kw}))


# ---------- Phase 2: analytics ----------
def test_analytics_empty(make_client):
    c, _ = make_client()
    r = c.get("/analytics").json()
    assert r["totals"]["questions"] == 0 and r["totals"]["helpful_rate"] is None
    assert any("Collect at least" in x for x in r["recommendations"])


def test_analytics_counts_ratings_gaps_and_documents(make_client):
    c, _ = make_client()
    setup_docs(c)
    m1 = chat(c, "battery hours", sid="1")[-1]["message_id"]
    m2 = chat(c, "battery warranty", sid="2")[-1]["message_id"]
    chat(c, "quantum physics", sid="3")
    c.post(f"/feedback/{m1}", json={"rating": 1})
    c.post(f"/feedback/{m2}", json={"rating": -1, "note": "wrong"})
    r = c.get("/analytics").json()
    t = r["totals"]
    assert (t["questions"], t["not_found"], t["thumbs_up"], t["thumbs_down"]) == (3, 1, 1, 1)
    assert t["helpful_rate"] == 0.5 and t["latency_ms_p50"] is not None
    assert r["content_gaps"][0]["question"] == "quantum physics"
    assert r["low_rated"][0]["note"] == "wrong"
    docs = {d["document"]: d for d in r["documents"]}
    assert docs["a.txt"]["times_cited"] == 2 and docs["b.txt"]["times_cited"] == 0
    assert r["retrieval_settings"][0]["config"] == "bm25+rerank / top_k=5"


def _turn(rating, method="bm25", rerank=False, content="ok", doc="d"):
    return {"id": "x", "question": "q", "content": content, "rating": rating, "note": None,
            "sources": [{"doc_name": doc}], "meta": {"method": method, "rerank": rerank, "top_k": 5}}


def test_report_recommends_best_setting_and_flags_bad_docs():
    turns = [_turn(1, "hybrid") for _ in range(5)] + [_turn(-1, "bm25", doc="old.pdf") for _ in range(5)]
    turns += [_turn(None, content=llm.REFUSAL) for _ in range(6)]
    r = build_report(turns, [{"name": "d"}, {"name": "old.pdf"}])
    recs = " ".join(r["recommendations"])
    assert "hybrid / top_k=5 (100% helpful over 5 ratings)" in recs
    assert "old.pdf" in recs and "not answerable" in recs


# ---------- Phase 5: corpus management ----------
def test_corpus_versioning_log(make_client):
    c, _ = make_client()
    setup_docs(c)
    doc = c.get("/documents").json()[0]
    c.post(f"/documents/{doc['id']}/reindex", json={"chunk_size": 50})
    c.delete(f"/documents/{doc['id']}")
    corp = c.get("/corpus").json()
    assert corp["version"] == 4 and corp["documents"] == 1
    assert [e["action"] for e in corp["log"]] == ["delete", "reindex", "add", "add"]


def test_reindex_changes_chunking_and_search_still_works(make_client):
    c, _ = make_client()
    text = " ".join(f"word{i}" for i in range(300)) + " giraffe"
    c.post("/documents", files=[("files", ("long.txt", text.encode()))], data={"chunk_size": 400})
    doc = c.get("/documents").json()[0]
    assert doc["n_chunks"] == 1
    new = c.post(f"/documents/{doc['id']}/reindex", json={"chunk_size": 100}).json()
    assert new["n_chunks"] == 4 and new["chunk_size"] == 100
    assert chat(c, "giraffe")[0]["sources"][0]["doc_name"] == "long.txt"
    assert c.post("/documents/nope/reindex", json={"chunk_size": 100}).status_code == 404
    assert c.post(f"/documents/{doc['id']}/reindex", json={"chunk_size": 10}).status_code == 422


def test_export_import_roundtrip(make_client, tmp_path):
    c, _ = make_client()
    setup_docs(c)
    r = c.get("/corpus/export")
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["documents"] == 2 and manifest["format"] == "ragchat-corpus"

    c2, _ = make_client()
    c2.post("/documents", files=[("files", ("a.txt", DOC_A))])
    res = c2.post("/corpus/import", files={"file": ("corpus.zip", r.content)}).json()
    assert [d["name"] for d in res["added"]] == ["b.txt"]
    assert "already indexed" in res["errors"][0]["error"]
    assert chat(c2, "giraffe")[0]["sources"][0]["doc_name"] == "b.txt"


def test_import_generic_jsonl(make_client):
    c, _ = make_client()
    data = "\n".join(json.dumps({"name": f"n{i}.txt", "text": f"note {i} about pandas"}) for i in range(3))
    res = c.post("/corpus/import", files={"file": ("notes.jsonl", data.encode())}).json()
    assert len(res["added"]) == 3


def test_import_rejects_bad_input(make_client):
    c, _ = make_client()
    for name, data in [("x.zip", b"nope"), ("x.jsonl", b'{"name": 1}'), ("x.csv", b"a,b")]:
        assert c.post("/corpus/import", files={"file": (name, data)}).status_code == 422


def test_parse_import_and_export_units():
    z = export_zip([{"name": "a", "text": "hello", "chunk_size": 100, "sha256": "h"}], 3)
    assert parse_import("c.zip", z) == [{"name": "a", "text": "hello", "chunk_size": 100}]
    try:
        parse_import("c.jsonl", b"not json")
        raise AssertionError
    except InvalidCorpus as e:
        assert "Line 1" in str(e)


# ---------- Phase 6: evaluation lab ----------
EVAL = "\n".join(json.dumps(x) for x in [
    {"id": "1", "question": "how long does the battery last", "evidence": "battery lasts 14 hours",
     "answer_keywords": ["14"]},
    {"id": "2", "question": "tallest land animal", "evidence": "giraffe is the tallest", "answer_keywords": ["giraffe"]},
    {"id": "3", "question": "what is the moon made of", "answerable": False},
]).encode()


def test_eval_retrieval_reports_all_configs(make_client):
    c, _ = make_client()
    setup_docs(c)
    r = c.post("/eval/retrieval", files={"file": ("e.jsonl", EVAL)}, data={"top_k": 3}).json()
    configs = {row["config"]: row for row in r["results"]}
    assert set(configs) == {"bm25", "semantic", "hybrid", "hybrid+rerank"}
    assert configs["bm25"]["hit@1"] == 1.0 and configs["bm25"]["mrr"] == 1.0
    assert configs["bm25"]["unanswerable_with_passages"] == "0/1"


def test_eval_answers_streams_progress_and_report(make_client):
    def fake(msgs):
        q = msgs[-1]["content"]
        yield "It lasts 14 hours [1]." if "battery" in q else "The giraffe [1]."

    c, _ = make_client(stream_fn=fake)
    setup_docs(c)
    ev = ndjson(c.post("/eval/answers", files={"file": ("e.jsonl", EVAL)}, data={"method": "bm25"}))
    assert [e["type"] for e in ev] == ["progress"] * 3 + ["report"]
    rep = ev[-1]["report"]
    assert rep["answer_accuracy"] == 1.0 and rep["refusal_accuracy"] == 1.0 and rep["citation_rate"] == 1.0


def test_eval_rejects_invalid_set(make_client):
    c, _ = make_client()
    r = c.post("/eval/retrieval", files={"file": ("e.jsonl", b'{"question": "x"}')})
    assert r.status_code == 422 and "evidence" in r.json()["detail"]
    r = c.post("/eval/answers", files={"file": ("e.jsonl", EVAL)}, data={"method": "magic"})
    assert r.status_code == 422


def test_score_answer_rules():
    item = {"answerable": True, "answer_keywords": ["14", "hours"]}
    assert score_answer(item, "14 hours [1]")["correct"]
    assert not score_answer(item, "14 days [1]")["correct"]
    assert not score_answer(item, llm.REFUSAL)["correct"]
    na = {"answerable": False, "answer_keywords": []}
    assert score_answer(na, llm.REFUSAL)["correct"] and not score_answer(na, "It is cheese.")["correct"]


def test_parse_eval_set_errors():
    for bad in ["", "not json", '{"answerable": true}']:
        try:
            parse_eval_set(bad)
            raise AssertionError(bad)
        except InvalidEvalSet:
            pass


def test_sample_eval_set_is_valid():
    from conftest import ROOT

    items = parse_eval_set((ROOT / "samples" / "eval_set.jsonl").read_text(encoding="utf-8"))
    assert len(items) == 24 and sum(not i["answerable"] for i in items) == 4
    texts = " ".join((ROOT / "samples" / f).read_text(encoding="utf-8")
                     for f in ("employee_handbook.md", "travel_policy.txt", "aurora_x2_manual.md"))
    norm = " ".join(texts.split())
    for i in items:
        if i["answerable"]:
            assert i["evidence"] in norm, i["id"]
