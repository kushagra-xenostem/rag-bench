from ragchat import llm
from conftest import ndjson

DOC = (b"The Zephyr-9 battery lasts 14 hours on a single charge. "
       b"It charges fully in 90 minutes using the USB-C port. "
       b"The warranty period is two years from purchase.")


def upload(client, name="manual.txt", data=DOC, chunk_size=200):
    return client.post("/documents", files=[("files", (name, data))], data={"chunk_size": chunk_size}).json()


def ask(client, q, sid="s1", **kw):
    return ndjson(client.post("/chat", json={"session_id": sid, "question": q, **kw}))


def test_health(make_client):
    c, _ = make_client()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["documents"] == 0 and h["semantic_available"] is True


def test_upload_list_get_delete(make_client):
    c, _ = make_client()
    res = upload(c)
    assert res["errors"] == [] and res["added"][0]["n_chunks"] == 1
    doc_id = res["added"][0]["id"]
    assert [d["id"] for d in c.get("/documents").json()] == [doc_id]
    detail = c.get(f"/documents/{doc_id}").json()
    assert detail["chunks"][0]["text"].startswith("The Zephyr-9")
    assert c.delete(f"/documents/{doc_id}").status_code == 200
    assert c.get("/documents").json() == []
    assert c.delete(f"/documents/{doc_id}").status_code == 404
    assert c.get(f"/documents/{doc_id}").status_code == 404


def test_upload_reports_bad_files(make_client):
    c, _ = make_client()
    res = c.post("/documents", files=[("files", ("x.exe", b"MZ")), ("files", ("e.txt", b"   ")),
                                      ("files", ("bad.pdf", b"not a pdf"))]).json()
    assert res["added"] == [] and len(res["errors"]) == 3


def test_duplicate_upload_is_rejected(make_client):
    c, _ = make_client()
    upload(c)
    res = upload(c, name="copy.txt", data=DOC.replace(b". ", b".  "))
    assert res["added"] == [] and "already indexed 'manual.txt'" in res["errors"][0]["error"]


def test_upload_rejects_bad_chunk_size(make_client):
    c, _ = make_client()
    assert c.post("/documents", files=[("files", ("a.txt", DOC))], data={"chunk_size": 5}).status_code == 422


def test_chat_streams_sources_tokens_done(make_client):
    c, calls = make_client()
    upload(c)
    ev = ask(c, "How long does the battery last?", method="bm25", rerank=False)
    assert ev[0]["type"] == "sources" and ev[0]["sources"][0]["doc_name"] == "manual.txt"
    assert ev[0]["sources"][0]["method"] == "bm25"
    assert "".join(e["text"] for e in ev if e["type"] == "token") == "The battery lasts 14 hours [1]."
    assert ev[-1]["type"] == "done" and ev[-1]["cited"] == [1] and ev[-1]["latency_ms"] >= 0
    assert "14 hours" in calls["stream"][0][-1]["content"]


def test_chat_refuses_without_llm_when_nothing_retrieved(make_client):
    c, calls = make_client()
    upload(c)
    ev = ask(c, "quantum chromodynamics", method="bm25")
    assert ev[0]["sources"] == [] and ev[1]["text"] == llm.REFUSAL
    assert calls["stream"] == []


def test_all_methods_and_rerank(make_client):
    c, _ = make_client()
    upload(c)
    upload(c, "zoo.txt", b"The giraffe is the tallest land animal on earth.")
    for method in ("bm25", "semantic", "hybrid"):
        for rerank in (False, True):
            src = ask(c, "tallest giraffe", sid=f"{method}{rerank}", method=method, rerank=rerank)[0]["sources"]
            assert src[0]["doc_name"] == "zoo.txt", (method, rerank)
            assert src[0]["method"] == method + ("+rerank" if rerank else "")


def test_semantic_falls_back_to_bm25_without_embedder(make_client):
    c, _ = make_client(embedder=None, reranker=None)
    upload(c)
    assert c.get("/health").json()["semantic_available"] is False
    src = ask(c, "battery", method="semantic", rerank=False)[0]["sources"]
    assert src and src[0]["method"] == "bm25"


def test_followup_is_rewritten_for_retrieval(make_client):
    c, calls = make_client()
    upload(c)
    ask(c, "battery hours?")
    ev = ask(c, "and how long is it covered?")
    assert ev[0] == {"type": "rewrite", "query": "Zephyr-9 warranty period"}
    hist = c.get("/sessions/s1").json()["messages"]
    assert hist[-1]["meta"]["retrieval_query"] == "Zephyr-9 warranty period"


def test_rewrite_disabled_and_first_turn_not_rewritten(make_client):
    c, calls = make_client()
    upload(c)
    ask(c, "battery hours?")
    ask(c, "warranty?", rewrite=False)
    assert all("standalone" not in m[0]["content"] for m in calls["complete"])


def test_rewrite_falls_back_when_llm_fails(make_client):
    def broken(msgs, **kw):
        raise llm.LLMError("down")

    c, _ = make_client(complete_fn=broken)
    upload(c)
    ask(c, "battery hours?")
    ev = ask(c, "warranty?")
    assert ev[0] == {"type": "rewrite", "query": "battery hours? warranty?"}


def test_history_and_rolling_summary(make_client):
    c, calls = make_client()
    upload(c)
    for i in range(6):
        ask(c, f"battery question {i}")
    sess = c.get("/sessions/s1").json()
    assert sess["summary"].startswith("- user asked")
    assert sess["summarized_upto"] >= 4
    last_prompt = calls["stream"][-1]
    assert any(m["role"] == "system" and "Summary of the earlier" in m["content"] for m in last_prompt)
    assert sum(1 for m in last_prompt if m["role"] in ("user", "assistant")) == 7


def test_sessions_are_isolated_and_clearable(make_client):
    c, _ = make_client()
    upload(c)
    ask(c, "battery", sid="a")
    assert c.get("/sessions/b").json()["messages"] == []
    c.delete("/sessions/a")
    assert c.get("/sessions/a").json()["messages"] == []


def test_feedback(make_client):
    c, _ = make_client()
    upload(c)
    mid = ask(c, "battery", sid="s")[-1]["message_id"]
    assert c.post(f"/feedback/{mid}", json={"rating": -1, "note": "too short"}).status_code == 200
    msg = c.get("/sessions/s").json()["messages"][1]
    assert msg["rating"] == -1 and msg["note"] == "too short"
    assert c.post("/feedback/nope", json={"rating": 1}).status_code == 404
    user_mid = c.get("/sessions/s").json()["messages"][0]["id"]
    assert c.post(f"/feedback/{user_mid}", json={"rating": 1}).status_code == 404
    assert c.post(f"/feedback/{mid}", json={"rating": 5}).status_code == 422


def test_llm_error_is_streamed_not_crashed(make_client):
    def broken(msgs):
        raise llm.LLMError("server down")
        yield

    c, _ = make_client(stream_fn=broken)
    upload(c)
    ev = ask(c, "battery")
    assert any(e["type"] == "error" and "server down" in e["error"] for e in ev)


def test_citations_out_of_range_ignored(make_client):
    c, _ = make_client(stream_fn=lambda m: iter(["See [1] and [9]."]))
    upload(c)
    assert ask(c, "battery")[-1]["cited"] == [1]


def test_request_validation(make_client):
    c, _ = make_client()
    bad = [{"top_k": 0}, {"question": ""}, {"method": "magic"}]
    for patch in bad:
        body = {"session_id": "s", "question": "x", **patch}
        assert c.post("/chat", json=body).status_code == 422


def test_index_updates_after_new_upload(make_client):
    c, _ = make_client()
    upload(c)
    assert ask(c, "giraffe", sid="a", method="bm25")[0]["sources"] == []
    upload(c, "zoo.txt", b"The giraffe is the tallest land animal.")
    assert ask(c, "giraffe", sid="b", method="bm25")[0]["sources"][0]["doc_name"] == "zoo.txt"


def test_build_messages_marks_passages_untrusted():
    msgs = llm.build_messages("q", [{"doc_name": "d", "text": "t"}], [], summary="- s")
    assert "untrusted" in msgs[0]["content"] and "[1] (from d)" in msgs[-1]["content"]
    assert msgs[1]["content"].endswith("- s")


def test_clean_rewrite():
    assert llm.clean_rewrite('Query: "battery life"', "fb") == "battery life"
    assert llm.clean_rewrite("<think>x</think>\nwarranty length\nextra", "fb") == "warranty length"
    assert llm.clean_rewrite("", "fb") == "fb"
    assert llm.clean_rewrite("x" * 400, "fb") == "fb"
