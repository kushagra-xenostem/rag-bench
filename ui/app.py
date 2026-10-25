import os
import json
import uuid
import httpx
import pandas as pd
import streamlit as st
from pathlib import Path

BACKEND = os.getenv("BACKEND_URL", f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8004')}")
SAMPLE_EVAL = Path(__file__).resolve().parents[1] / "samples" / "eval_set.jsonl"
METHODS = {"Hybrid (keyword + semantic)": "hybrid", "Keyword (BM25)": "bm25", "Semantic (embeddings)": "semantic"}

st.set_page_config(page_title="RAG Chat Hub", layout="wide")


def api(method: str, path: str, **kw) -> httpx.Response:
    return httpx.request(method, f"{BACKEND}{path}", timeout=600, **kw)


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []
sid = st.session_state.session_id

# ======================= sidebar =======================
with st.sidebar:
    st.subheader("Documents")
    try:
        health = api("GET", "/health", params={"check_llm": True}).json()
    except httpx.HTTPError:
        st.error(f"Backend not reachable at {BACKEND}")
        st.stop()
    llm_state = "connected" if health.get("llm") else "unreachable"
    models = health["models"]
    st.caption(f"LLM: {llm_state}  |  Embedding models: {models}  |  Corpus v{health['corpus_version']}")
    if not health.get("llm"):
        st.warning("The LLM server is unreachable. Check LLM_BASE_URL in .env.")

    chunk_size = st.select_slider("Chunk size (words)", options=[60, 100, 200, 400, 800], value=200)
    uploads = st.file_uploader("PDF, DOCX, TXT or Markdown", type=["pdf", "docx", "txt", "md", "markdown"],
                               accept_multiple_files=True)
    if st.button("Index documents", type="primary", disabled=not uploads, use_container_width=True):
        with st.spinner("Extracting, chunking and embedding..."):
            files = [("files", (f.name, f.getvalue())) for f in uploads]
            res = api("POST", "/documents", files=files, data={"chunk_size": chunk_size}).json()
        for d in res["added"]:
            st.success(f"{d['name']}: {d['n_chunks']} chunks")
        for e in res["errors"]:
            st.error(f"{e['file']}: {e['error']}")

    st.subheader("Retrieval")
    method_label = st.selectbox("Method", list(METHODS), disabled=not health["semantic_available"],
                                help="Semantic and hybrid need the local embedding model.")
    method = METHODS[method_label] if health["semantic_available"] else "bm25"
    rerank = st.toggle("Cross-encoder reranking", value=True,
                       help="Re-scores candidates with a local cross-encoder (about 0.2 s per question).")
    top_k = st.slider("Passages per answer (top-K)", 1, 10, 5)
    rewrite = st.toggle("Rewrite follow-up questions", value=True,
                        help="Uses the LLM to turn follow-ups like 'and how long is it?' into standalone queries.")

    st.subheader("Conversation")
    c1, c2 = st.columns(2)
    if c1.button("New chat", use_container_width=True):
        api("DELETE", f"/sessions/{sid}")
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()
    c2.download_button("Export", data=json.dumps(st.session_state.messages, indent=2),
                       file_name=f"chat_{sid}.json", mime="application/json", use_container_width=True)

docs = api("GET", "/documents").json()
st.title("RAG Chat Hub")
tab_chat, tab_lib, tab_ins, tab_eval = st.tabs(["Chat", "Library", "Insights", "Evaluation"])


# ======================= chat =======================
def render_sources(sources: list[dict], cited: list[int] | None = None) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            tag = "  |  cited" if cited and s["n"] in cited else ""
            st.markdown(f"**[{s['n']}] {s['doc_name']}** | {s.get('method', '')} score {s['score']}{tag}")
            st.caption(s["text"][:700] + ("..." if len(s["text"]) > 700 else ""))


def render_feedback(msg: dict, idx: int) -> None:
    if not msg.get("message_id"):
        return
    c1, c2, c3 = st.columns([1.2, 1.4, 8])
    if c1.button("Helpful", key=f"up{idx}", type="primary" if msg.get("rating") == 1 else "secondary"):
        api("POST", f"/feedback/{msg['message_id']}", json={"rating": 1})
        msg["rating"] = 1
        st.rerun()
    if c2.button("Not helpful", key=f"down{idx}", type="primary" if msg.get("rating") == -1 else "secondary"):
        api("POST", f"/feedback/{msg['message_id']}", json={"rating": -1})
        msg["rating"] = -1
        st.rerun()
    if msg.get("rating") == -1 and not msg.get("note"):
        with c3.form(f"note{idx}", border=False):
            n1, n2 = st.columns([5, 1])
            note = n1.text_input("What was wrong?", label_visibility="collapsed", placeholder="What was wrong?")
            if n2.form_submit_button("Send") and note.strip():
                api("POST", f"/feedback/{msg['message_id']}", json={"rating": -1, "note": note.strip()})
                msg["note"] = note.strip()
                st.rerun()


with tab_chat:
    st.caption("Answers come only from your indexed documents and cite the passages they use.")
    if not docs:
        st.info("Upload and index at least one document in the sidebar to start. "
                "Sample documents are in the samples/ folder.")
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("rewrite"):
                    st.caption(f"Searched for: {m['rewrite']}")
                render_sources(m.get("sources", []), m.get("cited"))
                render_feedback(m, i)

    question = st.chat_input("Ask about your documents...", disabled=not docs)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            state = {"sources": [], "cited": [], "message_id": None, "error": None, "rewrite": None}

            def tokens():
                body = {"session_id": sid, "question": question, "top_k": top_k, "method": method,
                        "rerank": rerank, "rewrite": rewrite}
                with httpx.stream("POST", f"{BACKEND}/chat", json=body, timeout=600) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        ev = json.loads(line)
                        if ev["type"] == "rewrite":
                            state["rewrite"] = ev["query"]
                        elif ev["type"] == "sources":
                            state["sources"] = ev["sources"]
                        elif ev["type"] == "token":
                            yield ev["text"]
                        elif ev["type"] == "error":
                            state["error"] = ev["error"]
                        elif ev["type"] == "done":
                            state["message_id"], state["cited"] = ev["message_id"], ev["cited"]

            with st.spinner("Searching documents..."):
                answer = st.write_stream(tokens())
            if state["error"]:
                st.error(state["error"])
        st.session_state.messages.append({
            "role": "assistant", "content": answer if isinstance(answer, str) else "",
            "sources": state["sources"], "cited": state["cited"], "message_id": state["message_id"],
            "rewrite": state["rewrite"],
        })
        st.rerun()

# ======================= library =======================
with tab_lib:
    corpus = api("GET", "/corpus").json()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Corpus version", corpus["version"])
    m2.metric("Documents", corpus["documents"])
    m3.metric("Chunks", corpus["chunks"])
    m4.metric("Characters", f"{corpus['characters']:,}")

    st.subheader("Documents")
    if not docs:
        st.caption("No documents indexed yet.")
    for d in docs:
        with st.expander(f"{d['name']} | {d['n_chunks']} chunks of {d['chunk_size']} words"):
            c1, c2, c3 = st.columns([2, 1, 1])
            new_size = c1.select_slider("Re-chunk with size", options=[60, 100, 200, 400, 800],
                                        value=d["chunk_size"] if d["chunk_size"] in (60, 100, 200, 400, 800) else 200,
                                        key=f"cs_{d['id']}")
            if c2.button("Re-index", key=f"ri_{d['id']}", use_container_width=True):
                api("POST", f"/documents/{d['id']}/reindex", json={"chunk_size": new_size})
                st.rerun()
            if c3.button("Delete", key=f"del_{d['id']}", use_container_width=True):
                api("DELETE", f"/documents/{d['id']}")
                st.rerun()
            if st.checkbox("Preview chunks", key=f"pv_{d['id']}"):
                for ch in api("GET", f"/documents/{d['id']}").json()["chunks"][:20]:
                    st.caption(f"Chunk {ch['position'] + 1}: {ch['text'][:400]}")

    st.subheader("Export and import")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Export the full corpus (text, chunk sizes, checksums) as a portable zip.")
        if st.button("Prepare export", disabled=not docs):
            st.session_state.export = api("GET", "/corpus/export").content
        if st.session_state.get("export"):
            st.download_button("Download corpus zip", st.session_state.export,
                               file_name=f"corpus_v{corpus['version']}.zip", mime="application/zip")
    with c2:
        st.caption('Import an exported zip, or a .jsonl file with one {"name", "text"} object per line.')
        imp = st.file_uploader("Corpus file", type=["zip", "jsonl"], key="imp", label_visibility="collapsed")
        if imp and st.button("Import"):
            res = api("POST", "/corpus/import", files={"file": (imp.name, imp.getvalue())},
                      data={"chunk_size": chunk_size})
            if res.status_code != 200:
                st.error(res.json().get("detail"))
            else:
                st.success(f"Imported {len(res.json()['added'])} documents, skipped {len(res.json()['errors'])}.")

    st.subheader("Version history")
    if corpus["log"]:
        log = pd.DataFrame(corpus["log"])[["version", "action", "doc_name", "detail", "created"]]
        log["created"] = pd.to_datetime(log["created"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(log, hide_index=True, use_container_width=True)

# ======================= insights =======================
with tab_ins:
    rep = api("GET", "/analytics").json()
    t = rep["totals"]

    def pct(v):
        return "n/a" if v is None else f"{v:.0%}"

    m = st.columns(6)
    m[0].metric("Questions", t["questions"])
    m[1].metric("Answered", t["answered"])
    m[2].metric("Not found", t["not_found"])
    m[3].metric("Helpful rate", pct(t["helpful_rate"]), help=f"{t['ratings']} ratings")
    m[4].metric("Latency p50", "n/a" if t["latency_ms_p50"] is None else f"{t['latency_ms_p50'] / 1000:.1f} s")
    m[5].metric("Latency p95", "n/a" if t["latency_ms_p95"] is None else f"{t['latency_ms_p95'] / 1000:.1f} s")

    st.subheader("Recommendations")
    for r in rep["recommendations"]:
        st.markdown(f"- {r}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Documents")
        if rep["documents"]:
            st.dataframe(pd.DataFrame(rep["documents"]), hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Retrieval settings by rating")
        if rep["retrieval_settings"]:
            st.dataframe(pd.DataFrame(rep["retrieval_settings"]), hide_index=True, use_container_width=True)
        else:
            st.caption("No rated answers yet.")
    st.subheader("Content gaps (questions the documents could not answer)")
    if rep["content_gaps"]:
        st.dataframe(pd.DataFrame(rep["content_gaps"])[["question"]], hide_index=True, use_container_width=True)
    else:
        st.caption("None so far.")
    st.subheader("Answers rated not helpful")
    if rep["low_rated"]:
        st.dataframe(pd.DataFrame(rep["low_rated"])[["question", "answer", "note"]], hide_index=True,
                     use_container_width=True)
    else:
        st.caption("None so far.")
    st.download_button("Download report (JSON)", json.dumps(rep, indent=2), file_name="feedback_report.json",
                       mime="application/json")

# ======================= evaluation =======================
with tab_eval:
    st.caption("Measure retrieval and answer quality on a labelled question set (JSONL). "
               "samples/eval_set.jsonl matches the sample documents.")
    c1, c2 = st.columns([3, 1])
    eval_file = c1.file_uploader("Eval set", type=["jsonl"], key="evalset", label_visibility="collapsed")
    use_sample = c2.checkbox("Use sample eval set", value=eval_file is None)
    data = eval_file.getvalue() if eval_file and not use_sample else (
        SAMPLE_EVAL.read_bytes() if SAMPLE_EVAL.exists() else None)

    st.subheader("Retrieval comparison")
    if st.button("Run retrieval evaluation", disabled=data is None or not docs):
        with st.spinner("Scoring every retrieval configuration..."):
            res = api("POST", "/eval/retrieval", files={"file": ("eval.jsonl", data)}, data={"top_k": top_k})
        if res.status_code != 200:
            st.error(res.json().get("detail"))
        else:
            st.session_state.ret_eval = res.json()
    if st.session_state.get("ret_eval"):
        r = st.session_state.ret_eval
        df = pd.DataFrame(r["results"])
        df["missed"] = df["missed"].apply(lambda m: ", ".join(m) or "-")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(f"{r['items']} questions, corpus v{r['corpus_version']}. A hit means the labelled evidence "
                   "sentence was inside a retrieved passage.")

    st.subheader("End-to-end answers")
    st.caption(f"Uses the current sidebar settings ({method}{' + rerank' if rerank else ''}, top-K {top_k}) "
               "and calls the LLM once per question.")
    if st.button("Run answer evaluation", disabled=data is None or not docs or not health.get("llm")):
        bar = st.progress(0.0, text="Starting...")
        report = None
        with httpx.stream("POST", f"{BACKEND}/eval/answers", files={"file": ("eval.jsonl", data)},
                          data={"method": method, "top_k": str(top_k), "rerank": str(rerank).lower()},
                          timeout=3600) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "progress":
                    bar.progress(ev["done"] / ev["total"], text=f"{ev['done']}/{ev['total']}: {ev['item']['question']}")
                elif ev["type"] == "report":
                    report = ev["report"]
        bar.empty()
        st.session_state.ans_eval = report
    if st.session_state.get("ans_eval"):
        r = st.session_state.ans_eval
        m = st.columns(5)
        m[0].metric("Answer accuracy", pct(r["answer_accuracy"]))
        m[1].metric("Refusal accuracy", pct(r["refusal_accuracy"]), help="Unanswerable questions correctly refused")
        m[2].metric("Citation rate", pct(r["citation_rate"]))
        m[3].metric("False refusals", r["false_refusals"])
        m[4].metric("Avg seconds", r["avg_seconds"])
        items = pd.DataFrame(r["items"])[["id", "question", "correct", "refused", "cited", "seconds", "answer"]]
        st.dataframe(items, hide_index=True, use_container_width=True)
        st.download_button("Download evaluation (JSON)", json.dumps(r, indent=2), file_name="answer_eval.json",
                           mime="application/json")
