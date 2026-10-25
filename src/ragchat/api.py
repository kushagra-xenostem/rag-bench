import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from collections.abc import Callable, Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import corpus_io, evaluation, llm
from .analytics import build_report
from .chunking import chunk_text
from .config import Settings, get_settings
from .embed import Embedder, FastEmbedder, FastReranker, Reranker
from .loaders import UnsupportedFile, extract_text
from .retrieval import METHODS, RetrievalConfig, Retriever
from .store import DuplicateDocument, Store

log = logging.getLogger("ragchat")
_CITE = re.compile(r"\[(\d+)\]")
RECENT_MESSAGES = 6  # last 3 turns go verbatim into the prompt
SUMMARY_BATCH = 4    # summarise older messages once this many are unsummarised


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    method: str = Field(default="hybrid", pattern="^(bm25|semantic|hybrid)$")
    rerank: bool = True
    rewrite: bool = True


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1)
    note: str | None = Field(default=None, max_length=1000)


class ReindexRequest(BaseModel):
    chunk_size: int = Field(ge=50, le=1000)


def create_app(settings: Settings | None = None, stream_fn: Callable | None = None,
               complete_fn: Callable | None = None, embedder: Embedder | None = None,
               reranker: Reranker | None = None, use_local_models: bool | None = None) -> FastAPI:
    settings = settings or get_settings()
    if use_local_models is None:
        use_local_models = os.getenv("RAGCHAT_DISABLE_MODELS", "0") != "1"
    if use_local_models:
        embedder = embedder or FastEmbedder()
        reranker = reranker or FastReranker()
    store = Store(settings.db_path)
    retriever = Retriever(store, embedder, reranker)
    stream = stream_fn or (lambda msgs: llm.stream_chat(settings, msgs))
    complete = complete_fn or (lambda msgs, **kw: llm.complete(settings, msgs, **kw))
    status = {"models": "ready" if embedder is None else "loading", "models_error": None}
    def embed_texts(texts: list[str]):
        if embedder is None:
            return None
        try:
            return list(embedder.embed(texts))
        except Exception as e:  # model download/load problems must not block keyword search
            log.warning("embedding failed, falling back to keyword search: %s", e)
            status["models_error"] = str(e)
            return None

    def warmup() -> None:
        try:
            missing = store.chunks_missing_embeddings()
            if embedder is not None:
                embedder.embed_query("warm up")
                if missing:
                    vecs = embedder.embed([m["text"] for m in missing])
                    store.set_embeddings([(m["id"], v) for m, v in zip(missing, vecs)])
                    retriever._version = -1
            if reranker is not None:
                reranker.score("warm up", ["warm up"])
            status["models"] = "ready"
        except Exception as e:
            status["models"], status["models_error"] = "unavailable", str(e)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if embedder is not None or reranker is not None:
            threading.Thread(target=warmup, daemon=True).start()
        yield

    app = FastAPI(title="RAG Chat Hub", version="1.0.0", lifespan=lifespan)
    app.state.warmup = warmup

    def index_document(name: str, text: str, chunk_size: int) -> dict:
        chunks = chunk_text(text, size=chunk_size, overlap=chunk_size // 5)
        if not chunks:
            raise UnsupportedFile("No extractable text (scanned PDF?)")
        return store.add_document(name, text, chunks, chunk_size, embed_texts(chunks))

    # ---------- health ----------
    @app.get("/health")
    def health(check_llm: bool = False) -> dict:
        out = {"status": "ok", "documents": len(store.list_documents()), "corpus_version": store.version,
               "models": status["models"], "semantic_available": retriever.semantic_available()}
        if status["models_error"]:
            out["models_error"] = status["models_error"][:300]
        if check_llm:
            out["llm"] = llm.ping(settings)
        return out

    # ---------- documents ----------
    @app.post("/documents")
    async def upload(files: list[UploadFile] = File(...), chunk_size: int = Form(200)) -> dict:
        if not 50 <= chunk_size <= 1000:
            raise HTTPException(422, "chunk_size must be between 50 and 1000 words")
        added, errors = [], []
        for f in files:
            name = f.filename or "untitled"
            try:
                text = extract_text(name, await f.read())
                added.append(index_document(name, text, chunk_size))
            except (UnsupportedFile, DuplicateDocument) as e:
                errors.append({"file": name, "error": str(e)})
            except Exception as e:  # corrupt PDFs/DOCX raise many library-specific errors
                errors.append({"file": name, "error": f"Could not read file: {e}"})
        return {"added": added, "errors": errors}

    @app.get("/documents")
    def list_documents() -> list[dict]:
        return store.list_documents()

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str) -> dict:
        doc = store.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "document not found")
        return {**doc, "chunks": store.document_chunks(doc_id)}

    @app.post("/documents/{doc_id}/reindex")
    def reindex(doc_id: str, req: ReindexRequest) -> dict:
        doc = store.get_document(doc_id, with_text=True)
        if not doc:
            raise HTTPException(404, "document not found")
        chunks = chunk_text(doc["text"], size=req.chunk_size, overlap=req.chunk_size // 5)
        return store.replace_chunks(doc_id, chunks, req.chunk_size, embed_texts(chunks))

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict:
        if not store.delete_document(doc_id):
            raise HTTPException(404, "document not found")
        return {"deleted": doc_id}

    # ---------- corpus management ----------
    @app.get("/corpus")
    def corpus() -> dict:
        docs = store.list_documents()
        return {"version": store.version, "documents": len(docs), "chunks": sum(d["n_chunks"] for d in docs),
                "characters": sum(d["n_chars"] for d in docs), "log": store.corpus_log()}

    @app.get("/corpus/export")
    def export_corpus() -> Response:
        data = corpus_io.export_zip(store.list_documents(with_text=True), store.version)
        name = f"corpus_v{store.version}.zip"
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/corpus/import")
    async def import_corpus(file: UploadFile = File(...), chunk_size: int = Form(200)) -> dict:
        if not 50 <= chunk_size <= 1000:
            raise HTTPException(422, "chunk_size must be between 50 and 1000 words")
        try:
            docs = corpus_io.parse_import(file.filename or "", await file.read())
        except corpus_io.InvalidCorpus as e:
            raise HTTPException(422, str(e)) from e
        added, skipped = [], []
        for d in docs:
            try:
                added.append(index_document(d["name"], d["text"], d["chunk_size"] or chunk_size))
            except (UnsupportedFile, DuplicateDocument) as e:
                skipped.append({"file": d["name"], "error": str(e)})
        return {"added": added, "errors": skipped}

    # ---------- chat ----------
    def summarize_if_needed(session_id: str) -> None:
        hist = store.history(session_id)
        older = hist[:-RECENT_MESSAGES] if len(hist) > RECENT_MESSAGES else []
        sess = store.get_session(session_id)
        pending = older[sess["summarized_upto"]:]
        if len(pending) < SUMMARY_BATCH:
            return
        try:
            summary = complete(llm.summary_messages(sess["summary"], pending), max_tokens=300)
        except llm.LLMError as e:
            log.warning("summary failed: %s", e)
            return
        if summary:
            store.set_summary(session_id, summary, len(older))

    def resolve_query(question: str, history: list[dict], rewrite: bool) -> str:
        if not rewrite or not history:
            return question
        fallback = f"{[m for m in history if m['role'] == 'user'][-1]['content']} {question}"
        try:
            out = complete(llm.rewrite_messages(question, history), max_tokens=80, timeout=30)
        except llm.LLMError:
            return fallback
        return llm.clean_rewrite(out, fallback)

    @app.post("/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        t0 = time.perf_counter()
        history = store.history(req.session_id)
        summary = store.get_session(req.session_id)["summary"]
        query = resolve_query(req.question, history, req.rewrite)
        cfg = RetrievalConfig(method=req.method, top_k=req.top_k, rerank=req.rerank)
        passages = retriever.search(query, cfg)
        store.add_message(req.session_id, "user", req.question)
        sources = [
            {"n": i, "doc_name": p["doc_name"], "doc_id": p["doc_id"], "score": p["score"],
             "method": p["method"], "text": p["text"]}
            for i, p in enumerate(passages, 1)
        ]
        meta = {"method": req.method, "rerank": req.rerank, "top_k": req.top_k,
                "retrieval_query": query if query != req.question else None}

        def events() -> Iterator[str]:
            if meta["retrieval_query"]:
                yield json.dumps({"type": "rewrite", "query": query}) + "\n"
            yield json.dumps({"type": "sources", "sources": sources}) + "\n"
            answer = ""
            if not passages:
                answer = llm.REFUSAL
                yield json.dumps({"type": "token", "text": answer}) + "\n"
            else:
                try:
                    for tok in stream(llm.build_messages(req.question, passages, history[-RECENT_MESSAGES:],
                                                         summary)):
                        answer += tok
                        yield json.dumps({"type": "token", "text": tok}) + "\n"
                except llm.LLMError as e:
                    meta["error"] = str(e)
                    yield json.dumps({"type": "error", "error": str(e)}) + "\n"
                    if not answer:
                        return
            cited = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(sources)})
            meta["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            used = [s for s in sources if s["n"] in cited]
            mid = store.add_message(req.session_id, "assistant", answer, used, meta)
            yield json.dumps({"type": "done", "message_id": mid, "cited": cited,
                              "latency_ms": meta["latency_ms"]}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 background=BackgroundTask(summarize_if_needed, req.session_id))

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        return {**store.get_session(session_id), "messages": store.history(session_id)}

    @app.delete("/sessions/{session_id}")
    def clear_session(session_id: str) -> dict:
        store.clear_session(session_id)
        return {"cleared": session_id}

    # ---------- feedback & analytics ----------
    @app.post("/feedback/{message_id}")
    def feedback(message_id: str, req: FeedbackRequest) -> dict:
        if not store.set_feedback(message_id, req.rating, req.note):
            raise HTTPException(404, "assistant message not found")
        return {"ok": True}

    @app.get("/analytics")
    def analytics() -> dict:
        return build_report(store.all_turns(), store.list_documents())

    # ---------- evaluation lab ----------
    async def read_eval(file: UploadFile) -> list[dict]:
        try:
            return evaluation.parse_eval_set((await file.read()).decode("utf-8", errors="replace"))
        except evaluation.InvalidEvalSet as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/eval/retrieval")
    async def eval_retrieval(file: UploadFile = File(...), top_k: int = Form(5)) -> dict:
        items = await read_eval(file)
        configs = [RetrievalConfig(method=m, top_k=top_k) for m in METHODS]
        if reranker is not None:
            configs.append(RetrievalConfig(method="hybrid", top_k=top_k, rerank=True))
        return {"items": len(items), "corpus_version": store.version,
                "results": evaluation.evaluate_retrieval(retriever, items, configs)}

    def answer_once(question: str, cfg: RetrievalConfig) -> str:
        passages = retriever.search(question, cfg)
        if not passages:
            return llm.REFUSAL
        return "".join(stream(llm.build_messages(question, passages, [])))

    @app.post("/eval/answers")
    async def eval_answers(file: UploadFile = File(...), method: str = Form("hybrid"),
                           top_k: int = Form(5), rerank: bool = Form(False)) -> StreamingResponse:
        if method not in METHODS:
            raise HTTPException(422, f"method must be one of {METHODS}")
        items = await read_eval(file)
        cfg = RetrievalConfig(method=method, top_k=top_k, rerank=rerank)

        def run() -> Iterator[str]:
            done: list[dict] = []
            for n, it in enumerate(items, 1):
                done.append(evaluation.evaluate_one(it, lambda q: answer_once(q, cfg)))
                yield json.dumps({"type": "progress", "done": n, "total": len(items), "item": done[-1]}) + "\n"
            report = evaluation.summarize(done)
            report["config"] = {"method": method, "top_k": top_k, "rerank": rerank}
            yield json.dumps({"type": "report", "report": report}) + "\n"

        return StreamingResponse(run(), media_type="application/x-ndjson")

    return app
