# RAG Chat Hub

A local, self-contained retrieval-augmented chat application. Upload documents, ask questions, and get streamed answers that cite the exact passages they come from. When the documents do not contain the answer, the assistant says so instead of guessing.

The project includes hybrid retrieval with cross-encoder reranking, follow-up question rewriting, conversation memory, feedback analytics, corpus versioning, and a built-in evaluation lab. A one-click launcher sets everything up on Windows.

![Chat](docs/screenshots/chat.png)

## Contents

- [Quick start](#quick-start)
- [Features](#features)
- [Results](#results)
- [Architecture](#architecture)
- [API](#api)
- [Evaluation lab](#evaluation-lab)
- [Development](#development)
- [Design decisions and limitations](#design-decisions-and-limitations)

## Quick start

**Requirements:** Windows 10/11 with Python 3.10 or newer ([python.org](https://www.python.org/downloads/), with "Add python.exe to PATH" ticked), and access to an OpenAI-compatible LLM endpoint.

1. Download or clone this repository.
2. Double-click `launcher.exe`.

On first run the launcher:

1. creates a virtual environment in `.venv/`
2. installs `requirements.txt`
3. creates `.env` from `.env.example`
4. starts the API and the web UI
5. opens `http://localhost:8504` in your browser

Later runs skip the installation and start in a few seconds. Closing the launcher window stops every process it started.

Configure the LLM endpoint in `.env`:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=your-key
LLM_MODEL=qwen3.6-35b-a3b
```

The local embedding and reranking models (about 90 MB) download automatically into `models/` on first start. Until they are ready, the app falls back to keyword search.

On Linux or macOS, run `./launch.sh`. To run the services manually:

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m uvicorn --factory ragchat.api:create_app --app-dir src --port 8004
.venv/Scripts/python -m streamlit run ui/app.py --server.port 8504
```

To try it quickly, upload the three files in `samples/`.

## Features

| Area | Capability |
|---|---|
| Ingestion | PDF, DOCX, TXT and Markdown; configurable chunk size with 20% overlap; duplicate detection by normalised content hash |
| Retrieval | BM25 keyword search, local semantic search (`bge-small-en-v1.5`, ONNX on CPU), hybrid fusion (reciprocal rank fusion) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |
| Answering | Token streaming, numbered citations, cited passages highlighted in the UI, grounded refusal when nothing relevant is found (no LLM call is made), passages treated as untrusted input |
| Conversation | Follow-up questions are rewritten into standalone search queries; the last three turns stay verbatim; older turns are compressed into a rolling summary |
| Feedback | Helpful / not-helpful ratings with optional notes; analytics on answer rate, helpfulness, latency, per-document usefulness, content gaps and best-rated retrieval settings |
| Corpus management | Version log of every add / re-index / delete, re-chunking of individual documents, portable zip export, and import from zip or generic JSONL |
| Evaluation | Retrieval comparison of all configurations (hit@k, MRR) and end-to-end answer scoring (accuracy, refusal accuracy, citation rate) from a JSONL question set |
| Operations | One-click launcher with dependency caching, port selection and guaranteed process cleanup; everything runs locally except the LLM call |

### Screenshots

| Library and corpus versioning | Feedback insights |
|---|---|
| ![Library](docs/screenshots/library.png) | ![Insights](docs/screenshots/insights.png) |

![Evaluation](docs/screenshots/evaluation.png)

## Results

Measured with the built-in evaluation lab on the sample corpus (3 documents, 60-word chunks, 21 chunks) and `samples/eval_set.jsonl`: 20 answerable questions, many paraphrased away from the source wording, plus 4 unanswerable ones. The LLM was Qwen3.6-35B-A3B served by llama.cpp.

**Retrieval (top-5).** A hit means the labelled evidence sentence appears in a retrieved passage.

| Configuration | Hit@1 | Hit@5 | MRR | Unanswerable questions that still returned passages | Latency per query |
|---|---|---|---|---|---|
| BM25 | 0.80 | 1.00 | 0.88 | 3 of 4 | < 1 ms |
| Semantic | 0.70 | 0.95 | 0.80 | 3 of 4 | 25 ms |
| Hybrid (RRF) | 0.75 | 1.00 | 0.87 | 3 of 4 | 23 ms |
| **Hybrid + rerank** | **0.80** | **1.00** | **0.90** | **1 of 4** | 167 ms |

**End-to-end answers**

| Configuration | Answer accuracy | Refusal accuracy | Citation rate | Average time per question |
|---|---|---|---|---|
| Hybrid | 95% (19/20) | 100% (4/4) | 100% | 3.3 s |
| **Hybrid + rerank** (default) | **100% (20/20)** | **100% (4/4)** | **100%** | 2.9 s |

Observations:

- Reranking makes the biggest difference. It moves the best passage to the top, and its score threshold filters off-topic questions before the LLM is called. The one hybrid-only error was a question about colour bands: the most relevant passage ranked below a similar one, and the model mixed up the bands.
- No semantic-similarity threshold separated answerable from unanswerable questions on this corpus (scores overlap between 0.62 and 0.65). For this reason the cross-encoder score, not cosine similarity, decides when to skip the LLM.
- Follow-up rewriting adds one short LLM call. On a single-slot local server this is about 4 s, so it can be switched off in the sidebar.

The sample set is small, so treat these numbers as a demonstration of the method rather than a benchmark. Run the lab on your own documents and questions for numbers that matter to you.

## Architecture

```
                     +--------------------------------------------+
 Browser :8504  -->  | Streamlit UI (ui/app.py)                   |
                     |  Chat | Library | Insights | Evaluation     |
                     +----------------------+---------------------+
                                            | HTTP / NDJSON stream
                     +----------------------v---------------------+
                     | FastAPI service (src/ragchat/api.py) :8004 |
                     |                                            |
                     |  loaders -> chunking -> embed (ONNX, CPU)  |
                     |  retrieval: BM25 + cosine -> RRF -> rerank |
                     |  llm: rewrite / stream answer / summarise  |-->  OpenAI-compatible LLM
                     |  analytics, corpus_io, evaluation          |
                     |  store: SQLite (documents, chunks,         |
                     |         embeddings, sessions, feedback,    |
                     |         corpus version log)                |
                     +--------------------------------------------+
 launcher.exe: venv bootstrap, dependency cache, port selection, Windows Job Object cleanup
```

| Module | Responsibility |
|---|---|
| `loaders.py` | Text extraction for PDF, DOCX, TXT and Markdown, with size and type validation |
| `chunking.py` | Word-window chunking with overlap |
| `bm25.py` | BM25 Okapi with an explicit tie-break, so rankings are identical across platforms |
| `embed.py` | Lazy-loaded local embedding and cross-encoder models |
| `retrieval.py` | Method selection, reciprocal rank fusion, reranking and thresholds; the index rebuilds when the corpus version changes |
| `llm.py` | Streaming and non-streaming chat calls, prompts for answering, rewriting and summarising |
| `analytics.py` | Feedback report and recommendations |
| `corpus_io.py` | Portable corpus format (manifest plus JSONL) |
| `evaluation.py` | Eval-set parsing, retrieval metrics and answer scoring |
| `store.py` | SQLite persistence with a schema-version guard |

## API

Interactive documentation is available at `http://localhost:8004/docs` while the app is running.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health?check_llm=true` | Service, model and LLM status |
| POST | `/documents` | Upload files (`files`, `chunk_size`) |
| GET | `/documents`, `/documents/{id}` | List documents, or one document with its chunks |
| POST | `/documents/{id}/reindex` | Re-chunk one document |
| DELETE | `/documents/{id}` | Remove a document |
| GET | `/corpus` | Corpus statistics and version log |
| GET | `/corpus/export` | Download the corpus as a zip |
| POST | `/corpus/import` | Import a zip or JSONL corpus (duplicates are skipped) |
| POST | `/chat` | Ask a question. Returns an NDJSON stream of `rewrite`, `sources`, `token`, `done` or `error` events |
| GET, DELETE | `/sessions/{id}` | Session history and summary, or clear the session |
| POST | `/feedback/{message_id}` | Rate an answer (-1, 0 or 1) with an optional note |
| GET | `/analytics` | Feedback analytics report |
| POST | `/eval/retrieval` | Compare retrieval configurations on an eval set |
| POST | `/eval/answers` | End-to-end answer evaluation (NDJSON progress, then a report) |

Example chat request:

```json
{"session_id": "abc", "question": "How much is the home office stipend?",
 "method": "hybrid", "rerank": true, "top_k": 5, "rewrite": true}
```

## Evaluation lab

An eval set is a JSONL file with one question per line:

```json
{"id": "q1", "question": "How much is the stipend?", "answerable": true,
 "evidence": "home office stipend of 600 USD", "answer_keywords": ["600"]}
{"id": "q2", "question": "What is the parental leave policy?", "answerable": false}
```

- `evidence` is a sentence fragment that must appear in a retrieved passage for a retrieval hit.
- `answer_keywords` must all appear in a correct answer.
- Unanswerable questions are correct only when the assistant refuses.

You can run the lab from the Evaluation tab, or from the command line against a running backend:

```
.venv/Scripts/python scripts/run_eval.py --index-samples --answers --rerank
.venv/Scripts/python scripts/analyze_feedback.py
```

Reports are written to `reports/`.

## Development

```
.venv/Scripts/python -m pytest              # 61 tests; no LLM or model download needed
powershell -File scripts/build_launcher.ps1 # rebuild launcher.exe (requires uv)
```

The tests replace the LLM, the embedding model and the reranker with deterministic fakes. Together they cover:

- ingestion, chunking, BM25 and the retrieval methods
- the streaming protocol, refusal, rewriting and summarisation
- feedback, analytics, versioning, export/import and evaluation
- the launcher, including a test that hard-kills it and checks its child processes are gone
- the Streamlit UI, rendered headlessly against an in-process backend

```
src/ragchat/    service code
ui/             Streamlit front end
launcher/       launcher source (built into launcher.exe)
samples/        sample documents and eval set
scripts/        evaluation, feedback report, launcher build
tests/          test suite
```

## Design decisions and limitations

- **Local first.** Ingestion, embeddings, reranking, storage and the UI all run on the user's machine. Only answer generation, rewriting and summarisation call the configured LLM endpoint.
- **CPU models.** The ONNX models need no GPU and no PyTorch, which keeps installation small. Embedding a document takes milliseconds per chunk.
- **Scanned PDFs** without a text layer are rejected, because OCR is out of scope.
- **Single user.** The app is designed for local use. It has no authentication and binds to `127.0.0.1` only.
- **Thresholds** were tuned on the sample corpus. Use the evaluation lab to confirm them on your own documents.
