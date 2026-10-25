import sys
import json
import pytest
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "launcher")]

from ragchat.bm25 import tokenize  # noqa: E402

DIM = 64


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words => higher cosine similarity."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    def score(self, query, texts):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) - 0.5 for t in texts]


@pytest.fixture
def make_client(tmp_path):
    from fastapi.testclient import TestClient

    from ragchat.api import create_app
    from ragchat.config import Settings

    counter = {"n": 0}

    def factory(stream_fn=None, complete_fn=None, embedder="hash", reranker="overlap"):
        calls = {"stream": [], "complete": []}

        def fake_stream(msgs):
            calls["stream"].append(msgs)
            yield "The battery lasts 14 hours "
            yield "[1]."

        def fake_complete(msgs, **kw):
            calls["complete"].append(msgs)
            if "standalone search query" in msgs[0]["content"]:
                return "Zephyr-9 warranty period"
            return "- user asked about the Zephyr-9 battery and warranty"

        counter["n"] += 1
        s = Settings("http://llm.invalid/v1", "k", "m", tmp_path / f"t{counter['n']}.db", 5)
        app = create_app(
            s,
            stream_fn=stream_fn or fake_stream,
            complete_fn=complete_fn or fake_complete,
            embedder=HashEmbedder() if embedder == "hash" else embedder,
            reranker=OverlapReranker() if reranker == "overlap" else reranker,
            use_local_models=False,
        )
        client = TestClient(app)
        client.__enter__()
        return client, calls

    return factory


def ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line]
