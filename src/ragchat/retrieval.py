import threading
import numpy as np
from .bm25 import BM25
from .store import Store
from dataclasses import dataclass
from .embed import Embedder, Reranker

METHODS = ("bm25", "semantic", "hybrid")
RRF_K = 60


@dataclass(frozen=True)
class RetrievalConfig:
    method: str = "hybrid"
    top_k: int = 5
    rerank: bool = False
    min_similarity: float = 0.5
    min_rerank_score: float = -6.0
    candidates: int = 30


class Retriever:
    """Keyword, semantic and hybrid (RRF) retrieval over every stored chunk, with optional
    cross-encoder reranking. The in-memory index is rebuilt whenever the corpus version changes."""

    def __init__(self, store: Store, embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._lock = threading.Lock()
        self._version = -1
        self._chunks: list[dict] = []
        self._matrix: np.ndarray | None = None
        self._bm25 = BM25([])

    def _refresh(self) -> None:
        with self._lock:
            v = self.store.version
            if v != self._version:
                self._chunks, self._matrix = self.store.all_chunks()
                self._bm25 = BM25([c["text"] for c in self._chunks])
                self._version = v

    def _bm25_ranked(self, query: str, n: int) -> list[tuple[int, float]]:
        return [(h.index, h.score) for h in self._bm25.search(query, n)]

    def _semantic_ranked(self, query: str, n: int, floor: float) -> list[tuple[int, float]]:
        if self._matrix is None or self.embedder is None or not self._chunks:
            return []
        sims = self._matrix @ self.embedder.embed_query(query)
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))[:n]
        return [(i, float(sims[i])) for i in order if sims[i] >= floor]

    def semantic_available(self) -> bool:
        self._refresh()
        return self.embedder is not None and (self._matrix is not None or not self._chunks)

    def search(self, query: str, cfg: RetrievalConfig = RetrievalConfig()) -> list[dict]:
        if cfg.method not in METHODS:
            raise ValueError(f"unknown method {cfg.method!r}")
        self._refresh()
        method = cfg.method if cfg.method == "bm25" or self.semantic_available() else "bm25"
        pool = max(cfg.candidates, cfg.top_k)

        if method == "bm25":
            ranked = self._bm25_ranked(query, pool)
        elif method == "semantic":
            ranked = self._semantic_ranked(query, pool, cfg.min_similarity)
        else:
            fused: dict[int, float] = {}
            for lst in (self._bm25_ranked(query, pool), self._semantic_ranked(query, pool, cfg.min_similarity)):
                for rank, (i, _) in enumerate(lst):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:pool]

        reranked = False
        if cfg.rerank and self.reranker is not None and ranked:
            scores = self.reranker.score(query, [self._chunks[i]["text"] for i, _ in ranked])
            ranked = sorted(
                ((i, s) for (i, _), s in zip(ranked, scores) if s >= cfg.min_rerank_score),
                key=lambda x: (-x[1], x[0]),
            )
            reranked = True

        return [
            {**self._chunks[i], "score": round(s, 4), "method": method + ("+rerank" if reranked else "")}
            for i, s in ranked[: cfg.top_k]
        ]
