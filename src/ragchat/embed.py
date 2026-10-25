import os
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ROOT

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CACHE_DIR = Path(os.getenv("RAGCHAT_MODEL_DIR", str(ROOT / "models")))


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class FastEmbedder:
    """Local ONNX embedding model (no GPU, no torch); downloaded once into ./models."""

    def __init__(self, model: str = EMBED_MODEL):
        self.model_name = model
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name, cache_dir=str(CACHE_DIR))
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return _normalize(np.vstack(list(self._load().embed(texts, batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.asarray(next(iter(self._load().query_embed(text)))))


class FastReranker:
    def __init__(self, model: str = RERANK_MODEL):
        self.model_name = model
        self._model = None
        self._lock = threading.Lock()

    def score(self, query: str, texts: list[str]) -> list[float]:
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(self.model_name, cache_dir=str(CACHE_DIR))
        return [float(s) for s in self._model.rerank(query, texts)] if texts else []
