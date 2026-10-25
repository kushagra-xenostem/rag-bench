import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what when "
    "where which who why will with do does did can you your we our they their".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    index: int
    score: float


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        hits = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = 1 - self.b + self.b * self.lens[i] / (self.avgdl or 1)
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if s > 0:
                hits.append(Hit(i, s))
        # explicit tie rule (lower index first) keeps ranking identical across platforms
        hits.sort(key=lambda h: (-h.score, h.index))
        return hits[:k]
