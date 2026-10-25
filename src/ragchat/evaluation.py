"""Built-in evaluation lab: retrieval metrics per configuration and end-to-end answer scoring.

Eval set (JSONL), one item per line:
  {"id": "q1", "question": "...", "answerable": true,
   "evidence": "phrase that must appear in a retrieved chunk",
   "answer_keywords": ["600", "receipts"]}
Unanswerable items set "answerable": false and omit evidence/keywords."""
import json
import re
import time
from collections.abc import Callable

from .llm import REFUSAL
from .retrieval import RetrievalConfig, Retriever


class InvalidEvalSet(ValueError):
    pass


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def parse_eval_set(text: str) -> list[dict]:
    items = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise InvalidEvalSet(f"Line {n}: invalid JSON") from e
        if not isinstance(obj.get("question"), str) or not obj["question"].strip():
            raise InvalidEvalSet(f"Line {n}: missing 'question'")
        answerable = bool(obj.get("answerable", True))
        if answerable and not isinstance(obj.get("evidence"), str):
            raise InvalidEvalSet(f"Line {n}: answerable items need an 'evidence' phrase")
        items.append({
            "id": str(obj.get("id", f"q{n}")), "question": obj["question"], "answerable": answerable,
            "evidence": obj.get("evidence"), "answer_keywords": list(obj.get("answer_keywords") or []),
        })
    if not items:
        raise InvalidEvalSet("Eval set is empty")
    return items


def _rank_of_evidence(results: list[dict], evidence: str) -> int | None:
    ev = _norm(evidence)
    for rank, r in enumerate(results, 1):
        if ev in _norm(r["text"]):
            return rank
    return None


def evaluate_retrieval(retriever: Retriever, items: list[dict], configs: list[RetrievalConfig]) -> list[dict]:
    rows = []
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    for cfg in configs:
        t0 = time.perf_counter()
        ranks, misses, false_hits = [], [], 0
        for it in answerable:
            r = _rank_of_evidence(retriever.search(it["question"], cfg), it["evidence"])
            ranks.append(r)
            if r is None:
                misses.append(it["id"])
        for it in unanswerable:
            if retriever.search(it["question"], cfg):
                false_hits += 1
        n = len(answerable) or 1
        rows.append({
            "config": f"{cfg.method}{'+rerank' if cfg.rerank else ''}",
            "top_k": cfg.top_k,
            "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
            f"hit@{cfg.top_k}": round(sum(1 for r in ranks if r) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "unanswerable_with_passages": f"{false_hits}/{len(unanswerable)}",
            "ms_per_query": round((time.perf_counter() - t0) * 1000 / max(1, len(items)), 1),
            "missed": misses,
        })
    return rows


def score_answer(item: dict, answer: str) -> dict:
    refused = REFUSAL.lower().rstrip(".") in answer.lower()
    cited = bool(re.search(r"\[\d+\]", answer))
    if not item["answerable"]:
        return {"correct": refused, "refused": refused, "cited": cited}
    norm = _norm(answer)
    kws = item["answer_keywords"]
    found = [k for k in kws if _norm(k) in norm]
    return {"correct": (not refused) and len(found) == len(kws), "refused": refused, "cited": cited,
            "missing_keywords": [k for k in kws if k not in found]}


def evaluate_one(item: dict, answer_fn: Callable[[str], str]) -> dict:
    t0 = time.perf_counter()
    try:
        answer, error = answer_fn(item["question"]), None
    except Exception as e:  # keep evaluating remaining items if one LLM call fails
        answer, error = "", str(e)
    return {"id": item["id"], "question": item["question"], "answerable": item["answerable"], "answer": answer,
            "seconds": round(time.perf_counter() - t0, 1), "error": error, **score_answer(item, answer)}


def evaluate_answers(items: list[dict], answer_fn: Callable[[str], str]) -> dict:
    return summarize([evaluate_one(it, answer_fn) for it in items])


def summarize(results: list[dict]) -> dict:
    if not results:
        raise InvalidEvalSet("No results to summarise")
    ans = [r for r in results if r["answerable"]]
    una = [r for r in results if not r["answerable"]]
    return {
        "answer_accuracy": round(sum(r["correct"] for r in ans) / len(ans), 3) if ans else None,
        "citation_rate": round(sum(r["cited"] for r in ans if not r["refused"]) /
                               max(1, sum(1 for r in ans if not r["refused"])), 3) if ans else None,
        "false_refusals": sum(r["refused"] for r in ans),
        "refusal_accuracy": round(sum(r["correct"] for r in una) / len(una), 3) if una else None,
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 1),
        "errors": sum(1 for r in results if r["error"]),
        "items": results,
    }
