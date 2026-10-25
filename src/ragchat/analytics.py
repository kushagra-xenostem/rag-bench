"""Feedback analytics: answer quality, content gaps and data-driven setting suggestions."""
import time
from .llm import REFUSAL
from collections import Counter, defaultdict

MIN_RATINGS_FOR_SUGGESTION = 5


def _rate(up: int, down: int) -> float | None:
    return round(up / (up + down), 3) if up + down else None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def build_report(turns: list[dict], documents: list[dict]) -> dict:
    total = len(turns)
    refused = [t for t in turns if t["content"].strip() == REFUSAL]
    errored = [t for t in turns if t["meta"].get("error")]
    rated = [t for t in turns if t["rating"] in (1, -1)]
    up = sum(1 for t in rated if t["rating"] == 1)
    latencies = [t["meta"]["latency_ms"] for t in turns if t["meta"].get("latency_ms") is not None]

    per_doc: dict[str, Counter] = defaultdict(Counter)
    for t in turns:
        for name in {s["doc_name"] for s in t["sources"]}:
            per_doc[name]["cited"] += 1
            if t["rating"] == 1:
                per_doc[name]["up"] += 1
            elif t["rating"] == -1:
                per_doc[name]["down"] += 1
    doc_names = {d["name"] for d in documents}
    documents_report = sorted(
        (
            {"document": name, "times_cited": per_doc[name]["cited"], "thumbs_up": per_doc[name]["up"],
             "thumbs_down": per_doc[name]["down"],
             "helpful_rate": _rate(per_doc[name]["up"], per_doc[name]["down"])}
            for name in doc_names | set(per_doc)
        ),
        key=lambda d: (-d["times_cited"], d["document"]),
    )

    config_stats: dict[str, Counter] = defaultdict(Counter)
    for t in rated:
        m = t["meta"]
        key = f"{m.get('method', 'bm25')}{'+rerank' if m.get('rerank') else ''} / top_k={m.get('top_k', 5)}"
        config_stats[key]["up" if t["rating"] == 1 else "down"] += 1
    configs = sorted(
        ({"config": k, "ratings": c["up"] + c["down"], "helpful_rate": _rate(c["up"], c["down"])}
         for k, c in config_stats.items()),
        key=lambda c: (-(c["helpful_rate"] or 0), -c["ratings"]),
    )
    eligible = [c for c in configs if c["ratings"] >= MIN_RATINGS_FOR_SUGGESTION]

    recommendations = []
    if eligible and len(eligible) > 1:
        best = eligible[0]
        recommendations.append(
            f"Best rated retrieval setting so far: {best['config']} "
            f"({best['helpful_rate']:.0%} helpful over {best['ratings']} ratings)."
        )
    elif not eligible:
        recommendations.append(
            f"Collect at least {MIN_RATINGS_FOR_SUGGESTION} ratings per setting before comparing retrieval settings."
        )
    if total and len(refused) / total > 0.3:
        recommendations.append(
            f"{len(refused)} of {total} questions were not answerable from the documents. "
            "Review the content gaps below and upload documents that cover them."
        )
    for d in documents_report:
        if d["thumbs_down"] >= 3 and (d["helpful_rate"] or 0) < 0.5:
            recommendations.append(
                f"Answers citing '{d['document']}' are often rated unhelpful; check it is current and well extracted."
            )
    unused = [d["document"] for d in documents_report if d["times_cited"] == 0]
    if unused and total >= 10:
        recommendations.append(f"Never cited so far: {', '.join(unused[:5])}.")

    def brief(t: dict) -> dict:
        return {"question": t["question"], "answer": t["content"][:400], "note": t["note"],
                "sources": [s["doc_name"] for s in t["sources"]], "message_id": t["id"]}

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": {
            "questions": total,
            "answered": total - len(refused) - len(errored),
            "not_found": len(refused),
            "errors": len(errored),
            "not_found_rate": round(len(refused) / total, 3) if total else None,
            "ratings": len(rated),
            "thumbs_up": up,
            "thumbs_down": len(rated) - up,
            "helpful_rate": _rate(up, len(rated) - up),
            "latency_ms_p50": _pct(latencies, 0.5),
            "latency_ms_p95": _pct(latencies, 0.95),
        },
        "documents": documents_report,
        "retrieval_settings": configs,
        "content_gaps": [brief(t) for t in refused][-20:],
        "low_rated": [brief(t) for t in rated if t["rating"] == -1][-20:],
        "recommendations": recommendations,
    }
