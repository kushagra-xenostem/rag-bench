"""Run the evaluation lab against a running backend and save a JSON report.

Usage: python scripts/run_eval.py [--eval samples/eval_set.jsonl] [--answers] [--method hybrid] [--top-k 5]
       [--index-samples] [--backend http://127.0.0.1:8004]"""
import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ["employee_handbook.md", "travel_policy.txt", "aurora_x2_manual.md"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8004")
    ap.add_argument("--eval", default=str(ROOT / "samples" / "eval_set.jsonl"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--method", default="hybrid", choices=["bm25", "semantic", "hybrid"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--answers", action="store_true", help="also run end-to-end answers (calls the LLM)")
    ap.add_argument("--index-samples", action="store_true", help="upload samples/ first (duplicates are skipped)")
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    a = ap.parse_args()

    c = httpx.Client(base_url=a.backend, timeout=3600)
    if a.index_samples:
        files = [("files", (n, (ROOT / "samples" / n).read_bytes())) for n in SAMPLES]
        res = c.post("/documents", files=files, data={"chunk_size": a.chunk_size}).json()
        print(f"indexed {len(res['added'])}, skipped {len(res['errors'])}")

    data = Path(a.eval).read_bytes()
    report = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "eval_set": Path(a.eval).name}
    ret = c.post("/eval/retrieval", files={"file": ("e.jsonl", data)}, data={"top_k": a.top_k})
    ret.raise_for_status()
    report["retrieval"] = ret.json()
    print(f"\nRetrieval (top_k={a.top_k}, {ret.json()['items']} questions)")
    for row in ret.json()["results"]:
        print(f"  {row['config']:14} hit@1={row['hit@1']:.2f} hit@{a.top_k}={row[f'hit@{a.top_k}']:.2f} "
              f"mrr={row['mrr']:.2f} unanswerable_with_passages={row['unanswerable_with_passages']} "
              f"{row['ms_per_query']} ms/q")

    if a.answers:
        body = {"method": a.method, "top_k": str(a.top_k), "rerank": str(a.rerank).lower()}
        with c.stream("POST", "/eval/answers", files={"file": ("e.jsonl", data)}, data=body) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "progress":
                    it = ev["item"]
                    print(f"  [{ev['done']}/{ev['total']}] {'OK ' if it['correct'] else 'BAD'} {it['id']}: "
                          f"{it['answer'][:90]!r}")
                else:
                    report["answers"] = ev["report"]
        r = report["answers"]
        print(f"\nAnswers ({a.method}{'+rerank' if a.rerank else ''}): accuracy={r['answer_accuracy']} "
              f"refusal_accuracy={r['refusal_accuracy']} citation_rate={r['citation_rate']} "
              f"false_refusals={r['false_refusals']} avg={r['avg_seconds']}s errors={r['errors']}")

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    path = out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
