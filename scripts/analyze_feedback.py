"""Print and save the feedback analytics report straight from the local database (no server needed).

Usage: python scripts/analyze_feedback.py [--db data/ragchat.db] [--out reports/feedback_report.json]"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragchat.analytics import build_report  # noqa: E402
from ragchat.store import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "ragchat.db"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "feedback_report.json"))
    a = ap.parse_args()
    if not Path(a.db).exists():
        sys.exit(f"No database at {a.db}. Run the app and ask some questions first.")
    store = Store(Path(a.db))
    report = build_report(store.all_turns(), store.list_documents())
    t = report["totals"]
    print(f"Questions: {t['questions']}  answered: {t['answered']}  not found: {t['not_found']}  "
          f"helpful: {t['helpful_rate']}  ({t['ratings']} ratings)")
    for r in report["recommendations"]:
        print(f"- {r}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
