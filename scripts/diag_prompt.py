"""Diagnostic: neutral judge-first prompt on 12 rows (acquiescence check)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kotoba.graph import make_llm_backend  # noqa: E402

P = ('文が正しいか誤りか、一語で最初に答えよ(正しい/誤り)。誤りの場合のみ次行にJSON: '
     '{"error_type": "particleの1つ", "correction": "正文"}。\n文: ')


def main() -> None:
    llm = make_llm_backend()
    rows = [json.loads(line) for line in
            open("results/m1_rows.jsonl", encoding="utf-8")]
    sample = ([r for r in rows if r["error_type"] == "none"][:6]
              + [r for r in rows if r["error_type"] != "none"][::25][:6])
    for r in sample:
        out = llm(P + r["learner"]).strip().splitlines()
        tag = "C " if r["error_type"] == "none" else "E " + r["error_type"][:4]
        print(tag, "|", (out[0][:60] if out else None))


if __name__ == "__main__":
    main()
