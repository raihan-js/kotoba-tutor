"""Probe: do type labels appear ANYWHERE in stage-B raw outputs? (8 rows)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kotoba.graph import TYPE_LABELS, TYPE_PROMPT, make_llm_backend


def main() -> None:
    llm = make_llm_backend()
    with open("results/m1_rows.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    sample = [r for r in rows if r["error_type"] != "none"][:8]
    for r in sample:
        span = r["detail"].split("->")[0] if "->" in r["detail"] else ""
        raw = llm(TYPE_PROMPT.format(text=r["learner"], span=span) + " /no_think")
        hit = [lab for lab in TYPE_LABELS if lab in raw.lower()]
        print(r["error_type"], "| raw:", repr(raw.strip()[:100]), "| labels in raw:", hit)


if __name__ == "__main__":
    main()
