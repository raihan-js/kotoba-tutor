"""Brain fix: k-candidate correction loop with rule-typer voting.

Per prereg amendment 2026-10-12. k=3 (temp 0.7, seed 0) candidates per row;
unflag when correction==learner; else rule-type each pair, majority vote.
Writes results/fixloop.json. ~20 min GPU on the Tatoeba 300.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "scripts")
from rule_type import type_from_pair  # noqa: E402


def main() -> None:
    import os

    from kotoba.graph import analyse_backend, make_llm_backend

    os.environ["LANGSMITH_API_KEY"] = (
        Path.home() / ".config/langsmith/key").read_text().strip()
    os.environ["LANGSMITH_TRACING"] = "false"
    from langsmith import Client

    llm_fn = make_llm_backend(temperature=0.7, seed=0)
    c = Client()
    ds = c.read_dataset(dataset_name="kotoba-corrections-300-v1")
    rows = [{**e.outputs, "learner": e.inputs["learner"]}
            for e in c.list_examples(dataset_id=ds.id)]
    print("rows:", len(rows))
    outs = []
    t0 = time.time()
    for i, r in enumerate(rows):
        cands = [analyse_backend(r["learner"], llm_fn) for _ in range(3)]
        flagged = [a for a in cands if a.get("has_error")
                   and a.get("correction", r["learner"]) != r["learner"]]
        if not flagged:
            outs.append({"has_error": False, "error_type": "none",
                         "correction": r["learner"]})
        else:
            votes = Counter(type_from_pair(r["learner"], a.get("correction", ""))
                            for a in flagged)
            et, _ = votes.most_common(1)[0]
            best = next(a for a in flagged
                        if type_from_pair(r["learner"],
                                          a.get("correction", "")) == et)
            outs.append({"has_error": True, "error_type": et,
                         "correction": best.get("correction", "")})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    cm = Counter((r["error_type"], a["error_type"]) for r, a in zip(rows, outs))
    per = {}
    for et in ["particle", "conjugation", "politeness", "choice"]:
        sub = [(r, a) for r, a in zip(rows, outs) if r["error_type"] == et]
        per[et] = {"recall": round(sum(1 for _, a in sub
                                       if a["error_type"] == et) / len(sub), 3),
                   "n": len(sub)}
    neg = [(r, a) for r, a in zip(rows, outs) if r["error_type"] == "none"]
    strict = [(r, a) for r, a in zip(rows, outs) if r["error_type"] != "none"]
    res = {"n": len(rows), "method": "k=3 retry + rule-typer majority vote",
           "false_correction": round(sum(1 for _, a in neg
                                         if a["has_error"]) / len(neg), 3),
           "per_type": per,
           "strict_correction": round(sum(1 for r, a in strict
                                          if a["correction"] == r["gold"]) / len(strict), 3),
           "confusion": {f"{g}>{p}": n for (g, p), n in sorted(cm.items())}}
    Path("results/fixloop.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
