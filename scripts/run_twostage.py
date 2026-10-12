"""Brain-fix probe: two-stage typing (stage A detect + stage B single-label
type call) on the Tatoeba 300. Per prereg amendment 2026-10-12: same rows,
same backend, temp 0. Writes results/twostage.json. ~10 min GPU."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    import os

    from kotoba.graph import analyse_backend, make_llm_backend, type_backend

    os.environ["LANGSMITH_API_KEY"] = (
        Path.home() / ".config/langsmith/key").read_text().strip()
    os.environ["LANGSMITH_TRACING"] = "false"
    from langsmith import Client

    llm_fn = make_llm_backend()
    c = Client()
    ds = c.read_dataset(dataset_name="kotoba-corrections-300-v1")
    rows = [{"learner": e.inputs["learner"], **e.outputs}
            for e in c.list_examples(dataset_id=ds.id)]
    print("rows:", len(rows))
    outs = []
    t0 = time.time()
    for i, r in enumerate(rows):
        a = analyse_backend(r["learner"], llm_fn)
        # Stage B overrides ONLY the type label; detection stays stage A.
        if a.get("has_error"):
            a["error_type"] = type_backend(
                r["learner"], str(a.get("span", "")), llm_fn)
        else:
            a["error_type"] = "none"
        outs.append(a)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    from collections import Counter
    cm = Counter((r["error_type"], a.get("error_type"))
                 for r, a in zip(rows, outs))
    print("confusion:", dict(cm))
    per = {}
    for et in ["particle", "conjugation", "politeness", "choice"]:
        sub = [(r, a) for r, a in zip(rows, outs) if r["error_type"] == et]
        hit = sum(1 for _, a in sub if a.get("error_type") == et)
        per[et] = {"recall": round(hit / len(sub), 3), "n": len(sub)}
    neg = [(r, a) for r, a in zip(rows, outs) if r["error_type"] == "none"]
    fc = sum(1 for _, a in neg if a.get("has_error"))
    res = {"n": len(rows), "method": "two-stage (A detect + B single-label)",
           "false_correction": round(fc / len(neg), 3), "per_type": per}
    Path("results/twostage.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
