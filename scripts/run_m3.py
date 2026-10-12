"""M3: memory vs memoryless analyse eval on the LangSmith 300.

Per results/prereg_m3.md (frozen before this run). Writes results/m3.json
+ LangSmith experiment rows. ~15 min GPU (600 analyse calls).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROFILE_LINE = ("この学習者は particle の誤りが多い(12回)、conjugation が4回、"
                "politeness が2回、choice が1回。これらに注意して判定して。")


def main() -> None:
    import os

    from kotoba.graph import ANALYSE_PROMPT, make_llm_backend

    ds_name = sys.argv[1] if len(sys.argv) > 1 else "kotoba-corrections-300-v1"
    tag = sys.argv[2] if len(sys.argv) > 2 else "m3"

    os.environ["LANGSMITH_API_KEY"] = (
        Path.home() / ".config/langsmith/key").read_text().strip()
    os.environ["LANGSMITH_TRACING"] = "false"
    from langsmith import Client

    llm_fn = make_llm_backend()
    mem_prompt = ANALYSE_PROMPT + PROFILE_LINE + "\n"

    def run_arm(rows: list[dict], prompt: str, done: dict[str, dict]) -> list[dict]:
        import kotoba.graph as G
        from kotoba.graph import analyse_backend as ab

        outs = []
        t0 = time.time()
        for i, r in enumerate(rows):
            if r["id"] in done:
                outs.append(done[r["id"]])
                continue
            old = G.ANALYSE_PROMPT
            G.ANALYSE_PROMPT = prompt
            try:
                outs.append(ab(r["learner"], llm_fn))
            finally:
                G.ANALYSE_PROMPT = old
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        return outs

    c = Client()
    ds = c.read_dataset(dataset_name=ds_name)
    ex = list(c.list_examples(dataset_id=ds.id))
    rows = []
    for i, e in enumerate(ex):
        r = {"id": f"r{i}", **e.outputs, "learner": e.inputs["learner"]}
        rows.append(r)
    print("rows:", len(rows))
    # Resumable: per-row outputs flushed incrementally (survives timeouts).
    ppath = Path(f"results/{tag}_partial.jsonl")
    partial: dict[str, dict] = {}
    if ppath.exists():
        for line in ppath.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            partial[d["id"]] = d
        print("resuming, partial rows:", len(partial))
    mem = run_arm(rows, mem_prompt,
                  {k: v["memory"] for k, v in partial.items() if "memory" in v})
    with open(ppath, "a", encoding="utf-8") as f:
        for r, a in zip(rows, mem):
            if r["id"] not in partial:
                partial[r["id"]] = {}
            partial[r["id"]]["memory"] = a
            f.write(json.dumps({"id": r["id"], "memory": a}, ensure_ascii=False) + "\n")
    less = run_arm(rows, ANALYSE_PROMPT,
                   {k: v["memoryless"] for k, v in partial.items() if "memoryless" in v})
    with open(ppath, "a", encoding="utf-8") as f:
        for r, a in zip(rows, less):
            partial[r["id"]]["memoryless"] = a
            f.write(json.dumps({"id": r["id"], "memoryless": a}, ensure_ascii=False) + "\n")

    def metrics(outs: list[dict]) -> dict:
        per: dict[str, dict[str, int]] = {}
        fc = fc_n = 0
        strict = strict_n = 0
        expl_ok = expl_n = 0
        for r, a in zip(rows, outs):
            et = r["error_type"]
            if et == "none":
                fc_n += 1
                fc += int(bool(a.get("has_error")))
            else:
                d = per.setdefault(et, {"hit": 0, "n": 0})
                d["n"] += 1
                d["hit"] += int(bool(a.get("has_error"))
                                and a.get("error_type") == et)
                strict_n += 1
                strict += int(a.get("correction", "") == r["gold"])
            if a.get("has_error"):
                expl_n += 1
                expl_ok += int(bool(a.get("explanation"))
                               and len(a["explanation"]) <= 80)
        return {"false_correction": round(fc / fc_n, 3),
                "recall": {k: round(v["hit"] / v["n"], 3) for k, v in per.items()},
                "strict_correction": round(strict / strict_n, 3),
                "explanation_ok": round(expl_ok / expl_n, 3) if expl_n else 1.0}

    res = {"n": len(rows), "dataset": ds_name,
           "memory": metrics(mem), "memoryless": metrics(less)}
    Path(f"results/{tag}.json").write_text(json.dumps(res, indent=2,
                                                      ensure_ascii=False) + "\n")
    with open(f"results/{tag}_rows.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"learner": r["learner"], "gold": r["gold"],
                                "error_type": r["error_type"], "memory": a,
                                "memoryless": b}, ensure_ascii=False) + "\n" for r, a, b in zip(rows, mem, less))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
