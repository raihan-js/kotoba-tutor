"""M3: memory vs memoryless analyse eval on the LangSmith 300.

Per results/prereg_m3.md (frozen before this run). Writes results/m3.json
+ LangSmith experiment rows. ~15 min GPU (600 analyse calls).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kotoba.graph import analyse_backend  # noqa: E402

PROFILE_LINE = ("この学習者は particle の誤りが多い(12回)、conjugation が4回、"
                "politeness が2回、choice が1回。これらに注意して判定して。")


def main() -> None:
    import os

    from kotoba.graph import ANALYSE_PROMPT, make_llm_backend

    os.environ["LANGSMITH_API_KEY"] = (
        Path.home() / ".config/langsmith/key").read_text().strip()
    os.environ["LANGSMITH_TRACING"] = "false"
    from langsmith import Client

    llm_fn = make_llm_backend()
    mem_prompt = ANALYSE_PROMPT + PROFILE_LINE + "\n"

    def run_arm(rows: list[dict], prompt: str) -> list[dict]:
        import kotoba.graph as G
        from kotoba.graph import analyse_backend as ab

        outs = []
        t0 = time.time()
        for i, r in enumerate(rows):
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
    ds = c.read_dataset(dataset_name="kotoba-corrections-300-v1")
    rows = [{"learner": e.inputs["learner"], **e.outputs}
            for e in c.list_examples(dataset_id=ds.id)]
    print("rows:", len(rows))
    mem = run_arm(rows, mem_prompt)
    less = run_arm(rows, ANALYSE_PROMPT)

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

    res = {"n": len(rows), "memory": metrics(mem), "memoryless": metrics(less)}
    Path("results/m3.json").write_text(json.dumps(res, indent=2,
                                                  ensure_ascii=False) + "\n")
    with open("results/m3_rows.jsonl", "w", encoding="utf-8") as f:
        for r, a, b in zip(rows, mem, less):
            f.write(json.dumps({"learner": r["learner"], "gold": r["gold"],
                                "error_type": r["error_type"], "memory": a,
                                "memoryless": b}, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
