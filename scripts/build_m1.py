"""M1: build the 300-sentence correction set.

150 correct (Tatoeba as-is, assumed gold) + 150 with one injected error
(round-robin over particle/conjugation/politeness/choice; exact tags by
construction). Writes results/m1.json + results/m1_rows.jsonl, uploads
LangSmith dataset kotoba-corrections-300-v1.
Tatoeba licence: CC-BY 2.0 FR (used under the licence; cited).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kotoba.inject import INJECTORS


def load_tatoeba(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            rows.append((parts[0], parts[2]))
    return rows


def ok(text: str) -> bool:
    if not (10 <= len(text) <= 80):
        return False
    latin = sum(c.isascii() and c.isalpha() for c in text)
    if latin / max(len(text), 1) > 0.15:
        return False
    return any("\u3040" <= c <= "\u309f" for c in text)  # hiragana present


def main() -> None:
    import os

    rows = load_tatoeba("data/jpn_sentences.tsv")
    seen: dict[str, None] = {}
    pool = []
    for sid, text in rows:
        if ok(text) and text not in seen:
            seen[text] = None
            pool.append((sid, text))
    print("candidate pool:", len(pool))

    correct = [{"id": f"c{i:03d}", "learner": t, "gold": t,
                "error_type": "none", "detail": "", "tatoeba_id": sid}
               for i, (sid, t) in enumerate(pool[:150])]
    # Per-type targeted scan (fire rates differ 97% vs 1%; round-robin skews).
    used = {t for _, t in pool[:150]}
    injected = []
    per = {}
    for et in ["particle", "conjugation", "politeness", "choice"]:
        per[et] = 0
        for sid, text in pool[150:]:
            if per[et] >= (37 if et != "choice" else 39):
                break
            if text in used:
                continue
            r = INJECTORS[et](text)
            if r is None:
                continue
            learner, detail = r
            injected.append({"id": f"e{len(injected):03d}", "learner": learner,
                             "gold": text, "error_type": et, "detail": detail,
                             "tatoeba_id": sid})
            used.add(text)
            per[et] += 1
    counts = per
    print("injected per type:", counts)
    allrows = correct + injected
    Path("results").mkdir(exist_ok=True)
    Path("results/m1_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in allrows) + "\n")
    stats = {"n": len(allrows), "n_correct": len(correct),
             "n_injected": len(injected), "per_type": counts,
             "source": "Tatoeba jpn_sentences (CC-BY 2.0 FR)",
             "gold": "Tatoeba original; injection tags exact by construction"}
    Path("results/m1.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    os.environ["LANGSMITH_API_KEY"] = (
        Path.home() / ".config/langsmith/key").read_text().strip()
    os.environ["LANGSMITH_TRACING"] = "false"
    from langsmith import Client

    c = Client()
    name = "kotoba-corrections-300-v1"
    from langsmith.utils import LangSmithNotFoundError

    try:
        c.delete_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        print("no prior dataset to delete")
    ds = c.create_dataset(name, description="Kotoba-Tutor M1: 150 correct + "
                          "150 rule-injected JA learner errors (Tatoeba)")
    c.create_examples(
        inputs=[{"learner": r["learner"]} for r in allrows],
        outputs=[{"gold": r["gold"], "error_type": r["error_type"],
                  "detail": r["detail"]} for r in allrows],
        dataset_id=ds.id)
    print("uploaded LangSmith dataset", name, len(allrows))


if __name__ == "__main__":
    main()
