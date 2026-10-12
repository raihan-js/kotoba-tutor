"""v1.1 eval set from TEC-JL human gold (research-only, never redistributed).

Single non-noop-edit sentences mapped to our 4 types; noop sentences are
corrects. Tag mapping (documented approximation, validated by sampling):
  particle:    R/M/U:PART
  conjugation: *:INFL, *:TENSE (VERB/AUX/ADJ)
  politeness:  R/M/U:AUX (non-INFL aux edits: ます/です/ない add/drop/swap)
  choice:      SPELL + OTHER + NOUN/VERB/ADJ/ADV/PRON/DET replacements
Excluded (unmappable): PUNCT, WO, CONJ.
Writes results/v11_rows.jsonl + results/v11.json. Uploads LangSmith
kotoba-tecjl-gold-v1 (inputs only + gold; corpus text stays, licence noted).
"""
import json
from pathlib import Path

PART = {"R:PART", "M:PART", "U:PART"}
POL = {"R:AUX", "M:AUX", "U:AUX"}
CHOICE = {"R:SPELL", "M:SPELL", "U:SPELL", "R:OTHER", "M:OTHER", "U:OTHER",
          "R:NOUN", "M:NOUN", "U:NOUN", "R:VERB", "M:VERB", "U:VERB",
          "R:ADJ", "U:ADJ", "R:ADV", "U:ADV", "R:PRON", "U:PRON",
          "R:DET", "M:DET", "U:DET"}


def tag_to_type(tag: str) -> str | None:
    if tag in PART:
        return "particle"
    if tag in POL:
        return "politeness"
    if tag in CHOICE:
        return "choice"
    base = tag.split(":")
    if "INFL" in base or "TENSE" in base:
        return "conjugation"
    return None


def parse_m2(path: str) -> list[list[str]]:
    """All A-line tags per sentence; 'noop' entries kept as 'noop'."""
    sents: list[list[str]] = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("S "):
            sents.append([])
        elif line.startswith("A "):
            sents[-1].append("noop" if "noop" in line else line.split("|||")[1])
    return sents


def main() -> None:
    base = Path("data/corpora/tec-jl/TEC_JL")
    src = (base / "tec-jl.src").read_text(encoding="utf-8").splitlines()
    ref0 = (base / "tec-jl.ref0").read_text(encoding="utf-8").splitlines()
    edits = parse_m2(str(base / "tec-jl.char.m2"))
    assert len(src) == len(ref0) == len(edits), (len(src), len(edits))
    rows = []
    for i, (s, r, es) in enumerate(zip(src, ref0, edits)):
        real = [e for e in es if e != "noop"]
        if not real:
            rows.append({"id": f"t{i:04d}", "learner": s, "gold": s,
                         "error_type": "none", "detail": "",
                         "source": "tec-jl/noop"})
            continue
        types = {tag_to_type(e) for e in real} - {None}
        if len(types) == 1:
            # Single-type (possibly multi-edit): pure typing signal.
            et = next(iter(types))
            rows.append({"id": f"t{i:04d}", "learner": s, "gold": r,
                         "error_type": et,
                         "detail": "+".join(real),
                         "source": "tec-jl/single-type"})
    from collections import Counter
    print(Counter(r["error_type"] for r in rows))
    Path("results").mkdir(exist_ok=True)
    Path("results/v11_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    stats = {"n": len(rows),
             "per_type": dict(Counter(r["error_type"] for r in rows)),
             "source": "TEC-JL (research-only, LREC 2020, human gold)",
             "mapping": "PART->particle, AUX->politeness, INFL/TENSE->conjugation, SPELL/OTHER/content->choice; PUNCT/WO/CONJ excluded"}
    Path("results/v11.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    import os
    os.environ["LANGSMITH_API_KEY"] = (
        Path.home() / ".config/langsmith/key").read_text().strip()
    os.environ["LANGSMITH_TRACING"] = "false"
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError

    c = Client()
    name = "kotoba-tecjl-gold-v1"
    try:
        c.delete_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        print("no prior dataset to delete")
    ds = c.create_dataset(name, description="Kotoba-Tutor v1.1: TEC-JL human-gold "
                          "single-edit sentences (research-only, never redistribute)")
    c.create_examples(
        inputs=[{"learner": r["learner"]} for r in rows],
        outputs=[{"gold": r["gold"], "error_type": r["error_type"],
                  "detail": r["detail"]} for r in rows],
        dataset_id=ds.id)
    print("uploaded", name, len(rows))


if __name__ == "__main__":
    main()
