"""Rule-based error typer v2: type from the (learner, correction) pair.

Align the two strings (difflib), inspect the differing chunks' morphology:
- either chunk a case/binding particle -> particle
- marker<->nothing, nothing<->marker, です<->だ, ください<->くれ -> politeness
- marker<->different marker (ます<->ました, ない<->なかった…) -> conjugation
- anything else (content swaps) -> choice
Identical strings -> none. CPU-only, deterministic.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kotoba.inject import toks

MARKERS = ("ください", "ました", "ません", "なかった", "でした",
           "ている", "ていた", "ます", "ない", "です", "だ", "くれ")
CONJ_FRAGS = MARKERS + ("まし", "なかっ", "た", "て", "よう", "ろう", "ます",
                        "れる", "られる", "せる", "させる")
PARTICLES = ("は", "が", "を", "に", "で", "へ", "と", "や", "も", "の")


def _pos(text: str) -> set[tuple[str, str]]:
    try:
        return {(p1, p2) for _, p1, p2 in toks(text)}
    except (RuntimeError, ValueError):
        return set()


def type_from_pair(learner: str, correction: str) -> str:
    if learner == correction:
        return "none"
    sm = difflib.SequenceMatcher(None, learner, correction, autojunk=False)
    chunks = [(learner[a:b], correction[c:d])
              for tag, a, b, c, d in sm.get_opcodes() if tag != "equal"]
    if not chunks:
        return "none"
    lchunk = "".join(l for l, _ in chunks)
    cchunk = "".join(c for _, c in chunks)
    lpos = _pos(lchunk) | _pos(learner)
    # particle: a standalone case/binding particle changed?
    if lchunk.strip() in PARTICLES or cchunk.strip() in PARTICLES:
        return "particle"
    lmark = any(m in lchunk for m in MARKERS)
    cmark = any(m in cchunk for m in MARKERS)
    if lmark != cmark:
        return "politeness"  # marker added or dropped = register break
    if lmark and cmark:
        pair = (lchunk.strip(), cchunk.strip())
        if pair in (("です", "だ"), ("だ", "です"), ("ください", "くれ"),
                    ("くれ", "ください")):
            return "politeness"
        return "conjugation"  # marker paradigm shift (ます/ました…)
    # tail-fragment conjugation (ます-stem/ない-stem/て-form splits)?
    def short(t: str) -> bool:
        return len(t.strip()) <= 2
    fragL = any(f in lchunk for f in CONJ_FRAGS)
    fragC = any(f in cchunk for f in CONJ_FRAGS)
    if (fragL and (fragC or short(cchunk))) or (fragC and short(lchunk)):
        return "conjugation"
    # content-word swap? confirm via POS if possible
    if ("助詞", "格助詞") in lpos or ("助詞", "係助詞") in lpos:
        for surf, p1, p2 in toks(lchunk):
            if p1 == "助詞" and p2 in ("格助詞", "係助詞") and surf in PARTICLES:
                return "particle"
    return "choice"


def main() -> None:
    import json
    from collections import Counter

    for name in ["results/m1_rows.jsonl", "results/v11_rows.jsonl"]:
        with open(name, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        cm = Counter()
        for r in rows:
            if r["error_type"] == "none":
                cm[("none", type_from_pair(r["learner"], r["gold"]))] += 1
                continue
            cm[(r["error_type"],
                type_from_pair(r["learner"], r["gold"]))] += 1
        print(name)
        for k in sorted(cm):
            print(f"  {k[0]:12s} -> {k[1]:12s} {cm[k]}")


if __name__ == "__main__":
    main()
